#!/usr/bin/env python3
"""
scan_judge.py — LLM Judge for flagged lazy training data

Phase 2: For each flagged candidate from scan_rules.py, use an LLM judge
to evaluate whether the training example truly contains lazy patterns.

Refactored to use nano-eval's OnlineBatchInferenceEngine for batch inference.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add nano-eval root to path for importing nanoeval modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nanoeval.backend.online import OnlineBatchInferenceEngine, APIConfig


# ── Logging ──────────────────────────────────────────────────

def log(msg: str, item_id: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}]"
    if item_id:
        prefix += f" [{item_id}]"
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


# ── Judge prompt ──────────────────────────────────────────────

JUDGE_SYSTEM = """你是一个训练数据质量审核专家。你需要判断给定的训练数据是否包含"偷懒"模式——即模型在长对话后期降低输出质量、跳过步骤、编造数据、或以虚假理由简化工作的行为。

这些偷懒模式如果被用于训练，会教导模型在长对话中退化（Context Rot）。

你需要对以下维度评分（0.0=无此问题, 0.5=轻微, 1.0=严重）：

1. **fabricated_pressure** — reasoning/thinking 中是否捏造了时间/token压力（而用户消息中无此约束）
2. **output_degradation** — 对话后期输出是否显著短于前期（内容量而非字数）
3. **step_skipping** — 是否跳过了应执行的步骤
4. **vague_generic** — 后期是否使用泛泛描述替代具体数据
5. **task_abandonment** — 是否有未尝试的任务被标记为"待处理"

输出格式（严格JSON，不要添加其他内容）：

```json
{
    "scores": {
        "fabricated_pressure": 0.0,
        "output_degradation": 0.0,
        "step_skipping": 0.0,
        "vague_generic": 0.0,
        "task_abandonment": 0.0
    },
    "overall": 0.0,
    "recommendation": "remove|flag|keep",
    "justification": "理由"
}
```

recommendation 判定标准:
- "remove": 严重偷懒，应从训练数据中移除 (overall >= 0.7)
- "flag": 有偷懒倾向，需人工审核 (0.3 <= overall < 0.7)
- "keep": 无偷懒问题或极轻微 (overall < 0.3)"""

JUDGE_USER_TEMPLATE = """## 规则筛选结果

以下训练样本被规则检测器标记为可能包含偷懒模式。

### 基本信息
- 来源文件: {source_file}
- 对话长度: {n_messages} 条消息 (约 {estimated_tokens} tokens)
- 匹配到的偷懒关键词: {keyword_summary}

### 早期Assistant输出（基准质量）
{early_samples}

### 后期Assistant输出（疑似退化）
{late_samples}

### Reasoning/Thinking中的偷懒片段
{reasoning_evidence}

请逐项评分，判断此训练样本是否包含偷懒模式。"""


# ── Read original training data ──────────────────────────────

def read_original_example(source_file: str, line_number: int) -> dict | None:
    """Read a specific line from the original JSONL file."""
    try:
        with open(source_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if i == line_number:
                    return json.loads(line.strip())
    except Exception as e:
        log(f"Error reading {source_file}:{line_number}: {e}")
    return None


def extract_assistant_content(msg: dict) -> str:
    """Extract text content from an assistant message."""
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return content


def build_judge_prompt(flagged: dict, example: dict) -> str:
    """Build the judge user prompt from flagged info and original example."""
    messages = example.get("messages", [])

    # Collect assistant messages with substantial content
    assistant_msgs = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = extract_assistant_content(msg)
        if len(content) > 50:
            assistant_msgs.append((i, content))

    # Early samples: first 2 substantive assistant messages
    early_samples = ""
    for idx, (msg_idx, content) in enumerate(assistant_msgs[:2]):
        truncated = content[:1500]
        if len(content) > 1500:
            truncated += "...[截断]"
        early_samples += f"\n**[消息 #{msg_idx}] ({len(content)} 字)**\n{truncated}\n"
    if not early_samples:
        early_samples = "(无实质性早期输出)"

    # Late samples: last 3 substantive assistant messages
    late_samples = ""
    for idx, (msg_idx, content) in enumerate(assistant_msgs[-3:]):
        truncated = content[:1500]
        if len(content) > 1500:
            truncated += "...[截断]"
        late_samples += f"\n**[消息 #{msg_idx}] ({len(content)} 字)**\n{truncated}\n"
    if not late_samples:
        late_samples = "(无后期输出)"

    # Reasoning evidence from keyword hits
    reasoning_evidence = ""
    seen_snippets = set()
    for hit in flagged.get("keyword_hits", [])[:5]:
        snippet = hit.get("snippet", "")
        if snippet and snippet not in seen_snippets:
            seen_snippets.add(snippet)
            reasoning_evidence += f"\n- [{hit['field']}] 关键词「{hit['keyword']}」: {snippet}\n"
    if not reasoning_evidence:
        reasoning_evidence = "(无reasoning证据)"

    # Keyword summary
    keywords = list(set(hit["keyword"] for hit in flagged.get("keyword_hits", [])))
    keyword_summary = ", ".join(f"「{kw}」" for kw in keywords[:10])

    return JUDGE_USER_TEMPLATE.format(
        source_file=os.path.basename(flagged["source_file"]),
        n_messages=flagged["n_messages"],
        estimated_tokens=flagged["estimated_tokens"],
        keyword_summary=keyword_summary,
        early_samples=early_samples,
        late_samples=late_samples,
        reasoning_evidence=reasoning_evidence,
    )


# ── Normalize scores ──────────────────────────────────────────

EXPECTED_KEYS = {
    "fabricated_pressure", "output_degradation",
    "step_skipping", "vague_generic", "task_abandonment",
}


def normalize_scores(result: dict) -> dict:
    """Normalize judge response to standard format."""
    if "scores" in result and isinstance(result["scores"], dict):
        scores = result["scores"]
    else:
        scores = {}
        for k, v in result.items():
            if k in EXPECTED_KEYS and isinstance(v, (int, float)):
                scores[k] = float(v) if v <= 1 else float(v) / 10.0
        result["scores"] = scores

    for k in EXPECTED_KEYS:
        if k not in scores:
            scores[k] = 0.0
        else:
            v = scores[k]
            if isinstance(v, (int, float)):
                scores[k] = float(v) if v <= 1 else float(v) / 10.0
            elif isinstance(v, dict):
                scores[k] = float(v.get("score", v.get("value", 0)))
                if scores[k] > 1:
                    scores[k] /= 10.0

    if scores:
        result["overall"] = sum(scores.values()) / len(scores)
    else:
        result["overall"] = 0.0

    overall = result["overall"]
    if overall >= 0.7:
        result["recommendation"] = "remove"
    elif overall >= 0.3:
        result["recommendation"] = "flag"
    else:
        result["recommendation"] = "keep"

    if "justification" not in result:
        result["justification"] = ""

    return result


def parse_judge_response(response_text: str) -> dict:
    """Parse judge response text, extract JSON, normalize scores."""
    content = response_text
    if "```json" in content:
        match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1)
    elif "```" in content:
        match = re.search(r"```\s*(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1)

    try:
        result = json.loads(content)
        return normalize_scores(result)
    except json.JSONDecodeError:
        return {
            "scores": {k: 0.0 for k in EXPECTED_KEYS},
            "overall": 0.0,
            "recommendation": "keep",
            "justification": f"JSON parse error: {content[:200]}",
        }


# ── Main pipeline ──────────────────────────────────────────────

def prepare_batch_input(flagged_items: list[dict], output_path: str) -> dict[str, dict]:
    """
    Build judge prompts for all flagged items and write to a JSONL for batch inference.

    Returns a mapping from item ID to the original flagged dict (for post-processing).
    """
    id_to_flagged = {}
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, flagged in enumerate(flagged_items):
            item_id = f"scan_{i}"

            # Read original example
            example = read_original_example(flagged["source_file"], flagged["line_number"])
            if not example:
                log(f"Could not read original file, skipping", item_id)
                skipped += 1
                continue

            user_msg = build_judge_prompt(flagged, example)
            del example  # free memory

            # Write batch input line (messages format for OnlineBatchInferenceEngine)
            batch_item = {
                "id": item_id,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            }
            f.write(json.dumps(batch_item, ensure_ascii=False) + "\n")
            id_to_flagged[item_id] = flagged

    log(f"Prepared {len(id_to_flagged)} items for batch inference ({skipped} skipped)")
    return id_to_flagged


def post_process_results(
    inference_output: str,
    id_to_flagged: dict[str, dict],
    final_output: str,
    judge_model: str,
):
    """Parse batch inference results and write final judged.jsonl."""
    results = []
    errors = 0

    with open(inference_output, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item_id = item.get("id", "")
            flagged = id_to_flagged.get(item_id)
            if not flagged:
                continue

            response_text = item.get("response", "")
            status = item.get("_status", "")

            if status == "failed" or not response_text:
                results.append({
                    "source_file": flagged["source_file"],
                    "line_number": flagged["line_number"],
                    "judge_error": item.get("_error", "no response"),
                })
                errors += 1
                continue

            judge_result = parse_judge_response(response_text)
            results.append({
                "source_file": flagged["source_file"],
                "line_number": flagged["line_number"],
                "estimated_tokens": flagged["estimated_tokens"],
                "n_messages": flagged["n_messages"],
                "n_keyword_hits": flagged["n_keyword_hits"],
                "keyword_hits": flagged["keyword_hits"][:5],
                "judge_scores": judge_result["scores"],
                "overall": judge_result["overall"],
                "recommendation": judge_result["recommendation"],
                "justification": judge_result.get("justification", ""),
                "judge_model": judge_model,
            })

    os.makedirs(os.path.dirname(final_output) or ".", exist_ok=True)
    with open(final_output, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    recommendations = {"remove": 0, "flag": 0, "keep": 0, "error": 0}
    for r in results:
        rec = r.get("recommendation", "error")
        if "judge_error" in r:
            rec = "error"
        recommendations[rec] = recommendations.get(rec, 0) + 1

    log("")
    log("=" * 60)
    log("JUDGE COMPLETE")
    log(f"  Total judged:  {len(results)}")
    log(f"  Remove:        {recommendations['remove']}")
    log(f"  Flag:          {recommendations['flag']}")
    log(f"  Keep:          {recommendations['keep']}")
    log(f"  Errors:        {recommendations['error']}")
    log(f"  Output:        {final_output}")
    log("=" * 60)


async def run_batch_judge(args):
    """Main pipeline: prepare -> batch inference -> post-process."""
    # Load flagged items
    flagged_items = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                flagged_items.append(json.loads(line))

    log(f"Loaded {len(flagged_items)} flagged items")

    if args.limit:
        flagged_items = flagged_items[:args.limit]
        log(f"Limited to {len(flagged_items)} items")

    # Step 1: Prepare batch input
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    batch_input_path = os.path.join(output_dir, ".scan_judge_input.jsonl")
    batch_output_path = os.path.join(output_dir, ".scan_judge_inference.jsonl")

    id_to_flagged = prepare_batch_input(flagged_items, batch_input_path)

    if not id_to_flagged:
        log("No items to judge after preparation")
        return

    # Step 2: Batch inference using nano-eval's OnlineBatchInferenceEngine
    log(f"Starting batch inference with concurrency={args.concurrency}")
    config = APIConfig(
        api_key=args.judge_api_key,
        base_url=args.judge_api_base.rstrip("/"),
        model=args.judge_model,
    )
    engine = OnlineBatchInferenceEngine(config, concurrency=args.concurrency)

    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    # Clean previous inference output if exists (avoid resume appending)
    if os.path.exists(batch_output_path):
        os.remove(batch_output_path)

    await engine.run(batch_input_path, batch_output_path, sampling_params)

    # Step 3: Post-process results
    log("Post-processing inference results...")
    post_process_results(batch_output_path, id_to_flagged, args.output, args.judge_model)

    # Clean up temp files
    for tmp in [batch_input_path, batch_output_path]:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    parser = argparse.ArgumentParser(description="LLM Judge for flagged lazy training data")
    parser.add_argument("--input", required=True, help="Input flagged.jsonl from scan_rules.py")
    parser.add_argument("--output", required=True, help="Output judged.jsonl")
    parser.add_argument("--judge-model", required=True, help="Judge model name")
    parser.add_argument("--judge-api-base", required=True, help="Judge API base URL")
    parser.add_argument("--judge-api-key", required=True, help="Judge API key")
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent API calls")
    parser.add_argument("--limit", type=int, default=None, help="Max items to judge")
    args = parser.parse_args()

    asyncio.run(run_batch_judge(args))


if __name__ == "__main__":
    main()
