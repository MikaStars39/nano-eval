#!/usr/bin/env python3
"""
scan_judge.py — LLM Judge for flagged lazy training data

Phase 2: For each flagged candidate from scan_rules.py, use an LLM judge
to evaluate whether the training example truly contains lazy patterns.

Uses nano-eval's OnlineInferenceActor (Ray) for batch inference.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

# Add nano-eval root to path for importing nanoeval modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import ray

from nanoeval.ray import init_ray
from nanoeval.ray.actors import OnlineInferenceActor

logger = logging.getLogger(__name__)


# ── Judge prompt ──────────────────────────────────────────────

JUDGE_SYSTEM = """你是一个训练数据质量审核专家。你需要判断给定的训练数据是否包含"偷懒"模式——即模型在长对话后期降低输出质量的行为。

偷懒模式包括但不限于：
- 在 reasoning/thinking 中捏造时间或 token 压力（而用户消息中无此约束）
- 对话后期输出质量显著下降（内容量而非字数）
- 跳过应执行的步骤
- 用泛泛描述替代具体内容
- 未尝试的任务被标记为"待处理"

评分标准（仅限三档）：
- 0.0：无偷懒问题
- 0.5：有轻微偷懒倾向
- 1.0：严重偷懒

输出格式（严格JSON，不要添加其他内容）：

```json
{
    "score": 0.0,
    "recommendation": "remove|flag|keep",
    "justification": "理由"
}
```

recommendation 判定标准:
- "remove": score = 1.0，严重偷懒，应从训练数据中移除
- "flag": score = 0.5，有偷懒倾向，需人工审核
- "keep": score = 0.0，无偷懒问题"""

JUDGE_USER_TEMPLATE = """## 规则筛选结果

匹配到的偷懒关键词: {keyword_summary}

## 对话内容

{conversation}

请判断此训练样本是否包含偷懒模式。"""


# ── Read original training data ──────────────────────────────

def batch_read_original_examples(
    flagged_items: list[dict],
) -> dict[tuple[str, int], dict]:
    """Read original examples in batch, grouped by source file.

    Returns a mapping of (source_file, line_number) -> parsed JSON example.
    Each file is opened at most once and scanned sequentially.
    """
    from collections import defaultdict

    file_lines: dict[str, set[int]] = defaultdict(set)
    for item in flagged_items:
        file_lines[item["source_file"]].add(item["line_number"])

    results: dict[tuple[str, int], dict] = {}
    for source_file, needed in file_lines.items():
        max_line = max(needed)
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if line_no in needed:
                        try:
                            results[(source_file, line_no)] = json.loads(line.strip())
                        except json.JSONDecodeError as e:
                            logger.error("JSON parse error at %s:%d: %s", source_file, line_no, e)
                    if line_no >= max_line:
                        break
        except Exception as e:
            logger.error("Error reading %s: %s", source_file, e)

    return results


def extract_assistant_content(msg: dict) -> str:
    """Extract text content from a message."""
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return content


def _truncate(text: str, limit: int = 3000) -> str:
    """Truncate long text, keeping head and tail."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[截断]...\n" + text[-half:]


def build_judge_prompt(flagged: dict, example: dict) -> str:
    """Build the judge user prompt from flagged info and original example."""
    messages = example.get("messages", [])

    # Format full conversation
    parts = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = extract_assistant_content(msg)
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

        lines = [f"### [{role}] 消息 #{i}"]
        if content:
            lines.append(_truncate(content))
        if isinstance(reasoning, str) and reasoning.strip():
            lines.append(f"<reasoning>\n{_truncate(reasoning)}\n</reasoning>")
        parts.append("\n".join(lines))

    conversation = "\n\n".join(parts)

    keywords = list(set(hit["keyword"] for hit in flagged.get("keyword_hits", [])))
    keyword_summary = ", ".join(f"「{kw}」" for kw in keywords[:10])

    return JUDGE_USER_TEMPLATE.format(
        keyword_summary=keyword_summary,
        conversation=conversation,
    )


# ── Normalize scores ──────────────────────────────────────────

VALID_SCORES = {0.0, 0.5, 1.0}


def normalize_score(result: dict) -> dict:
    """Normalize judge response to 3-point scale (0.0 / 0.5 / 1.0)."""
    score = result.get("score", 0.0)
    if isinstance(score, (int, float)):
        score = float(score)
        if score > 1:
            score /= 10.0
        # Snap to nearest valid value
        score = min(VALID_SCORES, key=lambda v: abs(v - score))
    else:
        score = 0.0

    result["score"] = score

    if score >= 1.0:
        result["recommendation"] = "remove"
    elif score >= 0.5:
        result["recommendation"] = "flag"
    else:
        result["recommendation"] = "keep"

    if "justification" not in result:
        result["justification"] = ""

    return result


def parse_judge_response(response_text: str) -> dict:
    """Parse judge response text, extract JSON, normalize score."""
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
        return normalize_score(result)
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "recommendation": "keep",
            "justification": f"JSON parse error: {content[:200]}",
        }


# ── Main pipeline ──────────────────────────────────────────────

def prepare_batch_input(flagged_items: list[dict], output_path: str) -> dict[str, dict]:
    """Build judge prompts and write batch input JSONL for OnlineInferenceActor.

    Returns a mapping from item ID to the original flagged dict.
    """
    id_to_flagged = {}
    skipped = 0

    # Prefer embedded raw_data; fall back to batch file reading
    needs_file_read = [f for f in flagged_items if "raw_data" not in f]
    file_examples = batch_read_original_examples(needs_file_read) if needs_file_read else {}

    with open(output_path, "w", encoding="utf-8") as f:
        for i, flagged in enumerate(flagged_items):
            item_id = f"scan_{i}"

            example = flagged.get("raw_data") or file_examples.get(
                (flagged["source_file"], flagged["line_number"])
            )
            if not example:
                logger.warning("[%s] Could not read %s:%d, skipping",
                               item_id, flagged["source_file"], flagged["line_number"])
                skipped += 1
                continue

            user_msg = build_judge_prompt(flagged, example)
            del example  # free memory

            # Write batch input line (messages format for OnlineInferenceActor)
            batch_item = {
                "id": item_id,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            }
            f.write(json.dumps(batch_item, ensure_ascii=False) + "\n")
            id_to_flagged[item_id] = flagged

    logger.info("Prepared %d items for batch inference (%d skipped)", len(id_to_flagged), skipped)
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
                "score": judge_result["score"],
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

    logger.info("=" * 60)
    logger.info("JUDGE COMPLETE")
    logger.info("  Total judged:  %d", len(results))
    logger.info("  Remove:        %d", recommendations['remove'])
    logger.info("  Flag:          %d", recommendations['flag'])
    logger.info("  Keep:          %d", recommendations['keep'])
    logger.info("  Errors:        %d", recommendations['error'])
    logger.info("  Output:        %s", final_output)
    logger.info("=" * 60)


async def run_batch_judge(args):
    """Main pipeline: prepare -> batch inference via Ray -> post-process."""
    # Load flagged items
    flagged_items = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                flagged_items.append(json.loads(line))

    logger.info("Loaded %d flagged items", len(flagged_items))

    if args.limit:
        flagged_items = flagged_items[:args.limit]
        logger.info("Limited to %d items", len(flagged_items))

    # Step 1: Prepare batch input
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    batch_input_path = os.path.join(output_dir, ".scan_judge_input.jsonl")
    batch_output_path = os.path.join(output_dir, ".scan_judge_inference.jsonl")

    id_to_flagged = prepare_batch_input(flagged_items, batch_input_path)

    if not id_to_flagged:
        logger.info("No items to judge after preparation")
        return

    # Step 2: Batch inference via Ray OnlineInferenceActor
    logger.info("Starting batch inference with concurrency=%d", args.concurrency)
    init_ray(address=args.ray_address)

    # Parse extra headers
    extra_headers = {}
    for h in args.extra_headers:
        key, _, value = h.partition(":")
        extra_headers[key.strip()] = value.strip()

    sampling_params = {
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    # Clean previous inference output if exists (avoid resume appending)
    if os.path.exists(batch_output_path):
        os.remove(batch_output_path)

    actor = OnlineInferenceActor.options(num_cpus=1).remote(
        api_key=args.judge_api_key,
        base_url=args.judge_api_base,
        model=args.judge_model,
        concurrency=args.concurrency,
        extra_headers=extra_headers or None,
        api_type=args.api_type,
    )
    ray.get(actor.run.remote(batch_input_path, batch_output_path, sampling_params))

    # Step 3: Post-process results
    logger.info("Post-processing inference results...")
    post_process_results(batch_output_path, id_to_flagged, args.output, args.judge_model)

    # Clean up temp files
    for tmp in [batch_input_path, batch_output_path]:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    from nanoeval.utils.logging_utils import configure_logger
    configure_logger()

    parser = argparse.ArgumentParser(description="LLM Judge for flagged lazy training data")
    parser.add_argument("--input", required=True, help="Input flagged.jsonl")
    parser.add_argument("--output", required=True, help="Output judged.jsonl")
    parser.add_argument("--judge-model", required=True, help="Judge model name")
    parser.add_argument("--judge-api-base", required=True, help="API base URL")
    parser.add_argument("--judge-api-key", required=True, help="API key")
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent API calls")
    parser.add_argument("--limit", type=int, default=None, help="Max items to judge")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray cluster address")
    parser.add_argument("--api-type", type=str, default="chat", choices=["chat", "responses"],
                        help="API type: chat (Chat Completions) or responses (Responses API)")
    parser.add_argument("--extra-headers", nargs="*", default=[],
                        help="Extra HTTP headers as 'Key: Value' pairs")
    args = parser.parse_args()

    import asyncio
    asyncio.run(run_batch_judge(args))


if __name__ == "__main__":
    main()
