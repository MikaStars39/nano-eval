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
import sys
import tqdm
from pathlib import Path

# Add nano-eval root to path for importing nanoeval modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
- "keep": score = 0.0，无偷懒问题

现在给你这个模型的一段对话记录，你开始判断：
"""

def build_judge_prompt(example: dict) -> str:
    """Build the judge user prompt from flagged info and original example."""
    messages = example.get("messages", [])
    
    conversation = JUDGE_SYSTEM + "\n" + json.dumps(messages)

    return conversation


def prepare_batch_input(
    input_path: str,
    output_dir: str,
    shard_size: int = 1000,
):
    """Build judge prompts and write sharded batch input JSONL.

    Output layout under output_dir:
        {stem}_000.jsonl, {stem}_001.jsonl, ...
        _files_info   (one JSON object per line: filename + lines)
    """
    input_p = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_p.stem  # e.g. "a" from "a.jsonl"

    shard_idx = 0
    line_count = 0
    files_info = []
    f_out = None

    def _open_new_shard():
        nonlocal f_out, shard_idx, line_count
        if f_out is not None:
            files_info.append({"filename": f_out.name, "lines": line_count})
            f_out.close()
        shard_name = f"{stem}_{shard_idx:03d}.jsonl"
        f_out = open(out_dir / shard_name, "w", encoding="utf-8")
        shard_idx += 1
        line_count = 0
        return f_out

    _open_new_shard()

    with open(input_path, "r", encoding="utf-8") as f_in:
        for line in tqdm.tqdm(f_in):
            if line_count >= shard_size:
                _open_new_shard()

            flagged = json.loads(line)
            example = flagged.get("raw_data")

            user_msg = build_judge_prompt(example)
            del example  # free memory

            batch_item = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": user_msg
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 1000
                }
            }
            f_out.write(json.dumps(batch_item, ensure_ascii=False) + "\n")
            line_count += 1

    # close last shard
    if f_out is not None:
        files_info.append({"filename": f_out.name, "lines": line_count})
        f_out.close()

    # write _files_info
    with open(out_dir / "_files_info", "w", encoding="utf-8") as f:
        for info in files_info:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")

def main():
    from nanoeval.utils.logging_utils import configure_logger
    configure_logger()

    parser = argparse.ArgumentParser(description="LLM Judge for flagged lazy training data")
    parser.add_argument("--input", required=True, help="Input flagged.jsonl")
    parser.add_argument("--output", required=True, help="Output directory for sharded JSONL")
    parser.add_argument("--shard-size", type=int, default=1000, help="Max lines per shard (default: 1000)")
    args = parser.parse_args()
    prepare_batch_input(
        input_path=args.input,
        output_dir=args.output,
        shard_size=args.shard_size,
    )



if __name__ == "__main__":
    main()
