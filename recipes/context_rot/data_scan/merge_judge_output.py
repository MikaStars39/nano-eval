#!/usr/bin/env python3
"""
merge_judge_output.py — 将 vulcan 推理输出 match 回原始输入，解析 judge 评分

用法:
    python merge_judge_output.py \
        --input-dir  /path/to/vulcan_110k_context_rot_data \
        --output-dir /path/to/vulcan_110k_context_rot_data/output \
        --original   /path/to/full_flagged.jsonl \
        --save       /path/to/merged_result.jsonl

数据链路:
    full_flagged.jsonl (原始数据, 有 source_file 等元数据)
        ↓ prepare_vulcan.py (--shard-size 5000, 顺序处理)
    full_flagged_000~022.jsonl (batch input, 只有 prompt)
        ↓ vulcan 推理
    output/full_flagged_000~022.jsonl (推理结果)
        ↓ 本脚本 (按 global_index == 原始行号 match)
    merged_result.jsonl

输出 JSONL 每行字段:
    source_file    — 原始数据来源文件
    line_number    — 在 source_file 中的行号
    shard          — 分片文件名
    index          — 行内序号 (0-based)
    global_index   — 全局序号 (= full_flagged.jsonl 行号)
    doc_id         — vulcan 生成的 doc_id
    score          — judge 评分 (0.0 / 0.5 / 1.0)，解析失败为 null
    recommendation — remove / flag / keep，解析失败为 null
    justification  — judge 理由
    finish_reason  — STOP / MAX_TOKENS / ...
    status         — ok / parse_error / max_tokens / api_error / bad_finish
    error_detail   — 错误信息 (仅失败时)
    original_meta  — 原始元数据 (source_file, line_number, keyword_hits 等，不含 raw_data)
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_judge_response(text: str) -> dict | None:
    """从 judge 输出文本中提取 JSON 评分结果。"""
    # 尝试 ```json ... ``` 代码块
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试直接匹配 { ... }
    m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def load_original_metadata(original_path: Path) -> list[dict]:
    """从原始 full_flagged.jsonl 加载每行的元数据（不含 raw_data）。"""
    logger.info("Loading original metadata from %s ...", original_path)
    metadata = []
    with open(original_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d.pop("raw_data", None)  # 不保留原始对话，太大
            metadata.append(d)
    logger.info("Loaded %d original records", len(metadata))
    return metadata


def process_shards(input_dir: Path, output_dir: Path, save_path: Path,
                   original_meta: list[dict] | None = None):
    """逐分片 match 输入输出，解析 judge 结果并写入合并文件。"""

    # 找到所有输入分片
    input_files = sorted(input_dir.glob("full_flagged_*.jsonl"))
    if not input_files:
        logger.error("No input shards found in %s", input_dir)
        sys.exit(1)

    stats = {"ok": 0, "parse_error": 0, "max_tokens": 0, "api_error": 0, "bad_finish": 0}
    global_idx = 0

    with open(save_path, "w", encoding="utf-8") as f_out:
        for input_file in input_files:
            shard_name = input_file.name
            output_file = output_dir / shard_name

            if not output_file.exists():
                logger.warning("Output shard missing: %s", output_file)
                # 写入全部 api_error
                with open(input_file, "r", encoding="utf-8") as fi:
                    for i, line in enumerate(fi):
                        meta = original_meta[global_idx] if original_meta else {}
                        record = {
                            "source_file": meta.get("source_file"),
                            "line_number": meta.get("line_number"),
                            "shard": shard_name,
                            "index": i,
                            "global_index": global_idx,
                            "doc_id": None,
                            "score": None,
                            "recommendation": None,
                            "justification": None,
                            "finish_reason": None,
                            "status": "api_error",
                            "error_detail": "output shard missing",
                            "original_meta": meta,
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stats["api_error"] += 1
                        global_idx += 1
                continue

            with open(input_file, "r", encoding="utf-8") as fi, \
                 open(output_file, "r", encoding="utf-8") as fo:
                for i, (in_line, out_line) in enumerate(zip(fi, fo)):
                    out_data = json.loads(out_line)
                    meta = original_meta[global_idx] if original_meta else {}

                    record = {
                        "source_file": meta.get("source_file"),
                        "line_number": meta.get("line_number"),
                        "shard": shard_name,
                        "index": i,
                        "global_index": global_idx,
                        "doc_id": out_data.get("doc_id"),
                        "score": None,
                        "recommendation": None,
                        "justification": None,
                        "finish_reason": None,
                        "status": None,
                        "error_detail": None,
                        "original_meta": meta,
                    }

                    # API 错误
                    if out_data.get("error_detail"):
                        record["status"] = "api_error"
                        record["error_detail"] = out_data["error_detail"]
                        stats["api_error"] += 1
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        global_idx += 1
                        continue

                    # 解析 response
                    try:
                        resp = json.loads(out_data["vulcan2_response"])
                        candidate = resp["candidates"][0]
                        finish_reason = candidate.get("finishReason", "UNKNOWN")
                        text = candidate["content"]["parts"][0]["text"]
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        record["status"] = "parse_error"
                        record["error_detail"] = f"response structure error: {e}"
                        stats["parse_error"] += 1
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        global_idx += 1
                        continue

                    record["finish_reason"] = finish_reason

                    # 非正常结束
                    if finish_reason not in ("STOP", "MAX_TOKENS"):
                        record["status"] = "bad_finish"
                        record["error_detail"] = f"finishReason={finish_reason}"
                        stats["bad_finish"] += 1
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        global_idx += 1
                        continue

                    # 解析 judge JSON
                    parsed = parse_judge_response(text)
                    if parsed:
                        record["score"] = parsed.get("score")
                        record["recommendation"] = parsed.get("recommendation")
                        record["justification"] = parsed.get("justification")
                        if finish_reason == "MAX_TOKENS":
                            record["status"] = "max_tokens"
                            stats["max_tokens"] += 1
                        else:
                            record["status"] = "ok"
                            stats["ok"] += 1
                    else:
                        record["status"] = "parse_error" if finish_reason == "STOP" else "max_tokens"
                        record["error_detail"] = f"cannot parse judge JSON from text (finishReason={finish_reason})"
                        stats["parse_error" if finish_reason == "STOP" else "max_tokens"] += 1

                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    global_idx += 1

            logger.info("Processed %s (running total: %d)", shard_name, global_idx)

    # 打印统计
    print(f"\n{'='*50}")
    print(f"Total: {global_idx}")
    for k, v in stats.items():
        pct = v / global_idx * 100 if global_idx else 0
        print(f"  {k}: {v} ({pct:.2f}%)")
    print(f"{'='*50}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Merge vulcan judge output back to input")
    parser.add_argument("--input-dir", required=True, help="Directory with input full_flagged_*.jsonl shards")
    parser.add_argument("--output-dir", required=True, help="Directory with output full_flagged_*.jsonl shards")
    parser.add_argument("--original", required=True, help="Original full_flagged.jsonl (with source_file etc.)")
    parser.add_argument("--save", required=True, help="Path to write merged JSONL result")
    args = parser.parse_args()

    original_meta = load_original_metadata(Path(args.original))

    process_shards(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        save_path=Path(args.save),
        original_meta=original_meta,
    )


if __name__ == "__main__":
    main()
