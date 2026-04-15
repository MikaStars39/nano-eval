#!/usr/bin/env python3
"""
sample_rot_per_source.py — 每个 source 抽取一条 context rot 样本（含完整 trajectory）

数据链路:
    merged_result.jsonl  →  筛选每个 source 的最佳 rot 样本，记录 global_index
    full_flagged.jsonl   →  按 global_index 提取对应行的 raw_data (完整对话)
    输出 JSONL:  judge 结果 + raw_data (messages/tools)

优先抽 score=1.0 (severe)，没有则抽 score=0.5 (mild)。

用法:
    python sample_rot_per_source.py \
        --input    /path/to/merged_result.jsonl \
        --original /path/to/full_flagged.jsonl \
        --save     /path/to/rot_samples.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def source_basename(source_file: str | None) -> str:
    if not source_file:
        return "<unknown>"
    return source_file.rsplit("/", 1)[-1]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Sample one context rot example per source with full trajectory")
    parser.add_argument("--input", required=True, help="Path to merged_result.jsonl")
    parser.add_argument("--original", required=True, help="Path to full_flagged.jsonl (with raw_data)")
    parser.add_argument("--save", required=True, help="Path to write sampled JSONL")
    args = parser.parse_args()

    # --- Pass 1: 从 merged_result 中找每个 source 的最佳 rot 样本 ---
    logger.info("Pass 1: scanning merged_result for best rot per source ...")
    best: dict[str, dict] = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            score = r.get("score", 0.0)
            if score == 0.0:
                continue

            src = source_basename(r.get("source_file"))
            prev = best.get(src)
            if prev is None or score > prev.get("score", 0):
                best[src] = r

    # 收集需要提取的 global_index
    needed_indices = {r["global_index"] for r in best.values()}
    logger.info("Found %d sources with rot, need %d trajectories", len(best), len(needed_indices))

    # --- Pass 2: 从 full_flagged.jsonl 提取对应行的 raw_data ---
    logger.info("Pass 2: extracting trajectories from full_flagged.jsonl ...")
    raw_by_index: dict[int, dict] = {}

    with open(args.original, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in needed_indices:
                d = json.loads(line)
                raw_by_index[idx] = d.get("raw_data", {})
                if len(raw_by_index) == len(needed_indices):
                    break  # 全部找到，提前退出

    logger.info("Extracted %d / %d trajectories", len(raw_by_index), len(needed_indices))

    # --- 合并并写出 ---
    samples = sorted(best.values(), key=lambda x: source_basename(x.get("source_file")))

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        for r in samples:
            gidx = r["global_index"]
            r["raw_data"] = raw_by_index.get(gidx, {})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    n_severe = sum(1 for r in samples if r.get("score") == 1.0)
    n_mild = sum(1 for r in samples if r.get("score") == 0.5)
    n_with_traj = sum(1 for r in samples if r.get("raw_data", {}).get("messages"))

    print(f"\nSampled {len(samples)} sources with context rot")
    print(f"  severe (1.0): {n_severe}")
    print(f"  mild   (0.5): {n_mild}")
    print(f"  with trajectory: {n_with_traj}")
    print(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
