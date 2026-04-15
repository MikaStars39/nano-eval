#!/usr/bin/env python3
"""
analyze_context_rot.py — 分析 merged_result.jsonl 中 context rot 的分布

输入: merge_judge_output.py 产出的 merged_result.jsonl
输出: 终端打印汇总 + 按 source 细分表格，可选 --save-csv 导出

用法:
    python analyze_context_rot.py \
        --input /path/to/merged_result.jsonl \
        [--save-csv /path/to/analysis.csv]
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def source_basename(source_file: str | None) -> str:
    if not source_file:
        return "<unknown>"
    return source_file.rsplit("/", 1)[-1]


def print_bar(ratio: float, width: int = 30) -> str:
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def analyze(records: list[dict], save_csv: Path | None = None):
    total = len(records)

    # --- 1. Status 分布 ---
    status_counts = Counter(r["status"] for r in records)
    print(f"\n{'=' * 70}")
    print(f"  CONTEXT ROT ANALYSIS  —  {total:,} records")
    print(f"{'=' * 70}")

    print(f"\n[1] Judge Status Distribution")
    print(f"    {'Status':<15} {'Count':>8}  {'Pct':>7}")
    print(f"    {'-'*15} {'-'*8}  {'-'*7}")
    for status in ["ok", "parse_error", "max_tokens", "api_error", "bad_finish"]:
        cnt = status_counts.get(status, 0)
        pct = cnt / total * 100 if total else 0
        print(f"    {status:<15} {cnt:>8,}  {pct:>6.2f}%")

    # --- 2. 仅分析 status=ok 的记录 ---
    ok_records = [r for r in records if r["status"] == "ok"]
    n_ok = len(ok_records)
    print(f"\n    Valid (ok) records for analysis: {n_ok:,} / {total:,}")

    # --- 3. Score / Recommendation 分布 ---
    score_counts = Counter()
    rec_counts = Counter()
    for r in ok_records:
        score_counts[r.get("score")] += 1
        rec_counts[r.get("recommendation")] += 1

    print(f"\n[2] Score Distribution (status=ok)")
    print(f"    {'Score':<10} {'Count':>8}  {'Pct':>7}  {'Meaning'}")
    print(f"    {'-'*10} {'-'*8}  {'-'*7}  {'-'*20}")
    score_labels = {0.0: "no context rot", 0.5: "mild context rot", 1.0: "severe context rot"}
    for score in [0.0, 0.5, 1.0]:
        cnt = score_counts.get(score, 0)
        pct = cnt / n_ok * 100 if n_ok else 0
        print(f"    {score:<10} {cnt:>8,}  {pct:>6.2f}%  {score_labels.get(score, '')}")

    n_rot = score_counts.get(0.5, 0) + score_counts.get(1.0, 0)
    rot_rate = n_rot / n_ok * 100 if n_ok else 0
    print(f"\n    Context Rot Total: {n_rot:,} / {n_ok:,} = {rot_rate:.2f}%")
    print(f"      - flag  (0.5): {score_counts.get(0.5, 0):,}")
    print(f"      - remove (1.0): {score_counts.get(1.0, 0):,}")

    print(f"\n[3] Recommendation Distribution (status=ok)")
    print(f"    {'Rec':<10} {'Count':>8}  {'Pct':>7}")
    print(f"    {'-'*10} {'-'*8}  {'-'*7}")
    for rec in ["keep", "flag", "remove"]:
        cnt = rec_counts.get(rec, 0)
        pct = cnt / n_ok * 100 if n_ok else 0
        print(f"    {rec:<10} {cnt:>8,}  {pct:>6.2f}%")

    # --- 4. 按 source_file 细分 ---
    per_source: dict[str, dict] = defaultdict(lambda: {"total": 0, "keep": 0, "flag": 0, "remove": 0})
    for r in ok_records:
        src = source_basename(r.get("source_file"))
        per_source[src]["total"] += 1
        rec = r.get("recommendation", "keep")
        if rec in ("keep", "flag", "remove"):
            per_source[src][rec] += 1

    # 按 rot count 降序
    source_rows = []
    for src, d in per_source.items():
        rot = d["flag"] + d["remove"]
        rot_pct = rot / d["total"] * 100 if d["total"] else 0
        source_rows.append({
            "source": src,
            "total": d["total"],
            "keep": d["keep"],
            "flag": d["flag"],
            "remove": d["remove"],
            "rot": rot,
            "rot_pct": rot_pct,
        })
    source_rows.sort(key=lambda x: x["rot"], reverse=True)

    print(f"\n[4] Context Rot by Source (sorted by rot count, top 40)")
    print(f"    {'Source':<65} {'Total':>6} {'Keep':>6} {'Flag':>5} {'Rm':>5} {'Rot':>5} {'Rot%':>7}  Bar")
    print(f"    {'-'*65} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*7}  {'-'*30}")
    for row in source_rows[:40]:
        bar = print_bar(row["rot_pct"] / 100, 20)
        src_display = row["source"][:63] if len(row["source"]) > 63 else row["source"]
        print(f"    {src_display:<65} {row['total']:>6} {row['keep']:>6} {row['flag']:>5} {row['remove']:>5} {row['rot']:>5} {row['rot_pct']:>6.1f}%  {bar}")

    # --- 5. 高 rot rate 的 source（至少 50 条数据）---
    high_rate = [r for r in source_rows if r["total"] >= 50 and r["rot_pct"] > 0]
    high_rate.sort(key=lambda x: x["rot_pct"], reverse=True)

    print(f"\n[5] Highest Rot Rate Sources (>=50 records, sorted by rot%)")
    print(f"    {'Source':<65} {'Total':>6} {'Rot':>5} {'Rot%':>7}")
    print(f"    {'-'*65} {'-'*6} {'-'*5} {'-'*7}")
    for row in high_rate[:30]:
        src_display = row["source"][:63] if len(row["source"]) > 63 else row["source"]
        print(f"    {src_display:<65} {row['total']:>6} {row['rot']:>5} {row['rot_pct']:>6.1f}%")

    # --- 6. 零 rot 的 source ---
    zero_rot = [r for r in source_rows if r["rot"] == 0]
    print(f"\n[6] Zero-Rot Sources: {len(zero_rot)} / {len(source_rows)} sources have 0 context rot")
    if zero_rot:
        print(f"    Examples: {', '.join(r['source'] for r in zero_rot[:10])}")

    # --- 7. 汇总 ---
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total records:        {total:>8,}")
    print(f"  Valid (ok):           {n_ok:>8,}")
    print(f"  Context rot:          {n_rot:>8,}  ({rot_rate:.2f}%)")
    print(f"    - flag (mild):      {score_counts.get(0.5, 0):>8,}")
    print(f"    - remove (severe):  {score_counts.get(1.0, 0):>8,}")
    print(f"  Clean (keep):         {score_counts.get(0.0, 0):>8,}  ({score_counts.get(0.0, 0) / n_ok * 100 if n_ok else 0:.2f}%)")
    print(f"  Sources with rot:     {len(source_rows) - len(zero_rot):>8}")
    print(f"  Sources without rot:  {len(zero_rot):>8}")
    print(f"{'=' * 70}\n")

    # --- 导出 CSV ---
    if save_csv:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "total", "keep", "flag", "remove", "rot", "rot_pct"])
            writer.writeheader()
            writer.writerows(source_rows)
        print(f"  CSV saved to: {save_csv}")


def main():
    parser = argparse.ArgumentParser(description="Analyze context rot distribution in merged judge results")
    parser.add_argument("--input", required=True, help="Path to merged_result.jsonl")
    parser.add_argument("--save-csv", default=None, help="Optional: save per-source breakdown to CSV")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    analyze(records, save_csv=Path(args.save_csv) if args.save_csv else None)


if __name__ == "__main__":
    main()
