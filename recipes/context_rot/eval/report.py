#!/usr/bin/env python3
"""
report.py — 从 results.jsonl 生成退化曲线和汇总报告

输出:
- 每个 case 的退化曲线 (P1-P5, original vs clean)
- ICL Gap 分析
- 按任务类型的汇总
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def print_degradation_curves(results: list[dict]):
    """打印每个 case 的退化曲线。"""
    # 按 case_index 分组
    by_case: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_case[r["case_index"]].append(r)

    for case_idx in sorted(by_case):
        case_results = by_case[case_idx]
        case_id = case_results[0]["case_id"]
        model = case_results[0]["model"]

        print(f"\n{'='*60}")
        print(f"Case {case_idx}: {case_id} (model: {model})")
        print(f"{'='*60}")

        # 按 test_point 排序
        by_point: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in case_results:
            by_point[r["test_point"]][r["condition"]] = r

        for point_name in sorted(by_point):
            original = by_point[point_name].get("original")
            clean = by_point[point_name].get("clean")

            if original:
                line = (
                    f"  {point_name} ({original['subtask_id']}, "
                    f"{original['n_prior_subtasks']} prior): "
                    f"{original['overall']:.3f}"
                )
                if clean:
                    icl_gap = clean["overall"] - original["overall"]
                    line += f"  |  clean: {clean['overall']:.3f}  |  ICL gap: {icl_gap:+.3f}"
                print(line)


def print_summary(results: list[dict]):
    """打印按任务类型的汇总表。"""
    # 按 case_id 和 test_point 分组
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    clean_by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        task = r["case_id"]
        point = r["test_point"]
        if r["condition"] == "original":
            by_task[task][point].append(r["overall"])
        else:
            clean_by_task[task][point].append(r["overall"])

    print(f"\n{'='*60}")
    print("Summary by Task Type")
    print(f"{'='*60}")
    print(f"{'Task Type':<25} {'P1':>6} {'P2':>6} {'P3':>6} {'P4':>6} {'P5':>6} {'Drop':>7} {'ICL':>7}")
    print("-" * 80)

    for task in sorted(by_task):
        scores = {}
        for point in ["P1", "P2", "P3", "P4", "P5"]:
            vals = by_task[task].get(point, [])
            scores[point] = sum(vals) / len(vals) if vals else None

        clean_scores = {}
        for point in ["P3", "P4", "P5"]:
            vals = clean_by_task[task].get(point, [])
            clean_scores[point] = sum(vals) / len(vals) if vals else None

        # 计算指标
        p1 = scores.get("P1")
        p5 = scores.get("P5")
        p5_clean = clean_scores.get("P5")

        total_drop = (p1 - p5) if (p1 is not None and p5 is not None) else None
        icl_gap = (p5_clean - p5) if (p5_clean is not None and p5 is not None) else None

        row = f"{task:<25}"
        for point in ["P1", "P2", "P3", "P4", "P5"]:
            v = scores.get(point)
            row += f" {v:.3f}" if v is not None else "   N/A"
        row += f" {total_drop:+.3f}" if total_drop is not None else "    N/A"
        row += f" {icl_gap:+.3f}" if icl_gap is not None else "    N/A"
        print(row)

    # 全局统计
    p1_scores = [r["overall"] for r in results if r["test_point"] == "P1" and r["condition"] == "original"]
    p5_scores = [r["overall"] for r in results if r["test_point"] == "P5" and r["condition"] == "original"]

    if p1_scores and p5_scores:
        avg_p1 = sum(p1_scores) / len(p1_scores)
        avg_p5 = sum(p5_scores) / len(p5_scores)
        print(f"\n  Average P1 score: {avg_p1:.3f}")
        print(f"  Average P5 score: {avg_p5:.3f}")
        print(f"  Average total drop: {avg_p1 - avg_p5:+.3f}")

        # 有多少 case 出现显著退化 (>20%)
        degraded = 0
        total_cases = 0
        by_case = defaultdict(dict)
        for r in results:
            if r["condition"] == "original" and r["test_point"] in ("P1", "P5"):
                by_case[r["case_index"]][r["test_point"]] = r["overall"]
        for case_idx, pts in by_case.items():
            if "P1" in pts and "P5" in pts:
                total_cases += 1
                ratio = pts["P5"] / pts["P1"] if pts["P1"] > 0 else 1.0
                if ratio < 0.8:
                    degraded += 1
        if total_cases:
            print(f"\n  Context Rot Susceptibility: {degraded}/{total_cases} cases ({100*degraded/total_cases:.0f}%) show >20% degradation")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Context Rot degradation report")
    parser.add_argument("--input", required=True, help="Input results.jsonl path")
    args = parser.parse_args()

    path = args.input

    if not Path(path).exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    results = load_results(path)
    print(f"Loaded {len(results)} results from {path}")

    print_degradation_curves(results)
    print_summary(results)


if __name__ == "__main__":
    main()
