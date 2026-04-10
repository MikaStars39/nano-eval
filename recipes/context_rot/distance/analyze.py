#!/usr/bin/env python3
"""
analyze_distance.py — Analyze distance sensitivity experiment results

Reads results_sp_distance.jsonl and/or results_query_distance.jsonl from a run
directory and produces analysis of how quality degrades with each distance type.

Usage:
    python analyze_distance.py <run_dir>
    python analyze_distance.py /path/to/runs_distance/20260410-123456
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def analyze_experiment(results: list[dict], experiment_type: str):
    """Analyze results for one experiment type (sp or query)."""
    if not results:
        print(f"  No results for {experiment_type} experiment")
        return

    # Parse padding level from condition field
    for r in results:
        cond = r.get("condition", "")
        if "sp_pad_" in cond:
            r["_padding"] = int(cond.split("sp_pad_")[1])
        elif "qpad_" in cond:
            r["_padding"] = int(cond.split("qpad_")[1])
        else:
            r["_padding"] = 0

    # Group by base test point (case_index + test_point)
    by_base = defaultdict(list)
    for r in results:
        base_key = f"case_{r['case_index']}_{r['test_point']}"
        by_base[base_key].append(r)

    # Print per-test-point curves
    for base_key in sorted(by_base):
        group = sorted(by_base[base_key], key=lambda x: x["_padding"])
        case_id = group[0]["case_id"]
        subtask = group[0]["subtask_id"]
        base_prefix = group[0]["prefix_msg_count"]

        print(f"\n  {base_key} ({case_id}, {subtask}, base_prefix={base_prefix})")
        print(f"  {'Padding':>10} {'Total msgs':>10} {'Score':>8} {'Resp len':>10} {'Tools':>6} {'Exit':>12}")
        print(f"  {'-'*60}")

        baseline_score = None
        for r in group:
            pad = r["_padding"]
            total_msgs = r["prefix_msg_count"]
            score = r["overall"]
            resp_len = r["response_length"]
            tools = r["tool_calls_made"]
            exit_r = r["exit_reason"]

            if baseline_score is None:
                baseline_score = score
                delta = ""
            else:
                delta = f"  ({score - baseline_score:+.3f})"

            print(f"  {pad:>10} {total_msgs:>10} {score:>8.3f}{delta:<10} {resp_len:>10} {tools:>6} {exit_r:>12}")

    # Aggregate curve across all test points
    by_padding = defaultdict(list)
    for r in results:
        by_padding[r["_padding"]].append(r["overall"])

    print(f"\n  Aggregate curve (averaged across {len(by_base)} test points):")
    print(f"  {'Padding':>10} {'Avg Score':>10} {'N':>4} {'Δ from 0':>10}")
    print(f"  {'-'*40}")
    baseline_avg = None
    for pad in sorted(by_padding):
        scores = by_padding[pad]
        avg = sum(scores) / len(scores)
        if baseline_avg is None:
            baseline_avg = avg
            delta = ""
        else:
            delta = f"{avg - baseline_avg:+.3f}"
        print(f"  {pad:>10} {avg:>10.3f} {len(scores):>4} {delta:>10}")

    # Compute Pearson correlation: padding vs overall
    all_pad = [r["_padding"] for r in results]
    all_scores = [r["overall"] for r in results]
    n = len(all_pad)
    if n > 2:
        mean_p = sum(all_pad) / n
        mean_s = sum(all_scores) / n
        cov = sum((p - mean_p) * (s - mean_s) for p, s in zip(all_pad, all_scores)) / n
        std_p = (sum((p - mean_p) ** 2 for p in all_pad) / n) ** 0.5
        std_s = (sum((s - mean_s) ** 2 for s in all_scores) / n) ** 0.5
        r = cov / (std_p * std_s) if std_p * std_s > 0 else 0
        print(f"\n  Pearson r(padding, score) = {r:.3f}")
        if abs(r) > 0.5:
            print(f"  → Strong {'negative' if r < 0 else 'positive'} correlation: "
                  f"{'quality degrades' if r < 0 else 'quality improves'} with {experiment_type} distance")
        elif abs(r) > 0.3:
            print(f"  → Moderate {'negative' if r < 0 else 'positive'} correlation")
        else:
            print(f"  → Weak correlation: {experiment_type} distance has limited effect")

    # Per-dimension analysis for the most degraded test point
    most_degraded = None
    max_drop = 0
    for base_key, group in by_base.items():
        group = sorted(group, key=lambda x: x["_padding"])
        if len(group) >= 2:
            drop = group[0]["overall"] - group[-1]["overall"]
            if drop > max_drop:
                max_drop = drop
                most_degraded = (base_key, group)

    if most_degraded and max_drop > 0.05:
        base_key, group = most_degraded
        print(f"\n  Most degraded test point: {base_key} (drop={max_drop:.3f})")
        print(f"  Per-dimension scores (baseline vs max padding):")
        baseline_scores = group[0].get("scores", {})
        final_scores = group[-1].get("scores", {})
        for dim in baseline_scores:
            b = baseline_scores.get(dim, 0)
            f_score = final_scores.get(dim, 0)
            delta = f_score - b
            bar = "▼" if delta < -0.1 else ("▲" if delta > 0.1 else "─")
            print(f"    {dim:<25} {b:.2f} → {f_score:.2f}  {delta:+.2f} {bar}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_distance.py <run_dir>", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"ERROR: {run_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"{'='*70}")
    print("Distance Sensitivity Analysis")
    print(f"{'='*70}")
    print(f"Run directory: {run_dir}")

    # SP distance results
    sp_results = load_results(run_dir / "results_sp_distance.jsonl")
    if sp_results:
        model = sp_results[0].get("model", "unknown")
        print(f"\n{'─'*70}")
        print(f"Experiment A: SP Distance (model: {model})")
        print(f"{'─'*70}")
        print("  Manipulation: irrelevant conversations inserted after system prompt")
        print("  Controls: user query always at end, same tools/simulator")
        analyze_experiment(sp_results, "SP")

    # Query distance results
    query_results = load_results(run_dir / "results_query_distance.jsonl")
    if query_results:
        model = query_results[0].get("model", "unknown")
        print(f"\n{'─'*70}")
        print(f"Experiment B: User Query Distance (model: {model})")
        print(f"{'─'*70}")
        print("  Manipulation: dummy tool-call rounds injected after continuation prompt")
        print("  Controls: same prefix (SP distance fixed), same tools/simulator")
        analyze_experiment(query_results, "query")

    # Comparative summary
    if sp_results and query_results:
        print(f"\n{'='*70}")
        print("Comparative Summary")
        print(f"{'='*70}")

        for label, results in [("SP Distance", sp_results), ("Query Distance", query_results)]:
            by_padding = defaultdict(list)
            for r in results:
                by_padding[r["_padding"]].append(r["overall"])

            pads = sorted(by_padding)
            if len(pads) >= 2:
                baseline = sum(by_padding[pads[0]]) / len(by_padding[pads[0]])
                worst = min(sum(by_padding[p]) / len(by_padding[p]) for p in pads)
                max_drop = baseline - worst
                print(f"  {label}: baseline={baseline:.3f}, worst={worst:.3f}, max_drop={max_drop:+.3f}")

        print()
        print("  If SP drop >> Query drop: model 'forgets' system prompt rules/procedures")
        print("  If Query drop >> SP drop: model 'forgets' what task it was asked to do")
        print("  If both similar: degradation is a general attention decay phenomenon")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
