#!/usr/bin/env python3
"""
scan_rules.py — Rule-based scanner for lazy patterns in M2.7 training data

Phase 1 of the training data quality scan:
1. Estimate token length per trajectory (chars / 2.5)
2. Filter trajectories >= min_tokens threshold
3. Search for lazy keywords in reasoning_content and content
4. Output flagged examples to JSONL
"""

import argparse
import json
import sys
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, Queue
from pathlib import Path


# ── Lazy keyword lists ──────────────────────────────────────────

LAZY_KEYWORDS = [
    # ── 中文 ──────────────────────────────────────────────

    # 虚构时间/token压力
    "赶时间",
    "时间有限",
    "时间和token",
    "由于时间",
    "token限制",
    "token不够",
    "篇幅有限",
    "受限于篇幅",
    "受限于长度",
    "字数限制",
    "空间有限",

    # 跳过/省略/简化
    "简化步骤",
    "简化处理",
    "简化流程",
    "批量处理",
    "不再逐一",
    "不再逐个",
    "不再逐条",
    "跳过",
    "省略",
    "以此类推",
    "依次类推",
    "限于篇幅",
    "为简洁起见",
    "不需要那么详细",
    "不用太详细",
    "直接给出结果",
    "直接给出结论",
    "不一一列举",
    "不再赘述",
    "不再展开",
    "不做详细",
    "这里不展开",
    "后续类似",
    "同理可得",
    "其余类似",
    "剩余的类似",

    # 放弃/搁置任务
    "留给用户",
    "后续再",
    "暂不处理",
    "待后续",

    # ── English ───────────────────────────────────────────

    # Fabricated time/token pressure
    "running out of tokens",
    "running out of space",
    "running low on",
    "token budget",
    "token limit",
    "space constraints",
    "length constraints",
    "due to length",
    "due to space",
    "context limit",
    "response limit",
    "nearing the limit",
    "limited space",
    "limited tokens",

    # Skipping / abbreviating
    "for brevity",
    "for the sake of brevity",
    "to keep this brief",
    "to keep it brief",
    "to save space",
    "in the interest of time",
    "I'll skip",
    "I'll abbreviate",
    "I'll summarize the rest",
    "I'll just provide",
    "skip the details",
    "skip the rest",
    "won't go through each",
    "won't enumerate",
    "won't list every",
    "instead of going through",
    "rather than listing",
    "without going into detail",

    # Vague/generic substitution
    "similar to above",
    "same as above",
    "same approach",
    "similar pattern",
    "following the same pattern",
    "repeat the same",
    "and so on",
    "and so forth",
    "et cetera",

    # Task abandonment
    "leave that for later",
    "beyond the scope",
    "I'll leave this",
    "out of scope",
    "won't cover",
    "I'll defer",
    "left as an exercise",
]

CHAR_PER_TOKEN = 2.5


# ── Core scanning logic ──────────────────────────────────────────

def estimate_tokens(messages: list[dict]) -> tuple[int, int]:
    """Estimate token count from message content. Returns (estimated_tokens, total_chars)."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            # Some messages have content as a list of dicts
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        total_chars += len(content)
        reasoning = msg.get("reasoning_content") or ""
        total_chars += len(reasoning)
    return int(total_chars / CHAR_PER_TOKEN), total_chars


def extract_snippet(text: str, keyword: str, context_chars: int = 80) -> str:
    """Extract a snippet around the keyword match."""
    idx = text.find(keyword)
    if idx == -1:
        return ""
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(keyword) + context_chars)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def scan_messages_for_keywords(messages: list[dict]) -> list[dict]:
    """Scan assistant messages for lazy keywords. Returns list of hits."""
    hits = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        for field in ("reasoning_content", "content"):
            text = msg.get(field) or ""
            if isinstance(text, list):
                text = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in text
                )
            if not text:
                continue

            for keyword in LAZY_KEYWORDS:
                if keyword in text:
                    hits.append({
                        "keyword": keyword,
                        "field": field,
                        "msg_index": i,
                        "snippet": extract_snippet(text, keyword),
                    })

    return hits


def scan_single_example(example: dict, line_number: int, source_file: str,
                        min_tokens: int) -> dict | None:
    """Scan a single training example. Returns flagged result or None."""
    messages = example.get("messages", [])
    if not messages:
        return None

    estimated_tokens, total_chars = estimate_tokens(messages)
    if estimated_tokens < min_tokens:
        return None

    keyword_hits = scan_messages_for_keywords(messages)
    if not keyword_hits:
        return None

    n_assistant = sum(1 for m in messages if m.get("role") == "assistant")

    meta = example.get("meta", example.get("metadata", {}))
    if isinstance(meta, dict):
        # Only keep small meta fields
        meta = {k: v for k, v in meta.items()
                if isinstance(v, (str, int, float, bool)) and len(str(v)) < 200}
    else:
        meta = {}

    return {
        "source_file": source_file,
        "line_number": line_number,
        "estimated_tokens": estimated_tokens,
        "total_chars": total_chars,
        "n_messages": len(messages),
        "n_assistant_messages": n_assistant,
        "n_keyword_hits": len(keyword_hits),
        "keyword_hits": keyword_hits,
        "meta": meta,
    }


# ── File-level processing ──────────────────────────────────────────

def scan_file(args: tuple) -> dict:
    """Process a single JSONL file. Called by multiprocessing pool."""
    filepath, min_tokens = args
    results = []
    stats = {
        "file": filepath,
        "total_lines": 0,
        "parse_errors": 0,
        "long_trajectories": 0,
        "flagged": 0,
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stats["total_lines"] += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    stats["parse_errors"] += 1
                    continue

                messages = example.get("messages", [])
                if not messages:
                    del example
                    continue

                # Quick char-count check before full scan
                estimated_tokens, _ = estimate_tokens(messages)
                if estimated_tokens >= min_tokens:
                    stats["long_trajectories"] += 1
                    result = scan_single_example(example, line_no, filepath, min_tokens)
                    if result:
                        stats["flagged"] += 1
                        results.append(result)

                del example

    except Exception as e:
        stats["error"] = str(e)

    return {"stats": stats, "results": results}


# ── Main ──────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Scan M2.7 training data for lazy patterns")
    parser.add_argument("--input-list", required=True, help="File listing JSONL paths (one per line)")
    parser.add_argument("--output", required=True, help="Output JSONL path for flagged examples")
    parser.add_argument("--min-tokens", type=int, default=20000,
                        help="Minimum estimated tokens to consider (default: 20000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    # Load file list
    with open(args.input_list) as f:
        files = [line.strip() for line in f if line.strip()]

    # Verify files exist
    missing = [fp for fp in files if not os.path.isfile(fp)]
    if missing:
        log(f"WARNING: {len(missing)} files not found, skipping:")
        for fp in missing[:10]:
            log(f"  {fp}")
        files = [fp for fp in files if os.path.isfile(fp)]

    log(f"Scanning {len(files)} files with {args.workers} workers, min_tokens={args.min_tokens}")
    log(f"Output: {args.output}")
    log(f"Keywords: {len(LAZY_KEYWORDS)}")

    # Prepare output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Run parallel scan
    work_items = [(fp, args.min_tokens) for fp in files]
    total_stats = defaultdict(int)
    all_results = []

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(scan_file, work_items)):
            stats = result["stats"]
            file_results = result["results"]

            total_stats["total_files"] += 1
            total_stats["total_lines"] += stats["total_lines"]
            total_stats["parse_errors"] += stats["parse_errors"]
            total_stats["long_trajectories"] += stats["long_trajectories"]
            total_stats["flagged"] += stats["flagged"]

            all_results.extend(file_results)

            # Progress
            if stats.get("error"):
                log(f"  [{i+1}/{len(files)}] ERROR {stats['file']}: {stats['error']}")
            elif stats["flagged"] > 0:
                log(f"  [{i+1}/{len(files)}] {os.path.basename(stats['file'])}: "
                    f"{stats['total_lines']} lines, {stats['long_trajectories']} long, "
                    f"{stats['flagged']} flagged")
            else:
                log(f"  [{i+1}/{len(files)}] {os.path.basename(stats['file'])}: "
                    f"{stats['total_lines']} lines, {stats['long_trajectories']} long, 0 flagged")

    # Sort by number of keyword hits (descending)
    all_results.sort(key=lambda x: x["n_keyword_hits"], reverse=True)

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    log("")
    log("=" * 60)
    log("SCAN COMPLETE")
    log(f"  Files scanned:       {total_stats['total_files']}")
    log(f"  Total examples:      {total_stats['total_lines']}")
    log(f"  Parse errors:        {total_stats['parse_errors']}")
    log(f"  Long trajectories:   {total_stats['long_trajectories']} (>= {args.min_tokens} est. tokens)")
    log(f"  Flagged (with lazy): {total_stats['flagged']}")
    log(f"  Output written to:   {args.output}")
    log("=" * 60)

    # Keyword frequency summary
    keyword_freq = defaultdict(int)
    for r in all_results:
        for hit in r["keyword_hits"]:
            keyword_freq[hit["keyword"]] += 1

    if keyword_freq:
        log("")
        log("Keyword frequency (top 20):")
        for kw, cnt in sorted(keyword_freq.items(), key=lambda x: -x[1])[:20]:
            log(f"  {kw:20s}  {cnt}")

    # Per-file summary
    file_counts = defaultdict(int)
    for r in all_results:
        file_counts[r["source_file"]] += 1

    if file_counts:
        log("")
        log("Per-file flagged counts (top 20):")
        for fp, cnt in sorted(file_counts.items(), key=lambda x: -x[1])[:20]:
            log(f"  {cnt:5d}  {os.path.basename(fp)}")


if __name__ == "__main__":
    main()
