"""
Ray utilities: initialization, JSONL shard splitting and merging.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List


def init_ray(address: str = "auto", **kwargs: Any) -> None:
    """Initialize Ray idempotently with sensible defaults."""
    import ray

    if ray.is_initialized():
        return
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level="ERROR",
        **kwargs,
    )


def shard_jsonl(input_file: str, num_actors: int, output_dir: str) -> List[str]:
    """Split a JSONL file into *num_actors* roughly equal shard files.

    Returns a list of shard file paths (only non-empty shards are created).
    """
    os.makedirs(output_dir, exist_ok=True)
    lines = Path(input_file).read_text("utf-8").splitlines()
    size = math.ceil(len(lines) / max(1, num_actors))
    paths: List[str] = []
    for i in range(num_actors):
        chunk = lines[i * size : (i + 1) * size]
        if not chunk:
            continue
        p = os.path.join(output_dir, f"shard_{i:05d}.jsonl")
        Path(p).write_text("\n".join(chunk) + "\n", "utf-8")
        paths.append(p)
    return paths


def merge_jsonl(shard_paths: List[str], output_file: str) -> int:
    """Merge shard JSONL files into a single output file.

    Returns the total number of lines written.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for p in sorted(shard_paths):
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n
