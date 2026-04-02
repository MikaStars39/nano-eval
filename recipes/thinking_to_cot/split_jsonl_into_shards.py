import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, TextIO


def _pick_shard(item_id: str, num_shards: int) -> int:
    digest = hashlib.md5(item_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def split_jsonl(source_jsonl: Path, output_dir: Path, num_shards: int, shard_prefix: str) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [output_dir / f"{shard_prefix}_{idx:05d}.jsonl" for idx in range(num_shards)]
    shard_counts = [0 for _ in range(num_shards)]
    total = 0
    written = 0
    invalid = 0

    files: List[TextIO] = [path.open("w", encoding="utf-8") for path in shard_paths]
    try:
        with source_jsonl.open("r", encoding="utf-8") as fin:
            for line_idx, line in enumerate(fin):
                if not line.strip():
                    continue
                total += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue

                item_id = str(item.get("id", f"line-{line_idx}"))
                item["id"] = item_id
                shard_idx = _pick_shard(item_id=item_id, num_shards=num_shards)
                files[shard_idx].write(json.dumps(item, ensure_ascii=False) + "\n")
                shard_counts[shard_idx] += 1
                written += 1
    finally:
        for f in files:
            f.close()

    stats: Dict[str, int] = {"total": total, "written": written, "invalid_json": invalid}
    for idx, count in enumerate(shard_counts):
        stats[f"shard_{idx:05d}"] = count
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a JSONL dataset into N shard JSONL files.")
    parser.add_argument("--source-jsonl", required=True, help="Input JSONL to split.")
    parser.add_argument("--output-dir", required=True, help="Directory for shard JSONL files.")
    parser.add_argument("--num-shards", required=True, type=int, help="Number of shards.")
    parser.add_argument("--shard-prefix", default="shard", help="Shard file prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")

    source_jsonl = Path(args.source_jsonl)
    output_dir = Path(args.output_dir)
    stats = split_jsonl(source_jsonl=source_jsonl, output_dir=output_dir, num_shards=args.num_shards, shard_prefix=args.shard_prefix)
    print("[split] " + ", ".join([f"{k}={v}" for k, v in stats.items()]) + f", output_dir={output_dir}")


if __name__ == "__main__":
    main()

