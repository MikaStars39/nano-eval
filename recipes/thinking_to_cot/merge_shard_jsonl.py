import argparse
import json
from pathlib import Path
from typing import Set, Tuple


def merge_jsonl_by_id(input_dir: Path, pattern: str, output_file: Path, dedup: bool) -> Tuple[int, int, int]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern `{pattern}` in {input_dir}")

    seen: Set[str] = set()
    written = 0
    duplicate = 0
    bad_json = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fout:
        for fp in files:
            with fp.open("r", encoding="utf-8") as fin:
                for line in fin:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        bad_json += 1
                        continue
                    item_id = str(obj.get("id", ""))
                    if dedup and item_id:
                        if item_id in seen:
                            duplicate += 1
                            continue
                        seen.add(item_id)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
    return written, duplicate, bad_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard JSONL files into one JSONL.")
    parser.add_argument("--input-dir", required=True, help="Directory containing shard files.")
    parser.add_argument("--glob", default="final_*.jsonl", help="Glob pattern under input-dir.")
    parser.add_argument("--output", required=True, help="Merged output JSONL path.")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication by id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written, duplicate, bad = merge_jsonl_by_id(
        input_dir=Path(args.input_dir),
        pattern=args.glob,
        output_file=Path(args.output),
        dedup=not args.no_dedup,
    )
    print(f"[merge] written={written}, duplicate_skipped={duplicate}, bad_json={bad}, output={args.output}")


if __name__ == "__main__":
    main()

