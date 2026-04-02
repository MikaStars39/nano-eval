import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_last_human_message(conversations: Any) -> str:
    if not isinstance(conversations, list):
        return ""
    for message in reversed(conversations):
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", "")).strip().lower()
        if role in {"human", "user"}:
            return str(message.get("value", "")).strip()
    return ""


def _keep_only_system_and_human(conversations: Any) -> List[Dict[str, Any]]:
    if not isinstance(conversations, list):
        return []
    kept: List[Dict[str, Any]] = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", "")).strip().lower()
        if role in {"system", "human", "user"}:
            msg = dict(message)
            if role == "user":
                msg["from"] = "human"
            kept.append(msg)
    return kept


def normalize_stage1(raw_jsonl: Path, normalized_jsonl: Path) -> Tuple[int, int]:
    normalized_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0

    with raw_jsonl.open("r", encoding="utf-8") as fin, normalized_jsonl.open("w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin):
            if not line.strip():
                continue
            total += 1
            try:
                sample: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            conversations = sample.get("conversations")
            question = _extract_last_human_message(conversations)
            if not question:
                continue

            slimmed = {
                "conversations": _keep_only_system_and_human(conversations),
            }
            fout.write(json.dumps(slimmed, ensure_ascii=False) + "\n")
            written += 1

    return total, written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 preprocess: keep rows with valid human question.")
    parser.add_argument("--raw-jsonl", required=True)
    parser.add_argument("--normalized-jsonl", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total, written = normalize_stage1(Path(args.raw_jsonl), Path(args.normalized_jsonl))
    print(f"[preprocess-stage1] total={total}, written={written}, output={args.normalized_jsonl}")


if __name__ == "__main__":
    main()

