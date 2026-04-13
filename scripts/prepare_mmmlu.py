"""
Prepare MMMLU (Multilingual MMLU) dataset for nano-eval evaluation.

Converts all language CSV subsets from OpenAI MMMLU into a single mmmlu.jsonl file.

Dataset: https://huggingface.co/datasets/openai/MMMLU

Usage:
    python scripts/prepare_mmmlu.py \
        --dataset-path /path/to/MMMLU \
        --output-dir /path/to/nano-eval/

    # Only specific languages:
    python scripts/prepare_mmmlu.py \
        --dataset-path /path/to/MMMLU \
        --output-dir /path/to/nano-eval/ \
        --languages ZH-CN,JA-JP,KO-KR
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def format_prompt(question: str, options: Dict[str, str]) -> str:
    """Format question and options into MCQ prompt matching existing mmlu.jsonl style."""
    lines = [question, "", "Options:"]
    for letter in ("A", "B", "C", "D"):
        lines.append(f"{letter}. {options[letter]}")
    lines.append(
        r"Choose an answer in A,B,C,D. Answer with \boxed{{A}}, \boxed{{B}}, \boxed{{C}}, or \boxed{{D}}."
    )
    return "\n".join(lines)


def convert_row(row: Dict[str, str], language: str) -> Dict:
    """Convert a single MMMLU CSV row to nano-eval JSONL format."""
    options = {letter: row[letter] for letter in ("A", "B", "C", "D")}
    prompt = format_prompt(row["Question"], options)
    return {
        "question_id": f"mmmlu_{language}_{row.get('', '0')}_{row['Subject']}",
        "prompt": prompt,
        "label": row["Answer"],
        "language": language,
        "subset": row["Subject"],
    }


def prepare_mmmlu(
    dataset_path: str,
    output_dir: Path,
    languages: Optional[List[str]] = None,
) -> None:
    test_dir = Path(dataset_path) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    csv_files = sorted(test_dir.glob("mmlu_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No mmlu_*.csv files found in {test_dir}")

    # Extract available language codes from filenames (e.g., mmlu_ZH-CN.csv -> ZH-CN)
    available_langs = {f.stem.replace("mmlu_", ""): f for f in csv_files}
    logger.info("Available language subsets (%d): %s", len(available_langs), sorted(available_langs.keys()))

    if languages is not None:
        missing = set(languages) - set(available_langs.keys())
        if missing:
            raise ValueError(f"Requested languages not found in dataset: {missing}")
        selected = {lang: available_langs[lang] for lang in languages}
    else:
        selected = available_langs

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mmmlu.jsonl"

    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for lang in sorted(selected.keys()):
            csv_path = selected[lang]
            logger.info("Processing language: %s (%s)", lang, csv_path.name)
            count = 0
            with csv_path.open("r", encoding="utf-8") as csvf:
                reader = csv.DictReader(csvf)
                for row in reader:
                    converted = convert_row(row, lang)
                    f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
            logger.info("  %s: %d records", lang, count)
            total += count

    logger.info("Done. Total records: %d -> %s", total, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MMMLU for nano-eval")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Local path to MMMLU dataset directory (containing test/ subdirectory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/nano_eval"),
        help="Output directory for mmmlu.jsonl (default: outputs/nano_eval/)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes to include (default: all). E.g., ZH-CN,JA-JP,KO-KR",
    )
    args = parser.parse_args()

    selected_langs = None
    if args.languages:
        selected_langs = [lang.strip() for lang in args.languages.split(",") if lang.strip()]

    prepare_mmmlu(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        languages=selected_langs,
    )


if __name__ == "__main__":
    main()
