"""
Prepare MMLU-ProX dataset for nano-eval evaluation.

Converts all language subsets from MMLU-ProX into a single mmlu_prox.jsonl file.

Usage:
    python scripts/prepare_mmlu_prox.py \
        --dataset-path /path/to/MMLU-ProX \
        --output-dir outputs/nano_eval/

    # Only specific languages:
    python scripts/prepare_mmlu_prox.py \
        --dataset-path /path/to/MMLU-ProX \
        --output-dir outputs/nano_eval/ \
        --languages zh,en,fr
"""

from __future__ import annotations

import argparse
import json
import logging
import string
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LETTERS = list(string.ascii_uppercase)  # A, B, C, ..., Z


def format_prompt(question: str, options: List[str]) -> str:
    """Format question and options into MCQ prompt."""
    lines = [question, ""]
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}. {opt}")
    lines.append("")
    lines.append(r"Please select the correct answer and put the letter in \boxed{}, e.g., \boxed{A}.")
    return "\n".join(lines)


def convert_record(record: Dict, language: str) -> Dict:
    """Convert a single MMLU-ProX record to nano-eval JSONL format."""
    options = []
    for i in range(10):
        val = record.get(f"option_{i}")
        if val is not None:
            options.append(str(val))

    prompt = format_prompt(record["question"], options)
    question_id = f"{language}_{record['question_id_src']}"

    return {
        "question_id": question_id,
        "prompt": prompt,
        "label": record["answer"],
        "language": language,
        "category": record.get("category", ""),
        "src": record.get("src", ""),
    }


def prepare_mmlu_prox(
    dataset_path: str,
    output_dir: Path,
    languages: Optional[List[str]] = None,
) -> None:
    from datasets import get_dataset_config_names, load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mmlu_prox.jsonl"

    available_langs = get_dataset_config_names(dataset_path)
    logger.info("Available language subsets (%d): %s", len(available_langs), available_langs)

    if languages is not None:
        missing = set(languages) - set(available_langs)
        if missing:
            raise ValueError(f"Requested languages not found in dataset: {missing}")
        selected_langs = languages
    else:
        selected_langs = available_langs

    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for lang in selected_langs:
            logger.info("Processing language: %s", lang)
            dataset = load_dataset(dataset_path, lang, split="test")
            count = 0
            for record in dataset:
                converted = convert_record(record, lang)
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
            logger.info("  %s: %d records", lang, count)
            total += count

    logger.info("Done. Total records: %d -> %s", total, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MMLU-ProX for nano-eval")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Local path to MMLU-ProX dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/nano_eval"),
        help="Output directory for mmlu_prox.jsonl (default: outputs/nano_eval/)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes to include (default: all). E.g., zh,en,fr",
    )
    args = parser.parse_args()

    selected_langs = None
    if args.languages:
        selected_langs = [lang.strip() for lang in args.languages.split(",") if lang.strip()]

    prepare_mmlu_prox(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        languages=selected_langs,
    )


if __name__ == "__main__":
    main()
