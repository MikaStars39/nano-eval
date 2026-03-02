import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

def load():
    """Load the AIME 2024 dataset as an iterator of dicts."""
    dataset = load_dataset("/mnt/llm-train/users/explore-train/qingyu/.cache/aime_2024", split="train")
    for idx, row in enumerate(tqdm(dataset, desc=f"Loading")):
        yield {
            "prompt": row["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            "label": row["answer"],
        }

def main():
    """Parse command line args and write AIME2024 as JSONL."""
    parser = argparse.ArgumentParser(description="Export AIME2024 dataset to JSONL.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file."
    )
    args = parser.parse_args()

    records = list(load())
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Written {len(records)} rows to {args.output}")

if __name__ == "__main__":
    main()