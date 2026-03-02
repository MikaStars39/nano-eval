import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

def load(split: str):
    """Load the HellaSwag dataset as an iterator of dicts."""
    dataset = load_dataset("/mnt/llm-train/users/explore-train/qingyu/.cache/hellaswag", split=split)
    for _, row in enumerate(tqdm(dataset, desc="Loading")):
        endings = row["endings"]
        if isinstance(endings, str):
            # Some exports may store endings as a serialized JSON string.
            endings = json.loads(endings)

        options_text = "\n".join(
            f"{i}. {ending}" for i, ending in enumerate(endings)
        )
        prompt = (
            f"Activity: {row['activity_label']}\n"
            f"Context: {row['ctx']}\n\n"
            "Choose the most plausible continuation from the options below.\n"
            f"{options_text}\n\n"
            "Answer with the option index only."
        )
        yield {
            "prompt": prompt,
            "label": str(row.get("label", "")),
        }

def main():
    """Parse command line args and write HellaSwag as JSONL."""
    parser = argparse.ArgumentParser(description="Export HellaSwag dataset to JSONL.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="HellaSwag split to export. Use validation/train for labels."
    )
    args = parser.parse_args()

    records = list(load(args.split))
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Written {len(records)} rows to {args.output}")

if __name__ == "__main__":
    main()