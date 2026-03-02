import json
import argparse
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Official CEVAL subset names.
CEVAL_SUBSETS = [
    'agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics',
    'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture',
    'chinese_foreign_policy', 'chinese_history', 'chinese_literature',
    'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science',
    'college_education', 'college_engineering_hydrology', 'college_law',
    'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
    'computer_security', 'conceptual_physics', 'construction_project_management', 'economics',
    'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense',
    'elementary_information_and_technology', 'elementary_mathematics', 'ethnology', 'food_science',
    'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry',
    'high_school_geography', 'high_school_mathematics', 'high_school_physics',
    'high_school_politics', 'human_sexuality', 'international_law', 'journalism',
    'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management',
    'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy',
    'professional_accounting', 'professional_law', 'professional_medicine',
    'professional_psychology', 'public_relations', 'security_study', 'sociology',
    'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions'
]


def _decode_line(raw_line: bytes):
    """
    Decode one jsonl line with common encodings.
    Return decoded text or None when decoding fails.
    """
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw_line.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _read_jsonl_safe(data_path: Path, subset_name: str):
    """
    Read jsonl robustly and keep valid rows even if partial corruption exists.
    """
    try:
        raw = data_path.read_bytes()
    except Exception as e:
        print(f"加载子集{subset_name}失败: {e}")
        return []

    if not raw.strip():
        print(f"加载子集{subset_name}失败: 文件为空")
        return []

    rows = []
    bad_lines = 0
    for raw_line in raw.splitlines():
        if not raw_line.strip():
            continue
        decoded_line = _decode_line(raw_line)
        if decoded_line is None:
            bad_lines += 1
            continue
        try:
            rows.append(json.loads(decoded_line))
        except json.JSONDecodeError:
            bad_lines += 1

    if bad_lines > 0:
        # Keep execution stable by skipping broken lines instead of dropping a whole subset.
        print(f"子集{subset_name}存在{bad_lines}条损坏样本，已跳过")
    return rows


def load():
    """
    Read all CEVAL subset jsonl files and merge records.
    """
    subset_names = CEVAL_SUBSETS
    all_records = []
    for subset_name in tqdm(subset_names, desc="Loading subsets"):
        data_path = Path(f"/mnt/llm-train/users/explore-train/qingyu/.cache/cmmlu/data/test/{subset_name}.jsonl")
        rows = _read_jsonl_safe(data_path=data_path, subset_name=subset_name)
        if not rows:
            continue

        for row in rows:
            question = row.get("question", "")
            # Support up to 26 options (A-Z).
            options = []
            for idx, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                try:
                    options.append(f"{letter}. {row['choices'][idx]}")
                except Exception:
                    break
            if options:
                prompt = f"{question}\n\nOptions:\n" + "\n".join(options)
            else:
                prompt = question

            all_records.append({
                "prompt": prompt + "\nChoose an answer in A,B,C,D. Answer with \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, or \\boxed{{D}}",
                "label": row.get("answer", ""),
                "subset": row.get("subject", subset_name),
            })
    return all_records

def main():
    """
    Export merged CEVAL subsets to a jsonl file.
    """
    parser = argparse.ArgumentParser(description="Export CEVAL subsets to JSONL.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="JSONL输出文件路径"
    )
    args = parser.parse_args()

    records = load()
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"已写入{len(records)}行至{args.output}")

if __name__ == "__main__":
    main()