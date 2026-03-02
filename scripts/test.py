import os
import json

def process_jsonl_file(filepath):
    """
    Replace '{problem} ' with '\n' in the 'prompt' field of each JSON object in a JSONL file,
    count the number of replacements, and overwrite the file if any replacements are made.
    """
    modified_lines = []
    replace_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "prompt" in data and "{problem} " in data["prompt"]:
                # 仅统计有替换的行
                data["prompt"] = data["prompt"].replace("{problem} ", "\n")
                replace_count += 1
            modified_lines.append(data)

    if replace_count > 0:
        # 只在有修改时覆盖原文件，防止无意义写入
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in modified_lines:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return replace_count

def scan_and_process_jsonl(directory):
    """
    Traverse all JSONL files in the given directory, replace patterns as needed, and print replacement stats.
    """
    stats = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            count = process_jsonl_file(filepath)
            stats[filename] = count
            print(f"{filename}: 替换 {count} 次")
    print("处理完成。处理文件数量:", len(stats))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="批量修正目录下所有JSONL文件的prompt字段内容。")
    parser.add_argument("--dir", required=True, help="指定包含jsonl文件的目录")
    args = parser.parse_args()
    scan_and_process_jsonl(args.dir)