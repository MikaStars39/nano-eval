import os
import json
import glob

from mpmath import j

def process_model_outputs(input_dir: str, output_file_with_think: str, output_file_without_think: str):
    pattern = os.path.join(input_dir, "model_output_*.jsonl")
    matched_files = glob.glob(pattern)

    with open(output_file_with_think, "w", encoding="utf-8") as f_out_think, \
        open(output_file_without_think, "w", encoding="utf-8") as f_out_nothink:
        for file in matched_files:
            with open(file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    content = json.loads(line)
                    prompt = content.get("prompt")
                    response = content.get("response")

                    if "</think>" in response:
                        response = response.split("</think>")[-1]
                        output = [
                            {"from": "system", "value": ""}, 
                            {"from": "human", "value": prompt}, 
                            {"from": "gpt", "value": response}
                        ]
                        f_out_think.write(json.dumps(output, ensure_ascii=False) + "\n")
                    else:
                        output = [
                            {"from": "system", "value": ""}, 
                            {"from": "human", "value": prompt}, 
                            {"from": "gpt", "value": response}
                        ]
                        f_out_nothink.write(json.dumps(output, ensure_ascii=False) + "\n")
    

if __name__ == "__main__":
    # 替换为你实际存放文件的目录路径
    TARGET_DIR = "/jpfs/qingyu/data/v32_two_stage/stage2/shards_out/" 
    OUT_THINK = "s"
    OUT_NO_THINK = "/jpfs/qingyu/data/v32_two_stage/processed_without_think.jsonl"
    
    process_model_outputs(TARGET_DIR, OUT_THINK, OUT_NO_THINK)