#!/bin/bash
# LLM Judge Scoring System - Example Run Script

# Configuration paths
INPUT_FILE="/mnt/llm-train/users/explore-train/wangzhenfang8/codes/generate/data/0309-distill-amthinking/deploy-sft-128k-s2-0318-s2-fixswe-shuf-64-1e-5-min1e-6/output_remaining.jsonl"
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/nano-eval/output/dpo_data_0319_yq4"
JUDGE_MODEL="/jpfs/models/DeepSeek-V3.2"
SCRIPT_DIR="/jpfs/qingyu/nano-eval"

python $SCRIPT_DIR/recipes/llm_judge/prepare_judge.py \
    --input $INPUT_FILE \
    --output "$OUTPUT_DIR/prepare.jsonl" \
    --tokenizer $JUDGE_MODEL \
    --workers 128 \
    --n_examples 4

python $SCRIPT_DIR/recipes/llm_judge/shard_jsonl.py \
    --input $OUTPUT_DIR/prepare.jsonl \
    --output-dir "$OUTPUT_DIR/shards_233_yuqi_new_data" \
    --num-shards 16