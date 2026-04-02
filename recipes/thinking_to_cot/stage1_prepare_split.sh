#!/usr/bin/env bash
set -euo pipefail

RAW_SOURCE_JSONL=/jpfs/qingyu/data/all_merged_3sources.jsonl
OUTPUT_DIR=/jpfs/qingyu/data/v32_two_stage/stage1
NUM_SHARDS=80

mkdir -p "$OUTPUT_DIR"
NORMALIZED_JSONL="$OUTPUT_DIR/source.normalized.jsonl"
SHARDS_DIR="$OUTPUT_DIR/shards"

echo "[stage1 step 1/2] preprocess..."
python recipe/thinking_to_cot/preprocess_stage1.py \
  --raw-jsonl "$RAW_SOURCE_JSONL" \
  --normalized-jsonl "$NORMALIZED_JSONL"

echo "[stage1 step 2/2] split..."
python recipe/thinking_to_cot/split_jsonl_into_shards.py \
  --source-jsonl "$NORMALIZED_JSONL" \
  --output-dir "$SHARDS_DIR" \
  --num-shards "$NUM_SHARDS" \
  --shard-prefix "shard"

echo "Done stage1 preprocess+split."

