#!/usr/bin/env bash
set -euo pipefail

STAGE1_MERGED_JSONL=/jpfs/qingyu/data/v32_two_stage/stage1/stage1_merged.jsonl
OUTPUT_DIR=/jpfs/qingyu/data/v32_two_stage/stage2
NUM_SHARDS=80

mkdir -p "$OUTPUT_DIR"
NORMALIZED_JSONL="$OUTPUT_DIR/source.normalized.jsonl"
SHARDS_DIR="$OUTPUT_DIR/shards"

echo "[stage2 step 1/2] preprocess..."
python recipe/thinking_to_cot/preprocess_stage2.py \
  --raw-jsonl "$STAGE1_MERGED_JSONL" \
  --normalized-jsonl "$NORMALIZED_JSONL"

echo "[stage2 step 2/2] split..."
python recipe/thinking_to_cot/split_jsonl_into_shards.py \
  --source-jsonl "$NORMALIZED_JSONL" \
  --output-dir "$SHARDS_DIR" \
  --num-shards "$NUM_SHARDS" \
  --shard-prefix "shard"

echo "Done stage2 preprocess+split."

