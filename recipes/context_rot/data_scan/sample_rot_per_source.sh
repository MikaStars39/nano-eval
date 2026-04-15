#!/usr/bin/env bash
# sample_rot_per_source.sh — 每个源抽一条偷懒样本（含完整 trajectory）

DATA_DIR="/jfs-dialogue-mmos-rs04/users/qingyu/data/context_rot/vulcan_110k_context_rot_data"
ORIGINAL="/jfs-dialogue-mmos-rs04/users/qingyu/data/context_rot/full_flagged.jsonl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/sample_rot_per_source.py" \
    --input    "$DATA_DIR/merged_result.jsonl" \
    --original "$ORIGINAL" \
    --save     "$DATA_DIR/rot_samples.jsonl"
