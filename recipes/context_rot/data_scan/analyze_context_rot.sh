#!/usr/bin/env bash
# analyze_context_rot.sh — 分析 merged_result.jsonl 中的 context rot 分布

DATA_DIR="/jfs-dialogue-mmos-rs04/users/qingyu/data/context_rot/vulcan_110k_context_rot_data"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/analyze_context_rot.py" \
    --input "$DATA_DIR/merged_result.jsonl" \
    --save-csv "$DATA_DIR/context_rot_analysis.csv"
