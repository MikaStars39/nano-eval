#!/bin/bash

DATA_DIR=/jfs-dialogue-mmos-rs04/users/qingyu/data/context_rot/vulcan_110k_context_rot_data
ORIGINAL=/jfs-dialogue-mmos-rs04/users/qingyu/data/context_rot/full_flagged.jsonl
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

python "$SCRIPT_DIR/merge_judge_output.py" \
    --input-dir  "$DATA_DIR" \
    --output-dir "$DATA_DIR/output" \
    --original   "$ORIGINAL" \
    --save       "$DATA_DIR/merged_result.jsonl"
