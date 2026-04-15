#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INPUT="${INPUT:?Set INPUT (path to full_flagged.jsonl)}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR (path to output directory)}"
SHARD_SIZE="${SHARD_SIZE:-5000}"

python3 "$SCRIPT_DIR/prepare_vulcan.py" \
    --input "$INPUT" \
    --output "$OUTPUT_DIR" \
    --shard-size "$SHARD_SIZE"
