#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SCAN_DIR="${SCAN_DIR:?Set SCAN_DIR (e.g. /jfs-dialogue-mmos-rs04/users/qingyu/data/context_rot/scan_results/20260410-082039)}"
INPUT_FILE="${SCAN_DIR}/flagged.jsonl"
OUTPUT_FILE="$(dirname "$SCAN_DIR")/full_flagged.jsonl"

python3 "$SCRIPT_DIR/map_jsonl.py" \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --workers "${WORKERS:-8}"
