#!/usr/bin/env bash
# run_scan.sh — Training data lazy pattern scan (two-phase pipeline)
#
# Phase 1: Rule-based keyword scan (multiprocess, no API needed)
# Phase 2: LLM judge verification (uses nano-eval's OnlineBatchInferenceEngine)
#
# Required env vars (Phase 2 only):
#   JUDGE_API_KEY, JUDGE_API_BASE, JUDGE_MODEL
#
# Usage:
#   # Phase 1 only (no API needed)
#   bash recipes/context_rot/data_scan/run_scan.sh --input-list /path/to/filelist.txt
#
#   # Both phases
#   export JUDGE_API_KEY=sk-xxx JUDGE_API_BASE=https://... JUDGE_MODEL=gpt-5.3-codex
#   bash recipes/context_rot/data_scan/run_scan.sh --input-list /path/to/filelist.txt --phase2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUTPUT_DIR="${REPO_ROOT}/outputs/context_rot/scan_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Parse args
INPUT_LIST=""
RUN_PHASE2=false
MIN_TOKENS="${MIN_TOKENS:-20000}"
WORKERS="${WORKERS:-8}"
CONCURRENCY="${CONCURRENCY:-16}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-list) INPUT_LIST="$2"; shift 2 ;;
        --phase2) RUN_PHASE2=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT_LIST" ]; then
    echo "Usage: bash run_scan.sh --input-list <filelist.txt> [--phase2]"
    exit 1
fi

echo "=== Training Data Lazy Pattern Scan ==="
echo "Output: $OUTPUT_DIR"
echo ""

# Phase 1: Rule-based scan
echo ">>> Phase 1: Rule-based scanning..."
python3 "$SCRIPT_DIR/scan_rules.py" \
    --input-list "$INPUT_LIST" \
    --output "$OUTPUT_DIR/flagged.jsonl" \
    --min-tokens "$MIN_TOKENS" \
    --workers "$WORKERS" \
    2>&1 | tee "$OUTPUT_DIR/scan.log"

echo ""
echo "Phase 1 complete. Flagged results: $OUTPUT_DIR/flagged.jsonl"

# Phase 2: LLM Judge (optional)
if [ "$RUN_PHASE2" = true ]; then
    echo ""
    echo ">>> Phase 2: LLM Judge verification..."
    python3 "$SCRIPT_DIR/scan_judge.py" \
        --input "$OUTPUT_DIR/flagged.jsonl" \
        --output "$OUTPUT_DIR/judged.jsonl" \
        --judge-model "${JUDGE_MODEL:?Set JUDGE_MODEL}" \
        --judge-api-base "${JUDGE_API_BASE:?Set JUDGE_API_BASE}" \
        --judge-api-key "${JUDGE_API_KEY:?Set JUDGE_API_KEY}" \
        --concurrency "$CONCURRENCY" \
        2>&1 | tee "$OUTPUT_DIR/judge.log"
else
    echo ""
    echo "To run Phase 2 (LLM Judge), execute:"
    echo ""
    echo "  export JUDGE_API_KEY=sk-xxx JUDGE_API_BASE=https://... JUDGE_MODEL=gpt-5.3-codex"
    echo "  python3 $SCRIPT_DIR/scan_judge.py \\"
    echo "    --input $OUTPUT_DIR/flagged.jsonl \\"
    echo "    --output $OUTPUT_DIR/judged.jsonl \\"
    echo "    --judge-model \$JUDGE_MODEL \\"
    echo "    --judge-api-base \$JUDGE_API_BASE \\"
    echo "    --judge-api-key \$JUDGE_API_KEY \\"
    echo "    --concurrency 16"
fi
