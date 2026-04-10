#!/usr/bin/env bash
# run_eval.sh — Run Context Rot evaluation
#
# Required env vars:
#   API_KEY       — Model API key
#   API_BASE      — Model API base URL (e.g., https://api.minimaxi.com/v1)
#   MODEL         — Model name (e.g., MiniMax-M2.7)
#   JUDGE_API_KEY — Judge API key
#   JUDGE_API_BASE— Judge API base URL
#   JUDGE_MODEL   — Judge model name (e.g., gpt-5.3-codex)
#   INPUT         — Input eval_set.jsonl path
#
# Optional env vars:
#   CONCURRENCY   — Max concurrent test points (default: 16)
#   MAX_TURNS     — Max agent loop turns (default: 10)
#   LIMIT         — Max test points to run (default: all)
#   JUDGE_EXTRA_HEADERS — Extra headers for judge API as JSON string
#
# Usage:
#   export API_KEY=sk-xxx API_BASE=https://... MODEL=MiniMax-M2.7
#   export JUDGE_API_KEY=sk-xxx JUDGE_API_BASE=https://... JUDGE_MODEL=gpt-5.3-codex
#   export INPUT=/path/to/eval_set.jsonl
#   bash recipes/context_rot/eval/run_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${REPO_ROOT}/outputs/context_rot/${MODEL:-model}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

EVAL_ARGS=(
    --model "${MODEL:?Set MODEL}"
    --judge-model "${JUDGE_MODEL:?Set JUDGE_MODEL}"
    --judge-api-base "${JUDGE_API_BASE:?Set JUDGE_API_BASE}"
    --judge-api-key "${JUDGE_API_KEY:?Set JUDGE_API_KEY}"
    --api-base "${API_BASE:?Set API_BASE}"
    --api-key "${API_KEY:?Set API_KEY}"
    --input "${INPUT:?Set INPUT}"
    --output "$RUN_DIR/results.jsonl"
    --concurrency "${CONCURRENCY:-16}"
    --max-turns "${MAX_TURNS:-10}"
)

if [ -n "${LIMIT:-}" ]; then
    EVAL_ARGS+=(--limit "$LIMIT")
fi

if [ -n "${JUDGE_EXTRA_HEADERS:-}" ]; then
    EVAL_ARGS+=(--judge-extra-headers "$JUDGE_EXTRA_HEADERS")
fi

echo "========================================"
echo "Context Rot Evaluation"
echo "========================================"
echo "  Model:   ${MODEL}"
echo "  Judge:   ${JUDGE_MODEL}"
echo "  Input:   ${INPUT}"
echo "  Output:  $RUN_DIR"
echo "========================================"

python3 "$SCRIPT_DIR/run_eval.py" "${EVAL_ARGS[@]}" 2>&1 | tee "$RUN_DIR/eval.log"

echo ""
echo "--- Generating report ---"
python3 "$SCRIPT_DIR/report.py" --input "$RUN_DIR/results.jsonl" 2>&1 | tee "$RUN_DIR/report.txt"

echo ""
echo "Done. Results in: $RUN_DIR"
