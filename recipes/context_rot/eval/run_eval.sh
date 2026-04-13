#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/context_rot/${MODEL:?Set MODEL}_${TIMESTAMP}"
mkdir -p "${WORKDIR}"

EVAL_ARGS=(
    --model "${MODEL}"
    --judge-model "${JUDGE_MODEL:?Set JUDGE_MODEL}"
    --judge-api-base "${JUDGE_API_BASE:?Set JUDGE_API_BASE}"
    --judge-api-key "${JUDGE_API_KEY:?Set JUDGE_API_KEY}"
    --api-base "${API_BASE:?Set API_BASE}"
    --api-key "${API_KEY:?Set API_KEY}"
    --input "${INPUT:?Set INPUT}"
    --output "${WORKDIR}/results.jsonl"
    --concurrency "${CONCURRENCY:-16}"
    --max-turns "${MAX_TURNS:-10}"
)

if [ -n "${LIMIT:-}" ]; then
    EVAL_ARGS+=(--limit "$LIMIT")
fi

if [ -n "${JUDGE_EXTRA_HEADERS:-}" ]; then
    EVAL_ARGS+=(--judge-extra-headers "$JUDGE_EXTRA_HEADERS")
fi

python3 "$SCRIPT_DIR/run_eval.py" "${EVAL_ARGS[@]}" 2>&1 | tee "${WORKDIR}/eval.log"

python3 "$SCRIPT_DIR/report.py" --input "${WORKDIR}/results.jsonl" 2>&1 | tee "${WORKDIR}/report.txt"
