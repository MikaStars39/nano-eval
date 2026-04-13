#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/context_rot/scan_${TIMESTAMP}"
mkdir -p "${WORKDIR}"

SCAN_ARGS=(
    --input-list "${INPUT_LIST:?Set INPUT_LIST}"
    --output "${WORKDIR}/flagged.jsonl"
    --min-tokens "${MIN_TOKENS:-20000}"
    --workers "${WORKERS:-8}"
)

python3 "$SCRIPT_DIR/scan_rules.py" "${SCAN_ARGS[@]}" 2>&1 | tee "${WORKDIR}/scan.log"

# Phase 2: LLM judge (runs when JUDGE_MODEL is set)
if [ -n "${JUDGE_MODEL:-}" ]; then
    JUDGE_ARGS=(
        --input "${WORKDIR}/flagged.jsonl"
        --output "${WORKDIR}/judged.jsonl"
        --judge-model "${JUDGE_MODEL}"
        --judge-api-base "${JUDGE_API_BASE:?Set JUDGE_API_BASE}"
        --judge-api-key "${JUDGE_API_KEY:?Set JUDGE_API_KEY}"
        --concurrency "${CONCURRENCY:-16}"
    )

    python3 "$SCRIPT_DIR/scan_judge.py" "${JUDGE_ARGS[@]}" 2>&1 | tee "${WORKDIR}/judge.log"
fi
