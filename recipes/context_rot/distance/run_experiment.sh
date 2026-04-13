#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_DIR="$SCRIPT_DIR/../eval"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/context_rot/distance_${TIMESTAMP}"
mkdir -p "${WORKDIR}"

EXPERIMENT="${EXPERIMENT:-both}"
FILTER_IDS="${FILTER_IDS:-case_0_P1,case_2_P1,case_8_P1}"
PADDING_LEVELS="${PADDING_LEVELS:-0,20,50,100,200}"

# Step 1: Generate eval sets
python3 "$SCRIPT_DIR/make_eval.py" \
    --experiment "$EXPERIMENT" \
    --input "${EVAL_SET:?Set EVAL_SET}" \
    --output-dir "$WORKDIR" \
    --filter-id "$FILTER_IDS" \
    --padding-levels "$PADDING_LEVELS"

# Step 2: Run evaluations
EVAL_ARGS=(
    --model "${MODEL:?Set MODEL}"
    --judge-model "${JUDGE_MODEL:?Set JUDGE_MODEL}"
    --judge-api-base "${JUDGE_API_BASE:?Set JUDGE_API_BASE}"
    --judge-api-key "${JUDGE_API_KEY:?Set JUDGE_API_KEY}"
    --api-base "${API_BASE:?Set API_BASE}"
    --api-key "${API_KEY:?Set API_KEY}"
    --concurrency "${CONCURRENCY:-8}"
    --max-turns "${MAX_TURNS:-10}"
    --limit "${LIMIT:-200}"
)

if [ -n "${JUDGE_EXTRA_HEADERS:-}" ]; then
    EVAL_ARGS+=(--judge-extra-headers "$JUDGE_EXTRA_HEADERS")
fi

if [[ "$EXPERIMENT" == "sp" || "$EXPERIMENT" == "both" ]]; then
    if [ -f "$WORKDIR/eval_set_sp_distance.jsonl" ]; then
        python3 "$EVAL_DIR/run_eval.py" \
            "${EVAL_ARGS[@]}" \
            --input "$WORKDIR/eval_set_sp_distance.jsonl" \
            --output "$WORKDIR/results_sp_distance.jsonl" \
            2>&1 | tee "$WORKDIR/eval_sp.log"
    fi
fi

if [[ "$EXPERIMENT" == "query" || "$EXPERIMENT" == "both" ]]; then
    if [ -f "$WORKDIR/eval_set_query_distance.jsonl" ]; then
        python3 "$EVAL_DIR/run_eval.py" \
            "${EVAL_ARGS[@]}" \
            --input "$WORKDIR/eval_set_query_distance.jsonl" \
            --output "$WORKDIR/results_query_distance.jsonl" \
            2>&1 | tee "$WORKDIR/eval_query.log"
    fi
fi

# Step 3: Analyze
python3 "$SCRIPT_DIR/analyze.py" "$WORKDIR"
