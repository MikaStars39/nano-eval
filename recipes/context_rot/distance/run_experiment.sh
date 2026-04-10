#!/usr/bin/env bash
# run_experiment.sh — Run SP/Query distance sensitivity experiments
#
# Required env vars:
#   API_KEY, API_BASE, MODEL — Model settings
#   JUDGE_API_KEY, JUDGE_API_BASE, JUDGE_MODEL — Judge settings
#   EVAL_SET — Base eval_set.jsonl path
#
# Usage:
#   export API_KEY=sk-xxx API_BASE=https://... MODEL=MiniMax-M2.7
#   export JUDGE_API_KEY=sk-xxx JUDGE_API_BASE=https://... JUDGE_MODEL=gpt-5.3-codex
#   export EVAL_SET=/path/to/eval_set.jsonl
#   bash recipes/context_rot/distance/run_experiment.sh [sp|query|both] [filter_ids] [padding_levels]

set -e

EXPERIMENT=${1:-both}
FILTER_IDS=${2:-"case_0_P1,case_2_P1,case_8_P1"}
PADDING_LEVELS=${3:-"0,20,50,100,200"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_DIR="$SCRIPT_DIR/../eval"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="${REPO_ROOT}/outputs/context_rot/distance_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "========================================"
echo "Distance Sensitivity Experiment"
echo "========================================"
echo "  Experiment:  $EXPERIMENT"
echo "  Model:       ${MODEL:?Set MODEL}"
echo "  Filter:      $FILTER_IDS"
echo "  Padding:     $PADDING_LEVELS"
echo "  Output:      $RUN_DIR"
echo "========================================"

# Step 1: Generate eval sets
echo ""
echo "--- Step 1: Generating distance eval sets ---"
python3 "$SCRIPT_DIR/make_eval.py" \
    --experiment "$EXPERIMENT" \
    --input "${EVAL_SET:?Set EVAL_SET}" \
    --output-dir "$RUN_DIR" \
    --filter-id "$FILTER_IDS" \
    --padding-levels "$PADDING_LEVELS"

# Step 2: Determine max turns
MAX_TURNS="${MAX_TURNS:-10}"

JUDGE_EXTRA_HEADERS_ARG=""
if [ -n "${JUDGE_EXTRA_HEADERS:-}" ]; then
    JUDGE_EXTRA_HEADERS_ARG="--judge-extra-headers $JUDGE_EXTRA_HEADERS"
fi

run_eval() {
    local input_file=$1
    local output_file=$2
    local log_file=$3
    local label=$4

    if [ ! -f "$input_file" ]; then
        echo "  Skipping $label (no input file)"
        return
    fi

    echo ""
    echo "--- Running $label ---"
    python3 "$EVAL_DIR/run_eval.py" \
        --model "${MODEL}" \
        --judge-model "${JUDGE_MODEL:?Set JUDGE_MODEL}" \
        --judge-api-base "${JUDGE_API_BASE:?Set JUDGE_API_BASE}" \
        --judge-api-key "${JUDGE_API_KEY:?Set JUDGE_API_KEY}" \
        ${JUDGE_EXTRA_HEADERS_ARG} \
        --api-base "${API_BASE:?Set API_BASE}" \
        --api-key "${API_KEY:?Set API_KEY}" \
        --input "$input_file" \
        --output "$output_file" \
        --concurrency "${CONCURRENCY:-8}" \
        --max-turns "$MAX_TURNS" \
        --limit "${LIMIT:-200}" 2>&1 | tee "$log_file"
}

if [[ "$EXPERIMENT" == "sp" || "$EXPERIMENT" == "both" ]]; then
    run_eval \
        "$RUN_DIR/eval_set_sp_distance.jsonl" \
        "$RUN_DIR/results_sp_distance.jsonl" \
        "$RUN_DIR/eval_sp.log" \
        "SP Distance Experiment"
fi

if [[ "$EXPERIMENT" == "query" || "$EXPERIMENT" == "both" ]]; then
    run_eval \
        "$RUN_DIR/eval_set_query_distance.jsonl" \
        "$RUN_DIR/results_query_distance.jsonl" \
        "$RUN_DIR/eval_query.log" \
        "Query Distance Experiment"
fi

# Step 3: Analyze results
echo ""
echo "--- Generating distance analysis report ---"
python3 "$SCRIPT_DIR/analyze.py" "$RUN_DIR"

echo ""
echo "========================================"
echo "Experiment complete. Results in: $RUN_DIR"
echo "========================================"
