#!/usr/bin/env bash
set -euo pipefail

# Required env vars:
#   PROFILING_INPUT — path to input JSONL
#   MODEL_PATH      — path to model
#
# Optional:
#   TOKENIZER_PATH  — defaults to MODEL_PATH
#   RAY_ADDRESS     — defaults to http://127.0.0.1:8265

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${PROFILING_WORKDIR:-${REPO_ROOT}/outputs/profiling_${TIMESTAMP}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${WORKDIR}"

PROFILING_INPUT="${PROFILING_INPUT:?Set PROFILING_INPUT}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"

RUNTIME_ENV=$(cat <<REOF
{"env_vars": {"PYTHONPATH": "${REPO_ROOT}"}}
REOF
)

ray job submit \
  --address "${RAY_ADDRESS}" \
  --runtime-env-json "${RUNTIME_ENV}" \
  -- python "${REPO_ROOT}/recipes/profiling/run.py" \
  --input     "${PROFILING_INPUT}" \
  --output-dir  "${WORKDIR}" \
  --stage       "all" \
  --model-path  "${MODEL_PATH}" \
  --tokenizer   "${TOKENIZER_PATH}" \
  --prompt-key  "prompt" \
  --label-key   "answer" \
  --num-examples 8 \
  --num-nodes   4 \
  --tp-size     1 \
  --dp-size     8 \
  --max-inflight 1024 \
  --temperature 1.0 \
  --top-p       0.95 \
  --max-new-tokens 30000 \
  --n-proc      32 \
  --acc-min     0.0 \
  --acc-max     0.7 \
  2>&1 | tee "${WORKDIR}/run.log"
