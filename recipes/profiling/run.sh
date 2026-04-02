#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${WORKDIR:-${REPO_ROOT}/outputs/profiling_${TIMESTAMP}}"
mkdir -p "${WORKDIR}"

python "${REPO_ROOT}/recipes/profiling/run.py" \
  --input       "${INPUT_FILE:?Set INPUT_FILE}" \
  --output-dir  "${WORKDIR}" \
  --stage       "${STAGE:-all}" \
  --model-path  "${MODEL_PATH:?Set MODEL_PATH}" \
  --tokenizer   "${TOKENIZER:-${MODEL_PATH}}" \
  --prompt-key  "${PROMPT_KEY:-messages}" \
  --label-key   "${LABEL_KEY:-label}" \
  --num-examples "${NUM_EXAMPLES:-1}" \
  --num-nodes   "${NUM_NODES:-1}" \
  --tp-size     "${TP_SIZE:-8}" \
  --dp-size     "${DP_SIZE:-1}" \
  --max-inflight "${MAX_INFLIGHT:-512}" \
  --temperature "${TEMPERATURE:-0.6}" \
  --top-p       "${TOP_P:-1.0}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
  --n-proc      "${N_PROC:-32}" \
  --enable-dp-attention \
 #  --resume \
  2>&1 | tee "${WORKDIR}/run.log"
