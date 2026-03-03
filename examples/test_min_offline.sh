#!/usr/bin/env bash
set -euo pipefail

# Minimal offline evaluation script.
REPO_ROOT=/mnt/llm-train/users/explore-train/qingyu/NanoEval
WORKDIR=${WORKDIR:-${REPO_ROOT}/outputs/test_min_offline}
LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

PREPARED_INPUT="${WORKDIR}/step01_prepared.jsonl"
INFERENCE_OUTPUT="${WORKDIR}/step02_inference.jsonl"
SCORE_OUTPUT="${WORKDIR}/step03_score.jsonl"
FINAL_EVAL_OUTPUT="${WORKDIR}/step03_final_eval.jsonl"

TASK_ARGS=(
  --stage all
  --task-dir ${REPO_ROOT}/outputs/nano_eval
  --tasks "aime2024@4"
  --output ${PREPARED_INPUT}
  --inference-output ${INFERENCE_OUTPUT}
  --score-output ${SCORE_OUTPUT}
  --final-eval-output ${FINAL_EVAL_OUTPUT}
)

MODEL_ARGS=(
  --model-path /mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-R1-Distill-Qwen-1.5B
)

ROLLOUT_ARGS=(
  --backend offline
  --temperature 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --presence-penalty 1.5
  --repetition-penalty 1.0
  --max-tokens 32768
  --concurrency 32
  --n-proc 8
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
