#!/usr/bin/env bash
set -euo pipefail

# Minimal online evaluation script.
REPO_ROOT=/mnt/llm-train/users/explore-train/qingyu/NanoEval
WORKDIR=/mnt/llm-train/users/explore-train/qingyu/NanoEval/outputs/online_qwen35_30b_thinking
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

ONLINE_ARGS=(
  # Replace these three values with your real endpoint config.
  --api-key "YOUR_API_KEY"
  --base-url "http://6.30.4.20:30339/v1"
  --model "qwen35-35b-a3b"
)

ROLLOUT_ARGS=(
  --backend online
  --temperature 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --presence-penalty 1.5
  --repetition-penalty 1.0
  --max-tokens 32768
  --concurrency 1024
  --n-proc 32
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ONLINE_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
