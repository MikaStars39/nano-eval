#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NLTK_DATA="/mnt/llm-train/users/explore-train/qingyu/.cache"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
REPO_ROOT=/mnt/llm-train/users/explore-train/qingyu/NanoEval
WORKDIR=/mnt/llm-train/users/explore-train/qingyu/NanoEval/outputs/qwen_4b_${TIMESTAMP}
LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

PREPARED_INPUT="${WORKDIR}/step01_prepared.jsonl"
INFERENCE_OUTPUT="${WORKDIR}/step02_inference.jsonl"
SCORE_OUTPUT="${WORKDIR}/step03_score.jsonl"
FINAL_EVAL_OUTPUT="${WORKDIR}/step03_final_eval.jsonl"

TASK_ARGS=(
  --stage all
  --task-dir ${REPO_ROOT}/outputs/nano_eval
  --tasks "ifeval@1"
  --output ${PREPARED_INPUT}
  --inference-output ${INFERENCE_OUTPUT}
  --score-output ${SCORE_OUTPUT}
  --final-eval-output ${FINAL_EVAL_OUTPUT}
)

ONLINE_ARGS=(
  # Replace these three values with your real endpoint config.
  # --api-key "YOUR_API_KEY"
  # --base-url "http://6.30.4.20:31859/v1"
)

ROLLOUT_ARGS=(
  --model-path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507
  --backend offline
  --tp-size 1
  --dp-size 8
  --temperature 0.6
  --top-p 0.95
  # --top-k 20
  # --min-p 0.0
  # --presence-penalty 0.2
  # --repetition-penalty 1.0
  --enable-thinking true
  --max-tokens 81920
  --concurrency 1024
  # Optional explicit split for online_ray:
  # --ray-num-actors 8
  # --ray-worker-concurrency 128
  # --online-request-timeout-s 3600
  # --online-stall-log-interval-s 60
  --n-proc 32
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ONLINE_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
