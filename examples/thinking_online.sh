#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export NLTK_DATA="/mnt/llm-train/users/explore-train/qingyu/.cache"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
# Minimal online evaluation script.
REPO_ROOT=/mnt/llm-train/users/explore-train/qingyu/NanoEval
WORKDIR=/mnt/llm-train/users/explore-train/qingyu/NanoEval/outputs/gptoss_low_${TIMESTAMP}
LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

PREPARED_INPUT="${WORKDIR}/step01_prepared.jsonl"
INFERENCE_OUTPUT="${WORKDIR}/step02_inference.jsonl"
SCORE_OUTPUT="${WORKDIR}/step03_score.jsonl"
FINAL_EVAL_OUTPUT="${WORKDIR}/step03_final_eval.jsonl"

TASK_ARGS=(
  --stage all
  --task-dir ${REPO_ROOT}/outputs/nano_eval
  --tasks "gpqa_diamond@4,math500@1,aime2025@8,ifeval@1"
  --output ${PREPARED_INPUT}
  --inference-output ${INFERENCE_OUTPUT}
  --score-output ${SCORE_OUTPUT}
  --final-eval-output ${FINAL_EVAL_OUTPUT}
)

ONLINE_ARGS=(
  # Replace these three values with your real endpoint config.
  --api-key "YOUR_API_KEY"
  --base-url "http://6.30.4.20:30865/v1"
  --model "gpt-oss-120b"
)

ROLLOUT_ARGS=(
  --backend online
  --temperature 1.0
  --enable-thinking True
  --top-p 0.95
  # --top-k 20
  # --min-p 0.0
  # --presence-penalty 0.3
  # --repetition-penalty 1.2
  --max-tokens 131072
  --concurrency 1024
  # Optional explicit split for online_ray:
  # --ray-num-actors 8 
  # --ray-worker-concurrency 128
  # --online-request-timeout-s 3600
  # --online-stall-log-interval-s 60
  --n-proc 32
  --reasoning-effort low
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ONLINE_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
