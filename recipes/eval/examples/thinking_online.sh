#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/online_${TIMESTAMP}"
mkdir -p "${WORKDIR}"

TASK_ARGS=(
  --stage all
  --task-dir "${REPO_ROOT}/outputs/nano_eval"
  --tasks "gpqa_diamond@4,math500@1,aime2025@8,ifeval@1"
  --output "${WORKDIR}/step01_prepared.jsonl"
  --inference-output "${WORKDIR}/step02_inference.jsonl"
  --score-output "${WORKDIR}/step03_score.jsonl"
  --final-eval-output "${WORKDIR}/step03_final_eval.jsonl"
)

ONLINE_ARGS=(
  --api-key "${API_KEY:?Set API_KEY}"
  --base-url "${BASE_URL:?Set BASE_URL}"
  --model "${MODEL_NAME:?Set MODEL_NAME}"
)

ROLLOUT_ARGS=(
  --backend online
  --temperature 1.0
  --top-p 0.95
  --enable-thinking true
  --max-tokens 131072
  --concurrency 1024
  --n-proc 32
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ONLINE_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${WORKDIR}/run.log"
