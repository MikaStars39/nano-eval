#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/online_${TIMESTAMP}"

python "${REPO_ROOT}/recipes/eval/run.py" \
  --tasks "gpqa_diamond@4,math500@1,aime2025@8,ifeval@1" \
  --task-dir "${REPO_ROOT}/outputs/nano_eval" \
  --output-dir "${WORKDIR}" \
  --stage all \
  --backend online \
  --api-key "${API_KEY:?Set API_KEY}" \
  --base-url "${BASE_URL:?Set BASE_URL}" \
  --model "${MODEL_NAME:?Set MODEL_NAME}" \
  --temperature 1.0 \
  --top-p 0.95 \
  --enable-thinking true \
  --max-tokens 131072 \
  --concurrency 1024 \
  --n-proc 32 \
  --num-actors "${NUM_ACTORS:-1}" \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/run.log"
