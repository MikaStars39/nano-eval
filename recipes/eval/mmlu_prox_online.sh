#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/mmlu_prox_online_${TIMESTAMP}"

python "${REPO_ROOT}/run.py" \
  --tasks "mmlu_prox@1" \
  --task-dir "${TASK_DIR:-/jfs-dialogue-mmos-rs04/users/qingyu/data/hf/nano-eval}" \
  --output-dir "${WORKDIR}" \
  --stage all \
  --backend online \
  --api-key "${API_KEY:?Set API_KEY}" \
  --base-url "${BASE_URL:?Set BASE_URL}" \
  --model "${MODEL_NAME:?Set MODEL_NAME}" \
  --temperature 0.6 \
  --top-p 0.95 \
  --enable-thinking true \
  --max-tokens 32768 \
  --concurrency 1024 \
  --n-proc 32 \
  --num-actors "${NUM_ACTORS:-1}" \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/run.log"
