#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/mmmlu_online_${TIMESTAMP}"
MODEL_NAME="MiniMax-M2.5"
BASE_URL="https://api.minimaxi.com/v1"

mkdir -p "${WORKDIR}"

python "${REPO_ROOT}/recipes/eval/run.py" \
  --tasks "mmmlu@1" \
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
  --concurrency 4 \
  --n-proc 32 \
  --num-actors "${NUM_ACTORS:-1}" \
  --max-examples 100 \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/run.log"
