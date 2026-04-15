#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/offline_${TIMESTAMP}"

python "${REPO_ROOT}/recipes/eval/run.py" \
  --tasks "ifeval@1" \
  --task-dir "${REPO_ROOT}/outputs/nano_eval" \
  --output-dir "${WORKDIR}" \
  --stage all \
  --backend offline \
  --model-path "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-dapo-rl-20260401_072346/iter_0000127-hf" \
  --tp-size "${TP_SIZE:-1}" \
  --dp-size "${DP_SIZE:-8}" \
  --temperature 0.6 \
  --top-p 0.95 \
  --enable-thinking true \
  --max-tokens 81920 \
  --n-proc 32 \
  --num-actors "${NUM_ACTORS:-1}" \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/run.log"
