#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/offline_${TIMESTAMP}"
mkdir -p "${WORKDIR}"

TASK_ARGS=(
  --stage all
  --task-dir "${REPO_ROOT}/outputs/nano_eval"
  --tasks "ifeval@1"
  --output "${WORKDIR}/step01_prepared.jsonl"
  --inference-output "${WORKDIR}/step02_inference.jsonl"
  --score-output "${WORKDIR}/step03_score.jsonl"
  --final-eval-output "${WORKDIR}/step03_final_eval.jsonl"
)

ROLLOUT_ARGS=(
  --model-path "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-dapo-rl-20260401_072346/iter_0000127-hf"
  --backend offline
  --tp-size "${TP_SIZE:-1}"
  --dp-size "${DP_SIZE:-8}"
  --temperature 0.6
  --top-p 0.95
  --enable-thinking true
  --max-tokens 81920
  --n-proc 32
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${WORKDIR}/run.log"
