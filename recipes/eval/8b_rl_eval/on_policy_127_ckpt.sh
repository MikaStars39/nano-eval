#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

REPO_ROOT=/jpfs-5p/qingyu/nano-eval
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/iter_0000127_qwen3_8b_rl_onpolicy_offline_${TIMESTAMP}"
mkdir -p "${WORKDIR}"

python "${REPO_ROOT}/recipes/eval/run.py" \
  --output-dir "${WORKDIR}" \
  --task-dir "/jpfs/chenyanxu.9/data/nano-eval" \
  --tasks "aime2024@32,aime2025@32,math500@4,gpqa_diamond@4" \
  --stage all \
  --backend offline \
  --model-path "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-dapo-rl-20260401_072346/iter_0000127-hf" \
  --tp-size 1 \
  --dp-size 8 \
  --temperature 1.0 \
  --top-p 0.95 \
  --enable-thinking true \
  --max-tokens 30000 \
  --n-proc 32 \
  --num-actors "${NUM_ACTORS:-1}" \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/run.log"
