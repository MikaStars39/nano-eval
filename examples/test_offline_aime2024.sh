#!/usr/bin/env bash
set -euo pipefail

# Real offline evaluation for AIME2024 with a local model.
REPO_ROOT=/mnt/llm-train/users/explore-train/qingyu/NanoEval
TASK_DIR="${REPO_ROOT}/outputs/nano_eval"
TASK_NAME="aime2024"
PASS_K=8
SYSTEM_PROMPT="You are a careful math solver. Show reasoning clearly and end with the final answer in \\boxed{}."

LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${TASK_DIR}/${TASK_NAME}.jsonl" ]]; then
  echo "Task file not found: ${TASK_DIR}/${TASK_NAME}.jsonl"
  exit 1
fi

TASK_ARGS=(
  --stage all \
  --task-dir "${TASK_DIR}" \
  --tasks "${TASK_NAME}" \
  --pass-k "${PASS_K}" \
  --system-prompt "${SYSTEM_PROMPT}"
  --n-proc 8
)

ROLLOUT_ARGS=(
  --backend offline \
  --model-path /mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-R1-Distill-Qwen-1.5B \
  --work-dir /mnt/llm-train/users/explore-train/qingyu/NanoEval/outputs/test \
  --temperature 1 \
  --max-tokens 32768
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
