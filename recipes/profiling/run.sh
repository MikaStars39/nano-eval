#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/jpfs-5p/qingyu/nano-eval/
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="/jpfs-5p/qingyu/data/profiling_${TIMESTAMP}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${WORKDIR}"

RUNTIME_ENV=$(cat <<'REOF'
{"env_vars": {"PYTHONPATH": "/jpfs-5p/qingyu/nano-eval/"}}
REOF
)

ray job submit \
  --address "http://127.0.0.1:8265" \
  --runtime-env-json "${RUNTIME_ENV}" \
  -- python "${REPO_ROOT}/recipes/profiling/run.py" \
  --input     "/jpfs/chenyanxu.9/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.jsonl" \
  --output-dir  "${WORKDIR}" \
  --stage       "all" \
  --model-path  "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-Base-sft-dolci-think/iter_0005375-hf" \
  --tokenizer   "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-Base-sft-dolci-think/iter_0005375-hf" \
  --prompt-key  "prompt" \
  --label-key   "answer" \
  --num-examples 8 \
  --num-nodes   4 \
  --tp-size     1 \
  --dp-size     8 \
  --max-inflight 1024 \
  --temperature 1.0 \
  --top-p       0.95 \
  --max-new-tokens 32768 \
  --n-proc      32 \
  --acc-min     0.1 \
  --acc-max     0.9 \
  2>&1 | tee "${WORKDIR}/run.log"
