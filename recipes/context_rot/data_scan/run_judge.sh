#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/context_rot/judge_${TIMESTAMP}"

mkdir -p "${WORKDIR}"

python "${REPO_ROOT}/recipes/context_rot/data_scan/scan_judge.py" \
  --input "/jfs-dialogue-mmos-rs04/users/qingyu/data/context_rot/scan_results/sampled_500_raw_context_rot_data.jsonl" \
  --output "${WORKDIR}/judged.jsonl" \
  --judge-model "gpt-5.4-thinking-xhigh" \
  --judge-api-base "https://talkie-ali-virginia-prod-internal.xaminim.com/llm/oai" \
  --judge-api-key "sk-esReHZqyjoLlZwRDdloc6muhI3zoDLqRzwYoNcvT9zsBnFMI" \
  --api-type "responses" \
  --extra-headers "X-Biz-Id: vela-admin" \
  --concurrency "${CONCURRENCY:-16}" \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/judge.log"
