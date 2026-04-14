#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/context_rot/judge_${TIMESTAMP}"

mkdir -p "${WORKDIR}"

python "${REPO_ROOT}/recipes/context_rot/data_scan/scan_judge.py" \
  --input "${SCAN_INPUT:?Set SCAN_INPUT}" \
  --output "${WORKDIR}/judged.jsonl" \
  --judge-model "${JUDGE_MODEL:-gpt-5.4-thinking-xhigh}" \
  --judge-api-base "${JUDGE_API_BASE:?Set JUDGE_API_BASE}" \
  --judge-api-key "${JUDGE_API_KEY:?Set JUDGE_API_KEY}" \
  --extra-headers "${EXTRA_HEADERS:-X-Biz-Id: vela-admin}" \
  --concurrency "${CONCURRENCY:-16}" 2>&1 | tee "${WORKDIR}/judge.log"
