#!/usr/bin/env bash
set -euo pipefail

PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

ray start --head \
  --port="${PORT}" \
  --dashboard-port="${DASHBOARD_PORT}" \
  --num-cpus="${RAY_NUM_CPUS:-$(nproc)}"

echo "Ray head started — RAY_ADDRESS=127.0.0.1:${PORT}"
