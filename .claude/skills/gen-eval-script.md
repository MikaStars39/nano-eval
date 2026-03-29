---
name: gen-eval-script
description: Generate a NanoEval evaluation bash script (offline or online backend)
user_invocable: true
---

# Generate NanoEval Evaluation Script

You are generating a bash evaluation script for the NanoEval framework. Ask the user for the required parameters, then produce a ready-to-run shell script.

## Information to Gather

Ask the user the following questions using `AskUserQuestion`. Group related questions together (max 4 per call) to minimize round-trips.

### Round 1 — Backend & Model

1. **Backend type**: `offline` or `online`?
2. **Run name**: A short label for this run (used in the output directory name, e.g. `qwen_4b`, `gptoss_low`).
3. **Tasks**: Which benchmarks and pass@k? Example: `"aime2025@8,math500@1,gpqa_diamond@4"`. Available tasks: `aime2024`, `aime2025`, `amc2023`, `math500`, `minerva`, `hmmt2025`, `gpqa_diamond`, `mmlu`, `mmlu_pro`, `ceval`, `ifeval`, `ifbench`.

### Round 2 — Backend-specific config

**If offline:**
4. **Model path**: Local path to the model weights.
5. **TP size / DP size**: Tensor-parallel and data-parallel sizes (defaults: tp=1, dp=8).

**If online:**
4. **Base URL**: API endpoint URL.
5. **API key**: API key (can leave as placeholder `YOUR_API_KEY`).
6. **Model name**: Model identifier for the API (e.g. `gpt-oss-120b`).

### Round 3 — Sampling & generation parameters

7. **Temperature** (default: 0.6 for offline, 1.0 for online)
8. **Top-p** (default: 0.95)
9. **Max tokens** (default: 81920 for offline, 131072 for online)
10. **Enable thinking** (default: true)
11. **Concurrency** (default: 1024)
12. **n-proc** for scoring (default: 32)
13. **Reasoning effort** (optional, only for online: `low`/`medium`/`high`)

For any parameter the user doesn't specify, use the defaults above.

## Script Template

Generate the script at the path the user specifies (or default to `scripts/eval_<run_name>.sh`).

### Offline template

```bash
#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NLTK_DATA="${NLTK_DATA:-/mnt/llm-train/users/explore-train/qingyu/.cache}"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR="${REPO_ROOT}/outputs/<RUN_NAME>_${TIMESTAMP}"
LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

PREPARED_INPUT="${WORKDIR}/step01_prepared.jsonl"
INFERENCE_OUTPUT="${WORKDIR}/step02_inference.jsonl"
SCORE_OUTPUT="${WORKDIR}/step03_score.jsonl"
FINAL_EVAL_OUTPUT="${WORKDIR}/step03_final_eval.jsonl"

TASK_ARGS=(
  --stage all
  --task-dir "${REPO_ROOT}/outputs/nano_eval"
  --tasks "<TASKS>"
  --output "${PREPARED_INPUT}"
  --inference-output "${INFERENCE_OUTPUT}"
  --score-output "${SCORE_OUTPUT}"
  --final-eval-output "${FINAL_EVAL_OUTPUT}"
)

ROLLOUT_ARGS=(
  --model-path <MODEL_PATH>
  --backend offline
  --tp-size <TP_SIZE>
  --dp-size <DP_SIZE>
  --temperature <TEMPERATURE>
  --top-p <TOP_P>
  --enable-thinking <ENABLE_THINKING>
  --max-tokens <MAX_TOKENS>
  --concurrency <CONCURRENCY>
  --n-proc <N_PROC>
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
```

### Online template

```bash
#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export NLTK_DATA="${NLTK_DATA:-/mnt/llm-train/users/explore-train/qingyu/.cache}"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR="${REPO_ROOT}/outputs/<RUN_NAME>_${TIMESTAMP}"
LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

PREPARED_INPUT="${WORKDIR}/step01_prepared.jsonl"
INFERENCE_OUTPUT="${WORKDIR}/step02_inference.jsonl"
SCORE_OUTPUT="${WORKDIR}/step03_score.jsonl"
FINAL_EVAL_OUTPUT="${WORKDIR}/step03_final_eval.jsonl"

TASK_ARGS=(
  --stage all
  --task-dir "${REPO_ROOT}/outputs/nano_eval"
  --tasks "<TASKS>"
  --output "${PREPARED_INPUT}"
  --inference-output "${INFERENCE_OUTPUT}"
  --score-output "${SCORE_OUTPUT}"
  --final-eval-output "${FINAL_EVAL_OUTPUT}"
)

ONLINE_ARGS=(
  --api-key "<API_KEY>"
  --base-url "<BASE_URL>"
  --model "<MODEL_NAME>"
)

ROLLOUT_ARGS=(
  --backend online
  --temperature <TEMPERATURE>
  --top-p <TOP_P>
  --enable-thinking <ENABLE_THINKING>
  --max-tokens <MAX_TOKENS>
  --concurrency <CONCURRENCY>
  --n-proc <N_PROC>
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ONLINE_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
```

## Rules

- Replace all `<PLACEHOLDER>` values with user-provided or default values.
- If the user specifies `--reasoning-effort`, add it to `ROLLOUT_ARGS`.
- Include commented-out lines for optional parameters the user didn't set (top-k, min-p, presence-penalty, repetition-penalty, ray-num-actors, ray-worker-concurrency, online-request-timeout-s, online-stall-log-interval-s) so the user can easily enable them later.
- Make the script executable after writing it (`chmod +x`).
- Always use `REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"` so the script works from any directory.
- After generating, show the user the full script path and remind them to sync it to the GPU server for execution.
