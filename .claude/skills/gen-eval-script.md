---
name: gen-eval-script
description: Generate a NanoEval evaluation bash script (offline or online backend)
user_invocable: true
---

> IMPORTANT
> Do not directly write into the scripts folder a new script!
> scripts is designed for .sh that are undergone ci test and are designed for examples!

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
5. **TP size / DP size**: Tensor-parallel and data-parallel sizes (defaults: tp=8, dp=1).
6. **Num actors**: Number of parallel inference actors (default: 1).

**If online:**
4. **Base URL**: API endpoint URL.
5. **API key**: API key (can leave as placeholder `YOUR_API_KEY`).
6. **Model name**: Model identifier for the API (e.g. `gpt-oss-120b`).
7. **Num actors**: Number of online inference actors (default: 1).

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

Generate the script at the path the user specifies (or default to `recipes/eval_<run_name>.sh`).

### Offline template

```bash
#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NLTK_DATA="${NLTK_DATA:-/mnt/llm-train/users/explore-train/qingyu/.cache}"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKDIR="${REPO_ROOT}/outputs/<RUN_NAME>_${TIMESTAMP}"
mkdir -p "${WORKDIR}"

python "${REPO_ROOT}/recipes/eval/run.py" \
  --output-dir "${WORKDIR}" \
  --task-dir "${REPO_ROOT}/outputs/nano_eval" \
  --tasks "<TASKS>" \
  --stage all \
  --backend offline \
  --model-path <MODEL_PATH> \
  --tp-size <TP_SIZE> \
  --dp-size <DP_SIZE> \
  --num-actors <NUM_ACTORS> \
  --temperature <TEMPERATURE> \
  --top-p <TOP_P> \
  --enable-thinking <ENABLE_THINKING> \
  --max-tokens <MAX_TOKENS> \
  --n-proc <N_PROC> \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/run.log"
```

### Online template

```bash
#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export NLTK_DATA="${NLTK_DATA:-/mnt/llm-train/users/explore-train/qingyu/.cache}"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKDIR="${REPO_ROOT}/outputs/<RUN_NAME>_${TIMESTAMP}"
mkdir -p "${WORKDIR}"

python "${REPO_ROOT}/recipes/eval/run.py" \
  --output-dir "${WORKDIR}" \
  --task-dir "${REPO_ROOT}/outputs/nano_eval" \
  --tasks "<TASKS>" \
  --stage all \
  --backend online \
  --api-key "<API_KEY>" \
  --base-url "<BASE_URL>" \
  --model "<MODEL_NAME>" \
  --num-actors <NUM_ACTORS> \
  --temperature <TEMPERATURE> \
  --top-p <TOP_P> \
  --enable-thinking <ENABLE_THINKING> \
  --max-tokens <MAX_TOKENS> \
  --concurrency <CONCURRENCY> \
  --n-proc <N_PROC> \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/run.log"
```

## Rules

- Replace all `<PLACEHOLDER>` values with user-provided or default values.
- If the user specifies `--reasoning-effort`, add it to the command.
- Include commented-out lines for optional parameters the user didn't set (top-k, min-p, presence-penalty, repetition-penalty, agent-loop, max-turns) so the user can easily enable them later.
- Make the script executable after writing it (`chmod +x`).
- Always use `REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"` so the script works from any directory.
- After generating, show the user the full script path and remind them to sync it to the GPU server for execution.
