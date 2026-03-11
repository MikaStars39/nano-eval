# NanoEval Evaluation Guide

A comprehensive guide for using NanoEval, a fast and lightweight evaluation toolkit for Large Language Models.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Design Principles](#design-principles)
4. [Quick Start](#quick-start)
5. [Evaluation Pipeline](#evaluation-pipeline)
6. [Backend Configuration](#backend-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Supported Tasks](#supported-tasks)
9. [Output Format](#output-format)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

NanoEval is a high-performance, lightweight evaluation framework designed for benchmarking Large Language Models across various reasoning and knowledge tasks. It features:

- **Three-stage pipeline**: Input preparation → Inference → Scoring
- **Multiple backends**: Local (SGLang), Online API, and Distributed (Ray)
- **High throughput**: Async I/O with producer-consumer architecture
- **Flexible task support**: Math, coding, instruction following, and multiple-choice tasks
- **Pass@k evaluation**: Built-in support for multiple sampling attempts per question

---

## Project Structure

```
NanoEval/
├── nanoeval/                   # Core evaluation library
│   ├── backend/               # Inference backends
│   │   ├── base.py             # Base SGLang engine with lifecycle management
│   │   ├── offline.py          # Local batch inference (SGLang)
│   │   ├── online.py         # API-based inference (OpenAI-compatible)
│   │   ├── online_ray.py     # Distributed inference with Ray
│   │   └── runner.py         # Backend router
│   ├── reward/                # Scoring and verification
│   │   ├── score.py          # Main scoring orchestrator
│   │   ├── reward.py         # Judge router for task-specific scoring
│   │   ├── math/             # Math verification (GSM8K, MATH, AIME, etc.)
│   │   ├── if_eval/          # Instruction following evaluation
│   │   └── gpqa/             # Multiple-choice verification
│   └── utils/                 # Utilities
│       ├── args.py           # CLI argument parsing
│       ├── task.py           # Task loading and preparation
│       └── logging_utils.py  # Logging configuration
├── examples/                   # Example evaluation scripts
│   ├── test_offline_aime2024.sh
│   ├── test_online.sh
│   └── test_min_offline.sh
├── run.py                     # Main entry point
└── docs/                      # Documentation
```

---

## Design Principles

### 1. Three-Stage Pipeline

NanoEval follows a clean separation of concerns with three distinct stages:

| Stage | Purpose | Key Operations |
|-------|---------|----------------|
| **Step01** | Input Preparation | Load tasks, apply chat templates, expand for pass@k |
| **Step02** | Inference | Generate responses using specified backend |
| **Step03** | Scoring | Judge responses, compute metrics (avg_k, pass@k) |

Each stage produces intermediate JSONL files, enabling:
- **Resumability**: Restart from any stage
- **Debugging**: Inspect intermediate outputs
- **Flexibility**: Use different backends for same input

### 2. Async Producer-Consumer Architecture

All backends use async I/O with three concurrent components:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Producer │────▶│  Queue   │────▶│  Worker  │
│ (Read)   │     │ (Buffer) │     │(Generate)│
└──────────┘     └──────────┘     └────┬─────┘
                                        │
                                        ▼
                                 ┌──────────┐
                                 │  Writer  │
                                 │ (Save)   │
                                 └──────────┘
```

Benefits:
- **Maximize GPU utilization**: Workers never wait for I/O
- **Memory efficiency**: Streaming processing of large datasets
- **Progress tracking**: Real-time throughput metrics

### 3. Unified Sampling Parameters

All backends share consistent sampling parameter interfaces:

```python
sampling_params = {
    "temperature": 0.7,        # Core parameter (always used)
    "max_tokens": 1024,        # Core parameter (always used)
    "top_p": 0.95,            # Optional nucleus sampling
    "top_k": 20,              # Optional top-k sampling
    "min_p": 0.0,             # Optional minimum probability
    "presence_penalty": 0.0,  # Optional presence penalty
    "repetition_penalty": 1.0, # Optional repetition penalty
}
```

### 4. Modular Backend System

| Backend | Use Case | Concurrency Model |
|---------|----------|-------------------|
| `offline` | Local GPU inference | `max_inflight` async workers |
| `online` | Remote API calls | `concurrency` semaphore-limited tasks |
| `online_ray` | Distributed API calls | Ray actors × worker concurrency |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd NanoEval

# Install dependencies
pip install -r requirements.txt

# For offline backend (SGLang)
pip install sglang flashinfer

# For online_ray backend
pip install ray
```

### Minimal Offline Evaluation

```bash
python run.py \
  --stage all \
  --task-dir ./outputs/nano_eval \
  --tasks "aime2024@4" \
  --output ./work/step01.jsonl \
  --inference-output ./work/step02.jsonl \
  --score-output ./work/step03_score.jsonl \
  --final-eval-output ./work/step03_metrics.jsonl \
  --backend offline \
  --model-path /path/to/your/model \
  --temperature 1.0 \
  --max-tokens 32768 \
  --concurrency 32
```

### Minimal Online Evaluation

```bash
python run.py \
  --stage all \
  --task-dir ./outputs/nano_eval \
  --tasks "ifeval@1" \
  --output ./work/step01.jsonl \
  --inference-output ./work/step02.jsonl \
  --score-output ./work/step03_score.jsonl \
  --final-eval-output ./work/step03_metrics.jsonl \
  --backend online \
  --api-key "your-api-key" \
  --base-url "https://api.example.com/v1" \
  --model "gpt-4o-mini" \
  --temperature 0.7 \
  --max-tokens 4096 \
  --concurrency 100
```

---

## Evaluation Pipeline

### Stage 1: Input Preparation

This stage:
1. Discovers task files from `--task-dir`
2. Applies chat templates (if `--chat-template-model-path` provided)
3. Expands each question for pass@k evaluation
4. Writes prepared prompts to `--output`

```bash
python run.py \
  --stage step01 \
  --tasks "aime2024@4,aime2025@8" \
  --pass-k 1 \
  --task-dir ./outputs/nano_eval \
  --output ./work/step01.jsonl \
  --chat-template-model-path /path/to/model \
  --system-prompt "You are a helpful assistant."
```

**Task specification syntax:**
- `taskname` — use default pass-k
- `taskname@k` — use k attempts per question
- `all` — auto-discover all tasks in directory

### Stage 2: Inference

Generates responses using the specified backend.

```bash
python run.py \
  --stage step02 \
  --input ./work/step01.jsonl \
  --inference-output ./work/step02.jsonl \
  --backend offline \
  --model-path /path/to/model \
  --tp-size 1 \
  --dp-size 8 \
  --temperature 0.6 \
  --max-tokens 81920 \
  --concurrency 1024
```

**Resume capability:** If `--inference-output` already exists, only processes missing IDs.

### Stage 3: Scoring

Evaluates responses and computes metrics.

```bash
python run.py \
  --stage step03 \
  --eval-input ./work/step02.jsonl \
  --score-output ./work/step03_score.jsonl \
  --final-eval-output ./work/step03_metrics.jsonl \
  --n-proc 32
```

---

## Backend Configuration

### Offline Backend (SGLang)

For local model inference with SGLang engine.

**Key Arguments:**
```bash
--backend offline
--model-path /path/to/model          # Local model directory
--tp-size 1                          # Tensor parallelism size
--dp-size 8                          # Data parallelism size
--concurrency 512                    # Max concurrent requests
```

**Example:**
```bash
ROLLOUT_ARGS=(
  --backend offline
  --model-path /mnt/cache/Qwen3-4B-Instruct
  --tp-size 1
  --dp-size 8
  --temperature 0.6
  --top-p 0.95
  --enable-thinking true             # For thinking models (Qwen3, etc.)
  --max-tokens 81920
  --concurrency 1024
)
```

**Performance Tips:**
- Set `--dp-size` to number of available GPUs for data parallelism
- Set `--concurrency` to 64-128× number of GPUs
- Use `--tp-size > 1` only for models that don't fit on single GPU

### Online Backend (API)

For remote API endpoints (OpenAI-compatible).

**Key Arguments:**
```bash
--backend online
--api-key "YOUR_API_KEY"
--base-url "http://host:port/v1"
--model "model-name"
--concurrency 100                    # Max parallel API calls
--online-request-timeout-s 3600      # Per-request timeout
```

**Example:**
```bash
ONLINE_ARGS=(
  --api-key "sk-..."
  --base-url "http://6.30.4.20:30339/v1"
  --model "qwen35-35b-a3b"
)

ROLLOUT_ARGS=(
  --backend online
  --temperature 1.0
  --top-p 0.95
  --top-k 20
  --presence-penalty 1.5
  --max-tokens 32768
  --concurrency 1024
)
```

### Online Ray Backend (Distributed)

For high-throughput distributed API inference using Ray.

**Key Arguments:**
```bash
--backend online_ray
--api-key "YOUR_API_KEY"
--base-url "http://host:port/v1"
--model "model-name"
--concurrency 256                    # Total concurrency
--ray-num-actors 8                 # Number of Ray actors
--ray-worker-concurrency 32          # Concurrency per actor
--online-stall-log-interval-s 60   # Progress heartbeat
```

**Architecture:**
```
Total Concurrency = ray-num-actors × ray-worker-concurrency
```

**Example:**
```bash
ROLLOUT_ARGS=(
  --backend online_ray
  --concurrency 256
  --ray-num-actors 8
  --ray-worker-concurrency 32
  --online-request-timeout-s 3600
  --online-stall-log-interval-s 60
)
```

**When to use:**
- High-latency APIs where single-process async is insufficient
- Handling thousands of concurrent requests
- Better fault tolerance (per-actor isolation)

---

## Performance Tuning

### General Guidelines

| Resource | Recommendation |
|----------|---------------|
| **Concurrency** | Start with 64×GPU count, increase until GPU saturation |
| **Batch Size** | Larger is better for throughput (limited by GPU memory) |
| **max_tokens** | Set based on task requirements; higher = slower |
| **n_proc (scoring)** | Match CPU core count (typically 16-32) |

### Offline Backend Tuning

```bash
# For 8×A100 GPUs with 4B model
--tp-size 1          # No tensor parallelism needed
--dp-size 8          # Data parallel across 8 GPUs
--concurrency 1024   # 128 per GPU

# For 70B model on 8×A100
--tp-size 8          # Tensor parallel (model shard)
--dp-size 1          # No data parallelism
--concurrency 128    # Adjust based on memory
```

### Online Backend Tuning

```bash
# Low-latency API (< 1s per request)
--concurrency 200

# High-latency API (10-30s per request)
--concurrency 2000   # Keep many in-flight requests

# With rate limits (e.g., 100 req/s)
--concurrency 100    # Match rate limit
```

### Online Ray Tuning

```bash
# For 1000 req/s target with 100ms per request
--ray-num-actors 20
--ray-worker-concurrency 50
# Total = 1000 concurrent requests

# Tune based on API capacity and latency
```

### Scoring Tuning

```bash
# CPU-bound operation
--n-proc 32          # Match CPU cores

# For small datasets
--n-proc 1           # Avoid multiprocessing overhead
```

### Memory Optimization

For large models or long contexts:

```bash
# Reduce memory fraction if encountering OOM
# (Modify in backend/base.py if needed)
mem_fraction_static=0.85

# Enable radix cache for repeated prefixes
enable_radix_cache=true
```

---

## Supported Tasks

| Task | Type | Metric | Description |
|------|------|--------|-------------|
| **aime2024/2025** | Math | pass@k | AIME competition problems |
| **amc2023** | Math | pass@k | AMC competition problems |
| **math500** | Math | pass@k | MATH dataset (500 problems) |
| **minerva** | Math | pass@k | Minerva math problems |
| **hmmt2025** | Math | pass@k | HMMT competition problems |
| **gpqa_diamond** | Science MC | pass@k | Graduate-level science QA |
| **mmlu/mmlu_pro** | MC | pass@k | Massive multitask language understanding |
| **ceval** | MC | pass@k | Chinese evaluation suite |
| **ifeval** | Instruction | prompt-level pass | Instruction following evaluation |
| **ifbench** | Instruction | prompt-level pass | Extended IF evaluation |

**Adding custom tasks:**

1. Create a JSONL file in your task directory:
```jsonl
{"question_id": "q1", "prompt": "What is 2+2?", "label": "4"}
{"question_id": "q2", "prompt": "Solve: x^2 = 4", "label": "2, -2"}
```

2. Register in `nanoeval/utils/task.py`:
```python
TASK_TO_JSONL = {
    "your_task": "your_task.jsonl",
    # ... existing tasks
}
```

---

## Output Format

### Step01 Output (Prepared Input)

```jsonl
{
  "question_id": "aime2024_1",
  "prompt": "<formatted prompt with chat template>",
  "label": "42",
  "id": "aime2024_1_0",
  "source": "aime2024",
  "sample_index": 0
}
```

### Step02 Output (Inference Results)

```jsonl
{
  "question_id": "aime2024_1",
  "prompt": "...",
  "label": "42",
  "id": "aime2024_1_0",
  "source": "aime2024",
  "sample_index": 0,
  "response": "The answer is 42.",
  "thinking": "<reasoning trace>",  // If model supports thinking
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  },
  "_latency": 2.345,
  "_status": "success"
}
```

### Step03 Score Output (Per-Instance)

```jsonl
{
  "question_id": "aime2024_1",
  "prompt": "...",
  "label": "42",
  "id": "aime2024_1_0",
  "source": "aime2024",
  "response": "The answer is 42.",
  "pred": "42",
  "pass": true,
  "pass_at_k": true  // True if any sample for this question passed
}
```

### Step03 Metrics Output (Aggregated)

```jsonl
["aime2024", {
  "avg_k": 0.25,           // Average accuracy across all attempts
  "pass_k": 0.5,           // Pass@k: proportion with at least 1 correct
  "avg_total_tokens": 150.5,
  "avg_thinking_tokens": 45.2,
  "max_thinking_tokens": 120,
  "min_thinking_tokens": 10
}]
["overall", { ... }]
```

### CSV Output

| task | avg_k | pass_k | avg_total_tokens | avg_thinking_tokens | max_thinking_tokens | min_thinking_tokens |
|------|-------|--------|------------------|---------------------|---------------------|---------------------|
| aime2024 | 0.25 | 0.5 | 150.5 | 45.2 | 120 | 10 |
| overall | 0.30 | 0.6 | 145.0 | 42.0 | 120 | 10 |

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions:**
1. Reduce `--concurrency`
2. Reduce `--max-tokens`
3. Increase `--tp-size` for tensor parallelism
4. Reduce `mem_fraction_static` in `base.py`

### Issue: Slow Inference

**Checklist:**
- [ ] Is GPU utilization at 100%? (`nvidia-smi`)
- [ ] Is `--concurrency` high enough?
- [ ] For online: Is API latency the bottleneck?
- [ ] Try `online_ray` for high-latency APIs

### Issue: Resume Not Working

**Cause:** Output file contains partial results but IDs don't match.

**Solution:**
```bash
# Remove corrupted output and restart
rm ./work/step02.jsonl
python run.py --stage step02 ...
```

### Issue: Chat Template Errors

**Cause:** Tokenizer doesn't support `apply_chat_template`.

**Solution:**
```bash
# Skip chat template application
# (Don't specify --chat-template-model-path)
```

### Issue: Ray Initialization Errors

**Solutions:**
```bash
# Clear Ray temp files
rm -rf /tmp/ray

# Or disable dashboard to save memory
ray.init(include_dashboard=False)
```

### Issue: Connection Timeout (Online)

**Solutions:**
```bash
# Increase timeout
--online-request-timeout-s 3600

# Reduce concurrency to avoid overwhelming API
--concurrency 50
```

---

## Best Practices

1. **Always use `set -euo pipefail`** in bash scripts for safety
2. **Organize outputs by timestamp** to avoid overwriting previous runs
3. **Use `tee`** to capture logs while viewing progress
4. **Start with small task sets** for debugging (`--tasks "aime2024@1"`)
5. **Monitor GPU utilization** with `nvidia-smi dmon` during runs
6. **Set appropriate max-tokens** based on task requirements
7. **Use pass@k wisely**: Higher k = more compute but better signal

---

## Example: Complete Evaluation Script

```bash
#!/usr/bin/env bash
set -euo pipefail

# Environment setup
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Configuration
TIMESTAMP=$(date +%Y%m%d%H%M%S)
REPO_ROOT=/path/to/NanoEval
WORKDIR=${REPO_ROOT}/outputs/eval_${TIMESTAMP}
LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

# Stage outputs
PREPARED_INPUT="${WORKDIR}/step01_prepared.jsonl"
INFERENCE_OUTPUT="${WORKDIR}/step02_inference.jsonl"
SCORE_OUTPUT="${WORKDIR}/step03_score.jsonl"
FINAL_EVAL_OUTPUT="${WORKDIR}/step03_final_eval.jsonl"

# Task configuration
TASK_ARGS=(
  --stage all
  --task-dir ${REPO_ROOT}/outputs/nano_eval
  --tasks "aime2024@4,aime2025@8,gpqa_diamond@1"
  --output ${PREPARED_INPUT}
  --inference-output ${INFERENCE_OUTPUT}
  --score-output ${SCORE_OUTPUT}
  --final-eval-output ${FINAL_EVAL_OUTPUT}
  --n-proc 32
)

# Inference configuration (offline example)
ROLLOUT_ARGS=(
  --backend offline
  --model-path /path/to/model
  --tp-size 1
  --dp-size 8
  --temperature 0.6
  --top-p 0.95
  --enable-thinking true
  --max-tokens 81920
  --concurrency 1024
)

# Run evaluation
python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

echo "Evaluation complete. Results in ${WORKDIR}"
```