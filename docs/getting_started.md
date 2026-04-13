# NanoEval 评测使用指南

本文档从架构到实操，手把手介绍如何使用 NanoEval 跑评测。参数速查和高级配置见 [`evaluation.md`](evaluation.md)。

## 目录

- [架构概览](#架构概览)
- [环境准备](#环境准备)
- [启动 Ray 集群](#启动-ray-集群)
- [Online 评测（API 推理）](#online-评测api-推理)
- [Offline 评测（本地 GPU 推理）](#offline-评测本地-gpu-推理)
- [参数详解](#参数详解)
- [查看评测结果](#查看评测结果)
- [进阶用法](#进阶用法)

---

## 架构概览

NanoEval 采用 **三阶段 Ray 流水线**，每个阶段由独立的 Ray Actor 执行：

```
                  ┌──────────────┐
                  │   run.py     │  统一入口
                  └──────┬───────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌─────────────┐ ┌──────────┐ ┌─────────────┐
   │ 1.Preprocess │ │2.Infer   │ │ 3.Score     │
   │             │ │          │ │             │
   │ 加载任务数据  │ │ 模型推理  │ │ 评分+聚合   │
   │ 应用模板     │ │ (多actor) │ │             │
   │ 展开 pass@k  │ │          │ │             │
   └──────┬──────┘ └────┬─────┘ └──────┬──────┘
          │              │              │
          ▼              ▼              ▼
    prepared.jsonl  inference.jsonl  score.jsonl
                                    final_eval.jsonl
                                    final_eval.csv
```

### 三阶段说明

| 阶段 | Actor | 做什么 | 产出 |
|------|-------|--------|------|
| **Preprocess** | `PreprocessActor` | 读取任务 JSONL，应用 chat template，按 pass@k 复制样本 | `prepared.jsonl` |
| **Inference** | `OfflineInferenceActor` 或 `OnlineInferenceActor` | 把 prepared 数据分片发给多个 actor 并行推理 | `inference.jsonl` |
| **Score** | `ScoringActor` | 对推理结果评分（数学题提取答案、选择题匹配、指令跟随检查），汇总指标 | `score.jsonl` + `final_eval.jsonl` + `final_eval.csv` |

**关键特性：**
- 每阶段产出独立文件 — 可以用 `--stage` 单独跑某个阶段，断点续跑
- Inference 阶段支持多 actor 并行 — 数据自动分片，推理完自动合并
- 支持 `--resume` 跳过已完成的样本

### 两种推理后端

| | Online | Offline |
|---|--------|---------|
| **推理方式** | 调用远程 API（OpenAI 兼容） | 本地 SGLang 引擎 |
| **资源** | 不需要 GPU，靠网络 | 需要 GPU |
| **并发控制** | `--concurrency`（异步 HTTP） | `--max-inflight`（异步队列） |
| **额外能力** | 支持 agent loop（多轮工具调用） | — |
| **典型场景** | 评测 API 服务、闭源模型 | 评测本地模型、不想部署服务 |

---

## 环境准备

### 1. 下载评测数据

从 HuggingFace 下载任务 JSONL 数据：

```bash
# 放到 outputs/nano_eval/ 目录下
mkdir -p outputs/nano_eval
# 从 https://huggingface.co/datasets/MikaStars39/nano-eval 下载
```

下载后目录结构：

```
outputs/nano_eval/
├── aime2024.jsonl
├── aime2025.jsonl
├── math500.jsonl
├── gpqa_diamond.jsonl
├── mmlu.jsonl
├── ifeval.jsonl
└── ...
```

### 2. 安装依赖

```bash
pip install ray aiohttp tqdm
# Offline 还需要：
pip install sglang torch transformers
# IFEval 评分需要：
pip install nltk
```

### 3. 环境变量

```bash
# Ray 分布式必需
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Offline (SGLang) 必需
export FLASHINFER_DISABLE_VERSION_CHECK=1

# IFEval 评分需要 NLTK 数据
export NLTK_DATA=/path/to/nltk_data
```

---

## 启动 Ray 集群

NanoEval 使用 Ray 编排所有阶段。两种用法：

### 单机模式（默认）

不需要手动启动 Ray — 脚本会自动 `ray.init(address="auto")` 拉起本地集群：

```bash
# 直接跑，不需要额外操作
python run.py --backend online --ray-address auto ...
```

### 多机模式

需要先手动组建 Ray 集群：

```bash
# 主节点
ray start --head --port=6379

# 工作节点（每台 GPU 机器上执行）
ray start --address="主节点IP:6379"

# 验证集群
ray status
```

然后在评测命令中指定：

```bash
python run.py --ray-address "主节点IP:6379" ...
```

> **Tip:** 单机多卡场景不需要手动启 Ray，`--ray-address auto` 即可。只有跨机器并行时才需要手动组集群。

---

## Online 评测（API 推理）

适用场景：评测已部署的 API 服务（OpenAI 兼容接口）、闭源模型。

### 最小示例

```bash
python run.py \
  --output-dir ./out/my_eval \
  --backend online \
  --api-key "$API_KEY" \
  --base-url "$BASE_URL" \
  --model "$MODEL_NAME" \
  --tasks "math500@1" \
  --task-dir outputs/nano_eval \
  --temperature 0.7 \
  --max-tokens 4096
```

### 完整示例（多任务 + 思考模式）

```bash
python run.py \
  --output-dir ./out/my_eval \
  --backend online \
  --api-key "$API_KEY" \
  --base-url "$BASE_URL" \
  --model "$MODEL_NAME" \
  --tasks "gpqa_diamond@4,math500@1,aime2025@8,ifeval@1" \
  --task-dir outputs/nano_eval \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 131072 \
  --enable-thinking true \
  --concurrency 1024 \
  --n-proc 32 \
  --num-actors 1
```

### 参数说明

- `--api-key`、`--base-url`、`--model`：**必填**，指定 API 连接信息
- `--concurrency`：每个 actor 的最大并发请求数。API 延迟高（10s+）可调大（1000+），有限流则匹配限流阈值
- `--num-actors`：Online 下每个 actor 只占 1 CPU，可以开多个加速。总并发 = `num_actors × concurrency`

### Shell 脚本模板

参考 `recipes/eval/examples/thinking_online.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/online_${TIMESTAMP}"

python "${REPO_ROOT}/run.py" \
  --tasks "gpqa_diamond@4,math500@1,aime2025@8,ifeval@1" \
  --task-dir "${REPO_ROOT}/outputs/nano_eval" \
  --output-dir "${WORKDIR}" \
  --stage all \
  --backend online \
  --api-key "${API_KEY:?Set API_KEY}" \
  --base-url "${BASE_URL:?Set BASE_URL}" \
  --model "${MODEL_NAME:?Set MODEL_NAME}" \
  --temperature 1.0 \
  --top-p 0.95 \
  --enable-thinking true \
  --max-tokens 131072 \
  --concurrency 1024 \
  --n-proc 32 \
  --num-actors "${NUM_ACTORS:-1}" \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/run.log"
```

---

## Offline 评测（本地 GPU 推理）

适用场景：评测本地模型权重（HuggingFace 格式），不需要部署 API 服务。

### 最小示例

```bash
python run.py \
  --output-dir ./out/my_eval \
  --backend offline \
  --model-path /path/to/model \
  --tasks "math500@1" \
  --task-dir outputs/nano_eval \
  --tp-size 1 \
  --dp-size 8 \
  --temperature 0.7 \
  --max-tokens 4096
```

### 完整示例（思考模式 + 大 token）

```bash
python run.py \
  --output-dir ./out/my_eval \
  --backend offline \
  --model-path /path/to/Qwen3-8B \
  --tasks "ifeval@1" \
  --task-dir outputs/nano_eval \
  --tp-size 1 \
  --dp-size 8 \
  --temperature 0.6 \
  --top-p 0.95 \
  --enable-thinking true \
  --max-tokens 81920 \
  --mem-fraction-static 0.90 \
  --n-proc 32 \
  --num-actors 1
```

### GPU 分配逻辑

Offline 的核心概念是 **张量并行 (TP)** 和 **数据并行 (DP)**：

```
每个 actor 占用 GPU 数 = tp_size × dp_size
总 GPU 数 = tp_size × dp_size × num_actors
```

| 场景 | tp_size | dp_size | 每 actor GPU | 说明 |
|------|---------|---------|-------------|------|
| 小模型 (4B~8B) × 8 GPU | 1 | 8 | 8 | 模型单卡放得下，8 路数据并行 |
| 大模型 (70B) × 8 GPU | 8 | 1 | 8 | 模型需要 8 卡才放得下 |
| 中模型 (32B) × 8 GPU | 4 | 2 | 8 | 4 卡放模型，2 路数据并行 |
| 多机 (8B) × 16 GPU | 1 | 8 | 8 | 2 个 actor，各占 8 GPU |

> **经验法则：** `--tp-size` 设为模型能放下的最小 GPU 数。剩余 GPU 数给 `--dp-size` 做数据并行，吞吐更高。

### Shell 脚本模板

参考 `recipes/eval/examples/thinking_offline.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
WORKDIR="${REPO_ROOT}/outputs/offline_${TIMESTAMP}"

python "${REPO_ROOT}/run.py" \
  --tasks "ifeval@1" \
  --task-dir "${REPO_ROOT}/outputs/nano_eval" \
  --output-dir "${WORKDIR}" \
  --stage all \
  --backend offline \
  --model-path "/path/to/model" \
  --tp-size "${TP_SIZE:-1}" \
  --dp-size "${DP_SIZE:-8}" \
  --temperature 0.6 \
  --top-p 0.95 \
  --enable-thinking true \
  --max-tokens 81920 \
  --n-proc 32 \
  --num-actors "${NUM_ACTORS:-1}" \
  --ray-address "${RAY_ADDRESS:-auto}" 2>&1 | tee "${WORKDIR}/run.log"
```

---

## 参数详解

### 任务指定 (`--tasks`)

使用 `task@k` 语法指定任务和 pass@k：

```bash
--tasks "math500@1"                        # 单任务，pass@1
--tasks "aime2025@8"                       # 单任务，pass@8（每题生成 8 次）
--tasks "gpqa_diamond@4,math500@1"         # 多任务，不同 pass@k
--tasks "all"                              # 自动发现 task-dir 下所有任务
```

- `@k` 省略时使用 `--pass-k` 的值（默认 1）
- pass@k 越大，每道题生成的回答越多，能更准确估计模型能力，但推理量也线性增长

**可用任务列表：**

| 类型 | 任务名 |
|------|--------|
| 数学竞赛 | `aime2024`, `aime2025`, `amc2023`, `math500`, `minerva`, `hmmt2025` |
| 科学问答 | `gpqa_diamond` |
| 多选题 | `mmlu`, `mmlu_pro`, `mmlu_prox`, `ceval` |
| 指令跟随 | `ifeval`, `ifbench` |

### 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--temperature` | `0.7` | 采样温度。0 = 贪心，1 = 标准采样 |
| `--max-tokens` | `1024` | 最大生成 token 数。思考模型建议 80000+ |
| `--top-p` | 不启用 | 核采样阈值 |
| `--top-k` | 不启用 | Top-k 采样 |
| `--min-p` | 不启用 | 最小概率阈值 |
| `--presence-penalty` | 不启用 | 存在惩罚 |
| `--repetition-penalty` | 不启用 | 重复惩罚 |
| `--reasoning-effort` | 不启用 | `low`/`medium`/`high`，控制推理力度 |
| `--enable-thinking` | 不启用 | `true`/`false`，启用思维链模式（Qwen3 等） |

### 并行与分片

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-actors` | `1` | 并行推理 actor 数。Offline: 每个占 `tp×dp` 个 GPU；Online: 每个占 1 CPU |
| `--concurrency` | `32` | Online 每个 actor 的并发请求数 |
| `--max-inflight` | `512` | Offline 每个 actor 的在途请求数 |
| `--n-proc` | `32` | Score 阶段的并行进程数 |

### 流程控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage` | `all` | 运行哪个阶段：`preprocess`、`inference`、`score`、`all` |
| `--resume` | `false` | 跳过已推理的样本，断点续跑 |
| `--max-examples` | 不限制 | 限制每个任务的样本数（调试用） |

---

## 查看评测结果

评测完成后，`--output-dir` 下会产出以下文件：

```
output-dir/
├── prepared.jsonl          # 预处理后的输入
├── inference.jsonl         # 推理结果
├── score.jsonl             # 逐条评分
├── final_eval.jsonl        # 聚合指标（JSON）
├── final_eval.csv          # 聚合指标（CSV）  ← 最终结果看这个
└── shards/                 # 中间分片（可忽略）
```

### 查看 CSV 结果

`final_eval.csv` 是最直接的结果汇总：

```bash
cat output-dir/final_eval.csv
```

输出示例：

```csv
task,avg_k,pass_k,avg_total_tokens,avg_thinking_tokens,max_thinking_tokens,min_thinking_tokens
aime2025,0.2500,0.5000,15234.3,12045.2,32000,1024
math500,0.8200,0.8200,2048.5,1536.1,8192,256
gpqa_diamond,0.4500,0.6250,4096.7,3012.4,16000,512
overall,0.5067,0.6483,7126.5,5531.2,32000,256
```

**指标含义：**

| 指标 | 含义 |
|------|------|
| `avg_k` | 平均通过率 = 正确样本数 / 总样本数 |
| `pass_k` | Pass@k 率 = 有至少 1 个正确样本的题目比例 |
| `avg_total_tokens` | 平均总 token 数 |
| `avg_thinking_tokens` | 平均思考 token 数（思考模型才有意义） |
| `max_thinking_tokens` | 最长思考 token 数 |
| `min_thinking_tokens` | 最短思考 token 数 |

> **`avg_k` vs `pass_k`：** 当 pass@k=1 时两者相等。pass@k>1 时，`pass_k` 更宽松（只要有一次答对就算过），`avg_k` 是严格平均。

### 查看逐条结果

`score.jsonl` 包含每个样本的详细评分：

```bash
# 查看某条的评分
head -1 output-dir/score.jsonl | python -m json.tool
```

```json
{
    "question_id": "aime2024_1",
    "id": "aime2024_1_0",
    "source": "aime2024",
    "prompt": "...",
    "label": "42",
    "response": "...",
    "thinking": "...",
    "pred": "42",
    "pass": true,
    "pass_at_k": true
}
```

### 查看推理日志

运行时的日志会打印到 stderr，包含每个阶段的进度和最终指标：

```
2026-04-13 10:00:00 [INFO] [preprocess] done: {'task_count': 3, 'instance_count': 520}
2026-04-13 10:00:05 [INFO] [inference] split into 1 actor(s)
2026-04-13 10:05:00 [INFO] [inference] merged 520 lines -> ./out/inference.jsonl
2026-04-13 10:05:10 [INFO]   aime2025: avg_k=0.2500  pass_k=0.5000
2026-04-13 10:05:10 [INFO]   math500: avg_k=0.8200  pass_k=0.8200
2026-04-13 10:05:10 [INFO] [done]
```

---

## 进阶用法

### 断点续跑

推理中途中断时，可以只重跑 inference 和 score 阶段：

```bash
# 从 inference 阶段恢复，跳过已完成的样本
python run.py \
  --output-dir ./out/my_eval \
  --backend online \
  --stage inference \
  --resume \
  ...其他参数不变
```

然后单独跑 score：

```bash
python run.py \
  --output-dir ./out/my_eval \
  --backend online \
  --stage score \
  --n-proc 32 \
  ...
```

### 调试模式

用 `--max-examples` 限制每个任务只跑几条，快速验证流程是否正确：

```bash
python run.py \
  --max-examples 5 \
  --tasks "aime2025@1" \
  ...
```

### Agent Loop（多轮工具调用）

仅 Online 后端支持。输入 JSONL 需要包含 `messages`、`tools`、`tool_responses` 字段：

```bash
python run.py \
  --backend online \
  --agent-loop \
  --max-turns 10 \
  ...
```

输入数据格式：

```json
{
  "id": "test_1",
  "messages": [{"role": "user", "content": "帮我查一下..."}],
  "tools": [{"type": "function", "function": {"name": "Search", "parameters": {...}}}],
  "tool_responses": [
    {"name": "Search", "arguments": {"query": "test"}, "response": "搜索结果..."}
  ]
}
```

模型会自动调用工具，`ToolResponseMatcher` 从预录的 `tool_responses` 中匹配返回。最多执行 `--max-turns` 轮。

### System Prompt 注入

```bash
# Online：通过内部键注入到 API 请求的 system message
python run.py --backend online --system-prompt "你是一个数学助手" ...

# Offline：在 preprocess 阶段通过 chat template 注入
python run.py --backend offline --system-prompt "你是一个数学助手" ...
```
