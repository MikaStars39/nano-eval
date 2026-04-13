# NanoEval 评测指南

一份全面的大模型评测工具使用指南，涵盖项目结构、设计原理、启动评测流程以及性能调优建议。

---

## 目录

1. [项目概述](#项目概述)
2. [项目结构](#项目结构)
3. [设计原理](#设计原理)
4. [快速开始](#快速开始)
5. [评测流水线](#评测流水线)
6. [后端配置](#后端配置)
7. [性能调优](#性能调优)
8. [支持的任务类型](#支持的任务类型)
9. [输出格式说明](#输出格式说明)
10. [常见问题排查](#常见问题排查)

---

## 项目概述

NanoEval 是一个高性能、轻量级的大语言模型评测框架，专为各类推理和知识任务的基准测试而设计。主要特性包括：

- **三阶段流水线**：输入准备 (preprocess) → 推理生成 (inference) → 评分计算 (score)
- **多种推理后端**：本地部署 (SGLang)、在线 API，均通过 Ray 分布式调度
- **高吞吐设计**：基于 Ray actor 的分片并行架构
- **灵活的任务支持**：数学推理、代码生成、指令遵循、选择题等
- **Pass@k 评测**：内置支持每道题多次采样评估

---

## 项目结构

```
NanoEval/
├── nanoeval/                   # 核心评测库
│   ├── backend/               # 推理后端实现
│   │   ├── base.py             # SGLang 引擎基类，管理生命周期
│   │   ├── offline.py          # 本地批量推理 (基于 SGLang)
│   │   └── online.py          # API 在线推理 (兼容 OpenAI 接口)
│   ├── ray/                   # Ray 分布式调度
│   │   ├── actors.py          # Ray actor 封装（Offline/Online/Scoring/Preprocess）
│   │   └── utils.py           # Ray init + JSONL shard/merge
│   ├── reward/                # 评分与验证模块
│   │   ├── score.py          # 评分主控器
│   │   ├── reward.py         # 任务特定的评分路由
│   │   ├── math/             # 数学答案验证 (GSM8K、MATH、AIME 等)
│   │   ├── if_eval/          # 指令遵循评估
│   │   └── gpqa/             # 选择题答案验证
│   └── utils/                 # 工具函数
│       ├── args.py           # 命令行参数解析
│       ├── task.py           # 任务加载与预处理
│       └── logging_utils.py  # 日志配置
├── run.py                     # 主入口程序（基于 Ray actor 调度）
└── docs/                      # 文档目录
    ├── evaluation_guide.md      # 英文版指南
    └── evaluation_guide_zh.md # 中文版指南 (本文档)
```

---

## 设计原理

### 1. 三阶段流水线架构

NanoEval 采用职责分离的三阶段设计：


| 阶段             | 功能   | 关键操作                     |
| -------------- | ---- | ------------------------ |
| **preprocess** | 输入准备 | 加载任务、应用对话模板、展开 pass@k    |
| **inference**  | 推理生成 | 使用指定后端生成回复（Ray 分片并行）    |
| **score**      | 评分计算 | 判断答案、计算指标 (avg_k、pass@k) |


每个阶段都会在 `--output-dir` 下生成对应的 JSONL 文件（prepared.jsonl、inference.jsonl、score.jsonl、final_eval.jsonl），带来以下优势：

- **可恢复性**：可从任意阶段重新开始
- **可调试性**：可随时检查中间结果
- **灵活性**：同一输入可用不同后端测试

### 2. 异步生产者-消费者架构

所有后端均采用异步 I/O，包含三个并发组件：

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ 生产者    │────▶│  队列     │──▶  │  工作器.  │
│ (读取)    │     │ (缓冲)    │    │ (生成回复) │
└──────────┘     └──────────┘     └────┬─────┘
                                        │
                                        ▼
                                 ┌──────────┐
                                 │  写入器   │
                                 │  (保存)   │
                                 └──────────┘
```

核心优势：

- **最大化 GPU 利用率**：工作器无需等待 I/O
- **内存高效**：流式处理大型数据集
- **实时监控**：实时显示吞吐率指标

### 3. 统一采样参数接口

所有后端共享一致的采样参数接口：

```python
sampling_params = {
    "temperature": 0.7,        # 核心参数 (必选)
    "max_tokens": 1024,        # 核心参数 (必选)
    "top_p": 0.95,            # 可选：核采样阈值
    "top_k": 20,              # 可选：Top-k 采样
    "min_p": 0.0,             # 可选：最小概率阈值
    "presence_penalty": 0.0,  # 可选：存在惩罚
    "repetition_penalty": 1.0, # 可选：重复惩罚
}
```

### 4. 模块化后端系统


| 后端           | 适用场景       | 并发模型                 |
| ------------ | ---------- | -------------------- |
| `offline`    | 本地 GPU 推理  | `max_inflight` 异步工作器 |
| `online`     | 远程 API 调用  | `concurrency` 信号量限制  |


---

## 快速开始

### 环境安装

```bash
# 克隆仓库
git clone <repository-url>
cd NanoEval

# 安装基础依赖
pip install -r requirements.txt

# 离线后端 (SGLang) 额外依赖
pip install sglang flashinfer

# Ray 分布式后端额外依赖
pip install ray
```

### 最小化离线评测示例

```bash
python run.py \
  --stage all \
  --output-dir ./work \
  --task-dir ./outputs/nano_eval \
  --tasks "aime2024@4" \
  --backend offline \
  --model-path /path/to/your/model \
  --temperature 1.0 \
  --max-tokens 32768 \
  --concurrency 32 \
  --num-shards 4 \
  --ray-address auto
```

### 最小化在线 API 评测示例

```bash
python run.py \
  --stage all \
  --output-dir ./work \
  --task-dir ./outputs/nano_eval \
  --tasks "ifeval@1" \
  --backend online \
  --api-key "your-api-key" \
  --base-url "https://api.example.com/v1" \
  --model "gpt-4o-mini" \
  --temperature 0.7 \
  --max-tokens 4096 \
  --concurrency 100 \
  --num-shards 4 \
  --ray-address auto
```

---

## 评测流水线

### 阶段 1：输入准备 (preprocess)

该阶段负责：

1. 从 `--task-dir` 发现任务文件
2. 应用对话模板（如指定了 `--chat-template-model-path`）
3. 为 pass@k 评测展开每道题目
4. 将准备好的 prompts 写入 `--output-dir/prepared.jsonl`

```bash
python run.py \
  --stage preprocess \
  --output-dir ./work \
  --tasks "aime2024@4,aime2025@8" \
  --pass-k 1 \
  --task-dir ./outputs/nano_eval \
  --backend online \
  --chat-template-model-path /path/to/model \
  --system-prompt "You are a helpful assistant." \
  --ray-address auto
```

**任务指定语法：**

- `taskname` — 使用默认 pass-k
- `taskname@k` — 使用 k 次采样
- `all` — 自动发现目录下所有任务

### 阶段 2：推理生成 (inference)

使用指定后端生成回复。输入自动读取 `--output-dir/prepared.jsonl`，输出写入 `--output-dir/inference.jsonl`。

```bash
python run.py \
  --stage inference \
  --output-dir ./work \
  --backend offline \
  --model-path /path/to/model \
  --tp-size 1 \
  --dp-size 8 \
  --temperature 0.6 \
  --max-tokens 81920 \
  --concurrency 1024 \
  --num-shards 4 \
  --ray-address auto
```

**断点续跑**：使用 `--resume` 参数，已完成的 ID 会被跳过。

### 阶段 3：评分计算 (score)

评估回复并计算指标。输入自动读取 `--output-dir/inference.jsonl`，输出写入 `--output-dir/score.jsonl` 和 `--output-dir/final_eval.jsonl`。

```bash
python run.py \
  --stage score \
  --output-dir ./work \
  --backend online \
  --n-proc 32 \
  --ray-address auto
```

---

## 后端配置

### 离线后端 (SGLang)

用于本地模型推理，基于 SGLang 引擎。

**关键参数：**

```bash
--backend offline
--model-path /path/to/model          # 本地模型路径
--tp-size 1                          # 张量并行大小
--dp-size 8                          # 数据并行大小
--concurrency 512                    # 最大并发请求数
```

**完整示例：**

```bash
ROLLOUT_ARGS=(
  --backend offline
  --model-path /mnt/cache/Qwen3-4B-Instruct
  --tp-size 1
  --dp-size 8
  --temperature 0.6
  --top-p 0.95
  --enable-thinking true             # 思维链模型 (如 Qwen3) 使用
  --max-tokens 81920
  --concurrency 1024
)
```

**性能建议：**

- `--dp-size` 设置为可用 GPU 数量
- `--concurrency` 设置为 GPU 数量的 64-128 倍
- 仅当模型无法放入单张 GPU 时使用 `--tp-size > 1`

### 在线后端 (API)

用于远程 API 推理（兼容 OpenAI 接口）。

**关键参数：**

```bash
--backend online
--api-key "YOUR_API_KEY"
--base-url "http://host:port/v1"
--model "model-name"
--concurrency 100                    # 最大并行 API 调用数
--online-request-timeout-s 3600      # 单次请求超时时间
```

**完整示例：**

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

---

## 性能调优

### 通用调优建议


| 资源             | 推荐配置                    |
| -------------- | ----------------------- |
| **并发数**        | 从 64×GPU 数开始，增加到 GPU 饱和 |
| **批次大小**       | 越大越好（受限于 GPU 内存）        |
| **max_tokens** | 根据任务需求设置，越高 = 越慢        |
| **评分进程数**      | 匹配 CPU 核心数（通常 16-32）    |


### 离线后端调优

```bash
# 8×A100 GPU + 4B 模型
--tp-size 1          # 无需张量并行
--dp-size 8          # 8 GPU 数据并行
--concurrency 1024   # 每 GPU 128 并发

# 8×A100 GPU + 70B 模型
--tp-size 8          # 张量并行（模型分片）
--dp-size 1          # 无数据并行
--concurrency 128    # 根据内存调整
```

### 在线后端调优

```bash
# 低延迟 API (< 1s)
--concurrency 200

# 高延迟 API (10-30s)
--concurrency 2000   # 保持大量在途请求

# 有限流场景 (如 100 req/s)
--concurrency 100    # 匹配限流阈值
```

### Ray 分布式调优

```bash
# 使用 --num-shards 控制并行 actor 数量
# 离线后端：每个 shard 使用 tp_size * dp_size 个 GPU
--num-shards 4   # 4 个推理 actor 并行

# 在线后端：每个 shard 独立并发调用 API
--num-shards 8   # 8 个 actor，各自使用 --concurrency 并发
--concurrency 100

# 连接到已有 Ray 集群
--ray-address auto
```

### 评分阶段调优

```bash
# CPU 密集型操作
--n-proc 32          # 匹配 CPU 核心数

# 小数据集
--n-proc 1           # 避免多进程开销
```

### 内存优化

大模型或长上下文场景：

```bash
# 如遇到 OOM，减少内存分配比例
# （在 backend/base.py 中修改）
mem_fraction_static=0.85

# 重复前缀场景启用 Radix Cache
enable_radix_cache=true
```

---

## 支持的任务类型


| 任务                | 类型    | 评测指标              | 说明              |
| ----------------- | ----- | ----------------- | --------------- |
| **aime2024/2025** | 数学推理  | pass@k            | AIME 竞赛题目       |
| **amc2023**       | 数学推理  | pass@k            | AMC 竞赛题目        |
| **math500**       | 数学推理  | pass@k            | MATH 数据集 (500题) |
| **minerva**       | 数学推理  | pass@k            | Minerva 数学题目    |
| **hmmt2025**      | 数学推理  | pass@k            | HMMT 竞赛题目       |
| **gpqa_diamond**  | 科学选择题 | pass@k            | 研究生级别科学 QA      |
| **mmlu/mmlu_pro** | 选择题   | pass@k            | 大规模多任务语言理解      |
| **ceval**         | 选择题   | pass@k            | 中文能力评测          |
| **ifeval**        | 指令遵循  | prompt-level pass | 指令遵循评估          |
| **ifbench**       | 指令遵循  | prompt-level pass | 扩展 IF 评估        |


**添加自定义任务：**

1. 在任务目录创建 JSONL 文件：

```jsonl
{"question_id": "q1", "prompt": "2+2=？", "label": "4"}
{"question_id": "q2", "prompt": "求解：x^2 = 4", "label": "2, -2"}
```

1. 在 `nanoeval/utils/task.py` 注册：

```python
TASK_TO_JSONL = {
    "your_task": "your_task.jsonl",
    # ... 现有任务
}
```

---

## 输出格式说明

### preprocess 输出（prepared.jsonl）

```jsonl
{
  "question_id": "aime2024_1",
  "prompt": "<带对话模板的格式化 prompt>",
  "label": "42",
  "id": "aime2024_1_0",
  "source": "aime2024",
  "sample_index": 0
}
```

### inference 输出（inference.jsonl）

```jsonl
{
  "question_id": "aime2024_1",
  "prompt": "...",
  "label": "42",
  "id": "aime2024_1_0",
  "source": "aime2024",
  "sample_index": 0,
  "response": "答案是 42。",
  "thinking": "<推理过程>",  // 模型支持思维链时
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  },
  "_latency": 2.345,
  "_status": "success"
}
```

### score 评分输出（score.jsonl）

```jsonl
{
  "question_id": "aime2024_1",
  "prompt": "...",
  "label": "42",
  "id": "aime2024_1_0",
  "source": "aime2024",
  "response": "答案是 42。",
  "pred": "42",
  "pass": true,
  "pass_at_k": true  // 该题目任一采样通过则为 true
}
```

### score 指标输出（final_eval.jsonl）

```jsonl
["aime2024", {
  "avg_k": 0.25,           // 所有尝试的平均准确率
  "pass_k": 0.5,           // Pass@k：至少 1 次正确的比例
  "avg_total_tokens": 150.5,
  "avg_thinking_tokens": 45.2,
  "max_thinking_tokens": 120,
  "min_thinking_tokens": 10
}]
["overall", { ... }]
```

### CSV 输出


| task     | avg_k | pass_k | avg_total_tokens | avg_thinking_tokens | max_thinking_tokens | min_thinking_tokens |
| -------- | ----- | ------ | ---------------- | ------------------- | ------------------- | ------------------- |
| aime2024 | 0.25  | 0.5    | 150.5            | 45.2                | 120                 | 10                  |
| overall  | 0.30  | 0.6    | 145.0            | 42.0                | 120                 | 10                  |


---

## 常见问题排查

### 问题：显存不足 (OOM)

**解决方案：**

1. 降低 `--concurrency`
2. 降低 `--max-tokens`
3. 增大 `--tp-size` 启用张量并行
4. 在 `base.py` 中降低 `mem_fraction_static`

### 问题：推理速度过慢

**排查清单：**

- GPU 利用率是否 100%？（`nvidia-smi` 查看）
- `--concurrency` 是否足够高？
- 在线模式：API 延迟是否是瓶颈？

### 问题：断点续跑失效

**原因：** 输出文件包含部分结果但 ID 不匹配。

**解决方案：**

```bash
# 删除损坏的输出重新运行
rm ./work/inference.jsonl
python run.py --stage inference --output-dir ./work ...
```

### 问题：对话模板错误

**原因：** Tokenizer 不支持 `apply_chat_template`。

**解决方案：**

```bash
# 跳过对话模板应用
# （不指定 --chat-template-model-path）
```

### 问题：Ray 初始化错误

**解决方案：**

```bash
# 清除 Ray 临时文件
rm -rf /tmp/ray

# 或禁用 dashboard 节省内存
ray.init(include_dashboard=False)
```

### 问题：连接超时（在线模式）

**解决方案：**

```bash
# 增加超时时间
--online-request-timeout-s 3600

# 降低并发避免压垮 API
--concurrency 50
```

---

## 最佳实践

1. **脚本中始终使用 `set -euo pipefail`**，确保错误及时退出
2. **按时间戳组织输出目录**，避免覆盖历史结果
3. **使用 `tee` 捕获日志**，同时查看实时进度
4. **先用小任务集调试**，如 `--tasks "aime2024@1"`
5. **运行期间监控 GPU 利用率**，使用 `nvidia-smi dmon`
6. **根据任务需求设置 max-tokens**
7. **合理使用 pass@k**：k 值越高 = 计算量越大，但信号越可靠
8. **在线模式设置合理超时**，防止无限等待

---

## 完整评测脚本示例

```bash
#!/usr/bin/env bash
set -euo pipefail

# 环境设置
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

# 配置
TIMESTAMP=$(date +%Y%m%d%H%M%S)
REPO_ROOT=/path/to/NanoEval
WORKDIR=${REPO_ROOT}/outputs/eval_${TIMESTAMP}
LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

# 任务配置
TASK_ARGS=(
  --stage all
  --output-dir ${WORKDIR}
  --task-dir ${REPO_ROOT}/outputs/nano_eval
  --tasks "aime2024@4,aime2025@8,gpqa_diamond@1"
  --n-proc 32
  --num-shards 4
  --ray-address auto
)

# 推理配置 (离线示例)
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

# 运行评测
python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

echo "评测完成。结果保存在 ${WORKDIR}"
```

