# NanoEval 评测参考

> 本文档是评测流水线的完整参考。项目结构和约定见 `CLAUDE.md`。

## 三阶段流水线

所有阶段通过 `recipes/eval/run.py` 统一调度，底层使用 Ray actors（`nanoeval/ray/actors.py`）执行。

| Stage | Ray Actor | 输入 | 输出 |
|-------|-----------|------|------|
| **preprocess** | `PreprocessActor` | 任务 JSONL + tokenizer | `prepared.jsonl` |
| **inference** | `OfflineInferenceActor` / `OnlineInferenceActor` | `prepared.jsonl` | `inference.jsonl` |
| **score** | `ScoringActor` | `inference.jsonl` | `score.jsonl` + `final_eval.jsonl` + `final_eval.csv` |

每个阶段产出独立文件，支持 `--stage preprocess/inference/score` 断点续跑。

## CLI 参数完整参考

> 以下与 `recipes/eval/run.py` 源码一致，按分组列出。

### 数据 / 任务

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tasks` | `"all"` | 逗号分隔任务名，支持 `task@k` 语法 |
| `--task-dir` | `outputs/nano_eval` | 任务 JSONL 数据目录 |
| `--pass-k` | `1` | 默认 pass@k 值（`@k` 省略时使用） |
| `--output-dir` | **必填** | 输出目录 |
| `--stage` | `all` | `preprocess` / `inference` / `score` / `all` |

### Preprocess

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chat-template-model-path` | `None`（回退到 `--model-path`） | Chat template 的 tokenizer 路径 |
| `--system-prompt` | `None` | System prompt 注入 |

### 后端选择

| 参数 | 说明 |
|------|------|
| `--backend` | **必填**，`offline` 或 `online` |

### Offline (SGLang)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `None` | 本地模型路径（offline 必填） |
| `--tp-size` | `8` | 张量并行度 |
| `--dp-size` | `1` | 数据并行度 |
| `--max-inflight` | `512` | 最大同时在途请求数 |
| `--mem-fraction-static` | `0.90` | GPU 显存分配比例 |
| `--enable-dp-attention` | `false` | 启用 DP attention |

### Online (API)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-key` | `None` | API key（online 必填） |
| `--base-url` | `None` | API endpoint（online 必填） |
| `--model` | `None` | 模型名（online 必填） |
| `--concurrency` | `32` | 最大并行请求数 |

### Agent Loop（仅 online）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent-loop` | `false` | 启用多轮 agent loop + 工具调用 |
| `--max-turns` | `10` | 每轮对话最大轮数 |

Agent loop 输入需包含 `messages`、`tools`、`tool_responses` 字段。

### 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--temperature` | `0.7` | 温度 |
| `--max-tokens` | `1024` | 最大生成 token 数 |
| `--top-p` | `None` | 核采样 |
| `--top-k` | `None` | Top-k 采样 |
| `--min-p` | `None` | 最小概率阈值 |
| `--presence-penalty` | `None` | 存在惩罚 |
| `--repetition-penalty` | `None` | 重复惩罚 |
| `--reasoning-effort` | `None` | `low`/`medium`/`high`，思考模型推理力度 |
| `--enable-thinking` | `None` | `true`/`false`，启用思维链模式（Qwen3 等） |

### Ray / 分片

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-actors` | `1` | 并行推理 actor 数量 |
| `--ray-address` | `auto` | Ray 集群地址 |
| `--resume` | `false` | 跳过已完成的 ID，断点续跑 |

### 评分

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-proc` | `32` | 评分并行进程数 |

## 后端详解

### Offline

- SGLang 引擎使用 `max_new_tokens` 而非 `max_tokens`，引擎自动转换
- `--dp-size` 设为 GPU 数量，`--tp-size` 仅在模型单卡放不下时 >1
- 每个 shard 占用 `tp_size × dp_size` 个 GPU

### Online

- 使用 aiohttp 异步客户端 + semaphore 控制并发
- 自动重试 429/5xx（指数退避）
- `--system-prompt` 通过内部键 `__system_prompt` 注入
- `--enable-thinking` 通过 `chat_template_kwargs` 传递

### Agent Loop

- 仅 online 后端支持
- `ToolResponseMatcher` 匹配策略：精确 ID → 函数名 → 按序回退
- 退出条件：`max_turns` / `completed` / `empty_response`
- 输入 JSONL 格式：

```jsonl
{"id": "test_1", "messages": [...], "tools": [...], "tool_responses": [...]}
```

## 支持的任务

> 前置准备：下载 JSONL 数据 https://huggingface.co/datasets/MikaStars39/nano-eval

| 类型 | 任务名 |
|------|--------|
| 数学竞赛 | `aime2024`, `aime2025`, `amc2023`, `math500`, `minerva`, `hmmt2025` |
| 科学问答 | `gpqa_diamond` |
| 多选题 | `mmlu`, `mmlu_pro`, `mmlu_prox`, `ceval` |
| 指令跟随 | `ifeval`, `ifbench` |

任务注册在 `nanoeval/utils/task.py:TASK_TO_JSONL`。未注册的任务名会自动 fallback 到 `{task_name}.jsonl`，无需改代码。

## 添加新评测任务

完整流程分四步：数据准备 → 注册任务 → 评分路由 → 上传数据。

### Step 1: 数据准备

将原始数据集（如 HuggingFace 上的公开数据）转换为 `--task-dir` 下的 JSONL 格式。

**目标格式**（每行一个 JSON 对象）：

```jsonl
{"question_id": "q1", "prompt": "题目文本", "label": "正确答案"}
```

- `question_id`：题目唯一标识
- `prompt`：完整的题目文本（含选项、指令等）
- `label`：标准答案（数学题为数值/表达式，选择题为字母）
- 可携带额外字段（`category`、`language` 等），不影响流水线

**参考脚本**：`scripts/prepare_mmlu_prox.py` — 从 HuggingFace MMLU-ProX 数据集转换为 JSONL，展示了完整模式：

```python
# 核心转换逻辑：原始 record → nano-eval JSONL record
def convert_record(record, language):
    prompt = format_prompt(record["question"], options)  # 拼装题目+选项
    return {
        "question_id": f"{language}_{record['question_id_src']}",
        "prompt": prompt,
        "label": record["answer"],  # 如 "A"
    }
```

**选择题 prompt 规范**：答案要求放在 `\boxed{}` 中，便于评分器提取：

```
题目文本

A. 选项1
B. 选项2
C. 选项3
D. 选项4

Please select the correct answer and put the letter in \boxed{}, e.g., \boxed{A}.
```

新的数据准备脚本放 `scripts/` 下，命名 `prepare_<task>.py`。

### Step 2: 注册任务（可选）

`TASK_TO_JSONL` 字典支持别名映射（task 名与文件名不同时使用）。
如果 JSONL 文件名就是 `{task_name}.jsonl`，无需注册——auto-discovery 会自动找到。

```python
# 仅在 task 名和文件名不一致时需要添加
TASK_TO_JSONL = {
    ...
    "your_task": "your_task.jsonl",  # 可省略，auto-discovery 会找到
}
```

### Step 3: 评分路由

`nanoeval/reward/reward.py:judge_router` 按 `source` 名分发评分器：

| source 包含 | 评分器 | 适用场景 |
|-------------|--------|----------|
| `ifeval` | `if_judge` | 指令遵循 |
| `gpqa` / `mmlu` | `gpqa_judge` | 选择题 |
| 其他（兜底） | `math_judge` | 数学题（`extract_answer` 提取 `\boxed{}` + 符号验证） |

- 数学类任务：无需改动，`math_judge` 兜底即可
- 选择题任务：source 名包含 `gpqa` 或 `mmlu` 即可，否则需在 `judge_router` 加分支
- 全新评分逻辑：写新 judge 函数，在 `judge_router` 加条件分支

### Step 4: 上传数据

将生成的 JSONL 上传到 https://huggingface.co/datasets/MikaStars39/nano-eval ，供其他用户下载使用。

## 数据流字段追踪

每条记录在三个阶段中依次被追加字段，原有字段全部透传。

### Preprocess 追加

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | `{question_id}_{sample_index}`，全局唯一 |
| `question_id` | str | 题目 ID（同题的 k 个样本共享） |
| `source` | str | 任务名（如 `aime2024`） |
| `sample_index` | int | 0 到 pass_k-1 |
| `prompt` | str | 若指定 chat template 则转换后的 prompt |

### Inference 追加

| 字段 | 类型 | Online | Offline | 说明 |
|------|------|--------|---------|------|
| `response` | str | ✓ | ✓ | 模型生成内容 |
| `thinking` | str? | ✓ | — | reasoning 字段（模型支持时） |
| `usage` | dict | ✓ | 写前移除 | token 统计 |
| `_latency` | float | ✓ | 写前移除 | 推理耗时（秒） |
| `_status` | str | ✓ | ✓ | `"success"` / `"failed"` |
| `_error` | str? | 仅失败 | 仅失败 | 错误信息 |
| `turns` | int | agent loop | — | 对话轮数 |
| `exit_reason` | str | agent loop | — | 退出原因 |

### Score 追加

| 字段 | 类型 | 说明 |
|------|------|------|
| `pred` | str? | 提取的预测答案 |
| `pass` | bool | 是否正确 |
| `pass_at_k` | bool | 该题在 k 个样本中是否有至少 1 个正确 |

> IFEval 额外追加 `instruction_count`（int）和 `instruction_pass_cnt`（int）。
> Score 阶段会从 `response` 中提取 `<think>`/`</thinking>` 块到 `thinking` 字段。

## 输出格式

### prepared.jsonl

```jsonl
{"question_id": "aime2024_1", "prompt": "...", "label": "42", "id": "aime2024_1_0", "source": "aime2024", "sample_index": 0}
```

### inference.jsonl

```jsonl
{"question_id": "aime2024_1", "id": "aime2024_1_0", "source": "aime2024", "response": "...", "thinking": "...", "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, "_latency": 2.345, "_status": "success"}
```

### score.jsonl（逐条）

```jsonl
{"question_id": "aime2024_1", "id": "aime2024_1_0", "source": "aime2024", "pred": "42", "pass": true, "pass_at_k": true}
```

### final_eval.jsonl（聚合指标）

```jsonl
["aime2024", {"avg_k": 0.25, "pass_k": 0.5, "avg_total_tokens": 150.5, "avg_thinking_tokens": 45.2, "max_thinking_tokens": 120, "min_thinking_tokens": 10}]
["overall", {...}]
```

## 性能调优速查

| 场景 | 推荐配置 |
|------|----------|
| 小模型 (4B) × 8 GPU | `--tp-size 1 --dp-size 8 --concurrency 1024` |
| 大模型 (70B) × 8 GPU | `--tp-size 8 --dp-size 1 --concurrency 128` |
| 低延迟 API | `--concurrency 200` |
| 高延迟 API (10-30s) | `--concurrency 2000` |
| 有限流 API | `--concurrency` 匹配限流阈值 |
| 多 actor 并行 | `--num-actors N`，总并发 = N × concurrency |
| 评分 | `--n-proc` 匹配 CPU 核心数 |

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| OOM | 降 `--concurrency`、降 `--max-tokens`、升 `--tp-size`、降 `--mem-fraction-static` |
| 推理慢 | 检查 GPU 利用率 (`nvidia-smi`)，提高 `--concurrency` |
| resume 失效 | 删除 `inference.jsonl` 重跑 |
| chat template 报错 | 不指定 `--chat-template-model-path` 跳过 |
| Ray 初始化失败 | `rm -rf /tmp/ray` 清理临时文件 |
| 连接超时 (online) | 降 `--concurrency`，API 端检查限流 |

## 环境变量

| 变量 | 用途 |
|------|------|
| `NLTK_DATA` | ifeval 评分所需 |
| `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` | Ray 分布式必需 |
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | SGLang offline 必需 |
