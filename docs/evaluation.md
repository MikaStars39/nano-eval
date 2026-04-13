# NanoEval 评测参考

> 本文档是评测流水线的完整参考。项目结构和约定见 `CLAUDE.md`。

## 三阶段流水线

所有阶段通过 `run.py` 统一调度，底层使用 Ray actors（`nanoeval/ray/actors.py`）执行。

| Stage | Ray Actor | 输入 | 输出 |
|-------|-----------|------|------|
| **preprocess** | `PreprocessActor` | 任务 JSONL + tokenizer | `prepared.jsonl` |
| **inference** | `OfflineInferenceActor` / `OnlineInferenceActor` | `prepared.jsonl` | `inference.jsonl` |
| **score** | `ScoringActor` | `inference.jsonl` | `score.jsonl` + `final_eval.jsonl` + `final_eval.csv` |

每个阶段产出独立文件，支持 `--stage preprocess/inference/score` 断点续跑。

## CLI 参数完整参考

> 以下与 `run.py` 源码一致，按分组列出。

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
| `--num-shards` | `1` | 推理 actor 数量 |
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
| 多选题 | `mmlu`, `mmlu_pro`, `ceval` |
| 指令跟随 | `ifeval`, `ifbench` |

任务注册在 `nanoeval/utils/task.py:TASK_TO_JSONL`。

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
| 多 shard 并行 | `--num-shards N`，总并发 = N × concurrency |
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
