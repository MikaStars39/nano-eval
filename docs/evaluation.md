> 本文档描述 NanoEval 的评测流水线。不包含：context_rot recipes 的使用（见 `recipes/context_rot/README.md`）、Ray 分布式部署（见 `recipes/eval/run_ray.py`）。

## 三阶段流水线

| Stage | 入口 | 输入 | 输出 |
|-------|------|------|------|
| **Step01** 准备 | `nanoeval/utils/__init__.py:prepare_eval_input` | 任务 JSONL + tokenizer | 合并后的 prompt JSONL |
| **Step02** 推理 | `nanoeval/backend/runner.py:run_inference` | prompt JSONL | response JSONL |
| **Step03** 评分 | `nanoeval/reward/score.py:eval_results` | response JSONL | score JSONL + metrics |

每个阶段产出独立的 JSONL 文件，支持从任意阶段断点续跑。

## 后端选择

| 后端 | 场景 | 关键参数 |
|------|------|----------|
| `offline` | 本地 GPU (SGLang) | `--model-path`, `--tp-size`, `--dp-size` |
| `online` | API 调用 (OpenAI 兼容) | `--api-key`, `--base-url`, `--model`, `--concurrency` |
| `online` + `--agent-loop` | 多轮对话 + 工具调用 | 加 `--max-turns`，输入需含 `messages`/`tools`/`tool_responses` |
| `mock` | 测试 | 无需额外参数 |

### Offline 注意事项

- SGLang 用 `max_new_tokens` 而非 `max_tokens`，runner.py 自动转换
- `--dp-size` 设为 GPU 数量，`--tp-size` 仅在单卡装不下模型时 >1
- `--concurrency` 建议 64-128× GPU 数

### Online 注意事项

- 支持 `enable_thinking`（通过 `chat_template_kwargs` 传递）
- 支持 `__system_prompt` 注入（通过 sampling_params 内部键）
- 自动 resume：已有输出文件时跳过已完成的 ID

## 支持的任务

任务文件存放于 `--task-dir`（默认 `outputs/nano_eval/`），命名规则见 `nanoeval/utils/task.py:TASK_TO_JSONL`。

> **前置准备**：需要先下载 JSONL 数据文件：https://huggingface.co/datasets/MikaStars39/nano-eval

| 类型 | 任务名 |
|------|--------|
| 数学竞赛 | `aime2024`, `aime2025`, `amc2023`, `math500`, `minerva`, `hmmt2025` |
| 科学问答 | `gpqa_diamond` |
| 多选题 | `mmlu`, `mmlu_pro`, `ceval` |
| 指令跟随 | `ifeval`, `ifbench` |

**Pass@k 语法**：`--tasks "aime2025@8,math500@1"` — `@k` 省略时用 `--pass-k` 默认值。

## JSONL 格式

### Step02 输入（prompt 或 messages）

```jsonl
{"id": "aime2024_1_0", "prompt": "...", "label": "42", "source": "aime2024"}
```

或 messages 格式（用于 agent loop）：

```jsonl
{"id": "test_1", "messages": [...], "tools": [...], "tool_responses": [...]}
```

### Step02 输出

```jsonl
{"id": "...", "response": "...", "thinking": "...", "usage": {...}, "_status": "success"}
```

### Step03 输出（聚合指标）

```jsonl
["aime2024", {"avg_k": 0.25, "pass_k": 0.5, "avg_total_tokens": 150}]
```

## Environment Variables

- `NLTK_DATA`：ifeval 评分所需的 NLTK 数据路径
- `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`：Ray 分布式模式下所需
- `FLASHINFER_DISABLE_VERSION_CHECK=1`：SGLang offline 需要
