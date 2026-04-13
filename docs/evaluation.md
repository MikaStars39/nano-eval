> 本文档描述 NanoEval 的评测流水线。不包含：context_rot recipes 的使用（见 `recipes/context_rot/README.md`）。

## 三阶段流水线

所有阶段通过 `run.py` 统一调度，底层使用 Ray actors（定义在 `nanoeval/ray/actors.py`）执行。

| Stage | Ray Actor | 输入 | 输出 |
|-------|-----------|------|------|
| **preprocess** 准备 | `PreprocessActor` | 任务 JSONL + tokenizer | 合并后的 prompt JSONL |
| **inference** 推理 | `OfflineInferenceActor` / `OnlineInferenceActor` | prompt JSONL | response JSONL |
| **score** 评分 | `ScoringActor` | response JSONL | score JSONL + metrics |

每个阶段产出独立的 JSONL 文件，支持从任意阶段断点续跑（`--stage preprocess/inference/score`）。

## 后端选择

| 后端 | 场景 | 关键参数 |
|------|------|----------|
| `offline` | 本地 GPU (SGLang) | `--model-path`, `--tp-size`, `--dp-size` |
| `online` | API 调用 (OpenAI 兼容) | `--api-key`, `--base-url`, `--model`, `--concurrency` |
| `online` + `--agent-loop` | 多轮对话 + 工具调用 | 加 `--max-turns`，输入需含 `messages`/`tools`/`tool_responses` |

### Offline 注意事项

- SGLang 用 `max_new_tokens` 而非 `max_tokens`，offline 引擎自动转换
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

### inference 输入（prompt 或 messages）

```jsonl
{"id": "aime2024_1_0", "prompt": "...", "label": "42", "source": "aime2024"}
```

或 messages 格式（用于 agent loop）：

```jsonl
{"id": "test_1", "messages": [...], "tools": [...], "tool_responses": [...]}
```

### inference 输出

```jsonl
{"id": "...", "response": "...", "thinking": "...", "usage": {...}, "_status": "success"}
```

### score 输出（聚合指标）

```jsonl
["aime2024", {"avg_k": 0.25, "pass_k": 0.5, "avg_total_tokens": 150}]
```

## Environment Variables

- `NLTK_DATA`：ifeval 评分所需的 NLTK 数据路径
- `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`：Ray 分布式模式下所需
- `FLASHINFER_DISABLE_VERSION_CHECK=1`：SGLang offline 需要
