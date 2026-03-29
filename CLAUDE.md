> IMPORTANT:
> - 除非特殊提醒，claude code默认是在一个开发机上运行的，这意味着这台机器上并没有安装任何能跑本项目相关代码的包。
> - 所以，不要跑需要特殊包的python命令，而尽量让用户去运行。用户会把这个代码同步到GPU服务器上运行。

NanoEval 是一个轻量高性能的 LLM 评测工具，采用三阶段流水线架构：**输入准备 → 推理 → 评分**。

## Architecture

### 三阶段流水线

**Step01** (`nanoeval/utils/__init__.py:prepare_eval_input`) — 加载任务 JSONL 文件，应用聊天模板（通过 HuggingFace tokenizer），按 pass@k 展开样本，写出合并后的 JSONL。

**Step02** (`nanoeval/backend/runner.py:run_inference`) — 路由到对应后端进行推理：
- `offline`：本地 SGLang 引擎，异步 producer-consumer 队列，支持 tp/dp 并行
- `online`：OpenAI 兼容 API，aiohttp 异步并发
- `online_ray`：Ray Actor 分布式推理
- `mock`：仅用于测试

**Step03** (`nanoeval/reward/score.py:eval_results`) — 按任务类型路由到对应评分器，聚合 pass@k 指标。

### 核心模块

```
nanoeval/
├── backend/
│   ├── base.py          # BaseSGLangEngine：生命周期、tokenizer、安全生成
│   ├── offline.py       # 本地 SGLang 批量推理
│   ├── online.py        # API 异步推理（支持 enable_thinking、system_prompt）
│   ├── online_ray.py    # Ray 分布式推理
│   └── runner.py        # 后端路由入口
├── reward/
│   ├── reward.py        # 任务类型 → 评分器路由（ifeval/gpqa/mmlu/math/livebench）
│   ├── score.py         # 评分主逻辑，聚合 pass@k 结果
│   ├── math/            # 数学验证
│   ├── if_eval/         # 指令跟随评测
│   └── gpqa/            # 多选题验证
└── utils/
    ├── args.py          # 全部 CLI 参数定义
    ├── task.py          # 任务名→文件映射，JSONL 读写，pass@k 展开
    └── logging_utils.py # 日志配置
```

## Supported Tasks

任务文件存放于 `--task-dir`（默认 `outputs/nano_eval/`），命名规则见 `nanoeval/utils/task.py:TASK_TO_JSONL`。

> **前置准备**：运行评测前需要先下载对应任务的 JSONL 数据文件到 `--task-dir` 目录下。如果用户本地没有这些文件，可以从 HuggingFace 下载：
> https://huggingface.co/datasets/MikaStars39/nano-eval

| 类型 | 任务名 |
|------|--------|
| 数学竞赛 | `aime2024`, `aime2025`, `amc2023`, `math500`, `minerva`, `hmmt2025` |
| 科学问答 | `gpqa_diamond` |
| 多选题 | `mmlu`, `mmlu_pro`, `ceval` |
| 指令跟随 | `ifeval`, `ifbench` |

**Pass@k 语法**：`--tasks "aime2025@8,math500@1"` 表示 aime2025 每题采样 8 次，`@k` 省略时使用 `--pass-k` 默认值。

## Directory Conventions

- **`recipes/`** — 用户自行编写的评测脚本（.sh），用于跑各种模型的评测实验。Claude 生成的评测脚本默认放在这里。
- **`scripts/`** — 经过良好 CI 测试的示例脚本，作为参考和模板使用。**不要直接往 scripts/ 里写新的评测脚本。**

## Environment Variables

- `NLTK_DATA`：ifeval 评分所需的 NLTK 数据路径
- `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`：Ray 分布式模式下所需

## Git Workflow

Every feature/fix follows: **new branch → commit → merge to main → push**.

## Status Tracker

### TODO
- (none currently — add items here as new work is identified)

### Active Branches
| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Stable trunk | current |
