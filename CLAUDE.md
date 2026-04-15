> IMPORTANT:
> - 除非特殊提醒，claude code 默认是在一个开发机上运行的，这意味着这台机器上并没有安装任何能跑本项目相关代码的包。
> - 所以，不要跑需要特殊包的 python 命令，而尽量让用户去运行。用户会把这个代码同步到 GPU 服务器上运行。

NanoEval 是一个轻量高性能的 LLM 评测工具，同时也兼具数据高效生产、数据处理清洗等功能。

## Entry Map

评测入口在 `recipes/eval/run.py`（Ray 编排三阶段流水线 preprocess→inference→score）。项目顶层没有暴露的 `run.py`，所有入口脚本都在 `recipes/` 下。

## 文档导航

| 文档 | 内容 | 何时查阅 |
|------|------|----------|
| [`docs/getting_started.md`](docs/getting_started.md) | 使用指南：架构概览、环境准备、Ray 集群、Online/Offline 评测示例、参数详解、结果查看 | 需要了解如何运行评测、查看参数配置时 |
| [`docs/evaluation.md`](docs/evaluation.md) | 技术参考：CLI 参数完整列表、后端详解、Agent Loop、添加新任务、数据流字段追踪、输出格式、性能调优、FAQ | 需要查参数、改评测逻辑、排查问题时 |
| [`README.md`](README.md) | 项目概述、Quick Start、支持的任务列表、项目结构 | 需要了解项目全貌时 |
| [`recipes/README.md`](recipes/README.md) | recipes 目录索引：各实验脚本的入口和用途 | 需要写新实验脚本或查找现有脚本时 |
| [`recipes/context_rot/README.md`](recipes/context_rot/README.md) | Context Rot 评测工具：eval / distance / data_scan 三模块说明 | 需要做 Context Rot 相关工作时 |
| [`recipes/context_rot/data_scan/README.md`](recipes/context_rot/data_scan/README.md) | 训练数据偷懒模式扫描：pipeline、各脚本用法、输出字段约定 | 需要做数据质量扫描时 |

## Anti-Patterns / No-Go Zones

> **核心模块 (`nanoeval/`) 只放可复用的基建**。推理引擎、评测引擎、通用工具是核心。特定任务的逻辑（如 context_rot 的 ToolSimulator、judge prompt）放 `recipes/`。判断标准：如果这段代码只为一个实验服务，就不要放核心模块。

- **不要**在 `nanoeval/` 下创建任务专用模块（曾经有 `nanoeval/context_rot/`，已移除）
- **不要**在各模块里调 `logging.basicConfig()`。日志统一通过 `configure_logger()` 在入口配置
- **不要**直接往 `scripts/` 里写新脚本。`scripts/` 是经过 CI 测试的参考模板，新脚本写 `recipes/`
- **不要**改 `nanoeval/backend/base.py` 除非你清楚 SGLang 引擎生命周期
- **Shell 脚本风格**：参照 `recipes/eval/examples/` — 无 banner echo、无 arg parsing、env vars + 数组 + 单次 python 调用

## Git Workflow

Every feature/fix follows: **new branch → commit → merge to main → push**.
