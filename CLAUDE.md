> IMPORTANT:
> - 除非特殊提醒，claude code默认是在一个开发机上运行的，这意味着这台机器上并没有安装任何能跑本项目相关代码的包。
> - 所以，不要跑需要特殊包的python命令，而尽量让用户去运行。用户会把这个代码同步到GPU服务器上运行。

NanoEval 是一个轻量高性能的 LLM 评测工具，同时也兼具数据高效生产、数据处理清洗等功能。

> 本文件描述项目整体结构、核心约定和禁区。CLI 参数、输出格式、性能调优等评测细节见 `docs/evaluation.md`。

## Entry Map

```
run.py                           # 主入口：Ray 编排三阶段流水线 (preprocess→inference→score)
nanoeval/backend/online.py       # API 推理引擎 + ToolResponseMatcher + agent loop
nanoeval/backend/offline.py      # SGLang 本地推理 (producer-worker-writer 队列)
nanoeval/backend/base.py         # SGLang 引擎生命周期管理（不要直接改）
nanoeval/reward/score.py         # 评分主逻辑，聚合 pass@k
nanoeval/reward/reward.py        # 任务类型 → 评分器路由
nanoeval/reward/extract.py       # 共享答案提取（\boxed{} 解析），math/gpqa 共用
nanoeval/utils/args.py           # CLI 参数辅助工具（parse_task_pass_k 等）
nanoeval/utils/task.py           # 任务名→文件映射，JSONL 读写，pass@k 展开
nanoeval/utils/logging_utils.py  # 统一日志配置（所有模块用 logging.getLogger(__name__)）
nanoeval/ray/actors.py           # Ray actor 封装（Offline/Online/Scoring/Preprocess）
nanoeval/ray/utils.py            # Ray init + JSONL shard/merge
```

## Anti-Patterns / No-Go Zones

> **核心模块 (`nanoeval/`) 只放可复用的基建**。推理引擎、评测引擎、通用工具是核心。特定任务的逻辑（如 context_rot 的 ToolSimulator、judge prompt）放 `recipes/`。判断标准：如果这段代码只为一个实验服务，就不要放核心模块。

- **不要**在 `nanoeval/` 下创建任务专用模块（曾经有 `nanoeval/context_rot/`，已移除）
- **不要**在各模块里调 `logging.basicConfig()`。日志统一通过 `configure_logger()` 在入口配置
- **不要**直接往 `scripts/` 里写新脚本。`scripts/` 是经过 CI 测试的参考模板，新脚本写 `recipes/`
- **不要**改 `nanoeval/backend/base.py` 除非你清楚 SGLang 引擎生命周期
- **Shell 脚本风格**：参照 `recipes/eval/examples/` — 无 banner echo、无 arg parsing、env vars + 数组 + 单次 python 调用

## Directory Structure

```
nanoeval/                  # 核心可复用模块（推理引擎、评分、工具）
recipes/                   # 实验脚本和任务专用代码
  eval/examples/           # 标准评测示例（脚本风格参考）
  context_rot/             # Context Rot 评测工具集
    eval/                  #   多轮 agent loop 评测 + LLM judge
    distance/              #   SP/Query 距离敏感性实验
    data_scan/             #   训练数据偷懒模式扫描
docs/                      # 文档
  evaluation.md            # 评测参考（CLI 参数、输出格式、调优、排错）
```

## Quick Commands

```bash
# 运行测试（dev 机可以跑，不需要 GPU）
python -m pytest tests/

# 评测（需要 GPU 服务器，所有阶段通过 Ray 编排）
python run.py --output-dir ./out --backend online --tasks "math500@1" \
  --api-key $API_KEY --base-url $BASE_URL --model $MODEL ...

# Offline 评测（多 shard 并行）
python run.py --output-dir ./out --backend offline --model-path /path/to/model \
  --num-actors 4 --tp-size 8 --tasks "aime2025@8" ...

# Agent loop 模式（多轮 + 工具调用）
python run.py --output-dir ./out --backend online --agent-loop --max-turns 10 \
  --api-key $API_KEY --base-url $BASE_URL --model $MODEL --tasks "aime2025@8" ...
```

## Git Workflow

Every feature/fix follows: **new branch → commit → merge to main → push**.
