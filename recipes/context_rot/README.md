# Context Rot - 长对话质量退化评测工具

Context Rot 是一套用于研究 LLM 在长上下文 Agent 场景中性能退化的评测和数据工具。包含 3 个主要任务模块：

## 模块

### 1. eval/ — 核心评测

在 3 类 Agent 任务上评估模型的长对话质量退化：
- **resume-screening** — 简历筛选（15份简历，7步流程）
- **competitive-analysis** — 竞品分析（12个竞品，8步流程）
- **stock-research** — 股票研究（20个问题，6步流程）

每个 case 生成 P1-P5 五个测试点，对应不同的上下文前缀长度，用于观察质量退化曲线。

**流程**: `prepare_eval.py` 生成 eval_set → `run_eval.py` Agent 循环 + LLM Judge 评分 → `report.py` 生成退化曲线

```bash
# 准备测试集
python3 recipes/context_rot/eval/prepare_eval.py \
    --input /path/to/context_rot_data.jsonl \
    --output /path/to/eval_set.jsonl

# 运行评测（参见 run_eval.sh 中的完整参数）
export API_KEY=sk-xxx API_BASE=https://... MODEL=MiniMax-M2.7
export JUDGE_API_KEY=sk-xxx JUDGE_API_BASE=https://... JUDGE_MODEL=gpt-5.3-codex
export INPUT=/path/to/eval_set.jsonl
bash recipes/context_rot/eval/run_eval.sh

# 生成报告
python3 recipes/context_rot/eval/report.py --input /path/to/results.jsonl
```

### 2. distance/ — 距离敏感性实验

分析模型退化是 System Prompt 距离还是 User Query 距离导致的：
- **SP Distance**: 在 System Prompt 和真实内容之间插入无关对话
- **Query Distance**: 在续写提示后插入虚拟工具调用

```bash
export API_KEY=sk-xxx API_BASE=https://... MODEL=MiniMax-M2.7
export JUDGE_API_KEY=sk-xxx JUDGE_API_BASE=https://... JUDGE_MODEL=gpt-5.3-codex
export EVAL_SET=/path/to/eval_set.jsonl
bash recipes/context_rot/distance/run_experiment.sh both "case_0_P1,case_2_P1" "0,50,100,200"
```

### 3. data_scan/ — 训练数据质量扫描

两阶段扫描训练数据中的偷懒模式（Phase 2 使用 nano-eval 的 `OnlineBatchInferenceEngine` 进行批量推理）：
- **Phase 1** (`scan_rules.py`): 规则关键词扫描，多进程并行
- **Phase 2** (`scan_judge.py`): LLM Judge 验证，复用 nano-eval 批量推理基建

```bash
# Phase 1（不需要 API）
bash recipes/context_rot/data_scan/run_scan.sh --input-list /path/to/filelist.txt

# Phase 1 + Phase 2
export JUDGE_API_KEY=sk-xxx JUDGE_API_BASE=https://... JUDGE_MODEL=gpt-5.3-codex
bash recipes/context_rot/data_scan/run_scan.sh --input-list /path/to/filelist.txt --phase2
```

## 数据准备

评测所需的 `context_rot_data.jsonl` 原始数据需要预先准备好。该文件包含完整的 Agent 对话轨迹，用于生成评测集。

## 依赖

- `openai` — 用于 Agent 循环和 Judge 的 API 调用（eval 模块）
- `nanoeval.backend.online` — 批量推理引擎（data_scan 模块 Phase 2 复用）
