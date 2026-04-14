# Recipes

实验脚本和任务专用代码。与核心模块 `nanoeval/` 的区别：recipes 只服务特定实验，不被其他模块 import。

## 目录索引

| 目录 | 用途 | 入口 |
|------|------|------|
| `eval/examples/` | 标准评测 shell 脚本模板（新脚本参照此风格） | `thinking_online.sh`, `thinking_offline.sh` |
| `eval/` | 具体评测任务的 shell 脚本 | 各 `.sh` 文件 |
| `context_rot/eval/` | Context Rot 多轮 agent 评测 + LLM judge | `run_eval.sh` → `run_eval.py` |
| `context_rot/distance/` | SP/Query 距离敏感性实验 | `run_experiment.sh` → `make_eval.py` |
| `context_rot/data_scan/` | 训练数据偷懒模式扫描 | `run_scan.sh` → `scan_rules.py`; `run_judge.sh` → `scan_judge.py` |
| `llm_judge/` | DPO 数据 LLM 打分流水线 | `run_dpo_llm_judge.sh` |
| `profiling/` | 推理性能 profiling | `run.sh` → `run.py` |
| `utils/` | Ray 集群启动等通用工具 | `ray_head_start.sh` |

## Shell 脚本风格

参照 `eval/examples/` — 无 banner echo、无 arg parsing、env vars + 数组 + 单次 python 调用。
