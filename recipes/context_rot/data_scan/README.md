# Data Scan — 训练数据偷懒模式扫描

扫描训练数据中的"偷懒"模式（模型在长对话后期降低输出质量），通过规则过滤 + LLM Judge 两阶段筛选问题数据。

## Pipeline

```
scan_rules.py          规则扫描，按关键词过滤
     ↓
map_jsonl.py           回填原始对话内容 (source_file → raw_data)
     ↓
scan_judge.py          LLM Judge 逐条评分 (通过 nano-eval Ray actor)
     ↓                      ── 或 ──
prepare_vulcan.py      转为 Vulcan 批量推理格式 (分片 JSONL)
     ↓ (Vulcan 推理)
merge_judge_output.py  合并推理结果，解析评分，match 回原始数据
```

有两条评分路径：`scan_judge.py` 直接调用 API 评分；或者 `prepare_vulcan.py` → Vulcan 批量推理 → `merge_judge_output.py` 解析合并。

## 文件说明

每个 `.py` 都有对应的同名 `.sh` 作为运行入口。

### scan_rules.py / scan_rules.sh

Phase 1：规则扫描。扫描训练数据 JSONL，按 token 长度阈值过滤 + 关键词匹配，输出命中的样本。

```bash
INPUT_LIST=/path/to/file_list.txt MIN_TOKENS=20000 bash scan_rules.sh
```

- 输入：`INPUT_LIST` — 文本文件，每行一个训练数据 JSONL 路径
- 输出：`flagged.jsonl` — 命中规则的样本（含 source_file、line_number、keyword_hits，不含原始对话）

### map_jsonl.py / map_jsonl.sh

从 `flagged.jsonl` 的 `source_file` + `line_number` 回到原始训练数据文件，读取完整对话内容填入 `raw_data` 字段。使用 mmap 高效随机读取。

```bash
SCAN_DIR=/path/to/scan_results/20260410-082039 bash map_jsonl.sh
```

- 输入：`flagged.jsonl`（scan_rules 输出）
- 输出：`full_flagged.jsonl`（补全了 raw_data 的完整数据）

### scan_judge.py / scan_judge.sh

Phase 2（路径 A）：通过 nano-eval 的 `OnlineInferenceActor` (Ray) 直接调用 LLM API 逐条评分。适合数据量较小或需要实时结果的场景。

```bash
SCAN_INPUT=/path/to/flagged.jsonl \
JUDGE_MODEL=gpt-5.4-thinking-xhigh \
JUDGE_API_BASE=https://... \
JUDGE_API_KEY=sk-... \
bash scan_judge.sh
```

- 输入：`flagged.jsonl` 或 `full_flagged.jsonl`
- 输出：`judged.jsonl`（含 score、recommendation、justification）

### prepare_vulcan.py / prepare_vulcan.sh

Phase 2（路径 B-1）：将 `full_flagged.jsonl` 转为 Vulcan 批量推理格式，按 `--shard-size` 分片输出。每条数据构造 judge prompt 后写入 Gemini `contents` 格式。

```bash
INPUT=/path/to/full_flagged.jsonl \
OUTPUT_DIR=/path/to/vulcan_batch \
SHARD_SIZE=5000 \
bash prepare_vulcan.sh
```

- 输入：`full_flagged.jsonl`
- 输出：`OUTPUT_DIR/full_flagged_000.jsonl`, `full_flagged_001.jsonl`, ... + `_files_info`

### merge_judge_output.py / merge_judge_output.sh

Phase 2（路径 B-2）：Vulcan 推理完成后，将结果 match 回原始数据。按 `global_index`（= `full_flagged.jsonl` 行号）精确对应，解析 judge 输出 JSON，提取 score/recommendation/justification。

```bash
bash merge_judge_output.sh
```

- 输入：Vulcan 输出分片 + 原始 `full_flagged.jsonl`
- 输出：`merged_result.jsonl`，每行包含：
  - `source_file` / `line_number` — 原始训练数据定位
  - `score` (0.0 / 0.5 / 1.0) / `recommendation` (keep / flag / remove) / `justification`
  - `status` — ok / parse_error / max_tokens / api_error / bad_finish
  - `original_meta` — 原始元数据（keyword_hits 等）

## 输出字段约定

Judge 评分三档：

| score | recommendation | 含义 |
|-------|---------------|------|
| 0.0   | keep          | 无偷懒问题 |
| 0.5   | flag          | 轻微偷懒倾向，需人工审核 |
| 1.0   | remove        | 严重偷懒，应移除 |
