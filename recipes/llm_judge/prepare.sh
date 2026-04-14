#!/bin/bash
# LLM Judge Scoring System - Example Run Script
#
# Required env vars:
#   INPUT_FILE    — path to input JSONL (model outputs)
#   OUTPUT_DIR    — directory for prepared outputs
#   JUDGE_MODEL   — path to judge model / tokenizer
#
# Example:
#   INPUT_FILE=/path/to/output.jsonl OUTPUT_DIR=/path/to/out JUDGE_MODEL=/path/to/model bash prepare.sh

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT_FILE="${INPUT_FILE:?Set INPUT_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
JUDGE_MODEL="${JUDGE_MODEL:?Set JUDGE_MODEL}"

python "${REPO_ROOT}/recipes/llm_judge/prepare_judge.py" \
    --input "${INPUT_FILE}" \
    --output "${OUTPUT_DIR}/prepare.jsonl" \
    --tokenizer "${JUDGE_MODEL}" \
    --workers 128 \
    --n_examples 4

python "${REPO_ROOT}/recipes/llm_judge/shard_jsonl.py" \
    --input "${OUTPUT_DIR}/prepare.jsonl" \
    --output-dir "${OUTPUT_DIR}/shards" \
    --num-shards 16