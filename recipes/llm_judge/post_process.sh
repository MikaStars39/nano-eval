INPUT_FILE="/jpfs/qingyu/data/0309-distill-amthinking.jsonl"
OUTPUT_DIR="/jpfs/qingyu/nano-eval/output/dpo_data_0310"
JUDGE_MODEL="/jpfs/models/DeepSeek-V3.2"
SCRIPT_DIR="/jpfs/qingyu/nano-eval"

python $SCRIPT_DIR/recipes/llm_judge/merge_and_extract.py \
    --response-dir $OUTPUT_DIR/responses \
    --output $OUTPUT_DIR/scores.jsonl \
    --failed $OUTPUT_DIR/failed.jsonl \
    --workers 128

python $SCRIPT_DIR/recipes/llm_judge/analyze_best_worst.py \
    --input $OUTPUT_DIR/scores.jsonl \
    --output $OUTPUT_DIR/final.jsonl

python $SCRIPT_DIR/recipes/llm_judge/analyze_scores.py \
    --scores $OUTPUT_DIR/final.jsonl