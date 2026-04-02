#!/bin/bash
set -euo pipefail

NAMESPACE="explore-train"
POD_PATTERN="deploy-ds32(-[234])?-"

BASE_DIR="/jpfs/qingyu/data/v32_two_stage/stage1"
SHARDS_DIR="$BASE_DIR/shards"
SHARDS_OUT_DIR="$BASE_DIR/shards_out"
LOG_DIR="/jpfs/qingyu/nano-eval/output/stage1_logs"

MODEL_PATH="/jpfs/models/DeepSeek-V3.2"
TP_SIZE=8
DP_SIZE=8
MAX_INFLIGHT=1024
TEMPERATURE=1
TOP_P=0.95
MAX_NEW_TOKENS=65536

mkdir -p "$LOG_DIR" "$SHARDS_OUT_DIR"

ALL_PODS=($(kubectl get pods -n $NAMESPACE --no-headers -o custom-columns=":metadata.name" | grep -E "$POD_PATTERN" | sort))
NUM_PODS=${#ALL_PODS[@]}

echo "Stage1 dispatch on $NUM_PODS pods..."

for ((i=0; i<$NUM_PODS; i++)); do
    POD=${ALL_PODS[$i]}
    SHARD_ID=$(printf "%05d" $i)
    LOG_FILE="$LOG_DIR/stage1_${POD}_shard_${SHARD_ID}.log"
    touch "$LOG_FILE"
    echo "[Pod $i/$NUM_PODS] $POD -> shard $SHARD_ID"

    (
        kubectl exec -n $NAMESPACE $POD -- bash -c "pkill -9 -f python || true; pkill -9 -f sglang || true; sleep 2" >/dev/null 2>&1 || true

        kubectl exec -n $NAMESPACE $POD -- bash -c "python /jpfs/qingyu/nano-eval/recipes/thinking_to_cot/stage1_run_shard.py \
          --shard-source-jsonl $SHARDS_DIR/shard_${SHARD_ID}.jsonl \
          --shard-prepared-jsonl $SHARDS_OUT_DIR/prepared_${SHARD_ID}.jsonl \
          --shard-model-output-jsonl $SHARDS_OUT_DIR/model_output_${SHARD_ID}.jsonl \
          --shard-final-jsonl $SHARDS_OUT_DIR/final_${SHARD_ID}.jsonl \
          --model-path $MODEL_PATH \
          --tp-size $TP_SIZE \
          --dp-size $DP_SIZE \
          --max-inflight $MAX_INFLIGHT \
          --enable-dp-attention \
          --temperature $TEMPERATURE \
          --top-p $TOP_P \
          --max-new-tokens $MAX_NEW_TOKENS \
          --apply-chat-template \
          --resume" > "$LOG_FILE" 2>&1
    ) &
done

wait
echo "Stage1 dispatch complete."
echo "Monitor: tail -f $LOG_DIR/stage1_*.log"

