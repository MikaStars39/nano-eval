#!/bin/bash
# LLM Judge Scoring System - Example Run Script
# Usage: TRAINING_JOB_NAME=<job-name> ./run_dpo_llm_judge.sh

# Configuration paths
INPUT_FILE="/jpfs/qingyu/data/deploy-sft-128k-s2-0318-s2-fixswe-shuf-64-1e-5-min1e-6-outputs_tmp.jsonl"
OUTPUT_DIR="/jpfs/qingyu/nano-eval/output/dpo_data_0319"
JUDGE_MODEL="/jpfs/models/DeepSeek-V3.2"
SCRIPT_DIR="/jpfs/qingyu/nano-eval"

# Get training job name from environment variable or first argument
TRAINING_JOB_NAME="qingyu-64gpu-data"

if [ -z "$TRAINING_JOB_NAME" ]; then
    echo "Error: TRAINING_JOB_NAME is not set."
    echo "Usage: TRAINING_JOB_NAME=<job-name> ./run_dpo_llm_judge.sh"
    echo "   or: ./run_dpo_llm_judge.sh <job-name>"
    exit 1
fi

echo "Using TrainingJob: $TRAINING_JOB_NAME"

# 1. 动态获取 Pod 列表（按 index 排序）
echo "Fetching pods for training job: $TRAINING_JOB_NAME..."
mapfile -t pods < <(kubectl get pods | grep "$TRAINING_JOB_NAME" | awk '{print $1}' | sort -V)

if [ ${#pods[@]} -eq 0 ]; then
    echo "Error: No pods found for training job '$TRAINING_JOB_NAME'"
    exit 1
fi

echo "Found ${#pods[@]} pods:"
for pod in "${pods[@]}"; do
    echo "  - $pod"
done

# 2. 清理所有 Pod 上的旧进程
echo ""
echo "正在清理所有 Pod 上的 python 和 sglang 进程..."
for pod in "${pods[@]}"; do
    kubectl exec $pod -- bash -c "pkill -9 python; pkill -9 sglang; pkill -9 python3" 2>/dev/null &
done
wait
echo "清理完成"

# 3. 启动所有任务（并行，无延迟）
echo "启动所有任务..."
for i in "${!pods[@]}"; do
    pod_name=${pods[$i]}
    shard_id=$i

    echo "  分配任务: Pod=$pod_name, Shard=$shard_id"

    # 后台并行执行
    kubectl exec $pod_name -- bash -c "cd $SCRIPT_DIR && python recipes/llm_judge/inference.py \
        --input '$OUTPUT_DIR/shards_233_yuqi_new_data/shard_${shard_id}.jsonl' \
        --output '$OUTPUT_DIR/responses/response_${shard_id}.jsonl' \
        --model_path '$JUDGE_MODEL' \
        --tp_size 8 \
        --dp_size 8 \
        --enable_dp_attention \
        --max_tokens 32768" &> $OUTPUT_DIR/log_shard_${shard_id}.log &
done

echo ""
echo "✅ 所有任务已启动！使用以下命令查看 pod 日志："
echo "   kubectl logs <pod-name> -f"
