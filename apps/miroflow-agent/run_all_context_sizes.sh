#!/bin/bash
# 自动运行32k、64k、128k三种context size的批量任务

echo "=========================================="
echo "开始运行 qwen3-30b-a3b 模型批量任务"
echo "=========================================="

# 等待32k任务完成（检查是否有正在运行的进程）
echo ""
echo "检查32k任务是否正在运行..."

# 等待当前32k任务完成
while pgrep -f "run_batch_folder_tasks.py.*32k.*qwen3-30b-a3b" > /dev/null 2>&1; do
    echo "32k任务正在运行中，等待完成..."
    sleep 60
done

echo "32k任务已完成或未运行"
echo ""

# 运行64k任务
echo "=========================================="
echo "开始运行 64k 任务"
echo "=========================================="
uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 64k --model qwen3-30b-a3b --llm-config qwen3_30b

echo ""
echo "64k任务完成"
echo ""

# 运行128k任务
echo "=========================================="
echo "开始运行 128k 任务"
echo "=========================================="
uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 128k --model qwen3-30b-a3b --llm-config qwen3_30b

echo ""
echo "=========================================="
echo "所有任务完成！"
echo "=========================================="
echo ""
echo "结果保存位置："
echo "  - 32k: results/results_32k/qwen3-30b-a3b/"
echo "  - 64k: results/results_64k/qwen3-30b-a3b/"
echo "  - 128k: results/results_128k/qwen3-30b-a3b/"
