#!/bin/bash
# 自动运行所有测试的脚本
# 使用方法: ./run_all_tests.sh
# 
# 注意: 此脚本会跳过已完成的任务 (使用 --skip-completed 参数)
# 如果需要重新运行所有任务，请删除 results 目录下的对应文件夹
#
# 配置文件说明:
#   - 使用 conf/llm/default.yaml 配置 (qwen2.5-coder-32b-instruct)
#   - 结果保存在 results/<日期>/results_<context_size>/<model>/
#   - 例如: results/20251210/results_64k/qwen2.5-coder-32b-instruct/

# 不使用 set -e，这样即使某个任务失败也会继续运行下一个

# 1. qwen2.5-coder-32b-instruct 64k
echo ""
echo "[1/2] 运行 qwen2.5-coder-32b-instruct 64k..."
echo "=========================================="
uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 64k --model qwen2.5-coder-32b-instruct --offline --skip-completed

# 2. qwen2.5-coder-32b-instruct 128k
echo ""
echo "[2/2] 运行 qwen2.5-coder-32b-instruct 128k..."
echo "=========================================="
uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 128k --model qwen2.5-coder-32b-instruct --offline --skip-completed 2>&1 | tee logs/qwen-coder-128k.log || echo "qwen-coder 128k 完成或出错"

echo ""
echo "=========================================="
echo "所有测试完成!"
echo "结束时间: $(date)"
echo "=========================================="
echo "结果保存在:"
echo "  - results/${TODAY}/results_64k/qwen2.5-coder-32b-instruct/"
echo "  - results/${TODAY}/results_128k/qwen2.5-coder-32b-instruct/"
echo ""
echo "日志保存在:"
echo "  - logs/qwen-coder-64k.log"
echo "  - logs/qwen-coder-128k.log"
