#!/bin/bash
# 重跑 claude37_sonnet 缺失和空报告的 case
# 更新时间: 2025-12-24
# 
# 缺失/空报告列表 (共14个):
# - 32k: 002(空), 003(空), 004(缺), 006(缺), 010(缺)
# - 64k: 002(缺), 004(空), 006(空)
# - 128k: 003(空), 006(空), 009(空)
# - 256k: 001(空), 006(空), 009(空)

set -e

cd "$(dirname "$0")"

# 保存到原来的 run_batch
RUN_BATCH="20251222_214548"
DATASET="datasets_batch2"
MODEL="claude37_sonnet"
LLM_CONFIG="claude37_sonnet"

echo "=========================================="
echo "重跑 claude37_sonnet 缺失/空报告的 case"
echo "Run Batch: $RUN_BATCH"
echo "=========================================="

# 32k 缺失/空报告的 case: 002, 003, 004, 006, 010
echo ""
echo "=== 32k 缺失/空报告的 case (5个) ==="
for case in 002 003 004 006 010; do
    echo "Running 32k/$case..."
    uv run python run_batch_folder_tasks.py \
        --data-dir $DATASET \
        --context-size 32k \
        --llm-config $LLM_CONFIG \
        --model $MODEL \
        --run-batch $RUN_BATCH \
        --tasks $case \
        --offline
done

# 64k 缺失/空报告的 case: 002, 004, 006
echo ""
echo "=== 64k 缺失/空报告的 case (3个) ==="
for case in 002 004 006; do
    echo "Running 64k/$case..."
    uv run python run_batch_folder_tasks.py \
        --data-dir $DATASET \
        --context-size 64k \
        --llm-config $LLM_CONFIG \
        --model $MODEL \
        --run-batch $RUN_BATCH \
        --tasks $case \
        --offline
done

# 128k 空报告的 case: 003, 006, 009
echo ""
echo "=== 128k 空报告的 case (3个) ==="
for case in 003 006 009; do
    echo "Running 128k/$case..."
    uv run python run_batch_folder_tasks.py \
        --data-dir $DATASET \
        --context-size 128k \
        --llm-config $LLM_CONFIG \
        --model $MODEL \
        --run-batch $RUN_BATCH \
        --tasks $case \
        --offline
done

# 256k 空报告的 case: 001, 006, 009
echo ""
echo "=== 256k 空报告的 case (3个) ==="
for case in 001 006 009; do
    echo "Running 256k/$case..."
    uv run python run_batch_folder_tasks.py \
        --data-dir $DATASET \
        --context-size 256k \
        --llm-config $LLM_CONFIG \
        --model $MODEL \
        --run-batch $RUN_BATCH \
        --tasks $case \
        --offline
done

echo ""
echo "=========================================="
echo "所有缺失/空报告的 case 已重跑完成!"
echo "=========================================="
echo ""
echo "重跑统计:"
echo "  32k:  5个 (002, 003, 004, 006, 010)"
echo "  64k:  3个 (002, 004, 006)"
echo "  128k: 3个 (003, 006, 009)"
echo "  256k: 3个 (001, 006, 009)"
echo "  总计: 14个 case"
echo ""
echo "结果保存到: result/$RUN_BATCH/datasets_batch2/"
