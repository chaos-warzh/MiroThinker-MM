#!/bin/bash
# Run batch folder tasks for datasets_batch2 with all context sizes and models
# All results will be saved to: result/<run_batch>/<dataset>/<context_size>/<model>/

# ============================================================
# Configuration
# ============================================================
export ENABLE_REPORT_VALIDATION="0"
RUN_BATCH="20251227-new"
DATA_DIR="datasets_batch2"

# ============================================================
# Run all models with different context sizes
# ============================================================

# qwen3_235b
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 32k --llm-config qwen3_235b --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 64k --llm-config qwen3_235b --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 128k --llm-config qwen3_235b --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 256k --llm-config qwen3_235b --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 512k --llm-config qwen3_235b --run-batch $RUN_BATCH --offline

# qwen3_30b
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 32k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 64k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 128k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 256k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 512k --llm-config qwen3_30b --run-batch $RUN_BATCH --offline

# claude35_sonnet
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 32k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 64k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 128k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 256k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 512k --llm-config claude35_sonnet --run-batch $RUN_BATCH --offline

# # claude37_sonnet
# uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 32k --llm-config claude37_sonnet --run-batch $RUN_BATCH --offline
# uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 64k --llm-config claude37_sonnet --run-batch $RUN_BATCH --offline
# uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 128k --llm-config claude37_sonnet --run-batch $RUN_BATCH --offline
# uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 256k --llm-config claude37_sonnet --run-batch $RUN_BATCH --offline
# uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 512k --llm-config claude37_sonnet --run-batch $RUN_BATCH --offline

# gpt-4
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 32k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 64k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 128k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 256k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir $DATA_DIR --context-size 512k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
