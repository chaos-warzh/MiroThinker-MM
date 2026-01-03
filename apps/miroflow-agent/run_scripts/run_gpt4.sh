#!/bin/bash
# Run batch folder tasks for GPT-4
# All results will be saved to: result/<run_batch>/<dataset>/<context_size>/<model>/
export ENABLE_REPORT_VALIDATION="0"
RUN_BATCH="20251229-real-noise"

# gpt-4
uv run python run_batch_folder_tasks.py --data-dir datasets_batch2_1228 --context-size 32k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir datasets_batch2_1228 --context-size 64k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir datasets_batch2_1228 --context-size 128k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir datasets_batch2_1228 --context-size 256k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir datasets_batch2_1228 --context-size 512k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
