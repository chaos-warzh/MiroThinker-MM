#!/bin/bash
export ENABLE_REPORT_VALIDATION="0"
RUN_BATCH="20251227"
uv run python run_batch_folder_tasks.py --data-dir datasets_batch2 --context-size 32k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir datasets_batch2 --context-size 64k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir datasets_batch2 --context-size 128k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
uv run python run_batch_folder_tasks.py --data-dir datasets_batch2 --context-size 256k --llm-config gpt-4 --run-batch $RUN_BATCH --offline
