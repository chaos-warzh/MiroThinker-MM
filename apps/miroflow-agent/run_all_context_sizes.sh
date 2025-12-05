#!/bin/bash

echo "=========================================="
echo "Starting batch runs for claude37_sonnet"
echo "=========================================="

# Run 32k
echo ""
echo "[$(date)] Starting 32k batch..."
uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 32k --model claude37_sonnet --offline --skip-completed
echo "[$(date)] 32k batch completed!"

# Run 64k
echo ""
echo "[$(date)] Starting 64k batch..."
uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 64k --model claude37_sonnet --offline --skip-completed
echo "[$(date)] 64k batch completed!"

# Run 128k
echo ""
echo "[$(date)] Starting 128k batch..."
uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 128k --model claude37_sonnet --offline --skip-completed
echo "[$(date)] 128k batch completed!"

echo ""
echo "=========================================="
echo "All batches completed!"
echo "=========================================="
