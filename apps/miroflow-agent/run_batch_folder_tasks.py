#!/usr/bin/env python3
# Copyright 2025 Miromind.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Batch Folder Task Runner

This script processes multiple folders in batch by reading task definitions
from a JSONL file and running each task using run_folder_task.py.

Usage:
    # Run all tasks from query.jsonl
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104
    
    # Run specific tasks by number
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --tasks 001 002 003
    
    # Skip already completed tasks
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --skip-completed
    
    # Preview tasks without running
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --preview

"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_folder_task import run_folder_task_simple, save_report_to_md, get_results_dir_for_folder


def load_tasks_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load tasks from a JSONL file.
    
    Args:
        jsonl_path: Path to the query.jsonl file
        
    Returns:
        List of task dictionaries with 'number' and 'query' keys
    """
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                if 'number' in task and 'query' in task:
                    tasks.append(task)
                else:
                    print(f"Warning: Line {line_num} missing 'number' or 'query' field, skipping")
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}, skipping")
    return tasks


def get_completed_tasks(results_dir: str) -> set:
    """
    Get set of task numbers that have already been completed.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Set of completed task numbers (e.g., {'001', '002'})
    """
    completed = set()
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.md'):
                task_num = filename.replace('.md', '')
                completed.add(task_num)
    return completed


async def run_single_task(
    data_dir: str,
    task: Dict,
    results_dir: str,
    config_overrides: List[str] = None
) -> bool:
    """
    Run a single task.
    
    Args:
        data_dir: Base data directory
        task: Task dictionary with 'number' and 'query'
        results_dir: Directory to save results
        config_overrides: Optional Hydra config overrides
        
    Returns:
        True if successful, False otherwise
    """
    task_number = task['number']
    query = task['query']
    folder_path = os.path.join(data_dir, task_number)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}, skipping task {task_number}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running Task {task_number}")
    print(f"Folder: {folder_path}")
    print(f"Query: {query[:100]}...")
    print(f"{'='*60}\n")
    
    try:
        start_time = time.time()
        
        final_summary, final_boxed_answer, log_file_path = await run_folder_task_simple(
            folder_path=folder_path,
            query=query,
            config_overrides=config_overrides
        )
        
        elapsed_time = time.time() - start_time
        
        # Save result using the shared function from run_folder_task
        report_path = save_report_to_md(
            results_dir=results_dir,
            folder_name=task_number,
            final_answer=final_boxed_answer,
            summary=final_summary,
            query=query,
        )
        print(f"Report saved to: {report_path}")
        
        print(f"\n✓ Task {task_number} completed in {elapsed_time:.1f}s")
        print(f"  Log file: {log_file_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Task {task_number} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_batch_tasks(
    data_dir: str,
    tasks: List[Dict],
    results_dir: str,
    skip_completed: bool = False,
    config_overrides: List[str] = None
) -> Dict:
    """
    Run multiple tasks in sequence.
    
    Args:
        data_dir: Base data directory
        tasks: List of task dictionaries
        results_dir: Directory to save results
        skip_completed: Whether to skip already completed tasks
        config_overrides: Optional Hydra config overrides
        
    Returns:
        Dictionary with statistics
    """
    completed_tasks = get_completed_tasks(results_dir) if skip_completed else set()
    
    stats = {
        'total': len(tasks),
        'skipped': 0,
        'success': 0,
        'failed': 0,
        'failed_tasks': []
    }
    
    for i, task in enumerate(tasks, 1):
        task_number = task['number']
        
        if task_number in completed_tasks:
            print(f"Skipping task {task_number} (already completed)")
            stats['skipped'] += 1
            continue
        
        print(f"\n[{i}/{len(tasks)}] Processing task {task_number}...")
        
        success = await run_single_task(
            data_dir=data_dir,
            task=task,
            results_dir=results_dir,
            config_overrides=config_overrides
        )
        
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
            stats['failed_tasks'].append(task_number)
    
    return stats


def preview_tasks(data_dir: str, tasks: List[Dict], results_dir: str):
    """
    Preview tasks without running them.
    
    Args:
        data_dir: Base data directory
        tasks: List of task dictionaries
        results_dir: Directory for results
    """
    completed = get_completed_tasks(results_dir)
    
    print("\n" + "=" * 60)
    print("BATCH TASK PREVIEW")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending: {len(tasks) - len([t for t in tasks if t['number'] in completed])}")
    
    print("\n" + "-" * 60)
    print("Tasks:")
    print("-" * 60)
    
    for task in tasks:
        task_number = task['number']
        folder_path = os.path.join(data_dir, task_number)
        folder_exists = os.path.exists(folder_path)
        is_completed = task_number in completed
        
        status = "✓ completed" if is_completed else ("✗ folder missing" if not folder_exists else "○ pending")
        
        print(f"\n[{task_number}] {status}")
        print(f"  Folder: {folder_path}")
        print(f"  Query: {task['query'][:80]}...")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch folder tasks from a JSONL file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tasks
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104
    
    # Run specific tasks
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --tasks 001 002
    
    # Skip completed tasks
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --skip-completed
    
    # Preview without running
    uv run python run_batch_folder_tasks.py --data-dir data/bench_case1104 --preview
        """
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Path to the data directory containing task folders and query.jsonl"
    )
    parser.add_argument(
        "--query-file", "-q",
        default="query.jsonl",
        help="Name of the JSONL file containing queries (default: query.jsonl)"
    )
    parser.add_argument(
        "--results-dir", "-r",
        default=None,
        help="Directory to save results (default: results/<data-dir-name>)"
    )
    parser.add_argument(
        "--tasks", "-t",
        nargs="+",
        default=None,
        help="Specific task numbers to run (e.g., 001 002 003)"
    )
    parser.add_argument(
        "--skip-completed", "-s",
        action="store_true",
        help="Skip tasks that have already been completed"
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Preview tasks without running them"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = args.data_dir
    query_file = os.path.join(data_dir, args.query_file)
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        data_dir_name = os.path.basename(os.path.abspath(data_dir))
        results_dir = os.path.join("results", data_dir_name)
    
    # Check query file exists
    if not os.path.exists(query_file):
        print(f"Error: Query file not found: {query_file}")
        sys.exit(1)
    
    # Load tasks
    tasks = load_tasks_from_jsonl(query_file)
    print(f"Loaded {len(tasks)} tasks from {query_file}")
    
    # Filter tasks if specific ones requested
    if args.tasks:
        tasks = [t for t in tasks if t['number'] in args.tasks]
        print(f"Filtered to {len(tasks)} tasks: {args.tasks}")
    
    if not tasks:
        print("No tasks to process")
        sys.exit(0)
    
    # Preview or run
    if args.preview:
        preview_tasks(data_dir, tasks, results_dir)
    else:
        print(f"\nStarting batch processing...")
        print(f"Results will be saved to: {results_dir}")
        
        start_time = time.time()
        
        stats = asyncio.run(
            run_batch_tasks(
                data_dir=data_dir,
                tasks=tasks,
                results_dir=results_dir,
                skip_completed=args.skip_completed
            )
        )
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"\nTotal time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
        print(f"Total tasks: {stats['total']}")
        print(f"Successful: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        print(f"Skipped: {stats['skipped']}")
        
        if stats['failed_tasks']:
            print(f"\nFailed tasks: {', '.join(stats['failed_tasks'])}")
        
        print(f"\nResults saved to: {results_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
