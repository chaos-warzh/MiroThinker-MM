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
    
    # Run with specific context size (32k, 64k, 128k)
    uv run python run_batch_folder_tasks.py --data-dir datasets --context-size 32k --model gpt4.1
    
    # Results will be saved to results/datasets_32k_gpt4.1/<task_number>_run_<timestamp>/

Output Structure:
    For each task run, a unique folder is created containing:
    - initial_report.md: The original report before validation
    - final_report.md: The final report after validation
    - tool_call_summary.json: Tool usage summary for each turn
    - tool_call_summary.md: Human-readable tool usage summary
"""

import argparse
import asyncio
import glob
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_folder_task import run_folder_task_simple, get_results_dir_for_folder


def prepare_folder_for_context_size(folder_path: str, context_size: str) -> bool:
    """
    Prepare a folder to use a specific context size by temporarily renaming db files.
    """
    db_files = glob.glob(os.path.join(folder_path, "*.chunks.db"))
    
    if not db_files:
        print(f"  Warning: No .chunks.db files found in {folder_path}")
        return False
    
    target_pattern = f"long_context_sampled_{context_size}.json.chunks.db"
    target_db = None
    
    for db_file in db_files:
        if os.path.basename(db_file) == target_pattern:
            target_db = db_file
            break
    
    if not target_db:
        print(f"  Warning: No db file found for context size {context_size} in {folder_path}")
        print(f"  Available db files: {[os.path.basename(f) for f in db_files]}")
        return False
    
    for db_file in db_files:
        if db_file != target_db:
            hidden_path = db_file + ".hidden"
            if not os.path.exists(hidden_path):
                os.rename(db_file, hidden_path)
    
    return True


def restore_folder_db_files(folder_path: str) -> None:
    """Restore all hidden db files in a folder."""
    hidden_files = glob.glob(os.path.join(folder_path, "*.chunks.db.hidden"))
    
    for hidden_file in hidden_files:
        original_path = hidden_file[:-7]
        if not os.path.exists(original_path):
            os.rename(hidden_file, original_path)


def load_tasks_from_jsonl(jsonl_path: str) -> List[Dict]:
    """Load tasks from a JSONL file."""
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                if 'task' in task and 'number' not in task:
                    task['number'] = task['task']
                if 'number' in task and 'query' in task:
                    tasks.append(task)
                else:
                    print(f"Warning: Line {line_num} missing 'number'/'task' or 'query' field, skipping")
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}, skipping")
    return tasks


def get_completed_tasks(results_dir: str) -> set:
    """Get set of task numbers that have already been completed."""
    completed = set()
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and '_run_' in item:
                task_num = item.split('_run_')[0]
                completed.add(task_num)
    return completed


def create_run_folder(results_dir: str, task_number: str, context_size: str = None, model: str = None) -> str:
    """Create a unique folder for this task run with timestamp.
    
    Directory structure: results/results_<context_size>/<model>/<task_number>_<timestamp>/
    Example: results/results_32k/gpt4.1/001_20251204_150425/
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Build folder path with model as subdirectory
    if model:
        model_dir = os.path.join(results_dir, model)
    else:
        model_dir = results_dir
    
    # Create folder name with task number and timestamp
    folder_name = f"{task_number}_{timestamp}"
    run_folder = os.path.join(model_dir, folder_name)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def parse_tool_calls_from_log(log_file_path: str) -> Dict:
    """Parse tool call information from the JSON log file."""
    if not os.path.exists(log_file_path):
        return {"error": f"Log file not found: {log_file_path}"}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except Exception as e:
        return {"error": f"Failed to parse log file: {e}"}
    
    tool_calls_by_turn = {}
    current_turn = 0
    
    step_logs = log_data.get('step_logs', [])
    
    for step in step_logs:
        step_name = step.get('step_name', '')
        message = step.get('message', '')
        metadata = step.get('metadata', {})
        
        # Detect turn changes
        if "Turn:" in step_name:
            turn_match = re.search(r'Turn:\s*(\d+)', step_name)
            if turn_match:
                current_turn = int(turn_match.group(1))
                if current_turn not in tool_calls_by_turn:
                    tool_calls_by_turn[current_turn] = {
                        "tools_used": [],
                        "rag_queries": []
                    }
        
        # Detect tool calls
        if "Tool Call Start" in step_name or "Tool Call Success" in step_name:
            tool_info = {}
            
            if "tool-" in step_name.lower():
                tool_match = re.search(r'tool-(\w+)', step_name.lower())
                if tool_match:
                    tool_info["tool_name"] = tool_match.group(1)
            
            if "tool_name" not in tool_info:
                if "rag_search" in message.lower() or "rag_get_context" in message.lower():
                    tool_info["tool_name"] = "rag"
                elif "google" in message.lower() or "search" in message.lower():
                    tool_info["tool_name"] = "google_search"
                elif "browser" in message.lower() or "playwright" in message.lower():
                    tool_info["tool_name"] = "browser"
                elif "python" in message.lower():
                    tool_info["tool_name"] = "python"
                else:
                    if metadata.get('tool_name'):
                        tool_info["tool_name"] = metadata.get('tool_name')
                    elif metadata.get('server_name'):
                        tool_info["tool_name"] = metadata.get('server_name')
            
            if tool_info.get("tool_name") and current_turn in tool_calls_by_turn:
                existing_tools = [t.get("tool_name") for t in tool_calls_by_turn[current_turn]["tools_used"]]
                if tool_info["tool_name"] not in existing_tools or "Tool Call Start" in step_name:
                    tool_calls_by_turn[current_turn]["tools_used"].append(tool_info)
    
    return {
        "task_id": log_data.get('task_id', 'unknown'),
        "status": log_data.get('status', 'unknown'),
        "start_time": log_data.get('start_time', ''),
        "end_time": log_data.get('end_time', ''),
        "total_turns": len(tool_calls_by_turn),
        "turns": tool_calls_by_turn
    }


def parse_rag_from_execution_log(log_file_path: str) -> Dict:
    """
    Parse RAG retrieval information directly from the execution log.
    
    This extracts RAG tool calls and their results from the main execution log,
    since RAG tools don't save separate log files.
    """
    if not os.path.exists(log_file_path):
        return {"message": "Execution log not found"}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except Exception as e:
        return {"message": f"Failed to parse execution log: {e}"}
    
    rag_summary = {
        "total_queries": 0,
        "queries": []
    }
    
    step_logs = log_data.get('step_logs', [])
    
    # Track RAG tool calls and their results
    current_rag_call = None
    
    for step in step_logs:
        step_name = step.get('step_name', '')
        message = step.get('message', '')
        metadata = step.get('metadata', {})
        
        # Detect RAG tool call start
        # Note: "tool-rag" is in the message field, not in step_name
        if "Tool Call Start" in step_name and "tool-rag" in message.lower():
            # Extract query from metadata or message
            tool_args = metadata.get('arguments', {})
            query = tool_args.get('query', '')
            # Extract tool name from message (e.g., "to call tool 'rag_search'")
            tool_name_match = re.search(r"to call tool '(\w+)'", message)
            tool_name = tool_name_match.group(1) if tool_name_match else metadata.get('tool_name', 'rag_search')
            
            current_rag_call = {
                "query": query,
                "tool": tool_name,
                "num_results": 0,
                "retrieved_documents": []
            }
        
        # Detect RAG tool call success with results
        # Note: "tool-rag" is in the message field, not in step_name
        elif "Tool Call Success" in step_name and "tool-rag" in message.lower():
            if current_rag_call:
                # Try to extract results from the tool response
                tool_result = metadata.get('result', '')
                
                # Parse the result to extract document titles
                if isinstance(tool_result, str):
                    # Try to extract document titles from the result text
                    docs = extract_doc_titles_from_result(tool_result)
                    current_rag_call["retrieved_documents"] = docs
                    current_rag_call["num_results"] = len(docs)
                elif isinstance(tool_result, list):
                    for item in tool_result:
                        if isinstance(item, dict):
                            doc_info = {
                                "title": item.get('title', 'Untitled'),
                                "score": item.get('score', 0),
                                "doc_index": item.get('doc_index', 0),
                                "chunk_index": item.get('chunk_index', 0)
                            }
                            current_rag_call["retrieved_documents"].append(doc_info)
                    current_rag_call["num_results"] = len(current_rag_call["retrieved_documents"])
                
                rag_summary["queries"].append(current_rag_call)
                rag_summary["total_queries"] += 1
                current_rag_call = None
        
        # Also check for RAG-related log messages that might contain query info
        elif "rag_search" in message.lower() or "rag_get_context" in message.lower():
            # Try to extract query from the message
            query_match = re.search(r'query["\s:]+([^"]+)"', message)
            if query_match and not current_rag_call:
                current_rag_call = {
                    "query": query_match.group(1),
                    "tool": "rag_search",
                    "num_results": 0,
                    "retrieved_documents": []
                }
        
        # Check for diverse search results in log messages
        elif "Diverse search returned" in message:
            match = re.search(r'(\d+) results from (\d+) documents', message)
            if match and current_rag_call:
                current_rag_call["num_results"] = int(match.group(1))
    
    return rag_summary


def extract_doc_titles_from_result(result_text: str) -> List[Dict]:
    """Extract document titles from RAG result text."""
    docs = []
    
    # Pattern 1: Look for "Title: xxx" or "title: xxx" patterns
    title_matches = re.findall(r'[Tt]itle[:\s]+([^\n]+)', result_text)
    for title in title_matches:
        title = title.strip().strip('"\'')
        if title and len(title) > 3:
            docs.append({"title": title, "score": 0})
    
    # Pattern 2: Look for "## Title" markdown headers
    header_matches = re.findall(r'##\s+([^\n]+)', result_text)
    for title in header_matches:
        title = title.strip()
        if title and len(title) > 3 and title not in [d["title"] for d in docs]:
            docs.append({"title": title, "score": 0})
    
    # Pattern 3: Look for "Document X:" patterns
    doc_matches = re.findall(r'Document\s+\d+[:\s]+([^\n]+)', result_text)
    for title in doc_matches:
        title = title.strip().strip('"\'')
        if title and len(title) > 3 and title not in [d["title"] for d in docs]:
            docs.append({"title": title, "score": 0})
    
    # Pattern 4: Look for score patterns like "(score: 0.85)"
    score_matches = re.findall(r'([^\n]+)\s*\(score:\s*([\d.]+)\)', result_text)
    for title, score in score_matches:
        title = title.strip().strip('"\'')
        if title and len(title) > 3:
            # Update existing or add new
            found = False
            for doc in docs:
                if doc["title"] == title:
                    doc["score"] = float(score)
                    found = True
                    break
            if not found:
                docs.append({"title": title, "score": float(score)})
    
    return docs


def save_tool_call_summary(run_folder: str, log_file_path: str, task_id: str, log_dir: str) -> tuple:
    """
    Save tool call summary to the run folder.
    
    Returns:
        Tuple of (json_path, md_path)
    """
    # Parse tool calls from main log
    tool_calls = parse_tool_calls_from_log(log_file_path)
    
    # Parse RAG retrieval information from execution log
    rag_summary = parse_rag_from_execution_log(log_file_path)
    
    # Combine into comprehensive summary
    summary = {
        "task_id": task_id,
        "execution_info": {
            "status": tool_calls.get("status", "unknown"),
            "start_time": tool_calls.get("start_time", ""),
            "end_time": tool_calls.get("end_time", ""),
            "total_turns": tool_calls.get("total_turns", 0)
        },
        "tool_calls_by_turn": tool_calls.get("turns", {}),
        "rag_retrieval": rag_summary
    }
    
    # Save JSON summary
    json_path = os.path.join(run_folder, "tool_call_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Generate human-readable markdown summary
    md_path = os.path.join(run_folder, "tool_call_summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Tool Call Summary\n\n")
        f.write(f"**Task ID:** {task_id}\n\n")
        f.write(f"## Execution Info\n\n")
        f.write(f"- **Status:** {summary['execution_info']['status']}\n")
        f.write(f"- **Start Time:** {summary['execution_info']['start_time']}\n")
        f.write(f"- **End Time:** {summary['execution_info']['end_time']}\n")
        f.write(f"- **Total Turns:** {summary['execution_info']['total_turns']}\n\n")
        
        f.write(f"## Tool Calls by Turn\n\n")
        turns = summary.get("tool_calls_by_turn", {})
        for turn_num in sorted(turns.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
            turn_data = turns[turn_num]
            f.write(f"### Turn {turn_num}\n\n")
            
            tools = turn_data.get("tools_used", [])
            if tools:
                f.write("**Tools Used:**\n")
                for tool in tools:
                    f.write(f"- {tool.get('tool_name', 'unknown')}\n")
            else:
                f.write("*No tools used in this turn*\n")
            f.write("\n")
        
        f.write(f"## RAG Retrieval Summary\n\n")
        rag = summary.get("rag_retrieval", {})
        if rag.get("message"):
            f.write(f"*{rag['message']}*\n\n")
        else:
            f.write(f"**Total RAG Queries:** {rag.get('total_queries', 0)}\n\n")
            
            for i, query in enumerate(rag.get("queries", []), 1):
                f.write(f"### RAG Query {i}\n\n")
                f.write(f"**Query:** {query.get('query', '')}\n\n")
                f.write(f"**Tool:** {query.get('tool', '')}\n\n")
                f.write(f"**Results:** {query.get('num_results', 0)}\n\n")
                
                docs = query.get("retrieved_documents", [])
                if docs:
                    f.write("**Retrieved Documents:**\n\n")
                    for doc in docs:
                        title = doc.get('title', 'Untitled')
                        url = doc.get('url', '')
                        score = doc.get('score', 0)
                        f.write(f"- **{title}**\n")
                        if url:
                            f.write(f"  - URL: {url}\n")
                        f.write(f"  - Score: {score:.4f}\n")
                f.write("\n")
    
    return json_path, md_path


def save_initial_report(run_folder: str, original_report: str, query: str) -> str:
    """Save the initial report (before validation) to the run folder."""
    report_path = os.path.join(run_folder, "initial_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Initial Report (Before Validation)\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if query:
            f.write("## Query\n\n")
            f.write(f"{query}\n\n")
        f.write("## Report\n\n")
        f.write(original_report if original_report else "No report generated.")
    return report_path


def save_final_report(run_folder: str, final_report: str, query: str) -> str:
    """Save the final report (after validation) to the run folder."""
    report_path = os.path.join(run_folder, "final_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Final Report (After Validation)\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if query:
            f.write("## Query\n\n")
            f.write(f"{query}\n\n")
        f.write("## Report\n\n")
        f.write(final_report if final_report else "No report generated.")
    return report_path


async def run_single_task(
    data_dir: str,
    task: Dict,
    results_dir: str,
    config_overrides: List[str] = None,
    context_size: str = None,
    model: str = None
) -> bool:
    """Run a single task and save results to a unique run folder."""
    task_number = task['number']
    query = task['query']
    folder_path = os.path.join(data_dir, task_number)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}, skipping task {task_number}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running Task {task_number}")
    print(f"Folder: {folder_path}")
    if context_size:
        print(f"Context Size: {context_size}")
    print(f"Query: {query[:100]}...")
    print(f"{'='*60}\n")
    
    # Prepare folder for specific context size if specified
    if context_size:
        if not prepare_folder_for_context_size(folder_path, context_size):
            print(f"Warning: Could not prepare folder for context size {context_size}, skipping task {task_number}")
            return False
    
    try:
        start_time = time.time()
        
        # Create unique run folder for this task execution
        run_folder = create_run_folder(results_dir, task_number, context_size, model)
        print(f"Results will be saved to: {run_folder}")
        
        result = await run_folder_task_simple(
            folder_path=folder_path,
            query=query,
            config_overrides=config_overrides
        )
        
        # Handle both old (3-tuple) and new (4-tuple) return formats
        if len(result) == 4:
            final_summary, final_boxed_answer, original_boxed_answer, log_file_path = result
        else:
            final_summary, final_boxed_answer, log_file_path = result
            original_boxed_answer = final_boxed_answer
        
        elapsed_time = time.time() - start_time
        
        # Save initial report (before validation)
        initial_report_path = save_initial_report(run_folder, original_boxed_answer, query)
        print(f"Initial report saved to: {initial_report_path}")
        
        # Save final report (after validation)
        final_report_path = save_final_report(run_folder, final_boxed_answer, query)
        print(f"Final report saved to: {final_report_path}")
        
        # Save tool call summary
        task_id = f"folder_task_{task_number}"
        log_dir = os.path.dirname(log_file_path) if log_file_path else "logs"
        json_path, md_path = save_tool_call_summary(run_folder, log_file_path, task_id, log_dir)
        print(f"Tool call summary saved to: {json_path}")
        print(f"Tool call summary (MD) saved to: {md_path}")
        
        # Copy the original log file to the run folder
        if log_file_path and os.path.exists(log_file_path):
            log_copy_path = os.path.join(run_folder, "execution_log.json")
            shutil.copy2(log_file_path, log_copy_path)
            print(f"Execution log copied to: {log_copy_path}")
        
        print(f"\n✓ Task {task_number} completed in {elapsed_time:.1f}s")
        print(f"  Run folder: {run_folder}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Task {task_number} failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore hidden db files
        if context_size:
            restore_folder_db_files(folder_path)


async def run_batch_tasks(
    data_dir: str,
    tasks: List[Dict],
    results_dir: str,
    skip_completed: bool = False,
    config_overrides: List[str] = None,
    context_size: str = None,
    model: str = None
) -> Dict:
    """Run multiple tasks in sequence."""
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
            config_overrides=config_overrides,
            context_size=context_size,
            model=model
        )
        
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
            stats['failed_tasks'].append(task_number)
    
    return stats


def preview_tasks(data_dir: str, tasks: List[Dict], results_dir: str):
    """Preview tasks without running them."""
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
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: no web search, only use long context (RAG) as information source"
    )
    parser.add_argument(
        "--context-size", "-c",
        type=str,
        choices=["32k", "64k", "128k"],
        default=None,
        help="Context size to use (32k, 64k, or 128k). Uses the corresponding sampled db file."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt4.1",
        help="Model name for result directory naming (default: gpt4.1)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = args.data_dir
    query_file = os.path.join(data_dir, args.query_file)
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Use results/results_<context_size>/ format when context_size is specified
        if args.context_size:
            results_dir = os.path.join("results", f"results_{args.context_size}")
        else:
            data_dir_name = os.path.basename(os.path.abspath(data_dir))
            results_dir = os.path.join("results", f"{data_dir_name}_{args.model}")
    
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
    
    # Prepare config overrides for offline mode
    config_overrides = []
    if args.offline:
        config_overrides.append("agent=evaluation_offline")
        print("🔒 Running in OFFLINE mode: No web search, using long context (RAG) only")
    
    # Print context size info
    if args.context_size:
        print(f"📊 Using context size: {args.context_size}")
        print(f"🤖 Model: {args.model}")
    
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
                skip_completed=args.skip_completed,
                config_overrides=config_overrides if config_overrides else None,
                context_size=args.context_size,
                model=args.model
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
