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
Folder Task Runner

This script demonstrates how to process a folder containing multiple files
(images, PDFs, videos, etc.) and run a task using the MiroFlow agent.

Usage:
    # Normal mode (with web search)
    uv run python run_folder_task.py --folder data/000 --query "请根据图片的重要属性，整理pdf中提到的重要文献，按照图片表格的形式整合，输出1000字左右的文献综述分析。"
    
    # Offline mode (no web search, only use long context as information source)
    uv run python run_folder_task.py --folder data/bench_case1104/005 --query "..." --offline
    
    # With Hydra config override
    uv run python run_folder_task.py agent=evaluation_offline --folder data/bench_case1104/005 --query "..."
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io.folder_processor import process_folder_for_task, scan_folder
from src.core.pipeline import create_pipeline_components, execute_task_pipeline
from src.logging.task_logger import bootstrap_logger

# Import RAG log path setter
try:
    from miroflow_tools.mcp_servers.rag_mcp_server import set_retrieval_log_path
except ImportError:
    set_retrieval_log_path = None

logger = bootstrap_logger()

# Default results directory
DEFAULT_RESULTS_DIR = "results"


def resolve_query_from_jsonl(query: str, folder_path: str) -> str:
    """
    Resolve query from a jsonl file if the query is a file path.
    
    If query ends with .jsonl, it reads the file and finds the matching query
    based on the folder number (e.g., folder '001' matches {"number": "001", ...}).
    
    Args:
        query: Either a direct query string or a path to a jsonl file
        folder_path: Path to the task folder (used to extract folder number)
        
    Returns:
        The resolved query string
    """
    # Check if query is a jsonl file path
    if not query.endswith('.jsonl'):
        return query
    
    # Check if the file exists
    if not os.path.exists(query):
        # Try relative to current directory
        if not os.path.isabs(query):
            # Query might be relative, return as-is if file doesn't exist
            return query
    
    # Extract folder number from folder_path
    folder_name = os.path.basename(os.path.abspath(folder_path))
    
    # Read jsonl file and find matching query
    try:
        with open(query, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Match by 'number' or 'task' field (different jsonl formats)
                    entry_id = entry.get('number') or entry.get('task')
                    if entry_id == folder_name:
                        resolved_query = entry.get('query', '')
                        if resolved_query:
                            logger.info(f"Resolved query from jsonl for folder '{folder_name}'")
                            return resolved_query
                except json.JSONDecodeError:
                    continue
        
        # If no match found, log warning and return original
        logger.warning(f"No matching query found in {query} for folder '{folder_name}'")
        return query
        
    except Exception as e:
        logger.warning(f"Failed to read jsonl file {query}: {e}")
        return query


def generate_turn_by_turn_log(log_file_path: str, output_path: str) -> str:
    """
    Parse the JSON log file and generate a human-readable turn-by-turn log.
    
    Args:
        log_file_path: Path to the JSON log file
        output_path: Path to save the turn-by-turn log
        
    Returns:
        Path to the generated log file
    """
    import json
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    lines = []
    lines.append("=" * 80)
    lines.append("TURN-BY-TURN EXECUTION LOG")
    lines.append("=" * 80)
    lines.append(f"Task ID: {log_data.get('task_id', 'N/A')}")
    lines.append(f"Status: {log_data.get('status', 'N/A')}")
    lines.append(f"Start Time: {log_data.get('start_time', 'N/A')}")
    lines.append(f"End Time: {log_data.get('end_time', 'N/A')}")
    lines.append("")
    
    # Parse step logs
    step_logs = log_data.get('step_logs', [])
    current_turn = 0
    current_agent = "Main Agent"
    
    for step in step_logs:
        step_name = step.get('step_name', '')
        message = step.get('message', '')
        timestamp = step.get('timestamp', '')
        info_level = step.get('info_level', 'info')
        
        # Detect turn changes
        if "Turn:" in step_name:
            import re
            turn_match = re.search(r'Turn:\s*(\d+)', step_name)
            if turn_match:
                new_turn = int(turn_match.group(1))
                if new_turn != current_turn:
                    current_turn = new_turn
                    lines.append("")
                    lines.append("-" * 80)
                    lines.append(f"TURN {current_turn}")
                    lines.append("-" * 80)
        
        # Detect agent changes
        if "agent-" in step_name.lower():
            agent_match = step_name.split("|")[0].strip()
            if agent_match != current_agent:
                current_agent = agent_match
                lines.append(f"\n>>> Agent: {current_agent}")
        
        # Format the step
        level_prefix = {
            'info': '[INFO]',
            'warning': '[WARN]',
            'error': '[ERROR]',
            'debug': '[DEBUG]'
        }.get(info_level, '[INFO]')
        
        lines.append(f"  {timestamp} {level_prefix} {step_name}")
        
        # Add message details for important steps
        if any(keyword in step_name.lower() for keyword in ['tool call', 'llm call', 'final answer', 'task description']):
            # Truncate very long messages
            if len(message) > 2000:
                message = message[:2000] + "... [truncated]"
            lines.append(f"    Message: {message}")
    
    # Add main agent message history
    main_history = log_data.get('main_agent_message_history', {})
    if main_history:
        lines.append("")
        lines.append("=" * 80)
        lines.append("MAIN AGENT MESSAGE HISTORY")
        lines.append("=" * 80)
        
        messages = main_history.get('message_history', [])
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            lines.append(f"\n--- Message {i+1} ({role}) ---")
            
            # Handle different content types
            if isinstance(content, str):
                if len(content) > 3000:
                    content = content[:3000] + "... [truncated]"
                lines.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get('type', 'unknown')
                        if item_type == 'text':
                            text = item.get('text', '')
                            if len(text) > 3000:
                                text = text[:3000] + "... [truncated]"
                            lines.append(f"[TEXT] {text}")
                        elif item_type == 'tool_use':
                            lines.append(f"[TOOL_USE] {item.get('name', 'unknown')}")
                            lines.append(f"  Input: {json.dumps(item.get('input', {}), ensure_ascii=False)[:1000]}")
                        elif item_type == 'tool_result':
                            result = str(item.get('content', ''))
                            if len(result) > 1000:
                                result = result[:1000] + "... [truncated]"
                            lines.append(f"[TOOL_RESULT] {result}")
                        else:
                            lines.append(f"[{item_type.upper()}] {str(item)[:500]}")
    
    # Add sub-agent sessions
    sub_sessions = log_data.get('sub_agent_message_history_sessions', {})
    if sub_sessions:
        lines.append("")
        lines.append("=" * 80)
        lines.append("SUB-AGENT SESSIONS")
        lines.append("=" * 80)
        
        for session_id, session_data in sub_sessions.items():
            lines.append(f"\n### Session: {session_id} ###")
            messages = session_data.get('message_history', [])
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                lines.append(f"\n--- Message {i+1} ({role}) ---")
                if isinstance(content, str):
                    if len(content) > 2000:
                        content = content[:2000] + "... [truncated]"
                    lines.append(content)
    
    # Add final answer
    final_answer = log_data.get('final_boxed_answer', '')
    if final_answer:
        lines.append("")
        lines.append("=" * 80)
        lines.append("FINAL ANSWER")
        lines.append("=" * 80)
        lines.append(final_answer)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path


def save_report_to_md(
    results_dir: str,
    folder_name: str,
    final_answer: str,
    summary: str = None,
    query: str = None,
) -> str:
    """
    Save the final report to a markdown file with timestamp.
    
    Args:
        results_dir: Directory to save results
        folder_name: Name of the task folder (used as filename)
        final_answer: The final boxed answer/report
        summary: Optional summary text
        query: Optional original query
        
    Returns:
        Path to the saved file
    """
    os.makedirs(results_dir, exist_ok=True)
    # Add timestamp to filename to avoid overwriting
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = os.path.join(results_dir, f"{folder_name}_{timestamp}.md")
    
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"# Task Report: {folder_name}\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if query:
            f.write("## Query\n\n")
            f.write(f"{query}\n\n")
        
        f.write("## Final Report\n\n")
        f.write(final_answer if final_answer else "No answer generated.")
        
        if summary:
            f.write("\n\n## Summary\n\n")
            f.write(summary)
    
    return result_path


def save_report_comparison(
    results_dir: str,
    folder_name: str,
    original_report: str,
    final_report: str,
    query: str = None,
) -> str:
    """
    Save both original and final reports to a comparison file.
    
    Args:
        results_dir: Directory to save results
        folder_name: Name of the task folder (used as filename)
        original_report: The original report before validation
        final_report: The final report after validation
        query: Optional original query
        
    Returns:
        Path to the saved comparison file
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    comparison_path = os.path.join(results_dir, f"{folder_name}_{timestamp}_comparison.txt")
    
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORT COMPARISON: ORIGINAL vs FINAL (After Validation)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Task: {folder_name}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if query:
            f.write("-" * 80 + "\n")
            f.write("QUERY\n")
            f.write("-" * 80 + "\n")
            f.write(f"{query}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("ORIGINAL REPORT (Before Validation)\n")
        f.write("=" * 80 + "\n\n")
        f.write(original_report if original_report else "No original report generated.")
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("FINAL REPORT (After Validation)\n")
        f.write("=" * 80 + "\n\n")
        f.write(final_report if final_report else "No final report generated.")
        f.write("\n\n")
        
        # Add comparison summary
        f.write("=" * 80 + "\n")
        f.write("COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        original_len = len(original_report) if original_report else 0
        final_len = len(final_report) if final_report else 0
        
        f.write(f"Original report length: {original_len} characters\n")
        f.write(f"Final report length: {final_len} characters\n")
        f.write(f"Length difference: {final_len - original_len} characters\n")
        
        if original_report == final_report:
            f.write("\n✅ Reports are IDENTICAL (no changes made during validation)\n")
        else:
            f.write("\n⚠️ Reports are DIFFERENT (changes were made during validation)\n")
    
    return comparison_path


async def run_folder_task(
    cfg: DictConfig,
    folder_path: str,
    query: str,
    task_id: str = None,
    recursive: bool = False,
) -> tuple:
    """
    Run a task on a folder containing multiple files.
    
    Args:
        cfg: Hydra configuration object
        folder_path: Path to the folder to process
        query: The user's query about the folder contents (can be a jsonl file path)
        task_id: Optional task ID (auto-generated if not provided)
        recursive: Whether to scan subdirectories recursively
        
    Returns:
        Tuple of (final_summary, final_boxed_answer, log_file_path)
    """
    # Generate task ID if not provided
    if task_id is None:
        folder_name = os.path.basename(os.path.abspath(folder_path))
        task_id = f"folder_task_{folder_name}"
    
    # Resolve query from jsonl file if needed
    query = resolve_query_from_jsonl(query, folder_path)
    
    logger.info(f"Processing folder: {folder_path}")
    logger.info(f"Query: {query}")
    
    # Process folder and prepare task description
    task_content, task_description, multimodal_files = process_folder_for_task(
        folder_path=folder_path,
        query=query,
        recursive=recursive,
        include_file_contents=True,
    )
    
    logger.info(f"Found {len(multimodal_files)} multimodal files")
    for f in multimodal_files:
        logger.info(f"  - {f}")
    
    # Create pipeline components
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg)
    )
    
    # Set up RAG retrieval log path
    if set_retrieval_log_path is not None:
        log_dir = cfg.get("debug_dir", "logs/debug")
        retrieval_log_path = os.path.join(log_dir, f"{task_id}_rag_retrieval.json")
        set_retrieval_log_path(retrieval_log_path)
        logger.info(f"RAG retrieval log will be saved to: {retrieval_log_path}")
    
    # For multimodal files, we pass the first one as task_file_name
    # The rest are included in the task_description
    task_file_name = multimodal_files[0] if multimodal_files else ""
    
    # Execute task
    result = await execute_task_pipeline(
        cfg=cfg,
        task_id=task_id,
        task_file_name=task_file_name,
        task_description=task_description,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        log_dir=cfg.debug_dir,
    )
    
    # Handle both old (3-tuple) and new (4-tuple) return formats
    if len(result) == 4:
        final_summary, final_boxed_answer, original_boxed_answer, log_file_path = result
    else:
        final_summary, final_boxed_answer, log_file_path = result
        original_boxed_answer = final_boxed_answer  # No validation was done
    
    return final_summary, final_boxed_answer, original_boxed_answer, log_file_path


async def run_folder_task_simple(
    folder_path: str,
    query: str,
    config_overrides: list = None,
) -> tuple:
    """
    Simplified interface to run a folder task with default config.
    
    Args:
        folder_path: Path to the folder to process
        query: The user's query about the folder contents (can be a jsonl file path)
        config_overrides: Optional list of Hydra config overrides
        
    Returns:
        Tuple of (final_summary, final_boxed_answer, log_file_path)
    """
    # Resolve query from jsonl file if needed
    query = resolve_query_from_jsonl(query, folder_path)
    
    # Initialize Hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Get config directory
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")
    
    # Initialize with config directory
    initialize_config_dir(config_dir=config_dir, version_base=None)
    
    # Compose config with overrides
    overrides = config_overrides if config_overrides else []
    cfg = compose(config_name="config", overrides=overrides)
    
    # Run task
    return await run_folder_task(cfg, folder_path, query)


def preview_folder(folder_path: str, recursive: bool = False) -> None:
    """
    Preview folder contents without running a task.
    
    Args:
        folder_path: Path to the folder to preview
        recursive: Whether to scan subdirectories recursively
    """
    contents = scan_folder(folder_path, recursive=recursive)
    
    print("\n" + "=" * 60)
    print("FOLDER CONTENTS PREVIEW")
    print("=" * 60)
    print(contents.get_summary())
    print("\n" + "-" * 60)
    
    if contents.images:
        print("\nImages:")
        for f in contents.images:
            print(f"  - {f.name} ({f.size_bytes / 1024:.1f} KB)")
    
    if contents.videos:
        print("\nVideos:")
        for f in contents.videos:
            print(f"  - {f.name} ({f.size_bytes / 1024 / 1024:.1f} MB)")
    
    if contents.audios:
        print("\nAudio files:")
        for f in contents.audios:
            print(f"  - {f.name} ({f.size_bytes / 1024:.1f} KB)")
    
    if contents.documents:
        print("\nDocuments:")
        for f in contents.documents:
            print(f"  - {f.name} ({f.size_bytes / 1024:.1f} KB)")
    
    if contents.spreadsheets:
        print("\nSpreadsheets:")
        for f in contents.spreadsheets:
            print(f"  - {f.name} ({f.size_bytes / 1024:.1f} KB)")
    
    if contents.data_files:
        print("\nData files:")
        for f in contents.data_files:
            print(f"  - {f.name} ({f.size_bytes / 1024:.1f} KB)")
    
    print("\n" + "=" * 60)


def get_results_dir_for_folder(folder_path: str) -> str:
    """
    Determine the results directory based on folder path structure.
    
    For paths like 'data/bench_case1104/001', returns 'results/bench_case1104'
    For paths like 'data/000', returns 'results'
    
    Args:
        folder_path: Path to the task folder
        
    Returns:
        Path to the results directory
    """
    abs_path = os.path.abspath(folder_path)
    parts = abs_path.split(os.sep)
    
    # Find 'data' in path and get the structure after it
    try:
        data_idx = parts.index('data')
        # If there's a parent folder between 'data' and the task folder
        if len(parts) > data_idx + 2:
            parent_folder = parts[data_idx + 1]
            return os.path.join(DEFAULT_RESULTS_DIR, parent_folder)
    except ValueError:
        pass
    
    return DEFAULT_RESULTS_DIR


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra config."""
    # Parse additional arguments
    parser = argparse.ArgumentParser(description="Run task on folder")
    parser.add_argument("--folder", "-f", required=True, help="Path to folder")
    parser.add_argument("--query", "-q", required=True, help="Query about folder contents")
    parser.add_argument("--task-id", "-t", default=None, help="Task ID")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan recursively")
    parser.add_argument("--preview", "-p", action="store_true", help="Preview folder only")
    parser.add_argument("--results-dir", "-o", default=None, help="Directory to save results (default: auto-detect)")
    
    # Parse known args (Hydra handles the rest)
    args, _ = parser.parse_known_args()
    
    if args.preview:
        preview_folder(args.folder, args.recursive)
        return
    
    # Run task
    final_summary, final_boxed_answer, original_boxed_answer, log_file_path = asyncio.run(
        run_folder_task(
            cfg=cfg,
            folder_path=args.folder,
            query=args.query,
            task_id=args.task_id,
            recursive=args.recursive,
        )
    )
    
    # Save report to markdown file
    folder_name = os.path.basename(os.path.abspath(args.folder))
    results_dir = args.results_dir if args.results_dir else get_results_dir_for_folder(args.folder)
    report_path = save_report_to_md(
        results_dir=results_dir,
        folder_name=folder_name,
        final_answer=final_boxed_answer,
        summary=final_summary,
        query=args.query,
    )
    
    # Save report comparison file
    comparison_path = save_report_comparison(
        results_dir=results_dir,
        folder_name=folder_name,
        original_report=original_boxed_answer,
        final_report=final_boxed_answer,
        query=args.query,
    )
    
    # Generate turn-by-turn log
    turn_log_path = report_path.replace('.md', '_turns.txt')
    try:
        generate_turn_by_turn_log(log_file_path, turn_log_path)
        print(f"Turn-by-turn log saved to: {turn_log_path}")
    except Exception as e:
        print(f"Warning: Failed to generate turn-by-turn log: {e}")
    
    print("\n" + "=" * 60)
    print("TASK COMPLETED")
    print("=" * 60)
    print(f"\nLog file: {log_file_path}")
    print(f"Report saved to: {report_path}")
    print(f"Report comparison saved to: {comparison_path}")
    print(f"Turn-by-turn log: {turn_log_path}")
    print(f"\nFinal Answer:\n{final_boxed_answer}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Check if running with Hydra or standalone
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        # Running standalone without Hydra
        parser = argparse.ArgumentParser(description="Run task on folder")
        parser.add_argument("--folder", "-f", required=True, help="Path to folder")
        parser.add_argument("--query", "-q", required=True, help="Query about folder contents")
        parser.add_argument("--preview", "-p", action="store_true", help="Preview folder only")
        parser.add_argument("--recursive", "-r", action="store_true", help="Scan recursively")
        parser.add_argument("--results-dir", "-o", default=None, help="Directory to save results")
        parser.add_argument("--offline", action="store_true", 
                          help="Offline mode: no web search, only use long context (RAG) as information source")
        
        args = parser.parse_args()
        
        if args.preview:
            preview_folder(args.folder, args.recursive)
        else:
            # Prepare config overrides
            config_overrides = []
            if args.offline:
                config_overrides.append("agent=evaluation_offline")
                print("🔒 Running in OFFLINE mode: No web search, using long context (RAG) only")
            
            # Run with config
            final_summary, final_boxed_answer, original_boxed_answer, log_file_path = asyncio.run(
                run_folder_task_simple(args.folder, args.query, config_overrides=config_overrides)
            )
            
            # Save report to markdown file
            folder_name = os.path.basename(os.path.abspath(args.folder))
            results_dir = args.results_dir if args.results_dir else get_results_dir_for_folder(args.folder)
            report_path = save_report_to_md(
                results_dir=results_dir,
                folder_name=folder_name,
                final_answer=final_boxed_answer,
                summary=final_summary,
                query=args.query,
            )
            
            # Save report comparison file
            comparison_path = save_report_comparison(
                results_dir=results_dir,
                folder_name=folder_name,
                original_report=original_boxed_answer,
                final_report=final_boxed_answer,
                query=args.query,
            )
            
            # Generate turn-by-turn log
            turn_log_path = report_path.replace('.md', '_turns.txt')
            try:
                generate_turn_by_turn_log(log_file_path, turn_log_path)
                print(f"Turn-by-turn log saved to: {turn_log_path}")
            except Exception as e:
                print(f"Warning: Failed to generate turn-by-turn log: {e}")
            
            print("\n" + "=" * 60)
            print("TASK COMPLETED")
            print("=" * 60)
            print(f"\nLog file: {log_file_path}")
            print(f"Report saved to: {report_path}")
            print(f"Report comparison saved to: {comparison_path}")
            print(f"Turn-by-turn log: {turn_log_path}")
            print(f"\nFinal Answer:\n{final_boxed_answer}")
            print("\n" + "=" * 60)
    else:
        # Running with Hydra
        main()
