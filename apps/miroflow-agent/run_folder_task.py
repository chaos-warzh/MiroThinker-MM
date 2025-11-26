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
    uv run python run_folder_task.py --folder data/000 --query "请根据图片的重要属性，整理pdf中提到的重要文献，按照图片表格的形式整合，输出1000字左右的文献综述分析。"
    uv run python run_folder_task.py --folder data/001 --query "请根据图片的重要属性，以及pdf的文献，同时参考long context里面的内容，输出1000字左右的文献综述分析。"
    uv run python run_folder_task.py --folder data/bench_case1104/001 --query "Assume you are a cultural travel researcher preparing a comprehensive tourism report for an international travel magazine. Based on the materials I provide, please write a report titled 'Exploring Cambodia: A Cultural and Culinary Journey'. The report should analyze Cambodia’s major tourist attractions, recommend representative local cuisines, and provide practical travel tips for international visitors.\n\nRequirements:\n1. The report must integrate information from the provided documents and, where necessary, additional verified online sources.\n2. All factual statements must be accurate, verifiable, and properly cited with clear references to both the source document and specific location.\n3. The writing style should be formal, engaging, and suitable for publication in an academic travel review.\n4. The report should be between 500 and 600 words."
"""

import argparse
import asyncio
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


def save_report_to_md(
    results_dir: str,
    folder_name: str,
    final_answer: str,
    summary: str = None,
    query: str = None,
) -> str:
    """
    Save the final report to a markdown file.
    
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
    result_path = os.path.join(results_dir, f"{folder_name}.md")
    
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
        query: The user's query about the folder contents
        task_id: Optional task ID (auto-generated if not provided)
        recursive: Whether to scan subdirectories recursively
        
    Returns:
        Tuple of (final_summary, final_boxed_answer, log_file_path)
    """
    # Generate task ID if not provided
    if task_id is None:
        folder_name = os.path.basename(os.path.abspath(folder_path))
        task_id = f"folder_task_{folder_name}"
    
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
    final_summary, final_boxed_answer, log_file_path = await execute_task_pipeline(
        cfg=cfg,
        task_id=task_id,
        task_file_name=task_file_name,
        task_description=task_description,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        log_dir=cfg.debug_dir,
    )
    
    return final_summary, final_boxed_answer, log_file_path


async def run_folder_task_simple(
    folder_path: str,
    query: str,
    config_overrides: list = None,
) -> tuple:
    """
    Simplified interface to run a folder task with default config.
    
    Args:
        folder_path: Path to the folder to process
        query: The user's query about the folder contents
        config_overrides: Optional list of Hydra config overrides
        
    Returns:
        Tuple of (final_summary, final_boxed_answer, log_file_path)
    """
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
    final_summary, final_boxed_answer, log_file_path = asyncio.run(
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
    
    print("\n" + "=" * 60)
    print("TASK COMPLETED")
    print("=" * 60)
    print(f"\nLog file: {log_file_path}")
    print(f"Report saved to: {report_path}")
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
        
        args = parser.parse_args()
        
        if args.preview:
            preview_folder(args.folder, args.recursive)
        else:
            # Run with default config
            final_summary, final_boxed_answer, log_file_path = asyncio.run(
                run_folder_task_simple(args.folder, args.query)
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
            
            print("\n" + "=" * 60)
            print("TASK COMPLETED")
            print("=" * 60)
            print(f"\nLog file: {log_file_path}")
            print(f"Report saved to: {report_path}")
            print(f"\nFinal Answer:\n{final_boxed_answer}")
            print("\n" + "=" * 60)
    else:
        # Running with Hydra
        main()
