import asyncio
import json
import os
# Usage: uv run scripts/generate_deep_research_data.py llm=qwen-3 agent=report +limit=2 +start_index=0
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Add parent directory to sys.path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.pipeline import create_pipeline_components, execute_task_pipeline
from src.logging.task_logger import bootstrap_logger

logger = bootstrap_logger()

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))

async def amain(cfg: DictConfig) -> None:
    logger.info("Starting Deep Research Data Generation...")
    
    # Get original working directory (before Hydra changed it)
    try:
        original_cwd = hydra.utils.get_original_cwd()
    except:
        original_cwd = os.getcwd()
        
    # Define workspace root relative to apps/miroflow-agent/
    # original_cwd should be apps/miroflow-agent/
    workspace_root = os.path.abspath(os.path.join(original_cwd, "../../"))
    
    # Define persistent report directory
    persistent_report_dir = os.path.join(original_cwd, "report", "deep_research_bench")
    os.makedirs(persistent_report_dir, exist_ok=True)
    logger.info(f"Persistent report directory: {persistent_report_dir}")

    # Input file path
    input_file = os.path.join(workspace_root, "deep_research_bench-main/data/prompt_data/query.jsonl")
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    # Output file path
    # Use model name from config
    model_name = cfg.llm.get("model_name", "unknown_model")
    logger.info(f"Resolved model_name: {model_name}")
    # Sanitize model name
    model_name = model_name.replace("/", "_").replace(":", "_")
    
    output_dir = os.path.join(workspace_root, "deep_research_bench-main/data/test_data/raw_data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}.jsonl")
    
    logger.info(f"Reading queries from: {input_file}")
    logger.info(f"Writing results to: {output_file}")

    # Load queries
    queries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line}")
    
    # Handle Start Index
    start_index = cfg.get("start_index", 0)
    if start_index:
        start_index = int(start_index)
        logger.info(f"Starting from index {start_index}.")
    else:
        start_index = 0

    # Handle Limit
    limit = cfg.get("limit", None)
    if limit:
        limit = int(limit)
        logger.info(f"Limiting to {limit} queries.")
        # Slice queries based on start_index and limit
        queries = queries[start_index : start_index + limit]
    else:
        queries = queries[start_index:]

    # Initialize pipeline
    logger.info("Initializing pipeline components...")
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = create_pipeline_components(cfg)
    
    # Check existing results in JSONL to avoid duplicates in the output file
    existing_jsonl_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        existing_jsonl_ids.add(data.get("id"))
                    except:
                        pass
    
    logger.info(f"Found {len(existing_jsonl_ids)} existing results in output JSONL.")

    for query in queries:
        # Use the original ID from query.jsonl (integer)
        # This ensures compatibility with deep_research_bench's extract.py which expects IDs to match
        task_id = query['id']
        
        # For internal file naming (report cache), we can still use a prefix if desired, 
        # or just use the ID. Let's use "task_{id}" for filenames to be safe, 
        # but keep the ID in the JSONL as the original integer/string from query.
        report_filename_id = f"task_{task_id}"
        prompt = query['prompt']
        
        # Path for the persistent markdown report
        persistent_report_path = os.path.join(persistent_report_dir, f"{report_filename_id}.md")
        
        article_content = ""
        
        # Check if report already exists in persistent storage
        if os.path.exists(persistent_report_path):
            logger.info(f"Report for {report_filename_id} already exists at {persistent_report_path}. Skipping execution.")
            with open(persistent_report_path, 'r', encoding='utf-8') as f:
                article_content = f.read()
        else:
            logger.info(f"Processing {report_filename_id}: {prompt[:50]}...")
            try:
                # Execute the pipeline
                # Note: execute_task_pipeline saves the report to "report/{task_id}.md" in the current working directory (Hydra run dir)
                # We use report_filename_id here for the pipeline execution to avoid potential issues with pure integer IDs in some systems
                await execute_task_pipeline(
                    cfg=cfg,
                    task_id=report_filename_id,
                    task_description=prompt,
                    task_file_paths=[],
                    main_agent_tool_manager=main_agent_tool_manager,
                    sub_agent_tool_managers=sub_agent_tool_managers,
                    output_formatter=output_formatter,
                    log_dir="logs", 
                )
                
                # Read the generated report from Hydra temp dir
                temp_report_path = os.path.join("report", f"{report_filename_id}.md")
                
                if os.path.exists(temp_report_path):
                    with open(temp_report_path, 'r', encoding='utf-8') as f:
                        article_content = f.read()
                    
                    # Save to persistent location
                    with open(persistent_report_path, 'w', encoding='utf-8') as f:
                        f.write(article_content)
                    logger.info(f"Saved persistent report to {persistent_report_path}")
                else:
                    logger.error(f"Report file not found for {report_filename_id} at {temp_report_path}")
                    continue # Skip writing to JSONL if report generation failed
                    
            except Exception as e:
                logger.error(f"Error processing {report_filename_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Append to output JSONL file if not already present
        
        if article_content:
            # Get language from query, default to 'en'
            language = query.get("language", "en")
            
            result_entry = {
                "id": task_id, # Use the original ID (likely integer)
                "prompt": prompt,
                "language": language,
                "article": article_content
            }
            
            # Check if this ID is already in the file
            # Note: existing_jsonl_ids stores whatever was in the file. 
            # If the file had "task_1", and now we are writing 1, we might duplicate if we are not careful.
            # But since we are regenerating, we assume the user cleared the file or we are appending consistent IDs.
            
            if task_id not in existing_jsonl_ids:
                try:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    existing_jsonl_ids.add(task_id)
                    logger.info(f"Added result for {task_id} to {output_file}")
                except Exception as e:
                    logger.error(f"Failed to write result to JSONL for {task_id}: {e}")
            else:
                 logger.info(f"Result for {task_id} already in JSONL. Skipping write.")

if __name__ == "__main__":
    main()
