#!/usr/bin/env python3
"""
Build Long Context from Useful + Noise

This script:
1. Reads useful_search.json and noise_search.json from each task folder
2. Builds context at different sizes: 32k, 64k, 128k, 256k, 512k, 1024k tokens
3. Starts from largest (1024k) and subsets down (smaller sizes are subsets of larger)
4. Shuffles documents at each level
5. Mixes useful documents randomly into the noise (not at beginning or end)
6. Saves as long_context_sampled_{size}k.json
7. Optionally runs embedding preprocessing

Usage:
    python build_long_context.py --dir datasets_batch2_1228
    python build_long_context.py --dir datasets_batch2_1228 --embed  # Also run embedding
    python build_long_context.py --dir datasets_batch2_1228 --tokens 32000 64000 128000
"""

import os
import sys
import json
import random
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

import tiktoken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text, disallowed_special=()))


def count_json_tokens(data: list, encoding_name: str = "o200k_base") -> int:
    """Count tokens in JSON data."""
    json_str = json.dumps(data, ensure_ascii=False)
    return count_tokens(json_str, encoding_name)


def count_doc_tokens(doc: dict, encoding_name: str = "o200k_base") -> int:
    """Count tokens for a single document."""
    doc_str = json.dumps(doc, ensure_ascii=False)
    return count_tokens(doc_str, encoding_name)


def sample_docs_to_target(
    docs: List[dict],
    target_tokens: int,
    encoding_name: str = "o200k_base"
) -> Tuple[List[dict], int]:
    """
    Sample documents to reach target token count.
    Assumes docs are already shuffled.
    
    Args:
        docs: List of documents (already shuffled)
        target_tokens: Target token count
        encoding_name: Tiktoken encoding name
        
    Returns:
        Tuple of (sampled_docs, actual_tokens)
    """
    sampled = []
    current_tokens = 2  # JSON array brackets []
    
    for doc in docs:
        doc_tokens = count_doc_tokens(doc, encoding_name)
        overhead = 2 if sampled else 0  # ", " between documents
        new_total = current_tokens + doc_tokens + overhead
        
        if new_total > target_tokens:
            # Check if adding this doc gets us closer to target
            diff_without = abs(target_tokens - current_tokens)
            diff_with = abs(target_tokens - new_total)
            
            if diff_with < diff_without:
                sampled.append(doc)
                current_tokens = new_total
            break
        
        sampled.append(doc)
        current_tokens = new_total
    
    actual_tokens = count_json_tokens(sampled, encoding_name) if sampled else 0
    return sampled, actual_tokens


def mix_useful_into_noise(
    useful_docs: List[dict],
    noise_docs: List[dict],
    seed: int = 42
) -> List[dict]:
    """
    Mix useful documents randomly into noise documents.
    Useful documents should NOT be at the very beginning or very end.
    
    Args:
        useful_docs: List of useful documents
        noise_docs: List of noise documents
        seed: Random seed
        
    Returns:
        Mixed list of documents
    """
    random.seed(seed)
    
    if not noise_docs:
        return useful_docs.copy()
    
    if len(noise_docs) < 2:
        # Not enough noise to avoid beginning/end, just interleave
        result = []
        for i, useful in enumerate(useful_docs):
            if i < len(noise_docs):
                result.append(noise_docs[i])
            result.append(useful)
        return result
    
    # Create result list starting with noise
    result = noise_docs.copy()
    
    # Insert useful documents at random positions (not first or last)
    for useful_doc in useful_docs:
        # Valid positions: 1 to len(result)-1 (not 0, not end)
        if len(result) <= 2:
            pos = 1
        else:
            pos = random.randint(1, len(result) - 1)
        result.insert(pos, useful_doc)
    
    return result


def find_task_folders(directory: str, tasks: List[str] = None) -> List[str]:
    """
    Find task folders (001, 002, etc.) in directory.
    
    Args:
        directory: Base directory
        tasks: Optional list of specific task IDs to process (e.g., ['001', '002'])
               If None, process all task folders
    
    Returns:
        List of task folder paths
    """
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.isdigit():
            # If specific tasks are requested, filter
            if tasks is None or item in tasks:
                folders.append(item_path)
    return sorted(folders)


def build_independent_contexts(
    task_folder: str,
    target_tokens_list: List[int],
    encoding_name: str = "o200k_base",
    seed: int = 42,
    force: bool = False
) -> List[dict]:
    """
    Build independent contexts for a single task folder.
    Each size is independently sampled from the full noise pool.
    
    Args:
        task_folder: Path to task folder
        target_tokens_list: List of target token counts
        encoding_name: Tiktoken encoding name
        seed: Random seed
        force: Force reprocessing
        
    Returns:
        List of processing results
    """
    results = []
    
    # Sort targets for consistent output
    sorted_targets = sorted(target_tokens_list)
    
    try:
        useful_file = os.path.join(task_folder, "useful_search.json")
        noise_file = os.path.join(task_folder, "noise_search.json")
        
        # Check if files exist
        if not os.path.exists(useful_file):
            raise FileNotFoundError(f"useful_search.json not found in {task_folder}")
        if not os.path.exists(noise_file):
            raise FileNotFoundError(f"noise_search.json not found in {task_folder}")
        
        # Load useful documents
        with open(useful_file, 'r', encoding='utf-8') as f:
            useful_docs = json.load(f)
        
        # Load noise documents
        with open(noise_file, 'r', encoding='utf-8') as f:
            noise_docs = json.load(f)
        
        useful_tokens = count_json_tokens(useful_docs, encoding_name)
        noise_total_tokens = count_json_tokens(noise_docs, encoding_name)
        logger.info(f"  Useful: {len(useful_docs)} docs, {useful_tokens:,} tokens")
        logger.info(f"  Noise: {len(noise_docs)} docs, {noise_total_tokens:,} tokens")
        
        for target_tokens in sorted_targets:
            result = {
                "folder": task_folder,
                "target_tokens": target_tokens,
                "status": "unknown",
                "useful_docs": len(useful_docs),
                "useful_tokens": useful_tokens,
                "noise_docs": 0,
                "noise_tokens": 0,
                "total_docs": 0,
                "total_tokens": 0,
                "output_file": None,
                "error": None
            }
            
            token_k = target_tokens // 1000
            output_file = os.path.join(task_folder, f"long_context_sampled_{token_k}k.json")
            result["output_file"] = output_file
            
            # Check if output already exists
            if not force and os.path.exists(output_file):
                logger.info(f"  [{token_k}k] Output already exists, skipping")
                result["status"] = "skipped"
                results.append(result)
                continue
            
            # Calculate how many tokens we need from noise
            noise_target = target_tokens - useful_tokens
            
            if noise_target <= 0:
                logger.warning(f"  [{token_k}k] Useful tokens ({useful_tokens}) >= target ({target_tokens}), using only useful docs")
                sampled_noise = []
                noise_tokens = 0
            else:
                # Each size independently samples from the FULL noise pool
                random.seed(seed + target_tokens)  # Different seed for each level
                shuffled_noise = noise_docs.copy()
                random.shuffle(shuffled_noise)
                
                # Sample from the shuffled full pool
                sampled_noise, noise_tokens = sample_docs_to_target(
                    shuffled_noise, noise_target, encoding_name
                )
            
            result["noise_docs"] = len(sampled_noise)
            result["noise_tokens"] = noise_tokens
            
            # Mix useful into noise
            mixed_docs = mix_useful_into_noise(useful_docs, sampled_noise, seed + target_tokens)
            
            result["total_docs"] = len(mixed_docs)
            result["total_tokens"] = count_json_tokens(mixed_docs, encoding_name)
            
            logger.info(f"  [{token_k}k] {len(mixed_docs)} docs, {result['total_tokens']:,} tokens (useful: {useful_tokens:,}, noise: {noise_tokens:,})")
            
            # Save output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(mixed_docs, f, ensure_ascii=False, indent=2)
            
            result["status"] = "success"
            results.append(result)
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        for target_tokens in sorted_targets:
            results.append({
                "folder": task_folder,
                "target_tokens": target_tokens,
                "status": "error",
                "error": str(e)
            })
    
    return results


def run_embedding_preprocessing(
    directory: str,
    pattern: str = "long_context_sampled_*.json",
    force: bool = False,
    tasks: List[str] = None
):
    """
    Run the embedding preprocessing script.
    
    Args:
        directory: Directory containing the files
        pattern: File pattern to match
        force: Force reprocessing
        tasks: Specific task IDs to process (if None, process all)
    """
    script_path = os.path.join(os.path.dirname(__file__), "preprocess_long_context.py")
    
    # If specific tasks are specified, run preprocessing for each task folder
    if tasks:
        for task in tasks:
            task_dir = os.path.join(directory, task)
            if os.path.isdir(task_dir):
                cmd = [
                    sys.executable, script_path,
                    "--dir", task_dir,
                    "--pattern", pattern
                ]
                
                if force:
                    cmd.append("--force")
                
                logger.info(f"\nRunning embedding preprocessing for task {task}...")
                logger.info(f"Command: {' '.join(cmd)}")
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Embedding preprocessing failed for task {task}: {e}")
                    raise
    else:
        # Process all tasks
        cmd = [
            sys.executable, script_path,
            "--dir", directory,
            "--pattern", pattern
        ]
        
        if force:
            cmd.append("--force")
        
        logger.info(f"\nRunning embedding preprocessing...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Embedding preprocessing failed: {e}")
            raise


def load_jsonl_docs(file_path: str) -> List[dict]:
    """
    Load documents from a JSONL file.
    Each line can be either:
    - A JSON array of documents
    - A single JSON object (document)
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of document dictionaries
    """
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, list):
                    docs.extend(data)
                elif isinstance(data, dict):
                    docs.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line in {file_path}: {e}")
                continue
    return docs


def build_independent_contexts_jsonl(
    task_folder: str,
    target_tokens_list: List[int],
    encoding_name: str = "o200k_base",
    seed: int = 42,
    force: bool = False,
    tolerance: float = 0.02
) -> List[dict]:
    """
    Build independent contexts for a single task folder using JSONL files.
    Only generates context sizes that can be achieved (within tolerance).
    
    Args:
        task_folder: Path to task folder
        target_tokens_list: List of target token counts
        encoding_name: Tiktoken encoding name
        seed: Random seed
        force: Force reprocessing
        tolerance: Tolerance for target token count (e.g., 0.02 = 2%)
        
    Returns:
        List of processing results
    """
    results = []
    
    # Sort targets for consistent output
    sorted_targets = sorted(target_tokens_list)
    
    try:
        # Try JSONL files first, then fall back to JSON
        useful_jsonl = os.path.join(task_folder, "useful_search.jsonl")
        noise_jsonl = os.path.join(task_folder, "noise.jsonl")
        useful_json = os.path.join(task_folder, "useful_search.json")
        noise_json = os.path.join(task_folder, "noise_search.json")
        
        # Load useful documents
        if os.path.exists(useful_jsonl):
            useful_docs = load_jsonl_docs(useful_jsonl)
            logger.info(f"  Loaded useful docs from JSONL: {len(useful_docs)} docs")
        elif os.path.exists(useful_json):
            with open(useful_json, 'r', encoding='utf-8') as f:
                useful_docs = json.load(f)
            logger.info(f"  Loaded useful docs from JSON: {len(useful_docs)} docs")
        else:
            raise FileNotFoundError(f"No useful_search.jsonl or useful_search.json found in {task_folder}")
        
        # Load noise documents
        if os.path.exists(noise_jsonl):
            noise_docs = load_jsonl_docs(noise_jsonl)
            logger.info(f"  Loaded noise docs from JSONL: {len(noise_docs)} docs")
        elif os.path.exists(noise_json):
            with open(noise_json, 'r', encoding='utf-8') as f:
                noise_docs = json.load(f)
            logger.info(f"  Loaded noise docs from JSON: {len(noise_docs)} docs")
        else:
            raise FileNotFoundError(f"No noise.jsonl or noise_search.json found in {task_folder}")
        
        useful_tokens = count_json_tokens(useful_docs, encoding_name)
        noise_total_tokens = count_json_tokens(noise_docs, encoding_name)
        total_available_tokens = useful_tokens + noise_total_tokens
        
        logger.info(f"  Useful: {len(useful_docs)} docs, {useful_tokens:,} tokens")
        logger.info(f"  Noise: {len(noise_docs)} docs, {noise_total_tokens:,} tokens")
        logger.info(f"  Total available: {total_available_tokens:,} tokens")
        
        for target_tokens in sorted_targets:
            result = {
                "folder": task_folder,
                "target_tokens": target_tokens,
                "status": "unknown",
                "useful_docs": len(useful_docs),
                "useful_tokens": useful_tokens,
                "noise_docs": 0,
                "noise_tokens": 0,
                "total_docs": 0,
                "total_tokens": 0,
                "output_file": None,
                "error": None
            }
            
            token_k = target_tokens // 1000
            output_file = os.path.join(task_folder, f"long_context_sampled_{token_k}k.json")
            result["output_file"] = output_file
            
            # Check if we have enough tokens (with tolerance)
            min_required = target_tokens * (1 - tolerance)
            if total_available_tokens < min_required:
                logger.info(f"  [{token_k}k] Skipping - not enough tokens (have {total_available_tokens:,}, need {min_required:,.0f})")
                result["status"] = "insufficient_tokens"
                result["error"] = f"Not enough tokens: have {total_available_tokens:,}, need {min_required:,.0f}"
                results.append(result)
                continue
            
            # Check if output already exists
            if not force and os.path.exists(output_file):
                logger.info(f"  [{token_k}k] Output already exists, skipping")
                result["status"] = "skipped"
                results.append(result)
                continue
            
            # Calculate how many tokens we need from noise
            noise_target = target_tokens - useful_tokens
            
            if noise_target <= 0:
                logger.warning(f"  [{token_k}k] Useful tokens ({useful_tokens}) >= target ({target_tokens}), using only useful docs")
                sampled_noise = []
                noise_tokens = 0
            else:
                # Each size independently samples from the FULL noise pool
                random.seed(seed + target_tokens)  # Different seed for each level
                shuffled_noise = noise_docs.copy()
                random.shuffle(shuffled_noise)
                
                # Sample from the shuffled full pool
                sampled_noise, noise_tokens = sample_docs_to_target(
                    shuffled_noise, noise_target, encoding_name
                )
            
            result["noise_docs"] = len(sampled_noise)
            result["noise_tokens"] = noise_tokens
            
            # Mix useful into noise
            mixed_docs = mix_useful_into_noise(useful_docs, sampled_noise, seed + target_tokens)
            
            result["total_docs"] = len(mixed_docs)
            result["total_tokens"] = count_json_tokens(mixed_docs, encoding_name)
            
            logger.info(f"  [{token_k}k] {len(mixed_docs)} docs, {result['total_tokens']:,} tokens (useful: {useful_tokens:,}, noise: {noise_tokens:,})")
            
            # Save output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(mixed_docs, f, ensure_ascii=False, indent=2)
            
            result["status"] = "success"
            results.append(result)
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        for target_tokens in sorted_targets:
            results.append({
                "folder": task_folder,
                "target_tokens": target_tokens,
                "status": "error",
                "error": str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build hierarchical long context from useful + noise documents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dir", "-d",
        type=str,
        required=True,
        help="Directory containing task folders (001, 002, etc.)"
    )
    parser.add_argument(
        "--tokens", "-t",
        type=int,
        nargs="+",
        default=[32000, 64000, 128000, 256000, 512000],
        help="Target token counts (default: 32000 64000 128000 256000 512000)"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="o200k_base",
        choices=["o200k_base", "cl100k_base"],
        help="Tiktoken encoding (default: o200k_base for GPT-4o)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output exists"
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Run embedding preprocessing after building contexts"
    )
    parser.add_argument(
        "--embed-pattern",
        type=str,
        default="long_context_sampled_*.json",
        help="Pattern for embedding preprocessing (default: long_context_sampled_*.json)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific task IDs to process (e.g., --tasks 001 002 003). If not specified, process all tasks."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Tolerance for target token count (default: 0.02 = 2%%). If available tokens are within tolerance of target, generate the file."
    )
    parser.add_argument(
        "--use-jsonl",
        action="store_true",
        default=True,
        help="Use JSONL files (useful_search.jsonl, noise.jsonl) instead of JSON files"
    )
    
    args = parser.parse_args()
    
    # Find task folders
    if not os.path.isdir(args.dir):
        logger.error(f"Directory not found: {args.dir}")
        sys.exit(1)
    
    task_folders = find_task_folders(args.dir, args.tasks)
    if not task_folders:
        logger.warning(f"No task folders found in: {args.dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(task_folders)} task folders")
    logger.info(f"Target token counts: {sorted(args.tokens)} (independent sampling)")
    
    # Process each folder
    all_results = []
    for task_folder in task_folders:
        logger.info(f"\nProcessing: {task_folder}")
        if args.use_jsonl:
            results = build_independent_contexts_jsonl(
                task_folder=task_folder,
                target_tokens_list=args.tokens,
                encoding_name=args.encoding,
                seed=args.seed,
                force=args.force,
                tolerance=args.tolerance
            )
        else:
            results = build_independent_contexts(
                task_folder=task_folder,
                target_tokens_list=args.tokens,
                encoding_name=args.encoding,
                seed=args.seed,
                force=args.force
            )
        all_results.extend(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BUILD SUMMARY")
    print("=" * 80)
    
    success = sum(1 for r in all_results if r["status"] == "success")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")
    insufficient = sum(1 for r in all_results if r["status"] == "insufficient_tokens")
    errors = sum(1 for r in all_results if r["status"] == "error")
    
    print(f"Total: {len(all_results)}, Success: {success}, Skipped: {skipped}, Insufficient: {insufficient}, Errors: {errors}")
    
    if success > 0:
        print("\nCreated files:")
        # Group by folder
        folders = {}
        for r in all_results:
            if r["status"] == "success":
                folder = r["folder"]
                if folder not in folders:
                    folders[folder] = []
                folders[folder].append(r)
        
        for folder, results in sorted(folders.items()):
            print(f"\n  {folder}:")
            for r in sorted(results, key=lambda x: x["target_tokens"]):
                token_k = r["target_tokens"] // 1000
                print(f"    {token_k:>4}k: {r['total_tokens']:>8,} tokens ({r['total_docs']:>3} docs)")
    
    if insufficient > 0:
        print("\nInsufficient tokens (skipped):")
        for r in all_results:
            if r["status"] == "insufficient_tokens":
                token_k = r["target_tokens"] // 1000
                print(f"  {r['folder']} [{token_k}k]: {r.get('error', 'Unknown')}")
    
    if errors > 0:
        print("\nErrors:")
        for r in all_results:
            if r["status"] == "error":
                print(f"  {r['folder']}: {r.get('error', 'Unknown error')}")
    
    print("=" * 80)
    
    # Run embedding preprocessing if requested
    if args.embed:
        run_embedding_preprocessing(
            directory=args.dir,
            pattern=args.embed_pattern,
            force=args.force,
            tasks=args.tasks
        )


if __name__ == "__main__":
    main()
