#!/usr/bin/env python3
"""
Sample Long Context by Token Count

This script samples documents from long_context.json to create subsets
with specific token counts (e.g., 32k, 64k tokens).

Usage:
    # Sample to 32k tokens
    python sample_by_tokens.py --file data/bench_case1114/005/long_context.json --tokens 32000
    
    # Sample to 64k tokens
    python sample_by_tokens.py --file data/bench_case1114/005/long_context.json --tokens 64000
    
    # Process all files in a directory
    python sample_by_tokens.py --dir data/bench_case1114 --tokens 32000 64000
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

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
    # Disable special token check to handle text containing special tokens
    return len(enc.encode(text, disallowed_special=()))


def count_json_tokens(data: list, encoding_name: str = "o200k_base") -> int:
    """Count tokens in JSON data."""
    json_str = json.dumps(data, ensure_ascii=False)
    return count_tokens(json_str, encoding_name)


def sample_documents_by_tokens(
    documents: List[dict],
    target_tokens: int,
    encoding_name: str = "o200k_base",
    seed: int = 42
) -> Tuple[List[dict], int]:
    """
    Sample documents to reach approximately the target token count.
    
    Strategy:
    1. Shuffle documents randomly
    2. Add documents one by one until we reach or exceed target tokens
    3. Return the sampled documents
    
    Args:
        documents: List of documents
        target_tokens: Target token count
        encoding_name: Tiktoken encoding name
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_documents, actual_token_count)
    """
    # Set random seed
    random.seed(seed)
    
    # Shuffle documents
    shuffled_docs = documents.copy()
    random.shuffle(shuffled_docs)
    
    # Calculate tokens for each document
    enc = tiktoken.get_encoding(encoding_name)
    doc_tokens = []
    for doc in shuffled_docs:
        doc_str = json.dumps(doc, ensure_ascii=False)
        tokens = len(enc.encode(doc_str, disallowed_special=()))
        doc_tokens.append((doc, tokens))
    
    # Sort by token count (smaller first) to pack more documents
    doc_tokens.sort(key=lambda x: x[1])
    
    # Greedily select documents
    sampled = []
    current_tokens = 0
    overhead = 10  # JSON array overhead: [ ] and commas
    
    for doc, tokens in doc_tokens:
        # Check if adding this document would exceed target
        new_total = current_tokens + tokens + 2  # +2 for comma and space
        if new_total > target_tokens:
            # If we haven't added any documents yet, add at least one
            if not sampled:
                sampled.append(doc)
                current_tokens = new_total
            break
        sampled.append(doc)
        current_tokens = new_total
    
    # Verify actual token count
    actual_tokens = count_json_tokens(sampled, encoding_name)
    
    return sampled, actual_tokens


def sample_documents_by_tokens_proportional(
    documents: List[dict],
    target_tokens: int,
    encoding_name: str = "o200k_base",
    seed: int = 42
) -> Tuple[List[dict], int]:
    """
    Sample documents to reach target token count as closely as possible.
    
    Strategy:
    1. Calculate tokens for each document
    2. Shuffle documents randomly
    3. Greedily add documents until we reach or slightly exceed target
    
    Args:
        documents: List of documents
        target_tokens: Target token count
        encoding_name: Tiktoken encoding name
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_documents, actual_token_count)
    """
    # Set random seed
    random.seed(seed)
    
    # Calculate total tokens first
    total_tokens = count_json_tokens(documents, encoding_name)
    
    if total_tokens <= target_tokens:
        logger.info(f"  Total tokens ({total_tokens}) <= target ({target_tokens}), using all documents")
        return documents, total_tokens
    
    logger.info(f"  Total tokens: {total_tokens}, target: {target_tokens}")
    
    # Calculate tokens for each document
    enc = tiktoken.get_encoding(encoding_name)
    doc_with_tokens = []
    for doc in documents:
        doc_str = json.dumps(doc, ensure_ascii=False)
        tokens = len(enc.encode(doc_str, disallowed_special=()))
        doc_with_tokens.append((doc, tokens))
    
    # Shuffle documents randomly
    random.shuffle(doc_with_tokens)
    
    # Greedily add documents until we reach target
    sampled = []
    current_tokens = 2  # Start with 2 for JSON array brackets []
    
    for doc, tokens in doc_with_tokens:
        # Calculate new total (add comma and space for each doc after first)
        overhead = 2 if sampled else 0  # ", " between documents
        new_total = current_tokens + tokens + overhead
        
        if new_total > target_tokens:
            # Check if adding this doc gets us closer to target
            diff_without = abs(target_tokens - current_tokens)
            diff_with = abs(target_tokens - new_total)
            
            if diff_with < diff_without:
                # Adding this doc gets us closer to target
                sampled.append(doc)
                current_tokens = new_total
            # Either way, we're done
            break
        
        sampled.append(doc)
        current_tokens = new_total
    
    # Verify actual token count
    actual_tokens = count_json_tokens(sampled, encoding_name)
    
    logger.info(f"  Sampling ratio: {len(sampled)/len(documents):.2%}, doc count: {len(sampled)}")
    
    return sampled, actual_tokens


def find_long_context_files(directory: str) -> List[str]:
    """Recursively find all long_context.json files in a directory."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename == "long_context.json":
                files.append(os.path.join(root, filename))
    return sorted(files)


def process_file(
    json_path: str,
    target_tokens: int,
    encoding_name: str = "o200k_base",
    seed: int = 42,
    force: bool = False
) -> dict:
    """
    Process a single long_context.json file.
    
    Args:
        json_path: Path to the long_context.json file
        target_tokens: Target token count
        encoding_name: Tiktoken encoding name
        seed: Random seed
        force: Force reprocessing even if output exists
        
    Returns:
        Dictionary with processing results
    """
    result = {
        "file": json_path,
        "target_tokens": target_tokens,
        "status": "unknown",
        "original_tokens": 0,
        "sampled_tokens": 0,
        "original_docs": 0,
        "sampled_docs": 0,
        "output_file": None,
        "error": None
    }
    
    try:
        # Generate output filename
        token_k = target_tokens // 1000
        output_path = json_path.replace(".json", f"_sampled_{token_k}k.json")
        result["output_file"] = output_path
        
        # Check if output already exists
        if not force and os.path.exists(output_path):
            logger.info(f"  Output already exists: {output_path}")
            result["status"] = "skipped"
            return result
        
        # Load documents
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict) and 'documents' in data:
            documents = data['documents']
        else:
            raise ValueError("Invalid long_context.json format")
        
        result["original_docs"] = len(documents)
        result["original_tokens"] = count_json_tokens(documents, encoding_name)
        
        logger.info(f"  Original: {result['original_docs']} docs, {result['original_tokens']} tokens")
        
        # Sample documents
        sampled, actual_tokens = sample_documents_by_tokens_proportional(
            documents, target_tokens, encoding_name, seed
        )
        
        result["sampled_docs"] = len(sampled)
        result["sampled_tokens"] = actual_tokens
        
        logger.info(f"  Sampled: {result['sampled_docs']} docs, {result['sampled_tokens']} tokens")
        
        # Save sampled documents
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sampled, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  ✓ Saved to: {output_path}")
        result["status"] = "success"
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Sample long_context.json by token count",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a single long_context.json file"
    )
    input_group.add_argument(
        "--dir", "-d",
        type=str,
        help="Directory to search for long_context.json files"
    )
    
    # Token options
    parser.add_argument(
        "--tokens", "-t",
        type=int,
        nargs="+",
        required=True,
        help="Target token count(s), e.g., 32000 64000"
    )
    
    # Other options
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
    
    args = parser.parse_args()
    
    # Get files to process
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        files = [args.file]
    else:
        if not os.path.isdir(args.dir):
            logger.error(f"Directory not found: {args.dir}")
            sys.exit(1)
        files = find_long_context_files(args.dir)
        if not files:
            logger.warning(f"No long_context.json files found in: {args.dir}")
            sys.exit(0)
    
    logger.info(f"Found {len(files)} file(s) to process")
    logger.info(f"Target token counts: {args.tokens}")
    logger.info(f"Encoding: {args.encoding}")
    
    # Process each file for each token target
    all_results = []
    for json_path in files:
        for target_tokens in args.tokens:
            logger.info(f"\nProcessing: {json_path} -> {target_tokens} tokens")
            result = process_file(
                json_path=json_path,
                target_tokens=target_tokens,
                encoding_name=args.encoding,
                seed=args.seed,
                force=args.force
            )
            all_results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success = sum(1 for r in all_results if r["status"] == "success")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")
    errors = sum(1 for r in all_results if r["status"] == "error")
    
    print(f"Total: {len(all_results)}, Success: {success}, Skipped: {skipped}, Errors: {errors}")
    
    if success > 0:
        print("\nCreated files:")
        for r in all_results:
            if r["status"] == "success":
                print(f"  {r['output_file']}")
                print(f"    {r['original_docs']} docs ({r['original_tokens']} tokens) -> {r['sampled_docs']} docs ({r['sampled_tokens']} tokens)")
    
    if errors > 0:
        print("\nErrors:")
        for r in all_results:
            if r["status"] == "error":
                print(f"  {r['file']}: {r['error']}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
