#!/usr/bin/env python3
"""
Sample Long Context by Token Count with Required Documents

This script samples documents from long_context.json to create subsets
with specific token counts, while ensuring that certain required documents
are always included.

Usage:
    # Sample to 64k tokens with required documents
    python sample_with_required_docs.py --file datasets1214/002/long_context.json --tokens 64000 --required-file datasets1214/002/required_docs.json
    
    # Sample to multiple token counts
    python sample_with_required_docs.py --file datasets1214/002/long_context.json --tokens 64000 128000 --required-titles "Doc Title 1" "Doc Title 2"
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Set

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


def find_required_documents(
    documents: List[dict],
    required_titles: List[str]
) -> Tuple[List[dict], List[dict], List[str]]:
    """
    Find required documents by matching titles.
    
    Args:
        documents: List of all documents
        required_titles: List of required document titles (can be partial matches)
        
    Returns:
        Tuple of (required_docs, other_docs, missing_titles)
    """
    required_docs = []
    other_docs = []
    found_titles = set()
    
    for doc in documents:
        doc_title = doc.get('title', '')
        is_required = False
        
        for req_title in required_titles:
            # Check for partial match (required title is substring of doc title or vice versa)
            if req_title in doc_title or doc_title in req_title:
                is_required = True
                found_titles.add(req_title)
                break
        
        if is_required:
            required_docs.append(doc)
        else:
            other_docs.append(doc)
    
    # Find missing titles
    missing_titles = [t for t in required_titles if t not in found_titles]
    
    return required_docs, other_docs, missing_titles


def sample_documents_with_required(
    documents: List[dict],
    required_titles: List[str],
    target_tokens: int,
    encoding_name: str = "o200k_base",
    seed: int = 42
) -> Tuple[List[dict], int, List[str]]:
    """
    Sample documents to reach target token count, ensuring required documents are included.
    
    Strategy:
    1. First, include all required documents
    2. Calculate remaining token budget
    3. Randomly sample from other documents to fill the budget
    
    Args:
        documents: List of all documents
        required_titles: List of required document titles
        target_tokens: Target token count
        encoding_name: Tiktoken encoding name
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_documents, actual_token_count, missing_titles)
    """
    # Set random seed
    random.seed(seed)
    
    # Find required documents
    required_docs, other_docs, missing_titles = find_required_documents(documents, required_titles)
    
    logger.info(f"  Found {len(required_docs)} required documents out of {len(required_titles)} requested")
    if missing_titles:
        logger.warning(f"  Missing required documents: {missing_titles}")
    
    # Calculate tokens for required documents
    enc = tiktoken.get_encoding(encoding_name)
    required_tokens = 0
    for doc in required_docs:
        doc_str = json.dumps(doc, ensure_ascii=False)
        required_tokens += len(enc.encode(doc_str, disallowed_special=()))
    
    logger.info(f"  Required documents: {len(required_docs)} docs, ~{required_tokens} tokens")
    
    # Check if required documents already exceed target
    if required_tokens >= target_tokens:
        logger.warning(f"  Required documents ({required_tokens} tokens) exceed target ({target_tokens} tokens)")
        actual_tokens = count_json_tokens(required_docs, encoding_name)
        return required_docs, actual_tokens, missing_titles
    
    # Calculate remaining budget
    remaining_budget = target_tokens - required_tokens - 10  # 10 for JSON overhead
    logger.info(f"  Remaining budget for other documents: ~{remaining_budget} tokens")
    
    # Calculate tokens for each other document
    doc_with_tokens = []
    for doc in other_docs:
        doc_str = json.dumps(doc, ensure_ascii=False)
        tokens = len(enc.encode(doc_str, disallowed_special=()))
        doc_with_tokens.append((doc, tokens))
    
    # Shuffle other documents randomly
    random.shuffle(doc_with_tokens)
    
    # Greedily add documents until we reach target
    sampled = list(required_docs)  # Start with required documents
    current_tokens = required_tokens + 2  # +2 for JSON array brackets
    
    for doc, tokens in doc_with_tokens:
        # Calculate new total
        overhead = 2 if len(sampled) > 0 else 0  # ", " between documents
        new_total = current_tokens + tokens + overhead
        
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
    
    # Verify actual token count
    actual_tokens = count_json_tokens(sampled, encoding_name)
    
    logger.info(f"  Final: {len(sampled)} docs, {actual_tokens} tokens")
    logger.info(f"  Sampling ratio: {len(sampled)/len(documents):.2%}")
    
    return sampled, actual_tokens, missing_titles


def process_file(
    json_path: str,
    required_titles: List[str],
    target_tokens: int,
    encoding_name: str = "o200k_base",
    seed: int = 42,
    force: bool = False
) -> dict:
    """
    Process a single long_context.json file.
    
    Args:
        json_path: Path to the long_context.json file
        required_titles: List of required document titles
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
        "required_docs_found": 0,
        "required_docs_missing": [],
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
        
        # Sample documents with required ones
        sampled, actual_tokens, missing_titles = sample_documents_with_required(
            documents, required_titles, target_tokens, encoding_name, seed
        )
        
        result["sampled_docs"] = len(sampled)
        result["sampled_tokens"] = actual_tokens
        result["required_docs_found"] = len(required_titles) - len(missing_titles)
        result["required_docs_missing"] = missing_titles
        
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
        description="Sample long_context.json by token count with required documents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Path to the long_context.json file"
    )
    
    # Token options
    parser.add_argument(
        "--tokens", "-t",
        type=int,
        nargs="+",
        required=True,
        help="Target token count(s), e.g., 64000 128000"
    )
    
    # Required documents options
    required_group = parser.add_mutually_exclusive_group(required=True)
    required_group.add_argument(
        "--required-titles",
        type=str,
        nargs="+",
        help="List of required document titles"
    )
    required_group.add_argument(
        "--required-file",
        type=str,
        help="JSON file containing required document titles (format: {\"gold_insights\": [{\"insight\": \"title1\"}, ...]})"
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
    
    # Check file exists
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
    
    # Get required titles
    if args.required_titles:
        required_titles = args.required_titles
    else:
        # Load from file
        with open(args.required_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'gold_insights' in data:
            required_titles = [item['insight'] for item in data['gold_insights']]
        elif isinstance(data, list):
            required_titles = data
        else:
            logger.error("Invalid required file format")
            sys.exit(1)
    
    logger.info(f"Required documents ({len(required_titles)}):")
    for title in required_titles:
        logger.info(f"  - {title}")
    
    logger.info(f"\nTarget token counts: {args.tokens}")
    logger.info(f"Encoding: {args.encoding}")
    
    # Process file for each token target
    all_results = []
    for target_tokens in args.tokens:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {args.file} -> {target_tokens} tokens")
        logger.info(f"{'='*60}")
        result = process_file(
            json_path=args.file,
            required_titles=required_titles,
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
    
    for r in all_results:
        print(f"\n{r['output_file']}:")
        print(f"  Status: {r['status']}")
        if r['status'] == 'success':
            print(f"  Documents: {r['original_docs']} -> {r['sampled_docs']}")
            print(f"  Tokens: {r['original_tokens']} -> {r['sampled_tokens']}")
            print(f"  Required docs found: {r['required_docs_found']}/{len(required_titles)}")
            if r['required_docs_missing']:
                print(f"  Missing: {r['required_docs_missing']}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
