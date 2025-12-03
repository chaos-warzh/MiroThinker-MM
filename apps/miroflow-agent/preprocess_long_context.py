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
Preprocess Long Context Documents

This script preprocesses long_context.json files by:
1. Optionally sampling a subset of documents (for large files)
2. Splitting documents into chunks
3. Generating embeddings for each chunk
4. Saving chunks and embeddings to SQLite database (.chunks.db)
5. Optionally saving the sampled documents to a new JSON file

This preprocessing step allows for faster RAG queries during runtime,
as the embeddings are cached and don't need to be regenerated.

Usage:
    # Process a single file
    python preprocess_long_context.py --file data/001/long_context.json
    
    # Process all long_context.json files in a directory (recursive)
    python preprocess_long_context.py --dir data/bench_case1104
    
    # Force reprocessing even if cache exists
    python preprocess_long_context.py --dir data/bench_case1104 --force
    
    # Sample 60 random documents from each file
    python preprocess_long_context.py --dir data/bench_case1114 --sample 60 --force

Environment Variables:
    OPENAI_API_KEY: Required. OpenAI API key for generating embeddings.
    OPENAI_BASE_URL: Optional. OpenAI API base URL (default: https://api.openai.com/v1)
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add the miroflow-tools library to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "miroflow-tools" / "src"))

from dotenv import load_dotenv
from miroflow_tools.tools.rag_tool import RAGTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_long_context_files(directory: str) -> List[str]:
    """
    Recursively find all long_context.json files in a directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of paths to long_context.json files
    """
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename == "long_context.json":
                files.append(os.path.join(root, filename))
    return sorted(files)


def check_cache_exists(json_path: str) -> bool:
    """
    Check if a valid cache database exists for the given JSON file.
    
    Args:
        json_path: Path to the long_context.json file
        
    Returns:
        True if cache exists, False otherwise
    """
    db_path = json_path + ".chunks.db"
    return os.path.exists(db_path)


def sample_documents(
    json_path: str,
    sample_size: int,
    seed: int = 42
) -> tuple:
    """
    Sample a subset of documents from a long_context.json file.
    
    Args:
        json_path: Path to the long_context.json file
        sample_size: Number of documents to sample
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_documents, original_count)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        documents = data
    elif isinstance(data, dict) and 'documents' in data:
        documents = data['documents']
    else:
        raise ValueError("Invalid long_context.json format")
    
    original_count = len(documents)
    
    if original_count <= sample_size:
        logger.info(f"  Document count ({original_count}) <= sample size ({sample_size}), using all documents")
        return documents, original_count
    
    # Set random seed for reproducibility
    random.seed(seed)
    sampled = random.sample(documents, sample_size)
    logger.info(f"  Sampled {sample_size} documents from {original_count} total")
    
    return sampled, original_count


def save_sampled_documents(
    sampled_docs: List[dict],
    output_path: str
) -> str:
    """
    Save sampled documents to a new JSON file.
    
    Args:
        sampled_docs: List of sampled documents
        output_path: Path to save the sampled documents
        
    Returns:
        Path to the saved file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_docs, f, ensure_ascii=False, indent=2)
    
    logger.info(f"  ✓ Saved sampled documents to: {output_path}")
    return output_path


def preprocess_file(
    json_path: str,
    api_key: str,
    base_url: str,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = None,
    chunk_overlap: int = None,
    force: bool = False,
    sample_size: int = None,
    seed: int = 42
) -> dict:
    """
    Preprocess a single long_context.json file.
    
    Args:
        json_path: Path to the long_context.json file
        api_key: OpenAI API key
        base_url: OpenAI API base URL
        embedding_model: Embedding model to use
        chunk_size: Maximum characters per chunk (default: 1500)
        chunk_overlap: Overlap between chunks (default: 200)
        force: Force reprocessing even if cache exists
        sample_size: Number of documents to randomly sample (None = use all)
        seed: Random seed for sampling reproducibility
        
    Returns:
        Dictionary with processing results
    """
    result = {
        "file": json_path,
        "status": "unknown",
        "chunks": 0,
        "documents": 0,
        "original_documents": 0,
        "cache_path": None,
        "sampled_file": None,
        "error": None,
        "skipped": False
    }
    
    try:
        # Determine the file to process (original or sampled)
        file_to_process = json_path
        
        # If sampling is requested, create a sampled version first
        if sample_size is not None:
            logger.info(f"Sampling {sample_size} documents from: {json_path}")
            sampled_docs, original_count = sample_documents(json_path, sample_size, seed)
            result["original_documents"] = original_count
            
            # Save sampled documents to a new file
            sampled_path = json_path.replace(".json", f"_sampled_{sample_size}.json")
            save_sampled_documents(sampled_docs, sampled_path)
            result["sampled_file"] = sampled_path
            file_to_process = sampled_path
        
        # Check if cache already exists for the file to process
        if not force and check_cache_exists(file_to_process):
            logger.info(f"Cache already exists for: {file_to_process}")
            result["status"] = "skipped"
            result["skipped"] = True
            result["cache_path"] = file_to_process + ".chunks.db"
            return result
        
        logger.info(f"Processing: {file_to_process}")
        
        # Initialize RAG tool with same parameters as production
        rag = RAGTool(
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Load documents - this will:
        # 1. Parse the JSON file
        # 2. Split documents into chunks
        # 3. Generate embeddings for each chunk
        # 4. Save to SQLite database
        chunk_count = rag.load_documents(file_to_process)
        
        # Get statistics
        stats = rag.get_stats()
        
        result["status"] = "success"
        result["chunks"] = chunk_count
        result["documents"] = stats.get("total_documents", 0)
        result["cache_path"] = rag.db_path
        
        logger.info(f"  ✓ Created {chunk_count} chunks from {result['documents']} documents")
        logger.info(f"  ✓ Cache saved to: {rag.db_path}")
        
        # Close the database connection
        rag.close()
        
    except Exception as e:
        logger.error(f"  ✗ Error processing {json_path}: {e}")
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def preprocess_directory(
    directory: str,
    api_key: str,
    base_url: str,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = None,
    chunk_overlap: int = None,
    force: bool = False,
    sample_size: int = None,
    seed: int = 42
) -> List[dict]:
    """
    Preprocess all long_context.json files in a directory.
    
    Args:
        directory: Root directory to search
        api_key: OpenAI API key
        base_url: OpenAI API base URL
        embedding_model: Embedding model to use
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        force: Force reprocessing even if cache exists
        sample_size: Number of documents to randomly sample (None = use all)
        seed: Random seed for sampling reproducibility
        
    Returns:
        List of processing results
    """
    # Find all long_context.json files
    files = find_long_context_files(directory)
    
    if not files:
        logger.warning(f"No long_context.json files found in: {directory}")
        return []
    
    logger.info(f"Found {len(files)} long_context.json files to process")
    if sample_size:
        logger.info(f"Will sample {sample_size} documents from each file")
    
    results = []
    for i, json_path in enumerate(files, 1):
        logger.info(f"\n[{i}/{len(files)}] Processing: {json_path}")
        result = preprocess_file(
            json_path=json_path,
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force=force,
            sample_size=sample_size,
            seed=seed
        )
        results.append(result)
    
    return results


def print_summary(results: List[dict]):
    """Print a summary of processing results."""
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    
    total_chunks = sum(r["chunks"] for r in results if r["status"] == "success")
    total_docs = sum(r["documents"] for r in results if r["status"] == "success")
    total_original = sum(r.get("original_documents", 0) for r in results if r["status"] == "success")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files:     {total}")
    print(f"  ✓ Success:     {success}")
    print(f"  ○ Skipped:     {skipped}")
    print(f"  ✗ Errors:      {errors}")
    print(f"\nTotal documents processed: {total_docs}")
    if total_original > 0:
        print(f"Total original documents:  {total_original}")
    print(f"Total chunks:    {total_chunks}")
    
    # Show sampled files
    sampled_files = [r.get("sampled_file") for r in results if r.get("sampled_file")]
    if sampled_files:
        print("\nSampled files created:")
        for f in sampled_files:
            print(f"  - {f}")
    
    if errors > 0:
        print("\nErrors:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['file']}: {r['error']}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess long_context.json files for RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single file
    python preprocess_long_context.py --file data/001/long_context.json
    
    # Process all files in a directory
    python preprocess_long_context.py --dir data/bench_case1104
    
    # Force reprocessing
    python preprocess_long_context.py --dir data/bench_case1104 --force
    
    # Use custom embedding model
    python preprocess_long_context.py --dir data --model text-embedding-3-large
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a single long_context.json file to process"
    )
    input_group.add_argument(
        "--dir", "-d",
        type=str,
        help="Directory to search for long_context.json files (recursive)"
    )
    
    # Processing options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if cache exists"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use (default: text-embedding-3-small)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Maximum characters per chunk (default: 1500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap between chunks (default: 200)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of documents to randomly sample from each file (default: use all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility (default: 42)"
    )
    
    # API options
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: from OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI API base URL (default: from OPENAI_BASE_URL env var or https://api.openai.com/v1)"
    )
    
    # Environment file
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file (default: .env in current directory)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    env_file = args.env_file or os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"Loaded environment from: {env_file}")
    
    # Get API credentials
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        logger.error("OPENAI_API_KEY is required. Set it via --api-key or OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    logger.info(f"Using embedding model: {args.model}")
    logger.info(f"Using API base URL: {base_url}")
    
    start_time = datetime.now()
    
    # Process files
    if args.file:
        # Single file mode
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        
        result = preprocess_file(
            json_path=args.file,
            api_key=api_key,
            base_url=base_url,
            embedding_model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            force=args.force,
            sample_size=args.sample,
            seed=args.seed
        )
        results = [result]
    else:
        # Directory mode
        if not os.path.isdir(args.dir):
            logger.error(f"Directory not found: {args.dir}")
            sys.exit(1)
        
        results = preprocess_directory(
            directory=args.dir,
            api_key=api_key,
            base_url=base_url,
            embedding_model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            force=args.force,
            sample_size=args.sample,
            seed=args.seed
        )
    
    # Print summary
    print_summary(results)
    
    elapsed = datetime.now() - start_time
    logger.info(f"Total time: {elapsed}")
    
    # Exit with error code if any failures
    if any(r["status"] == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
