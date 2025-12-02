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
Long Context Reader MCP Server

Provides long context reading and comprehension functionality as MCP tools.
Unlike RAG which retrieves specific chunks, this tool reads and comprehends
the entire document collection, extracting and synthesizing key information.

Features:
- Parallel chunk processing for efficiency
- Intelligent information extraction and filtering
- Hierarchical summarization for long documents
- Question-aware reading for targeted comprehension
- SQLite-based caching for processed results
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import asdict
from mcp.server.fastmcp import FastMCP

from ..tools.long_context_reader import LongContextReader, FullContextReading

logger = logging.getLogger(__name__)

# Global reader instance
_reader_instance: Optional[LongContextReader] = None
_current_json_path: Optional[str] = None

# Global reading log
_reading_log_path: Optional[str] = None
_reading_results: List[Dict[str, Any]] = []
_read_counter: int = 0


def set_reading_log_path(log_path: str):
    """Set the path for saving reading results."""
    global _reading_log_path, _reading_results, _read_counter
    _reading_log_path = log_path
    _reading_results = []
    _read_counter = 0
    print(f"[LongContextReader] Reading log path set to: {log_path}")


def _save_reading_result(question: str, result: FullContextReading, tool_name: str, json_path: str):
    """Save reading result to the log file."""
    global _reading_log_path, _reading_results
    
    if not _reading_log_path:
        return
    
    # Build result entry (simplified for logging)
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "question": question,
        "source_file": json_path,
        "total_documents": result.total_documents,
        "total_chunks_processed": result.total_chunks_processed,
        "confidence_score": result.confidence_score,
        "compression_ratio": result.compression_ratio,
        "key_findings": result.key_findings[:10],
        "synthesized_understanding_preview": result.synthesized_understanding[:500] + "..." if len(result.synthesized_understanding) > 500 else result.synthesized_understanding,
        "processing_stats": result.processing_stats
    }
    
    _reading_results.append(result_entry)
    
    # Save to file
    try:
        os.makedirs(os.path.dirname(_reading_log_path), exist_ok=True)
        with open(_reading_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_reads": len(_reading_results),
                "reads": _reading_results
            }, f, ensure_ascii=False, indent=2)
        print(f"[LongContextReader] Saved reading results to: {_reading_log_path}")
    except Exception as e:
        print(f"[LongContextReader] Warning: Failed to save reading log: {e}")


mcp = FastMCP("long_context_reader")


def _get_reader_instance(json_path: str) -> LongContextReader:
    """Get or create reader instance for the given JSON path."""
    global _reader_instance, _current_json_path
    
    if _reader_instance is None or _current_json_path != json_path:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("READER_MODEL", "gpt-4o-mini")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        print(f"[LongContextReader] Initializing reader for: {json_path}")
        print(f"[LongContextReader] Using model: {model}")
        
        _reader_instance = LongContextReader(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        doc_count = _reader_instance.load_documents(json_path)
        stats = _reader_instance.get_stats()
        
        print(f"[LongContextReader] Loaded {doc_count} documents")
        print(f"[LongContextReader] Total characters: {stats.get('total_characters', 0):,}")
        print(f"[LongContextReader] Estimated chunks: {stats.get('estimated_chunks', 0)}")
        print(f"[LongContextReader] Cache path: {_reader_instance.db_path}")
        
        _current_json_path = json_path
    else:
        print(f"[LongContextReader] Using cached instance for: {json_path}")
    
    return _reader_instance


@mcp.tool()
def read_long_context(
    json_path: str,
    question: str = "",
    max_docs: int = None
) -> str:
    """
    Read and comprehend the entire long context document collection.
    
    This tool reads ALL documents in the long_context.json file, extracts key information
    from each chunk, and synthesizes a comprehensive understanding. Unlike RAG which
    retrieves specific chunks based on similarity, this tool provides a complete
    understanding of the entire document collection.
    
    Use this tool when you need to:
    - Understand the overall content of a large document collection
    - Extract all key facts and entities from documents
    - Get a comprehensive summary before diving into details
    - Answer questions that require understanding the full context
    
    Args:
        json_path: Path to the long_context.json file containing the documents
        question: Optional question or focus area for targeted reading. If provided,
                  the reader will prioritize information relevant to this question.
        max_docs: Maximum number of documents to process (None = all documents)
        
    Returns:
        Comprehensive understanding including:
        - Synthesized understanding of all documents
        - Key findings prioritized by relevance
        - Document summaries with key facts and entities
        - Confidence score and compression statistics
    """
    global _read_counter
    
    try:
        _read_counter += 1
        
        print(f"\n{'='*60}")
        print(f"[LongContextReader] Read #{_read_counter}")
        print(f"[LongContextReader] Question: '{question[:100]}...' " if question else "[LongContextReader] No specific question - full comprehension mode")
        print(f"[LongContextReader] Max docs: {max_docs if max_docs else 'All'}")
        print(f"{'='*60}")
        
        if not os.path.exists(json_path):
            return f"Error: File not found: {json_path}"
        
        reader = _get_reader_instance(json_path)
        result = reader.read_full_context(question=question, max_docs=max_docs)
        
        print(f"[LongContextReader] Processed {result.total_documents} documents, {result.total_chunks_processed} chunks")
        print(f"[LongContextReader] Confidence: {result.confidence_score}%")
        print(f"[LongContextReader] Compression ratio: {result.compression_ratio:.2%}")
        
        # Save reading results to log file
        _save_reading_result(question, result, "read_long_context", json_path)
        
        # Format output
        output_parts = [
            f"=== Long Context Reading Results ===",
            f"Question/Focus: '{question}'" if question else "Mode: Full Comprehension",
            f"Source File: {json_path}",
            f"Documents Processed: {result.total_documents}",
            f"Total Chunks: {result.total_chunks_processed}",
            f"Confidence Score: {result.confidence_score}%",
            f"Compression Ratio: {result.processing_stats.get('compression_ratio', 'N/A')}",
            "",
            "=" * 60,
            "SYNTHESIZED UNDERSTANDING",
            "=" * 60,
            result.synthesized_understanding,
            "",
            "=" * 60,
            "KEY FINDINGS",
            "=" * 60,
        ]
        
        for i, finding in enumerate(result.key_findings, 1):
            output_parts.append(f"{i}. {finding}")
        
        if result.relevant_quotes:
            output_parts.append("")
            output_parts.append("=" * 60)
            output_parts.append("RELEVANT EVIDENCE/QUOTES")
            output_parts.append("=" * 60)
            for i, quote in enumerate(result.relevant_quotes[:5], 1):
                output_parts.append(f"{i}. {quote}")
        
        # Add document summaries
        output_parts.append("")
        output_parts.append("=" * 60)
        output_parts.append("DOCUMENT SUMMARIES")
        output_parts.append("=" * 60)
        
        for dr in result.document_readings:
            avg_relevance = sum(cr.relevance_score for cr in dr.chunk_readings) / len(dr.chunk_readings)
            output_parts.append(f"\n--- Document {dr.doc_index + 1}: {dr.title} ---")
            output_parts.append(f"URL: {dr.url}")
            output_parts.append(f"Relevance: {avg_relevance:.1f}/10")
            output_parts.append(f"Summary: {dr.overall_summary}")
            if dr.key_facts:
                output_parts.append(f"Key Facts: {', '.join(dr.key_facts[:5])}")
        
        output_parts.append("")
        output_parts.append("=" * 60)
        output_parts.append("USAGE NOTE")
        output_parts.append("=" * 60)
        output_parts.append("This is a comprehensive reading of the entire document collection.")
        output_parts.append("For specific information retrieval, consider using the RAG tool (rag_search).")
        output_parts.append("=" * 60)
        
        return "\n".join(output_parts)
        
    except Exception as e:
        logger.exception(f"Error during long context reading: {e}")
        return f"Error during long context reading: {str(e)}"


@mcp.tool()
def get_document_overview(
    json_path: str
) -> str:
    """
    Get a quick overview of the long context document collection without full reading.
    
    This tool provides statistics and sample information about the documents
    without performing the full reading process. Use this to understand the
    scope of the document collection before deciding to do a full read.
    
    Args:
        json_path: Path to the long_context.json file
        
    Returns:
        Overview statistics including document count, total size, and sample titles
    """
    try:
        if not os.path.exists(json_path):
            return f"Error: File not found: {json_path}"
        
        reader = _get_reader_instance(json_path)
        stats = reader.get_stats()
        
        output_parts = [
            "=== Long Context Document Overview ===",
            f"Source File: {json_path}",
            f"Total Documents: {stats.get('total_documents', 0)}",
            f"Total Characters: {stats.get('total_characters', 0):,}",
            f"Estimated Chunks: {stats.get('estimated_chunks', 0)}",
            f"Chunk Size: {stats.get('chunk_size', 0)} chars",
            f"Parallel Workers: {stats.get('max_workers', 0)}",
            f"Reading Model: {stats.get('model', 'unknown')}",
            f"Cache Path: {stats.get('cache_path', 'none')}",
            "",
            "Sample Document Titles:",
        ]
        
        for i, title in enumerate(stats.get('sample_titles', []), 1):
            output_parts.append(f"  {i}. {title}")
        
        output_parts.append("")
        output_parts.append("To read and comprehend the full context, use: read_long_context")
        output_parts.append("For specific information retrieval, use: rag_search")
        
        return "\n".join(output_parts)
        
    except Exception as e:
        logger.exception(f"Error getting document overview: {e}")
        return f"Error getting document overview: {str(e)}"


@mcp.tool()
def read_single_document(
    json_path: str,
    doc_index: int,
    question: str = ""
) -> str:
    """
    Read and comprehend a single document from the collection.
    
    Use this when you want to focus on understanding one specific document
    rather than the entire collection.
    
    Args:
        json_path: Path to the long_context.json file
        doc_index: Index of the document to read (0-based)
        question: Optional question or focus area for targeted reading
        
    Returns:
        Comprehensive understanding of the single document
    """
    try:
        if not os.path.exists(json_path):
            return f"Error: File not found: {json_path}"
        
        reader = _get_reader_instance(json_path)
        
        if doc_index < 0 or doc_index >= len(reader.documents):
            return f"Error: Invalid document index {doc_index}. Valid range: 0-{len(reader.documents)-1}"
        
        # Read just this one document
        result = reader.read_full_context(question=question, max_docs=doc_index + 1)
        
        # Get the specific document reading
        if doc_index < len(result.document_readings):
            dr = result.document_readings[doc_index]
        else:
            return f"Error: Could not read document at index {doc_index}"
        
        output_parts = [
            f"=== Single Document Reading ===",
            f"Document Index: {doc_index}",
            f"Title: {dr.title}",
            f"URL: {dr.url}",
            f"Chunks Processed: {len(dr.chunk_readings)}",
            f"Original Length: {dr.total_original_length:,} chars",
            f"Compressed Length: {dr.total_compressed_length:,} chars",
            "",
            "=" * 50,
            "DOCUMENT SUMMARY",
            "=" * 50,
            dr.overall_summary,
            "",
            "=" * 50,
            "KEY FACTS",
            "=" * 50,
        ]
        
        for i, fact in enumerate(dr.key_facts, 1):
            output_parts.append(f"{i}. {fact}")
        
        output_parts.append("")
        output_parts.append("=" * 50)
        output_parts.append("KEY ENTITIES")
        output_parts.append("=" * 50)
        output_parts.append(", ".join(dr.key_entities))
        
        # Add chunk details
        output_parts.append("")
        output_parts.append("=" * 50)
        output_parts.append("CHUNK DETAILS")
        output_parts.append("=" * 50)
        
        for cr in dr.chunk_readings:
            output_parts.append(f"\nChunk {cr.chunk_index + 1}:")
            output_parts.append(f"  Relevance: {cr.relevance_score}/10")
            output_parts.append(f"  Summary: {cr.summary[:200]}...")
        
        return "\n".join(output_parts)
        
    except Exception as e:
        logger.exception(f"Error reading single document: {e}")
        return f"Error reading single document: {str(e)}"


if __name__ == "__main__":
    mcp.run()
