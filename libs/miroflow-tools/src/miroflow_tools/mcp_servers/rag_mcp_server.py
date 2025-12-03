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
RAG MCP Server

Provides RAG (Retrieval-Augmented Generation) functionality as MCP tools.
Uses OpenAI embeddings for semantic search over long context documents.

Features:
- SQLite-based embedding cache for fast subsequent queries
- Semantic search using OpenAI embeddings
- Support for long_context.json format
- Automatic logging of retrieval results to separate files
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP

from ..tools.rag_tool import RAGTool

logger = logging.getLogger(__name__)

# Global RAG instance
_rag_instance: Optional[RAGTool] = None
_current_json_path: Optional[str] = None

# Global retrieval log path
_retrieval_log_path: Optional[str] = None
_retrieval_results: List[Dict[str, Any]] = []
_query_counter: int = 0
_turn_counter: int = 0
_last_turn_query_count: int = 0


def set_retrieval_log_path(log_path: str):
    """Set the path for saving retrieval results."""
    global _retrieval_log_path, _retrieval_results, _query_counter, _turn_counter, _last_turn_query_count
    _retrieval_log_path = log_path
    _retrieval_results = []
    _query_counter = 0
    _turn_counter = 0
    _last_turn_query_count = 0
    print(f"[RAG] Retrieval log path set to: {log_path}")


def increment_turn():
    """Increment turn counter (called externally when a new turn starts)."""
    global _turn_counter, _last_turn_query_count
    _turn_counter += 1
    _last_turn_query_count = 0
    print(f"\n{'='*60}")
    print(f"[RAG] === Turn {_turn_counter} Started ===")
    print(f"{'='*60}")


def _save_retrieval_result(query: str, results: List[Any], tool_name: str, json_path: str):
    """Save retrieval result to the log file."""
    global _retrieval_log_path, _retrieval_results
    
    if not _retrieval_log_path:
        return
    
    # Build result entry
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "query": query,
        "source_file": json_path,
        "num_results": len(results),
        "results": []
    }
    
    for i, r in enumerate(results, 1):
        result_entry["results"].append({
            "rank": i,
            "citation_id": f"[RAG-{i}]" if tool_name == "rag_search" else f"[Context-{i}]",
            "score": round(r.score, 4),
            "doc_index": r.doc_index,
            "chunk_index": r.chunk_index,
            "title": r.title,
            "url": r.metadata.get('url', ''),
            "start_char": r.metadata.get('start_char', 0),
            "end_char": r.metadata.get('end_char', 0),
            "content_length": len(r.content),
            "content": r.content  # Full content for reference
        })
    
    _retrieval_results.append(result_entry)
    
    # Save to file
    try:
        os.makedirs(os.path.dirname(_retrieval_log_path), exist_ok=True)
        with open(_retrieval_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_queries": len(_retrieval_results),
                "queries": _retrieval_results
            }, f, ensure_ascii=False, indent=2)
        print(f"[RAG] Saved retrieval results to: {_retrieval_log_path}")
    except Exception as e:
        print(f"[RAG] Warning: Failed to save retrieval log: {e}")

mcp = FastMCP("rag")


def _get_rag_instance(json_path: str) -> RAGTool:
    """Get or create RAG instance for the given JSON path."""
    global _rag_instance, _current_json_path
    
    if _rag_instance is None or _current_json_path != json_path:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        print(f"[RAG] Initializing RAG tool for: {json_path}")
        _rag_instance = RAGTool(api_key=api_key, base_url=base_url)
        chunk_count = _rag_instance.load_documents(json_path)
        stats = _rag_instance.get_stats()
        print(f"[RAG] Loaded {stats.get('total_documents', 0)} documents, {chunk_count} chunks from {json_path}")
        print(f"[RAG] Chunk size: {stats.get('chunk_size', 0)}, Overlap: {stats.get('chunk_overlap', 0)}")
        print(f"[RAG] Cache path: {_rag_instance.db_path}")
        _current_json_path = json_path
    else:
        print(f"[RAG] Using cached instance for: {json_path}")
    
    return _rag_instance


@mcp.tool()
def rag_search(
    query: str,
    json_path: str,
    top_k: int = 10,
    diverse: bool = True,
    min_docs: int = 5,
    max_per_doc: int = 2
) -> str:
    """
    Search for relevant information in a long context document using semantic search.
    
    This tool uses RAG (Retrieval-Augmented Generation) with OpenAI embeddings
    to find the most relevant passages from a large document collection.
    Embeddings are cached in SQLite for fast subsequent queries.
    
    By default, uses diverse search to ensure results come from multiple different
    documents, not just the most similar chunks from a few documents.
    
    Args:
        query: The search query - what information you're looking for
        json_path: Path to the long_context.json file containing the documents
        top_k: Number of top results to return (default: 10)
        diverse: If True, ensure results come from different documents (default: True)
        min_docs: Minimum number of different documents to include when diverse=True (default: 5)
        max_per_doc: Maximum chunks per document when diverse=True (default: 2)
        
    Returns:
        Formatted search results with relevant passages and their sources
    """
    global _query_counter, _turn_counter, _last_turn_query_count
    try:
        _query_counter += 1
        _last_turn_query_count += 1
        
        # Enhanced logging with turn and query info
        print(f"\n{'='*60}")
        print(f"[RAG] Turn {_turn_counter} | Query #{_query_counter} (Turn Query #{_last_turn_query_count})")
        print(f"[RAG] Search Query: '{query}'")
        print(f"[RAG] top_k={top_k}, diverse={diverse}, min_docs={min_docs}, max_per_doc={max_per_doc}")
        print(f"{'='*60}")
        
        if not os.path.exists(json_path):
            return f"Error: File not found: {json_path}"
        
        rag = _get_rag_instance(json_path)
        
        # Use diverse search by default to ensure results from multiple documents
        if diverse:
            results = rag.diverse_search(query, top_k=top_k, min_docs=min_docs, max_per_doc=max_per_doc)
        else:
            results = rag.search(query, top_k=top_k)
        
        print(f"[RAG] Found {len(results)} results")
        for i, r in enumerate(results[:3]):
            print(f"[RAG]   Result {i+1}: score={r.score:.3f}, title='{r.title[:40]}...'")
        
        if not results:
            return f"No relevant results found for query: '{query}'"
        
        # Save retrieval results to log file
        _save_retrieval_result(query, results, "rag_search", json_path)
        
        # Format results with detailed source information for citation
        output_parts = [
            f"=== RAG Search Results ===",
            f"Query: '{query}'",
            f"Source File: {json_path}",
            f"Results Found: {len(results)}",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            # Create a detailed citation ID for this result
            # Format: [long_context: "文档标题", chunk X]
            title_short = result.title[:50] if result.title else "Untitled"
            chunk_idx = result.metadata.get('chunk_index', result.chunk_index if hasattr(result, 'chunk_index') else 0)
            citation_id = f'[long_context: "{title_short}", chunk {chunk_idx}]'
            
            output_parts.append(f"\n{'='*60}")
            output_parts.append(f"Result {i}")
            output_parts.append(f"Citation: {citation_id}")
            output_parts.append(f"{'='*60}")
            output_parts.append(f"Relevance Score: {result.score:.3f}")
            output_parts.append(f"Document Index: {result.doc_index}")
            output_parts.append(f"Chunk Index: {chunk_idx}")
            output_parts.append(f"Title: {result.title}")
            
            if result.metadata.get('url'):
                output_parts.append(f"Source URL: {result.metadata['url']}")
            if result.metadata.get('source'):
                output_parts.append(f"Source Type: {result.metadata['source']}")
            
            output_parts.append(f"\n--- Content ---")
            output_parts.append(f"(Cite as: {citation_id})")
            # Show more content for better context
            content_preview = result.content[:3000] if len(result.content) > 3000 else result.content
            output_parts.append(content_preview)
            
            if len(result.content) > 3000:
                output_parts.append(f"\n[... Content truncated. Full length: {len(result.content)} chars]")
            
            output_parts.append("")  # Empty line between results
        
        # Add citation guidance
        output_parts.append("\n" + "="*60)
        output_parts.append("CITATION GUIDANCE:")
        output_parts.append("When citing information from long_context.json, use the format:")
        output_parts.append('  [long_context: "Document Title", chunk N]')
        output_parts.append("For PDF/image files, use: [filename.pdf, page X] or [filename.png]")
        output_parts.append("="*60)
        
        return "\n".join(output_parts)
    except Exception as e:
        logger.exception(f"Error during RAG search: {e}")
        return f"Error during RAG search: {str(e)}"


@mcp.tool()
def rag_get_context(
    query: str,
    json_path: str,
    max_tokens: int = 4000,
    top_k: int = 10,
    diverse: bool = True,
    min_docs: int = 5,
    max_per_doc: int = 2
) -> str:
    """
    Get relevant context from a long document for answering a specific question.
    
    This tool retrieves the most relevant passages that can help answer the query,
    formatted as context that can be used for further analysis.
    Each passage includes source information for proper citation.
    
    By default, uses diverse search to ensure context comes from multiple different
    documents, not just the most similar chunks from a few documents.
    
    Args:
        query: The question or topic to find context for
        json_path: Path to the long_context.json file
        max_tokens: Maximum approximate tokens of context to return (default: 4000)
        top_k: Number of documents to consider (default: 10)
        diverse: If True, ensure results come from different documents (default: True)
        min_docs: Minimum number of different documents to include when diverse=True (default: 5)
        max_per_doc: Maximum chunks per document when diverse=True (default: 2)
        
    Returns:
        Concatenated relevant context passages with source information
    """
    try:
        print(f"[RAG] rag_get_context called with query: '{query[:50]}...' max_tokens={max_tokens}, diverse={diverse}")
        
        if not os.path.exists(json_path):
            return f"Error: File not found: {json_path}"
        
        rag = _get_rag_instance(json_path)
        
        # Get search results with full metadata, using diverse search by default
        if diverse:
            results = rag.diverse_search(query, top_k=top_k, min_docs=min_docs, max_per_doc=max_per_doc)
        else:
            results = rag.search(query, top_k=top_k)
        
        if not results:
            return "No relevant context found for the query."
        
        # Save retrieval results to log file
        _save_retrieval_result(query, results, "rag_get_context", json_path)
        
        # Build context with source citations
        output_parts = [
            f"=== RAG Context for Query ===",
            f"Query: '{query}'",
            f"Source File: {json_path}",
            f"Documents Retrieved: {len(results)}",
            "",
            "--- Retrieved Context with Sources ---",
            ""
        ]
        
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate chars per token
        
        for i, result in enumerate(results, 1):
            # Create detailed citation format: [long_context: "文档标题", chunk X]
            title_short = result.title[:50] if result.title else "Untitled"
            chunk_idx = result.metadata.get('chunk_index', result.chunk_index if hasattr(result, 'chunk_index') else 0)
            citation_id = f'[long_context: "{title_short}", chunk {chunk_idx}]'
            
            # Build document header
            doc_header = [
                f"\n{'='*50}",
                f"Source {i}",
                f"Citation: {citation_id}",
                f"Title: {result.title}",
                f"Document Index: {result.doc_index}",
                f"Chunk Index: {chunk_idx}",
                f"Relevance Score: {result.score:.3f}",
            ]
            
            if result.metadata.get('url'):
                doc_header.append(f"URL: {result.metadata['url']}")
            
            doc_header.append(f"{'='*50}")
            header_text = "\n".join(doc_header)
            
            # Check if we have room for this document
            doc_text = f"{header_text}\n\n{result.content}\n"
            if total_chars + len(doc_text) > max_chars:
                # Add partial content if possible
                remaining = max_chars - total_chars - len(header_text) - 100
                if remaining > 500:
                    output_parts.append(header_text)
                    output_parts.append(f"\n{result.content[:remaining]}...")
                    output_parts.append(f"\n[Content truncated due to token limit]")
                break
            
            output_parts.append(doc_text)
            total_chars += len(doc_text)
        
        # Add citation guidance
        output_parts.append("\n" + "="*50)
        output_parts.append("CITATION GUIDANCE:")
        output_parts.append("When citing information from long_context.json, use the format:")
        output_parts.append('  [long_context: "Document Title", chunk N]')
        output_parts.append("For PDF/image files, use: [filename.pdf, page X] or [filename.png]")
        output_parts.append("="*50)
        
        print(f"[RAG] Retrieved context length: {sum(len(p) for p in output_parts)} chars")
        
        return "\n".join(output_parts)
    except Exception as e:
        logger.exception(f"Error getting context: {e}")
        return f"Error getting context: {str(e)}"


@mcp.tool()
def rag_document_stats(json_path: str) -> str:
    """
    Get statistics about the loaded document collection.
    
    Args:
        json_path: Path to the long_context.json file
        
    Returns:
        Statistics about the document collection (number of documents, chunks, cache info, etc.)
    """
    try:
        if not os.path.exists(json_path):
            return f"Error: File not found: {json_path}"
        
        rag = _get_rag_instance(json_path)
        stats = rag.get_stats()
        
        output_parts = [
            "Document Collection Statistics:",
            f"- Source file: {json_path}",
            f"- Total documents: {stats.get('total_documents', 0)}",
            f"- Total chunks: {stats.get('total_chunks', 0)}",
            f"- Chunk size: {stats.get('chunk_size', 0)} chars",
            f"- Chunk overlap: {stats.get('chunk_overlap', 0)} chars",
            f"- Total characters: {stats.get('total_characters', 0):,}",
            f"- Average chunk length: {stats.get('average_chunk_length', 0)} chars",
            f"- Embedding model: {stats.get('embedding_model', 'unknown')}",
            f"- Cache path: {stats.get('cache_path', 'none')}",
            f"\nSample titles:",
        ]
        
        for i, title in enumerate(stats.get('sample_titles', []), 1):
            output_parts.append(f"  {i}. {title}")
        
        return "\n".join(output_parts)
    except Exception as e:
        logger.exception(f"Error getting document stats: {e}")
        return f"Error getting document stats: {str(e)}"


if __name__ == "__main__":
    mcp.run()
