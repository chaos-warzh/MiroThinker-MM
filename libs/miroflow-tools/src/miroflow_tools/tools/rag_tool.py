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
RAG Tool for Long Context Document Search

This module provides RAG (Retrieval-Augmented Generation) functionality
for searching and retrieving relevant passages from long context documents.

Features:
- Chunk-based document processing for handling long documents
- SQLite-based embedding cache for fast subsequent queries
- Semantic search using OpenAI embeddings
- Support for long_context.json format
- Batch embedding generation with progress tracking
"""

import json
import hashlib
import sqlite3
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a document chunk."""
    doc_index: int
    chunk_index: int
    title: str
    content: str
    url: str
    start_char: int
    end_char: int


@dataclass
class SearchResult:
    """A single search result from RAG."""
    title: str
    content: str
    score: float
    doc_index: int
    chunk_index: int
    metadata: Dict[str, Any]


class RAGTool:
    """
    RAG Tool for semantic search over long context documents.
    
    Uses chunk-based processing to handle long documents and
    SQLite to cache embeddings for fast subsequent queries.
    """
    
    # Default chunk configuration
    DEFAULT_CHUNK_SIZE = 1500  # Characters per chunk (safe for ~750-1000 tokens)
    DEFAULT_CHUNK_OVERLAP = 200  # Overlap between chunks for context continuity
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        cache_dir: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize RAG Tool.
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            embedding_model: Model to use for embeddings
            cache_dir: Directory to store SQLite cache (default: same as document)
            chunk_size: Maximum characters per chunk (default: 1500)
            chunk_overlap: Overlap between chunks (default: 200)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
        
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[ChunkInfo] = []
        self.embeddings: np.ndarray = None
        self.current_file: str = None
        self.db_path: str = None
        self.conn: sqlite3.Connection = None
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for cache validation."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_db_path(self, file_path: str) -> str:
        """Get SQLite database path for caching embeddings."""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            base_name = os.path.basename(file_path)
            return os.path.join(self.cache_dir, f"{base_name}.chunks.db")
        else:
            return file_path + ".chunks.db"
    
    def _init_db(self, db_path: str):
        """Initialize SQLite database for embedding cache."""
        self.conn = sqlite3.connect(db_path)
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_index INTEGER,
                chunk_index INTEGER,
                title TEXT,
                content TEXT,
                url TEXT,
                start_char INTEGER,
                end_char INTEGER,
                embedding BLOB
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_doc_chunk 
            ON chunks(doc_index, chunk_index)
        ''')
        
        self.conn.commit()
    
    def _check_cache_valid(self, file_hash: str) -> bool:
        """Check if cache is valid for the current file."""
        if not self.conn:
            return False
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'file_hash'")
        result = cursor.fetchone()
        
        if result and result[0] == file_hash:
            cursor.execute("SELECT value FROM metadata WHERE key = 'embedding_model'")
            model_result = cursor.fetchone()
            if model_result and model_result[0] == self.embedding_model:
                # Also check chunk configuration
                cursor.execute("SELECT value FROM metadata WHERE key = 'chunk_size'")
                chunk_size_result = cursor.fetchone()
                cursor.execute("SELECT value FROM metadata WHERE key = 'chunk_overlap'")
                chunk_overlap_result = cursor.fetchone()
                if (chunk_size_result and int(chunk_size_result[0]) == self.chunk_size and
                    chunk_overlap_result and int(chunk_overlap_result[0]) == self.chunk_overlap):
                    return True
        
        return False
    
    def _save_cache_metadata(self, file_hash: str):
        """Save cache metadata."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ('file_hash', file_hash)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ('embedding_model', self.embedding_model)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ('chunk_size', str(self.chunk_size))
        )
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ('chunk_overlap', str(self.chunk_overlap))
        )
        self.conn.commit()
    
    def _split_into_chunks(self, text: str, doc_index: int, title: str, url: str) -> List[ChunkInfo]:
        """
        Split a document into chunks with overlap.
        
        Args:
            text: The document text to split
            doc_index: Index of the source document
            title: Document title
            url: Document URL
            
        Returns:
            List of ChunkInfo objects
        """
        if not text or not text.strip():
            # Return a single empty chunk for empty documents
            return [ChunkInfo(
                doc_index=doc_index,
                chunk_index=0,
                title=title,
                content="[Empty Document]",
                url=url,
                start_char=0,
                end_char=0
            )]
        
        chunks = []
        text_len = len(text)
        start = 0
        chunk_index = 0
        
        while start < text_len:
            # Calculate end position
            end = min(start + self.chunk_size, text_len)
            
            # Try to find a good break point (sentence or paragraph boundary)
            if end < text_len:
                # Look for paragraph break first
                break_point = text.rfind('\n\n', start, end)
                if break_point == -1 or break_point <= start:
                    # Look for sentence break
                    for sep in ['。', '！', '？', '. ', '! ', '? ', '\n']:
                        break_point = text.rfind(sep, start, end)
                        if break_point > start:
                            end = break_point + len(sep)
                            break
                else:
                    end = break_point + 2
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunks.append(ChunkInfo(
                    doc_index=doc_index,
                    chunk_index=chunk_index,
                    title=title,
                    content=chunk_content,
                    url=url,
                    start_char=start,
                    end_char=end
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= text_len:
                break
            # Ensure we make progress
            if start <= chunks[-1].start_char if chunks else 0:
                start = end
        
        return chunks if chunks else [ChunkInfo(
            doc_index=doc_index,
            chunk_index=0,
            title=title,
            content="[Empty Document]",
            url=url,
            start_char=0,
            end_char=0
        )]
    
    def _load_from_cache(self) -> bool:
        """Load chunks and embeddings from cache."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT doc_index, chunk_index, title, content, url, start_char, end_char, embedding 
            FROM chunks ORDER BY doc_index, chunk_index
        """)
        rows = cursor.fetchall()
        
        if not rows:
            return False
        
        self.chunks = []
        embeddings_list = []
        
        for row in rows:
            doc_index, chunk_index, title, content, url, start_char, end_char, embedding_blob = row
            self.chunks.append(ChunkInfo(
                doc_index=doc_index,
                chunk_index=chunk_index,
                title=title,
                content=content,
                url=url,
                start_char=start_char,
                end_char=end_char
            ))
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embeddings_list.append(embedding)
        
        self.embeddings = np.array(embeddings_list)
        logger.info(f"Loaded {len(self.chunks)} chunks from cache")
        return True
    
    def _save_to_cache(self):
        """Save chunks and embeddings to cache."""
        cursor = self.conn.cursor()
        
        # Clear existing chunks
        cursor.execute("DELETE FROM chunks")
        
        # Insert chunks with embeddings
        for chunk, embedding in zip(self.chunks, self.embeddings):
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute(
                """INSERT INTO chunks 
                   (doc_index, chunk_index, title, content, url, start_char, end_char, embedding) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk.doc_index, chunk.chunk_index, chunk.title, chunk.content, 
                 chunk.url, chunk.start_char, chunk.end_char, embedding_blob)
            )
        
        self.conn.commit()
        logger.info(f"Saved {len(self.chunks)} chunks to cache")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        # Ensure text is a non-empty string with actual content
        if not text or not isinstance(text, str) or not text.strip():
            text = "[Empty]"
        
        # Truncate if still too long (shouldn't happen with proper chunking)
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def _get_embeddings_for_chunks(self, chunks: List[ChunkInfo]) -> np.ndarray:
        """Get embeddings for all chunks one by one."""
        all_embeddings = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            text = chunk.content
            
            # Ensure text is valid
            if not text or not isinstance(text, str) or not text.strip():
                text = "[Empty]"
            
            if (i + 1) % 10 == 0 or i == total - 1:
                logger.info(f"Generating embeddings: {i + 1}/{total}")
            
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                embedding = np.array(response.data[0].embedding)
                all_embeddings.append(embedding)
            except Exception as e:
                if "maximum context length" in str(e).lower() or "token" in str(e).lower():
                    logger.warning(f"Chunk {i} too long, truncating...")
                    text = text[:3000]
                    try:
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=text
                        )
                        embedding = np.array(response.data[0].embedding)
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        logger.error(f"Failed to get embedding for chunk {i}: {e2}")
                        all_embeddings.append(np.zeros(1536))
                else:
                    logger.error(f"Failed to get embedding for chunk {i}: {e}")
                    all_embeddings.append(np.zeros(1536))
        
        return np.array(all_embeddings)
    
    def load_documents(self, file_path: str) -> int:
        """
        Load documents from a long_context.json file and split into chunks.
        Uses SQLite cache if available and valid.
        
        Args:
            file_path: Path to the long_context.json file
            
        Returns:
            Number of chunks created
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.current_file = file_path
        self.db_path = self._get_db_path(file_path)
        file_hash = self._get_file_hash(file_path)
        
        # Initialize database
        self._init_db(self.db_path)
        
        # Check if cache is valid
        if self._check_cache_valid(file_hash):
            logger.info(f"Loading from cache: {self.db_path}")
            if self._load_from_cache():
                return len(self.chunks)
        
        # Load documents from JSON
        logger.info(f"Loading documents from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.documents = data
        elif isinstance(data, dict) and 'documents' in data:
            self.documents = data['documents']
        else:
            raise ValueError("Invalid long_context.json format")
        
        logger.info(f"Loaded {len(self.documents)} documents, splitting into chunks...")
        
        # Split documents into chunks
        self.chunks = []
        for doc_index, doc in enumerate(self.documents):
            title = doc.get('title', '')
            url = doc.get('url', '')
            # Support both 'content' and 'page_body' field names
            content = doc.get('content', '') or doc.get('page_body', '')
            
            # Prepend title to content for better context
            full_text = f"{title}\n\n{content}" if title else content
            
            doc_chunks = self._split_into_chunks(full_text, doc_index, title, url)
            self.chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")
        logger.info(f"Generating embeddings for chunks...")
        
        # Generate embeddings for all chunks
        self.embeddings = self._get_embeddings_for_chunks(self.chunks)
        
        # Save to cache
        self._save_cache_metadata(file_hash)
        self._save_to_cache()
        
        logger.info(f"Embeddings generated and cached for {len(self.chunks)} chunks")
        return len(self.chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                continue
            
            chunk = self.chunks[idx]
            results.append(SearchResult(
                title=chunk.title,
                content=chunk.content,
                score=score,
                doc_index=chunk.doc_index,
                chunk_index=chunk.chunk_index,
                metadata={
                    'url': chunk.url,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                }
            ))
        
        return results
    
    def get_context(
        self,
        query: str,
        max_tokens: int = 4000,
        top_k: int = 10
    ) -> str:
        """
        Get concatenated context from top relevant chunks.
        
        Args:
            query: Search query
            max_tokens: Maximum approximate tokens in context
            top_k: Number of chunks to consider
            
        Returns:
            Concatenated context string
        """
        results = self.search(query, top_k=top_k)
        
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate chars per token
        
        for result in results:
            # Include title only for first chunk of each document
            if result.chunk_index == 0:
                chunk_text = f"## {result.title}\n\n{result.content}\n\n"
            else:
                chunk_text = f"{result.content}\n\n"
            
            if total_chars + len(chunk_text) > max_chars:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents and chunks."""
        if not self.chunks:
            return {"status": "no_documents_loaded"}
        
        # Count unique documents
        unique_docs = set(chunk.doc_index for chunk in self.chunks)
        total_chars = sum(len(chunk.content) for chunk in self.chunks)
        
        return {
            "total_documents": len(unique_docs),
            "total_chunks": len(self.chunks),
            "total_characters": total_chars,
            "average_chunk_length": total_chars // len(self.chunks) if self.chunks else 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "cache_path": self.db_path,
            "embedding_model": self.embedding_model,
            "sample_titles": list(set(chunk.title[:50] for chunk in self.chunks[:10]))
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
