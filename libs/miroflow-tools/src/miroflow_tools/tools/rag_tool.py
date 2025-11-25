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
class SearchResult:
    """A single search result from RAG."""
    title: str
    content: str
    score: float
    doc_index: int
    metadata: Dict[str, Any]


class RAGTool:
    """
    RAG Tool for semantic search over long context documents.
    
    Uses SQLite to cache embeddings for fast subsequent queries.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        cache_dir: str = None
    ):
        """
        Initialize RAG Tool.
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            embedding_model: Model to use for embeddings
            cache_dir: Directory to store SQLite cache (default: same as document)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        
        self.documents: List[Dict[str, Any]] = []
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
            return os.path.join(self.cache_dir, f"{base_name}.embeddings.db")
        else:
            return file_path + ".embeddings.db"
    
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
            CREATE TABLE IF NOT EXISTS documents (
                doc_index INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                url TEXT,
                embedding BLOB
            )
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
        self.conn.commit()
    
    def _load_from_cache(self) -> bool:
        """Load documents and embeddings from cache."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT doc_index, title, content, url, embedding FROM documents ORDER BY doc_index")
        rows = cursor.fetchall()
        
        if not rows:
            return False
        
        self.documents = []
        embeddings_list = []
        
        for row in rows:
            doc_index, title, content, url, embedding_blob = row
            self.documents.append({
                'title': title,
                'content': content,
                'url': url
            })
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embeddings_list.append(embedding)
        
        self.embeddings = np.array(embeddings_list)
        logger.info(f"Loaded {len(self.documents)} documents from cache")
        return True
    
    def _save_to_cache(self):
        """Save documents and embeddings to cache."""
        cursor = self.conn.cursor()
        
        # Clear existing documents
        cursor.execute("DELETE FROM documents")
        
        # Insert documents with embeddings
        for i, (doc, embedding) in enumerate(zip(self.documents, self.embeddings)):
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute(
                "INSERT INTO documents (doc_index, title, content, url, embedding) VALUES (?, ?, ?, ?, ?)",
                (i, doc.get('title', ''), doc.get('content', ''), doc.get('url', ''), embedding_blob)
            )
        
        self.conn.commit()
        logger.info(f"Saved {len(self.documents)} documents to cache")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        # Truncate text if too long (max ~8000 tokens for embedding models)
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Get embeddings for multiple texts in batches."""
        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            # Truncate each text
            batch = [t[:30000] if len(t) > 30000 else t for t in batch]
            
            logger.info(f"Generating embeddings: {i + len(batch)}/{total}")
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            
            batch_embeddings = [np.array(d.embedding) for d in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def load_documents(self, file_path: str) -> int:
        """
        Load documents from a long_context.json file.
        Uses SQLite cache if available and valid.
        
        Args:
            file_path: Path to the long_context.json file
            
        Returns:
            Number of documents loaded
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
                return len(self.documents)
        
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
        
        logger.info(f"Loaded {len(self.documents)} documents, generating embeddings...")
        
        # Generate embeddings
        texts = []
        for doc in self.documents:
            title = doc.get('title', '')
            # Support both 'content' and 'page_body' field names
            content = doc.get('content', '') or doc.get('page_body', '')
            text = f"{title}\n\n{content}" if title else content
            texts.append(text)
        
        self.embeddings = self._get_embeddings_batch(texts)
        
        # Save to cache
        self._save_cache_metadata(file_hash)
        self._save_to_cache()
        
        logger.info(f"Embeddings generated and cached for {len(self.documents)} documents")
        return len(self.documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        if self.embeddings is None or len(self.documents) == 0:
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
            
            doc = self.documents[idx]
            # Support both 'content' and 'page_body' field names
            content = doc.get('content', '') or doc.get('page_body', '')
            results.append(SearchResult(
                title=doc.get('title', ''),
                content=content,
                score=score,
                doc_index=int(idx),
                metadata={
                    'url': doc.get('url', ''),
                    'source': doc.get('source', '')
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
        Get concatenated context from top relevant documents.
        
        Args:
            query: Search query
            max_tokens: Maximum approximate tokens in context
            top_k: Number of documents to consider
            
        Returns:
            Concatenated context string
        """
        results = self.search(query, top_k=top_k)
        
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate chars per token
        
        for result in results:
            doc_text = f"## {result.title}\n\n{result.content}\n\n"
            if total_chars + len(doc_text) > max_chars:
                break
            context_parts.append(doc_text)
            total_chars += len(doc_text)
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        if not self.documents:
            return {"status": "no_documents_loaded"}
        
        total_chars = sum(len(d.get('content', '')) for d in self.documents)
        
        return {
            "total_documents": len(self.documents),
            "total_characters": total_chars,
            "average_doc_length": total_chars // len(self.documents) if self.documents else 0,
            "cache_path": self.db_path,
            "embedding_model": self.embedding_model,
            "sample_titles": [d.get('title', '')[:50] for d in self.documents[:5]]
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
