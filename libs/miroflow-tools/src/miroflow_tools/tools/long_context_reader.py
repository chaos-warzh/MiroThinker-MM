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
Long Context Reader Tool

This module provides a tool for reading and comprehending long context documents
that exceed LLM context limits. It processes documents in parallel chunks,
extracts key information, and synthesizes a comprehensive understanding.

Features:
- Parallel chunk processing for efficiency
- Intelligent information extraction and filtering
- Hierarchical summarization for long documents
- Question-aware reading for targeted comprehension
- Caching of processed results

Difference from RAG:
- RAG: Retrieves specific chunks based on semantic similarity (precision-focused)
- Long Context Reader: Reads and comprehends the entire document (comprehension-focused)
"""

import json
import hashlib
import sqlite3
import os
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class ChunkReading:
    """Reading result of a single chunk."""
    chunk_index: int
    doc_index: int
    title: str
    key_facts: List[str]
    key_entities: List[str]
    relevance_score: float  # 0-10, how relevant to the question
    summary: str
    original_length: int
    compressed_length: int


@dataclass
class DocumentReading:
    """Complete reading of a document."""
    doc_index: int
    title: str
    url: str
    overall_summary: str
    key_facts: List[str]
    key_entities: List[str]
    chunk_readings: List[ChunkReading]
    total_original_length: int
    total_compressed_length: int


@dataclass
class FullContextReading:
    """Result of reading the entire long context."""
    question: str
    total_documents: int
    total_chunks_processed: int
    document_readings: List[DocumentReading]
    synthesized_understanding: str
    key_findings: List[str]
    relevant_quotes: List[str]
    confidence_score: float
    compression_ratio: float
    processing_stats: Dict[str, Any] = field(default_factory=dict)


class LongContextReader:
    """
    Long Context Reader for comprehending documents that exceed LLM context limits.
    
    This tool reads documents in parallel chunks, extracts key information,
    and synthesizes a comprehensive understanding like a human reader would.
    
    Unlike RAG which retrieves specific chunks, this tool:
    1. Reads ALL chunks to build complete understanding
    2. Filters out irrelevant/redundant information
    3. Synthesizes information across all documents
    4. Provides a compressed but comprehensive view
    """
    
    # Default configuration
    DEFAULT_CHUNK_SIZE = 4000  # Characters per chunk for reading
    DEFAULT_MAX_WORKERS = 4   # Parallel processing workers
    
    # Prompts for different stages
    CHUNK_READING_PROMPT = """You are a careful reader extracting key information from a document chunk.

Document Title: {title}
Chunk {chunk_index} of {total_chunks}

Question/Focus (if provided): {question}

--- Document Content ---
{content}
--- End of Content ---

Please analyze this chunk and extract:

1. **Key Facts**: List the most important factual information (dates, numbers, events, claims, findings)
2. **Key Entities**: List important people, organizations, places, concepts, or technical terms
3. **Relevance**: Rate how relevant this chunk is to the question/focus (0-10, use 5 if no question provided)
4. **Summary**: Write a concise summary (2-4 sentences) capturing the essential information

IMPORTANT: Focus on extracting USEFUL information. Skip boilerplate, navigation text, or irrelevant content.

Respond in JSON format:
```json
{{
    "key_facts": ["fact1", "fact2", ...],
    "key_entities": ["entity1", "entity2", ...],
    "relevance_score": 8,
    "summary": "Concise summary here..."
}}
```"""

    DOCUMENT_SYNTHESIS_PROMPT = """You are synthesizing information from multiple chunks of a single document.

Document Title: {title}
URL: {url}
Total Chunks: {total_chunks}

Chunk Summaries and Key Information:
{chunk_info}

Please provide:
1. **Overall Summary**: A comprehensive summary of the entire document (3-5 sentences)
2. **Consolidated Key Facts**: The most important facts from all chunks (deduplicated)
3. **Consolidated Key Entities**: Important entities mentioned across chunks (deduplicated)

Respond in JSON format:
```json
{{
    "overall_summary": "Comprehensive document summary...",
    "key_facts": ["fact1", "fact2", ...],
    "key_entities": ["entity1", "entity2", ...]
}}
```"""

    FULL_SYNTHESIS_PROMPT = """You are synthesizing information from multiple documents to build a comprehensive understanding.

Question/Focus: {question}

I have read through {total_docs} documents with {total_chunks} total chunks. Here are the document summaries and key findings:

{document_summaries}

Based on all the information gathered, please provide:

1. **Synthesized Understanding**: A comprehensive understanding of all the information, answering the question if provided
2. **Key Findings**: The most important discoveries from reading all documents (prioritized by relevance)
3. **Relevant Quotes/Evidence**: Key pieces of evidence or quotes that support the findings
4. **Confidence Score**: Your confidence level (0-100) in the completeness of understanding
5. **Information Gaps**: What important information might still be missing

Respond in JSON format:
```json
{{
    "synthesized_understanding": "Your comprehensive understanding here...",
    "key_findings": ["finding1", "finding2", ...],
    "relevant_quotes": ["quote1", "quote2", ...],
    "confidence_score": 85,
    "information_gaps": ["gap1", "gap2", ...]
}}
```"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        cache_dir: str = None,
        chunk_size: int = None,
        max_workers: int = None
    ):
        """
        Initialize Long Context Reader.
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            model: Model to use for reading comprehension
            cache_dir: Directory to store cache (default: same as document)
            chunk_size: Characters per chunk (default: 4000)
            max_workers: Number of parallel workers (default: 4)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.max_workers = max_workers or self.DEFAULT_MAX_WORKERS
        
        self.documents: List[Dict[str, Any]] = []
        self.current_file: str = None
        self.db_path: str = None
        self.conn: sqlite3.Connection = None
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for cache validation."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_question_hash(self, question: str) -> str:
        """Get hash of question for cache key."""
        return hashlib.md5(question.encode()).hexdigest()[:16]
    
    def _get_db_path(self, file_path: str) -> str:
        """Get SQLite database path for caching."""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            base_name = os.path.basename(file_path)
            return os.path.join(self.cache_dir, f"{base_name}.reader.db")
        else:
            return file_path + ".reader.db"
    
    def _init_db(self, db_path: str):
        """Initialize SQLite database for caching."""
        self.conn = sqlite3.connect(db_path)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_hash TEXT,
                file_hash TEXT,
                doc_index INTEGER,
                chunk_index INTEGER,
                title TEXT,
                key_facts TEXT,
                key_entities TEXT,
                relevance_score REAL,
                summary TEXT,
                original_length INTEGER,
                compressed_length INTEGER,
                UNIQUE(question_hash, file_hash, doc_index, chunk_index)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_question_file 
            ON chunk_readings(question_hash, file_hash)
        ''')
        
        self.conn.commit()
    
    def _split_into_chunks(self, text: str) -> List[Tuple[int, str]]:
        """Split text into chunks with smart boundaries."""
        if not text or not text.strip():
            return [(0, "[Empty Document]")]
        
        chunks = []
        text_len = len(text)
        start = 0
        chunk_index = 0
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # Try to find a good break point
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
            if chunk_content:
                chunks.append((chunk_index, chunk_content))
                chunk_index += 1
            
            start = end
        
        return chunks if chunks else [(0, "[Empty Document]")]
    
    def _read_single_chunk(
        self,
        chunk_index: int,
        chunk_content: str,
        doc_index: int,
        title: str,
        total_chunks: int,
        question: str,
        question_hash: str,
        file_hash: str
    ) -> ChunkReading:
        """Read and extract information from a single chunk."""
        
        # Check cache first
        if self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT key_facts, key_entities, relevance_score, summary, original_length, compressed_length
                FROM chunk_readings
                WHERE question_hash = ? AND file_hash = ? AND doc_index = ? AND chunk_index = ?
            ''', (question_hash, file_hash, doc_index, chunk_index))
            cached = cursor.fetchone()
            
            if cached:
                logger.debug(f"Cache hit for doc {doc_index}, chunk {chunk_index}")
                return ChunkReading(
                    chunk_index=chunk_index,
                    doc_index=doc_index,
                    title=title,
                    key_facts=json.loads(cached[0]),
                    key_entities=json.loads(cached[1]),
                    relevance_score=cached[2],
                    summary=cached[3],
                    original_length=cached[4],
                    compressed_length=cached[5]
                )
        
        # Call LLM to read the chunk
        prompt = self.CHUNK_READING_PROMPT.format(
            title=title,
            chunk_index=chunk_index + 1,
            total_chunks=total_chunks,
            question=question if question else "No specific question - extract all important information",
            content=chunk_content
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Try to parse the whole content as JSON
                result = json.loads(content)
            
            reading = ChunkReading(
                chunk_index=chunk_index,
                doc_index=doc_index,
                title=title,
                key_facts=result.get("key_facts", []),
                key_entities=result.get("key_entities", []),
                relevance_score=float(result.get("relevance_score", 5)),
                summary=result.get("summary", ""),
                original_length=len(chunk_content),
                compressed_length=len(result.get("summary", ""))
            )
            
            # Cache the result
            if self.conn:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO chunk_readings
                    (question_hash, file_hash, doc_index, chunk_index, title, key_facts, key_entities, 
                     relevance_score, summary, original_length, compressed_length)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    question_hash, file_hash, doc_index, chunk_index, title,
                    json.dumps(reading.key_facts, ensure_ascii=False),
                    json.dumps(reading.key_entities, ensure_ascii=False),
                    reading.relevance_score, reading.summary,
                    reading.original_length, reading.compressed_length
                ))
                self.conn.commit()
            
            return reading
            
        except Exception as e:
            logger.error(f"Error reading chunk {chunk_index} of doc {doc_index}: {e}")
            return ChunkReading(
                chunk_index=chunk_index,
                doc_index=doc_index,
                title=title,
                key_facts=[],
                key_entities=[],
                relevance_score=0,
                summary=f"[Error reading chunk: {str(e)}]",
                original_length=len(chunk_content),
                compressed_length=0
            )
    
    def _synthesize_document(self, doc_index: int, title: str, url: str, 
                            chunk_readings: List[ChunkReading]) -> DocumentReading:
        """Synthesize information from all chunks of a document."""
        
        total_original = sum(cr.original_length for cr in chunk_readings)
        total_compressed = sum(cr.compressed_length for cr in chunk_readings)
        
        if len(chunk_readings) == 1:
            # Single chunk, no need for synthesis
            cr = chunk_readings[0]
            return DocumentReading(
                doc_index=doc_index,
                title=title,
                url=url,
                overall_summary=cr.summary,
                key_facts=cr.key_facts,
                key_entities=cr.key_entities,
                chunk_readings=chunk_readings,
                total_original_length=total_original,
                total_compressed_length=total_compressed
            )
        
        # Build chunk info for synthesis
        chunk_info_parts = []
        for cr in chunk_readings:
            chunk_info_parts.append(f"""
Chunk {cr.chunk_index + 1} (Relevance: {cr.relevance_score}/10):
Summary: {cr.summary}
Key Facts: {', '.join(cr.key_facts[:5])}
Key Entities: {', '.join(cr.key_entities[:5])}
""")
        
        prompt = self.DOCUMENT_SYNTHESIS_PROMPT.format(
            title=title,
            url=url,
            total_chunks=len(chunk_readings),
            chunk_info="\n".join(chunk_info_parts)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(content)
            
            return DocumentReading(
                doc_index=doc_index,
                title=title,
                url=url,
                overall_summary=result.get("overall_summary", ""),
                key_facts=result.get("key_facts", []),
                key_entities=result.get("key_entities", []),
                chunk_readings=chunk_readings,
                total_original_length=total_original,
                total_compressed_length=total_compressed
            )
            
        except Exception as e:
            logger.error(f"Error synthesizing document {doc_index}: {e}")
            # Fallback: combine chunk summaries
            all_facts = []
            all_entities = []
            summaries = []
            for cr in chunk_readings:
                all_facts.extend(cr.key_facts)
                all_entities.extend(cr.key_entities)
                summaries.append(cr.summary)
            
            return DocumentReading(
                doc_index=doc_index,
                title=title,
                url=url,
                overall_summary=" ".join(summaries[:3]),
                key_facts=list(set(all_facts))[:10],
                key_entities=list(set(all_entities))[:10],
                chunk_readings=chunk_readings,
                total_original_length=total_original,
                total_compressed_length=total_compressed
            )
    
    def _synthesize_full_context(
        self,
        question: str,
        document_readings: List[DocumentReading]
    ) -> Tuple[str, List[str], List[str], float]:
        """Synthesize understanding from all documents."""
        
        total_chunks = sum(len(dr.chunk_readings) for dr in document_readings)
        
        # Build document summaries
        doc_summaries_parts = []
        for dr in document_readings:
            avg_relevance = sum(cr.relevance_score for cr in dr.chunk_readings) / len(dr.chunk_readings)
            doc_summaries_parts.append(f"""
=== Document {dr.doc_index + 1}: {dr.title} ===
URL: {dr.url}
Average Relevance: {avg_relevance:.1f}/10
Summary: {dr.overall_summary}
Key Facts: {', '.join(dr.key_facts[:5])}
Key Entities: {', '.join(dr.key_entities[:5])}
""")
        
        prompt = self.FULL_SYNTHESIS_PROMPT.format(
            question=question if question else "Build a comprehensive understanding of all documents",
            total_docs=len(document_readings),
            total_chunks=total_chunks,
            document_summaries="\n".join(doc_summaries_parts)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(content)
            
            return (
                result.get("synthesized_understanding", ""),
                result.get("key_findings", []),
                result.get("relevant_quotes", []),
                float(result.get("confidence_score", 50))
            )
            
        except Exception as e:
            logger.error(f"Error synthesizing full context: {e}")
            # Fallback
            all_summaries = [dr.overall_summary for dr in document_readings]
            all_facts = []
            for dr in document_readings:
                all_facts.extend(dr.key_facts[:3])
            
            return (
                " ".join(all_summaries),
                all_facts[:10],
                [],
                30.0
            )
    
    def load_documents(self, file_path: str) -> int:
        """
        Load documents from a long_context.json file.
        
        Args:
            file_path: Path to the long_context.json file
            
        Returns:
            Number of documents loaded
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.current_file = file_path
        self.db_path = self._get_db_path(file_path)
        
        # Initialize database
        self._init_db(self.db_path)
        
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
        
        logger.info(f"Loaded {len(self.documents)} documents")
        return len(self.documents)
    
    def read_full_context(
        self,
        question: str = "",
        max_docs: int = None
    ) -> FullContextReading:
        """
        Read and comprehend the entire long context.
        
        This is the main method that:
        1. Splits all documents into chunks
        2. Reads each chunk in parallel
        3. Synthesizes document-level understanding
        4. Synthesizes full context understanding
        
        Args:
            question: Optional question/focus for targeted reading
            max_docs: Maximum number of documents to process (None = all)
            
        Returns:
            FullContextReading with comprehensive understanding
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        file_hash = self._get_file_hash(self.current_file)
        question_hash = self._get_question_hash(question)
        
        docs_to_process = self.documents[:max_docs] if max_docs else self.documents
        
        logger.info(f"Reading {len(docs_to_process)} documents with question: '{question[:50]}...'")
        
        document_readings = []
        total_chunks_processed = 0
        total_original_length = 0
        total_compressed_length = 0
        
        for doc_index, doc in enumerate(docs_to_process):
            title = doc.get('title', f'Document {doc_index + 1}')
            url = doc.get('url', '')
            content = doc.get('content', '') or doc.get('page_body', '')
            
            # Prepend title to content
            full_text = f"{title}\n\n{content}" if title else content
            
            # Split into chunks
            chunks = self._split_into_chunks(full_text)
            total_chunks = len(chunks)
            
            logger.info(f"Processing document {doc_index + 1}/{len(docs_to_process)}: {title[:50]}... ({total_chunks} chunks)")
            
            # Read chunks in parallel
            chunk_readings = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._read_single_chunk,
                        chunk_index, chunk_content, doc_index, title,
                        total_chunks, question, question_hash, file_hash
                    ): chunk_index
                    for chunk_index, chunk_content in chunks
                }
                
                for future in as_completed(futures):
                    try:
                        reading = future.result()
                        chunk_readings.append(reading)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
            
            # Sort by chunk index
            chunk_readings.sort(key=lambda x: x.chunk_index)
            total_chunks_processed += len(chunk_readings)
            
            # Synthesize document
            doc_reading = self._synthesize_document(doc_index, title, url, chunk_readings)
            document_readings.append(doc_reading)
            
            total_original_length += doc_reading.total_original_length
            total_compressed_length += doc_reading.total_compressed_length
        
        # Synthesize full context
        synthesized, key_findings, quotes, confidence = self._synthesize_full_context(
            question, document_readings
        )
        
        compression_ratio = total_compressed_length / total_original_length if total_original_length > 0 else 0
        
        return FullContextReading(
            question=question,
            total_documents=len(document_readings),
            total_chunks_processed=total_chunks_processed,
            document_readings=document_readings,
            synthesized_understanding=synthesized,
            key_findings=key_findings,
            relevant_quotes=quotes,
            confidence_score=confidence,
            compression_ratio=compression_ratio,
            processing_stats={
                "total_original_chars": total_original_length,
                "total_compressed_chars": total_compressed_length,
                "compression_ratio": f"{compression_ratio:.2%}",
                "avg_chunks_per_doc": total_chunks_processed / len(document_readings) if document_readings else 0
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        if not self.documents:
            return {"status": "no_documents_loaded"}
        
        total_chars = sum(
            len(doc.get('content', '') or doc.get('page_body', ''))
            for doc in self.documents
        )
        
        return {
            "total_documents": len(self.documents),
            "total_characters": total_chars,
            "estimated_chunks": total_chars // self.chunk_size + 1,
            "chunk_size": self.chunk_size,
            "max_workers": self.max_workers,
            "cache_path": self.db_path,
            "model": self.model,
            "sample_titles": [doc.get('title', '')[:50] for doc in self.documents[:5]]
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
