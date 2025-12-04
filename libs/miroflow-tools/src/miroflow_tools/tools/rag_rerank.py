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
RAG Rerank and Summary Module

This module provides post-retrieval processing for RAG results:
- Rerank: Use LLM to re-score and filter retrieved chunks based on relevance
- Summary: Extract key information from each chunk to avoid truncation issues

Features:
- LLM-based relevance scoring (0-10 scale)
- Noise filtering based on relevance threshold
- Key fact extraction from each chunk
- Configurable via environment variables
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result of reranking a single chunk."""
    original_index: int
    relevance_score: float
    relevance_reason: str
    is_relevant: bool


@dataclass
class SummaryResult:
    """Result of summarizing a single chunk."""
    key_facts: List[str]
    summary: str
    relevance_note: str
    citation: str


class RAGReranker:
    """
    Rerank retrieved chunks using LLM-based relevance scoring.
    
    This class uses an LLM to evaluate the relevance of each retrieved chunk
    to the original query, filtering out noise and irrelevant content.
    
    Uses the same model as the Agent system (configured via OPENAI_API_KEY and 
    OPENAI_BASE_URL environment variables) to avoid extra model calls.
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        """
        Initialize the reranker.
        
        Args:
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            base_url: OpenAI API base URL (default: from OPENAI_BASE_URL env var)
            model: Model to use for reranking (default: uses agent's model from OPENAI_MODEL env var)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        # Use agent's model by default (same model as the main agent)
        # Priority: explicit model > RAG_RERANK_MODEL > OPENAI_MODEL > fallback
        self.model = model or os.environ.get("RAG_RERANK_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4.1")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for reranking")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"RAGReranker initialized with model: {self.model}")
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        relevance_threshold: float = 0.6,
        max_content_preview: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on relevance to the query.
        
        Only filters out irrelevant chunks (below threshold), keeps all relevant ones.
        
        Args:
            query: The original search query
            chunks: List of retrieved chunks (each with 'title', 'content', 'score', etc.)
            relevance_threshold: Minimum relevance score (0-1) to keep a chunk
            max_content_preview: Maximum characters of content to show in rerank prompt
        
        Returns:
            List of reranked and filtered chunks with added 'rerank_score' and 'rerank_reason'
        """
        if not chunks:
            return []
        
        # Build rerank prompt
        rerank_prompt = self._build_rerank_prompt(query, chunks, max_content_preview)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个信息相关性评估专家。请严格按照要求的JSON格式输出。"},
                    {"role": "user", "content": rerank_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            rankings = self._parse_rankings(response_text)
            
            if not rankings:
                logger.warning("Failed to parse rerank results, returning original chunks")
                return chunks
            
            # Filter by relevance threshold only (no top_n limit)
            # Keep all chunks that are above the threshold
            reranked_chunks = []
            filtered_count = 0
            for rank in sorted(rankings, key=lambda x: x.get('score', 0), reverse=True):
                idx = rank.get('index', 0) - 1  # Convert 1-indexed to 0-indexed
                if 0 <= idx < len(chunks):
                    score = rank.get('score', 0) / 10.0  # Normalize to 0-1
                    if score >= relevance_threshold:
                        chunk = chunks[idx].copy()
                        chunk['rerank_score'] = score
                        chunk['rerank_reason'] = rank.get('reason', '')
                        reranked_chunks.append(chunk)
                    else:
                        filtered_count += 1
            
            logger.info(f"Reranked {len(chunks)} chunks: kept {len(reranked_chunks)}, filtered {filtered_count} irrelevant")
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            return chunks
    
    def _build_rerank_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_content_preview: int
    ) -> str:
        """Build the rerank prompt."""
        prompt = f"""请评估以下文档片段与查询的相关性。

**查询**: {query}

**评分标准** (0-10分):
- 10分：完全相关，直接回答查询，包含关键信息
- 7-9分：高度相关，包含有用信息，但可能不是直接答案
- 4-6分：部分相关，可能有用但信息价值有限
- 1-3分：略微相关，提到了相关词汇但内容不相关
- 0分：完全不相关或是干扰信息

**文档片段**:
"""
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get('title', 'Untitled')[:100]
            content = chunk.get('content', '')[:max_content_preview]
            prompt += f"\n[片段 {i}]\n标题: {title}\n内容: {content}...\n"
        
        prompt += """
**请以JSON格式返回评分结果**:
```json
{
    "rankings": [
        {"index": 1, "score": 8, "reason": "直接包含查询相关的具体信息"},
        {"index": 2, "score": 3, "reason": "虽然提到相关词汇但内容是关于其他主题的"}
    ]
}
```

注意：
1. 必须对每个片段都给出评分
2. score 必须是 0-10 的整数
3. reason 要简洁说明评分理由
"""
        return prompt
    
    def _parse_rankings(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse the rankings from LLM response."""
        # Try to extract JSON from response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{[\s\S]*"rankings"[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group()
            else:
                return []
        
        try:
            data = json.loads(json_str)
            return data.get('rankings', [])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse rerank JSON: {e}")
            return []


class RAGSummarizer:
    """
    Summarize retrieved chunks to extract key information.
    
    This class uses an LLM to extract the most relevant information from each chunk,
    avoiding truncation issues and focusing on query-relevant content.
    
    Uses the same model as the Agent system (configured via OPENAI_API_KEY and 
    OPENAI_BASE_URL environment variables) to avoid extra model calls.
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        """
        Initialize the summarizer.
        
        Args:
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            base_url: OpenAI API base URL (default: from OPENAI_BASE_URL env var)
            model: Model to use for summarization (default: uses agent's model from OPENAI_MODEL env var)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        # Use agent's model by default (same model as the main agent)
        # Priority: explicit model > RAG_SUMMARY_MODEL > OPENAI_MODEL > fallback
        self.model = model or os.environ.get("RAG_SUMMARY_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4.1")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for summarization")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"RAGSummarizer initialized with model: {self.model}")
    
    def summarize_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_summary_length: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Summarize each chunk to extract key information.
        
        Args:
            query: The original search query
            chunks: List of chunks to summarize
            max_summary_length: Maximum length of each summary
        
        Returns:
            List of chunks with added 'key_facts', 'summary', and 'relevance_note'
        """
        if not chunks:
            return []
        
        summarized_chunks = []
        
        # Process chunks in batch for efficiency
        batch_prompt = self._build_batch_summary_prompt(query, chunks, max_summary_length)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个信息提取专家。请严格按照要求的JSON格式输出。"},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            summaries = self._parse_summaries(response_text)
            
            for i, chunk in enumerate(chunks):
                chunk_copy = chunk.copy()
                if i < len(summaries):
                    summary_data = summaries[i]
                    chunk_copy['key_facts'] = summary_data.get('key_facts', [])
                    chunk_copy['summary'] = summary_data.get('summary', chunk.get('content', '')[:max_summary_length])
                    chunk_copy['relevance_note'] = summary_data.get('relevance_note', '')
                else:
                    # Fallback if parsing failed for this chunk
                    chunk_copy['key_facts'] = []
                    chunk_copy['summary'] = chunk.get('content', '')[:max_summary_length]
                    chunk_copy['relevance_note'] = ''
                
                summarized_chunks.append(chunk_copy)
            
            logger.info(f"Summarized {len(chunks)} chunks")
            return summarized_chunks
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Return chunks with truncated content as fallback
            for chunk in chunks:
                chunk_copy = chunk.copy()
                chunk_copy['key_facts'] = []
                chunk_copy['summary'] = chunk.get('content', '')[:max_summary_length]
                chunk_copy['relevance_note'] = ''
                summarized_chunks.append(chunk_copy)
            return summarized_chunks
    
    def _build_batch_summary_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_summary_length: int
    ) -> str:
        """Build the batch summary prompt."""
        prompt = f"""请从以下文档片段中提取与查询最相关的关键信息。

**查询**: {query}

**要求**:
1. 提取与查询直接相关的事实和数据
2. 保留关键的名称、日期、数字等
3. 保留原文的准确表述，不要改写
4. 每个摘要控制在 {max_summary_length} 字以内
5. 如果片段与查询不太相关，摘要可以更短

**文档片段**:
"""
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get('title', 'Untitled')
            content = chunk.get('content', '')
            prompt += f"\n[片段 {i}]\n标题: {title}\n内容: {content}\n"
        
        prompt += """
**请以JSON格式返回摘要结果**:
```json
{
    "summaries": [
        {
            "index": 1,
            "key_facts": ["事实1", "事实2"],
            "summary": "精炼的摘要内容",
            "relevance_note": "与查询的关联说明"
        }
    ]
}
```

注意：
1. 必须对每个片段都给出摘要
2. key_facts 是关键事实列表，每个事实一句话
3. summary 是整体摘要，保留最重要的信息
4. relevance_note 简要说明该片段与查询的关联
"""
        return prompt
    
    def _parse_summaries(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse the summaries from LLM response."""
        # Try to extract JSON from response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{[\s\S]*"summaries"[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group()
            else:
                return []
        
        try:
            data = json.loads(json_str)
            summaries = data.get('summaries', [])
            # Sort by index to ensure correct order
            summaries.sort(key=lambda x: x.get('index', 0))
            return summaries
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse summary JSON: {e}")
            return []


class RAGPostProcessor:
    """
    Combined post-processor that applies both reranking and summarization.
    
    This is the main entry point for RAG post-processing, combining
    reranking (to filter noise) and summarization (to extract key info).
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        rerank_model: str = None,
        summary_model: str = None
    ):
        """
        Initialize the post-processor.
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            rerank_model: Model for reranking
            summary_model: Model for summarization
        """
        self.reranker = RAGReranker(api_key, base_url, rerank_model)
        self.summarizer = RAGSummarizer(api_key, base_url, summary_model)
    
    def process(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        enable_rerank: bool = True,
        enable_summary: bool = True,
        relevance_threshold: float = 0.6,
        max_summary_length: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Process retrieved chunks with reranking and summarization.
        
        Args:
            query: The original search query
            chunks: List of retrieved chunks
            enable_rerank: Whether to apply reranking
            enable_summary: Whether to apply summarization
            relevance_threshold: Minimum relevance score for reranking (filters out irrelevant chunks)
            max_summary_length: Maximum length of each summary
        
        Returns:
            List of processed chunks (only irrelevant ones filtered, all relevant ones kept)
        """
        if not chunks:
            return []
        
        processed_chunks = chunks
        
        # Step 1: Rerank (filter noise only, keep all relevant chunks)
        if enable_rerank:
            logger.info(f"Reranking {len(processed_chunks)} chunks...")
            processed_chunks = self.reranker.rerank(
                query=query,
                chunks=processed_chunks,
                relevance_threshold=relevance_threshold
            )
            logger.info(f"After reranking: {len(processed_chunks)} chunks")
        
        # Step 2: Summarize (extract key info)
        if enable_summary:
            logger.info(f"Summarizing {len(processed_chunks)} chunks...")
            processed_chunks = self.summarizer.summarize_chunks(
                query=query,
                chunks=processed_chunks,
                max_summary_length=max_summary_length
            )
            logger.info(f"Summarization complete")
        
        return processed_chunks
    
    def format_results(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        include_original_content: bool = False
    ) -> str:
        """
        Format processed chunks into a readable string.
        
        Args:
            query: The original search query
            chunks: List of processed chunks
            include_original_content: Whether to include original content
        
        Returns:
            Formatted string with search results
        """
        if not chunks:
            return f"No relevant results found for query: '{query}'"
        
        output_parts = [
            f"=== RAG Search Results (Enhanced) ===",
            f"Query: '{query}'",
            f"Results Found: {len(chunks)}",
            ""
        ]
        
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get('title', 'Untitled')[:50]
            chunk_idx = chunk.get('chunk_index', 0)
            citation = f'[long_context: "{title}", chunk {chunk_idx}]'
            
            output_parts.append(f"\n{'='*60}")
            output_parts.append(f"Result {i}")
            output_parts.append(f"Citation: {citation}")
            output_parts.append(f"{'='*60}")
            
            # Show rerank info if available
            if 'rerank_score' in chunk:
                output_parts.append(f"Relevance Score: {chunk['rerank_score']:.2f}")
                if chunk.get('rerank_reason'):
                    output_parts.append(f"Relevance Reason: {chunk['rerank_reason']}")
            
            output_parts.append(f"Title: {chunk.get('title', 'Untitled')}")
            
            if chunk.get('url'):
                output_parts.append(f"Source URL: {chunk['url']}")
            
            # Show key facts if available
            if chunk.get('key_facts'):
                output_parts.append(f"\n--- Key Facts ---")
                for fact in chunk['key_facts']:
                    output_parts.append(f"• {fact}")
            
            # Show summary or content
            output_parts.append(f"\n--- Content ---")
            output_parts.append(f"(Cite as: {citation})")
            
            if chunk.get('summary'):
                output_parts.append(chunk['summary'])
            elif include_original_content:
                content = chunk.get('content', '')[:3000]
                output_parts.append(content)
                if len(chunk.get('content', '')) > 3000:
                    output_parts.append(f"\n[... Content truncated]")
            
            if chunk.get('relevance_note'):
                output_parts.append(f"\nRelevance: {chunk['relevance_note']}")
            
            output_parts.append("")
        
        # Add citation guidance
        output_parts.append("\n" + "="*60)
        output_parts.append("CITATION GUIDANCE:")
        output_parts.append("When citing information from long_context.json, use the format:")
        output_parts.append('  [long_context: "Document Title", chunk N]')
        output_parts.append("="*60)
        
        return "\n".join(output_parts)
