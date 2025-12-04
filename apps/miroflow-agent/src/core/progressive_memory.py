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
Progressive Memory Management Module

This module implements a priority-based memory management system with rolling compression.
When context limit is reached, it compresses lower-priority content first before adding new content.

Priority levels (lower number = higher priority):
1. USER_QUERY - User's original query (never compressed)
2. USER_FILE - User uploaded files (can be compressed but not deleted)
3. RAG_RESULT - RAG search results (can be compressed)
4. TOOL_RESULT - Other tool results (can be compressed)
5. CONVERSATION - Intermediate conversation (can be deleted)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import tiktoken

logger = logging.getLogger(__name__)


class ContentPriority(Enum):
    """Priority levels for memory content"""
    USER_QUERY = 1      # User's original query - never compressed
    USER_FILE = 2       # User uploaded files - can be compressed but not deleted
    RAG_RESULT = 3      # RAG search results - can be compressed
    TOOL_RESULT = 4     # Other tool results - can be compressed
    CONVERSATION = 5    # Intermediate conversation - can be deleted


@dataclass
class MemoryItem:
    """A single item in the memory"""
    content: str
    priority: ContentPriority
    original_tokens: int
    current_tokens: int
    is_compressed: bool = False
    compression_level: int = 0  # 0=original, 1=light compression, 2=heavy compression
    metadata: Dict[str, Any] = field(default_factory=dict)
    role: str = "user"  # "user", "assistant", or "system"
    
    def can_compress(self) -> bool:
        """Check if this item can be further compressed"""
        if self.priority == ContentPriority.USER_QUERY:
            return False  # Never compress user query
        if self.priority == ContentPriority.USER_FILE:
            return self.compression_level < 2  # Can compress up to level 2
        if self.priority == ContentPriority.CONVERSATION:
            return True  # Can always be deleted
        return self.compression_level < 2  # Other types can compress up to level 2
    
    def can_delete(self) -> bool:
        """Check if this item can be deleted"""
        return self.priority == ContentPriority.CONVERSATION


class ProgressiveMemory:
    """
    Progressive Memory Manager with priority-based compression.
    
    When context limit is reached:
    1. First, compress lower-priority items (CONVERSATION -> TOOL_RESULT -> RAG_RESULT -> USER_FILE)
    2. If still over limit, delete CONVERSATION items
    3. USER_QUERY is never compressed or deleted
    """
    
    def __init__(
        self,
        max_tokens: int = 100000,
        reserved_tokens: int = 20000,  # Reserved for new content and response
        compress_callback: Optional[Callable[[str, int], str]] = None,
    ):
        """
        Initialize Progressive Memory.
        
        Args:
            max_tokens: Maximum tokens allowed in memory
            reserved_tokens: Tokens reserved for new content and LLM response
            compress_callback: Async function to compress content, signature: (content, level) -> compressed_content
        """
        self.items: List[MemoryItem] = []
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.compress_callback = compress_callback
        self._encoding = None
        
    @property
    def encoding(self):
        """Lazy load tiktoken encoding"""
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            # Use disallowed_special=() to allow all special tokens to be encoded as normal text
            # This prevents errors when encountering tokens like <|endofprompt|>
            return len(self.encoding.encode(text, disallowed_special=()))
        except Exception:
            # Fallback: ~4 chars per token
            return len(text) // 4
    
    @property
    def current_tokens(self) -> int:
        """Get current total tokens in memory"""
        return sum(item.current_tokens for item in self.items)
    
    @property
    def available_tokens(self) -> int:
        """Get available tokens for new content"""
        return self.max_tokens - self.current_tokens - self.reserved_tokens
    
    def add(
        self,
        content: str,
        priority: ContentPriority,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add content to memory.
        
        Args:
            content: The content to add
            priority: Priority level of the content
            role: Message role ("user", "assistant", "system")
            metadata: Optional metadata
            
        Returns:
            True if added successfully, False if failed
        """
        tokens = self.estimate_tokens(content)
        
        item = MemoryItem(
            content=content,
            priority=priority,
            original_tokens=tokens,
            current_tokens=tokens,
            role=role,
            metadata=metadata or {},
        )
        
        self.items.append(item)
        
        logger.debug(
            f"Added memory item: priority={priority.name}, tokens={tokens}, "
            f"total_tokens={self.current_tokens}"
        )
        
        return True
    
    def needs_compression(self, incoming_tokens: int = 0) -> bool:
        """Check if compression is needed to fit incoming content"""
        return self.current_tokens + incoming_tokens > self.max_tokens - self.reserved_tokens
    
    async def compress_if_needed(
        self,
        incoming_tokens: int = 0,
        compress_func: Optional[Callable] = None,
    ) -> Tuple[bool, int]:
        """
        Compress memory if needed to fit incoming content.
        
        Args:
            incoming_tokens: Tokens of incoming content
            compress_func: Async function to compress content
            
        Returns:
            Tuple of (success, tokens_freed)
        """
        if not self.needs_compression(incoming_tokens):
            return True, 0
        
        total_freed = 0
        compress_fn = compress_func or self.compress_callback
        
        # Sort items by priority (higher value = lower priority = compress first)
        # and compression level (lower level = compress first)
        compressible = [
            (i, item) for i, item in enumerate(self.items)
            if item.can_compress()
        ]
        compressible.sort(key=lambda x: (-x[1].priority.value, x[1].compression_level))
        
        for idx, item in compressible:
            if not self.needs_compression(incoming_tokens):
                break
            
            # Try to compress this item
            freed = await self._compress_item(idx, compress_fn)
            total_freed += freed
            
            logger.debug(
                f"Compressed item {idx}: priority={item.priority.name}, "
                f"freed={freed} tokens, total_freed={total_freed}"
            )
        
        # If still over limit, try deleting CONVERSATION items
        if self.needs_compression(incoming_tokens):
            deletable = [
                (i, item) for i, item in enumerate(self.items)
                if item.can_delete()
            ]
            # Delete from newest to oldest (reverse order)
            for idx, item in reversed(deletable):
                if not self.needs_compression(incoming_tokens):
                    break
                
                freed = item.current_tokens
                self.items.pop(idx)
                total_freed += freed
                
                logger.debug(
                    f"Deleted item {idx}: priority={item.priority.name}, "
                    f"freed={freed} tokens"
                )
        
        success = not self.needs_compression(incoming_tokens)
        return success, total_freed
    
    async def _compress_item(
        self,
        idx: int,
        compress_func: Optional[Callable] = None,
    ) -> int:
        """
        Compress a single item.
        
        Args:
            idx: Index of item to compress
            compress_func: Async function to compress content
            
        Returns:
            Tokens freed by compression
        """
        item = self.items[idx]
        
        if not item.can_compress():
            return 0
        
        old_tokens = item.current_tokens
        new_level = item.compression_level + 1
        
        if compress_func:
            # Use provided compression function
            try:
                compressed = await compress_func(item.content, new_level)
                new_tokens = self.estimate_tokens(compressed)
                
                item.content = compressed
                item.current_tokens = new_tokens
                item.is_compressed = True
                item.compression_level = new_level
                
                return old_tokens - new_tokens
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
                return 0
        else:
            # Use simple truncation as fallback
            if new_level == 1:
                # Light compression: keep first 1/3
                target_len = len(item.content) // 3
                compressed = item.content[:target_len] + "\n... [Content compressed]"
            else:
                # Heavy compression: keep first 1/6
                target_len = len(item.content) // 6
                compressed = item.content[:target_len] + "\n... [Content heavily compressed]"
            
            new_tokens = self.estimate_tokens(compressed)
            
            item.content = compressed
            item.current_tokens = new_tokens
            item.is_compressed = True
            item.compression_level = new_level
            
            return old_tokens - new_tokens
    
    def to_message_history(self) -> List[Dict[str, str]]:
        """Convert memory to message history format"""
        messages = []
        
        for item in self.items:
            messages.append({
                "role": item.role,
                "content": item.content,
            })
        
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            "total_items": len(self.items),
            "total_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "available_tokens": self.available_tokens,
            "items_by_priority": {},
            "compressed_items": 0,
        }
        
        for priority in ContentPriority:
            items = [i for i in self.items if i.priority == priority]
            stats["items_by_priority"][priority.name] = {
                "count": len(items),
                "tokens": sum(i.current_tokens for i in items),
            }
        
        stats["compressed_items"] = sum(1 for i in self.items if i.is_compressed)
        
        return stats
    
    def clear(self):
        """Clear all memory items"""
        self.items.clear()
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __repr__(self) -> str:
        return (
            f"ProgressiveMemory(items={len(self.items)}, "
            f"tokens={self.current_tokens}/{self.max_tokens})"
        )


class MemoryCompressor:
    """
    Helper class to compress memory content using LLM.
    """
    
    def __init__(self, llm_client: Any):
        """
        Initialize compressor with LLM client.
        
        Args:
            llm_client: LLM client for compression
        """
        self.llm_client = llm_client
    
    async def compress(self, content: str, level: int) -> str:
        """
        Compress content using LLM.
        
        Args:
            content: Content to compress
            level: Compression level (1=light, 2=heavy)
            
        Returns:
            Compressed content
        """
        if level == 1:
            prompt = self._get_light_compression_prompt(content)
        else:
            prompt = self._get_heavy_compression_prompt(content)
        
        try:
            # Use a simple completion call for compression
            # This should use a fast/cheap model
            response = await self.llm_client.quick_complete(prompt)
            return response
        except Exception as e:
            logger.warning(f"LLM compression failed: {e}, using fallback")
            return self._fallback_compress(content, level)
    
    def _get_light_compression_prompt(self, content: str) -> str:
        """Get prompt for light compression"""
        return f"""请将以下内容压缩为原来的1/3长度，保留所有关键信息、数据和引用：

{content}

压缩后的内容（保留关键信息）："""
    
    def _get_heavy_compression_prompt(self, content: str) -> str:
        """Get prompt for heavy compression"""
        return f"""请将以下内容压缩为3-5个核心要点，只保留最重要的信息：

{content}

核心要点："""
    
    def _fallback_compress(self, content: str, level: int) -> str:
        """Fallback compression using simple truncation"""
        if level == 1:
            target_len = len(content) // 3
            return content[:target_len] + "\n... [Content compressed]"
        else:
            target_len = len(content) // 6
            return content[:target_len] + "\n... [Content heavily compressed]"
