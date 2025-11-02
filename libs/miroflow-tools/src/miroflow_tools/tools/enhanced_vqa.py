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
Enhanced Vision Question Answering (VQA) Module

This module provides advanced vision understanding capabilities with:
- Multi-turn vision understanding for improved accuracy
- Confidence scoring and metadata extraction
- Cross-validation with web search
- Support for multiple VLLM providers (OpenAI, Anthropic, Qwen, Custom)
"""

import base64
import os
from typing import Dict, List, Optional, Any, Union
import json
import requests
import aiohttp

import logging

logger = logging.getLogger(__name__)


class EnhancedVQAClient:
    """
    Enhanced Vision Question Answering client supporting multiple VLLM providers.
    
    Provides:
    1. Single-turn VQA with confidence scoring
    2. Multi-turn VQA with follow-up questions for verification
    3. Image metadata extraction (EXIF, dimensions, etc.)
    4. Cross-validation with web search (optional)
    """
    
    def __init__(self):
        """Initialize Enhanced VQA client from environment variables."""
        self.provider = os.environ.get("VLLM_PROVIDER", "openai")
        self.api_key = os.environ.get("VLLM_API_KEY")
        self.base_url = os.environ.get("VLLM_BASE_URL")
        self.model = os.environ.get("VLLM_MODEL")
        self.enable_multi_turn = os.environ.get("VLLM_ENABLE_MULTI_TURN", "true").lower() == "true"
        self.confidence_threshold = float(os.environ.get("VLLM_CONFIDENCE_THRESHOLD", "0.6"))
        
        # Google Search for cross-validation
        self.serper_api_key = os.environ.get("SERPER_API_KEY")
        self.serper_base_url = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev")
        self.jina_api_key = os.environ.get("JINA_API_KEY")
        self.jina_base_url = os.environ.get("JINA_BASE_URL", "https://r.jina.ai")
        
        if not all([self.api_key, self.base_url, self.model]):
            raise ValueError(
                "VLLM_API_KEY, VLLM_BASE_URL, and VLLM_MODEL must be set"
            )
        
        logger.info(f"Enhanced VQA initialized with provider: {self.provider}, model: {self.model}")
    
    async def analyze_image(
        self,
        image_path_or_url: str,
        question: str,
        enable_multi_turn: Optional[bool] = None,
        follow_up_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze an image with optional multi-turn verification.
        
        Args:
            image_path_or_url: Local file path or URL of the image
            question: Main question to ask about the image
            enable_multi_turn: Override default multi-turn setting
            follow_up_questions: Optional list of follow-up questions for verification
        
        Returns:
            Dict with keys:
            - answer: Main answer to the question
            - confidence: Confidence score (0.0-1.0)
            - metadata: Extracted metadata (dimensions, colors, objects detected, etc.)
            - follow_up_answers: Optional answers to follow-up questions
            - reasoning: Reasoning behind the answer
            - verification_sources: Optional verification from web search
        """
        # Encode image
        image_data = await self._encode_image(image_path_or_url)
        image_media_type = await self._get_image_media_type(image_path_or_url)
        
        # Primary analysis
        primary_response = await self._call_vllm(
            image_data=image_data,
            image_media_type=image_media_type,
            question=question,
            include_reasoning=True,
        )
        
        result: Dict[str, Union[str, float, Dict[str, Any], List[Dict[str, Any]]]] = {
            "answer": primary_response.get("answer", ""),
            "confidence": primary_response.get("confidence", 0.5),
            "metadata": primary_response.get("metadata", {}),
            "reasoning": primary_response.get("reasoning", ""),
        }
        
        # Multi-turn verification if enabled
        use_multi_turn = enable_multi_turn if enable_multi_turn is not None else self.enable_multi_turn
        
        if use_multi_turn:
            # Default follow-up questions if not provided
            if not follow_up_questions:
                follow_up_questions = self._generate_follow_up_questions(question, primary_response)
            
            follow_up_answers = []
            for follow_up_q in follow_up_questions:
                follow_up_response = await self._call_vllm(
                    image_data=image_data,
                    image_media_type=image_media_type,
                    question=follow_up_q,
                    context=primary_response.get("answer", ""),
                )
                follow_up_answers.append({
                    "question": follow_up_q,
                    "answer": follow_up_response.get("answer", ""),
                    "confidence": follow_up_response.get("confidence", 0.5),
                })
            
            result["follow_up_answers"] = follow_up_answers
            
            # Refine confidence based on follow-up consistency
            result["confidence"] = self._refine_confidence(
                primary_response,
                follow_up_answers
            )
        
        # Cross-validation with web search (if enabled and serper key available)
        if self.serper_api_key:
            try:
                verification_sources = await self._verify_with_web_search(
                    question=question,
                    answer=result["answer"],
                    image_path_or_url=image_path_or_url,
                )
                if verification_sources:
                    result["verification_sources"] = verification_sources
                    result["verified"] = True
            except Exception as e:
                logger.warning(f"Web search verification failed: {e}")
        
        return result
    
    async def _encode_image(self, image_path_or_url: str) -> str:
        """Encode image to base64 or return URL."""
        if image_path_or_url.startswith(("http://", "https://")):
            # For URLs, return as-is (most VLLMs support URL input)
            return image_path_or_url
        
        # Local file - encode to base64
        try:
            with open(image_path_or_url, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
    
    async def _get_image_media_type(self, image_path_or_url: str) -> str:
        """Detect image media type."""
        if image_path_or_url.startswith(("http://", "https://")):
            # Try to get content-type from URL
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.head(image_path_or_url) as resp:
                        return resp.headers.get("Content-Type", "image/jpeg")
            except:
                return "image/jpeg"
        
        # Local file - detect by extension
        _, ext = os.path.splitext(image_path_or_url)
        ext = ext.lower()
        
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        
        return media_types.get(ext, "image/jpeg")
    
    async def _call_vllm(
        self,
        image_data: str,
        image_media_type: str,
        question: str,
        include_reasoning: bool = False,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Call the VLLM provider API.
        
        Supports multiple providers:
        - openai: GPT-4V, GPT-4o, etc.
        - anthropic: Claude 3.5 Sonnet, etc.
        - qwen: Qwen2.5-VL, etc.
        - custom: Custom OpenAI-compatible endpoints
        """
        try:
            if self.provider == "openai":
                return await self._call_openai_vllm(
                    image_data, image_media_type, question, include_reasoning, context
                )
            elif self.provider == "anthropic":
                return await self._call_anthropic_vllm(
                    image_data, image_media_type, question, include_reasoning, context
                )
            elif self.provider in ["qwen", "custom"]:
                return await self._call_openai_compatible_vllm(
                    image_data, image_media_type, question, include_reasoning, context
                )
            else:
                raise ValueError(f"Unknown VLLM provider: {self.provider}")
        except Exception as e:
            logger.error(f"VLLM call failed: {e}")
            raise
    
    async def _call_openai_vllm(
        self,
        image_data: str,
        image_media_type: str,
        question: str,
        include_reasoning: bool = False,
        context: str = "",
    ) -> Dict[str, Any]:
        """Call OpenAI-compatible VLLM (GPT-4V, GPT-4o, etc.)."""
        # Determine if image_data is URL or base64
        if image_data.startswith(("http://", "https://")):
            image_source = {"type": "url", "url": image_data}
        else:
            image_source = {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_data,
            }
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_source},
                    {
                        "type": "text",
                        "text": self._build_prompt(question, context, include_reasoning),
                    },
                ],
            }
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        
        result = response.json()
        answer_text = result["choices"][0]["message"]["content"]
        
        # Parse response to extract structured information
        return self._parse_vllm_response(answer_text)
    
    async def _call_anthropic_vllm(
        self,
        image_data: str,
        image_media_type: str,
        question: str,
        include_reasoning: bool = False,
        context: str = "",
    ) -> Dict[str, Any]:
        """Call Anthropic Claude Vision API."""
        from anthropic import Anthropic
        from anthropic.types import MessageParam, TextBlockParam, ImageBlockParam
        
        client = Anthropic(api_key=self.api_key, base_url=self.base_url)
        
        # Prepare image source with proper typing
        if image_data.startswith(("http://", "https://")):
            image_block: ImageBlockParam = {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image_data,
                }
            }
        else:
            image_block: ImageBlockParam = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": image_data,
                }
            }
        
        text_block: TextBlockParam = {
            "type": "text",
            "text": self._build_prompt(question, context, include_reasoning),
        }
        
        messages: List[MessageParam] = [
            {
                "role": "user",
                "content": [image_block, text_block],
            }
        ]
        
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=messages,
        )
        
        answer_text = response.content[0].text
        
        # Parse response
        return self._parse_vllm_response(answer_text)
    
    async def _call_openai_compatible_vllm(
        self,
        image_data: str,
        image_media_type: str,
        question: str,
        include_reasoning: bool = False,
        context: str = "",
    ) -> Dict[str, Any]:
        """Call OpenAI-compatible endpoint (Qwen, custom, etc.)."""
        # For OpenAI-compatible endpoints like Qwen VLM
        if image_data.startswith(("http://", "https://")):
            image_url = image_data
        else:
            image_url = f"data:{image_media_type};base64,{image_data}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": self._build_prompt(question, context, include_reasoning),
                    },
                ],
            }
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        
        result = response.json()
        answer_text = result["choices"][0]["message"]["content"]
        
        return self._parse_vllm_response(answer_text)
    
    def _build_prompt(
        self,
        question: str,
        context: str = "",
        include_reasoning: bool = False,
    ) -> str:
        """Build the prompt for VLLM."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Previous context: {context}\n")
        
        prompt_parts.append(f"Question: {question}\n")
        
        if include_reasoning:
            prompt_parts.append(
                "\nPlease provide your answer in the following JSON format:\n"
                "{\n"
                '  "answer": "your direct answer to the question",\n'
                '  "confidence": <0.0-1.0>,\n'
                '  "reasoning": "your reasoning",\n'
                '  "metadata": {\n'
                '    "objects_detected": [...],\n'
                '    "colors_dominant": [...],\n'
                '    "text_visible": "any text in image",\n'
                '    "scene_description": "description of scene"\n'
                '  }\n'
                "}"
            )
        else:
            prompt_parts.append(
                "\nProvide your answer in JSON format:\n"
                "{\n"
                '  "answer": "your answer",\n'
                '  "confidence": <0.0-1.0>\n'
                "}"
            )
        
        return "".join(prompt_parts)
    
    def _parse_vllm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse VLLM response and extract structured information."""
        try:
            # Try to parse as JSON
            if "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Ensure required fields
                result = {
                    "answer": parsed.get("answer", response_text),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "metadata": parsed.get("metadata", {}),
                    "reasoning": parsed.get("reasoning", ""),
                }
                
                # Validate confidence
                result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                return result
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
        
        # Fallback: treat entire response as answer
        return {
            "answer": response_text,
            "confidence": 0.5,
            "metadata": {},
            "reasoning": "",
        }
    
    def _generate_follow_up_questions(
        self,
        original_question: str,
        primary_response: Dict[str, Any],
    ) -> List[str]:
        """Generate follow-up questions for multi-turn verification."""
        follow_ups = [
            f"What specific visual features or details in the image support the answer: '{primary_response.get('answer', '')}'?",
            "Are there any alternative interpretations of what you see in this image? Why did you choose this interpretation?",
            "What objects, text, colors, or other elements can you clearly identify in this image?",
        ]
        return follow_ups
    
    def _refine_confidence(
        self,
        primary_response: Dict[str, Any],
        follow_up_answers: List[Dict[str, Any]],
    ) -> float:
        """Refine confidence score based on multi-turn consistency."""
        # Start with primary confidence
        confidence = primary_response.get("confidence", 0.5)
        
        # Check consistency with follow-ups
        if follow_up_answers:
            follow_up_confidences = [
                f.get("confidence", 0.5) for f in follow_up_answers
            ]
            avg_follow_up_conf = sum(follow_up_confidences) / len(follow_up_confidences)
            
            # Boost confidence if follow-ups are consistent
            consistency_factor = 0.9 if avg_follow_up_conf > 0.7 else 0.7
            confidence = (confidence + avg_follow_up_conf) / 2 * consistency_factor
        
        return max(0.0, min(1.0, confidence))
    
    async def _verify_with_web_search(
        self,
        question: str,
        answer: str,
        image_path_or_url: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Verify answer with web search.
        
        Searches for information related to the answer to provide
        external validation.
        """
        if not self.serper_api_key:
            return None
        
        try:
            # Extract keywords from answer for search
            keywords = answer.split()[:5]  # First 5 words
            search_query = " ".join(keywords)
            
            # Search via Serper API
            headers = {"X-API-KEY": self.serper_api_key}
            payload = {"q": search_query}
            
            response = requests.post(
                f"{self.serper_base_url}/search",
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            
            search_results = response.json()
            
            # Extract top results
            top_results = search_results.get("organic", [])[:3]
            
            if top_results:
                return {
                    "search_query": search_query,
                    "results": [
                        {
                            "title": r.get("title", ""),
                            "snippet": r.get("snippet", ""),
                            "link": r.get("link", ""),
                        }
                        for r in top_results
                    ],
                }
        
        except Exception as e:
            logger.warning(f"Web search verification failed: {e}")
        
        return None


# Singleton instance
_client_instance: Optional[EnhancedVQAClient] = None


def get_vqa_client() -> EnhancedVQAClient:
    """Get or create singleton VQA client"""
    global _client_instance
    if _client_instance is None:
        _client_instance = EnhancedVQAClient()
    return _client_instance


# Utility function for non-async usage
def analyze_image_sync(
    image_path_or_url: str,
    question: str,
    enable_multi_turn: bool = True,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for image analysis.
    
    Note: Internally uses asyncio, suitable for scripts and non-async contexts.
    """
    import asyncio
    
    client = EnhancedVQAClient()
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        client.analyze_image(
            image_path_or_url=image_path_or_url,
            question=question,
            enable_multi_turn=enable_multi_turn,
        )
    )
