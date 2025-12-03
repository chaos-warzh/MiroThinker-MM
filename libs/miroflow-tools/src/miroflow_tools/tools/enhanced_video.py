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
Enhanced Video Understanding Client
Provides multi-turn verification, confidence scoring, and multi-provider support for video processing.
Primary provider: Google Gemini 2.0 Flash Exp (best video understanding as of 2025)
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp

# Import shared utilities
from ..utils.file_utils import ensure_local_file, cleanup_temp_file_async

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedVideoClient:
    """Enhanced Video Understanding Client with multi-turn verification and multi-provider support"""

    def __init__(self):
        """Initialize from environment variables"""
        # Provider selection
        self.provider = os.getenv("VIDEO_PROVIDER", "gemini").lower()

        # Google Gemini configuration (Primary - best video understanding)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.gemini_base_url = os.getenv(
            "GEMINI_BASE_URL", 
            "https://generativelanguage.googleapis.com/v1beta"
        )

        # GPT-4o-video configuration (Backup)
        self.gpt4o_video_api_key = os.getenv("GPT4O_VIDEO_API_KEY", "")
        self.gpt4o_video_base_url = os.getenv(
            "GPT4O_VIDEO_BASE_URL", "https://api.openai.com/v1"
        )
        self.gpt4o_video_model = os.getenv("GPT4O_VIDEO_MODEL", "gpt-4o")

        # Qwen-VL-Video configuration (Open-source option)
        self.qwen_video_api_key = os.getenv("QWEN_VIDEO_API_KEY", "")
        self.qwen_video_base_url = os.getenv(
            "QWEN_VIDEO_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.qwen_video_model = os.getenv("QWEN_VIDEO_MODEL", "qwen-vl-max")

        # Custom video API configuration
        self.custom_video_api_key = os.getenv("CUSTOM_VIDEO_API_KEY", "")
        self.custom_video_base_url = os.getenv("CUSTOM_VIDEO_BASE_URL", "")
        self.custom_video_model = os.getenv("CUSTOM_VIDEO_MODEL", "")

        # Frames-based video analysis configuration (uses VQA tool)
        self.frames_api_key = os.getenv("FRAMES_VIDEO_API_KEY", os.getenv("VLLM_OPENAI_API_KEY", ""))
        self.frames_base_url = os.getenv("FRAMES_VIDEO_BASE_URL", os.getenv("VLLM_OPENAI_BASE_URL", ""))
        self.frames_model = os.getenv("FRAMES_VIDEO_MODEL", os.getenv("VLLM_OPENAI_MODEL", "gpt-4o"))
        self.frames_count = int(os.getenv("FRAMES_VIDEO_COUNT", "8"))  # Number of frames to extract
        self.frames_max_size = int(os.getenv("FRAMES_VIDEO_MAX_SIZE", "512"))  # Max dimension for resizing

        # Processing configuration
        self.processing_check_interval = float(os.getenv("VIDEO_PROCESSING_CHECK_INTERVAL", "2.0"))
        self.processing_timeout = float(os.getenv("VIDEO_PROCESSING_TIMEOUT", "300.0"))

        # Feature flags
        self.enable_multi_turn = os.getenv("VIDEO_ENABLE_MULTI_TURN", "true").lower() == "true"
        self.enable_web_search = os.getenv("VIDEO_ENABLE_WEB_SEARCH", "false").lower() == "true"
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Validation
        self._validate_config()

    def _validate_config(self):
        """Validate configuration"""
        if not self.provider:
            raise ValueError("VIDEO_PROVIDER environment variable is required")

        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for gemini provider")
        elif self.provider == "gpt4o_video" and not self.gpt4o_video_api_key:
            raise ValueError("GPT4O_VIDEO_API_KEY is required for gpt4o_video provider")
        elif self.provider == "qwen_video" and not self.qwen_video_api_key:
            raise ValueError("QWEN_VIDEO_API_KEY is required for qwen_video provider")
        elif self.provider == "custom" and not self.custom_video_base_url:
            raise ValueError("CUSTOM_VIDEO_BASE_URL is required for custom provider")
        elif self.provider == "frames" and not self.frames_api_key:
            raise ValueError("FRAMES_VIDEO_API_KEY or VLLM_OPENAI_API_KEY is required for frames provider")

    def _validate_input(self, video_path: str, question: Optional[str] = None) -> None:
        """
        Validate input parameters.
        
        Args:
            video_path: Path to video file or URL
            question: Optional question about the video
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not video_path or not video_path.strip():
            raise ValueError("video_path cannot be empty")
        
        if question is not None and not question.strip():
            raise ValueError("question cannot be empty when provided")
        
        # For local files, basic existence check will be done by ensure_local_file

    async def analyze_video(
        self,
        video_path: str,
        question: Optional[str] = None,
        enable_verification: bool = True,
    ) -> Dict[str, Any]:
        """
        Main entry point for enhanced video understanding.

        Args:
            video_path: Path to video file (local or URL)
            question: Optional question to answer about the video
            enable_verification: Whether to enable multi-turn verification

        Returns:
            Dict with:
                - description: Overall video description
                - answer: Answer to question (if provided)
                - confidence: 0.0-1.0 confidence score
                - metadata: Video features (duration, resolution, fps, key_moments)
                - reasoning: Explanation of confidence score
                - verification_results: Key aspects verified (if multi-turn enabled)
                - verification_sources: External validation (if web search enabled)
        """
        # Validate inputs
        self._validate_input(video_path, question)
        
        logger.info(f"Starting video analysis with provider: {self.provider}")

        # Step 1: Initial video understanding
        initial_result = await self._call_video_service(
            video_path=video_path,
            question=question or "Describe this video in detail, including actions, scenes, objects, and any notable events."
        )

        if "error" in initial_result:
            return initial_result

        description = initial_result.get("description", "")
        answer = initial_result.get("answer", "")
        initial_confidence = initial_result.get("confidence", 0.6)
        metadata = initial_result.get("metadata", {})

        result = {
            "description": description,
            "answer": answer if question else description,
            "confidence": initial_confidence,
            "metadata": metadata,
            "reasoning": f"Initial {self.provider} video analysis",
            "verification_results": [],
            "verification_sources": [],
        }

        # Step 2: Multi-turn verification (if enabled)
        if enable_verification and self.enable_multi_turn and description:
            verification_results = await self._verify_key_aspects(
                video_path=video_path,
                description=description,
                metadata=metadata
            )
            result["verification_results"] = verification_results

            # Refine confidence based on consistency
            refined_confidence = self._refine_confidence(
                initial_confidence=initial_confidence,
                description=description,
                verification_results=verification_results
            )
            result["confidence"] = refined_confidence
            result["reasoning"] = (
                f"Refined from {initial_confidence:.2f} to {refined_confidence:.2f} "
                f"based on {len(verification_results)} aspect verifications"
            )

        # Step 3: Web search validation (if enabled)
        if self.enable_web_search and self.serper_api_key and (answer or description):
            web_validation = await self._verify_with_web_search(answer or description)
            result["verification_sources"] = web_validation
            if web_validation:
                result["confidence"] = min(1.0, result["confidence"] + 0.1)
                result["reasoning"] += f"; Validated with {len(web_validation)} web sources"

        return result

    async def _call_video_service(
        self,
        video_path: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Call appropriate video service based on provider.

        Args:
            video_path: Path to video file
            question: Question about the video

        Returns:
            Dict with description, answer, confidence, metadata
        """
        try:
            if self.provider == "gemini":
                return await self._call_gemini(video_path, question)
            elif self.provider == "gpt4o_video":
                return await self._call_gpt4o_video(video_path, question)
            elif self.provider == "qwen_video":
                return await self._call_qwen_video(video_path, question)
            elif self.provider == "custom":
                return await self._call_custom_video(video_path, question)
            elif self.provider == "frames":
                return await self._call_frames_video(video_path, question)
            else:
                return {"error": f"Unsupported provider: {self.provider}"}
        except Exception as e:
            logger.error(f"Video service call failed: {e}")
            return {"error": str(e)}

    async def _call_gemini(
        self,
        video_path: str,
        question: str
    ) -> Dict[str, Any]:
        """Call Google Gemini API for video understanding"""
        local_path = None
        is_temp = False
        
        try:
            # Import new Google GenAI SDK (supports custom base_url)
            from google import genai
            from google.genai.types import HttpOptions
            
            # Create client with custom base_url support
            http_options = None
            if self.gemini_base_url:
                # Remove trailing /v1beta or /v1 from base_url as SDK will add it
                base_url = self.gemini_base_url.rstrip('/').rstrip('/v1beta').rstrip('/v1')
                http_options = HttpOptions(base_url=base_url)
                self.logger.info(f"Using custom Gemini base_url: {base_url}")
            
            client = genai.Client(
                api_key=self.gemini_api_key,
                http_options=http_options
            )
            
            # Ensure local file (download if URL)
            local_path, is_temp = await ensure_local_file(video_path)

            # Upload video to Gemini
            self.logger.info(f"Uploading video file: {local_path}")
            video_file = client.files.upload(file=local_path)
            
            # Wait for video processing
            self.logger.info(f"Waiting for video processing... Initial state: {video_file.state}")
            if hasattr(video_file.state, 'name'):
                while video_file.state.name == "PROCESSING":
                    await asyncio.sleep(self.processing_check_interval)
                    video_file = client.files.get(name=video_file.name)
                    self.logger.info(f"Video processing state: {video_file.state.name}")

                if video_file.state.name == "FAILED":
                    return {"error": "Video processing failed"}
            
            self.logger.info("Video ready, generating content...")
            
            # Generate content using new SDK
            prompt = f"""{question}

Please provide a detailed analysis in JSON format with:
- description: Overall description of the video
- key_moments: List of important moments with timestamps (if identifiable)
- objects_seen: Main objects/people visible
- actions: Key actions taking place
- scene_changes: Notable scene transitions
- confidence: Your confidence in this analysis (0.0-1.0)
"""

            response = client.models.generate_content(
                model=self.gemini_model,
                contents=[video_file, prompt]
            )
            
            # Parse response
            response_text = response.text
            
            try:
                # Try to extract JSON
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response_text
                
                parsed = json.loads(json_str)
                
                # Extract metadata
                metadata = self._extract_metadata(local_path)
                metadata.update({
                    "key_moments": parsed.get("key_moments", []),
                    "objects_seen": parsed.get("objects_seen", []),
                    "actions": parsed.get("actions", []),
                    "scene_changes": parsed.get("scene_changes", [])
                })
                
                result = {
                    "description": parsed.get("description", response_text),
                    "answer": parsed.get("description", response_text),
                    "confidence": parsed.get("confidence", 0.85),  # Gemini 2.0 is highly reliable
                    "metadata": metadata
                }
                
            except json.JSONDecodeError:
                # Fallback: use raw text
                metadata = self._extract_metadata(local_path)
                result = {
                    "description": response_text,
                    "answer": response_text,
                    "confidence": 0.75,
                    "metadata": metadata
                }

            # Clean up uploaded file using new SDK
            client.files.delete(name=video_file.name)
            
            # Clean up temp file if downloaded (async)
            await cleanup_temp_file_async(local_path, is_temp)

            return result
            
        except Exception as e:
            logger.error(f"Gemini video call failed: {e}")
            # Ensure cleanup on error
            if local_path:
                await cleanup_temp_file_async(local_path, is_temp)
            return {"error": str(e)}

    async def _call_gpt4o_video(
        self,
        video_path: str,
        question: str
    ) -> Dict[str, Any]:
        """Call GPT-4o-video API"""
        # GPT-4o supports video through frames extraction
        # Extract key frames and analyze them
        try:
            # For simplicity, we'll use a placeholder
            # In production, you'd extract frames and send them
            return {
                "description": "GPT-4o-video analysis (requires frame extraction)",
                "answer": "GPT-4o-video currently requires frame-based processing",
                "confidence": 0.5,
                "metadata": self._extract_metadata(video_path)
            }
        except Exception as e:
            return {"error": str(e)}

    async def _call_qwen_video(
        self,
        video_path: str,
        question: str
    ) -> Dict[str, Any]:
        """Call Qwen-VL-Video API"""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                api_key=self.qwen_video_api_key,
                base_url=self.qwen_video_base_url
            )

            # Upload or encode video
            video_url = await self._upload_or_encode_video(video_path)

            response = await client.chat.completions.create(
                model=self.qwen_video_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video_url": video_url},
                            {"type": "text", "text": question}
                        ]
                    }
                ]
            )

            content = response.choices[0].message.content
            metadata = self._extract_metadata(video_path)

            return {
                "description": content,
                "answer": content,
                "confidence": 0.75,
                "metadata": metadata
            }

        except Exception as e:
            return {"error": str(e)}

    async def _call_frames_video(
        self,
        video_path: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Analyze video by extracting key frames and using VQA model.
        
        This approach:
        1. Extracts N evenly-spaced frames from the video
        2. Resizes frames to reduce token usage
        3. Sends all frames to a vision model (like GPT-4o)
        4. Gets a comprehensive analysis
        """
        local_path = None
        is_temp = False
        
        try:
            from openai import AsyncOpenAI
            import cv2
            import base64
            import numpy as np
            
            # Ensure local file (download if URL)
            local_path, is_temp = await ensure_local_file(video_path)
            
            # Extract metadata first
            metadata = self._extract_metadata(local_path)
            
            # Extract frames
            self.logger.info(f"Extracting {self.frames_count} frames from video: {local_path}")
            frames = await self._extract_video_frames(local_path, self.frames_count, self.frames_max_size)
            
            if not frames:
                return {
                    "error": "Failed to extract frames from video",
                    "metadata": metadata
                }
            
            self.logger.info(f"Extracted {len(frames)} frames, sending to vision model...")
            
            # Build message content with all frames
            content = []
            
            # Add instruction text
            content.append({
                "type": "text",
                "text": f"""I'm showing you {len(frames)} frames extracted from a video (evenly spaced throughout the video).
                
Video metadata:
- Duration: {metadata.get('duration_seconds', 'unknown')} seconds
- Resolution: {metadata.get('resolution', 'unknown')}
- FPS: {metadata.get('fps', 'unknown')}

Please analyze these frames to understand the video content and answer the following question:

{question}

Provide a detailed analysis including:
1. Overall description of what's happening in the video
2. Key objects, people, or elements visible
3. Main actions or events
4. Any notable changes or transitions between frames
5. Your confidence level (0.0-1.0) in this analysis"""
            })
            
            # Add each frame as an image
            for i, frame_b64 in enumerate(frames):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                        "detail": "low"  # Use low detail to reduce tokens
                    }
                })
            
            # Call vision model
            client = AsyncOpenAI(
                api_key=self.frames_api_key,
                base_url=self.frames_base_url
            )
            
            response = await client.chat.completions.create(
                model=self.frames_model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=2048
            )
            
            answer = response.choices[0].message.content
            
            # Try to extract confidence from response
            confidence = 0.75  # Default confidence
            if "confidence" in answer.lower():
                import re
                conf_match = re.search(r'confidence[:\s]+([0-9.]+)', answer.lower())
                if conf_match:
                    try:
                        confidence = float(conf_match.group(1))
                        confidence = min(1.0, max(0.0, confidence))
                    except ValueError:
                        pass
            
            # Update metadata with frame info
            metadata["frames_analyzed"] = len(frames)
            metadata["analysis_method"] = "frames_extraction"
            
            # Clean up temp file
            await cleanup_temp_file_async(local_path, is_temp)
            
            return {
                "description": answer,
                "answer": answer,
                "confidence": confidence,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Frames-based video analysis failed: {e}")
            if local_path:
                await cleanup_temp_file_async(local_path, is_temp)
            return {"error": str(e)}
    
    async def _extract_video_frames(
        self,
        video_path: str,
        num_frames: int = 8,
        max_size: int = 512
    ) -> List[str]:
        """
        Extract evenly-spaced frames from video and convert to base64.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            max_size: Maximum dimension for resizing
            
        Returns:
            List of base64-encoded JPEG images
        """
        import cv2
        import base64
        
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return frames
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                logger.error(f"Video has no frames: {video_path}")
                cap.release()
                return frames
            
            # Calculate frame indices to extract (evenly spaced)
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Resize frame to reduce size
                h, w = frame.shape[:2]
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Convert to JPEG and base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frames.append(frame_b64)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
        
        return frames

    async def _call_custom_video(
        self,
        video_path: str,
        question: str
    ) -> Dict[str, Any]:
        """Call custom video API (OpenAI-compatible)"""
        try:
            # Most OpenAI-compatible APIs don't support video content type
            # We need to either:
            # 1. Extract keyframes and send as images
            # 2. Send video URL if API supports it
            # 3. Return placeholder with metadata
            
            # For now, extract metadata and return a placeholder
            metadata = self._extract_metadata(video_path)
            
            # Try to send as file URL (some APIs might support this)
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.custom_video_api_key}",
                "Content-Type": "application/json",
            }
            
            # Try simple text-only request first (to test API connectivity)
            payload = {
                "model": self.custom_video_model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"This is a video file at {video_path}. {question}\n\nNote: Video analysis not fully supported yet. Metadata: {metadata}"
                    }
                ],
                "max_tokens": 1024,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.custom_video_base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return {
                            "error": f"API request failed: {resp.status} - {error_text}",
                            "description": f"Video analysis failed. Metadata: {metadata}",
                            "confidence": 0.0,
                            "metadata": metadata
                        }
                    
                    result = await resp.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    return {
                        "description": content,
                        "answer": content,
                        "confidence": 0.5,  # Lower confidence since we can't fully analyze video
                        "metadata": metadata
                    }

        except Exception as e:
            logger.error(f"Custom video API call failed: {e}")
            metadata = self._extract_metadata(video_path)
            return {
                "error": str(e),
                "description": f"Video analysis failed: {e}. File metadata: {metadata}",
                "confidence": 0.0,
                "metadata": metadata
            }

    async def _verify_key_aspects(
        self,
        video_path: str,
        description: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Re-analyze key aspects for verification.

        Strategy:
        1. Identify 3 key aspects to verify (actions, objects, temporal sequence)
        2. Ask specific questions about each
        3. Check consistency with original description
        """
        verification_results = []

        questions = [
            "What are the main actions or activities happening in this video?",
            "What objects, people, or animals are visible in the video?",
            "Describe the sequence of events from beginning to end."
        ]

        for i, question in enumerate(questions):
            try:
                result = await self._call_video_service(
                    video_path=video_path,
                    question=question
                )
                verification_results.append({
                    "aspect_id": i + 1,
                    "question": question,
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.5)
                })
            except Exception as e:
                logger.warning(f"Aspect verification {i+1} failed: {e}")

        return verification_results

    def _refine_confidence(
        self,
        initial_confidence: float,
        description: str,
        verification_results: List[Dict[str, Any]]
    ) -> float:
        """
        Refine confidence based on verification consistency.

        Args:
            initial_confidence: Initial confidence from primary analysis
            description: Original description
            verification_results: Results from aspect verifications

        Returns:
            Refined confidence score (0.0-1.0)
        """
        if not verification_results:
            return initial_confidence

        # Check consistency: Do verification answers align with description?
        consistency_scores = []

        for verification in verification_results:
            answer = verification.get("answer", "").lower()
            # Simple keyword overlap check
            description_words = set(description.lower().split())
            answer_words = set(answer.split())
            overlap = len(description_words & answer_words)
            total = len(answer_words)
            
            if total > 0:
                consistency = overlap / total
                consistency_scores.append(consistency)

        if not consistency_scores:
            return initial_confidence

        avg_consistency = sum(consistency_scores) / len(consistency_scores)

        # Adjust confidence
        if avg_consistency > 0.6:
            # High consistency: boost confidence
            return min(1.0, initial_confidence + 0.15)
        elif avg_consistency > 0.4:
            # Medium consistency: slight boost
            return min(1.0, initial_confidence + 0.05)
        else:
            # Low consistency: reduce confidence
            return max(0.0, initial_confidence - 0.15)

    async def _verify_with_web_search(
        self,
        content: str
    ) -> List[Dict[str, str]]:
        """
        Verify video content with web search.

        Args:
            content: Video description or answer to verify

        Returns:
            List of validation sources with titles and URLs
        """
        if not self.serper_api_key:
            return []

        # Extract key phrases for search (first 100 chars)
        search_query = content[:100] if len(content) < 200 else content[:100]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": self.serper_api_key,
                        "Content-Type": "application/json"
                    },
                    json={"q": search_query, "num": 3}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get("organic", [])[:3]:
                            results.append({
                                "title": item.get("title", ""),
                                "url": item.get("link", ""),
                                "snippet": item.get("snippet", "")
                            })
                        return results
        except Exception as e:
            logger.warning(f"Web search validation failed: {e}")

        return []

    def _get_video_extension(self, url: str, content_type: str = None) -> str:
        """Determine video file extension"""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]
        for ext in video_extensions:
            if path.endswith(ext):
                return ext

        if content_type:
            content_type = content_type.lower()
            if "mp4" in content_type:
                return ".mp4"
            elif "webm" in content_type:
                return ".webm"
            elif "quicktime" in content_type or "mov" in content_type:
                return ".mov"

        return ".mp4"

    def extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Public method to extract video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with metadata:
            - duration_seconds: Video duration
            - width: Video width in pixels
            - height: Video height in pixels
            - fps: Frames per second
            - file_size_bytes: File size
        """
        return self._extract_metadata(video_path)

    def _extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata: duration, resolution, fps, etc."""
        metadata = {}

        try:
            # Try using moviepy
            from moviepy.editor import VideoFileClip
            
            clip = VideoFileClip(video_path)
            metadata["duration_seconds"] = round(clip.duration, 2)
            metadata["fps"] = clip.fps
            metadata["resolution"] = f"{clip.w}x{clip.h}"
            metadata["width"] = clip.w
            metadata["height"] = clip.h
            clip.close()
            
        except Exception:
            # Fallback: try opencv
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
                metadata["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata["resolution"] = f"{metadata['width']}x{metadata['height']}"
                if metadata["fps"] > 0:
                    metadata["duration_seconds"] = round(metadata["frame_count"] / metadata["fps"], 2)
                cap.release()
            except Exception as e:
                logger.warning(f"Failed to extract metadata: {e}")

        # File size
        if os.path.exists(video_path):
            metadata["file_size_bytes"] = os.path.getsize(video_path)

        return metadata

    def _upload_or_encode_video(self, video_path: str) -> str:
        """For providers that need upload or encoding"""
        # Simplified: return file path or data URL
        if os.path.exists(video_path):
            return f"file://{os.path.abspath(video_path)}"
        else:
            return video_path


# Singleton instance
_client_instance: Optional[EnhancedVideoClient] = None


def get_video_client() -> EnhancedVideoClient:
    """Get or create singleton video client"""
    global _client_instance
    if _client_instance is None:
        _client_instance = EnhancedVideoClient()
    return _client_instance
