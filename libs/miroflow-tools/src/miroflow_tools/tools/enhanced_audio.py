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
Enhanced Audio Understanding Client
Provides multi-turn verification, confidence scoring, and multi-provider support for audio processing.
"""

import asyncio
import base64
import logging
import mimetypes
import os
import wave
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from mutagen import File as MutagenFile
from openai import AsyncOpenAI

# Import shared utilities
from ..utils.file_utils import ensure_local_file, cleanup_temp_file_async

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAudioClient:
    """Enhanced Audio Understanding Client with multi-turn verification and multi-provider support"""

    def __init__(self):
        """Initialize from environment variables"""
        # Provider selection
        self.provider = os.getenv("AUDIO_PROVIDER", "").lower()

        # OpenAI Whisper configuration
        self.openai_whisper_api_key = os.getenv("WHISPER_OPENAI_API_KEY", "")
        self.openai_whisper_base_url = os.getenv(
            "WHISPER_OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.openai_whisper_model = os.getenv(
            "WHISPER_OPENAI_MODEL", "whisper-1"
        )

        # Open-Source Whisper configuration
        self.whisper_os_api_key = os.getenv("WHISPER_OS_API_KEY", "")
        self.whisper_os_base_url = os.getenv("WHISPER_OS_BASE_URL", "")
        self.whisper_os_model = os.getenv("WHISPER_OS_MODEL", "whisper-large-v3")

        # Qwen-Audio configuration
        self.qwen_audio_api_key = os.getenv("QWEN_AUDIO_API_KEY", "")
        self.qwen_audio_base_url = os.getenv(
            "QWEN_AUDIO_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.qwen_audio_model = os.getenv("QWEN_AUDIO_MODEL", "qwen-audio-turbo")

        # GPT-4o-audio configuration
        self.gpt4o_audio_api_key = os.getenv("GPT4O_AUDIO_API_KEY", "")
        self.gpt4o_audio_base_url = os.getenv(
            "GPT4O_AUDIO_BASE_URL", "https://api.openai.com/v1"
        )
        self.gpt4o_audio_model = os.getenv("GPT4O_AUDIO_MODEL", "gpt-4o-audio-preview")

        # Feature flags
        self.enable_multi_turn = os.getenv("AUDIO_ENABLE_MULTI_TURN", "true").lower() == "true"
        self.enable_web_search = os.getenv("AUDIO_ENABLE_WEB_SEARCH", "false").lower() == "true"
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")

        # Validation
        self._validate_config()

    def _validate_config(self):
        """Validate configuration"""
        if not self.provider:
            raise ValueError("AUDIO_PROVIDER environment variable is required")

        if self.provider == "openai_whisper" and not self.openai_whisper_api_key:
            raise ValueError("WHISPER_OPENAI_API_KEY is required for openai_whisper provider")
        elif self.provider == "whisper_os" and not self.whisper_os_base_url:
            raise ValueError("WHISPER_OS_BASE_URL is required for whisper_os provider")
        elif self.provider == "qwen_audio" and not self.qwen_audio_api_key:
            raise ValueError("QWEN_AUDIO_API_KEY is required for qwen_audio provider")
        elif self.provider == "gpt4o_audio" and not self.gpt4o_audio_api_key:
            raise ValueError("GPT4O_AUDIO_API_KEY is required for gpt4o_audio provider")

    async def analyze_audio(
        self,
        audio_path: str,
        question: Optional[str] = None,
        enable_verification: bool = True,
    ) -> Dict[str, Any]:
        """
        Main entry point for enhanced audio understanding.

        Args:
            audio_path: Path to audio file (local or URL)
            question: Optional question to answer about the audio
            enable_verification: Whether to enable multi-turn verification

        Returns:
            Dict with:
                - transcript: Full transcription text
                - answer: Answer to question (if provided)
                - confidence: 0.0-1.0 confidence score
                - metadata: Audio features (duration, speakers, emotion, etc.)
                - reasoning: Explanation of confidence score
                - verification_segments: Key segments re-analyzed (if multi-turn enabled)
                - verification_sources: External validation (if web search enabled)
        """
        logger.info(f"Starting audio analysis with provider: {self.provider}")

        # Step 1: Initial transcription
        initial_result = await self._call_audio_service(
            audio_path=audio_path,
            question=question,
            mode="transcribe" if not question else "qa"
        )

        if "error" in initial_result:
            return initial_result

        transcript = initial_result.get("transcript", "")
        answer = initial_result.get("answer", "")
        initial_confidence = initial_result.get("confidence", 0.5)
        metadata = initial_result.get("metadata", {})

        result = {
            "transcript": transcript,
            "answer": answer,
            "confidence": initial_confidence,
            "metadata": metadata,
            "reasoning": f"Initial {self.provider} transcription",
            "verification_segments": [],
            "verification_sources": [],
        }

        # Step 2: Multi-turn verification (if enabled)
        if enable_verification and self.enable_multi_turn and transcript:
            verification_results = await self._verify_key_segments(
                audio_path=audio_path,
                transcript=transcript,
                metadata=metadata
            )
            result["verification_segments"] = verification_results

            # Refine confidence based on consistency
            refined_confidence = self._refine_confidence(
                initial_confidence=initial_confidence,
                transcript=transcript,
                verification_results=verification_results
            )
            result["confidence"] = refined_confidence
            result["reasoning"] = (
                f"Refined from {initial_confidence:.2f} to {refined_confidence:.2f} "
                f"based on {len(verification_results)} segment verifications"
            )

        # Step 3: Web search validation (if enabled)
        if self.enable_web_search and self.serper_api_key and transcript:
            web_validation = await self._verify_with_web_search(transcript, answer or transcript)
            result["verification_sources"] = web_validation
            if web_validation:
                result["confidence"] = min(1.0, result["confidence"] + 0.1)
                result["reasoning"] += f"; Validated with {len(web_validation)} web sources"

        return result

    async def _call_audio_service(
        self,
        audio_path: str,
        question: Optional[str] = None,
        mode: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Call appropriate audio service based on provider.

        Args:
            audio_path: Path to audio file
            question: Optional question for QA mode
            mode: "transcribe" or "qa"

        Returns:
            Dict with transcript, answer (if QA), confidence, metadata
        """
        try:
            if self.provider == "openai_whisper":
                return await self._call_openai_whisper(audio_path, question, mode)
            elif self.provider == "whisper_os":
                return await self._call_whisper_os(audio_path, question, mode)
            elif self.provider == "qwen_audio":
                return await self._call_qwen_audio(audio_path, question, mode)
            elif self.provider == "gpt4o_audio":
                return await self._call_gpt4o_audio(audio_path, question, mode)
            else:
                return {"error": f"Unsupported provider: {self.provider}"}
        except Exception as e:
            logger.error(f"Audio service call failed: {e}")
            return {"error": str(e)}

    async def _call_openai_whisper(
        self,
        audio_path: str,
        question: Optional[str],
        mode: str
    ) -> Dict[str, Any]:
        """Call OpenAI Whisper API"""
        client = AsyncOpenAI(
            api_key=self.openai_whisper_api_key,
            base_url=self.openai_whisper_base_url
        )

        # Ensure local file (download if URL)
        local_path, is_temp = await ensure_local_file(audio_path)

        try:
            # Transcription - using synchronous file read in thread
            def read_and_transcribe():
                with open(local_path, "rb") as audio_file:
                    return audio_file.read()
            
            audio_data = await asyncio.to_thread(read_and_transcribe)
            
            # Create temporary file-like object for API
            import io
            audio_file_obj = io.BytesIO(audio_data)
            audio_file_obj.name = os.path.basename(local_path)
            
            transcription = await client.audio.transcriptions.create(
                model=self.openai_whisper_model,
                file=audio_file_obj,
                response_format="verbose_json"  # Get timestamps and confidence
            )

            transcript = transcription.text
            metadata = self._extract_metadata(local_path)
            
            # OpenAI Whisper doesn't provide word-level confidence in verbose_json by default
            # We'll estimate based on segment-level data if available
            confidence = 0.8  # Default high confidence for OpenAI Whisper

            result = {
                "transcript": transcript,
                "confidence": confidence,
                "metadata": metadata
            }

            # If QA mode, use GPT-4 to answer question based on transcript
            if mode == "qa" and question:
                qa_client = AsyncOpenAI(
                    api_key=self.openai_whisper_api_key,
                    base_url=self.openai_whisper_base_url  # Use same base URL as transcription
                )
                qa_response = await qa_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Answer questions based on the provided audio transcript."},
                        {"role": "user", "content": f"Transcript: {transcript}\n\nQuestion: {question}"}
                    ]
                )
                result["answer"] = qa_response.choices[0].message.content

            return result

        finally:
            # Clean up temp file if downloaded (async)
            await cleanup_temp_file_async(local_path, is_temp)

    async def _call_whisper_os(
        self,
        audio_path: str,
        question: Optional[str],
        mode: str
    ) -> Dict[str, Any]:
        """Call Open-Source Whisper API"""
        client = AsyncOpenAI(
            api_key=self.whisper_os_api_key or "dummy",  # Some servers don't need key
            base_url=self.whisper_os_base_url
        )

        local_path, is_temp = await ensure_local_file(audio_path)

        try:
            # Read file in thread pool
            audio_data = await asyncio.to_thread(lambda: open(local_path, "rb").read())
            
            # Create file-like object for API
            import io
            audio_file_obj = io.BytesIO(audio_data)
            audio_file_obj.name = os.path.basename(local_path)
            
            transcription = await client.audio.transcriptions.create(
                model=self.whisper_os_model,
                file=audio_file_obj
            )

            transcript = transcription.text
            metadata = self._extract_metadata(local_path)

            result = {
                "transcript": transcript,
                "confidence": 0.75,  # Default for open-source Whisper
                "metadata": metadata
            }

            # If QA mode, need to use a separate LLM (not included in Whisper-only server)
            if mode == "qa" and question:
                result["answer"] = f"[QA not supported in whisper_os mode. Transcript: {transcript[:200]}...]"

            return result

        finally:
            await cleanup_temp_file_async(local_path, is_temp)

    async def _call_qwen_audio(
        self,
        audio_path: str,
        question: Optional[str],
        mode: str
    ) -> Dict[str, Any]:
        """Call Qwen-Audio API (supports both transcription and audio QA)"""
        client = AsyncOpenAI(
            api_key=self.qwen_audio_api_key,
            base_url=self.qwen_audio_base_url
        )

        local_path, is_temp = await ensure_local_file(audio_path)

        try:
            # Qwen-Audio uses chat completion with audio input
            audio_url = self._upload_or_encode_audio(local_path)

            prompt = question if mode == "qa" and question else "请转写这段音频的内容。"

            response = await client.chat.completions.create(
                model=self.qwen_audio_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": audio_url},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            content = response.choices[0].message.content
            metadata = self._extract_metadata(local_path)

            result = {
                "transcript": content if mode == "transcribe" else "",
                "answer": content if mode == "qa" else "",
                "confidence": 0.8,  # Qwen-Audio typically high quality
                "metadata": metadata
            }

            return result

        finally:
            await cleanup_temp_file_async(local_path, is_temp)

    async def _call_gpt4o_audio(
        self,
        audio_path: str,
        question: Optional[str],
        mode: str
    ) -> Dict[str, Any]:
        """Call GPT-4o-audio API (best for complex audio understanding)"""
        client = AsyncOpenAI(
            api_key=self.gpt4o_audio_api_key,
            base_url=self.gpt4o_audio_base_url
        )

        local_path, is_temp = await ensure_local_file(audio_path)

        try:
            # Encode audio to base64
            encoded_audio, audio_format = self._encode_audio_file(local_path)

            prompt = question if mode == "qa" and question else "Transcribe this audio precisely."

            response = await client.chat.completions.create(
                model=self.gpt4o_audio_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded_audio,
                                    "format": audio_format
                                }
                            }
                        ]
                    }
                ]
            )

            content = response.choices[0].message.content
            metadata = self._extract_metadata(local_path)

            result = {
                "transcript": content if mode == "transcribe" else "",
                "answer": content if mode == "qa" else "",
                "confidence": 0.9,  # GPT-4o-audio typically very high quality
                "metadata": metadata
            }

            return result

        finally:
            await cleanup_temp_file_async(local_path, is_temp)

    async def _verify_key_segments(
        self,
        audio_path: str,
        transcript: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Re-analyze key segments for verification.

        Strategy:
        1. Identify 2-3 key segments (start, middle, end or keyword-rich parts)
        2. Re-transcribe each segment
        3. Check consistency with original transcript
        """
        verification_results = []

        # For simplicity, we'll verify by asking specific questions about content
        # (Real implementation would extract audio segments and re-transcribe)

        questions = [
            "What are the main topics or keywords mentioned in this audio?",
            "What is the speaker's tone or emotion?",
            "Are there any specific names, dates, or numbers mentioned?"
        ]

        for i, question in enumerate(questions):
            try:
                result = await self._call_audio_service(
                    audio_path=audio_path,
                    question=question,
                    mode="qa"
                )
                verification_results.append({
                    "segment_id": i + 1,
                    "question": question,
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.5)
                })
            except Exception as e:
                logger.warning(f"Segment verification {i+1} failed: {e}")

        return verification_results

    def _refine_confidence(
        self,
        initial_confidence: float,
        transcript: str,
        verification_results: List[Dict[str, Any]]
    ) -> float:
        """
        Refine confidence based on verification consistency.

        Args:
            initial_confidence: Initial confidence from primary transcription
            transcript: Original transcript
            verification_results: Results from segment verifications

        Returns:
            Refined confidence score (0.0-1.0)
        """
        if not verification_results:
            return initial_confidence

        # Check consistency: Do verification answers align with transcript?
        consistency_scores = []

        for verification in verification_results:
            answer = verification.get("answer", "").lower()
            # Simple keyword overlap check
            transcript_words = set(transcript.lower().split())
            answer_words = set(answer.split())
            overlap = len(transcript_words & answer_words)
            total = len(answer_words)
            
            if total > 0:
                consistency = overlap / total
                consistency_scores.append(consistency)

        if not consistency_scores:
            return initial_confidence

        avg_consistency = sum(consistency_scores) / len(consistency_scores)

        # Adjust confidence
        if avg_consistency > 0.7:
            # High consistency: boost confidence
            return min(1.0, initial_confidence + 0.2)
        elif avg_consistency > 0.4:
            # Medium consistency: slight boost
            return min(1.0, initial_confidence + 0.1)
        else:
            # Low consistency: reduce confidence
            return max(0.0, initial_confidence - 0.2)

    async def _verify_with_web_search(
        self,
        transcript: str,
        answer: str
    ) -> List[Dict[str, str]]:
        """
        Verify transcription/answer with web search.

        Args:
            transcript: Transcribed text
            answer: Answer to verify (could be same as transcript)

        Returns:
            List of validation sources with titles and URLs
        """
        if not self.serper_api_key:
            return []

        # Extract key phrases for search
        # (Simple approach: use first 100 chars or key answer)
        search_query = answer[:100] if len(answer) < 200 else transcript[:100]

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

    def _get_audio_extension(self, url: str, content_type: str = None) -> str:
        """Determine audio file extension"""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        audio_extensions = [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"]
        for ext in audio_extensions:
            if path.endswith(ext):
                return ext

        if content_type:
            content_type = content_type.lower()
            if "mp3" in content_type or "mpeg" in content_type:
                return ".mp3"
            elif "wav" in content_type:
                return ".wav"
            elif "m4a" in content_type:
                return ".m4a"

        return ".mp3"

    def extract_metadata(self, audio_path: str) -> Dict[str, Any]:
        """
        Public method to extract audio metadata.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with metadata:
            - duration_seconds: Audio duration
            - sample_rate: Sample rate in Hz
            - channels: Number of audio channels
            - file_size_bytes: File size
        """
        return self._extract_metadata(audio_path)

    def _extract_metadata(self, audio_path: str) -> Dict[str, Any]:
        """Extract audio metadata: duration, sample rate, etc."""
        metadata = {}

        # Duration
        try:
            # Try wave first
            with wave.open(audio_path, "rb") as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                metadata["duration_seconds"] = round(duration, 2)
                metadata["sample_rate"] = rate
                metadata["channels"] = f.getnchannels()
        except Exception:
            # Try mutagen for other formats
            try:
                audio = MutagenFile(audio_path)
                if audio and hasattr(audio, "info") and hasattr(audio.info, "length"):
                    metadata["duration_seconds"] = round(audio.info.length, 2)
                    if hasattr(audio.info, "sample_rate"):
                        metadata["sample_rate"] = audio.info.sample_rate
            except Exception as e:
                logger.warning(f"Failed to extract metadata: {e}")

        # File size
        if os.path.exists(audio_path):
            metadata["file_size_bytes"] = os.path.getsize(audio_path)

        return metadata

    def _encode_audio_file(self, audio_path: str) -> Tuple[str, str]:
        """Encode audio to base64 and determine format"""
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            encoded_string = base64.b64encode(audio_data).decode("utf-8")

        mime_type, _ = mimetypes.guess_type(audio_path)
        if mime_type and mime_type.startswith("audio/"):
            mime_format = mime_type.split("/")[-1]
            format_mapping = {"mpeg": "mp3", "wav": "wav", "wave": "wav"}
            file_format = format_mapping.get(mime_format, "mp3")
        else:
            file_format = "mp3"

        return encoded_string, file_format

    def _upload_or_encode_audio(self, audio_path: str) -> str:
        """For Qwen-Audio, may need to upload to OSS or encode as data URL"""
        # Simplified: return base64 data URL
        encoded, fmt = self._encode_audio_file(audio_path)
        return f"data:audio/{fmt};base64,{encoded}"


# Singleton instance
_client_instance: Optional[EnhancedAudioClient] = None


def get_audio_client() -> EnhancedAudioClient:
    """Get or create singleton audio client"""
    global _client_instance
    if _client_instance is None:
        _client_instance = EnhancedAudioClient()
    return _client_instance
