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
Enhanced Audio MCP Server
Exposes advanced audio understanding tools via FastMCP protocol.
"""

import json
import logging
import os
from typing import Optional

from fastmcp import FastMCP

# Import the enhanced audio client from miroflow_tools.tools package
from miroflow_tools.tools.enhanced_audio import get_audio_client

# Initialize FastMCP server
mcp = FastMCP("audio-mcp-server-enhanced")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@mcp.tool()
async def audio_understanding_advanced(
    audio_path_or_url: str,
    question: Optional[str] = None,
    enable_verification: bool = True
) -> str:
    """
    Advanced audio understanding with multi-turn verification and confidence scoring.

    This tool provides:
    - High-quality transcription with confidence scores
    - Optional question answering about audio content
    - Multi-turn verification for accuracy (3 follow-up checks)
    - Audio metadata extraction (duration, sample rate, etc.)
    - Web search validation (if enabled)

    Args:
        audio_path_or_url: Local path or URL to audio file (mp3, wav, m4a, etc.)
        question: Optional question to answer about the audio content
        enable_verification: Whether to perform multi-turn verification (default: True)
                            Set to False for faster processing when high accuracy is not critical

    Returns:
        JSON string with:
        {
            "transcript": "Full transcription text",
            "answer": "Answer to question (if provided)",
            "confidence": 0.85,  # 0.0-1.0 score
            "metadata": {
                "duration_seconds": 120.5,
                "sample_rate": 44100,
                "file_size_bytes": 1024000
            },
            "reasoning": "Explanation of confidence score",
            "verification_segments": [
                {
                    "segment_id": 1,
                    "question": "What are the main topics?",
                    "answer": "...",
                    "confidence": 0.9
                }
            ],
            "verification_sources": [
                {"title": "...", "url": "...", "snippet": "..."}
            ]
        }

    Usage recommendations:
    - Use enable_verification=true for critical transcriptions (e.g., interviews, lectures)
    - Check confidence score: >= 0.7 is good, < 0.4 may need manual review
    - If confidence is low, consider re-recording or using different audio quality
    """
    try:
        client = get_audio_client()
        result = await client.analyze_audio(
            audio_path=audio_path_or_url,
            question=question,
            enable_verification=enable_verification
        )

        # Return as JSON string for LLM parsing
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Audio understanding failed: {e}")
        return json.dumps({
            "error": str(e),
            "transcript": "",
            "confidence": 0.0
        })


@mcp.tool()
async def audio_quick_transcription(audio_path_or_url: str) -> str:
    """
    Fast single-pass audio transcription without verification.

    Use this tool when:
    - Speed is more important than perfect accuracy
    - Audio quality is known to be good
    - Content is not mission-critical

    Args:
        audio_path_or_url: Local path or URL to audio file

    Returns:
        JSON string with:
        {
            "transcript": "Transcribed text",
            "confidence": 0.75,  # Base confidence without verification
            "metadata": {...}
        }
    """
    try:
        client = get_audio_client()
        result = await client.analyze_audio(
            audio_path=audio_path_or_url,
            question=None,
            enable_verification=False  # Disable multi-turn for speed
        )

        return json.dumps({
            "transcript": result.get("transcript", ""),
            "confidence": result.get("confidence", 0.5),
            "metadata": result.get("metadata", {})
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Quick transcription failed: {e}")
        return json.dumps({
            "error": str(e),
            "transcript": ""
        })


@mcp.tool()
async def audio_extract_metadata(audio_path_or_url: str) -> str:
    """
    Extract audio metadata without transcription.

    Use this tool to get audio file information:
    - Duration in seconds
    - Sample rate (Hz)
    - Number of channels
    - File size
    - Format/codec information (if available)

    Args:
        audio_path_or_url: Local path or URL to audio file

    Returns:
        JSON string with metadata:
        {
            "duration_seconds": 120.5,
            "sample_rate": 44100,
            "channels": 2,
            "file_size_bytes": 1024000,
            "format": "mp3"
        }
    """
    try:
        from miroflow_tools.utils.file_utils import ensure_local_file, cleanup_temp_file_async
        
        client = get_audio_client()

        # Download if URL (use shared utility)
        local_path, is_temp = await ensure_local_file(audio_path_or_url)

        try:
            metadata = client.extract_metadata(local_path)
            return json.dumps(metadata, ensure_ascii=False, indent=2)
        finally:
            # Clean up temp file if downloaded (async)
            await cleanup_temp_file_async(local_path, is_temp)

    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def audio_question_answering_enhanced(
    audio_path_or_url: str,
    question: str
) -> str:
    """
    Answer questions about audio content with confidence scoring.

    This tool:
    1. Analyzes the audio to understand content
    2. Answers the specific question
    3. Provides confidence score
    4. Includes reasoning for the answer

    Use this tool when:
    - You need to extract specific information from audio
    - Understanding context and nuance is important
    - You want confidence assessment of the answer

    Args:
        audio_path_or_url: Local path or URL to audio file
        question: Specific question about the audio content
                 Examples:
                 - "Who is the speaker?"
                 - "What is the main topic discussed?"
                 - "Are there any specific dates or numbers mentioned?"
                 - "What is the speaker's emotional tone?"

    Returns:
        JSON string with:
        {
            "question": "Your question",
            "answer": "Detailed answer based on audio",
            "confidence": 0.85,
            "reasoning": "Why this confidence score",
            "transcript_excerpt": "Relevant portion of transcript supporting the answer",
            "metadata": {...}
        }
    """
    try:
        client = get_audio_client()
        result = await client.analyze_audio(
            audio_path=audio_path_or_url,
            question=question,
            enable_verification=True  # Enable verification for QA
        )

        # Extract relevant portions for QA response
        answer = result.get("answer", result.get("transcript", ""))
        
        return json.dumps({
            "question": question,
            "answer": answer,
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("reasoning", ""),
            "transcript_excerpt": result.get("transcript", "")[:500] + "..." if len(result.get("transcript", "")) > 500 else result.get("transcript", ""),
            "metadata": result.get("metadata", {}),
            "verification_segments": result.get("verification_segments", [])
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Audio QA failed: {e}")
        return json.dumps({
            "question": question,
            "answer": "",
            "error": str(e),
            "confidence": 0.0
        })


if __name__ == "__main__":
    # Run MCP server
    mcp.run(transport="stdio")
