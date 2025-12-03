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
MCP Server for Enhanced Video Understanding Tools
Exposes 4 video tools for LLM agents via FastMCP
"""

import json
import logging
import os
from pathlib import Path

# Load environment variables from .env file
# Try multiple possible locations for .env file
def load_env():
    """Load .env file from various possible locations"""
    try:
        from dotenv import load_dotenv
        
        # Possible .env locations (in order of priority)
        possible_paths = [
            Path.cwd() / ".env",  # Current working directory
            Path(__file__).parent.parent.parent.parent.parent.parent / "apps" / "miroflow-agent" / ".env",
            Path.home() / ".miroflow" / ".env",
        ]
        
        for env_path in possible_paths:
            if env_path.exists():
                load_dotenv(env_path)
                logging.info(f"Loaded .env from: {env_path}")
                return True
        
        # Also try load_dotenv() which searches parent directories
        load_dotenv()
        return True
    except ImportError:
        logging.warning("python-dotenv not installed, skipping .env loading")
        return False

load_env()

from fastmcp import FastMCP

# Import video client singleton
from miroflow_tools.tools.enhanced_video import get_video_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("Enhanced Video Understanding")


@mcp.tool()
async def video_understanding_advanced(
    video_path: str,
    question: str = None,
    enable_verification: bool = True,
) -> str:
    """
    Advanced video understanding with multi-turn verification.
    
    Use this for complex video analysis requiring high accuracy. Includes:
    - Detailed scene analysis with temporal understanding
    - Multi-turn verification of key aspects (actions, objects, sequence)
    - Confidence scoring based on consistency
    - Optional web search validation
    - Key moments extraction with timestamps
    
    Args:
        video_path: Path to video file (local path or URL)
        question: Optional specific question about the video. If not provided, performs general analysis.
        enable_verification: Whether to enable multi-turn verification (default: True). 
                           Improves accuracy but takes longer.
    
    Returns:
        JSON string with:
        - description: Overall video description
        - answer: Answer to the question (or description if no question)
        - confidence: 0.0-1.0 confidence score
        - metadata: Video features (duration, resolution, fps, key_moments)
        - reasoning: Explanation of confidence
        - verification_results: Aspect verifications performed
    
    Example:
        >>> result = await video_understanding_advanced(
        ...     video_path="/path/to/video.mp4",
        ...     question="What sport is being played in this video?"
        ... )
    
    Best for:
    - Detailed action recognition
    - Scene understanding with temporal context
    - Multi-object tracking
    - Event sequence analysis
    """
    try:
        client = get_video_client()
        result = await client.analyze_video(
            video_path=video_path,
            question=question,
            enable_verification=enable_verification
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"video_understanding_advanced failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def video_quick_analysis(
    video_path: str,
    question: str = None,
) -> str:
    """
    Quick video analysis without multi-turn verification.
    
    Fast single-pass analysis. Use when:
    - Speed is more important than maximum accuracy
    - Video content is straightforward
    - You need a quick overview
    
    Args:
        video_path: Path to video file (local path or URL)
        question: Optional question about the video
    
    Returns:
        JSON string with:
        - description: Video description
        - answer: Answer to question (if provided)
        - confidence: Initial confidence (typically 0.6-0.8)
        - metadata: Basic video info (duration, resolution, fps)
    
    Example:
        >>> result = await video_quick_analysis(
        ...     video_path="https://example.com/video.mp4",
        ...     question="Is this video indoors or outdoors?"
        ... )
    
    Best for:
    - Quick previews
    - Simple yes/no questions
    - Initial exploration
    """
    try:
        client = get_video_client()
        result = await client.analyze_video(
            video_path=video_path,
            question=question,
            enable_verification=False  # Disable multi-turn for speed
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"video_quick_analysis failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def video_extract_keyframes(
    video_path: str,
    max_keyframes: int = 10
) -> str:
    """
    Extract key moments and metadata from video without deep analysis.
    
    Provides technical information and structure without detailed understanding:
    - Duration, resolution, fps
    - Key moments identification (scene changes, important frames)
    - Temporal structure
    
    Args:
        video_path: Path to video file (local path or URL)
        max_keyframes: Maximum number of key moments to identify (default: 10)
    
    Returns:
        JSON string with:
        - metadata: Video technical specs
        - key_moments: List of timestamps with brief descriptions
        - duration_seconds: Total video length
        - resolution: Video dimensions
        - fps: Frames per second
    
    Example:
        >>> result = await video_extract_keyframes(
        ...     video_path="/videos/lecture.mp4",
        ...     max_keyframes=5
        ... )
    
    Best for:
    - Checking video properties before analysis
    - Finding important timestamps
    - Video preprocessing
    - Quality validation
    """
    try:
        client = get_video_client()
        
        # Get metadata through quick analysis
        result = await client.analyze_video(
            video_path=video_path,
            question="Identify the key moments and scene changes in this video.",
            enable_verification=False
        )
        
        # Focus on metadata and key_moments
        response = {
            "metadata": result.get("metadata", {}),
            "key_moments": result.get("metadata", {}).get("key_moments", [])[:max_keyframes],
            "duration_seconds": result.get("metadata", {}).get("duration_seconds"),
            "resolution": result.get("metadata", {}).get("resolution"),
            "fps": result.get("metadata", {}).get("fps"),
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"video_extract_keyframes failed: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def video_temporal_qa(
    video_path: str,
    question: str,
    start_time: float = None,
    end_time: float = None,
    enable_verification: bool = True
) -> str:
    """
    Answer questions about specific time ranges in the video.
    
    Focuses analysis on a particular temporal segment. Useful for:
    - Analyzing specific scenes
    - Understanding action sequences in a timeframe
    - Detailed examination of video segments
    
    Args:
        video_path: Path to video file (local path or URL)
        question: Question about the video or time range
        start_time: Start time in seconds (optional, analyzes from beginning if not specified)
        end_time: End time in seconds (optional, analyzes until end if not specified)
        enable_verification: Whether to use multi-turn verification (default: True)
    
    Returns:
        JSON string with:
        - answer: Answer specific to the time range
        - confidence: 0.0-1.0 confidence score
        - time_range: The analyzed time segment
        - metadata: Video features for the segment
    
    Example:
        >>> result = await video_temporal_qa(
        ...     video_path="/path/to/game.mp4",
        ...     question="What happens in this segment?",
        ...     start_time=30.0,
        ...     end_time=60.0
        ... )
    
    Best for:
    - Scene-specific questions
    - Action analysis in time windows
    - Segment-by-segment understanding
    - Detailed temporal reasoning
    """
    try:
        client = get_video_client()
        
        # Enhance question with time range context
        time_context = ""
        if start_time is not None and end_time is not None:
            time_context = f" Focus on the segment from {start_time}s to {end_time}s."
        elif start_time is not None:
            time_context = f" Focus on the segment starting from {start_time}s."
        elif end_time is not None:
            time_context = f" Focus on the segment up to {end_time}s."
        
        enhanced_question = question + time_context
        
        result = await client.analyze_video(
            video_path=video_path,
            question=enhanced_question,
            enable_verification=enable_verification
        )
        
        # Add time range to result
        response = result.copy()
        response["time_range"] = {
            "start_time": start_time,
            "end_time": end_time
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"video_temporal_qa failed: {e}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
