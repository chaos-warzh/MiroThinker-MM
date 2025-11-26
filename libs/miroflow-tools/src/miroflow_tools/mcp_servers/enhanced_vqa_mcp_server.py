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
Enhanced VQA MCP Server

This MCP server exposes the Enhanced VQA tool as a callable interface for LLM agents.
It wraps the enhanced_vqa module to provide:

1. Multi-turn vision understanding with automatic follow-up questions
2. Confidence scoring and metadata extraction
3. Cross-validation with web search
4. Support for multiple VLLM providers (OpenAI, Anthropic, Qwen, Custom)

Usage by LLM:
    The LLM can call this tool via MCP with:
    - Tool: vision_understanding_advanced
    - Parameters:
        - image_path_or_url: Path or URL to image
        - question: Main question about the image
        - enable_verification: Whether to perform multi-turn verification (default: true)
"""

import json

from fastmcp import FastMCP

# Import the enhanced VQA client from miroflow_tools.tools package
from miroflow_tools.tools.enhanced_vqa import get_vqa_client

# Initialize FastMCP server
mcp = FastMCP("enhanced-vqa-mcp-server")


@mcp.tool()
async def vision_understanding_advanced(
    image_path_or_url: str,
    question: str,
    enable_verification: bool = True,
) -> str:
    """
    Advanced vision understanding with multi-turn verification and confidence scoring.
    
    This tool provides enhanced image understanding capabilities:
    - Multi-turn verification for higher accuracy
    - Confidence scoring for reliability assessment
    - Metadata extraction (objects, colors, text, scene)
    - Optional cross-validation with web search
    - Support for multiple VLLM providers
    
    Args:
        image_path_or_url: Local file path (e.g., 'data/image.jpg') or URL (e.g., 'https://...')
        question: The question to ask about the image
        enable_verification: Whether to perform multi-turn verification (default: true)
    
    Returns:
        JSON string with:
        - answer: Direct answer to the question
        - confidence: Confidence score (0.0-1.0)
        - metadata: Extracted visual information
        - reasoning: Reasoning behind the answer
        - follow_up_answers: Optional answers to verification questions
        - verification_sources: Optional web search verification
    
    Example:
        Input:
            image_path_or_url: "data/character.jpg"
            question: "Who is this character and what anime are they from?"
            enable_verification: true
        
        Output (JSON):
        {
            "answer": "This is Bocchi from the anime 'Bocchi the Rock!'",
            "confidence": 0.92,
            "metadata": {
                "objects_detected": ["character", "guitar", "pink hair"],
                "colors_dominant": ["pink", "black", "white"],
                "scene_description": "A teenage girl with pink hair holding a guitar"
            },
            "reasoning": "The character has distinctive pink hair and is holding a guitar, which are signature characteristics of Bocchi from the anime 'Bocchi the Rock!'",
            "follow_up_answers": [
                {
                    "question": "What specific visual features support this identification?",
                    "answer": "The pink hair, guitar, and art style are distinctive to this character",
                    "confidence": 0.88
                }
            ],
            "verified": true,
            "verification_sources": {
                "search_query": "Bocchi Rock anime character",
                "results": [...]
            }
        }
    """
    try:
        client = get_vqa_client()
        
        # Call enhanced VQA
        result = await client.analyze_image(
            image_path_or_url=image_path_or_url,
            question=question,
            enable_multi_turn=enable_verification,
        )
        
        # Return as JSON string
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
            "image_path_or_url": image_path_or_url,
            "question": question,
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)


@mcp.tool()
async def vision_quick_answer(
    image_path_or_url: str,
    question: str,
    enable_verification: bool = False,
) -> str:
    """
    Quick vision question answering without multi-turn verification.
    
    Use this for simpler questions that don't require extensive verification.
    Faster response time compared to vision_understanding_advanced.
    
    Args:
        image_path_or_url: Local file path or URL of the image
        question: Question about the image
        enable_verification: (Optional) Ignored for quick answer, included for compatibility.
    
    Returns:
        JSON with answer, confidence, and basic metadata
    """
    try:
        client = get_vqa_client()
        
        result = await client.analyze_image(
            image_path_or_url=image_path_or_url,
            question=question,
            enable_multi_turn=False,
        )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)


@mcp.tool()
async def vision_extract_metadata(
    image_path_or_url: str,
    enable_verification: bool = False,
) -> str:
    """
    Extract visual metadata from an image without answering a specific question.
    
    Useful for understanding what's in an image before formulating questions.
    
    Args:
        image_path_or_url: Local file path or URL of the image
        enable_verification: (Optional) Ignored for metadata extraction, included for compatibility.
    
    Returns:
        JSON with extracted visual information:
        - objects detected
        - dominant colors
        - visible text
        - scene description
        - estimated confidence of observations
    """
    try:
        client = get_vqa_client()
        
        metadata_question = (
            "Please analyze this image and provide detailed metadata. "
            "What objects do you see? What are the dominant colors? "
            "Is there any visible text? Describe the overall scene and setting."
        )
        
        result = await client.analyze_image(
            image_path_or_url=image_path_or_url,
            question=metadata_question,
            enable_multi_turn=False,
        )
        
        # Focus on metadata
        metadata_result = {
            "metadata": result.get("metadata", {}),
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence", 0),
        }
        
        return json.dumps(metadata_result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)


@mcp.tool()
async def vision_comparative_analysis(
    image_path_or_url_1: str,
    image_path_or_url_2: str,
    comparison_question: str,
) -> str:
    """
    Compare two images to answer a question about their relationship.
    
    Useful for verification tasks (e.g., "Are these the same character?")
    
    Args:
        image_path_or_url_1: First image path or URL
        image_path_or_url_2: Second image path or URL
        comparison_question: Question comparing the two images
    
    Returns:
        JSON with comparative analysis
    """
    try:
        client = get_vqa_client()
        
        # Analyze first image
        result1 = await client.analyze_image(
            image_path_or_url=image_path_or_url_1,
            question="Describe what you see in detail",
            enable_multi_turn=False,
        )
        
        # Analyze second image
        result2 = await client.analyze_image(
            image_path_or_url=image_path_or_url_2,
            question="Describe what you see in detail",
            enable_multi_turn=False,
        )
        
        # Compare
        comparison_result = await client.analyze_image(
            image_path_or_url=image_path_or_url_1,
            question=f"First image: {result1['answer']}. "
                     f"Second image (for comparison): {result2['answer']}. "
                     f"Now answer: {comparison_question}",
            enable_multi_turn=False,
        )
        
        return json.dumps({
            "image1_analysis": {
                "description": result1.get("answer", ""),
                "metadata": result1.get("metadata", {}),
            },
            "image2_analysis": {
                "description": result2.get("answer", ""),
                "metadata": result2.get("metadata", {}),
            },
            "comparison": {
                "answer": comparison_result.get("answer", ""),
                "confidence": comparison_result.get("confidence", 0),
                "reasoning": comparison_result.get("reasoning", ""),
            },
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # For testing and debugging
    mcp.run(transport="stdio")
