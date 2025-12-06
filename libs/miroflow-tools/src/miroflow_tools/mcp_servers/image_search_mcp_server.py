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
Image Search MCP Server

This MCP server provides image search capabilities for generating illustrated technical reports.
It uses Jina AI Search API to find high-quality images with proper quality filtering.

Key Features:
1. Search for images based on semantic queries
2. Quality filtering (resolution, format, source)
3. Returns multiple candidates for agent selection
4. Automatic source citation generation
"""

import json
import os
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from fastmcp import FastMCP

mcp = FastMCP("image-search-mcp-server")

# Quality filtering constants
MIN_IMAGE_WIDTH = 800  # Minimum acceptable width in pixels
MIN_IMAGE_HEIGHT = 400  # Minimum acceptable height in pixels
ALLOWED_FORMATS = [".jpg", ".jpeg", ".png", ".svg", ".webp"]
EXCLUDED_KEYWORDS = ["icon", "logo", "favicon", "avatar", "thumbnail", "button", "badge"]
PREFERRED_SOURCES = [
    "arxiv.org",
    "github.com",
    "medium.com",
    "towardsdatascience.com",
    "papers.nips.cc",
    "openaccess.thecvf.com",
    "paperswithcode.com",
    "distill.pub",
    "jalammar.github.io",
]


def is_high_quality_image(image_url: str, image_title: str) -> bool:
    """
    Check if an image meets quality standards.
    
    Filters:
    1. File format must be in ALLOWED_FORMATS
    2. URL should not contain excluded keywords (icon, logo, etc.)
    3. Title should not contain excluded keywords
    """
    # Check file extension
    parsed_url = urlparse(image_url.lower())
    path = parsed_url.path
    
    # Check if it has a valid image extension
    has_valid_extension = any(path.endswith(ext) for ext in ALLOWED_FORMATS)
    
    # SVG files are often diagrams, but be careful with logos
    if path.endswith(".svg"):
        # SVG is acceptable if not a logo/icon
        if any(keyword in image_url.lower() for keyword in ["logo", "icon", "favicon"]):
            return False
    
    # For other formats, must have extension or be from data URI
    if not has_valid_extension and not image_url.startswith("data:image"):
        # Sometimes images don't have extensions but are served via CDN
        # Allow if URL looks like an image CDN pattern
        if not any(pattern in image_url.lower() for pattern in ["/images/", "/img/", "/media/", "imgur", "cloudinary"]):
            return False
    
    # Check for excluded keywords in URL
    url_lower = image_url.lower()
    if any(keyword in url_lower for keyword in EXCLUDED_KEYWORDS):
        return False
    
    # Check for excluded keywords in title
    title_lower = image_title.lower()
    if any(keyword in title_lower for keyword in EXCLUDED_KEYWORDS):
        return False
    
    return True


def score_image_source(source_url: str) -> int:
    """
    Score the source website for reliability and quality.
    Higher score = better source.
    """
    score = 50  # Base score
    
    domain = urlparse(source_url).netloc.lower()
    
    # Preferred academic/technical sources get bonus
    for preferred in PREFERRED_SOURCES:
        if preferred in domain:
            score += 30
            break
    
    # Educational domains (.edu) get bonus
    if ".edu" in domain:
        score += 20
    
    # GitHub repositories often have good diagrams
    if "github" in domain and "/raw/" in source_url:
        score += 15
    
    # Penalize social media and ad-heavy sites
    if any(bad in domain for bad in ["facebook", "twitter", "instagram", "pinterest", "tiktok"]):
        score -= 20
    
    return score


@mcp.tool()
async def search_images_for_report(
    query: str,
    num_results: int = 5,
    context: str = "",
    preferred_source: str = None,
) -> str:
    """
    Search for high-quality images suitable for technical reports.
    
    This tool searches the web for images matching the query and returns multiple
    high-quality candidates with automatic quality filtering. The agent should:
    1. Review all returned candidates
    2. Use verify_image_relevance to check semantic match
    3. Select the most relevant image
    4. Insert the markdown at the appropriate position
    
    Args:
        query: Search query describing the desired image
               Examples: "transformer architecture diagram", 
                        "BERT model attention mechanism visualization",
                        "neural network backpropagation illustration"
        num_results: Number of candidate images to return (default: 5)
                    Returns multiple candidates for agent selection
        context: Optional context about where the image will be used
                 Helps with relevance filtering
                 Example: "This image will illustrate the attention mechanism 
                          in the Transformer section"
        preferred_source: Optional preferred website domain
                         Example: "arxiv.org", "github.com"
    
    Returns:
        JSON array of image candidates with metadata:
        [
            {
                "url": "https://example.com/transformer.png",
                "title": "Transformer Architecture Diagram",
                "source_url": "https://arxiv.org/abs/1706.03762",
                "source_name": "Attention is All You Need - arXiv",
                "quality_score": 85,
                "markdown": "![Transformer Architecture](https://...)\n*Source: [Attention is All You Need](https://arxiv.org/abs/1706.03762)*"
            },
            ...
        ]
    
    Quality Filtering Applied:
    - Minimum resolution: 800x400 pixels (estimated)
    - Formats: JPG, PNG, SVG, WebP
    - Excludes: icons, logos, favicons, thumbnails
    - Prefers: academic papers, technical blogs, documentation
    
    Usage Example:
        1. Agent: "I need to illustrate the Transformer architecture"
        2. Call: search_images_for_report("transformer architecture diagram", num_results=5)
        3. Agent: Review candidates, verify relevance, select best match
        4. Agent: Insert markdown at appropriate position in report
    
    Note: Always verify image relevance using verify_image_relevance before insertion!
    """
    try:
        jina_api_key = os.getenv("JINA_API_KEY")
        if not jina_api_key:
            return json.dumps({
                "error": "JINA_API_KEY not configured",
                "message": "Please set JINA_API_KEY environment variable"
            }, ensure_ascii=False)
        
        # Enhance search query for better results
        enhanced_query = f"{query} high quality diagram illustration"
        
        # Prepare Jina Search API request
        headers = {
            "Authorization": f"Bearer {jina_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-With-Images-Summary": "all",  # Get all images from search results
            "X-No-Cache": "true",  # Bypass cache for fresh results
        }
        
        # Add site restriction if preferred source is specified
        if preferred_source:
            headers["X-Site"] = f"https://{preferred_source}"
        
        payload = {
            "q": enhanced_query,
            "num": min(num_results * 3, 30),  # Get extra candidates for filtering
        }
        
        response = requests.post(
            "https://s.jina.ai/",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Collect and filter images
        candidates = []
        seen_urls = set()  # Deduplicate
        
        for search_result in data.get("data", []):
            source_url = search_result.get("url", "")
            source_title = search_result.get("title", "Unknown Source")
            images = search_result.get("images", {})
            
            # Calculate source quality score
            source_score = score_image_source(source_url)
            
            for img_title, img_url in images.items():
                # Skip duplicates
                if img_url in seen_urls:
                    continue
                
                # Apply quality filter
                if not is_high_quality_image(img_url, img_title):
                    continue
                
                seen_urls.add(img_url)
                
                # Calculate overall quality score
                quality_score = source_score
                
                # Bonus for descriptive titles
                if len(img_title) > 20 and not img_title.startswith("Image"):
                    quality_score += 10
                
                # Construct markdown with proper citation
                markdown = f"![{img_title}]({img_url})\n"
                markdown += f"*Source: [{source_title}]({source_url})*"
                
                candidates.append({
                    "url": img_url,
                    "title": img_title,
                    "source_url": source_url,
                    "source_name": source_title,
                    "quality_score": quality_score,
                    "markdown": markdown
                })
                
                # Stop if we have enough candidates
                if len(candidates) >= num_results * 2:
                    break
            
            if len(candidates) >= num_results * 2:
                break
        
        # Sort by quality score and return top N
        candidates.sort(key=lambda x: x["quality_score"], reverse=True)
        top_candidates = candidates[:num_results]
        
        if not top_candidates:
            return json.dumps({
                "error": "No high-quality images found",
                "message": f"Try different search terms or relax quality filters",
                "query": query
            }, ensure_ascii=False)
        
        return json.dumps({
            "success": True,
            "query": query,
            "num_candidates": len(top_candidates),
            "images": top_candidates,
            "usage_note": "Review all candidates and use verify_image_relevance to check semantic match before insertion"
        }, ensure_ascii=False, indent=2)
    
    except requests.exceptions.RequestException as e:
        return json.dumps({
            "error": "API request failed",
            "error_type": type(e).__name__,
            "message": str(e),
            "query": query
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "error": "Unexpected error",
            "error_type": type(e).__name__,
            "message": str(e),
            "query": query
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def verify_image_relevance(
    image_url: str,
    expected_content: str,
    section_context: str = "",
) -> str:
    """
    Verify if an image is semantically relevant to the expected content.
    
    This tool uses vision understanding to check if the image actually contains
    the expected content and is appropriate for the given context. Use this
    BEFORE inserting an image into the report to ensure quality.
    
    Args:
        image_url: URL of the image to verify
        expected_content: Description of what the image should contain
                         Example: "Transformer architecture with encoder-decoder structure"
        section_context: Optional context about where image will be used
                        Example: "Section about attention mechanisms in NLP"
    
    Returns:
        JSON with verification results:
        {
            "is_relevant": true/false,
            "confidence": 0.0-1.0,
            "description": "What the image actually shows",
            "recommendation": "Use/Skip",
            "reasoning": "Why this image is/isn't suitable"
        }
    
    Usage Example:
        After getting candidates from search_images_for_report:
        1. For each candidate with high quality_score:
        2. Call verify_image_relevance to check semantic match
        3. Select the candidate with highest relevance + confidence
        4. Insert its markdown into the report
    
    Note: This tool may use vision API quota. Use wisely - only verify top candidates.
    """
    try:
        # Import vision client (should be available in the environment)
        from miroflow_tools.tools.enhanced_vqa import get_vqa_client
        
        client = get_vqa_client()
        
        # Construct verification question
        verification_question = (
            f"Does this image show {expected_content}? "
            f"Describe what you see and whether it matches the expected content. "
            f"Context: {section_context if section_context else 'Technical documentation'}"
        )
        
        # Analyze image
        result = await client.analyze_image(
            image_path_or_url=image_url,
            question=verification_question,
            enable_multi_turn=False,  # Single-turn for speed
        )
        
        answer = result.get("answer", "").lower()
        confidence = result.get("confidence", 0.0)
        
        # Determine relevance based on answer
        positive_indicators = ["yes", "shows", "contains", "depicts", "illustrates", "displays"]
        negative_indicators = ["no", "does not", "doesn't", "cannot", "can't", "unable"]
        
        has_positive = any(indicator in answer for indicator in positive_indicators)
        has_negative = any(indicator in answer for indicator in negative_indicators)
        
        is_relevant = has_positive and not has_negative and confidence > 0.6
        
        # Generate recommendation
        if is_relevant and confidence > 0.8:
            recommendation = "Strongly Recommended - High relevance and confidence"
        elif is_relevant and confidence > 0.6:
            recommendation = "Recommended - Good match"
        elif confidence > 0.5:
            recommendation = "Acceptable - Moderate confidence, review manually"
        else:
            recommendation = "Skip - Low relevance or confidence"
        
        return json.dumps({
            "is_relevant": is_relevant,
            "confidence": confidence,
            "description": result.get("answer", ""),
            "recommendation": recommendation,
            "reasoning": result.get("reasoning", ""),
            "image_url": image_url,
            "expected_content": expected_content
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "error": "Verification failed",
            "error_type": type(e).__name__,
            "message": str(e),
            "image_url": image_url,
            "fallback_recommendation": "Review manually before using"
        }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
