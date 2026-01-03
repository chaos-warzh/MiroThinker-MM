#!/usr/bin/env python3
"""
Generate search terms and initial query from user files using Gemini 2.5 Pro.

This script processes user files (PDF, video, images, markdown, etc.) and generates:
1. 10 search terms for web search
2. An initial query that can be answered by combining user files and search results

Usage:
    uv run python generate_search_terms_and_query.py --data-dir datasets_batch3
    uv run python generate_search_terms_and_query.py --data-dir datasets_batch3 --tasks 011 012
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# File type extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".html", ".htm", ".md"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}
PRESENTATION_EXTENSIONS = {".pptx", ".ppt"}

# Files to exclude (system files, not user files)
EXCLUDE_PATTERNS = {
    ".json", ".jsonl", ".db", ".hidden", ".chunks.db"
}

# API Configuration - Uses OPENAI_API_KEY and OPENAI_BASE_URL from environment
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
MODEL = "gemini-2.5-pro-06-17"


def is_user_file(filename: str) -> bool:
    """Check if a file is a user file (not system file)."""
    lower_name = filename.lower()
    
    # Exclude system files
    for pattern in EXCLUDE_PATTERNS:
        if lower_name.endswith(pattern):
            return False
    
    # Check if it's a supported file type
    ext = Path(filename).suffix.lower()
    all_extensions = (
        IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS |
        DOCUMENT_EXTENSIONS | SPREADSHEET_EXTENSIONS | PRESENTATION_EXTENSIONS
    )
    
    return ext in all_extensions


def get_user_files(folder_path: str) -> List[str]:
    """Get all user files from a folder."""
    files = []
    for item in os.listdir(folder_path):
        if is_user_file(item):
            files.append(os.path.join(folder_path, item))
    return files


def get_file_type(filepath: str) -> str:
    """Determine the type of file."""
    ext = Path(filepath).suffix.lower()
    
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext in DOCUMENT_EXTENSIONS:
        return "document"
    elif ext in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    elif ext in PRESENTATION_EXTENSIONS:
        return "presentation"
    else:
        return "unknown"


def compress_video(input_path: str, output_path: str, scale: int = 320, crf: int = 32) -> str:
    """Compress video to reduce size while keeping full duration."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale={scale}:-2",
        "-c:v", "libx264", "-crf", str(crf),
        "-preset", "fast",
        "-an",  # Remove audio
        output_path
    ]
    subprocess.run(cmd, capture_output=True)
    return output_path


def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64."""
    with open(file_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_mime_type(filepath: str) -> str:
    """Get MIME type for a file."""
    ext = Path(filepath).suffix.lower()
    
    mime_types = {
        # Images
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        # Videos
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        # Documents
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
    }
    
    return mime_types.get(ext, "application/octet-stream")


def prepare_file_content(filepath: str) -> Tuple[List[Dict], str]:
    """
    Prepare file content for API request.
    
    Returns:
        Tuple of (content_parts, file_description)
    """
    file_type = get_file_type(filepath)
    filename = os.path.basename(filepath)
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    
    content_parts = []
    
    if file_type == "video":
        # Compress video if too large
        if size_mb > 5:
            print(f"    Compressing video ({size_mb:.2f} MB)...")
            compressed_path = "/tmp/compressed_video.mp4"
            compress_video(filepath, compressed_path, scale=320, crf=32)
            filepath = compressed_path
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"    Compressed to {size_mb:.2f} MB")
        
        video_base64 = encode_file_to_base64(filepath)
        content_parts.append({
            "type": "video_url",
            "video_url": {
                "url": f"data:video/mp4;base64,{video_base64}"
            }
        })
        description = f"视频文件: {filename} ({size_mb:.2f} MB)"
    
    elif file_type == "audio":
        # Handle audio files (mp3, wav, m4a)
        ext = Path(filepath).suffix.lower()
        mime_map = {
            ".mp3": "audio/mp3",
            ".wav": "audio/wav",
            ".m4a": "audio/m4a"
        }
        mime_type = mime_map.get(ext, "audio/mpeg")
        
        audio_base64 = encode_file_to_base64(filepath)
        content_parts.append({
            "type": "input_audio",
            "input_audio": {
                "data": audio_base64,
                "format": ext.lstrip(".")
            }
        })
        description = f"音频文件: {filename} ({size_mb:.2f} MB)"
        
    elif file_type == "image":
        image_base64 = encode_file_to_base64(filepath)
        mime_type = get_mime_type(filepath)
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            }
        })
        description = f"图片文件: {filename}"
        
    elif file_type == "document":
        ext = Path(filepath).suffix.lower()
        
        if ext == ".pdf":
            pdf_base64 = encode_file_to_base64(filepath)
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:application/pdf;base64,{pdf_base64}"
                }
            })
            description = f"PDF文档: {filename}"
            
        elif ext in {".txt", ".md", ".html", ".htm"}:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
            content_parts.append({
                "type": "text",
                "text": f"文档内容 ({filename}):\n\n{text_content}"
            })
            description = f"文本文档: {filename}"
        else:
            description = f"不支持的文档格式: {filename}"
    else:
        description = f"不支持的文件类型: {filename}"
    
    return content_parts, description


def generate_search_terms_and_query(
    client: OpenAI,
    filepaths: List[str],
    task_number: str,
    max_tokens: int = 8192
) -> Optional[Dict]:
    """
    Generate search terms and initial query from user files.
    
    Returns:
        Dict with search_terms and query, or None if failed
    """
    print(f"  Processing {len(filepaths)} files...")
    
    # Prepare all file contents
    all_content_parts = []
    file_descriptions = []
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print(f"    Loading: {filename}")
        content_parts, description = prepare_file_content(filepath)
        if content_parts:
            all_content_parts.extend(content_parts)
            file_descriptions.append(description)
    
    if not all_content_parts:
        print(f"    Error: No valid files to process")
        return None
    
    # Build the prompt
    files_summary = "\n".join(f"- {desc}" for desc in file_descriptions)
    
    prompt = f"""你是一个专业的研究任务设计师。请根据以下用户文件，设计一个研究任务。

## 用户文件
{files_summary}

## 任务要求

### 1. 生成10个检索词（必须使用中文）
基于用户文件的内容，生成10个用于网络搜索的**中文**检索词。这些检索词应该：
- **必须使用中文**，不要使用英文
- 能够检索到与用户文件主题相关的补充信息
- 覆盖不同的角度和方面（如背景知识、案例分析、政策法规、技术细节、行业动态等）
- 每个检索词应该是简短的关键词组合（2-4词）
- 检索词应该能够找到**用户文件中没有但对理解主题有帮助的信息**

### 2. 生成初步的研究任务（Query，必须使用中文）
设计一个综合性的研究任务，这个任务需要：
- **必须使用中文**
- 结合用户文件中的信息
- 结合通过检索词搜索到的外部信息
- 包含3-5个子任务
- 每个子任务应该明确、具体、可验证

## 语言要求
**重要**：所有输出内容（检索词、query、topic_summary）都必须使用**中文**。

## 输出格式
请以JSON格式输出：
```json
{{
  "search_terms": [
    "中文检索词1",
    "中文检索词2",
    "中文检索词3",
    "中文检索词4",
    "中文检索词5",
    "中文检索词6",
    "中文检索词7",
    "中文检索词8",
    "中文检索词9",
    "中文检索词10"
  ],
  "query": "基于用户文件，结合外部资料，完成以下研究任务：\\n1. 子任务1描述\\n2. 子任务2描述\\n3. 子任务3描述\\n4. 子任务4描述",
  "topic_summary": "用户文件主题的简要概述（1-2句话）"
}}
```

只输出JSON，不要其他内容。"""

    # Add prompt to content
    all_content_parts.append({
        "type": "text",
        "text": prompt
    })
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": all_content_parts
                }
            ],
            max_tokens=max_tokens
        )
        
        if response.choices:
            content = response.choices[0].message.content
            
            # Try to parse JSON from response
            try:
                # Find JSON object in response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # Validate result
                    if "search_terms" in result and "query" in result:
                        print(f"    Generated {len(result['search_terms'])} search terms")
                        print(f"    Query: {result['query'][:100]}...")
                        return result
                    else:
                        print(f"    Warning: Missing required fields in response")
                        return None
            except json.JSONDecodeError as e:
                print(f"    Warning: Could not parse JSON: {e}")
                return None
        
        return None
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


def process_task_folder(
    client: OpenAI,
    folder_path: str,
    task_number: str
) -> Optional[Dict]:
    """Process all user files in a task folder.
    
    Returns:
        Dict with search_terms, query, and topic_summary
    """
    print(f"\n{'='*60}")
    print(f"Processing Task {task_number}")
    print(f"Folder: {folder_path}")
    print(f"{'='*60}")
    
    user_files = get_user_files(folder_path)
    print(f"Found {len(user_files)} user files:")
    for f in user_files:
        print(f"  - {os.path.basename(f)}")
    
    if not user_files:
        print(f"  Warning: No user files found")
        return None
    
    result = generate_search_terms_and_query(client, user_files, task_number)
    
    if result:
        result["task"] = task_number
        result["files"] = [os.path.basename(f) for f in user_files]
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate search terms and initial query from user files using Gemini 2.5 Pro"
    )
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Path to the data directory containing task folders"
    )
    parser.add_argument(
        "--tasks", "-t",
        nargs="+",
        default=None,
        help="Specific task numbers to process (e.g., 011 012)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: <data-dir>)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for Gemini (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for API (or set OPENAI_BASE_URL env var)"
    )
    
    args = parser.parse_args()
    
    # Set API credentials
    global API_KEY, BASE_URL
    if args.api_key:
        API_KEY = args.api_key
    if args.base_url:
        BASE_URL = args.base_url
    
    if not API_KEY:
        print("Error: API key not provided. Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)
    
    if not BASE_URL:
        print("Error: Base URL not provided. Set OPENAI_BASE_URL or use --base-url")
        sys.exit(1)
    
    # Get all task folders
    all_task_folders = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and d.isdigit()
    ])
    
    # Determine tasks to process
    if args.tasks:
        task_numbers = args.tasks
    else:
        task_numbers = all_task_folders
    
    print(f"Found {len(all_task_folders)} task folders in {args.data_dir}")
    print(f"Will process {len(task_numbers)} tasks: {task_numbers}")
    
    # Set up output directory
    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create API client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Process each task
    all_results = []
    query_lines = []
    
    for task_number in task_numbers:
        folder_path = os.path.join(args.data_dir, task_number)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found for task {task_number}, skipping")
            continue
        
        # Check if output file already exists
        output_file = os.path.join(output_dir, task_number, "search_terms.json")
        if os.path.exists(output_file):
            print(f"Skipping task {task_number}: output file already exists")
            # Load existing result for query.jsonl
            with open(output_file, "r", encoding="utf-8") as f:
                existing_result = json.load(f)
                query_lines.append({"task": task_number, "query": existing_result.get("query", "")})
            continue
        
        result = process_task_folder(client, folder_path, task_number)
        
        if result:
            all_results.append(result)
            query_lines.append({"task": task_number, "query": result.get("query", "")})
            
            # Save individual task result
            task_output_dir = os.path.join(output_dir, task_number)
            os.makedirs(task_output_dir, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved to: {output_file}")
    
    # Save query.jsonl (just task and query)
    query_file = os.path.join(output_dir, "query.jsonl")
    with open(query_file, "w", encoding="utf-8") as f:
        for item in query_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved queries to: {query_file}")
    
    # Save all_results.jsonl (complete results with search_terms, query, topic_summary, etc.)
    all_results_file = os.path.join(output_dir, "all_results.jsonl")
    with open(all_results_file, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved all results to: {all_results_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Tasks processed: {len(all_results)}")
    print(f"Total queries generated: {len(query_lines)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
