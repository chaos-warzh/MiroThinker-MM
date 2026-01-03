#!/usr/bin/env python3
"""
Extract insights from source files using Gemini 2.5 Pro.

This script processes user files (PDF, video, images, markdown, etc.) and extracts
insights that are required for answering the query. Each insight follows these criteria:
- 可验证性: Can be clearly verified in the report
- 必要性: Essential for a complete report
- 独立性: Independent from other insights
- 来源明确: Clearly marked as [用户文档]
- 粒度适中: 1-2 sentences, verifiable

Usage:
    uv run python extract_insights_from_sources.py --data-dir datasets_batch2_1228
    uv run python extract_insights_from_sources.py --data-dir datasets_batch2_1228 --tasks 001 002
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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


def extract_insights_from_file(
    client: OpenAI,
    filepath: str,
    query: str,
    max_tokens: int = 8192
) -> Optional[Dict]:
    """
    Extract insights from a single file.
    
    Returns:
        Dict with insights or None if failed
    """
    filename = os.path.basename(filepath)
    print(f"  Processing: {filename}")
    
    content_parts, description = prepare_file_content(filepath)
    
    if not content_parts:
        print(f"    Skipped: unsupported file type")
        return None
    
    # Build the prompt
    file_type = get_file_type(filepath)
    
    # Different prompts for different file types
    if file_type == "image":
        type_specific_instruction = """
## 重要提示（图片文件）
你正在处理的是一张图片。请只抽取**能从图片中直接观察到的视觉信息**，例如：
- 图片中显示的场景、物体、人物
- 图片中可见的文字、标识、数字
- 图片的构图、色彩、光影特征
- 图片中展示的具体细节

**严禁**抽取以下内容：
- 需要外部知识才能得知的历史事件、日期、统计数据
- 图片中未直接显示的背景信息
- 需要网络搜索才能获得的信息
- 你的推测或联想"""
    elif file_type == "video":
        type_specific_instruction = """
## 重要提示（视频文件）
你正在处理的是一个视频。请只抽取**能从视频中直接观察到的信息**，例如：
- 视频中出现的场景、人物、物体
- 视频中的对话、旁白、字幕内容
- 视频中展示的动作、事件、过程
- 视频中可见的文字、数据、图表

**严禁**抽取以下内容：
- 需要外部知识才能得知的信息
- 视频中未直接展示的背景知识
- 需要网络搜索才能获得的信息"""
    else:
        type_specific_instruction = """
## 重要提示（文档文件）
请只抽取**文档中明确写明的信息**，不要添加外部知识或推测。"""

    prompt = f"""你是一个专业的信息抽取助手。请从以下文件中抽取与查询相关的关键信息点（insights）。

## 查询任务
{query}
{type_specific_instruction}

## 抽取要求
每个insight必须满足以下标准：
1. **直接可得**: 信息必须能从当前文件中直接获取，不能依赖外部知识或网络搜索
2. **可验证性**: 能够明确判断报告是否包含这个信息
3. **必要性**: 缺少这个信息，报告将不完整
4. **独立性**: 与其他insight不重复
5. **粒度适中**: 1-2句话可验证

## 语言要求
**重要**: 输出的insight必须使用与上述"查询任务"相同的语言。如果查询是中文，insight必须用中文；如果查询是英文，insight必须用英文。

## 输出格式
请以JSON数组格式输出insights：
```json
[
  {{"insight": "从文件中直接获取的具体信息点"}}
]
```

如果文件中没有与查询相关的可直接获取的信息，请返回空数组 []。

只输出JSON数组，不要其他内容。"""

    # Add prompt to content
    content_parts.append({
        "type": "text",
        "text": prompt
    })
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            max_tokens=max_tokens
        )
        
        if response.choices:
            content = response.choices[0].message.content
            
            # Try to parse JSON array from response
            try:
                # Find JSON array in response
                json_start = content.find("[")
                json_end = content.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    insights_list = json.loads(json_str)
                    # Force set the correct source filename (override LLM's output)
                    for insight in insights_list:
                        insight["source"] = filename
                    print(f"    Extracted {len(insights_list)} insights")
                    return insights_list
            except json.JSONDecodeError:
                # Return empty list if JSON parsing fails
                print(f"    Warning: Could not parse JSON")
                return []
        
        return None
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


def process_task_folder(
    client: OpenAI,
    folder_path: str,
    query: str,
    task_number: str
) -> Dict:
    """Process all user files in a task folder.
    
    Returns:
        Dict with format: {"gold_insights": [{"insight": "...", "source": "..."}]}
    """
    print(f"\n{'='*60}")
    print(f"Processing Task {task_number}")
    print(f"Folder: {folder_path}")
    print(f"Query: {query[:100]}...")
    print(f"{'='*60}")
    
    user_files = get_user_files(folder_path)
    print(f"Found {len(user_files)} user files:")
    for f in user_files:
        print(f"  - {os.path.basename(f)}")
    
    all_insights = []
    
    for filepath in user_files:
        insights_list = extract_insights_from_file(client, filepath, query)
        
        if insights_list:
            all_insights.extend(insights_list)
    
    print(f"Total insights extracted: {len(all_insights)}")
    return {"gold_insights": all_insights}


def load_queries(data_dir: str) -> Dict[str, str]:
    """Load queries from query.jsonl file."""
    query_file = os.path.join(data_dir, "query.jsonl")
    queries = {}
    
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                task = json.loads(line)
                task_number = task.get("task") or task.get("number")
                queries[task_number] = task["query"]
    
    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Extract insights from source files using Gemini 2.5 Pro"
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
        help="Specific task numbers to process (e.g., 001 002)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for insights (default: insights/<data-dir-name>)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for Gemini (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for API (or set GEMINI_BASE_URL env var)"
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
    
    # Load queries
    queries = load_queries(args.data_dir)
    print(f"Loaded {len(queries)} queries from {args.data_dir}/query.jsonl")
    
    # Determine tasks to process
    if args.tasks:
        task_numbers = args.tasks
    else:
        # Get all task folders
        task_numbers = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d in queries
        ])
    
    print(f"Will process {len(task_numbers)} tasks: {task_numbers}")
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        data_dir_name = os.path.basename(os.path.abspath(args.data_dir))
        output_dir = os.path.join("insights", data_dir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create API client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Process each task
    all_results = []
    skipped_count = 0
    for task_number in task_numbers:
        if task_number not in queries:
            print(f"Warning: No query found for task {task_number}, skipping")
            continue
        
        folder_path = os.path.join(args.data_dir, task_number)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found for task {task_number}, skipping")
            continue
        
        # Check if output file already exists
        task_output_dir = os.path.join(output_dir, task_number)
        output_file = os.path.join(task_output_dir, "gold_insights_from_source.json")
        if os.path.exists(output_file):
            print(f"Skipping task {task_number}: output file already exists")
            skipped_count += 1
            continue
        
        query = queries[task_number]
        result = process_task_folder(client, folder_path, query, task_number)
        all_results.append(result)
        
        # Save individual task result
        os.makedirs(task_output_dir, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved insights to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Tasks processed: {len(all_results)}")
    print(f"Tasks skipped (already exist): {skipped_count}")
    total_insights = sum(len(r.get("gold_insights", [])) for r in all_results)
    print(f"Total insights extracted: {total_insights}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
