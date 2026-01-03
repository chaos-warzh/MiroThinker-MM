#!/usr/bin/env python3
"""
使用 Gemini 生成符合要求的短 Query

工作流程：
1. 读取每个任务文件夹中的用户文件（PDF、图片、视频等）
2. 读取 useful_search.json 中的搜索结果
3. 调用 Gemini 生成短 query（≤50字，不暴露答案）

Usage:
    uv run python generate_short_query_gemini.py --data-dir datasets_batch2_1228
    uv run python generate_short_query_gemini.py --data-dir datasets_batch2_1228 --tasks 001 002
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


def load_useful_search(folder_path: str) -> List[Dict]:
    """Load useful_search.json from folder."""
    useful_search_path = os.path.join(folder_path, "useful_search.json")
    if os.path.exists(useful_search_path):
        with open(useful_search_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def generate_short_query(
    client: OpenAI,
    user_file_contents: List[Dict],
    user_file_names: List[str],
    useful_search: List[Dict],
    max_tokens: int = 4096
) -> Optional[Dict]:
    """
    Generate short query using Gemini.
    
    Returns:
        Dict with query info or None if failed
    """
    # Build search results summary
    search_summaries = []
    for i, item in enumerate(useful_search, 1):
        title = item.get("title", "无标题")
        # Truncate to first 500 chars
        body = item.get("page_body", "")[:500]
        search_summaries.append(f"{i}. 【{title}】\n{body}...")
    
    search_text = "\n\n".join(search_summaries)
    
    prompt = f"""## 任务
根据用户提供的文件和搜索结果，生成一个带有多个子任务的详细 Query。

**关键要求**：每个子任务必须包含**只有对应搜索结果才有的独特信息**，确保其他搜索结果无法回答。

## 用户文件
{chr(10).join(f'- {name}' for name in user_file_names)}

## 搜索结果（共{len(useful_search)}条，必须全部覆盖）
{search_text}

## Query 设计要求

### 核心原则（重要！）
1. **独特性约束**：每个子任务必须包含只有对应搜索结果才有的**独特事实/数据/案例名称**
2. **不可替代性**：设计的子任务必须确保其他搜索结果无法回答
3. **全覆盖**：设计 3-5 个子任务，确保所有 {len(useful_search)} 条搜索结果都被覆盖

### 如何确保独特性
分析每条搜索结果，找出其**独特标识**：
- 独特的地名/机构名（如"静安区拾影花园"、"嘉定松茗园"）
- 独特的数据（如"建成区海绵城市达标率38%"、"3800平方米雨水花园"）
- 独特的时间/事件（如"2024年上海口袋公园优秀案例"）
- 独特的技术/方法（如"VR沉浸游"、"悬河设计"）

### 子任务设计示例
❌ 错误示例（太泛，其他结果也能回答）：
- "分析口袋公园建设案例"
- "评估智慧园林技术"

✅ 正确示例（包含独特标识）：
- "分析2024年上海口袋公园优秀案例评选中的获奖项目"
- "评估采用VR沉浸游技术的古典园林数字化方案"
- "分析建成区海绵城市达标率达38%的城市实践"

### 设计步骤
1. 为每条搜索结果提取1-2个**独特标识**（只有这条结果才有的信息）
2. 按主题将搜索结果聚类成 3-5 组
3. 为每组设计一个包含独特标识的子任务

## 输出格式
请直接输出 JSON 格式：
{{
  "main_query": "基于用户文件的主题，完成以下子任务：",
  "subtasks": [
    {{
      "id": 1,
      "description": "子任务描述（包含独特标识，15-30字）",
      "covers_results": [1, 3, 5],
      "unique_identifiers": ["只有这些结果才有的独特信息1", "独特信息2"]
    }},
    {{
      "id": 2, 
      "description": "子任务描述（包含独特标识，15-30字）",
      "covers_results": [2, 4],
      "unique_identifiers": ["独特信息"]
    }}
  ],
  "full_query": "完整的 Query 文本（主题 + 所有子任务）",
  "user_doc_concepts": ["从用户文件中提取的核心概念"],
  "uniqueness_analysis": [
    {{"result_id": 1, "unique_fact": "只有结果1才有的独特事实"}},
    {{"result_id": 2, "unique_fact": "只有结果2才有的独特事实"}}
  ],
  "coverage_verification": {{
    "total_results": {len(useful_search)},
    "covered_results": [1,2,3,...],
    "all_covered": true
  }}
}}

注意：只输出JSON，不要其他内容。"""

    # Build content parts: first add all file contents, then add prompt
    content_parts = []
    for file_content in user_file_contents:
        content_parts.extend(file_content)
    
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
            
            # Try to parse JSON from response
            try:
                # Find JSON in response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                    return result
            except json.JSONDecodeError:
                print(f"    Warning: Could not parse JSON")
                print(f"    Raw response: {content[:500]}")
                return {"query": content[:100], "error": "JSON parse failed"}
        
        return None
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


def process_task_folder(
    client: OpenAI,
    folder_path: str,
    task_number: str
) -> Dict:
    """Process a task folder and generate short query.
    
    Returns:
        Dict with query info
    """
    print(f"\n{'='*60}")
    print(f"Processing Task {task_number}")
    print(f"Folder: {folder_path}")
    print(f"{'='*60}")
    
    # Get user files
    user_files = get_user_files(folder_path)
    print(f"Found {len(user_files)} user files:")
    for f in user_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load useful search results
    useful_search = load_useful_search(folder_path)
    print(f"Found {len(useful_search)} useful search results")
    
    if not user_files:
        print(f"  Warning: No user files found")
        return {"task": task_number, "error": "No user files"}
    
    if not useful_search:
        print(f"  Warning: No useful search results found")
        return {"task": task_number, "error": "No useful search results"}
    
    # Prepare file contents
    all_file_contents = []
    user_file_names = []
    
    for filepath in user_files:
        filename = os.path.basename(filepath)
        print(f"  Processing: {filename}")
        
        content_parts, description = prepare_file_content(filepath)
        
        if content_parts:
            all_file_contents.append(content_parts)
            user_file_names.append(filename)
            print(f"    {description}")
        else:
            print(f"    Skipped: unsupported file type")
    
    if not all_file_contents:
        print(f"  Warning: No valid file contents")
        return {"task": task_number, "error": "No valid file contents"}
    
    # Generate short query
    print(f"  Generating short query...")
    result = generate_short_query(
        client,
        all_file_contents,
        user_file_names,
        useful_search
    )
    
    if result:
        result["task"] = task_number
        result["user_files"] = user_file_names
        print(f"  Generated Query: {result.get('query', 'N/A')}")
        return result
    else:
        return {"task": task_number, "error": "Failed to generate query"}


def main():
    parser = argparse.ArgumentParser(
        description="Generate short queries using Gemini"
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
        "--output", "-o",
        default=None,
        help="Output file path (default: <data-dir>/query-short-gemini.jsonl)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (or set OPENAI_API_KEY env var)"
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
    
    # Determine tasks to process
    if args.tasks:
        task_numbers = args.tasks
    else:
        # Get all task folders (directories with numeric names)
        task_numbers = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d.isdigit()
        ])
    
    print(f"Will process {len(task_numbers)} tasks: {task_numbers}")
    
    # Set up output file
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.data_dir, "query-short-gemini.jsonl")
    
    print(f"Output file: {output_path}")
    
    # Create API client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Process each task
    results = []
    for task_number in task_numbers:
        folder_path = os.path.join(args.data_dir, task_number)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found for task {task_number}, skipping")
            continue
        
        result = process_task_folder(client, folder_path, task_number)
        results.append(result)
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Tasks processed: {len(results)}")
    print(f"Output file: {output_path}")
    
    print("\nGenerated Queries:")
    for result in results:
        task_id = result.get("task", "?")
        query = result.get("query", result.get("error", "N/A"))
        print(f"  {task_id}: {query[:60]}...")


if __name__ == "__main__":
    main()
