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
Folder Processor Module

This module provides functionality to process all files in a folder
and prepare them for multi-modal LLM processing.

Supports:
- Images: jpg, jpeg, png, gif, webp
- Videos: mp4, avi, mov, mkv, webm, flv, wmv, m4v
- Audio: wav, mp3, m4a
- Documents: pdf, docx, doc, txt, xlsx, xls, pptx, ppt, html, htm
- Data: json, jsonld, csv
- Archives: zip

Usage:
    from src.io.folder_processor import process_folder_for_task
    
    task_content, task_description, multimodal_files = process_folder_for_task(
        folder_path="data/000",
        query="请根据这个pdf文件和图片的内容，整理重要文献"
    )
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Import existing converters from input_handler
from .input_handler import (
    DocumentConverterResult,
    XlsxConverter,
    DocxConverter,
    HtmlConverter,
    PptxConverter,
    ZipConverter,
)

# Try to import optional dependencies
try:
    import pdfminer.high_level
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False

try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import json


# File type categories
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".html", ".htm"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}
PRESENTATION_EXTENSIONS = {".pptx", ".ppt"}
DATA_EXTENSIONS = {".jsonld", ".json"}
ARCHIVE_EXTENSIONS = {".zip"}


@dataclass
class FileInfo:
    """Information about a single file."""
    path: str
    name: str
    extension: str
    category: str
    size_bytes: int
    
    @property
    def is_multimodal(self) -> bool:
        """Check if file requires multimodal processing (image/video/audio)."""
        return self.category in ["image", "video", "audio"]


@dataclass
class FolderContents:
    """Structured representation of folder contents."""
    folder_path: str
    files: List[FileInfo] = field(default_factory=list)
    
    @property
    def images(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "image"]
    
    @property
    def videos(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "video"]
    
    @property
    def audios(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "audio"]
    
    @property
    def documents(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "document"]
    
    @property
    def spreadsheets(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "spreadsheet"]
    
    @property
    def presentations(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "presentation"]
    
    @property
    def data_files(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "data"]
    
    @property
    def archives(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "archive"]
    
    @property
    def other_files(self) -> List[FileInfo]:
        return [f for f in self.files if f.category == "other"]
    
    @property
    def multimodal_files(self) -> List[FileInfo]:
        """Get all files that require multimodal processing."""
        return [f for f in self.files if f.is_multimodal]
    
    @property
    def text_extractable_files(self) -> List[FileInfo]:
        """Get all files that can have text extracted."""
        return [f for f in self.files if f.category in 
                ["document", "spreadsheet", "presentation", "data"]]
    
    def get_summary(self) -> str:
        """Get a summary of folder contents."""
        summary_parts = [f"Folder: {self.folder_path}"]
        summary_parts.append(f"Total files: {len(self.files)}")
        
        categories = {}
        for f in self.files:
            categories[f.category] = categories.get(f.category, 0) + 1
        
        for cat, count in sorted(categories.items()):
            summary_parts.append(f"  - {cat}: {count}")
        
        return "\n".join(summary_parts)


def get_file_category(extension: str) -> str:
    """Determine the category of a file based on its extension."""
    ext = extension.lower()
    
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
    elif ext in DATA_EXTENSIONS:
        return "data"
    elif ext in ARCHIVE_EXTENSIONS:
        return "archive"
    else:
        return "other"


def scan_folder(folder_path: str, recursive: bool = False) -> FolderContents:
    """
    Scan a folder and categorize all files.
    
    Args:
        folder_path: Path to the folder to scan
        recursive: Whether to scan subdirectories recursively
        
    Returns:
        FolderContents object with categorized files
    """
    folder_path = os.path.abspath(folder_path)
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")
    
    contents = FolderContents(folder_path=folder_path)
    
    if recursive:
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.startswith("."):  # Skip hidden files
                    continue
                file_path = os.path.join(root, filename)
                _add_file_info(contents, file_path, filename)
    else:
        for filename in os.listdir(folder_path):
            if filename.startswith("."):  # Skip hidden files
                continue
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                _add_file_info(contents, file_path, filename)
    
    return contents


def _add_file_info(contents: FolderContents, file_path: str, filename: str) -> None:
    """Add file information to FolderContents."""
    _, ext = os.path.splitext(filename)
    category = get_file_category(ext)
    
    try:
        size = os.path.getsize(file_path)
    except OSError:
        size = 0
    
    contents.files.append(FileInfo(
        path=file_path,
        name=filename,
        extension=ext.lower(),
        category=category,
        size_bytes=size
    ))


def _extract_file_content(file_info: FileInfo, max_content_length: int = 200_000) -> Optional[str]:
    """
    Extract text content from a file using existing converters.
    
    Args:
        file_info: FileInfo object for the file
        max_content_length: Maximum length of content to return
        
    Returns:
        Extracted text content or None if extraction failed
    """
    file_path = file_info.path
    ext = file_info.extension.lower()
    
    try:
        parsing_result = None
        
        # Use existing converters from input_handler
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            parsing_result = DocumentConverterResult(title=None, text_content=content)
        
        elif ext in [".json", ".jsonld"]:
            # Check if this is a long_context.json file (RAG candidate)
            if "long_context" in os.path.basename(file_path).lower():
                # For long context files, just note that RAG should be used
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    doc_count = len(data)
                    sample_titles = [d.get("title", "")[:50] for d in data[:3]]
                    content = f"[Long Context Document Collection]\n"
                    content += f"Total documents: {doc_count}\n"
                    content += f"Sample titles: {sample_titles}\n"
                    content += f"\n**Use RAG tools (rag_search, rag_get_context) to search this document.**"
                    parsing_result = DocumentConverterResult(title=None, text_content=content)
                else:
                    content = json.dumps(data, ensure_ascii=False, indent=2)
                    parsing_result = DocumentConverterResult(title=None, text_content=content)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.dumps(json.load(f), ensure_ascii=False, indent=2)
                parsing_result = DocumentConverterResult(title=None, text_content=content)
        
        elif ext in [".xlsx", ".xls"]:
            parsing_result = XlsxConverter(local_path=file_path)
        
        elif ext == ".pdf":
            if HAS_PDFMINER:
                content = pdfminer.high_level.extract_text(file_path)
                parsing_result = DocumentConverterResult(title=None, text_content=content)
        
        elif ext in [".docx", ".doc"]:
            parsing_result = DocxConverter(local_path=file_path)
        
        elif ext in [".html", ".htm"]:
            parsing_result = HtmlConverter(local_path=file_path)
        
        elif ext in [".pptx", ".ppt"]:
            parsing_result = PptxConverter(local_path=file_path)
        
        elif ext == ".zip":
            parsing_result = ZipConverter(local_path=file_path)
        
        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            parsing_result = DocumentConverterResult(title=None, text_content=content)
        
        # Try MarkItDown as fallback for other file types
        if parsing_result is None and HAS_MARKITDOWN:
            try:
                md = MarkItDown(enable_plugins=True)
                parsing_result = md.convert(file_path)
            except Exception:
                pass
        
        # Extract content from result
        if parsing_result:
            content = parsing_result.text_content
            if content and len(content) > max_content_length:
                content = content[:max_content_length] + "\n... [Content truncated]"
            return content
        
    except Exception as e:
        return f"[Error extracting content: {str(e)}]"
    
    return None


def _get_image_info(file_info: FileInfo) -> str:
    """Get image information string."""
    info_parts = [f"Image file: {file_info.name}"]
    info_parts.append(f"Path: {file_info.path}")
    
    if HAS_PIL:
        try:
            with Image.open(file_info.path) as img:
                width, height = img.size
                info_parts.append(f"Dimensions: {width}x{height} pixels")
                info_parts.append(f"Format: {img.format}")
        except Exception:
            pass
    
    return "\n".join(info_parts)


def _get_video_info(file_info: FileInfo) -> str:
    """Get video information string."""
    info_parts = [f"Video file: {file_info.name}"]
    info_parts.append(f"Path: {file_info.path}")
    
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(file_info.path)
        info_parts.append(f"Duration: {clip.duration:.2f} seconds")
        info_parts.append(f"Resolution: {clip.w}x{clip.h}")
        info_parts.append(f"FPS: {clip.fps:.1f}")
        clip.close()
    except Exception:
        try:
            import cv2
            cap = cv2.VideoCapture(file_info.path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps > 0:
                duration = frame_count / fps
                info_parts.append(f"Duration: {duration:.2f} seconds")
            info_parts.append(f"Resolution: {width}x{height}")
            info_parts.append(f"FPS: {fps:.1f}")
            cap.release()
        except Exception:
            pass
    
    return "\n".join(info_parts)


def _get_audio_info(file_info: FileInfo) -> str:
    """Get audio information string."""
    info_parts = [f"Audio file: {file_info.name}"]
    info_parts.append(f"Path: {file_info.path}")
    
    ext = file_info.extension.lower()
    
    if ext == ".wav":
        try:
            import wave
            with wave.open(file_info.path, "rb") as audio_file:
                duration = audio_file.getnframes() / float(audio_file.getframerate())
                sample_rate = audio_file.getframerate()
                channels = audio_file.getnchannels()
                info_parts.append(f"Duration: {duration:.2f} seconds")
                info_parts.append(f"Sample rate: {sample_rate} Hz")
                info_parts.append(f"Channels: {channels}")
        except Exception:
            pass
    else:
        try:
            from mutagen import File as MutagenFile
            audio = MutagenFile(file_info.path)
            if audio and hasattr(audio, "info") and hasattr(audio.info, "length"):
                info_parts.append(f"Duration: {audio.info.length:.2f} seconds")
                if hasattr(audio.info, "sample_rate"):
                    info_parts.append(f"Sample rate: {audio.info.sample_rate} Hz")
        except Exception:
            pass
    
    return "\n".join(info_parts)


def process_folder_for_task(
    folder_path: str,
    query: str,
    recursive: bool = False,
    include_file_contents: bool = True,
    max_content_length: int = 200_000
) -> Tuple[str, str, List[str]]:
    """
    Process all files in a folder and prepare task description for LLM.
    
    This function:
    1. Scans the folder and categorizes all files
    2. Extracts text content from documents, spreadsheets, etc.
    3. Prepares multimodal file information (images, videos, audio)
    4. Generates a comprehensive task description with tool usage guidance
    
    Args:
        folder_path: Path to the folder to process
        query: The user's query/question about the folder contents
        recursive: Whether to scan subdirectories recursively
        include_file_contents: Whether to include extracted file contents
        max_content_length: Maximum length of content per file
        
    Returns:
        Tuple of:
        - task_content: Full content string for LLM (includes file contents)
        - task_description: Task description with tool guidance
        - multimodal_files: List of paths to multimodal files (images, videos, audio)
    """
    # Scan folder
    contents = scan_folder(folder_path, recursive=recursive)
    
    # Build task description
    task_parts = []
    task_parts.append(f"# Task\n\n{query}\n")
    
    # Add folder summary
    task_parts.append(f"\n## Folder Contents Summary\n\n{contents.get_summary()}\n")
    
    # Process text-extractable files
    if include_file_contents and contents.text_extractable_files:
        task_parts.append("\n## Document Contents\n")
        
        for file_info in contents.text_extractable_files:
            content = _extract_file_content(file_info, max_content_length)
            if content:
                task_parts.append(f"\n### {file_info.name}\n")
                task_parts.append(f"<file path=\"{file_info.path}\">\n{content}\n</file>\n")
    
    # Process multimodal files
    multimodal_files = []
    
    # Images
    if contents.images:
        task_parts.append("\n## Image Files\n")
        task_parts.append("\nThe following image files are available for analysis:\n")
        
        for file_info in contents.images:
            multimodal_files.append(file_info.path)
            task_parts.append(f"\n### {file_info.name}\n")
            task_parts.append(_get_image_info(file_info))
        
        task_parts.append("\n\n**IMPORTANT**: Use the 'vision_understanding_advanced' tool to analyze these images.")
        task_parts.append("This tool provides multi-turn verification, confidence scoring, and cross-validation.")
        task_parts.append("Recommended approach:")
        task_parts.append("1. Call vision_understanding_advanced with a specific question about each image")
        task_parts.append("2. Review the confidence score and metadata")
        task_parts.append("3. If confidence < 0.75, use follow-up analysis or web search for verification\n")
    
    # Videos
    if contents.videos:
        task_parts.append("\n## Video Files\n")
        task_parts.append("\nThe following video files are available for analysis:\n")
        
        for file_info in contents.videos:
            multimodal_files.append(file_info.path)
            task_parts.append(f"\n### {file_info.name}\n")
            task_parts.append(_get_video_info(file_info))
        
        task_parts.append("\n\n**IMPORTANT**: Use the 'video_understanding_advanced' tool to analyze these videos.")
        task_parts.append("Recommendation:")
        task_parts.append("- Use enable_verification=true for detailed action/scene analysis")
        task_parts.append("- For quick preview, use 'video_quick_analysis' tool")
        task_parts.append("- To analyze specific time ranges, use 'video_temporal_qa' with start_time and end_time")
        task_parts.append("- To extract key moments/frames, use 'video_extract_keyframes' tool\n")
    
    # Audio
    if contents.audios:
        task_parts.append("\n## Audio Files\n")
        task_parts.append("\nThe following audio files are available for analysis:\n")
        
        for file_info in contents.audios:
            multimodal_files.append(file_info.path)
            task_parts.append(f"\n### {file_info.name}\n")
            task_parts.append(_get_audio_info(file_info))
        
        task_parts.append("\n\n**IMPORTANT**: Use the 'audio_understanding_advanced' tool to analyze these audio files.")
        task_parts.append("Recommendation:")
        task_parts.append("- Use enable_verification=true for critical transcriptions")
        task_parts.append("- For quick transcription, use 'audio_quick_transcription' tool")
        task_parts.append("- To answer specific questions about the audio, use 'audio_question_answering_enhanced'\n")
    
    # Long context files (RAG)
    # Check for both .json files and pre-built .db files
    long_context_files = [f for f in contents.data_files if "long_context" in f.name.lower()]
    
    # Also check for .db files (pre-built embedding databases)
    db_files = [f for f in contents.other_files if f.name.endswith('.chunks.db') or f.name.endswith('.db')]
    
    if long_context_files or db_files:
        task_parts.append("\n## Long Context Documents (RAG)\n")
        task_parts.append("\nThe following long context document files are available for semantic search:\n")
        
        for file_info in long_context_files:
            # Check if there's a corresponding .db file (pre-built embeddings)
            db_path = file_info.path + ".chunks.db"
            if os.path.exists(db_path):
                db_size = os.path.getsize(db_path)
                task_parts.append(f"\n### {file_info.name}\n")
                task_parts.append(f"Path: {file_info.path}\n")
                task_parts.append(f"Size: {file_info.size_bytes / 1024:.1f} KB\n")
                task_parts.append(f"**Pre-built embedding database available**: {db_path} ({db_size / 1024:.1f} KB)\n")
            else:
                task_parts.append(f"\n### {file_info.name}\n")
                task_parts.append(f"Path: {file_info.path}\n")
                task_parts.append(f"Size: {file_info.size_bytes / 1024:.1f} KB\n")
        
        # List standalone .db files (without corresponding .json)
        for file_info in db_files:
            # Check if this db file has a corresponding json file already listed
            json_path = file_info.path.replace('.chunks.db', '').replace('.db', '')
            if not any(f.path == json_path or f.path == json_path + '.json' for f in long_context_files):
                task_parts.append(f"\n### {file_info.name} (Pre-built Database)\n")
                task_parts.append(f"Path: {file_info.path}\n")
                task_parts.append(f"Size: {file_info.size_bytes / 1024:.1f} KB\n")
                task_parts.append("**This is a pre-built embedding database that can be loaded directly.**\n")
        
        task_parts.append("\n**IMPORTANT**: Use RAG tools to search these long context documents:")
        task_parts.append("- `rag_search`: Semantic search to find relevant passages (use 1-3 times with different keywords)")
        task_parts.append("- `rag_get_context`: Get concatenated context for answering questions")
        task_parts.append("- `rag_document_stats`: Get document collection statistics")
        task_parts.append("\nDo NOT attempt to read these files directly - they are too large. Use RAG tools instead.")
        task_parts.append("If a pre-built .db file exists, the RAG tool will automatically use it for faster loading.\n")
    
    # Other files
    other_files = [f for f in contents.other_files if "long_context" not in f.name.lower()]
    if other_files:
        task_parts.append("\n## Other Files\n")
        for file_info in other_files:
            task_parts.append(f"- {file_info.name} ({file_info.extension})\n")
    
    # Add output format requirement
    use_cn_prompt = os.environ.get("USE_CN_PROMPT", "0")
    if use_cn_prompt == "1":
        task_parts.append("\n请通过任务分解和MCP工具调用来解决给定的问题。**你必须严格遵循请求中的格式要求，并将最终答案包裹在 \\boxed{} 中。**")
    else:
        task_parts.append("\nYou should follow the format instruction in the request strictly and wrap the final answer in \\boxed{}.")
    
    task_content = "\n".join(task_parts)
    task_description = task_content
    
    return task_content, task_description, multimodal_files


def process_folder_batch(
    folder_paths: List[str],
    query: str,
    recursive: bool = False,
    include_file_contents: bool = True,
    max_content_length: int = 200_000
) -> List[Tuple[str, str, str, List[str]]]:
    """
    Process multiple folders in batch.
    
    Args:
        folder_paths: List of folder paths to process
        query: The user's query/question about the folder contents
        recursive: Whether to scan subdirectories recursively
        include_file_contents: Whether to include extracted file contents
        max_content_length: Maximum length of content per file
        
    Returns:
        List of tuples, each containing:
        - folder_path: The original folder path
        - task_content: Full content string for LLM
        - task_description: Task description with tool guidance
        - multimodal_files: List of paths to multimodal files
    """
    results = []
    
    for folder_path in folder_paths:
        try:
            task_content, task_description, multimodal_files = process_folder_for_task(
                folder_path=folder_path,
                query=query,
                recursive=recursive,
                include_file_contents=include_file_contents,
                max_content_length=max_content_length
            )
            results.append((folder_path, task_content, task_description, multimodal_files))
        except Exception as e:
            error_msg = f"Error processing folder {folder_path}: {str(e)}"
            results.append((folder_path, error_msg, error_msg, []))
    
    return results
