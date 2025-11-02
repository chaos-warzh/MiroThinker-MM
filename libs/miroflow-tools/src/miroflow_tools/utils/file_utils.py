"""
Common utility functions for file operations
Extracted to eliminate code duplication across enhanced_vqa, enhanced_audio, and enhanced_video
"""
import aiohttp
import asyncio
import os
import tempfile
from typing import Tuple
from urllib.parse import urlparse


async def ensure_local_file(file_path: str) -> Tuple[str, bool]:
    """
    Ensure file is available locally, downloading if it's a URL.
    
    Args:
        file_path: Local file path or URL
        
    Returns:
        Tuple of (local_path, is_temp_file)
        - local_path: Path to the local file
        - is_temp_file: True if file was downloaded and should be cleaned up
        
    Raises:
        ValueError: If URL download fails
        FileNotFoundError: If local file doesn't exist
    """
    # If it's a URL, download it
    if file_path.startswith(("http://", "https://")):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(file_path, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    response.raise_for_status()
                    
                    # Create temp file with appropriate extension
                    parsed_url = urlparse(file_path)
                    _, ext = os.path.splitext(parsed_url.path)
                    if not ext:
                        ext = ".tmp"
                    
                    # Use thread pool for file I/O operations
                    content = await response.read()
                    
                    def write_temp_file():
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode='wb') as temp_file:
                            temp_file.write(content)
                            return temp_file.name
                    
                    temp_file_path = await asyncio.to_thread(write_temp_file)
                    return temp_file_path, True
                    
        except Exception as e:
            raise ValueError(f"Failed to download file from URL: {e}")
    
    # Local file - verify it exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Local file not found: {file_path}")
    
    return file_path, False


def cleanup_temp_file(file_path: str, is_temp: bool) -> None:
    """
    Clean up temporary file if needed.
    
    Args:
        file_path: Path to file
        is_temp: Whether file is temporary and should be deleted
    """
    if is_temp and file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass  # Ignore cleanup errors


async def cleanup_temp_file_async(file_path: str, is_temp: bool) -> None:
    """
    Async version of cleanup_temp_file for use in async contexts.
    
    Args:
        file_path: Path to file
        is_temp: Whether file is temporary and should be deleted
    """
    if is_temp and file_path and os.path.exists(file_path):
        try:
            await asyncio.to_thread(os.remove, file_path)
        except Exception:
            pass  # Ignore cleanup errors
