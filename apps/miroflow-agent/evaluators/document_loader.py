"""
文档加载器 - 加载和管理源文档、long-context 等
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False


def normalize_title(text: str) -> str:
    """标准化标题：繁简转换、全半角转换、去除多余空白"""
    if not text:
        return ""
    
    # 全角转半角
    result = ""
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result += chr(code - 0xFEE0)
        elif code == 0x3000:
            result += ' '
        else:
            result += char
    
    # 标准化引号
    result = result.replace('"', '"').replace('"', '"')
    result = result.replace(''', "'").replace(''', "'")
    result = result.replace('「', '"').replace('」', '"')
    result = result.replace('『', '"').replace('』', '"')
    
    # 去除多余空白和标点
    result = re.sub(r'\s+', ' ', result).strip()
    result = result.lower()
    
    return result


def fuzzy_title_match(needle: str, haystack: str, threshold: float = 0.7) -> bool:
    """模糊标题匹配"""
    needle_norm = normalize_title(needle)
    haystack_norm = normalize_title(haystack)
    
    if not needle_norm or not haystack_norm:
        return False
    
    # 精确匹配
    if needle_norm == haystack_norm:
        return True
    
    # 包含匹配
    if needle_norm in haystack_norm or haystack_norm in needle_norm:
        return True
    
    # 分词匹配：检查关键词是否大部分出现
    # 去除常见停用词
    stopwords = {'的', '了', '是', '在', '和', '与', 'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with'}
    needle_words = [w for w in needle_norm.split() if w not in stopwords and len(w) > 1]
    
    if len(needle_words) > 1:
        matched = sum(1 for w in needle_words if w in haystack_norm)
        if matched / len(needle_words) >= threshold:
            return True
    
    return False


class DocumentLoader:
    """文档加载器"""
    
    def __init__(self):
        self.source_documents: Dict[str, str] = {}
        self.long_context: List[Dict] = []
        self._title_to_idx: Dict[str, int] = {}
    
    def load_json(self, path: Path) -> Any:
        """加载 JSON 文件"""
        return json.loads(path.read_text(encoding='utf-8'))
    
    def load_long_context(self, path: Path) -> None:
        """加载 long-context 文件"""
        data = self.load_json(path)
        self.long_context = data if isinstance(data, list) else [data] if data else []
        self._build_index()
    
    def _build_index(self) -> None:
        """构建标题索引"""
        self._title_to_idx.clear()
        for idx, item in enumerate(self.long_context):
            if isinstance(item, dict):
                title = item.get('title', '').lower().strip()
                if title:
                    self._title_to_idx[title] = idx
    
    def load_source_folder(self, source_folder: Path) -> None:
        """加载源文档文件夹"""
        if not source_folder.exists():
            return
        
        # 检查是否有 source 子文件夹
        source_subfolder = source_folder / "source"
        if source_subfolder.exists() and source_subfolder.is_dir():
            source_folder = source_subfolder
        
        for file_path in source_folder.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                self._load_single_file(file_path)
    
    def _load_single_file(self, file_path: Path) -> None:
        """加载单个文件"""
        suffix = file_path.suffix.lower()
        
        # 纯文本类文件
        if suffix in {".txt", ".md", ".json"}:
            try:
                content = file_path.read_text(encoding="utf-8")
                self.source_documents[file_path.name] = content
            except Exception as e:
                self.source_documents[file_path.name] = f"[Text file unreadable: {file_path.name}, error: {e}]"
            return
        
        # 二进制文档（docx, pdf 等）
        if suffix in {".doc", ".docx", ".pdf", ".ppt", ".pptx", ".html", ".htm"}:
            if HAS_MARKITDOWN:
                try:
                    md = MarkItDown()
                    result = md.convert(str(file_path))
                    text = getattr(result, "text", None) or str(result)
                    self.source_documents[file_path.name] = text
                except Exception as e:
                    self.source_documents[file_path.name] = f"[Binary file unreadable: {file_path.name}, error: {e}]"
            else:
                self.source_documents[file_path.name] = f"[Binary file (install markitdown to parse): {file_path.name}]"
            return
        
        # 其他类型文件
        try:
            content = file_path.read_text(encoding="utf-8")
            self.source_documents[file_path.name] = content
        except Exception:
            self.source_documents[file_path.name] = f"[Binary file: {file_path.name}]"
    
    def get_content_by_index(self, idx: int) -> Tuple[Optional[str], Optional[str]]:
        """通过索引获取 long-context 内容"""
        if 0 <= idx < len(self.long_context):
            item = self.long_context[idx]
            if isinstance(item, dict):
                return item.get('page_body', ''), item.get('title', f'long_context[{idx}]')
        return None, None
    
    def get_content_by_title(self, title: str) -> Tuple[Optional[str], Optional[str]]:
        """通过标题获取 long-context 内容（支持模糊匹配）"""
        title_norm = normalize_title(title)
        
        # 精确匹配（标准化后）
        for idx, item in enumerate(self.long_context):
            if isinstance(item, dict):
                item_title = item.get('title', '')
                if normalize_title(item_title) == title_norm:
                    return item.get('page_body', ''), item.get('title', f'long_context[{idx}]')
        
        # 模糊匹配
        for idx, item in enumerate(self.long_context):
            if isinstance(item, dict):
                item_title = item.get('title', '')
                if fuzzy_title_match(title, item_title):
                    return item.get('page_body', ''), item.get('title', f'long_context[{idx}]')
        
        return None, None
    
    def get_source_document(self, filename: str) -> Optional[str]:
        """获取源文档内容"""
        return self.source_documents.get(filename)
