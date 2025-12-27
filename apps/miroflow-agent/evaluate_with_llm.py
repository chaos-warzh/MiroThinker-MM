#!/usr/bin/env python3
"""
评估生成的 report 质量 - 重构版本

主要改进：
1. 代码结构：拆分成多个职责单一的类
2. 时间复杂度：使用索引实现 O(1) 查找，避免重复遍历
3. 代码复用：消除重复代码
4. 可维护性：更清晰的类型注解和文档

Usage:
    uv run python evaluate_with_llm.py \
        --result examples/001/001.md \
        --gold examples/001/gold_insights.json \
        --gold-source examples/001/gold_insights_from_source.json \
        --metadata examples/001/metadata.json \
        --source-folder examples/001/source \
        --long-context examples/001/long_context.json \
        --output examples/001/evaluation_report.txt
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import os

from openai import OpenAI

# 尝试使用 MarkItDown 读取二进制文档
try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class EvalConfig:
    """评估配置"""
    api_key: str = field(default_factory=lambda: os.getenv(
        "OPENAI_API_KEY", "sk-D4FTmbfAeJlrFBQfU9enqHs9Vu3Lvu2JANL1z9jD3KlWCq9F"
    ))
    base_url: str = field(default_factory=lambda: os.getenv(
        "OPENAI_BASE_URL", "http://14.103.68.46/v1"
    ))
    model_name: str = "gpt-51-1113-global"
    max_segment_length: int = 8000
    batch_size: int = 10
    temperature: float = 0.1
    
    # 评估权重
    weights: Dict[str, float] = field(default_factory=lambda: {
        'information_recall': 0.25,
        'factual_accuracy': 0.25,
        'overall_quality': 0.15,
        'checklist_compliance': 0.15,  # 替代 format_compliance
        'citation_coverage': 0.10,  # 新增引用覆盖率
        'expression_requirement': 0.10
    })
    
    # Checklist 配置
    max_checklist_items: int = 10  # 限制项数以优化时间复杂度


@dataclass
class EvalScore:
    """评估分数结构"""
    information_recall: float
    factual_accuracy: float
    overall_quality: float
    checklist_compliance: float  # 替代 format_compliance
    citation_coverage: float  # 新增引用覆盖率
    expression_requirement: float
    total_score: float
    details: Dict[str, Any]


@dataclass
class ChecklistItem:
    """单个 Checklist 项"""
    id: int
    category: str
    requirement: str
    importance: str
    weight: float


@dataclass
class ChecklistEvaluation:
    """单个 Checklist 项的评估结果"""
    item_id: int
    requirement: str
    satisfied: bool
    score: float
    evidence: str
    explanation: str


# =============================================================================
# 文档加载器 - 负责加载和索引文档
# =============================================================================

class DocumentLoader:
    """文档加载器，支持 O(1) 标题查找"""
    
    TEXT_EXTENSIONS = {".txt", ".md", ".json"}
    BINARY_EXTENSIONS = {".doc", ".docx", ".pdf", ".ppt", ".pptx", ".html", ".htm"}
    
    def __init__(self):
        self.source_documents: Dict[str, str] = {}
        self.long_context: List[Dict] = []
        # 索引：实现 O(1) 查找
        self._title_to_idx: Dict[str, int] = {}
        self._word_to_indices: Dict[str, List[int]] = {}
    
    def load_json(self, path: Path) -> Any:
        """加载 JSON 文件"""
        return json.loads(path.read_text(encoding='utf-8'))
    
    def load_source_folder(self, folder: Path) -> None:
        """加载源文档文件夹"""
        if not folder.exists():
            return
        
        # 检查 source 子文件夹
        subfolder = folder / "source"
        if subfolder.exists() and subfolder.is_dir():
            folder = subfolder
        
        for file_path in folder.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                self.source_documents[file_path.name] = self._read_file(file_path)
    
    def load_long_context(self, path: Path) -> None:
        """加载并索引 long context"""
        data = self.load_json(path)
        self.long_context = data if isinstance(data, list) else [data] if data else []
        self._build_index()
    
    def _build_index(self) -> None:
        """构建标题索引，实现 O(1) 查找"""
        self._title_to_idx.clear()
        self._word_to_indices.clear()
        
        for idx, item in enumerate(self.long_context):
            if not isinstance(item, dict):
                continue
            title = item.get('title', '').lower().strip()
            if not title:
                continue
            
            # 精确匹配索引
            self._title_to_idx[title] = idx
            
            # 词级索引（用于模糊匹配）
            for word in title.split():
                if word not in self._word_to_indices:
                    self._word_to_indices[word] = []
                self._word_to_indices[word].append(idx)
    
    def _read_file(self, path: Path) -> str:
        """读取文件内容"""
        suffix = path.suffix.lower()
        
        if suffix in self.TEXT_EXTENSIONS:
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                return f"[无法读取文本文件: {path.name}, 错误: {e}]"
        
        if suffix in self.BINARY_EXTENSIONS:
            if HAS_MARKITDOWN:
                try:
                    result = MarkItDown().convert(str(path))
                    return getattr(result, "text", None) or str(result)
                except Exception as e:
                    return f"[无法解析二进制文件: {path.name}, 错误: {e}]"
            return f"[二进制文件（需安装 markitdown）: {path.name}]"
        
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return f"[二进制文件: {path.name}]"
    
    def get_content(self, source_file: str, source_type: str) -> Tuple[Optional[str], Optional[str]]:
        """获取源内容，使用索引实现快速查找"""
        if source_type == 'doc':
            return self._get_doc(source_file)
        elif source_type == 'rag':
            return self._get_rag(source_file)
        elif source_type == 'long_context':
            return self._get_by_title(source_file)
        return None, None
    
    def _get_doc(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """获取文档内容"""
        if filename in self.source_documents:
            return self.source_documents[filename], filename
        
        # 回退：在 long_context 中搜索
        filename_lower = filename.lower()
        for idx, item in enumerate(self.long_context):
            if isinstance(item, dict):
                if filename_lower in item.get('page_body', '').lower() or \
                   filename_lower in item.get('title', '').lower():
                    return item.get('page_body', ''), f"long_context[{idx}]"
        return None, None
    
    def _get_rag(self, source_file: str) -> Tuple[Optional[str], Optional[str]]:
        """获取 RAG 内容（O(1) 索引访问）"""
        match = re.search(r'RAG-(\d+)', source_file)
        if not match:
            return None, None
        
        idx = int(match.group(1)) - 1  # RAG-1 -> 索引 0
        if 0 <= idx < len(self.long_context):
            item = self.long_context[idx]
            if isinstance(item, dict):
                return item.get('page_body', ''), f"long_context[{idx}] ({item.get('title', '')})"
        return None, None
    
    def _get_by_title(self, title: str) -> Tuple[Optional[str], Optional[str]]:
        """通过标题获取内容（O(1) 精确匹配 + 模糊匹配）"""
        title_lower = title.lower().strip()
        
        # 1. 精确匹配 O(1)
        if title_lower in self._title_to_idx:
            idx = self._title_to_idx[title_lower]
            item = self.long_context[idx]
            return item.get('page_body', ''), f"long_context[{idx}] ({item.get('title', '')})"
        
        # 2. 包含匹配
        for stored_title, idx in self._title_to_idx.items():
            if title_lower in stored_title or stored_title in title_lower:
                item = self.long_context[idx]
                return item.get('page_body', ''), f"long_context[{idx}] ({item.get('title', '')})"
        
        # 3. 模糊匹配（使用词级索引）
        title_words = set(title_lower.split())
        if not title_words:
            return None, None
        
        # 统计候选项的匹配词数
        candidate_scores: Dict[int, int] = {}
        for word in title_words:
            for idx in self._word_to_indices.get(word, []):
                candidate_scores[idx] = candidate_scores.get(idx, 0) + 1
        
        # 找最佳匹配
        best_idx, best_score = -1, 0.3
        for idx, overlap in candidate_scores.items():
            item_words = set(self.long_context[idx].get('title', '').lower().split())
            if item_words:
                similarity = overlap / len(title_words | item_words)
                if similarity > best_score:
                    best_score, best_idx = similarity, idx
        
        if best_idx >= 0:
            item = self.long_context[best_idx]
            return item.get('page_body', ''), f"long_context[{best_idx}] ({item.get('title', '')})"
        
        return None, None


# =============================================================================
# 引用提取器 - 负责从文本中提取引用
# =============================================================================

@dataclass
class CitationRef:
    """引用信息"""
    citation: str
    source_file: str
    source_type: str  # 'doc', 'rag', 'long_context'
    chunk_info: Optional[str] = None


@dataclass
class CitationContext:
    """引用及其上下文"""
    citations: List[str]
    context: str
    refs: List[CitationRef]


class CitationExtractor:
    """引用提取器"""
    
    PATTERN = re.compile(
        r'\[(?:Doc:\s*[^\]]+|文档:\s*[^\]]+|RAG-\d+|long_context:\s*"[^"]+",\s*chunk\s*[\d,\s]+)\]'
    )
    
    def extract(self, text: str) -> List[CitationContext]:
        """提取所有引用及其上下文"""
        positions = [
            {'start': m.start(), 'end': m.end(), 'text': m.group(0)}
            for m in self.PATTERN.finditer(text)
        ]
        
        if not positions:
            return []
        
        # 分组连续引用
        groups = self._group_consecutive(positions, text)
        
        # 为每组创建 CitationContext
        results = []
        for group in groups:
            ctx = self._create_context(group, text)
            if ctx:
                results.append(ctx)
        return results
    
    def _group_consecutive(self, positions: List[Dict], text: str) -> List[List[Dict]]:
        """将连续的引用分组"""
        if not positions:
            return []
        
        groups = []
        current = [positions[0]]
        
        for pos in positions[1:]:
            gap = text[current[-1]['end']:pos['start']]
            if re.match(r'^[\s\[\]]*$', gap):
                current.append(pos)
            else:
                groups.append(current)
                current = [pos]
        
        groups.append(current)
        return groups
    
    def _create_context(self, group: List[Dict], text: str) -> Optional[CitationContext]:
        """创建引用上下文"""
        if not group:
            return None
        
        # 提取引用前的文本作为上下文
        first_start = group[0]['start']
        text_before = text[:first_start]
        
        # 获取最后一个句子
        sentences = re.split(r'([.!?。！？]\s*)', text_before)
        if len(sentences) >= 2:
            context = ''.join(sentences[-2:]).strip()
        else:
            context = text_before[-200:].strip()
        context = re.sub(r'\s+', ' ', context)
        
        # 解析每个引用
        citations = []
        refs = []
        
        for pos in group:
            citation_text = pos['text']
            citations.append(citation_text)
            ref = self._parse_citation(citation_text)
            if ref:
                refs.append(ref)
        
        if not context or not refs:
            return None
        
        return CitationContext(citations=citations, context=context, refs=refs)
    
    def _parse_citation(self, text: str) -> Optional[CitationRef]:
        """解析单个引用"""
        if text.startswith('[Doc:') or text.startswith('[文档:'):
            match = re.search(r'\[(?:Doc|文档):\s*([^\]]+)\]', text)
            if match:
                return CitationRef(text, match.group(1).strip(), 'doc')
        
        elif text.startswith('[RAG-'):
            match = re.search(r'\[RAG-(\d+)\]', text)
            if match:
                return CitationRef(text, f"RAG-{match.group(1)}", 'rag')
        
        elif text.startswith('[long_context:'):
            match = re.search(r'\[long_context:\s*"([^"]+)",\s*chunk\s*([\d,\s]+)\]', text)
            if match:
                return CitationRef(text, match.group(1).strip(), 'long_context', match.group(2).strip())
        
        return None


# =============================================================================
# LLM 客户端 - 封装 LLM 调用
# =============================================================================

class LLMClient:
    """LLM 客户端，封装所有 LLM 调用"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    
    def call(self, system: str, user: str, json_mode: bool = True) -> Optional[Dict]:
        """调用 LLM"""
        try:
            kwargs = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": self.config.temperature
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return json.loads(content) if json_mode else {"text": content}
        except Exception as e:
            print(f"LLM 调用错误: {e}")
            return None


# =============================================================================
# 内容分段器 - 处理长文本分段
# =============================================================================

class ContentSegmenter:
    """内容分段器"""
    
    def __init__(self, max_length: int = 8000):
        self.max_length = max_length
    
    def segment(self, content: str) -> List[str]:
        """将长内容分段"""
        if len(content) <= self.max_length:
            return [content]
        
        segments = []
        paragraphs = content.split('\n\n')
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.max_length:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    segments.append(current)
                
                if len(para) > self.max_length:
                    # 按句子分割超长段落
                    segments.extend(self._split_by_sentences(para))
                    current = ""
                else:
                    current = para
        
        if current:
            segments.append(current)
        
        return segments if segments else [content]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分割"""
        sentences = re.split(r'([.!?。！？]\s*)', text)
        segments = []
        current = ""
        
        for i in range(0, len(sentences), 2):
            sent = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            if len(current) + len(sent) <= self.max_length:
                current += sent
            else:
                if current:
                    segments.append(current)
                current = sent
        
        if current:
            segments.append(current)
        return segments


# =============================================================================
# 评估器 - 核心评估逻辑
# =============================================================================

class Evaluator:
    """评估器"""
    
    def __init__(self, config: EvalConfig, doc_loader: DocumentLoader, 
                 llm: LLMClient, result_text: str, gold_data: Dict,
                 source_gold_data: Dict, metadata: Dict, query_data: Optional[Dict] = None):
        self.config = config
        self.docs = doc_loader
        self.llm = llm
        self.result_text = result_text
        self.gold_data = gold_data
        self.source_gold_data = source_gold_data
        self.metadata = metadata
        self.query_data = query_data
        self.extractor = CitationExtractor()
        self.segmenter = ContentSegmenter(config.max_segment_length)
    
    def evaluate(self) -> EvalScore:
        """执行完整评估"""
        print("开始 LLM 评估...")
        
        print("1. 评估信息召回率...")
        info_score, info_details = self._eval_information_recall()
        
        print("2. 评估事实准确性...")
        fact_score, fact_details = self._eval_factual_accuracy()
        
        print("3. 评估整体质量...")
        quality_score, quality_details = self._eval_overall_quality()
        
        print("4. 评估 Checklist 符合性...")
        checklist_score, checklist_details = self._eval_checklist_compliance()
        
        print("5. 评估引用覆盖率...")
        citation_score, citation_details = self._eval_citation_coverage()
        
        print("6. 评估表达要求...")
        expr_score, expr_details = self._eval_expression_requirement()
        
        # 计算总分
        w = self.config.weights
        total = (
            info_score * w['information_recall'] +
            fact_score * w['factual_accuracy'] +
            quality_score * w['overall_quality'] +
            checklist_score * w['checklist_compliance'] +
            citation_score * w['citation_coverage'] +
            expr_score * w['expression_requirement']
        )
        
        return EvalScore(
            information_recall=info_score,
            factual_accuracy=fact_score,
            overall_quality=quality_score,
            checklist_compliance=checklist_score,
            citation_coverage=citation_score,
            expression_requirement=expr_score,
            total_score=total,
            details={
                'information_recall': info_details,
                'factual_accuracy': fact_details,
                'overall_quality': quality_details,
                'checklist_compliance': checklist_details,
                'citation_coverage': citation_details,
                'expression_requirement': expr_details
            }
        )
    
    # -------------------------------------------------------------------------
    # 信息召回评估
    # -------------------------------------------------------------------------
    
    def _eval_information_recall(self) -> Tuple[float, Dict]:
        """评估信息召回率（50% long_context + 50% source_documents）"""
        long_insights = self.gold_data.get('gold_insights', [])
        source_insights = self.source_gold_data.get('gold_insights', [])
        
        long_score, long_details, has_long = self._eval_recall_component(long_insights, 'long_context')
        source_score, source_details, has_source = self._eval_recall_component(source_insights, 'source_documents')
        
        # 加权平均
        total_weight = (0.5 if has_long else 0) + (0.5 if has_source else 0)
        if total_weight > 0:
            combined = ((long_score * 0.5 if has_long else 0) + 
                       (source_score * 0.5 if has_source else 0)) / total_weight
        else:
            combined = 100.0
        
        return combined, {
            'combined_score': combined,
            'components': {
                'long_context': {'score': long_score, 'details': long_details, 'available': has_long},
                'source_documents': {'score': source_score, 'details': source_details, 'available': has_source}
            }
        }
    
    def _eval_recall_component(self, insights: List[Dict], label: str) -> Tuple[float, Dict, bool]:
        """评估单个来源的召回率"""
        if not insights:
            return 100.0, {'message': f'No gold insights for {label}'}, False
        
        insights_text = "\n".join([f"{i+1}. {ins['insight']}" for i, ins in enumerate(insights)])
        
        prompt = f"""You are an expert evaluator. Analyze how the report covers the required insights.

Required Insights:
{insights_text}

Report:
{self.result_text}

Be GENEROUS. For each insight, determine if the report covers its factual core.
Respond in JSON: {{"covered_insights": [{{"id": 1, "covered": true/false, "explanation": "..."}}]}}"""
        
        result = self.llm.call("You are a precise text analysis expert.", prompt)
        
        if not result:
            return self._fallback_recall(insights), {'fallback': True}, True
        
        covered = result.get('covered_insights', [])
        total = len(covered) if covered else len(insights)
        count = sum(1 for c in covered if c.get('covered'))
        score = 100.0 * count / total if total > 0 else 100.0
        
        result['total_insights'] = total
        result['covered_count'] = count
        result['recall_percentage'] = score
        
        return score, {'evaluation_result': result}, True
    
    def _fallback_recall(self, insights: List[Dict]) -> float:
        """降级召回评估"""
        count = sum(1 for ins in insights if ins['insight'].lower() in self.result_text.lower())
        return 100.0 * count / len(insights) if insights else 100.0
    
    # -------------------------------------------------------------------------
    # 事实准确性评估
    # -------------------------------------------------------------------------
    
    def _eval_factual_accuracy(self) -> Tuple[float, Dict]:
        """评估事实准确性"""
        citations = self.extractor.extract(self.result_text)
        
        if not citations:
            return self._basic_factual_check()
        
        # 准备验证项
        items = []
        for ctx in citations:
            for ref in ctx.refs:
                content, name = self.docs.get_content(ref.source_file, ref.source_type)
                items.append({
                    'citation': ref.citation,
                    'context': ctx.context,
                    'source_content': content,
                    'source_name': name
                })
        
        if not items:
            return self._basic_factual_check()
        
        # 分批验证
        all_results = []
        supported = not_found = unsupported = 0
        
        for i in range(0, len(items), self.config.batch_size):
            batch = items[i:i + self.config.batch_size]
            print(f"  处理批次 {i // self.config.batch_size + 1}...")
            
            for item in batch:
                result = self._verify_citation(item)
                all_results.append(result)
                
                if not result.get('source_found'):
                    not_found += 1
                elif result.get('supported'):
                    supported += 1
                else:
                    unsupported += 1
        
        total = len(items)
        score = 100.0 * supported / total if total > 0 else 0
        
        return score, {
            'verification_result': {
                'verifications': all_results,
                'total_citations': total,
                'supported_count': supported,
                'not_found_count': not_found,
                'unsupported_count': unsupported,
                'accuracy_score': score
            }
        }
    
    def _verify_citation(self, item: Dict) -> Dict:
        """验证单个引用"""
        if not item['source_content']:
            return {'citation': item['citation'], 'supported': False, 'source_found': False, 
                    'explanation': 'Source not found'}
        
        content = item['source_content']
        segments = self.segmenter.segment(content)
        
        # 检查每个分段
        for seg_idx, segment in enumerate(segments):
            prompt = f"""Verify if the statement is supported by the source content.

Citation: {item['citation']}
Statement: {item['context']}
Source ({item['source_name']}), segment {seg_idx + 1}/{len(segments)}:
{segment}

A citation is WRONG only if it CONFLICTS with or CANNOT be inferred from the source.
Respond in JSON: {{"supported": true/false, "explanation": "..."}}"""
            
            result = self.llm.call("You are a fact-checker.", prompt)
            if result and result.get('supported'):
                return {'citation': item['citation'], 'supported': True, 'source_found': True,
                        'explanation': f"Supported by segment {seg_idx + 1}. {result.get('explanation', '')}"}
        
        return {'citation': item['citation'], 'supported': False, 'source_found': True,
                'explanation': 'Not supported by any segment'}
    
    def _basic_factual_check(self) -> Tuple[float, Dict]:
        """基础事实检查"""
        citations = re.findall(r'\[[^\]]+\]', self.result_text)
        score = 70 if citations else 50
        return score, {'method': 'basic', 'citation_count': len(citations)}
    
    # -------------------------------------------------------------------------
    # 整体质量评估
    # -------------------------------------------------------------------------
    
    def _eval_overall_quality(self) -> Tuple[float, Dict]:
        """评估整体质量"""
        prompt = f"""Assess the overall quality of this report.

Report:
{self.result_text}

Evaluate: Clarity, Structure, Engagement, Depth, Coherence.
Respond in JSON: {{
    "clarity_score": 0-100, "structure_score": 0-100, "engagement_score": 0-100,
    "depth_score": 0-100, "coherence_score": 0-100, "overall_score": 0-100,
    "strengths": [...], "weaknesses": [...], "explanation": "..."
}}"""
        
        result = self.llm.call("You are an expert writing evaluator.", prompt)
        
        if not result:
            return self._basic_quality_check()
        
        return result.get('overall_score', 70), {'quality_assessment': result}
    
    def _basic_quality_check(self) -> Tuple[float, Dict]:
        """基础质量检查"""
        words = len(self.result_text.split())
        paragraphs = len([p for p in self.result_text.split('\n\n') if p.strip()])
        score = 70 + (10 if words >= 300 else 0) + (10 if 3 <= paragraphs <= 8 else 0)
        return min(100, score), {'method': 'basic', 'word_count': words}
    
    # -------------------------------------------------------------------------
    # Checklist 符合性评估
    # -------------------------------------------------------------------------
    
    def _eval_checklist_compliance(self) -> Tuple[float, Dict]:
        """评估 Checklist 符合性"""
        # 获取查询文本
        query_text = self._get_query_text()
        if not query_text:
            return 100.0, {'message': 'No query found for checklist generation'}
        
        # 生成 checklist
        checklist = self._generate_checklist(query_text)
        if not checklist:
            return 100.0, {'message': 'Failed to generate checklist'}
        
        # 一次性评估所有项
        evaluations = self._evaluate_checklist(checklist)
        
        # 计算得分
        total_score = sum(e.score for e in evaluations) / len(evaluations) if evaluations else 100.0
        satisfied_count = sum(1 for e in evaluations if e.satisfied)
        
        # 计算加权得分（考虑重要性）
        eval_map = {e.item_id: e for e in evaluations}
        weighted_sum = sum(
            eval_map[item.id].score * item.weight
            for item in checklist if item.id in eval_map
        )
        weight_sum = sum(item.weight for item in checklist if item.id in eval_map)
        weighted_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        return weighted_score, {
            'checklist': [self._item_to_dict(item) for item in checklist],
            'evaluations': [self._eval_to_dict(e) for e in evaluations],
            'total_score': total_score,
            'weighted_score': weighted_score,
            'satisfied_count': satisfied_count,
            'total_items': len(checklist)
        }
    
    def _get_query_text(self) -> Optional[str]:
        """获取查询文本"""
        # 优先从 query_data 获取
        if self.query_data and self.query_data.get('query'):
            return self.query_data['query']
        
        # 从报告中提取
        query_match = re.search(r'##\s*Query\s*(.*?)(?=\n##\s+|\Z)', self.result_text, re.S | re.I)
        if query_match:
            return query_match.group(1).strip()
        
        return None
    
    def _generate_checklist(self, query: str) -> List[ChecklistItem]:
        """生成 Checklist"""
        prompt = f"""根据查询问题，生成一个标准答案应该包含的 checklist。

查询问题：
{query}

要求：
1. 分析查询中的每个具体要求
2. 生成不超过 {self.config.max_checklist_items} 个检查项
3. 为每项分配权重（总和为1.0）
4. 标注重要性：critical/important/nice_to_have

输出 JSON：
{{
    "checklist": [
        {{"id": 1, "category": "content", "requirement": "具体要求", "importance": "critical", "weight": 0.2}},
        ...
    ]
}}"""
        
        result = self.llm.call("你是一个专业的内容评估专家。", prompt)
        if not result or "checklist" not in result:
            return []
        
        items = []
        for item_data in result["checklist"]:
            try:
                items.append(ChecklistItem(
                    id=item_data["id"],
                    category=item_data.get("category", "content"),
                    requirement=item_data["requirement"],
                    importance=item_data.get("importance", "important"),
                    weight=float(item_data.get("weight", 0.1))
                ))
            except (KeyError, ValueError):
                continue
        
        # 归一化权重
        total_weight = sum(item.weight for item in items)
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            for item in items:
                item.weight = item.weight / total_weight
        
        return items
    
    def _evaluate_checklist(self, checklist: List[ChecklistItem]) -> List[ChecklistEvaluation]:
        """一次性评估所有 checklist 项"""
        if not checklist:
            return []
        
        checklist_text = "\n".join([
            f"[{item.id}] {item.category} | {item.importance} | {item.requirement}"
            for item in checklist
        ])
        
        prompt = f"""评估报告是否满足以下所有 checklist 要求，为每项打分。

Checklist：
{checklist_text}

报告内容：
{self.result_text[:15000]}  # 限制长度

评分标准：
- 完全满足：90-100分
- 大部分满足：70-89分
- 部分满足：50-69分
- 少量涉及：30-49分
- 未涉及：0-29分

输出 JSON：
{{
    "evaluations": [
        {{"item_id": 1, "satisfied": true/false, "score": 0-100, "evidence": "证据", "explanation": "说明"}},
        ...
    ]
}}"""
        
        result = self.llm.call("你是一个严格但公正的内容评估专家。", prompt)
        if not result or "evaluations" not in result:
            return []
        
        evaluations = []
        eval_map = {e.get("item_id"): e for e in result["evaluations"]}
        
        for item in checklist:
            if item.id in eval_map:
                e = eval_map[item.id]
                evaluations.append(ChecklistEvaluation(
                    item_id=item.id,
                    requirement=item.requirement,
                    satisfied=e.get("satisfied", False),
                    score=float(e.get("score", 0)),
                    evidence=str(e.get("evidence", ""))[:150],
                    explanation=str(e.get("explanation", ""))[:80]
                ))
            else:
                evaluations.append(ChecklistEvaluation(
                    item_id=item.id,
                    requirement=item.requirement,
                    satisfied=False,
                    score=0,
                    evidence="未评估",
                    explanation="该项未在评估结果中"
                ))
        
        return evaluations
    
    def _item_to_dict(self, item: ChecklistItem) -> Dict:
        """ChecklistItem 转字典"""
        return {
            'id': item.id,
            'category': item.category,
            'requirement': item.requirement,
            'importance': item.importance,
            'weight': item.weight
        }
    
    def _eval_to_dict(self, eval: ChecklistEvaluation) -> Dict:
        """ChecklistEvaluation 转字典"""
        return {
            'item_id': eval.item_id,
            'requirement': eval.requirement,
            'satisfied': eval.satisfied,
            'score': eval.score,
            'evidence': eval.evidence,
            'explanation': eval.explanation
        }
    
    # -------------------------------------------------------------------------
    # 引用覆盖率评估
    # -------------------------------------------------------------------------
    
    def _eval_citation_coverage(self) -> Tuple[float, Dict]:
        """评估引用覆盖率 - 检查是否引用了 useful_search.json 中的必需文档"""
        # 加载 useful_search.json
        useful_titles = self._load_useful_search()
        if not useful_titles:
            return 100.0, {'message': 'No useful_search.json found or no titles to check'}
        
        # 标准化报告内容中的引号
        report_normalized = self._normalize_quotes(self.result_text)
        
        # 检查每个标题是否被引用
        cited = []
        missing = []
        
        for title in useful_titles:
            # 提取标题的主要部分（去掉网站后缀）
            main_title = title.split("-")[0].strip() if "-" in title else title
            
            # 标准化标题中的引号
            main_title_normalized = self._normalize_quotes(main_title)
            title_normalized = self._normalize_quotes(title)
            
            # 检查是否在报告中出现
            if (main_title in self.result_text or title in self.result_text or 
                main_title_normalized in report_normalized or 
                title_normalized in report_normalized):
                cited.append(title)
            else:
                missing.append(title)
        
        # 计算覆盖率
        total = len(useful_titles)
        cited_count = len(cited)
        coverage_rate = (cited_count / total * 100) if total > 0 else 100.0
        
        return coverage_rate, {
            'total_required': total,
            'cited_count': cited_count,
            'coverage_rate': coverage_rate,
            'cited_titles': cited,
            'missing_titles': missing
        }
    
    def _load_useful_search(self) -> List[str]:
        """加载 useful_search.json 中的标题列表"""
        # 尝试多个可能的路径
        possible_paths = []
        
        # 从 metadata 或 gold_data 获取 case 编号
        case_num = self.metadata.get('number') or self.gold_data.get('number')
        if case_num:
            case_id = f"{int(case_num):03d}" if str(case_num).isdigit() else str(case_num)
            possible_paths.extend([
                Path(f"datasets_batch2/{case_id}/useful_search.json"),
                Path(f"datasets/{case_id}/useful_search.json"),
                Path(f"data/{case_id}/useful_search.json")
            ])
        
        # 尝试从源文件夹路径推断
        if hasattr(self.docs, 'source_folder_path'):
            folder = Path(self.docs.source_folder_path)
            possible_paths.append(folder / "useful_search.json")
            possible_paths.append(folder.parent / "useful_search.json")
        
        # 查找文件
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    titles = [item.get("title", "") for item in data if item.get("title")]
                    return titles
                except Exception:
                    continue
        
        return []
    
    def _normalize_quotes(self, s: str) -> str:
        """标准化引号：将中文引号转换为英文引号"""
        # 中文双引号
        s = s.replace('"', '"').replace('"', '"')
        # 中文单引号
        s = s.replace(''', "'").replace(''', "'")
        return s
    
    # -------------------------------------------------------------------------
    # 表达要求评估
    # -------------------------------------------------------------------------
    
    def _eval_expression_requirement(self) -> Tuple[float, Dict]:
        """评估表达要求"""
        # 从 query 提取表达要求
        expr_req = None
        source = None
        
        if self.query_data and self.query_data.get('query'):
            expr_req = self._extract_expression_requirements(self.query_data['query'])
            source = "query.jsonl"
        
        if not expr_req or not expr_req.get('has_expression_requirements'):
            style = self.metadata.get('language_style', '')
            reqs = self.metadata.get('language_requirements', [])
            if style or reqs:
                expr_req = {
                    'has_expression_requirements': True,
                    'extracted_requirements': f"Style: {style}, Requirements: {reqs}"
                }
                source = "metadata"
        
        if not expr_req or not expr_req.get('has_expression_requirements'):
            return 100.0, {'message': 'No expression requirements'}
        
        prompt = f"""Evaluate if the report meets the expression style requirements.

Requirements: {expr_req.get('extracted_requirements', '')}

Report:
{self.result_text}

Respond in JSON: {{
    "style_match_score": 0-100, "requirements_met": 0-100,
    "overall_score": 0-100, "explanation": "..."
}}"""
        
        result = self.llm.call("You are a language style evaluator.", prompt)
        if not result:
            return 70.0, {'fallback': True}
        
        return result.get('overall_score', 70), {'evaluation_result': result, 'source': source}
    
    def _extract_expression_requirements(self, query_text: str) -> Dict:
        """从 query 提取表达要求"""
        prompt = f"""Analyze the query for expression style requirements.

Query: {query_text}

Extract: language style, tone, format preferences, specific instructions.
Respond in JSON: {{
    "has_expression_requirements": true/false,
    "extracted_requirements": "summary of requirements"
}}"""
        
        return self.llm.call("You are an expert at analyzing text.", prompt) or {}


# =============================================================================
# 报告生成器
# =============================================================================

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, config: EvalConfig, gold_data: Dict):
        self.config = config
        self.gold_data = gold_data
    
    def generate(self, score: EvalScore, compact: bool = False) -> str:
        """生成评估报告
        
        Args:
            score: 评估分数
            compact: 是否生成紧凑格式（用于日志文件）
        """
        case_num = self.gold_data.get('number', 'Unknown')
        
        if compact:
            # 紧凑格式：一行摘要
            return (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                   f"Case {case_num}: Total={score.total_score:.1f} | "
                   f"Recall={score.information_recall:.1f} | "
                   f"Accuracy={score.factual_accuracy:.1f} | "
                   f"Quality={score.overall_quality:.1f} | "
                   f"Checklist={score.checklist_compliance:.1f} | "
                   f"Citation={score.citation_coverage:.1f} | "
                   f"Expression={score.expression_requirement:.1f}")
        
        # 完整格式
        lines = [
            "=" * 60,
            "📊 LLM 评估报告",
            "=" * 60,
            f"\n📁 案例编号: {case_num}",
            f"📅 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🤖 模型: {self.config.model_name}",
            f"\n🎯 总分: {score.total_score:.1f}/100",
            self._get_grade(score.total_score),
            "\n📈 各维度得分:",
            f"  1. 信息召回 (25%): {score.information_recall:.1f}/100",
            f"  2. 事实准确 (25%): {score.factual_accuracy:.1f}/100",
            f"  3. 整体质量 (15%): {score.overall_quality:.1f}/100",
            f"  4. Checklist符合 (15%): {score.checklist_compliance:.1f}/100",
            f"  5. 引用覆盖 (10%): {score.citation_coverage:.1f}/100",
            f"  6. 表达要求 (10%): {score.expression_requirement:.1f}/100",
            "\n" + "=" * 60
        ]
        return "\n".join(lines)
    
    def _get_grade(self, score: float) -> str:
        """获取等级"""
        if score >= 90: return "⭐ 等级: 优秀 (A)"
        if score >= 80: return "✨ 等级: 良好 (B)"
        if score >= 70: return "👍 等级: 合格 (C)"
        if score >= 60: return "📌 等级: 及格 (D)"
        return "❌ 等级: 不及格 (F)"
    
    def to_json_record(self, score: EvalScore) -> Dict:
        """生成 JSON 记录（用于追加到 JSONL 文件）"""
        return {
            'case_number': self.gold_data.get('number', 'Unknown'),
            'evaluation_time': datetime.now().isoformat(),
            'model_used': self.config.model_name,
            'scores': {
                'total': round(score.total_score, 2),
                'information_recall': round(score.information_recall, 2),
                'factual_accuracy': round(score.factual_accuracy, 2),
                'overall_quality': round(score.overall_quality, 2),
                'checklist_compliance': round(score.checklist_compliance, 2),
                'citation_coverage': round(score.citation_coverage, 2),
                'expression_requirement': round(score.expression_requirement, 2)
            },
            'details': score.details
        }


# =============================================================================
# 日志管理器 - 统一管理评测结果输出
# =============================================================================

class EvalLogger:
    """评测日志管理器 - 所有结果写入统一的日志文件"""
    
    DEFAULT_LOG_DIR = Path("evaluation_logs")
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or self.DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        self.summary_log = self.log_dir / "evaluation_summary.log"
        self.details_jsonl = self.log_dir / "evaluation_details.jsonl"
    
    def log_result(self, generator: ReportGenerator, score: EvalScore) -> None:
        """记录评测结果到统一日志文件"""
        # 1. 追加紧凑摘要到 .log 文件
        compact_line = generator.generate(score, compact=True)
        with open(self.summary_log, 'a', encoding='utf-8') as f:
            f.write(compact_line + '\n')
        
        # 2. 追加详细 JSON 到 .jsonl 文件
        json_record = generator.to_json_record(score)
        with open(self.details_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(json_record, ensure_ascii=False) + '\n')
        
        print(f"📝 结果已追加到: {self.summary_log}")
    
    def get_summary(self) -> str:
        """获取所有评测结果的汇总统计"""
        if not self.details_jsonl.exists():
            return "暂无评测记录"
        
        records = []
        for line in self.details_jsonl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                records.append(json.loads(line))
        
        if not records:
            return "暂无评测记录"
        
        # 计算统计信息
        total_scores = [r['scores']['total'] for r in records]
        avg_score = sum(total_scores) / len(total_scores)
        
        lines = [
            "=" * 60,
            f"📊 评测汇总统计 (共 {len(records)} 个案例)",
            "=" * 60,
            f"平均总分: {avg_score:.1f}/100",
            f"最高分: {max(total_scores):.1f}",
            f"最低分: {min(total_scores):.1f}",
            "",
            "各维度平均分:",
            f"  信息召回: {sum(r['scores']['information_recall'] for r in records) / len(records):.1f}",
            f"  事实准确: {sum(r['scores']['factual_accuracy'] for r in records) / len(records):.1f}",
            f"  整体质量: {sum(r['scores']['overall_quality'] for r in records) / len(records):.1f}",
            f"  Checklist符合: {sum(r['scores'].get('checklist_compliance', r['scores'].get('format_compliance', 0)) for r in records) / len(records):.1f}",
            f"  引用覆盖: {sum(r['scores'].get('citation_coverage', 100) for r in records) / len(records):.1f}",
            f"  表达要求: {sum(r['scores']['expression_requirement'] for r in records) / len(records):.1f}",
            "=" * 60
        ]
        return "\n".join(lines)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用 LLM 评估文本生成质量")
    parser.add_argument("--result", type=str, required=True, help="结果 markdown 文件路径")
    parser.add_argument("--gold", type=str, required=True, help="gold standard JSON 文件路径")
    parser.add_argument("--gold-source", type=str, help="源文档 gold insights JSON 路径")
    parser.add_argument("--metadata", type=str, help="metadata JSON 文件路径")
    parser.add_argument("--source-folder", type=str, help="源文档文件夹路径")
    parser.add_argument("--long-context", type=str, required=True, help="long context JSON 文件路径")
    parser.add_argument("--query-file", type=str, help="query.jsonl 文件路径")
    parser.add_argument("--log-dir", type=str, help="日志输出目录（默认: evaluation_logs）")
    parser.add_argument("--show-summary", action="store_true", help="显示所有评测结果的汇总统计")
    
    args = parser.parse_args()
    
    # 初始化日志管理器
    log_dir = Path(args.log_dir) if args.log_dir else None
    logger = EvalLogger(log_dir)
    
    # 如果只是查看汇总
    if args.show_summary:
        print(logger.get_summary())
        return
    
    # 初始化配置
    config = EvalConfig()
    
    # 加载文档
    doc_loader = DocumentLoader()
    doc_loader.load_long_context(Path(args.long_context))
    if args.source_folder:
        doc_loader.load_source_folder(Path(args.source_folder))
    
    # 加载数据
    result_text = Path(args.result).read_text(encoding='utf-8')
    gold_data = doc_loader.load_json(Path(args.gold))
    source_gold_data = doc_loader.load_json(Path(args.gold_source)) if args.gold_source else {}
    metadata = doc_loader.load_json(Path(args.metadata)) if args.metadata else {}
    
    # 加载 query
    query_data = None
    if args.query_file:
        query_path = Path(args.query_file)
        if query_path.exists():
            case_id = metadata.get('number') or gold_data.get('number')
            if case_id:
                case_id = f"{int(case_id):03d}" if str(case_id).isdigit() else case_id
                for line in query_path.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        data = json.loads(line)
                        qid = data.get('id') or data.get('number')
                        if qid and f"{int(qid):03d}" == case_id:
                            query_data = data
                            break
    
    # 创建评估器并执行评估
    llm = LLMClient(config)
    evaluator = Evaluator(config, doc_loader, llm, result_text, gold_data, source_gold_data, metadata, query_data)
    score = evaluator.evaluate()
    
    # 生成报告
    generator = ReportGenerator(config, gold_data)
    report = generator.generate(score)
    
    # 打印完整报告到控制台
    print("\n" + report)
    
    # 记录到统一日志文件
    logger.log_result(generator, score)


if __name__ == "__main__":
    main()
