#!/usr/bin/env python3
"""
事实准确性评估器 - 验证报告中的引用是否被源文档支持

Usage:
    python -m evaluators.factual_accuracy \
        --result examples/001/final_report.md \
        --long-context examples/001/long_context.json \
        --source-folder examples/001/source \
        --output examples/001/eval_factual_accuracy.json
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import unicodedata

from .base import BaseEvaluator, EvalConfig, EvalResult
from .llm_client import LLMClient
from .document_loader import DocumentLoader


def normalize_text(text: str) -> str:
    """标准化文本：繁简转换、全半角转换、去除多余空白"""
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
    
    # 去除多余空白
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result.lower()


def fuzzy_match(needle: str, haystack: str, threshold: float = 0.8) -> bool:
    """模糊匹配：检查 needle 是否在 haystack 中（支持部分匹配）"""
    needle_norm = normalize_text(needle)
    haystack_norm = normalize_text(haystack)
    
    # 精确匹配
    if needle_norm in haystack_norm:
        return True
    
    # 分词匹配：检查关键词是否大部分出现
    words = needle_norm.split()
    if len(words) > 2:
        matched = sum(1 for w in words if w in haystack_norm)
        if matched / len(words) >= threshold:
            return True
    
    return False


class FactualAccuracyEvaluator(BaseEvaluator):
    """事实准确性评估器"""
    
    metric_name = "factual_accuracy"
    weight = 1.0  # 满分 100 分
    
    def __init__(self, config: Optional[EvalConfig] = None):
        super().__init__(config)
        self.llm = LLMClient(self.config)
        self.doc_loader = DocumentLoader()
    
    def evaluate(self, result_text: str,
                 long_context_path: Optional[Path] = None,
                 source_folder: Optional[Path] = None,
                 **kwargs) -> EvalResult:
        """评估事实准确性
        
        Args:
            result_text: 待评估的报告文本
            long_context_path: long-context JSON 文件路径
            source_folder: 源文档文件夹路径
            
        Returns:
            EvalResult 包含分数和详细信息
        """
        # 加载文档
        if long_context_path:
            self.doc_loader.load_long_context(long_context_path)
        if source_folder:
            self.doc_loader.load_source_folder(source_folder)
        
        # 提取引用及其上下文
        citations_with_context = self._extract_citations_with_context(result_text)
        
        if not citations_with_context:
            return self._basic_factual_check(result_text)
        
        # 准备验证数据
        verification_items = []
        total_citations = 0
        
        for item in citations_with_context:
            context = item['context']
            for ref in item['citation_refs']:
                total_citations += 1
                source_content, source_name = self._get_source_content(
                    ref['source_file'], ref['source_type']
                )
                verification_items.append({
                    'citation': ref['citation'],
                    'context': context,
                    'source_content': source_content,
                    'source_name': source_name
                })
        
        if not verification_items:
            return self._basic_factual_check(result_text)
        
        # 分批验证
        batch_size = 10
        all_verifications = []
        all_supported_count = 0
        all_not_found_count = 0
        all_unsupported_count = 0
        
        total_batches = (len(verification_items) + batch_size - 1) // batch_size
        
        print(f"Processing {len(verification_items)} citations in {total_batches} batches...")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(verification_items))
            batch_items = verification_items[start_idx:end_idx]
            
            print(f"  Processing batch {batch_idx + 1}/{total_batches}...")
            
            batch_result = self._verify_citations_batch(batch_items)
            
            all_verifications.extend(batch_result.get('verifications', []))
            all_supported_count += batch_result.get('supported_count', 0)
            all_not_found_count += batch_result.get('not_found_count', 0)
            all_unsupported_count += batch_result.get('unsupported_count', 0)
        
        # 计算准确率
        accuracy_score = (all_supported_count / total_citations * 100) if total_citations > 0 else 0
        
        details = {
            'verification_result': {
                'verifications': all_verifications,
                'total_citations': total_citations,
                'supported_count': all_supported_count,
                'not_found_count': all_not_found_count,
                'unsupported_count': all_unsupported_count,
                'accuracy_score': accuracy_score
            },
            'source_documents_checked': len(self.doc_loader.source_documents),
            'long_context_checked': len(self.doc_loader.long_context),
            'batches_processed': total_batches
        }
        
        return EvalResult(
            metric_name=self.metric_name,
            score=accuracy_score,
            details=details,
            weight=self.weight
        )
    
    def _extract_citations_with_context(self, result_text: str) -> List[Dict]:
        """提取引用及其前面的句子上下文"""
        citations_with_context = []
        
        # 匹配引用格式
        citation_pattern = r'\[(?:Doc:\s*[^\]]+|文档:\s*[^\]]+|RAG-\d+|long_context:\s*"[^"]+",\s*chunk\s*[\d,\s]+)\]'
        
        citation_positions = []
        for match in re.finditer(citation_pattern, result_text):
            citation_positions.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group(0)
            })
        
        if not citation_positions:
            return citations_with_context
        
        # 分组连续引用
        citation_groups = []
        current_group = []
        
        for i, pos in enumerate(citation_positions):
            if not current_group:
                current_group.append(pos)
            else:
                prev_end = current_group[-1]['end']
                gap = result_text[prev_end:pos['start']]
                if re.match(r'^[\s\[\]]*$', gap):
                    current_group.append(pos)
                else:
                    citation_groups.append(current_group)
                    current_group = [pos]
        
        if current_group:
            citation_groups.append(current_group)
        
        # 提取上下文
        for group in citation_groups:
            first_citation_start = group[0]['start']
            text_before = result_text[:first_citation_start]
            
            sentences = re.split(r'([.!?]\s+)', text_before)
            if len(sentences) >= 2:
                context = ''.join(sentences[-2:]).strip()
            else:
                context = text_before[-200:].strip()
            
            context = re.sub(r'\s+', ' ', context)
            
            citation_refs = []
            for pos in group:
                citation_text = pos['text']
                ref = self._parse_citation(citation_text)
                if ref:
                    citation_refs.append(ref)
            
            if context and citation_refs:
                citations_with_context.append({
                    'context': context,
                    'citation_refs': citation_refs
                })
        
        return citations_with_context
    
    def _parse_citation(self, citation_text: str) -> Optional[Dict]:
        """解析单个引用"""
        if citation_text.startswith('[Doc:') or citation_text.startswith('[文档:'):
            doc_match = re.search(r'\[(?:Doc|文档):\s*([^\]]+)\]', citation_text)
            if doc_match:
                return {
                    'citation': citation_text,
                    'source_file': doc_match.group(1).strip(),
                    'source_type': 'doc'
                }
        elif citation_text.startswith('[RAG-'):
            rag_match = re.search(r'\[RAG-(\d+)\]', citation_text)
            if rag_match:
                return {
                    'citation': citation_text,
                    'source_file': f"RAG-{rag_match.group(1)}",
                    'source_type': 'rag'
                }
        elif citation_text.startswith('[long_context:'):
            long_ctx_match = re.search(r'\[long_context:\s*"([^"]+)",\s*chunk\s*([\d,\s]+)\]', citation_text)
            if long_ctx_match:
                return {
                    'citation': citation_text,
                    'source_file': long_ctx_match.group(1).strip(),
                    'source_type': 'long_context'
                }
        return None
    
    def _get_source_content(self, source_file: str, source_type: str) -> Tuple[Optional[str], Optional[str]]:
        """获取源内容"""
        if source_type == 'doc':
            content = self.doc_loader.get_source_document(source_file)
            if content:
                return content, source_file
            # 尝试从 long-context 查找
            content, name = self.doc_loader.get_content_by_title(source_file)
            return content, name
        
        elif source_type == 'rag':
            try:
                rag_num = int(re.search(r'RAG-(\d+)', source_file).group(1))
                idx = rag_num - 1
                return self.doc_loader.get_content_by_index(idx)
            except:
                return None, None
        
        elif source_type == 'long_context':
            return self.doc_loader.get_content_by_title(source_file)
        
        return None, None
    
    def _verify_citations_batch(self, verification_items: List[Dict]) -> Dict:
        """验证一批引用 - 逐个引用单独验证"""
        all_verifications = []
        supported_count = 0
        not_found_count = 0
        unsupported_count = 0
        total = len(verification_items)
        
        for i, item in enumerate(verification_items):
            citation = item['citation']
            context = item['context']
            source_name = item['source_name'] or 'NOT FOUND'
            source_content = item['source_content']
            
            print(f"    [{i+1}/{total}] Verifying citation: {citation}")
            
            if not source_content:
                all_verifications.append({
                    'citation': citation,
                    'context': context[:100] + '...' if len(context) > 100 else context,
                    'supported': False,
                    'source_found': False,
                    'explanation': 'Source content not found'
                })
                not_found_count += 1
                continue
            
            # 如果源内容太长，分段验证
            if len(source_content) > self.config.max_segment_length:
                result = self._verify_with_segments(citation, context, source_name, source_content)
            else:
                result = self._verify_single(citation, context, source_name, source_content)
            
            # 添加上下文信息
            result['context'] = context[:100] + '...' if len(context) > 100 else context
            all_verifications.append(result)
            
            status = "✅" if result.get('supported') else "❌"
            print(f"      {status} {result.get('explanation', '')[:80]}")
            
            if result.get('supported'):
                supported_count += 1
            else:
                unsupported_count += 1
        
        return {
            'verifications': all_verifications,
            'total_citations': len(verification_items),
            'supported_count': supported_count,
            'not_found_count': not_found_count,
            'unsupported_count': unsupported_count
        }
    
    def _verify_single(self, citation: str, context: str, 
                      source_name: str, source_content: str) -> Dict:
        """验证单个引用"""
        prompt = f"""You are a fact-checking expert. Please verify if the statement before a citation is accurately supported by the source content.

IMPORTANT: A citation is considered WRONG only if:
1. The statement CONFLICTS with the source content
2. The source content does NOT support the statement

A citation is considered CORRECT if:
- The source content supports the statement (explicitly or implicitly)
- The statement is a reasonable interpretation of the source content

Citation: {citation}
Statement before citation: {context}
Source: {source_name}
Source content:
{source_content[:self.config.max_segment_length]}

Respond in JSON format:
{{
    "supported": true/false,
    "explanation": "brief explanation"
}}
"""
        
        result = self.llm.call(
            system="You are a meticulous fact-checker.",
            user=prompt
        )
        
        if result:
            return {
                'citation': citation,
                'supported': result.get('supported', False),
                'source_found': True,
                'explanation': result.get('explanation', '')
            }
        else:
            return {
                'citation': citation,
                'supported': False,
                'source_found': True,
                'explanation': 'Verification failed'
            }
    
    def _verify_with_segments(self, citation: str, context: str,
                             source_name: str, source_content: str) -> Dict:
        """分段验证长内容"""
        segments = self._split_content(source_content)
        
        for seg_idx, segment in enumerate(segments):
            result = self._verify_single(citation, context, source_name, segment)
            if result.get('supported'):
                result['explanation'] = f"Supported by segment {seg_idx + 1}/{len(segments)}. " + result.get('explanation', '')
                return result
        
        return {
            'citation': citation,
            'supported': False,
            'source_found': True,
            'explanation': f'Checked all {len(segments)} segments, none support the statement.'
        }
    
    def _split_content(self, content: str) -> List[str]:
        """分段内容"""
        max_len = self.config.max_segment_length
        if len(content) <= max_len:
            return [content]
        
        segments = []
        paragraphs = content.split('\n\n')
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_len:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    segments.append(current)
                current = para if len(para) <= max_len else para[:max_len]
        
        if current:
            segments.append(current)
        
        return segments if segments else [content[:max_len]]
    
    def _basic_factual_check(self, result_text: str) -> EvalResult:
        """基础事实检查"""
        citations = re.findall(r'\[[^\]]+\]', result_text)
        has_citations = len(citations) > 0
        score = 70 if has_citations else 50
        
        return EvalResult(
            metric_name=self.metric_name,
            score=score,
            details={
                'method': 'basic_check',
                'has_citations': has_citations,
                'citation_count': len(citations)
            },
            weight=self.weight
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate factual accuracy")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--long-context", type=str, help="Path to long context JSON")
    parser.add_argument("--source-folder", type=str, help="Path to source documents folder")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
