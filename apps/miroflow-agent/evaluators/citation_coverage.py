#!/usr/bin/env python3
"""
引用覆盖率评估器 - 检查报告是否引用了必需的文档

基于 check_citation.py 的逻辑

Usage:
    python -m evaluators.citation_coverage \
        --result examples/001/final_report.md \
        --useful-search examples/001/useful_search.json \
        --output examples/001/eval_citation_coverage.json
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseEvaluator, EvalConfig, EvalResult
from .document_loader import normalize_title, fuzzy_title_match


class CitationCoverageEvaluator(BaseEvaluator):
    """引用覆盖率评估器"""
    
    metric_name = "citation_coverage"
    weight = 1.0  # 满分 100 分
    
    def evaluate(self, result_text: str,
                 useful_search_path: Optional[Path] = None,
                 required_titles: Optional[List[str]] = None,
                 **kwargs) -> EvalResult:
        """评估引用覆盖率
        
        Args:
            result_text: 待评估的报告文本
            useful_search_path: useful_search.json 文件路径
            required_titles: 必需引用的标题列表（可选，直接传入）
            
        Returns:
            EvalResult 包含分数和详细信息
        """
        # 加载必需的标题
        titles = required_titles or []
        if useful_search_path and useful_search_path.exists():
            titles = self._load_useful_search(useful_search_path)
        
        if not titles:
            return EvalResult(
                metric_name=self.metric_name,
                score=100.0,
                details={
                    'message': 'No required titles specified',
                    'cited': [],
                    'missing': [],
                    'total_required': 0,
                    'total_cited': 0
                },
                weight=self.weight
            )
        
        # 检查是否是空报告
        if "No final answer" in result_text:
            return EvalResult(
                metric_name=self.metric_name,
                score=0.0,
                details={
                    'message': 'Empty report (No final answer)',
                    'cited': [],
                    'missing': titles,
                    'total_required': len(titles),
                    'total_cited': 0
                },
                weight=self.weight
            )
        
        # 检查引用
        result = self._check_citations(result_text, titles)
        
        cited_count = len(result['cited'])
        total_count = len(titles)
        coverage_rate = (cited_count / total_count * 100) if total_count > 0 else 100.0
        
        details = {
            'cited': result['cited'],
            'missing': result['missing'],
            'total_required': total_count,
            'total_cited': cited_count,
            'coverage_rate': coverage_rate,
            'full_coverage': cited_count == total_count
        }
        
        return EvalResult(
            metric_name=self.metric_name,
            score=coverage_rate,
            details=details,
            weight=self.weight
        )
    
    def _load_useful_search(self, path: Path) -> List[str]:
        """加载 useful_search.json 或 useful_search.jsonl 中的标题列表
        
        支持两种格式：
        - JSON: 一个包含对象的数组
        - JSONL: 每行一个 JSON 对象，或者第一行是整个数组
        """
        try:
            content = path.read_text(encoding='utf-8')
            titles = []
            
            # 尝试作为 JSON 数组解析
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    # 标准 JSON 数组格式
                    for item in data:
                        if isinstance(item, dict) and item.get("title"):
                            titles.append(item["title"])
                    return titles
            except json.JSONDecodeError:
                pass
            
            # 尝试作为 JSONL 格式解析（每行一个 JSON）
            for line in content.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # 如果这一行是一个数组（整个文件就是一行 JSON 数组）
                    if isinstance(item, list):
                        for obj in item:
                            if isinstance(obj, dict) and obj.get("title"):
                                titles.append(obj["title"])
                    # 如果这一行是一个对象
                    elif isinstance(item, dict) and item.get("title"):
                        titles.append(item["title"])
                except json.JSONDecodeError:
                    continue
            
            return titles
        except Exception:
            return []
    
    def _normalize_quotes(self, s: str) -> str:
        """标准化引号：将中文引号转换为英文引号"""
        # 中文双引号
        s = s.replace('"', '"').replace('"', '"')
        # 中文单引号
        s = s.replace(''', "'").replace(''', "'")
        return s
    
    def _check_citations(self, content: str, titles: List[str]) -> Dict:
        """检查报告中是否引用了指定的标题（支持模糊匹配）"""
        cited = []
        missing = []
        
        # 标准化报告内容
        content_normalized = normalize_title(content)
        
        for title in titles:
            found = False
            
            # 提取标题的主要部分（去掉网站后缀）
            main_title = title.split("-")[0].strip() if "-" in title else title
            
            # 1. 精确匹配（标准化后）
            title_norm = normalize_title(title)
            main_title_norm = normalize_title(main_title)
            
            if title_norm in content_normalized or main_title_norm in content_normalized:
                found = True
            
            # 2. 模糊匹配：检查报告中的每一行
            if not found:
                for line in content.split('\n'):
                    if fuzzy_title_match(main_title, line):
                        found = True
                        break
            
            # 3. 关键词匹配：提取标题中的关键词
            if not found:
                # 去除常见后缀和网站名
                clean_title = re.sub(r'\s*[-|–]\s*[^-|–]+$', '', title).strip()
                clean_title_norm = normalize_title(clean_title)
                if clean_title_norm and len(clean_title_norm) > 5:
                    if clean_title_norm in content_normalized:
                        found = True
            
            if found:
                cited.append(title)
            else:
                missing.append(title)
        
        return {"cited": cited, "missing": missing}


def main():
    parser = argparse.ArgumentParser(description="Evaluate citation coverage")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--useful-search", type=str, help="Path to useful_search.json")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
    
    # 加载文件
    result_text = Path(args.result).read_text(encoding='utf-8')
    
    # 评估
    evaluator = CitationCoverageEvaluator()
    result = evaluator.evaluate(
        result_text,
        useful_search_path=Path(args.useful_search) if args.useful_search else None
    )
    
    # 输出结果
    print(f"\n🔗 Citation Coverage Score: {result.score:.1f}/100")
    print(result.to_json())
    
    if args.output:
        evaluator.save_result(result, Path(args.output))
        print(f"\n📄 Result saved to: {args.output}")


if __name__ == "__main__":
    main()
