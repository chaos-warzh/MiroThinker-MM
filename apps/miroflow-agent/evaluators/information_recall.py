#!/usr/bin/env python3
"""
信息召回率评估器 - 评估报告是否覆盖了 gold_insights 中的关键信息点

Usage:
    python -m evaluators.information_recall \
        --result examples/001/final_report.md \
        --gold examples/001/gold_insights.json \
        --output examples/001/eval_information_recall.json
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .base import BaseEvaluator, EvalConfig, EvalResult
from .llm_client import LLMClient


class InformationRecallEvaluator(BaseEvaluator):
    """信息召回率评估器"""
    
    metric_name = "information_recall"
    weight = 1.0  # 满分 100 分
    
    def __init__(self, config: Optional[EvalConfig] = None):
        super().__init__(config)
        self.llm = LLMClient(self.config)
    
    def evaluate(self, result_text: str, 
                 gold_insights: List[Dict] = None,
                 source_gold_insights: List[Dict] = None,
                 **kwargs) -> EvalResult:
        """评估信息召回率
        
        Args:
            result_text: 待评估的报告文本
            gold_insights: 从 long-context 提取的 gold insights
            source_gold_insights: 从 source documents 提取的 gold insights
            
        Returns:
            EvalResult 包含分数和详细信息
        """
        gold_insights = gold_insights or []
        source_gold_insights = source_gold_insights or []
        
        # 分别评估两个来源
        long_score, long_details, has_long = self._evaluate_component(
            result_text, gold_insights, 'long_context'
        )
        source_score, source_details, has_source = self._evaluate_component(
            result_text, source_gold_insights, 'source_documents'
        )
        
        # 分开两部分分数，不加权平均
        details = {
            'long_context_score': long_score if has_long else None,
            'source_documents_score': source_score if has_source else None,
            'components': {
                'long_context': {
                    'score': long_score,
                    'details': long_details,
                    'available': has_long
                },
                'source_documents': {
                    'score': source_score,
                    'details': source_details,
                    'available': has_source
                }
            },
            'note': 'Scores are reported separately for long_context and source_documents'
        }
        
        # 计算综合分数（用于排序，但两个分数都会单独展示）
        if has_long and has_source:
            combined_score = (long_score + source_score) / 2
        elif has_long:
            combined_score = long_score
        elif has_source:
            combined_score = source_score
        else:
            combined_score = 100.0
        
        return EvalResult(
            metric_name=self.metric_name,
            score=combined_score,
            details=details,
            weight=self.weight
        )
    
    def _evaluate_component(self, result_text: str, 
                           gold_insights: List[Dict], 
                           label: str) -> Tuple[float, Dict, bool]:
        """评估单个来源的召回率 - 逐个 insight 单独验证"""
        if not gold_insights:
            return 100.0, {
                'message': f'No gold insights provided for {label}',
                'evaluation_result': {
                    'covered_insights': [],
                    'total_insights': 0,
                    'covered_count': 0,
                    'recall_percentage': 100.0
                }
            }, False
        
        # 逐个验证每个 insight
        covered_insights = []
        total = len(gold_insights)
        
        print(f"  Evaluating {total} insights for {label}...")
        
        for i, insight in enumerate(gold_insights):
            insight_text = insight['insight']
            print(f"    [{i+1}/{total}] Checking insight: {insight_text[:50]}...")
            
            result = self._verify_single_insight(result_text, insight_text, i + 1)
            covered_insights.append(result)
        
        # 统计结果
        covered = sum(1 for item in covered_insights if item.get('covered') is True)
        recall_percentage = 100.0 * covered / total if total > 0 else 100.0
        
        evaluation_result = {
            'covered_insights': covered_insights,
            'total_insights': total,
            'covered_count': covered,
            'recall_percentage': recall_percentage
        }
        
        details = {
            'evaluation_result': evaluation_result,
            'recall_rate': recall_percentage / 100,
            'covered_count': covered,
            'total_count': total,
            'original_gold_count': len(gold_insights),
            'source_label': label
        }
        
        return recall_percentage, details, True
    
    def _verify_single_insight(self, result_text: str, insight_text: str, insight_id: int) -> Dict:
        """验证单个 insight 是否被报告覆盖
        
        Args:
            result_text: 报告文本
            insight_text: 要验证的 insight 文本
            insight_id: insight 的编号
            
        Returns:
            验证结果字典
        """
        prompt = f"""You are an expert evaluator. Please determine if the following report covers the required insight.

Required Insight:
{insight_text}

Report to Evaluate:
{result_text}

Your evaluation should be GENEROUS and focus on whether the factual content of the report is supported, not on matching every subjective planning detail.

Determine whether the report adequately covers the factual core of this insight.
If the report mentions closely related facts or a more general version of the same idea, you should usually consider it COVERED.

Respond in JSON format:
{{
    "covered": true/false,
    "explanation": "brief explanation of why it is covered or not covered"
}}
"""
        
        result = self.llm.call(
            system="You are a precise text analysis expert.",
            user=prompt
        )
        
        if not result:
            # 降级到简单关键词匹配
            is_covered = insight_text.lower() in result_text.lower()
            return {
                'id': insight_id,
                'covered': is_covered,
                'explanation': 'Fallback: keyword matching',
                'insight_text': insight_text[:100] + '...' if len(insight_text) > 100 else insight_text
            }
        
        return {
            'id': insight_id,
            'covered': result.get('covered', False),
            'explanation': result.get('explanation', ''),
            'insight_text': insight_text[:100] + '...' if len(insight_text) > 100 else insight_text
        }
    
    def _fallback_recall(self, result_text: str, 
                        gold_insights: List[Dict], 
                        label: str) -> Tuple[float, Dict, bool]:
        """降级的召回率评估"""
        found_count = 0
        for insight in gold_insights:
            if insight['insight'].lower() in result_text.lower():
                found_count += 1
        
        recall_rate = found_count / len(gold_insights) if gold_insights else 1.0
        return recall_rate * 100, {
            'fallback_method': True, 
            'recall_rate': recall_rate,
            'source_label': label
        }, True


def main():
    parser = argparse.ArgumentParser(description="Evaluate information recall")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--gold", type=str, help="Path to gold insights JSON (from long context)")
    parser.add_argument("--gold-source", type=str, help="Path to source gold insights JSON")
    parser.add_argument("--insights-dir", type=str, help="Path to insights directory containing gold_insights_from_longcontext.json and gold_insights_from_source.json")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
    
    # 加载文件
    result_text = Path(args.result).read_text(encoding='utf-8')
    
    import json
    
    # 处理 insights 路径
    gold_insights = []
    source_gold_insights = []
    
    if args.insights_dir:
        # 使用 insights-dir 自动查找两个 insight 文件
        insights_dir = Path(args.insights_dir)
        longcontext_file = insights_dir / "gold_insights_from_longcontext.json"
        source_file = insights_dir / "gold_insights_from_source.json"
        
        if longcontext_file.exists():
            gold_data = json.loads(longcontext_file.read_text(encoding='utf-8'))
            gold_insights = gold_data.get('gold_insights', [])
        if source_file.exists():
            source_data = json.loads(source_file.read_text(encoding='utf-8'))
            source_gold_insights = source_data.get('gold_insights', [])
    else:
        # 使用单独指定的路径
        if args.gold:
            gold_data = json.loads(Path(args.gold).read_text(encoding='utf-8'))
            gold_insights = gold_data.get('gold_insights', [])
        if args.gold_source:
            source_data = json.loads(Path(args.gold_source).read_text(encoding='utf-8'))
            source_gold_insights = source_data.get('gold_insights', [])
    
    # 评估
    evaluator = InformationRecallEvaluator()
    result = evaluator.evaluate(
        result_text,
        gold_insights=gold_insights,
        source_gold_insights=source_gold_insights
    )
    
    # 输出结果
    print(f"\n📋 Information Recall Score: {result.score:.1f}/100")
    print(result.to_json())
    
    if args.output:
        evaluator.save_result(result, Path(args.output))
        print(f"\n📄 Result saved to: {args.output}")


if __name__ == "__main__":
    main()
