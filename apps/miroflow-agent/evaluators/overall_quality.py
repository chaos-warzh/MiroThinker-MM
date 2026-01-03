#!/usr/bin/env python3
"""
整体质量评估器 - 评估报告的写作质量

Usage:
    python -m evaluators.overall_quality \
        --result examples/001/final_report.md \
        --output examples/001/eval_overall_quality.json
"""

import argparse
from pathlib import Path
from typing import Optional

from .base import BaseEvaluator, EvalConfig, EvalResult
from .llm_client import LLMClient


class OverallQualityEvaluator(BaseEvaluator):
    """整体质量评估器"""
    
    metric_name = "overall_quality"
    weight = 1.0  # 满分 100 分
    
    def __init__(self, config: Optional[EvalConfig] = None):
        super().__init__(config)
        self.llm = LLMClient(self.config)
    
    def evaluate(self, result_text: str, **kwargs) -> EvalResult:
        """评估整体质量
        
        Args:
            result_text: 待评估的报告文本
            
        Returns:
            EvalResult 包含分数和详细信息
        """
        prompt = f"""You are an expert writing evaluator. Please assess the overall quality of this report based on TWO key dimensions.

Report:
{result_text}

Evaluation Criteria:

1. **Clarity (清晰度)** - 50% weight
   - Is the writing clear and easy to understand?
   - Is the report well-organized with logical flow?
   - Are ideas well-connected and transitions smooth?
   - Is the language precise and unambiguous?

2. **Insight (洞察力)** - 50% weight
   - Does the report provide meaningful insights and analysis?
   - Does it go beyond surface-level information?
   - Are the conclusions well-supported and valuable?
   - Does it demonstrate deep understanding of the topic?

Respond in JSON format:
{{
    "clarity_score": number (0-100),
    "clarity_explanation": "explanation for clarity score",
    "insight_score": number (0-100),
    "insight_explanation": "explanation for insight score",
    "overall_score": number (0-100),
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"]
}}
"""
        
        result = self.llm.call(
            system="You are an expert writing quality assessor focusing on clarity and insight.",
            user=prompt
        )
        
        if result:
            # 计算加权平均分 (各占 50%)
            clarity_score = result.get('clarity_score', 70)
            insight_score = result.get('insight_score', 70)
            overall_score = result.get('overall_score', (clarity_score + insight_score) / 2)
            
            details = {
                'quality_assessment': result,
                'sub_scores': {
                    'clarity': clarity_score,
                    'insight': insight_score
                },
                'clarity_explanation': result.get('clarity_explanation', ''),
                'insight_explanation': result.get('insight_explanation', ''),
                'strengths': result.get('strengths', []),
                'weaknesses': result.get('weaknesses', [])
            }
            
            return EvalResult(
                metric_name=self.metric_name,
                score=overall_score,
                details=details,
                weight=self.weight
            )
        else:
            # 降级到基础检查
            return self._basic_quality_check(result_text)
    
    def _basic_quality_check(self, result_text: str) -> EvalResult:
        """基础质量检查"""
        word_count = len(result_text.split())
        paragraph_count = len([p for p in result_text.split('\n\n') if p.strip()])
        sentence_count = max(1, result_text.count('.') + result_text.count('。'))
        avg_sentence_length = word_count / sentence_count
        
        # 基础评分
        score = 70
        if 15 <= avg_sentence_length <= 25:
            score += 10
        if 3 <= paragraph_count <= 8:
            score += 10
        if word_count >= 300:
            score += 10
        
        return EvalResult(
            metric_name=self.metric_name,
            score=min(100, score),
            details={
                'method': 'basic_check',
                'word_count': word_count,
                'paragraph_count': paragraph_count,
                'avg_sentence_length': avg_sentence_length
            },
            weight=self.weight
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate overall quality")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
    
    # 加载文件
    result_text = Path(args.result).read_text(encoding='utf-8')
    
    # 评估
    evaluator = OverallQualityEvaluator()
    result = evaluator.evaluate(result_text)
    
    # 输出结果
    print(f"\n📝 Overall Quality Score: {result.score:.1f}/100")
    print(result.to_json())
    
    if args.output:
        evaluator.save_result(result, Path(args.output))
        print(f"\n📄 Result saved to: {args.output}")


if __name__ == "__main__":
    main()
