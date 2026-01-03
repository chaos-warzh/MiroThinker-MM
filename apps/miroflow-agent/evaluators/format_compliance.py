#!/usr/bin/env python3
"""
格式符合性评估器 - 基于 Checklist 评估报告是否满足要求

Usage:
    python -m evaluators.format_compliance \
        --result examples/001/final_report.md \
        --checklist checklists/batch2_checklists/001/checklist.json \
        --output examples/001/eval_format_compliance.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseEvaluator, EvalConfig, EvalResult
from .llm_client import LLMClient


class FormatComplianceEvaluator(BaseEvaluator):
    """格式符合性评估器 - 基于 Checklist"""
    
    metric_name = "format_compliance"
    weight = 1.0  # 满分 100 分
    
    def __init__(self, config: Optional[EvalConfig] = None):
        super().__init__(config)
        self.llm = LLMClient(self.config)
    
    def evaluate(self, result_text: str,
                 checklist_path: Optional[Path] = None,
                 checklist_items: Optional[List[Dict]] = None,
                 **kwargs) -> EvalResult:
        """评估格式符合性
        
        Args:
            result_text: 待评估的报告文本
            checklist_path: checklist.json 文件路径
            checklist_items: 直接传入的 checklist 项目列表
            
        Returns:
            EvalResult 包含分数和详细信息
        """
        # 加载 checklist
        items = checklist_items or []
        if checklist_path and checklist_path.exists():
            items = self._load_checklist(checklist_path)
        
        if not items:
            return EvalResult(
                metric_name=self.metric_name,
                score=100.0,
                details={
                    'message': 'No checklist provided',
                    'checklist_items': [],
                    'satisfied_count': 0,
                    'total_count': 0
                },
                weight=self.weight
            )
        
        # 使用 LLM 评估每个 checklist 项目
        evaluation_results = self._evaluate_checklist(result_text, items)
        
        # 计算分数
        satisfied_count = sum(1 for r in evaluation_results if r.get('satisfied'))
        total_count = len(evaluation_results)
        score = (satisfied_count / total_count * 100) if total_count > 0 else 100.0
        
        # 按类别统计
        category_stats = {}
        for r in evaluation_results:
            cat = r.get('category', 'content')
            if cat not in category_stats:
                category_stats[cat] = {'satisfied': 0, 'total': 0}
            category_stats[cat]['total'] += 1
            if r.get('satisfied'):
                category_stats[cat]['satisfied'] += 1
        
        details = {
            'checklist_evaluation': evaluation_results,
            'satisfied_count': satisfied_count,
            'total_count': total_count,
            'satisfaction_rate': score,
            'category_stats': category_stats,
            'unsatisfied_items': [
                r for r in evaluation_results if not r.get('satisfied')
            ]
        }
        
        return EvalResult(
            metric_name=self.metric_name,
            score=score,
            details=details,
            weight=self.weight
        )
    
    def _load_checklist(self, path: Path) -> List[Dict]:
        """加载 checklist 文件"""
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            return data.get('checklist', [])
        except Exception:
            return []
    
    def _evaluate_checklist(self, result_text: str, items: List[Dict]) -> List[Dict]:
        """使用 LLM 逐个评估 checklist 项目"""
        results = []
        total = len(items)
        
        print(f"  Evaluating {total} checklist items...")
        
        for i, item in enumerate(items):
            item_id = item.get('id', i+1)
            requirement = item.get('requirement', '')
            category = item.get('category', 'content')
            
            print(f"    [{i+1}/{total}] Checking: {requirement[:50]}...")
            
            eval_result = self._verify_single_item(result_text, item_id, requirement, category)
            results.append(eval_result)
            
            status = "✅" if eval_result.get('satisfied') else "❌"
            print(f"      {status} {eval_result.get('explanation', '')[:80]}")
        
        return results
    
    def _verify_single_item(self, result_text: str, item_id: int, 
                           requirement: str, category: str) -> Dict:
        """验证单个 checklist 项目
        
        Args:
            result_text: 报告文本
            item_id: 项目 ID
            requirement: 要求描述
            category: 类别
            
        Returns:
            验证结果字典
        """
        prompt = f"""You are an expert evaluator. Please check if the following report satisfies this specific requirement.

Requirement:
[{category}] {requirement}

Report to Evaluate:
{result_text}

Determine whether the report satisfies this requirement.
Be fair but thorough - a requirement is satisfied if the report reasonably addresses it.

Respond in JSON format:
{{
    "satisfied": true/false,
    "explanation": "brief explanation of why satisfied or not"
}}
"""
        
        result = self.llm.call(
            system="You are a meticulous requirements checker.",
            user=prompt
        )
        
        if not result:
            return {
                'id': item_id,
                'requirement': requirement,
                'category': category,
                'satisfied': False,
                'explanation': 'Evaluation failed'
            }
        
        return {
            'id': item_id,
            'requirement': requirement,
            'category': category,
            'satisfied': result.get('satisfied', False),
            'explanation': result.get('explanation', '')
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate format compliance using checklist")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--checklist", type=str, help="Path to checklist.json file")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
    
    # 加载文件
    result_text = Path(args.result).read_text(encoding='utf-8')
    
    # 评估
    evaluator = FormatComplianceEvaluator()
    result = evaluator.evaluate(
        result_text,
        checklist_path=Path(args.checklist) if args.checklist else None
    )
    
    # 输出结果
    print(f"\n📋 Format Compliance Score: {result.score:.1f}/100")
    print(result.to_json())
    
    if args.output:
        evaluator.save_result(result, Path(args.output))
        print(f"\n📄 Result saved to: {args.output}")


if __name__ == "__main__":
    main()
