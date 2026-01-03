#!/usr/bin/env python3
"""
工具使用效率评估器 - 评估 Agent 的工具调用效率

基于 evaluate_with_llm.py 中的 ToolUsageMetrics 逻辑

Usage:
    python -m evaluators.tool_usage \
        --execution-log examples/001/execution_log.json \
        --gold examples/001/gold_insights.json \
        --output examples/001/eval_tool_usage.json
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .base import BaseEvaluator, EvalConfig, EvalResult


@dataclass
class ToolUsageMetrics:
    """工具使用指标"""
    total_tool_calls: int = 0
    unique_tools_used: int = 0
    tool_call_breakdown: Dict[str, int] = None
    total_duration_seconds: float = 0.0
    main_agent_turns: int = 0
    sub_agent_turns: Dict[str, int] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    task_completed: bool = False
    has_final_answer: bool = False
    rag_tool_calls: int = 0
    rag_sub_agent_turns: int = 0
    info_density: float = 0.0
    rag_call_ratio: float = 0.0
    calls_per_insight: float = 0.0
    rag_search_depth: float = 0.0
    covered_insights: int = 0
    total_insights: int = 0
    
    def __post_init__(self):
        if self.tool_call_breakdown is None:
            self.tool_call_breakdown = {}
        if self.sub_agent_turns is None:
            self.sub_agent_turns = {}
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ToolMetricsExtractor:
    """工具指标提取器"""
    
    def __init__(self, execution_log_path: Optional[Path] = None):
        self.execution_log_path = execution_log_path
        self.step_logs: List[Dict] = []
    
    def load(self) -> bool:
        """加载执行日志"""
        if not self.execution_log_path or not self.execution_log_path.exists():
            return False
        try:
            data = json.loads(self.execution_log_path.read_text(encoding='utf-8'))
            self.step_logs = data.get('step_logs', [])
            return True
        except Exception as e:
            print(f"  ⚠️ 加载 execution_log.json 失败: {e}")
            return False
    
    def extract(self) -> Optional[ToolUsageMetrics]:
        """提取工具使用指标"""
        if not self.step_logs:
            return None
        
        # 提取工具调用
        tool_calls = [log for log in self.step_logs if 'Tool Call Start' in log.get('step_name', '')]
        tool_breakdown = {}
        for call in tool_calls:
            msg = call.get('message', '')
            match = re.search(r"to call tool '([^']+)'", msg)
            if match:
                tool_name = match.group(1)
                tool_breakdown[tool_name] = tool_breakdown.get(tool_name, 0) + 1
        
        # 提取统计信息
        stats = self._extract_statistics()
        tokens = self._extract_token_usage()
        
        # 检查任务完成状态
        task_completed = any('task_execution_finished' in log.get('step_name', '') for log in self.step_logs)
        has_final_answer = any('Final Answer' in log.get('step_name', '') for log in self.step_logs)
        
        # RAG 相关指标
        rag_tool_calls = sum(count for tool, count in tool_breakdown.items() 
                            if 'rag' in tool.lower() or 'search' in tool.lower())
        sub_agent_turns = stats.get('sub_agent_turns', {})
        rag_sub_agent_turns = sum(turns for agent, turns in sub_agent_turns.items() if 'rag' in agent.lower())
        
        return ToolUsageMetrics(
            total_tool_calls=len(tool_calls),
            unique_tools_used=len(tool_breakdown),
            tool_call_breakdown=tool_breakdown,
            total_duration_seconds=stats.get('duration', 0.0),
            main_agent_turns=stats.get('main_turns', 0),
            sub_agent_turns=sub_agent_turns,
            total_input_tokens=tokens.get('input', 0),
            total_output_tokens=tokens.get('output', 0),
            task_completed=task_completed,
            has_final_answer=has_final_answer,
            rag_tool_calls=rag_tool_calls,
            rag_sub_agent_turns=rag_sub_agent_turns
        )
    
    def _extract_statistics(self) -> Dict[str, Any]:
        """提取统计信息"""
        for log in self.step_logs:
            if 'Main Agent | Statistics' in log.get('step_name', ''):
                message = log.get('message', '')
                result = {}
                duration_match = re.search(r'Total Duration:\s*([\d.]+)\s*seconds', message)
                if duration_match:
                    result['duration'] = float(duration_match.group(1))
                turns_match = re.search(r'Main Agent Turns:\s*(\d+)', message)
                if turns_match:
                    result['main_turns'] = int(turns_match.group(1))
                sub_agent_turns = {}
                for match in re.finditer(r'-\s*([^:]+):\s*(\d+)\s*turns', message):
                    sub_agent_turns[match.group(1).strip()] = int(match.group(2))
                result['sub_agent_turns'] = sub_agent_turns
                return result
        return {}
    
    def _extract_token_usage(self) -> Dict[str, int]:
        """提取 token 使用量"""
        token_logs = [log for log in self.step_logs if 'Token Usage' in log.get('step_name', '')]
        if not token_logs:
            return {'input': 0, 'output': 0}
        last_log = token_logs[-1]
        message = last_log.get('message', '')
        input_match = re.search(r'Input:\s*(\d+)', message)
        output_match = re.search(r'Output:\s*(\d+)', message)
        return {
            'input': int(input_match.group(1)) if input_match else 0,
            'output': int(output_match.group(1)) if output_match else 0
        }


class ToolUsageEvaluator(BaseEvaluator):
    """工具使用效率评估器"""
    
    metric_name = "tool_usage"
    weight = 1.0  # 满分 100 分
    
    def evaluate(self, result_text: str = "",
                 execution_log_path: Optional[Path] = None,
                 gold_insights: Optional[List[Dict]] = None,
                 covered_insights: int = 0,
                 **kwargs) -> EvalResult:
        """评估工具使用效率
        
        Args:
            result_text: 报告文本（可选）
            execution_log_path: execution_log.json 文件路径
            gold_insights: gold insights 列表
            covered_insights: 已覆盖的 insights 数量
            
        Returns:
            EvalResult 包含分数和详细信息
        """
        # 提取工具指标
        extractor = ToolMetricsExtractor(execution_log_path)
        if not extractor.load():
            return EvalResult(
                metric_name=self.metric_name,
                score=50.0,
                details={
                    'message': 'No execution log available',
                    'metrics': None
                },
                weight=self.weight
            )
        
        metrics = extractor.extract()
        if not metrics:
            return EvalResult(
                metric_name=self.metric_name,
                score=50.0,
                details={
                    'message': 'Failed to extract metrics',
                    'metrics': None
                },
                weight=self.weight
            )
        
        # 计算高级指标
        total_insights = len(gold_insights) if gold_insights else 0
        metrics.total_insights = total_insights
        metrics.covered_insights = covered_insights
        
        # 信息密度 = 覆盖的 insights / 总工具调用数
        if metrics.total_tool_calls > 0:
            metrics.info_density = covered_insights / metrics.total_tool_calls
        
        # RAG 调用比例
        if metrics.total_tool_calls > 0:
            metrics.rag_call_ratio = metrics.rag_tool_calls / metrics.total_tool_calls
        
        # 每个 insight 的调用数
        if covered_insights > 0:
            metrics.calls_per_insight = metrics.total_tool_calls / covered_insights
        
        # RAG 搜索深度
        if metrics.rag_tool_calls > 0 and total_insights > 0:
            metrics.rag_search_depth = metrics.rag_tool_calls / total_insights
        
        # 计算各项指标分数（不加权求和）
        task_completion_score = 100.0 if metrics.task_completed else 0.0
        has_final_answer_score = 100.0 if metrics.has_final_answer else 0.0
        efficiency_score = self._calculate_efficiency_score(metrics)
        diversity_score = self._calculate_diversity_score(metrics)
        
        details = {
            'metrics': metrics.to_dict(),
            'individual_scores': {
                'task_completion': {
                    'score': task_completion_score,
                    'description': 'Whether the task was completed',
                    'value': metrics.task_completed
                },
                'has_final_answer': {
                    'score': has_final_answer_score,
                    'description': 'Whether a final answer was generated',
                    'value': metrics.has_final_answer
                },
                'efficiency': {
                    'score': efficiency_score,
                    'description': 'Information density (insights per tool call)',
                    'value': metrics.info_density
                },
                'tool_diversity': {
                    'score': diversity_score,
                    'description': 'Number of unique tools used',
                    'value': metrics.unique_tools_used
                }
            },
            'raw_metrics': {
                'total_tool_calls': metrics.total_tool_calls,
                'unique_tools_used': metrics.unique_tools_used,
                'rag_tool_calls': metrics.rag_tool_calls,
                'info_density': metrics.info_density,
                'rag_call_ratio': metrics.rag_call_ratio,
                'calls_per_insight': metrics.calls_per_insight,
                'total_input_tokens': metrics.total_input_tokens,
                'total_output_tokens': metrics.total_output_tokens,
                'total_duration_seconds': metrics.total_duration_seconds,
                'main_agent_turns': metrics.main_agent_turns
            }
        }
        
        # 计算各项指标的分数
        info_density_score = self._score_info_density(metrics.info_density)
        rag_ratio_score = self._score_rag_ratio(metrics.rag_call_ratio)
        calls_per_insight_score = self._score_calls_per_insight(metrics.calls_per_insight)
        rag_depth_score = self._score_rag_depth(metrics.rag_search_depth)
        
        # 添加各项指标分数到 details
        details['metric_scores'] = {
            'info_density': {
                'value': metrics.info_density,
                'score': info_density_score,
                'formula': 'covered_insights / total_tool_calls',
                'description': '信息密度：每次工具调用获取的 insight 数量'
            },
            'rag_call_ratio': {
                'value': metrics.rag_call_ratio,
                'score': rag_ratio_score,
                'formula': 'rag_tool_calls / total_tool_calls',
                'description': 'RAG 调用比例：RAG 工具调用占总调用的比例'
            },
            'calls_per_insight': {
                'value': metrics.calls_per_insight,
                'score': calls_per_insight_score,
                'formula': 'total_tool_calls / covered_insights',
                'description': '每个 insight 的调用成本：获取一个 insight 需要的工具调用数'
            },
            'rag_search_depth': {
                'value': metrics.rag_search_depth,
                'score': rag_depth_score,
                'formula': 'rag_tool_calls / total_insights',
                'description': 'RAG 搜索深度：每个 insight 的 RAG 搜索次数'
            }
        }
        
        # 综合分数：四个指标的平均分
        overall_score = (info_density_score + rag_ratio_score + calls_per_insight_score + rag_depth_score) / 4
        
        return EvalResult(
            metric_name=self.metric_name,
            score=overall_score,
            details=details,
            weight=self.weight
        )
    
    def _score_info_density(self, value: float) -> float:
        """信息密度分数 (0-100)：越高越好"""
        if value >= 1.0:
            return 100
        elif value >= 0.5:
            return 80
        elif value >= 0.3:
            return 60
        elif value >= 0.1:
            return 40
        elif value > 0:
            return 20
        else:
            return 0
    
    def _score_rag_ratio(self, value: float) -> float:
        """RAG 调用比例分数 (0-100)：适中最好"""
        # RAG 比例在 50%-80% 之间最好
        if 0.5 <= value <= 0.8:
            return 100
        elif 0.3 <= value < 0.5 or 0.8 < value <= 0.9:
            return 80
        elif 0.2 <= value < 0.3 or 0.9 < value <= 1.0:
            return 60
        elif value < 0.2:
            return 40
        else:
            return 50
    
    def _score_calls_per_insight(self, value: float) -> float:
        """每个 insight 调用成本分数 (0-100)：越低越好"""
        if value == 0 or value == float('inf'):
            return 0
        elif value <= 1:
            return 100
        elif value <= 2:
            return 80
        elif value <= 3:
            return 60
        elif value <= 5:
            return 40
        else:
            return 20
    
    def _score_rag_depth(self, value: float) -> float:
        """RAG 搜索深度分数 (0-100)：适中最好"""
        # 搜索深度在 0.3-0.7 之间最好
        if 0.3 <= value <= 0.7:
            return 100
        elif 0.2 <= value < 0.3 or 0.7 < value <= 1.0:
            return 80
        elif 0.1 <= value < 0.2 or 1.0 < value <= 1.5:
            return 60
        elif value < 0.1:
            return 40
        else:
            return 30
    
    def _calculate_score(self, metrics: ToolUsageMetrics) -> float:
        """计算综合分数"""
        score = 0.0
        
        # 任务完成 (30%)
        if metrics.task_completed:
            score += 30
        
        # 有最终答案 (20%)
        if metrics.has_final_answer:
            score += 20
        
        # 效率分数 (25%)
        score += self._calculate_efficiency_score(metrics)
        
        # 工具多样性 (25%)
        score += self._calculate_diversity_score(metrics)
        
        return min(100, score)
    
    def _calculate_efficiency_score(self, metrics: ToolUsageMetrics) -> float:
        """计算效率分数 (0-25)"""
        if metrics.total_tool_calls == 0:
            return 0
        
        # 信息密度越高越好
        if metrics.info_density >= 0.5:
            return 25
        elif metrics.info_density >= 0.3:
            return 20
        elif metrics.info_density >= 0.1:
            return 15
        elif metrics.info_density > 0:
            return 10
        else:
            return 5
    
    def _calculate_diversity_score(self, metrics: ToolUsageMetrics) -> float:
        """计算工具多样性分数 (0-25)"""
        unique_tools = metrics.unique_tools_used
        
        if unique_tools >= 5:
            return 25
        elif unique_tools >= 4:
            return 20
        elif unique_tools >= 3:
            return 15
        elif unique_tools >= 2:
            return 10
        elif unique_tools >= 1:
            return 5
        else:
            return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool usage efficiency")
    parser.add_argument("--execution-log", type=str, required=True, help="Path to execution_log.json")
    parser.add_argument("--gold", type=str, help="Path to gold insights JSON")
    parser.add_argument("--covered-insights", type=int, default=0, help="Number of covered insights")
    parser.add_argument("--output", type=str, help="Output file for evaluation result")
    
    args = parser.parse_args()
    
    # 加载 gold insights
    gold_insights = []
    if args.gold:
        gold_path = Path(args.gold)
        if gold_path.exists():
            gold_data = json.loads(gold_path.read_text(encoding='utf-8'))
            gold_insights = gold_data.get('gold_insights', [])
    
    # 评估
    evaluator = ToolUsageEvaluator()
    result = evaluator.evaluate(
        execution_log_path=Path(args.execution_log),
        gold_insights=gold_insights,
        covered_insights=args.covered_insights
    )
    
    # 输出结果
    print(f"\n🔧 Tool Usage Score: {result.score:.1f}/100")
    print(result.to_json())
    
    if args.output:
        evaluator.save_result(result, Path(args.output))
        print(f"\n📄 Result saved to: {args.output}")


if __name__ == "__main__":
    main()
