#!/usr/bin/env python3
"""
运行所有评估器的主脚本

Usage:
    python -m evaluators.run_all \
        --result examples/001/final_report.md \
        --gold examples/001/gold_insights.json \
        --long-context examples/001/long_context.json \
        --output-dir examples/001/eval_results
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from .base import EvalConfig, EvalResult

# 获取 logger（与 run_batch_eval.py 共享同一个 logger）
logger = logging.getLogger("batch_eval")
from .information_recall import InformationRecallEvaluator
from .factual_accuracy import FactualAccuracyEvaluator
from .overall_quality import OverallQualityEvaluator
from .format_compliance import FormatComplianceEvaluator
from .citation_coverage import CitationCoverageEvaluator
from .tool_usage import ToolUsageEvaluator


@dataclass
class CombinedEvalResult:
    """综合评估结果"""
    total_score: float
    results: Dict[str, EvalResult]
    evaluation_time: str
    
    def to_dict(self) -> Dict:
        return {
            'total_score': self.total_score,
            'evaluation_time': self.evaluation_time,
            'metrics': {
                name: result.to_dict() for name, result in self.results.items()
            }
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# 所有可用的评估指标
ALL_METRICS = [
    'information_recall',
    'factual_accuracy', 
    'overall_quality',
    'format_compliance',
    'citation_coverage',
    'tool_usage',
]


class EvaluationRunner:
    """评估运行器 - 运行所有评估器并汇总结果"""
    
    # 类级别的可用指标列表
    AVAILABLE_METRICS = ALL_METRICS
    
    def __init__(self, config: Optional[EvalConfig] = None, metrics: Optional[List[str]] = None):
        """初始化评估运行器
        
        Args:
            config: 评估配置
            metrics: 要运行的指标列表，如果为 None 则运行所有指标
        """
        self.config = config or EvalConfig()
        self.enabled_metrics = metrics if metrics else ALL_METRICS
        
        # 验证指标名称
        invalid_metrics = set(self.enabled_metrics) - set(ALL_METRICS)
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Available metrics: {ALL_METRICS}")
        
        # 初始化所有评估器（但只运行启用的）
        self._all_evaluators = {
            'information_recall': InformationRecallEvaluator(self.config),
            'factual_accuracy': FactualAccuracyEvaluator(self.config),
            'overall_quality': OverallQualityEvaluator(self.config),
            'format_compliance': FormatComplianceEvaluator(self.config),
            'citation_coverage': CitationCoverageEvaluator(self.config),
            'tool_usage': ToolUsageEvaluator(self.config),
        }
        
        # 只保留启用的评估器
        self.evaluators = {k: v for k, v in self._all_evaluators.items() if k in self.enabled_metrics}
    
    def run(self, 
            result_path: Path,
            gold_path: Optional[Path] = None,
            gold_source_path: Optional[Path] = None,
            long_context_path: Optional[Path] = None,
            source_folder: Optional[Path] = None,
            query_file: Optional[Path] = None,
            case_id: Optional[str] = None,
            useful_search_path: Optional[Path] = None,
            execution_log_path: Optional[Path] = None,
            checklist_path: Optional[Path] = None) -> CombinedEvalResult:
        """运行所有评估
        
        Args:
            result_path: 待评估的报告文件路径
            gold_path: gold insights JSON 文件路径
            gold_source_path: source gold insights JSON 文件路径
            long_context_path: long-context JSON 文件路径
            source_folder: 源文档文件夹路径
            query_file: query.jsonl 文件路径
            case_id: 案例 ID
            useful_search_path: useful_search.json 文件路径
            execution_log_path: execution_log.json 文件路径
            checklist_path: checklist.json 文件路径
            
        Returns:
            CombinedEvalResult 包含所有评估结果
        """
        # 加载报告文本
        result_text = result_path.read_text(encoding='utf-8')
        
        # 加载 gold insights
        gold_insights = []
        source_gold_insights = []
        if gold_path and gold_path.exists():
            gold_data = json.loads(gold_path.read_text(encoding='utf-8'))
            gold_insights = gold_data.get('gold_insights', [])
        if gold_source_path and gold_source_path.exists():
            source_data = json.loads(gold_source_path.read_text(encoding='utf-8'))
            source_gold_insights = source_data.get('gold_insights', [])
        
        # 加载 query 数据
        query_data = None
        if query_file and query_file.exists() and case_id:
            query_data = self._load_query_data(query_file, case_id)
        
        results = {}
        step = 0
        
        # 显示将要运行的指标
        print(f"\n📊 Running {len(self.enabled_metrics)} metrics: {', '.join(self.enabled_metrics)}")
        
        # 1. 信息召回率评估
        if 'information_recall' in self.enabled_metrics:
            step += 1
            print(f"\n📋 {step}. Evaluating Information Recall...")
            results['information_recall'] = self.evaluators['information_recall'].evaluate(
                result_text,
                gold_insights=gold_insights,
                source_gold_insights=source_gold_insights
            )
            print(f"   Score: {results['information_recall'].score:.1f}/100")
        
        # 2. 事实准确性评估
        if 'factual_accuracy' in self.enabled_metrics:
            step += 1
            print(f"\n✅ {step}. Evaluating Factual Accuracy...")
            results['factual_accuracy'] = self.evaluators['factual_accuracy'].evaluate(
                result_text,
                long_context_path=long_context_path,
                source_folder=source_folder
            )
            print(f"   Score: {results['factual_accuracy'].score:.1f}/100")
        
        # 3. 整体质量评估
        if 'overall_quality' in self.enabled_metrics:
            step += 1
            print(f"\n📝 {step}. Evaluating Overall Quality...")
            results['overall_quality'] = self.evaluators['overall_quality'].evaluate(result_text)
            print(f"   Score: {results['overall_quality'].score:.1f}/100")
        
        # 4. 格式符合性评估（基于 Checklist）
        if 'format_compliance' in self.enabled_metrics:
            step += 1
            print(f"\n📋 {step}. Evaluating Format Compliance (Checklist)...")
            results['format_compliance'] = self.evaluators['format_compliance'].evaluate(
                result_text,
                checklist_path=checklist_path
            )
            print(f"   Score: {results['format_compliance'].score:.1f}/100")
        
        # 5. 引用覆盖率评估
        if 'citation_coverage' in self.enabled_metrics:
            step += 1
            print(f"\n🔗 {step}. Evaluating Citation Coverage...")
            results['citation_coverage'] = self.evaluators['citation_coverage'].evaluate(
                result_text,
                useful_search_path=useful_search_path
            )
            print(f"   Score: {results['citation_coverage'].score:.1f}/100")
        
        # 6. 工具使用效率评估
        if 'tool_usage' in self.enabled_metrics:
            step += 1
            print(f"\n🔧 {step}. Evaluating Tool Usage...")
            # 获取已覆盖的 insights 数量（从信息召回率评估结果中）
            covered_insights = 0
            total_insights = 0
            if 'information_recall' in results:
                ir_details = results['information_recall'].details
                if 'components' in ir_details:
                    for comp in ir_details['components'].values():
                        if comp.get('available') and 'details' in comp:
                            covered_insights += comp['details'].get('covered_count', 0)
                            total_insights += comp['details'].get('total_count', 0)
            
            # 合并 gold_insights 和 source_gold_insights 的总数
            all_gold_insights = gold_insights + source_gold_insights
            
            results['tool_usage'] = self.evaluators['tool_usage'].evaluate(
                result_text,
                execution_log_path=execution_log_path,
                gold_insights=all_gold_insights,
                covered_insights=covered_insights
            )
            print(f"   Score: {results['tool_usage'].score:.1f}/100")
        
        # 计算加权总分
        total_score = self._calculate_total_score(results)
        
        return CombinedEvalResult(
            total_score=total_score,
            results=results,
            evaluation_time=datetime.now().isoformat()
        )
    
    def _load_query_data(self, query_file: Path, case_id: str) -> Optional[Dict]:
        """从 query.jsonl 加载指定案例的查询数据"""
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    query_id = data.get('id') or data.get('number')
                    if str(query_id) == str(case_id):
                        return data
                except json.JSONDecodeError:
                    continue
        return None
    
    def _calculate_total_score(self, results: Dict[str, EvalResult]) -> float:
        """计算加权总分"""
        total_weight = sum(r.weight for r in results.values())
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(r.score * r.weight for r in results.values())
        return weighted_sum / total_weight * (total_weight / sum(e.weight for e in self.evaluators.values()))
    
    def generate_report(self, combined_result: CombinedEvalResult) -> str:
        """生成评估报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("📊 EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"\n📅 Evaluation Time: {combined_result.evaluation_time}")
        lines.append(f"\n🎯 TOTAL SCORE: {combined_result.total_score:.1f}/100")
        lines.append(self._get_grade(combined_result.total_score))
        lines.append(self._make_progress_bar(combined_result.total_score, "Overall"))
        
        lines.append("\n📈 DIMENSION SCORES:")
        for name, result in combined_result.results.items():
            weight_pct = int(result.weight * 100)
            lines.append(f"\n  {self._get_metric_emoji(name)} {name} ({weight_pct}%): {result.score:.1f}/100")
            lines.append(f"  {self._make_progress_bar(result.score, name)}")
        
        # 添加工具使用效率指标专区
        if 'tool_usage' in combined_result.results:
            lines.append(self._format_tool_efficiency_section(combined_result.results['tool_usage']))
        
        lines.append("\n📝 DETAILED RESULTS:")
        
        # 1. Information Recall 详情
        if 'information_recall' in combined_result.results:
            lines.append(self._format_information_recall_details(combined_result.results['information_recall']))
        
        # 2. Factual Accuracy 详情
        if 'factual_accuracy' in combined_result.results:
            lines.append(self._format_factual_accuracy_details(combined_result.results['factual_accuracy']))
        
        # 3. Format Compliance 详情
        if 'format_compliance' in combined_result.results:
            lines.append(self._format_format_compliance_details(combined_result.results['format_compliance']))
        
        # 4. Citation Coverage 详情
        if 'citation_coverage' in combined_result.results:
            lines.append(self._format_citation_coverage_details(combined_result.results['citation_coverage']))
        
        # 5. Overall Quality 详情
        if 'overall_quality' in combined_result.results:
            lines.append(self._format_overall_quality_details(combined_result.results['overall_quality']))
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
    
    def _format_tool_efficiency_section(self, tool_result: EvalResult) -> str:
        """格式化工具使用效率指标专区"""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("🔧 TOOL USAGE EFFICIENCY METRICS")
        lines.append("-" * 60)
        
        details = tool_result.details
        metrics = details.get('metrics', {})
        
        if not metrics:
            lines.append("  ⚠️ No execution log available")
            return "\n".join(lines)
        
        # 基础统计
        lines.append("\n📊 Basic Statistics:")
        lines.append(f"  • Total Tool Calls: {metrics.get('total_tool_calls', 0)}")
        lines.append(f"  • Unique Tools Used: {metrics.get('unique_tools_used', 0)}")
        lines.append(f"  • RAG Tool Calls: {metrics.get('rag_tool_calls', 0)}")
        lines.append(f"  • Main Agent Turns: {metrics.get('main_agent_turns', 0)}")
        lines.append(f"  • RAG Sub-Agent Turns: {metrics.get('rag_sub_agent_turns', 0)}")
        lines.append(f"  • Total Duration: {metrics.get('total_duration_seconds', 0):.1f}s")
        lines.append(f"  • Total Tokens: {metrics.get('total_input_tokens', 0) + metrics.get('total_output_tokens', 0):,}")
        
        # 效率指标
        lines.append("\n📈 Efficiency Metrics:")
        
        # 信息密度
        info_density = metrics.get('info_density', 0)
        lines.append(f"  • Info Density: {info_density:.3f}")
        lines.append(f"    (Covered Insights / Total Tool Calls = {metrics.get('covered_insights', 0)} / {metrics.get('total_tool_calls', 0)})")
        
        # RAG 调用比例
        rag_ratio = metrics.get('rag_call_ratio', 0)
        lines.append(f"  • RAG Call Ratio: {rag_ratio:.1%}")
        lines.append(f"    (RAG Tool Calls / Total Tool Calls = {metrics.get('rag_tool_calls', 0)} / {metrics.get('total_tool_calls', 0)})")
        
        # 每 insight 调用成本
        calls_per_insight = metrics.get('calls_per_insight', 0)
        if calls_per_insight == float('inf') or calls_per_insight > 1000:
            lines.append(f"  • Calls per Insight: N/A (no insights covered)")
        else:
            lines.append(f"  • Calls per Insight: {calls_per_insight:.2f}")
            lines.append(f"    (Total Tool Calls / Covered Insights = {metrics.get('total_tool_calls', 0)} / {metrics.get('covered_insights', 0)})")
        
        # RAG 搜索深度
        rag_depth = metrics.get('rag_search_depth', 0)
        lines.append(f"  • RAG Search Depth: {rag_depth:.2f}")
        lines.append(f"    (RAG Tool Calls / Total Insights = {metrics.get('rag_tool_calls', 0)} / {metrics.get('total_insights', 0)})")
        
        # Insight 覆盖情况
        lines.append("\n📋 Insight Coverage:")
        lines.append(f"  • Covered Insights: {metrics.get('covered_insights', 0)}")
        lines.append(f"  • Total Insights: {metrics.get('total_insights', 0)}")
        covered = metrics.get('covered_insights', 0)
        total = metrics.get('total_insights', 0)
        if total > 0:
            lines.append(f"  • Coverage Rate: {100 * covered / total:.1f}%")
        
        # 工具调用分布
        tool_breakdown = metrics.get('tool_call_breakdown', {})
        if tool_breakdown:
            lines.append("\n🔨 Tool Call Breakdown:")
            for tool, count in sorted(tool_breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"  • {tool}: {count}")
        
        # 任务完成状态
        lines.append("\n✅ Task Status:")
        lines.append(f"  • Task Completed: {'Yes' if metrics.get('task_completed') else 'No'}")
        lines.append(f"  • Has Final Answer: {'Yes' if metrics.get('has_final_answer') else 'No'}")
        
        lines.append("-" * 60)
        return "\n".join(lines)
    
    def _get_grade(self, score: float) -> str:
        """获取等级评定"""
        if score >= 90:
            return "⭐ Grade: EXCELLENT (A)"
        elif score >= 80:
            return "✨ Grade: GOOD (B)"
        elif score >= 70:
            return "👍 Grade: SATISFACTORY (C)"
        elif score >= 60:
            return "📌 Grade: PASS (D)"
        else:
            return "❌ Grade: FAIL (F)"
    
    def _make_progress_bar(self, score: float, label: str = "", width: int = 30) -> str:
        """生成进度条"""
        filled = int(score / 100 * width)
        empty = width - filled
        
        # 根据分数选择颜色/符号
        if score >= 90:
            fill_char = "█"
            color_indicator = "🟢"
        elif score >= 70:
            fill_char = "█"
            color_indicator = "🟡"
        elif score >= 50:
            fill_char = "█"
            color_indicator = "🟠"
        else:
            fill_char = "█"
            color_indicator = "🔴"
        
        bar = fill_char * filled + "░" * empty
        return f"{color_indicator} [{bar}] {score:.1f}%"
    
    def _get_metric_emoji(self, metric_name: str) -> str:
        """获取指标对应的 emoji"""
        emoji_map = {
            'information_recall': '📋',
            'factual_accuracy': '✅',
            'overall_quality': '📝',
            'format_compliance': '📐',
            'citation_coverage': '🔗',
            'tool_usage': '🔧'
        }
        return emoji_map.get(metric_name, '📊')
    
    def _format_information_recall_details(self, result: EvalResult) -> str:
        """格式化信息召回率详情"""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("📋 INFORMATION RECALL DETAILS")
        lines.append("-" * 60)
        lines.append(f"Score: {result.score:.1f}/100")
        lines.append(self._make_progress_bar(result.score))
        
        details = result.details
        components = details.get('components', {})
        
        # Long Context 部分
        lc = components.get('long_context', {})
        if lc.get('available'):
            lc_details = lc.get('details', {})
            lc_score = lc.get('score', 0)
            covered = lc_details.get('covered_count', 0)
            total = lc_details.get('total_count', 0)
            lines.append(f"\n📄 Long Context Recall: {lc_score:.1f}/100")
            lines.append(f"   Covered: {covered}/{total} insights")
            lines.append(f"   {self._make_progress_bar(lc_score, width=25)}")
            
            # 显示覆盖的 insights
            eval_result = lc_details.get('evaluation_result', {})
            covered_insights = eval_result.get('covered_insights', [])
            
            covered_list = [i for i in covered_insights if i.get('covered')]
            uncovered_list = [i for i in covered_insights if not i.get('covered')]
            
            if covered_list:
                lines.append(f"\n   ✅ Covered Insights ({len(covered_list)}):")
                for item in covered_list[:5]:
                    lines.append(f"      • [{item.get('id')}] {item.get('explanation', '')[:50]}...")
                if len(covered_list) > 5:
                    lines.append(f"      ... and {len(covered_list) - 5} more")
            
            if uncovered_list:
                lines.append(f"\n   ❌ Missing Insights ({len(uncovered_list)}):")
                for item in uncovered_list[:5]:
                    lines.append(f"      • [{item.get('id')}] {item.get('explanation', '')[:50]}...")
                if len(uncovered_list) > 5:
                    lines.append(f"      ... and {len(uncovered_list) - 5} more")
        else:
            lines.append("\n📄 Long Context Recall: N/A (no gold insights)")
        
        # Source 部分
        src = components.get('source_documents', {})
        if src.get('available'):
            src_details = src.get('details', {})
            src_score = src.get('score', 0)
            covered = src_details.get('covered_count', 0)
            total = src_details.get('total_count', 0)
            lines.append(f"\n📁 Source Documents Recall: {src_score:.1f}/100")
            lines.append(f"   Covered: {covered}/{total} insights")
            lines.append(f"   {self._make_progress_bar(src_score, width=25)}")
            
            # 显示覆盖的 insights
            eval_result = src_details.get('evaluation_result', {})
            covered_insights = eval_result.get('covered_insights', [])
            
            covered_list = [i for i in covered_insights if i.get('covered')]
            uncovered_list = [i for i in covered_insights if not i.get('covered')]
            
            if covered_list:
                lines.append(f"\n   ✅ Covered Insights ({len(covered_list)}):")
                for item in covered_list[:5]:
                    lines.append(f"      • [{item.get('id')}] {item.get('explanation', '')[:50]}...")
                if len(covered_list) > 5:
                    lines.append(f"      ... and {len(covered_list) - 5} more")
            
            if uncovered_list:
                lines.append(f"\n   ❌ Missing Insights ({len(uncovered_list)}):")
                for item in uncovered_list[:5]:
                    lines.append(f"      • [{item.get('id')}] {item.get('explanation', '')[:50]}...")
                if len(uncovered_list) > 5:
                    lines.append(f"      ... and {len(uncovered_list) - 5} more")
        else:
            lines.append("\n📁 Source Documents Recall: N/A (no source gold insights)")
        
        return "\n".join(lines)
    
    def _format_factual_accuracy_details(self, result: EvalResult) -> str:
        """格式化事实准确性详情"""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("✅ FACTUAL ACCURACY DETAILS")
        lines.append("-" * 60)
        lines.append(f"Score: {result.score:.1f}/100")
        lines.append(self._make_progress_bar(result.score))
        
        details = result.details
        vr = details.get('verification_result', {})
        
        if vr:
            total = vr.get('total_citations', 0)
            supported = vr.get('supported_count', 0)
            not_found = vr.get('not_found_count', 0)
            unsupported = vr.get('unsupported_count', 0)
            
            lines.append(f"\n📊 Citation Verification:")
            lines.append(f"   • Total Citations: {total}")
            lines.append(f"   • ✅ Supported: {supported}")
            lines.append(f"   • ❓ Source Not Found: {not_found}")
            lines.append(f"   • ❌ Unsupported: {unsupported}")
            
            if total > 0:
                accuracy = supported / total * 100
                lines.append(f"   • Accuracy: {accuracy:.1f}%")
                lines.append(f"   {self._make_progress_bar(accuracy, width=25)}")
            
            # 显示验证详情
            verifications = vr.get('verifications', [])
            unsupported_items = [v for v in verifications if not v.get('supported')]
            if unsupported_items:
                lines.append(f"\n⚠️ Unsupported Citations ({len(unsupported_items)}):")
                for item in unsupported_items[:3]:
                    citation = item.get('citation', '')[:50]
                    explanation = item.get('explanation', '')[:80]
                    lines.append(f"   • {citation}")
                    lines.append(f"     Reason: {explanation}")
        else:
            lines.append("\n⚠️ No verification data available")
            lines.append(f"   Method: {details.get('method', 'unknown')}")
        
        lines.append(f"\n📚 Sources Checked:")
        lines.append(f"   • Long Context Documents: {details.get('long_context_checked', 0)}")
        lines.append(f"   • Source Documents: {details.get('source_documents_checked', 0)}")
        
        return "\n".join(lines)
    
    def _format_format_compliance_details(self, result: EvalResult) -> str:
        """格式化格式符合性详情"""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("📐 FORMAT COMPLIANCE DETAILS (Checklist)")
        lines.append("-" * 60)
        lines.append(f"Score: {result.score:.1f}/100")
        lines.append(self._make_progress_bar(result.score))
        
        details = result.details
        
        if details.get('method') == 'checklist':
            satisfied = details.get('satisfied_count', 0)
            total = details.get('total_items', 0)
            lines.append(f"\n📋 Checklist Items: {satisfied}/{total} satisfied")
            
            if total > 0:
                rate = satisfied / total * 100
                lines.append(f"   {self._make_progress_bar(rate, width=25)}")
            
            # 显示未满足的项目
            unsatisfied = details.get('unsatisfied_items', [])
            if unsatisfied:
                lines.append(f"\n❌ Unsatisfied Items ({len(unsatisfied)}):")
                for item in unsatisfied[:5]:
                    lines.append(f"   • {item[:70]}...")
                if len(unsatisfied) > 5:
                    lines.append(f"   ... and {len(unsatisfied) - 5} more")
            
            # 显示满足的项目
            satisfied_items = details.get('satisfied_items', [])
            if satisfied_items:
                lines.append(f"\n✅ Satisfied Items ({len(satisfied_items)}):")
                for item in satisfied_items[:3]:
                    lines.append(f"   • {item[:70]}...")
        else:
            lines.append("\n⚠️ No checklist available, using basic format check")
        
        return "\n".join(lines)
    
    def _format_citation_coverage_details(self, result: EvalResult) -> str:
        """格式化引用覆盖率详情"""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("🔗 CITATION COVERAGE DETAILS")
        lines.append("-" * 60)
        lines.append(f"Score: {result.score:.1f}/100")
        lines.append(self._make_progress_bar(result.score))
        
        details = result.details
        
        total_useful = details.get('total_useful_sources', 0)
        cited = details.get('cited_count', 0)
        
        lines.append(f"\n📊 Coverage Statistics:")
        lines.append(f"   • Useful Sources: {total_useful}")
        lines.append(f"   • Cited in Report: {cited}")
        
        if total_useful > 0:
            coverage = cited / total_useful * 100
            lines.append(f"   • Coverage Rate: {coverage:.1f}%")
            lines.append(f"   {self._make_progress_bar(coverage, width=25)}")
        
        # 显示未引用的来源
        uncited = details.get('uncited_sources', [])
        if uncited:
            lines.append(f"\n⚠️ Uncited Sources ({len(uncited)}):")
            for src in uncited[:5]:
                lines.append(f"   • {src[:60]}...")
            if len(uncited) > 5:
                lines.append(f"   ... and {len(uncited) - 5} more")
        
        return "\n".join(lines)
    
    def _format_overall_quality_details(self, result: EvalResult) -> str:
        """格式化整体质量详情"""
        lines = []
        lines.append("\n" + "-" * 60)
        lines.append("📝 OVERALL QUALITY DETAILS")
        lines.append("-" * 60)
        lines.append(f"Score: {result.score:.1f}/100")
        lines.append(self._make_progress_bar(result.score))
        
        details = result.details
        sub_scores = details.get('sub_scores', {})
        
        if sub_scores:
            lines.append("\n📊 Sub-Scores:")
            for name, score in sub_scores.items():
                lines.append(f"   • {name}: {score}/100")
                lines.append(f"     {self._make_progress_bar(score, width=20)}")
        
        # 显示问题
        issues = details.get('issues', [])
        if issues:
            lines.append(f"\n⚠️ Issues ({len(issues)}):")
            for issue in issues[:5]:
                lines.append(f"   • {issue}")
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run all evaluators")
    parser.add_argument("--result", type=str, required=True, help="Path to result markdown file")
    parser.add_argument("--gold", type=str, help="Path to gold insights JSON (from long context)")
    parser.add_argument("--gold-source", type=str, help="Path to source gold insights JSON")
    parser.add_argument("--insights-dir", type=str, help="Path to insights directory containing gold_insights_from_longcontext.json and gold_insights_from_source.json")
    parser.add_argument("--long-context", type=str, help="Path to long context JSON")
    parser.add_argument("--source-folder", type=str, help="Path to source documents folder")
    parser.add_argument("--query-file", type=str, help="Path to query.jsonl file")
    parser.add_argument("--case-id", type=str, help="Case ID")
    parser.add_argument("--useful-search", type=str, help="Path to useful_search.json file")
    parser.add_argument("--execution-log", type=str, help="Path to execution_log.json file")
    parser.add_argument("--output-dir", type=str, help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    # 处理 insights 路径
    gold_path = None
    gold_source_path = None
    
    if args.insights_dir:
        # 使用 insights-dir 自动查找两个 insight 文件
        insights_dir = Path(args.insights_dir)
        longcontext_file = insights_dir / "gold_insights_from_longcontext.json"
        source_file = insights_dir / "gold_insights_from_source.json"
        
        if longcontext_file.exists():
            gold_path = longcontext_file
        if source_file.exists():
            gold_source_path = source_file
    else:
        # 使用单独指定的路径
        if args.gold:
            gold_path = Path(args.gold)
        if args.gold_source:
            gold_source_path = Path(args.gold_source)
    
    # 创建运行器
    runner = EvaluationRunner()
    
    # 运行评估
    result = runner.run(
        result_path=Path(args.result),
        gold_path=gold_path,
        gold_source_path=gold_source_path,
        long_context_path=Path(args.long_context) if args.long_context else None,
        source_folder=Path(args.source_folder) if args.source_folder else None,
        query_file=Path(args.query_file) if args.query_file else None,
        case_id=args.case_id,
        useful_search_path=Path(args.useful_search) if args.useful_search else None,
        execution_log_path=Path(args.execution_log) if args.execution_log else None
    )
    
    # 生成报告
    report = runner.generate_report(result)
    print("\n" + report)
    
    # 保存结果
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存综合结果
        combined_path = output_dir / "combined_result.json"
        combined_path.write_text(result.to_json(), encoding='utf-8')
        print(f"\n📊 Combined result saved to: {combined_path}")
        
        # 保存报告
        report_path = output_dir / "evaluation_report.txt"
        report_path.write_text(report, encoding='utf-8')
        print(f"📄 Report saved to: {report_path}")
        
        # 保存各个指标的详细结果
        for name, eval_result in result.results.items():
            metric_path = output_dir / f"eval_{name}.json"
            metric_path.write_text(eval_result.to_json(), encoding='utf-8')
            print(f"📁 {name} result saved to: {metric_path}")


if __name__ == "__main__":
    main()
