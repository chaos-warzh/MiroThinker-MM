"""
评估器模块 - 将评估逻辑解构为独立的脚本

每个评估器负责一个指标：
1. information_recall - 信息召回率评估 (30%)
2. factual_accuracy - 事实准确性评估 (30%)
3. overall_quality - 整体质量评估 (15%) - 清晰度 + 洞察力
4. format_compliance - 格式符合性评估 (15%)
5. citation_coverage - 引用覆盖率评估 (10%) - 基于 check_citation.py
6. tool_usage - 工具使用效率评估 (10%) - 基于 evaluate_with_llm.py

共享模块：
- base - 基础类和配置
- llm_client - LLM 客户端
- document_loader - 文档加载器

主脚本：
- run_all - 运行所有评估器

Insights 文件结构：
    insights/batch2_insights/001/
    ├── gold_insights_from_longcontext.json  # 从 long-context 提取的 insights
    └── gold_insights_from_source.json       # 从 source documents 提取的 insights

Usage:
    # 运行所有评估器（使用 --insights-dir 自动查找 insight 文件）
    python -m evaluators.run_all \
        --result examples/001/final_report.md \
        --insights-dir insights/batch2_insights/001 \
        --long-context examples/001/long_context.json \
        --execution-log examples/001/execution_log.json \
        --output-dir examples/001/eval_results
    
    # 或者分别指定 gold 文件
    python -m evaluators.run_all \
        --result examples/001/final_report.md \
        --gold insights/batch2_insights/001/gold_insights_from_longcontext.json \
        --gold-source insights/batch2_insights/001/gold_insights_from_source.json \
        --long-context examples/001/long_context.json \
        --output-dir examples/001/eval_results
    
    # 单独运行信息召回率评估（使用 --insights-dir）
    python -m evaluators.information_recall \
        --result examples/001/final_report.md \
        --insights-dir insights/batch2_insights/001 \
        --output examples/001/eval_information_recall.json
    
    # 运行引用覆盖率评估
    python -m evaluators.citation_coverage \
        --result examples/001/final_report.md \
        --useful-search examples/001/useful_search.json \
        --output examples/001/eval_citation_coverage.json
    
    # 运行工具使用效率评估
    python -m evaluators.tool_usage \
        --execution-log examples/001/execution_log.json \
        --gold insights/batch2_insights/001/gold_insights_from_longcontext.json \
        --output examples/001/eval_tool_usage.json
"""

from .base import EvalConfig, EvalResult, BaseEvaluator
from .llm_client import LLMClient
from .document_loader import DocumentLoader
from .information_recall import InformationRecallEvaluator
from .factual_accuracy import FactualAccuracyEvaluator
from .overall_quality import OverallQualityEvaluator
from .format_compliance import FormatComplianceEvaluator
from .citation_coverage import CitationCoverageEvaluator
from .tool_usage import ToolUsageEvaluator, ToolUsageMetrics, ToolMetricsExtractor
from .run_all import EvaluationRunner, CombinedEvalResult

__all__ = [
    # 基础类
    'EvalConfig',
    'EvalResult', 
    'BaseEvaluator',
    'LLMClient',
    'DocumentLoader',
    # 评估器
    'InformationRecallEvaluator',
    'FactualAccuracyEvaluator',
    'OverallQualityEvaluator',
    'FormatComplianceEvaluator',
    'CitationCoverageEvaluator',
    'ToolUsageEvaluator',
    # 工具使用相关
    'ToolUsageMetrics',
    'ToolMetricsExtractor',
    # 运行器
    'EvaluationRunner',
    'CombinedEvalResult',
]
