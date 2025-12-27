#!/usr/bin/env python3
"""
基于 Checklist 的 Report 评估系统

工作流程：
1. 根据 query 生成标准答案应该包含的 checklist
2. 根据 checklist 一次性评估 report，计算得分（每项都有分数）

Usage:
    uv run python evaluate_with_checklist.py \
        --query "你的查询问题" \
        --report path/to/report.md \
        --output path/to/evaluation_result.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os

from openai import OpenAI


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class ChecklistConfig:
    """Checklist 评估配置"""
    api_key: str = field(default_factory=lambda: os.getenv(
        "ALIBABA_API_KEY", os.getenv("OPENAI_API_KEY", "")
    ))
    base_url: str = field(default_factory=lambda: os.getenv(
        "ALIBABA_BASE_URL", os.getenv("OPENAI_BASE_URL", "")
    ))
    model_name: str = "gpt-51-1113-global"
    temperature: float = 0.1
    max_checklist_items: int = 15


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


@dataclass
class EvaluationResult:
    """完整评估结果"""
    query: str
    checklist: List[ChecklistItem]
    evaluations: List[ChecklistEvaluation]
    total_score: float
    weighted_score: float
    summary: str
    timestamp: str


# =============================================================================
# LLM 客户端
# =============================================================================

class LLMClient:
    def __init__(self, config: ChecklistConfig):
        self.config = config
        if not config.api_key or not config.base_url:
            raise ValueError("请设置 OPENAI_API_KEY 和 OPENAI_BASE_URL 环境变量")
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    
    def call(self, system: str, user: str, json_mode: bool = True) -> Optional[Dict]:
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
# Checklist 生成器
# =============================================================================

class ChecklistGenerator:
    SYSTEM_PROMPT = """你是一个专业的内容评估专家。根据用户的查询问题，生成一个"标准答案应该包含的内容"的 checklist。请以 JSON 格式输出。

生成 checklist 时请注意：
1. 仔细分析 query 中的每个具体要求和子问题
2. 将要求分解为可验证的具体条目
3. 为每个条目分配合理的权重（所有权重之和应为 1.0）
4. 区分重要性级别：critical（必须有）、important（应该有）、nice_to_have（有更好）

类别：content/accuracy/depth/structure/relevance"""

    USER_PROMPT_TEMPLATE = """请根据以下查询问题，生成一个标准答案应该包含的 checklist。

## 查询问题
{query}

## 输出格式
{{
    "checklist": [
        {{"id": 1, "category": "content", "requirement": "具体要求", "importance": "critical", "weight": 0.15}},
        ...
    ]
}}

注意：生成 {max_items} 个左右的项，所有 weight 之和必须等于 1.0"""

    def __init__(self, llm: LLMClient, config: ChecklistConfig):
        self.llm = llm
        self.config = config
    
    def generate(self, query: str) -> List[ChecklistItem]:
        print("📋 正在生成 Checklist...")
        result = self.llm.call(
            self.SYSTEM_PROMPT,
            self.USER_PROMPT_TEMPLATE.format(query=query, max_items=self.config.max_checklist_items)
        )
        
        if not result or "checklist" not in result:
            return self._default_checklist()
        
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
        
        print(f"✅ 生成了 {len(items)} 个 Checklist 项")
        return items
    
    def _default_checklist(self) -> List[ChecklistItem]:
        return [
            ChecklistItem(1, "content", "回答涵盖了主要问题", "critical", 0.3),
            ChecklistItem(2, "content", "回答涵盖了所有子问题", "critical", 0.2),
            ChecklistItem(3, "accuracy", "信息准确可靠", "critical", 0.2),
            ChecklistItem(4, "depth", "分析有深度", "important", 0.15),
            ChecklistItem(5, "structure", "结构清晰", "important", 0.1),
            ChecklistItem(6, "relevance", "紧扣主题", "nice_to_have", 0.05),
        ]


# =============================================================================
# Report 评估器 - 一次性评估所有项，每项都有分数
# =============================================================================

class ReportEvaluator:
    SYSTEM_PROMPT = """你是一个严格但公正的内容评估专家。一次性评估报告是否满足所有 checklist 要求。请以 JSON 格式输出评估结果。

评估原则：
1. 客观公正：基于报告实际内容评估
2. 证据导向：为每个评估提供具体证据
3. 合理评分：完全满足90-100分，大部分满足70-89分，部分满足50-69分，少量涉及30-49分，未涉及0-29分
4. 宽容原则：不同表达方式表达相同意思也应认可"""

    EVAL_PROMPT_TEMPLATE = """请根据以下 checklist 一次性评估报告内容，为每一项打分。

## Checklist 项目列表
{checklist_items}

## 报告内容
{report}

## 输出格式
{{
    "evaluations": [
        {{"item_id": 1, "satisfied": true/false, "score": 0-100, "evidence": "证据(限80字)", "explanation": "说明(限40字)"}},
        {{"item_id": 2, "satisfied": true/false, "score": 0-100, "evidence": "...", "explanation": "..."}},
        ...
    ]
}}

注意：必须对每个 checklist 项都给出评估结果和分数！"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def evaluate(self, report: str, checklist: List[ChecklistItem]) -> List[ChecklistEvaluation]:
        print(f"📊 正在评估报告（共 {len(checklist)} 项，一次性评估）...")
        
        checklist_text = "\n".join([
            f"[{item.id}] {item.category} | {item.importance} | {item.requirement}"
            for item in checklist
        ])
        
        result = self.llm.call(
            self.SYSTEM_PROMPT,
            self.EVAL_PROMPT_TEMPLATE.format(
                checklist_items=checklist_text,
                report=report[:20000]
            )
        )
        
        if not result or "evaluations" not in result:
            return self._default_evaluations(checklist)
        
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
        
        print("✅ 评估完成")
        return evaluations
    
    def _default_evaluations(self, checklist: List[ChecklistItem]) -> List[ChecklistEvaluation]:
        return [
            ChecklistEvaluation(item.id, item.requirement, False, 0, "评估失败", "LLM调用失败")
            for item in checklist
        ]


# =============================================================================
# 结果计算器
# =============================================================================

class ScoreCalculator:
    def calculate(self, checklist: List[ChecklistItem], 
                  evaluations: List[ChecklistEvaluation]) -> Tuple[float, float, str]:
        if not evaluations:
            return 0.0, 0.0, "无评估结果"
        
        total_score = sum(e.score for e in evaluations) / len(evaluations)
        
        eval_map = {e.item_id: e for e in evaluations}
        weighted_sum = sum(
            eval_map[item.id].score * item.weight
            for item in checklist if item.id in eval_map
        )
        weight_sum = sum(item.weight for item in checklist if item.id in eval_map)
        weighted_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        satisfied_count = sum(1 for e in evaluations if e.satisfied)
        critical_items = [item for item in checklist if item.importance == "critical"]
        critical_satisfied = sum(
            1 for item in critical_items 
            if item.id in eval_map and eval_map[item.id].satisfied
        )
        
        summary = (
            f"总体得分: {total_score:.1f}/100 (加权: {weighted_score:.1f}/100)\n"
            f"满足项: {satisfied_count}/{len(evaluations)}\n"
            f"关键项满足: {critical_satisfied}/{len(critical_items)}"
        )
        
        return total_score, weighted_score, summary


# =============================================================================
# 报告生成器
# =============================================================================

class ResultReporter:
    def generate_report(self, result: EvaluationResult) -> str:
        lines = [
            "=" * 70,
            "📊 Checklist 评估报告",
            "=" * 70,
            f"\n📅 评估时间: {result.timestamp}",
            f"\n📝 Query (前200字): {result.query[:200]}...\n" if len(result.query) > 200 else f"\n📝 Query: {result.query}\n",
            "-" * 70,
            "📋 Checklist 项目:",
            "-" * 70,
        ]
        
        # 按类别分组
        categories = {}
        for item in result.checklist:
            categories.setdefault(item.category, []).append(item)
        
        for category, items in categories.items():
            lines.append(f"\n【{category.upper()}】")
            for item in items:
                icon = {"critical": "🔴", "important": "🟡", "nice_to_have": "🟢"}.get(item.importance, "⚪")
                lines.append(f"  {icon} [{item.id}] {item.requirement} (权重: {item.weight:.2f})")
        
        lines.extend(["\n" + "-" * 70, "📈 评估结果:", "-" * 70])
        
        eval_map = {e.item_id: e for e in result.evaluations}
        for item in result.checklist:
            if item.id in eval_map:
                e = eval_map[item.id]
                status = "✅" if e.satisfied else "❌"
                lines.append(f"\n{status} [{item.id}] {item.requirement}")
                lines.append(f"   得分: {e.score:.0f}/100")
                lines.append(f"   证据: {e.evidence[:100]}..." if len(e.evidence) > 100 else f"   证据: {e.evidence}")
                lines.append(f"   说明: {e.explanation}")
        
        grade = self._get_grade(result.weighted_score)
        lines.extend([
            "\n" + "=" * 70,
            "📊 总结",
            "=" * 70,
            f"\n{result.summary}",
            f"\n🎯 最终得分: {result.weighted_score:.1f}/100",
            grade,
            "\n" + "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def _get_grade(self, score: float) -> str:
        if score >= 90: return "⭐ 等级: 优秀 (A)"
        if score >= 80: return "✨ 等级: 良好 (B)"
        if score >= 70: return "👍 等级: 合格 (C)"
        if score >= 60: return "📌 等级: 及格 (D)"
        return "❌ 等级: 不及格 (F)"
    
    def to_json(self, result: EvaluationResult) -> Dict:
        return {
            "query": result.query,
            "timestamp": result.timestamp,
            "scores": {"total": round(result.total_score, 2), "weighted": round(result.weighted_score, 2)},
            "summary": result.summary,
            "checklist": [asdict(item) for item in result.checklist],
            "evaluations": [asdict(e) for e in result.evaluations]
        }


# =============================================================================
# 主评估流程
# =============================================================================

class ChecklistEvaluationPipeline:
    def __init__(self, config: Optional[ChecklistConfig] = None):
        self.config = config or ChecklistConfig()
        self.llm = LLMClient(self.config)
        self.generator = ChecklistGenerator(self.llm, self.config)
        self.evaluator = ReportEvaluator(self.llm)
        self.calculator = ScoreCalculator()
        self.reporter = ResultReporter()
    
    def run(self, query: str, report: str) -> EvaluationResult:
        print("\n" + "=" * 50)
        print("🚀 开始 Checklist 评估")
        print("=" * 50 + "\n")
        
        checklist = self.generator.generate(query)
        evaluations = self.evaluator.evaluate(report, checklist)
        total_score, weighted_score, summary = self.calculator.calculate(checklist, evaluations)
        
        return EvaluationResult(
            query=query,
            checklist=checklist,
            evaluations=evaluations,
            total_score=total_score,
            weighted_score=weighted_score,
            summary=summary,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def run_and_report(self, query: str, report: str, output_path: Optional[Path] = None) -> str:
        result = self.run(query, report)
        report_text = self.reporter.generate_report(result)
        print("\n" + report_text)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(self.reporter.to_json(result), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"\n📁 结果已保存到: {output_path}")
        
        return report_text


# =============================================================================
# 辅助函数
# =============================================================================

def load_query_from_jsonl(file_path: Path, query_id: str) -> Optional[str]:
    if not file_path.exists():
        return None
    
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            item_id = data.get("id") or data.get("number")
            if item_id:
                normalized_id = f"{int(item_id):03d}" if str(item_id).isdigit() else str(item_id)
                normalized_query_id = f"{int(query_id):03d}" if str(query_id).isdigit() else str(query_id)
                if normalized_id == normalized_query_id:
                    return data.get("query", "")
    return None


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="基于 Checklist 的 Report 评估系统")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", type=str, help="直接指定查询问题")
    query_group.add_argument("--query-file", type=str, help="query.jsonl 文件路径")
    
    parser.add_argument("--query-id", type=str, help="query ID")
    parser.add_argument("--report", type=str, required=True, help="Report 文件路径")
    parser.add_argument("--output", type=str, help="输出 JSON 文件路径")
    parser.add_argument("--model", type=str, default="gpt-51-1113-global", help="模型名称")
    parser.add_argument("--max-items", type=int, default=15, help="最大 checklist 项数")
    
    args = parser.parse_args()
    
    if args.query:
        query = args.query
    else:
        if not args.query_id:
            parser.error("使用 --query-file 时必须指定 --query-id")
        query = load_query_from_jsonl(Path(args.query_file), args.query_id)
        if not query:
            print(f"❌ 未找到 query")
            return
    
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"❌ Report 文件不存在: {report_path}")
        return
    
    report = report_path.read_text(encoding="utf-8")
    
    config = ChecklistConfig()
    config.model_name = args.model
    config.max_checklist_items = args.max_items
    
    pipeline = ChecklistEvaluationPipeline(config)
    output_path = Path(args.output) if args.output else None
    pipeline.run_and_report(query, report, output_path)


if __name__ == "__main__":
    main()
