#!/usr/bin/env python3
"""
增强版评测脚本：评测所有报告并进行模型间和上下文大小的比较分析

评测维度：
1. 回答要点覆盖率 (002_in.json) - 检查报告是否覆盖了必须回答的要点
2. 文档引用覆盖率 (002_out.json) - 检查报告是否引用了必须引用的文档

比较维度：
1. 模型间比较：在相同上下文大小下，不同模型的表现
2. 上下文比较：同一模型在不同上下文大小下的表现

使用方法：
    python evaluate_and_compare.py
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI


@dataclass
class EvaluationResult:
    """单个评测项的结果"""
    item: str
    score: int
    evidence: str
    reasoning: str


@dataclass
class ReportEvaluation:
    """报告的完整评测结果"""
    report_name: str
    model_name: str
    context_size: str
    insight_results: List[EvaluationResult]
    reference_results: List[EvaluationResult]
    insight_score: float
    reference_score: float
    total_score: float


def parse_filename(filename: str) -> Tuple[str, str]:
    """解析文件名，提取模型名称和上下文大小"""
    stem = Path(filename).stem
    match = re.match(r'^(.+)_(\d+k)$', stem)
    if match:
        return match.group(1), match.group(2)
    return stem, "unknown"


def load_json(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_markdown(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def create_insight_evaluation_prompt(report: str, insight: str) -> str:
    return f"""你是一个非常严格的评测专家。请判断以下报告是否**精确、完整地**覆盖了指定的要点。

## 评测要点
{insight}

## 待评测报告
{report}

## 严格评测标准
1. 报告必须**明确、具体地**论述要点中的核心内容，不能只是泛泛而谈
2. 要点中的**每个关键概念**都必须在报告中有对应的论述
3. 如果要点包含具体的定义、数据、案例要求，报告必须有相应的具体内容
4. 仅仅提及相关关键词但没有深入展开，**不算覆盖**
5. 论述内容与要点只是"相关"但不是"精确对应"，**不算覆盖**
6. 如果要点要求说明某个概念的定义，报告必须给出明确的定义，而不是只提到这个概念

## 评分标准
- 得分1：报告中有明确、具体、完整的论述，精确对应要点的核心内容
- 得分0：报告中没有相关内容，或只是泛泛提及，或论述不够具体完整

## 输出格式
请严格按照以下JSON格式输出，不要输出其他内容：
{{
    "score": 0或1,
    "evidence": "报告中与该要点相关的具体内容（如果有的话，摘录原文）",
    "reasoning": "评分理由，说明为什么给出这个分数，特别是如果给0分要说明缺少什么"
}}
"""


def create_reference_evaluation_prompt(report: str, reference: str) -> str:
    return f"""你是一个评测专家。请判断以下报告是否提到了指定的文档。

## 必须引用的文档
{reference}

## 待评测报告
{report}

## 评测要求（宽松标准）
1. 只需要检查报告中是否**提到了该文档的名称**（可以是文档名称的一部分，或者通过long_context标注引用）
2. 文档名称可能出现在：
   - 正文中的引用标注，如 [long_context: "文档名称", chunk X]
   - 参考文献列表
   - 正文中直接提及文档名称
3. **只要提到了文档名称，就算通过**，不需要有实质性展开
4. 文档名称匹配可以是部分匹配，例如"字节跳动短视频平台2025算法"可以匹配"字节跳动短视频平台2025算法在内容监管与风险控制中的应用报告"

## 评分标准
- 得分1：报告中提到了该文档的名称（完整或部分）
- 得分0：报告中完全没有提到该文档

## 输出格式
请严格按照以下JSON格式输出，不要输出其他内容：
{{
    "score": 0或1,
    "evidence": "报告中提到该文档的位置（如果有的话，摘录原文）",
    "reasoning": "评分理由"
}}
"""


def call_llm_for_evaluation(prompt: str, client: OpenAI, model: str = "gpt-4o-mini") -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个严谨的评测专家，请严格按照要求进行评测并输出JSON格式结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        return json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return {"score": 0, "evidence": "", "reasoning": f"解析错误: {e}"}
    except Exception as e:
        print(f"API调用错误: {e}")
        return {"score": 0, "evidence": "", "reasoning": f"API错误: {e}"}


def evaluate_report(
    report_path: str,
    insights: List[str],
    references: List[str],
    client: OpenAI,
    model: str = "gpt-4o-mini"
) -> ReportEvaluation:
    report_name = Path(report_path).stem
    model_name, context_size = parse_filename(report_path)
    report_content = load_markdown(report_path)
    
    print(f"\n{'='*60}")
    print(f"评测报告: {report_name}")
    print(f"模型: {model_name}, 上下文: {context_size}")
    print(f"{'='*60}")
    
    insight_results = []
    print("\n--- 要点覆盖评测 ---")
    for i, insight in enumerate(insights, 1):
        print(f"  评测要点 {i}/{len(insights)}...")
        prompt = create_insight_evaluation_prompt(report_content, insight)
        result = call_llm_for_evaluation(prompt, client, model)
        
        eval_result = EvaluationResult(
            item=insight,
            score=result.get("score", 0),
            evidence=result.get("evidence", ""),
            reasoning=result.get("reasoning", "")
        )
        insight_results.append(eval_result)
        print(f"    得分: {eval_result.score}")
    
    reference_results = []
    print("\n--- 文档引用评测 ---")
    for i, reference in enumerate(references, 1):
        print(f"  评测引用 {i}/{len(references)}...")
        prompt = create_reference_evaluation_prompt(report_content, reference)
        result = call_llm_for_evaluation(prompt, client, model)
        
        eval_result = EvaluationResult(
            item=reference,
            score=result.get("score", 0),
            evidence=result.get("evidence", ""),
            reasoning=result.get("reasoning", "")
        )
        reference_results.append(eval_result)
        print(f"    得分: {eval_result.score}")
    
    insight_score = sum(r.score for r in insight_results) / len(insight_results) if insight_results else 0
    reference_score = sum(r.score for r in reference_results) / len(reference_results) if reference_results else 0
    total_score = (insight_score + reference_score) / 2
    
    return ReportEvaluation(
        report_name=report_name,
        model_name=model_name,
        context_size=context_size,
        insight_results=insight_results,
        reference_results=reference_results,
        insight_score=insight_score,
        reference_score=reference_score,
        total_score=total_score
    )


def generate_comparison_report(evaluations: List[ReportEvaluation], output_path: str):
    lines = []
    lines.append("# 报告评测与比较分析结果\n")
    lines.append(f"评测时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    by_model = defaultdict(list)
    by_context = defaultdict(list)
    
    for eval_result in evaluations:
        by_model[eval_result.model_name].append(eval_result)
        by_context[eval_result.context_size].append(eval_result)
    
    context_order = ['32k', '64k', '128k']
    sorted_contexts = sorted(by_context.keys(), key=lambda x: context_order.index(x) if x in context_order else 999)
    
    # 总览表格
    lines.append("## 1. 评分总览\n")
    lines.append("| 报告 | 模型 | 上下文 | 要点覆盖率 | 文档引用率 | 总分 |")
    lines.append("|------|------|--------|-----------|-----------|------|")
    
    sorted_evals = sorted(evaluations, key=lambda x: x.total_score, reverse=True)
    for eval_result in sorted_evals:
        lines.append(
            f"| {eval_result.report_name} | "
            f"{eval_result.model_name} | "
            f"{eval_result.context_size} | "
            f"{eval_result.insight_score:.1%} | "
            f"{eval_result.reference_score:.1%} | "
            f"{eval_result.total_score:.1%} |"
        )
    
    lines.append("\n")
    
    # 模型间比较
    lines.append("## 2. 模型间比较（相同上下文大小）\n")
    
    for context in sorted_contexts:
        evals_in_context = by_context[context]
        if len(evals_in_context) > 1:
            lines.append(f"### 上下文大小: {context}\n")
            lines.append("| 模型 | 要点覆盖率 | 文档引用率 | 总分 | 排名 |")
            lines.append("|------|-----------|-----------|------|------|")
            
            sorted_by_score = sorted(evals_in_context, key=lambda x: x.total_score, reverse=True)
            for rank, eval_result in enumerate(sorted_by_score, 1):
                medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else ""))
                lines.append(
                    f"| {eval_result.model_name} | "
                    f"{eval_result.insight_score:.1%} | "
                    f"{eval_result.reference_score:.1%} | "
                    f"{eval_result.total_score:.1%} | "
                    f"{medal} {rank} |"
                )
            lines.append("\n")
            
            best = sorted_by_score[0]
            worst = sorted_by_score[-1]
            diff = best.total_score - worst.total_score
            lines.append(f"**分析**: 在 {context} 上下文下，**{best.model_name}** 表现最佳（总分 {best.total_score:.1%}），")
            lines.append(f"**{worst.model_name}** 表现最差（总分 {worst.total_score:.1%}），差距为 {diff:.1%}。\n")
    
    # 上下文大小比较
    lines.append("## 3. 上下文大小比较（同一模型）\n")
    
    for model_name in sorted(by_model.keys()):
        evals_for_model = by_model[model_name]
        if len(evals_for_model) > 1:
            lines.append(f"### 模型: {model_name}\n")
            lines.append("| 上下文 | 要点覆盖率 | 文档引用率 | 总分 | 变化趋势 |")
            lines.append("|--------|-----------|-----------|------|----------|")
            
            sorted_by_context = sorted(evals_for_model, 
                key=lambda x: context_order.index(x.context_size) if x.context_size in context_order else 999)
            
            prev_score = None
            for eval_result in sorted_by_context:
                if prev_score is not None:
                    diff = eval_result.total_score - prev_score
                    trend = "📈 +" if diff > 0 else ("📉 " if diff < 0 else "➡️ ")
                    trend += f"{abs(diff):.1%}"
                else:
                    trend = "-"
                
                lines.append(
                    f"| {eval_result.context_size} | "
                    f"{eval_result.insight_score:.1%} | "
                    f"{eval_result.reference_score:.1%} | "
                    f"{eval_result.total_score:.1%} | "
                    f"{trend} |"
                )
                prev_score = eval_result.total_score
            lines.append("\n")
            
            # 分析趋势
            first = sorted_by_context[0]
            last = sorted_by_context[-1]
            overall_diff = last.total_score - first.total_score
            if overall_diff > 0.05:
                lines.append(f"**分析**: 随着上下文增大，{model_name} 的表现**提升**（从 {first.total_score:.1%} 到 {last.total_score:.1%}）。\n")
            elif overall_diff < -0.05:
                lines.append(f"**分析**: 随着上下文增大，{model_name} 的表现**下降**（从 {first.total_score:.1%} 到 {last.total_score:.1%}）。\n")
            else:
                lines.append(f"**分析**: {model_name} 在不同上下文大小下表现**稳定**（{first.total_score:.1%} ~ {last.total_score:.1%}）。\n")
    
    # 综合分析
    lines.append("## 4. 综合分析\n")
    
    # 找出最佳模型
    best_overall = max(evaluations, key=lambda x: x.total_score)
    lines.append(f"### 最佳表现\n")
    lines.append(f"- **最高分报告**: {best_overall.report_name}（{best_overall.model_name} @ {best_overall.context_size}）")
    lines.append(f"- **总分**: {best_overall.total_score:.1%}")
    lines.append(f"- **要点覆盖率**: {best_overall.insight_score:.1%}")
    lines.append(f"- **文档引用率**: {best_overall.reference_score:.1%}\n")
    
    # 模型平均分排名
    lines.append("### 模型平均分排名\n")
    lines.append("| 排名 | 模型 | 平均总分 | 平均要点覆盖 | 平均文档引用 |")
    lines.append("|------|------|----------|-------------|-------------|")
    
    model_avg = {}
    for model_name, evals in by_model.items():
        avg_total = sum(e.total_score for e in evals) / len(evals)
        avg_insight = sum(e.insight_score for e in evals) / len(evals)
        avg_ref = sum(e.reference_score for e in evals) / len(evals)
        model_avg[model_name] = (avg_total, avg_insight, avg_ref)
    
    sorted_models = sorted(model_avg.items(), key=lambda x: x[1][0], reverse=True)
    for rank, (model_name, (avg_total, avg_insight, avg_ref)) in enumerate(sorted_models, 1):
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else ""))
        lines.append(f"| {medal} {rank} | {model_name} | {avg_total:.1%} | {avg_insight:.1%} | {avg_ref:.1%} |")
    
    lines.append("\n")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n比较分析报告已保存到: {output_path}")


def main():
    base_dir = Path(__file__).parent
    
    # 加载评测标准
    insights_data = load_json(base_dir / "002_in.json")
    references_data = load_json(base_dir / "002_out.json")
    
    insights = [item["insight"] for item in insights_data["gold_insights"]]
    references = [item["insight"] for item in references_data["gold_insights"]]
    
    print("=" * 60)
    print("报告评测与比较分析系统")
    print("=" * 60)
    print(f"要点数量: {len(insights)}")
    print(f"必须引用文档数量: {len(references)}")
    
    # 获取所有md报告文件
    report_files = [f for f in base_dir.glob("*.md") 
                    if f.stem not in ["evaluation_results", "comparison_results"]]
    print(f"待评测报告数量: {len(report_files)}")
    
    if not report_files:
        print("未找到任何md报告文件！")
        return
    
    # 显示文件列表
    print("\n待评测文件:")
    for f in sorted(report_files):
        model_name, context_size = parse_filename(str(f))
        print(f"  - {f.name} (模型: {model_name}, 上下文: {context_size})")
    
    # 初始化OpenAI客户端
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.getenv("EVAL_MODEL", "gpt-4.1")
    
    if not api_key:
        env_path = base_dir.parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("OPENAI_BASE_URL="):
                        api_base = line.split("=", 1)[1].strip().strip('"').strip("'")
    
    if not api_key:
        print("错误: 未找到OPENAI_API_KEY环境变量")
        return
    
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    print(f"\n使用模型: {model}")
    print(f"API Base: {api_base}")
    
    # 评测所有报告
    evaluations = []
    for report_path in sorted(report_files):
        eval_result = evaluate_report(
            str(report_path),
            insights,
            references,
            client,
            model
        )
        evaluations.append(eval_result)
    
    # 生成比较分析报告
    output_path = base_dir / "comparison_results.md"
    generate_comparison_report(evaluations, str(output_path))
    
    # 保存JSON结果
    json_output = {
        "evaluation_time": datetime.datetime.now().isoformat(),
        "insights_count": len(insights),
        "references_count": len(references),
        "results": []
    }
    
    for eval_result in evaluations:
        json_output["results"].append({
            "report_name": eval_result.report_name,
            "model_name": eval_result.model_name,
            "context_size": eval_result.context_size,
            "insight_score": eval_result.insight_score,
            "reference_score": eval_result.reference_score,
            "total_score": eval_result.total_score,
            "insight_details": [
                {"item": r.item, "score": r.score, "evidence": r.evidence, "reasoning": r.reasoning}
                for r in eval_result.insight_results
            ],
            "reference_details": [
                {"item": r.item, "score": r.score, "evidence": r.evidence, "reasoning": r.reasoning}
                for r in eval_result.reference_results
            ]
        })
    
    json_output_path = base_dir / "comparison_results.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    print(f"JSON结果已保存到: {json_output_path}")
    
    # 打印最终摘要
    print("\n" + "=" * 60)
    print("评测完成！最终结果摘要：")
    print("=" * 60)
    print(f"{'报告名称':<25} {'模型':<20} {'上下文':<8} {'总分':<10}")
    print("-" * 60)
    
    sorted_evals = sorted(evaluations, key=lambda x: x.total_score, reverse=True)
    for eval_result in sorted_evals:
        print(f"{eval_result.report_name:<25} {eval_result.model_name:<20} {eval_result.context_size:<8} {eval_result.total_score:.1%}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
