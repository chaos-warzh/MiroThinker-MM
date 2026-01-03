#!/usr/bin/env python3
"""
简化版评测脚本：只检查文档引用覆盖率

评测维度：
- 文档引用覆盖率 (002_out.json) - 使用字符串匹配（检查[long_context: "文档名称"格式）

使用方法：
    uv run python evaluate_references_only.py
"""

import json
import re
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import datetime


@dataclass
class EvaluationResult:
    item: str
    score: int
    evidence: str


@dataclass
class ReportEvaluation:
    report_name: str
    model_name: str
    context_size: str
    reference_results: List[EvaluationResult]
    reference_score: float
    reference_count: int
    total_references: int


def parse_filename(filename: str) -> Tuple[str, str]:
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


def check_reference_in_report(report: str, reference: str) -> Tuple[int, str]:
    """
    使用字符串匹配检查报告中是否引用了指定文档
    引用格式: [long_context: "文档名称", chunk X]
    
    返回: (score, evidence)
    """
    reference_clean = reference.strip()
    
    # 1. 直接在报告中搜索完整文档名
    if reference_clean in report:
        idx = report.find(reference_clean)
        start = max(0, idx - 50)
        end = min(len(report), idx + len(reference_clean) + 50)
        evidence = report[start:end].replace('\n', ' ')
        return 1, f"...{evidence}..."
    
    # 2. 搜索文档名的主要部分（去掉后缀和来源）
    name_parts = reference_clean.split('-')
    main_name = name_parts[0].strip()
    main_name = re.sub(r'\.(docx?|pdf|txt|md)$', '', main_name, flags=re.IGNORECASE)
    
    if main_name and len(main_name) > 5 and main_name in report:
        idx = report.find(main_name)
        start = max(0, idx - 50)
        end = min(len(report), idx + len(main_name) + 50)
        evidence = report[start:end].replace('\n', ' ')
        return 1, f"...{evidence}..."
    
    # 3. 搜索long_context格式的引用
    pattern = r'\[long_context:\s*"([^"]+)"'
    matches = re.findall(pattern, report)
    
    for match in matches:
        if is_similar_reference(match, reference_clean):
            idx = report.find(match)
            start = max(0, idx - 30)
            end = min(len(report), idx + len(match) + 50)
            evidence = report[start:end].replace('\n', ' ')
            return 1, f"...{evidence}..."
    
    return 0, "未找到引用"


def is_similar_reference(found: str, target: str) -> bool:
    """检查找到的引用是否与目标文档相似"""
    found_lower = found.lower()
    target_lower = target.lower()
    
    if found_lower in target_lower or target_lower in found_lower:
        return True
    
    def extract_keywords(text):
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [w for w in words if len(w) > 2]
    
    found_keywords = set(extract_keywords(found_lower))
    target_keywords = set(extract_keywords(target_lower))
    
    if len(found_keywords) > 0 and len(target_keywords) > 0:
        overlap = found_keywords & target_keywords
        if len(overlap) >= 2 or len(overlap) / min(len(found_keywords), len(target_keywords)) > 0.5:
            return True
    
    return False


def evaluate_report(report_path: str, references: List[str]) -> ReportEvaluation:
    report_name = Path(report_path).stem
    model_name, context_size = parse_filename(report_path)
    report_content = load_markdown(report_path)
    
    print(f"\n{'='*60}")
    print(f"评测报告: {report_name}")
    print(f"模型: {model_name}, 上下文: {context_size}")
    print(f"{'='*60}")
    
    reference_results = []
    print("\n--- 文档引用评测 (字符串匹配) ---")
    for i, reference in enumerate(references, 1):
        display_name = reference[:40] + "..." if len(reference) > 40 else reference
        print(f"  检查引用 {i}/{len(references)}: {display_name}")
        score, evidence = check_reference_in_report(report_content, reference)
        
        eval_result = EvaluationResult(
            item=reference,
            score=score,
            evidence=evidence
        )
        reference_results.append(eval_result)
        status = "✓" if score == 1 else "✗"
        print(f"    {status} 得分: {score}")
    
    reference_count = sum(r.score for r in reference_results)
    reference_score = reference_count / len(reference_results) if reference_results else 0
    
    print(f"\n  总计: {reference_count}/{len(references)} ({reference_score:.1%})")
    
    return ReportEvaluation(
        report_name=report_name,
        model_name=model_name,
        context_size=context_size,
        reference_results=reference_results,
        reference_score=reference_score,
        reference_count=reference_count,
        total_references=len(references)
    )


def generate_comparison_report(evaluations: List[ReportEvaluation], references: List[str], output_path: str):
    lines = []
    lines.append("# 文档引用评测与比较分析结果\n")
    lines.append(f"评测时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"必须引用的文档数量: {len(references)}\n")
    
    lines.append("## 必须引用的文档列表\n")
    for i, ref in enumerate(references, 1):
        lines.append(f"{i}. {ref}")
    lines.append("\n")
    
    by_model = defaultdict(list)
    by_context = defaultdict(list)
    
    for eval_result in evaluations:
        by_model[eval_result.model_name].append(eval_result)
        by_context[eval_result.context_size].append(eval_result)
    
    context_order = ['32k', '64k', '128k']
    sorted_contexts = sorted(by_context.keys(), key=lambda x: context_order.index(x) if x in context_order else 999)
    
    # 总览表格
    lines.append("## 1. 评分总览\n")
    lines.append("| 报告 | 模型 | 上下文 | 引用数 | 引用率 |")
    lines.append("|------|------|--------|--------|--------|")
    
    sorted_evals = sorted(evaluations, key=lambda x: x.reference_score, reverse=True)
    for eval_result in sorted_evals:
        lines.append(
            f"| {eval_result.report_name} | "
            f"{eval_result.model_name} | "
            f"{eval_result.context_size} | "
            f"{eval_result.reference_count}/{eval_result.total_references} | "
            f"{eval_result.reference_score:.1%} |"
        )
    
    lines.append("\n")
    
    # 模型间比较
    lines.append("## 2. 模型间比较（相同上下文大小）\n")
    
    for context in sorted_contexts:
        evals_in_context = by_context[context]
        if len(evals_in_context) > 1:
            lines.append(f"### 上下文大小: {context}\n")
            lines.append("| 模型 | 引用数 | 引用率 | 排名 |")
            lines.append("|------|--------|--------|------|")
            
            sorted_by_score = sorted(evals_in_context, key=lambda x: x.reference_score, reverse=True)
            for rank, eval_result in enumerate(sorted_by_score, 1):
                medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else ""))
                lines.append(
                    f"| {eval_result.model_name} | "
                    f"{eval_result.reference_count}/{eval_result.total_references} | "
                    f"{eval_result.reference_score:.1%} | "
                    f"{medal} {rank} |"
                )
            lines.append("\n")
            
            best = sorted_by_score[0]
            worst = sorted_by_score[-1]
            diff = best.reference_score - worst.reference_score
            lines.append(f"**分析**: 在 {context} 上下文下，**{best.model_name}** 表现最佳（引用率 {best.reference_score:.1%}），")
            lines.append(f"**{worst.model_name}** 表现最差（引用率 {worst.reference_score:.1%}），差距为 {diff:.1%}。\n")
    
    # 上下文大小比较
    lines.append("## 3. 上下文大小比较（同一模型）\n")
    
    for model_name in sorted(by_model.keys()):
        evals_for_model = by_model[model_name]
        if len(evals_for_model) > 1:
            lines.append(f"### 模型: {model_name}\n")
            lines.append("| 上下文 | 引用数 | 引用率 | 变化趋势 |")
            lines.append("|--------|--------|--------|----------|")
            
            sorted_by_context = sorted(evals_for_model, 
                key=lambda x: context_order.index(x.context_size) if x.context_size in context_order else 999)
            
            prev_score = None
            for eval_result in sorted_by_context:
                if prev_score is not None:
                    diff = eval_result.reference_score - prev_score
                    trend = "📈 +" if diff > 0 else ("📉 " if diff < 0 else "➡️ ")
                    trend += f"{abs(diff):.1%}"
                else:
                    trend = "-"
                
                lines.append(
                    f"| {eval_result.context_size} | "
                    f"{eval_result.reference_count}/{eval_result.total_references} | "
                    f"{eval_result.reference_score:.1%} | "
                    f"{trend} |"
                )
                prev_score = eval_result.reference_score
            lines.append("\n")
    
    # 各文档被引用情况
    lines.append("## 4. 各文档被引用情况\n")
    lines.append("| 文档 | 被引用次数 | 引用率 |")
    lines.append("|------|-----------|--------|")
    
    for ref in references:
        count = sum(1 for e in evaluations for r in e.reference_results if r.item == ref and r.score == 1)
        rate = count / len(evaluations) if evaluations else 0
        display_name = ref[:50] + "..." if len(ref) > 50 else ref
        lines.append(f"| {display_name} | {count}/{len(evaluations)} | {rate:.1%} |")
    
    lines.append("\n")
    
    # 综合分析
    lines.append("## 5. 综合分析\n")
    
    best_overall = max(evaluations, key=lambda x: x.reference_score)
    lines.append(f"### 最佳表现\n")
    lines.append(f"- **最高分报告**: {best_overall.report_name}（{best_overall.model_name} @ {best_overall.context_size}）")
    lines.append(f"- **引用率**: {best_overall.reference_score:.1%} ({best_overall.reference_count}/{best_overall.total_references})\n")
    
    # 模型平均分排名
    lines.append("### 模型平均引用率排名\n")
    lines.append("| 排名 | 模型 | 平均引用率 |")
    lines.append("|------|------|----------|")
    
    model_avg = {}
    for model_name, evals in by_model.items():
        avg_ref = sum(e.reference_score for e in evals) / len(evals)
        model_avg[model_name] = avg_ref
    
    sorted_models = sorted(model_avg.items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, avg_ref) in enumerate(sorted_models, 1):
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else ""))
        lines.append(f"| {medal} {rank} | {model_name} | {avg_ref:.1%} |")
    
    lines.append("\n")
    
    # 详细引用情况
    lines.append("## 6. 详细引用情况\n")
    
    for eval_result in sorted_evals:
        lines.append(f"### {eval_result.report_name}\n")
        lines.append(f"模型: {eval_result.model_name}, 上下文: {eval_result.context_size}\n")
        lines.append(f"引用率: {eval_result.reference_score:.1%} ({eval_result.reference_count}/{eval_result.total_references})\n")
        
        lines.append("| 文档 | 状态 | 证据 |")
        lines.append("|------|------|------|")
        
        for r in eval_result.reference_results:
            status = "✓" if r.score == 1 else "✗"
            display_name = r.item[:40] + "..." if len(r.item) > 40 else r.item
            evidence = r.evidence[:50] + "..." if len(r.evidence) > 50 else r.evidence
            lines.append(f"| {display_name} | {status} | {evidence} |")
        
        lines.append("\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n比较分析报告已保存到: {output_path}")


def main():
    base_dir = Path(__file__).parent
    
    references_data = load_json(base_dir / "002_out.json")
    references = [item["insight"] for item in references_data["gold_insights"]]
    
    print("=" * 60)
    print("文档引用评测系统 (只检查002_out.json)")
    print("=" * 60)
    print(f"必须引用文档数量: {len(references)}")
    
    print("\n必须引用的文档:")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {ref}")
    
    report_files = [f for f in base_dir.glob("*.md") 
                    if f.stem not in ["evaluation_results", "comparison_results", "reference_results"]]
    print(f"\n待评测报告数量: {len(report_files)}")
    
    if not report_files:
        print("未找到任何md报告文件！")
        return
    
    print("\n待评测文件:")
    for f in sorted(report_files):
        model_name, context_size = parse_filename(str(f))
        print(f"  - {f.name} (模型: {model_name}, 上下文: {context_size})")
    
    evaluations = []
    for report_path in sorted(report_files):
        eval_result = evaluate_report(str(report_path), references)
        evaluations.append(eval_result)
    
    output_path = base_dir / "reference_results.md"
    generate_comparison_report(evaluations, references, str(output_path))
    
    # 保存JSON结果
    json_output = {
        "evaluation_time": datetime.datetime.now().isoformat(),
        "references_count": len(references),
        "references": references,
        "results": []
    }
    
    for eval_result in evaluations:
        json_output["results"].append({
            "report_name": eval_result.report_name,
            "model_name": eval_result.model_name,
            "context_size": eval_result.context_size,
            "reference_score": eval_result.reference_score,
            "reference_count": eval_result.reference_count,
            "total_references": eval_result.total_references,
            "reference_details": [
                {"item": r.item, "score": r.score, "evidence": r.evidence}
                for r in eval_result.reference_results
            ]
        })
    
    json_output_path = base_dir / "reference_results.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    print(f"JSON结果已保存到: {json_output_path}")
    
    # 打印最终结果摘要
    print("\n" + "=" * 60)
    print("评测完成！最终结果摘要：")
    print("=" * 60)
    print(f"{'报告名称':<30} {'模型':<20} {'上下文':<8} {'引用率':<10}")
    print("-" * 70)
    
    sorted_evals = sorted(evaluations, key=lambda x: x.reference_score, reverse=True)
    for eval_result in sorted_evals:
        print(f"{eval_result.report_name:<30} {eval_result.model_name:<20} {eval_result.context_size:<8} {eval_result.reference_score:.1%} ({eval_result.reference_count}/{eval_result.total_references})")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
