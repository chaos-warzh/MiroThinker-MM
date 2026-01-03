#!/usr/bin/env python3
"""
优化版评测脚本：评测所有报告并进行模型间和上下文大小的比较分析

评测维度：
1. 文档引用覆盖率 (002_out.json) - 使用字符串匹配
2. 噪声引用率 - 引用了多少不相关的文档（除了6篇必须引用的文档以外的long_context引用）

使用方法：
    uv run python evaluate_and_compare_v2.py
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Set
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
    # 必须引用的文档
    reference_results: List[EvaluationResult]
    reference_score: float
    reference_count: int
    total_references: int
    # 噪声引用
    all_citations: List[str]  # 所有long_context引用
    valid_citations: List[str]  # 有效引用（匹配必须引用的文档）
    noise_citations: List[str]  # 噪声引用（不匹配必须引用的文档）
    noise_count: int
    noise_rate: float  # 噪声引用率 = 噪声引用数 / 总引用数


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


def extract_all_long_context_citations(report: str) -> List[str]:
    """
    提取报告中所有的long_context引用
    引用格式: [long_context: "文档名称", chunk X] 或 [long_context: "文档名称"]
    
    返回: 去重后的文档名称列表
    """
    # 匹配 [long_context: "文档名称", chunk X] 或 [long_context: "文档名称"]
    pattern = r'\[long_context:\s*"([^"]+)"'
    matches = re.findall(pattern, report)
    
    # 去重但保持顺序
    seen = set()
    unique_citations = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            unique_citations.append(match)
    
    return unique_citations


def is_citation_matching_reference(citation: str, reference: str) -> bool:
    """检查引用是否匹配必须引用的文档"""
    citation_lower = citation.lower()
    reference_lower = reference.lower()
    
    # 直接包含
    if citation_lower in reference_lower or reference_lower in citation_lower:
        return True
    
    # 提取关键词进行匹配
    def extract_keywords(text):
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [w for w in words if len(w) > 2]
    
    citation_keywords = set(extract_keywords(citation_lower))
    reference_keywords = set(extract_keywords(reference_lower))
    
    if len(citation_keywords) > 0 and len(reference_keywords) > 0:
        overlap = citation_keywords & reference_keywords
        # 至少有2个关键词重叠，或者重叠率超过50%
        if len(overlap) >= 2 or len(overlap) / min(len(citation_keywords), len(reference_keywords)) > 0.5:
            return True
    
    return False


def check_reference_in_report(report: str, reference: str) -> Tuple[int, str]:
    """
    使用字符串匹配检查报告中是否引用了指定文档
    
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
        if is_citation_matching_reference(match, reference_clean):
            idx = report.find(match)
            start = max(0, idx - 30)
            end = min(len(report), idx + len(match) + 50)
            evidence = report[start:end].replace('\n', ' ')
            return 1, f"...{evidence}..."
    
    return 0, "未找到引用"


def evaluate_report(report_path: str, references: List[str]) -> ReportEvaluation:
    report_name = Path(report_path).stem
    model_name, context_size = parse_filename(report_path)
    report_content = load_markdown(report_path)
    
    print(f"\n{'='*60}")
    print(f"评测报告: {report_name}")
    print(f"模型: {model_name}, 上下文: {context_size}")
    print(f"{'='*60}")
    
    # 1. 提取所有long_context引用
    all_citations = extract_all_long_context_citations(report_content)
    print(f"\n--- 提取到的所有引用 ({len(all_citations)}个) ---")
    for i, citation in enumerate(all_citations, 1):
        display = citation[:50] + "..." if len(citation) > 50 else citation
        print(f"  {i}. {display}")
    
    # 2. 评测必须引用的文档
    reference_results = []
    valid_citations = []
    print(f"\n--- 文档引用评测 ({len(references)}个必须引用) ---")
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
    
    # 3. 计算噪声引用
    # 找出哪些引用是有效的（匹配必须引用的文档）
    for citation in all_citations:
        for reference in references:
            if is_citation_matching_reference(citation, reference):
                if citation not in valid_citations:
                    valid_citations.append(citation)
                break
    
    # 噪声引用 = 所有引用 - 有效引用
    noise_citations = [c for c in all_citations if c not in valid_citations]
    
    reference_count = sum(r.score for r in reference_results)
    reference_score = reference_count / len(reference_results) if reference_results else 0
    noise_count = len(noise_citations)
    total_citations = len(all_citations)
    noise_rate = noise_count / total_citations if total_citations > 0 else 0
    
    print(f"\n--- 噪声引用分析 ---")
    print(f"  总引用数: {total_citations}")
    print(f"  有效引用数: {len(valid_citations)}")
    print(f"  噪声引用数: {noise_count}")
    print(f"  噪声引用率: {noise_rate:.1%}")
    
    if noise_citations:
        print(f"\n  噪声引用列表:")
        for i, citation in enumerate(noise_citations, 1):
            display = citation[:50] + "..." if len(citation) > 50 else citation
            print(f"    {i}. {display}")
    
    print(f"\n  必须引用覆盖率: {reference_count}/{len(references)} ({reference_score:.1%})")
    
    return ReportEvaluation(
        report_name=report_name,
        model_name=model_name,
        context_size=context_size,
        reference_results=reference_results,
        reference_score=reference_score,
        reference_count=reference_count,
        total_references=len(references),
        all_citations=all_citations,
        valid_citations=valid_citations,
        noise_citations=noise_citations,
        noise_count=noise_count,
        noise_rate=noise_rate
    )


def generate_comparison_report(evaluations: List[ReportEvaluation], references: List[str], output_path: str):
    lines = []
    lines.append("# 文档引用评测与噪声分析结果\n")
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
    lines.append("| 报告 | 模型 | 上下文 | 引用覆盖 | 总引用数 | 噪声引用数 | 噪声率 |")
    lines.append("|------|------|--------|----------|----------|-----------|--------|")
    
    sorted_evals = sorted(evaluations, key=lambda x: (x.reference_score, -x.noise_rate), reverse=True)
    for eval_result in sorted_evals:
        lines.append(
            f"| {eval_result.report_name} | "
            f"{eval_result.model_name} | "
            f"{eval_result.context_size} | "
            f"{eval_result.reference_count}/{eval_result.total_references} ({eval_result.reference_score:.1%}) | "
            f"{len(eval_result.all_citations)} | "
            f"{eval_result.noise_count} | "
            f"{eval_result.noise_rate:.1%} |"
        )
    
    lines.append("\n")
    
    # 模型间比较
    lines.append("## 2. 模型间比较（相同上下文大小）\n")
    
    for context in sorted_contexts:
        evals_in_context = by_context[context]
        if len(evals_in_context) > 1:
            lines.append(f"### 上下文大小: {context}\n")
            lines.append("| 模型 | 引用覆盖 | 总引用数 | 噪声引用数 | 噪声率 | 排名 |")
            lines.append("|------|----------|----------|-----------|--------|------|")
            
            # 按引用覆盖率排序，噪声率作为次要排序
            sorted_by_score = sorted(evals_in_context, key=lambda x: (x.reference_score, -x.noise_rate), reverse=True)
            for rank, eval_result in enumerate(sorted_by_score, 1):
                medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else ""))
                lines.append(
                    f"| {eval_result.model_name} | "
                    f"{eval_result.reference_count}/{eval_result.total_references} ({eval_result.reference_score:.1%}) | "
                    f"{len(eval_result.all_citations)} | "
                    f"{eval_result.noise_count} | "
                    f"{eval_result.noise_rate:.1%} | "
                    f"{medal} {rank} |"
                )
            lines.append("\n")
    
    # 上下文大小比较
    lines.append("## 3. 上下文大小比较（同一模型）\n")
    
    for model_name in sorted(by_model.keys()):
        evals_for_model = by_model[model_name]
        if len(evals_for_model) > 1:
            lines.append(f"### 模型: {model_name}\n")
            lines.append("| 上下文 | 引用覆盖 | 总引用数 | 噪声引用数 | 噪声率 | 趋势 |")
            lines.append("|--------|----------|----------|-----------|--------|------|")
            
            sorted_by_context = sorted(evals_for_model, 
                key=lambda x: context_order.index(x.context_size) if x.context_size in context_order else 999)
            
            prev_noise_rate = None
            for eval_result in sorted_by_context:
                if prev_noise_rate is not None:
                    diff = eval_result.noise_rate - prev_noise_rate
                    trend = "📈 +" if diff > 0 else ("📉 " if diff < 0 else "➡️ ")
                    trend += f"{abs(diff):.1%}"
                else:
                    trend = "-"
                
                lines.append(
                    f"| {eval_result.context_size} | "
                    f"{eval_result.reference_count}/{eval_result.total_references} ({eval_result.reference_score:.1%}) | "
                    f"{len(eval_result.all_citations)} | "
                    f"{eval_result.noise_count} | "
                    f"{eval_result.noise_rate:.1%} | "
                    f"{trend} |"
                )
                prev_noise_rate = eval_result.noise_rate
            lines.append("\n")
    
    # 噪声引用详情
    lines.append("## 4. 噪声引用详情\n")
    
    for eval_result in sorted_evals:
        if eval_result.noise_citations:
            lines.append(f"### {eval_result.report_name}\n")
            lines.append(f"模型: {eval_result.model_name}, 上下文: {eval_result.context_size}\n")
            lines.append(f"噪声引用数: {eval_result.noise_count}, 噪声率: {eval_result.noise_rate:.1%}\n")
            lines.append("| 序号 | 噪声引用文档 |")
            lines.append("|------|-------------|")
            for i, citation in enumerate(eval_result.noise_citations, 1):
                display = citation[:60] + "..." if len(citation) > 60 else citation
                lines.append(f"| {i} | {display} |")
            lines.append("\n")
    
    # 综合分析
    lines.append("## 5. 综合分析\n")
    
    # 按上下文大小统计平均噪声率
    lines.append("### 各上下文大小的平均噪声率\n")
    lines.append("| 上下文 | 平均噪声率 | 平均引用覆盖率 |")
    lines.append("|--------|-----------|---------------|")
    
    for context in sorted_contexts:
        evals_in_context = by_context[context]
        avg_noise = sum(e.noise_rate for e in evals_in_context) / len(evals_in_context)
        avg_ref = sum(e.reference_score for e in evals_in_context) / len(evals_in_context)
        lines.append(f"| {context} | {avg_noise:.1%} | {avg_ref:.1%} |")
    
    lines.append("\n")
    
    # 模型平均噪声率排名
    lines.append("### 模型平均噪声率排名（越低越好）\n")
    lines.append("| 排名 | 模型 | 平均噪声率 | 平均引用覆盖率 |")
    lines.append("|------|------|-----------|---------------|")
    
    model_avg = {}
    for model_name, evals in by_model.items():
        avg_noise = sum(e.noise_rate for e in evals) / len(evals)
        avg_ref = sum(e.reference_score for e in evals) / len(evals)
        model_avg[model_name] = (avg_noise, avg_ref)
    
    # 按噪声率升序排序（越低越好）
    sorted_models = sorted(model_avg.items(), key=lambda x: x[1][0])
    for rank, (model_name, (avg_noise, avg_ref)) in enumerate(sorted_models, 1):
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else ""))
        lines.append(f"| {medal} {rank} | {model_name} | {avg_noise:.1%} | {avg_ref:.1%} |")
    
    lines.append("\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n比较分析报告已保存到: {output_path}")


def main():
    base_dir = Path(__file__).parent
    
    references_data = load_json(base_dir / "002_out.json")
    references = [item["insight"] for item in references_data["gold_insights"]]
    
    print("=" * 60)
    print("文档引用评测系统 (含噪声引用分析)")
    print("=" * 60)
    print(f"必须引用文档数量: {len(references)}")
    
    print("\n必须引用的文档:")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {ref}")
    
    report_files = [f for f in base_dir.glob("*.md") 
                    if f.stem not in ["evaluation_results", "comparison_results", "reference_results", "model_context_analysis"]]
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
    
    output_path = base_dir / "noise_analysis_results.md"
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
            "all_citations_count": len(eval_result.all_citations),
            "valid_citations_count": len(eval_result.valid_citations),
            "noise_count": eval_result.noise_count,
            "noise_rate": eval_result.noise_rate,
            "all_citations": eval_result.all_citations,
            "valid_citations": eval_result.valid_citations,
            "noise_citations": eval_result.noise_citations,
            "reference_details": [
                {"item": r.item, "score": r.score, "evidence": r.evidence}
                for r in eval_result.reference_results
            ]
        })
    
    json_output_path = base_dir / "noise_analysis_results.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    print(f"JSON结果已保存到: {json_output_path}")
    
    # 打印最终结果摘要
    print("\n" + "=" * 60)
    print("评测完成！最终结果摘要：")
    print("=" * 60)
    print(f"{'报告名称':<30} {'模型':<18} {'上下文':<6} {'引用覆盖':<12} {'噪声率':<10}")
    print("-" * 80)
    
    sorted_evals = sorted(evaluations, key=lambda x: (x.reference_score, -x.noise_rate), reverse=True)
    for eval_result in sorted_evals:
        ref_str = f"{eval_result.reference_count}/{eval_result.total_references} ({eval_result.reference_score:.1%})"
        print(f"{eval_result.report_name:<30} {eval_result.model_name:<18} {eval_result.context_size:<6} {ref_str:<12} {eval_result.noise_rate:.1%}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
