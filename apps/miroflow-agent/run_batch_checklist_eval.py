#!/usr/bin/env python3
"""
批量运行 Checklist 评估并生成综合报告

Usage:
    export $(cat .env | grep -v '^#' | xargs)
    uv run python run_batch_checklist_eval.py \
        --result-dir result/20251222_214548 \
        --query-file datasets_batch2/query.jsonl \
        --output-dir evaluation_logs/batch_20251222
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import os
import sys

# 禁用输出缓冲
os.environ['PYTHONUNBUFFERED'] = '1'

def log(msg: str):
    """实时输出日志"""
    print(msg, flush=True)

# 导入评估模块
from evaluate_with_checklist import (
    ChecklistConfig, ChecklistEvaluationPipeline, 
    load_query_from_jsonl, EvaluationResult
)


def find_all_reports(result_dir: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """
    查找所有 final_report.md 文件
    
    Returns:
        {context_size: {model: {case_id: report_path}}}
    """
    reports = defaultdict(lambda: defaultdict(dict))
    
    for report_path in result_dir.rglob("final_report.md"):
        parts = report_path.relative_to(result_dir).parts
        # 结构: datasets_batch2/64k/gpt-4.1/001/final_report.md
        if len(parts) >= 4:
            context_size = parts[1]  # 64k, 128k, etc.
            model = parts[2]  # gpt-4.1, claude37_sonnet, etc.
            case_id = parts[3]  # 001, 002, etc.
            reports[context_size][model][case_id] = report_path
    
    return dict(reports)


def run_batch_evaluation(
    result_dir: Path,
    query_file: Path,
    output_dir: Path,
    config: ChecklistConfig
) -> Dict[str, Any]:
    """运行批量评估"""
    
    # 查找所有报告
    log("🔍 查找所有报告...")
    reports = find_all_reports(result_dir)
    
    total_reports = sum(
        len(cases) 
        for models in reports.values() 
        for cases in models.values()
    )
    log(f"📊 找到 {total_reports} 个报告")
    log(f"   上下文大小: {list(reports.keys())}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化评估器
    pipeline = ChecklistEvaluationPipeline(config)
    
    # 存储所有结果
    all_results = defaultdict(lambda: defaultdict(dict))
    checklist_cache = {}  # 缓存每个 case 的 checklist
    
    # 统计
    processed = 0
    errors = []
    
    # 按 case_id 分组处理（这样可以复用 checklist）
    case_ids = set()
    for context_size, models in reports.items():
        for model, cases in models.items():
            case_ids.update(cases.keys())
    
    case_ids = sorted(case_ids)
    log(f"   Case IDs: {case_ids}")
    
    for case_id in case_ids:
        log(f"\n{'='*60}")
        log(f"📋 处理 Case {case_id}")
        log(f"{'='*60}")
        
        # 加载 query
        query = load_query_from_jsonl(query_file, case_id)
        if not query:
            log(f"⚠️ 跳过 Case {case_id}: 无法加载 query")
            continue
        
        # 生成 checklist（只需要生成一次）
        if case_id not in checklist_cache:
            log(f"📋 为 Case {case_id} 生成 Checklist...")
            checklist = pipeline.generator.generate(query)
            checklist_cache[case_id] = checklist
        else:
            checklist = checklist_cache[case_id]
        
        # 评估每个模型和上下文大小的报告
        for context_size in sorted(reports.keys()):
            for model in sorted(reports[context_size].keys()):
                if case_id not in reports[context_size][model]:
                    continue
                
                report_path = reports[context_size][model][case_id]
                log(f"\n  📄 评估: {context_size}/{model}/{case_id}")
                
                try:
                    # 读取报告
                    report = report_path.read_text(encoding="utf-8")
                    
                    # 评估
                    evaluations = pipeline.evaluator.evaluate(report, checklist)
                    total_score, weighted_score, summary = pipeline.calculator.calculate(
                        checklist, evaluations
                    )
                    
                    # 存储结果
                    result = {
                        "case_id": case_id,
                        "context_size": context_size,
                        "model": model,
                        "total_score": round(total_score, 2),
                        "weighted_score": round(weighted_score, 2),
                        "satisfied_count": sum(1 for e in evaluations if e.satisfied),
                        "total_items": len(evaluations),
                        "evaluations": [
                            {
                                "item_id": e.item_id,
                                "requirement": e.requirement[:100],
                                "satisfied": e.satisfied,
                                "score": e.score
                            }
                            for e in evaluations
                        ]
                    }
                    
                    all_results[context_size][model][case_id] = result
                    processed += 1
                    
                    log(f"     ✅ 得分: {weighted_score:.1f}/100 ({sum(1 for e in evaluations if e.satisfied)}/{len(evaluations)} 满足)")
                    
                except Exception as e:
                    log(f"     ❌ 错误: {e}")
                    errors.append({
                        "context_size": context_size,
                        "model": model,
                        "case_id": case_id,
                        "error": str(e)
                    })
    
    # 保存详细结果
    detail_path = output_dir / "detailed_results.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(dict(all_results), f, ensure_ascii=False, indent=2)
    log(f"\n📁 详细结果已保存到: {detail_path}")
    
    # 生成综合报告
    summary_report = generate_summary_report(all_results, checklist_cache, errors)
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text(summary_report, encoding="utf-8")
    log(f"📁 综合报告已保存到: {report_path}")
    
    return {
        "processed": processed,
        "errors": len(errors),
        "results": dict(all_results)
    }


def generate_summary_report(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    checklist_cache: Dict[str, List],
    errors: List[Dict]
) -> str:
    """生成 Markdown 格式的综合报告"""
    
    lines = [
        "# Checklist 评估综合报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 📊 总体统计",
        "",
    ]
    
    # 计算总体统计
    context_sizes = sorted(results.keys())
    all_models = set()
    for ctx in results.values():
        all_models.update(ctx.keys())
    all_models = sorted(all_models)
    
    # 按模型和上下文大小统计平均分
    model_ctx_scores = defaultdict(lambda: defaultdict(list))
    for ctx_size, models in results.items():
        for model, cases in models.items():
            for case_id, result in cases.items():
                model_ctx_scores[model][ctx_size].append(result["weighted_score"])
    
    # 生成汇总表格
    lines.extend([
        "### 各模型在不同上下文大小下的平均得分",
        "",
        "| 模型 | " + " | ".join(context_sizes) + " | 平均 |",
        "|" + "---|" * (len(context_sizes) + 2),
    ])
    
    for model in all_models:
        row = [model]
        all_scores = []
        for ctx_size in context_sizes:
            scores = model_ctx_scores[model].get(ctx_size, [])
            if scores:
                avg = sum(scores) / len(scores)
                row.append(f"{avg:.1f}")
                all_scores.extend(scores)
            else:
                row.append("-")
        
        if all_scores:
            row.append(f"**{sum(all_scores)/len(all_scores):.1f}**")
        else:
            row.append("-")
        
        lines.append("| " + " | ".join(row) + " |")
    
    lines.extend(["", "---", ""])
    
    # 按上下文大小分组的详细结果
    lines.extend([
        "## 📈 按上下文大小分组的结果",
        "",
    ])
    
    for ctx_size in context_sizes:
        lines.extend([
            f"### {ctx_size}",
            "",
            "| Case | " + " | ".join(all_models) + " |",
            "|" + "---|" * (len(all_models) + 1),
        ])
        
        # 获取所有 case_id
        case_ids = set()
        for model in results.get(ctx_size, {}).values():
            case_ids.update(model.keys())
        case_ids = sorted(case_ids)
        
        for case_id in case_ids:
            row = [case_id]
            for model in all_models:
                result = results.get(ctx_size, {}).get(model, {}).get(case_id)
                if result:
                    score = result["weighted_score"]
                    satisfied = result["satisfied_count"]
                    total = result["total_items"]
                    # 根据分数添加颜色标记
                    if score >= 80:
                        row.append(f"✅ {score:.0f} ({satisfied}/{total})")
                    elif score >= 60:
                        row.append(f"🟡 {score:.0f} ({satisfied}/{total})")
                    else:
                        row.append(f"❌ {score:.0f} ({satisfied}/{total})")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        
        lines.extend(["", ""])
    
    lines.extend(["---", ""])
    
    # 按模型分组的详细结果
    lines.extend([
        "## 🤖 按模型分组的结果",
        "",
    ])
    
    for model in all_models:
        lines.extend([
            f"### {model}",
            "",
            "| Case | " + " | ".join(context_sizes) + " |",
            "|" + "---|" * (len(context_sizes) + 1),
        ])
        
        # 获取所有 case_id
        case_ids = set()
        for ctx_size in context_sizes:
            cases = results.get(ctx_size, {}).get(model, {})
            case_ids.update(cases.keys())
        case_ids = sorted(case_ids)
        
        for case_id in case_ids:
            row = [case_id]
            for ctx_size in context_sizes:
                result = results.get(ctx_size, {}).get(model, {}).get(case_id)
                if result:
                    score = result["weighted_score"]
                    if score >= 80:
                        row.append(f"✅ {score:.0f}")
                    elif score >= 60:
                        row.append(f"🟡 {score:.0f}")
                    else:
                        row.append(f"❌ {score:.0f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        
        lines.extend(["", ""])
    
    # 错误信息
    if errors:
        lines.extend([
            "---",
            "",
            "## ⚠️ 错误信息",
            "",
        ])
        for err in errors:
            lines.append(f"- {err['context_size']}/{err['model']}/{err['case_id']}: {err['error']}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="批量运行 Checklist 评估")
    parser.add_argument("--result-dir", type=str, required=True, help="结果目录")
    parser.add_argument("--query-file", type=str, required=True, help="query.jsonl 文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model", type=str, default="gpt-51-1113-global", help="评估模型")
    
    args = parser.parse_args()
    
    config = ChecklistConfig()
    config.model_name = args.model
    
    run_batch_evaluation(
        result_dir=Path(args.result_dir),
        query_file=Path(args.query_file),
        output_dir=Path(args.output_dir),
        config=config
    )


if __name__ == "__main__":
    main()
