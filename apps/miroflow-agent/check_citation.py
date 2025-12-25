#!/usr/bin/env python3
"""
检查 final_report.md 中是否引用了 useful_search.json 中的必需文档标题
"""

import json
import os
import sys
from pathlib import Path

def load_useful_search(dataset_dir: str) -> list[str]:
    """加载 useful_search.json 中的标题列表"""
    useful_search_path = Path(dataset_dir) / "useful_search.json"
    if not useful_search_path.exists():
        return []
    
    with open(useful_search_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    titles = [item.get("title", "") for item in data if item.get("title")]
    return titles

def normalize_quotes(s: str) -> str:
    """标准化引号：将中文引号转换为英文引号"""
    # 中文双引号
    s = s.replace('"', '"').replace('"', '"')
    # 中文单引号
    s = s.replace(''', "'").replace(''', "'")
    return s

def check_citation_in_report(report_path: str, titles: list[str]) -> dict:
    """检查报告中是否引用了指定的标题"""
    if not os.path.exists(report_path):
        return {"exists": False, "cited": [], "missing": titles}
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否是空报告
    if "No final answer" in content:
        return {"exists": True, "empty": True, "cited": [], "missing": titles}
    
    # 标准化报告内容中的引号
    content_normalized = normalize_quotes(content)
    
    cited = []
    missing = []
    
    for title in titles:
        # 检查标题是否在报告中出现（可能是部分匹配）
        # 提取标题的主要部分（去掉网站后缀）
        main_title = title.split("-")[0].strip() if "-" in title else title
        
        # 标准化标题中的引号
        main_title_normalized = normalize_quotes(main_title)
        title_normalized = normalize_quotes(title)
        
        # 同时检查原始和标准化后的版本
        if (main_title in content or title in content or 
            main_title_normalized in content_normalized or 
            title_normalized in content_normalized):
            cited.append(title)
        else:
            missing.append(title)
    
    return {"exists": True, "empty": False, "cited": cited, "missing": missing}

def main():
    # 数据集目录
    datasets_dir = Path("datasets_batch2")
    # 结果目录
    results_base = Path("result/20251222_214548/datasets_batch2")
    
    # 获取所有 case 编号
    cases = sorted([d.name for d in datasets_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    # 上下文大小和模型
    context_sizes = ["32k", "64k", "128k", "256k"]
    models = ["gpt-4.1", "claude35_sonnet", "Qwen3-235B-A22B", "Qwen3-30B-A3B"]
    
    # 统计结果
    results = {}
    
    print("# 引用检查报告\n")
    print("检查每个 case 的 final_report.md 是否引用了 useful_search.json 中的必需文档\n")
    
    # 按模型和上下文大小统计
    for model in models:
        results[model] = {}
        for size in context_sizes:
            results[model][size] = {
                "total": 0,
                "full_citation": 0,
                "partial_citation": 0,
                "no_citation": 0,
                "missing_report": 0,
                "empty_report": 0,
                "details": []
            }
    
    # 遍历所有 case
    for case in cases:
        # 加载该 case 的必需标题
        titles = load_useful_search(datasets_dir / case)
        if not titles:
            continue
        
        for size in context_sizes:
            for model in models:
                report_path = results_base / size / model / case / "final_report.md"
                result = check_citation_in_report(str(report_path), titles)
                
                stats = results[model][size]
                stats["total"] += 1
                
                if not result["exists"]:
                    stats["missing_report"] += 1
                    stats["details"].append({
                        "case": case,
                        "status": "missing",
                        "cited": 0,
                        "total": len(titles)
                    })
                elif result.get("empty", False):
                    stats["empty_report"] += 1
                    stats["details"].append({
                        "case": case,
                        "status": "empty",
                        "cited": 0,
                        "total": len(titles)
                    })
                else:
                    cited_count = len(result["cited"])
                    total_count = len(titles)
                    
                    if cited_count == total_count:
                        stats["full_citation"] += 1
                    elif cited_count > 0:
                        stats["partial_citation"] += 1
                    else:
                        stats["no_citation"] += 1
                    
                    stats["details"].append({
                        "case": case,
                        "status": "ok",
                        "cited": cited_count,
                        "total": total_count,
                        "missing_titles": result["missing"]
                    })
    
    # 输出汇总表格
    print("## 汇总统计\n")
    print("| 模型 | Context | 完整引用 | 部分引用 | 无引用 | 空报告 | 缺失报告 |")
    print("|------|---------|----------|----------|--------|--------|----------|")
    
    for model in models:
        for size in context_sizes:
            stats = results[model][size]
            print(f"| {model} | {size} | {stats['full_citation']}/{stats['total']} | {stats['partial_citation']} | {stats['no_citation']} | {stats['empty_report']} | {stats['missing_report']} |")
    
    # 计算引用百分率随上下文长度变化
    print("\n## 引用百分率随上下文长度变化\n")
    print("计算方式: (已引用文档数 / 应引用文档总数) × 100%\n")
    print("注意: 空报告和缺失报告按 0% 计算\n")
    
    # 表头
    print("| 模型 | 32k | 64k | 128k | 256k | 趋势 |")
    print("|------|-----|-----|------|------|------|")
    
    for model in models:
        row = f"| {model} |"
        percentages = []
        
        for size in context_sizes:
            stats = results[model][size]
            total_cited = 0
            total_required = 0
            
            for detail in stats["details"]:
                if detail["status"] == "ok":
                    total_cited += detail["cited"]
                    total_required += detail["total"]
                else:
                    # 空报告或缺失报告，引用数为0
                    total_required += detail["total"]
            
            if total_required > 0:
                percentage = (total_cited / total_required) * 100
            else:
                percentage = 0
            
            percentages.append(percentage)
            row += f" {percentage:.1f}% |"
        
        # 计算趋势
        if len(percentages) >= 2:
            diff = percentages[-1] - percentages[0]
            if diff > 5:
                trend = "📈 上升"
            elif diff < -5:
                trend = "📉 下降"
            else:
                trend = "➡️ 稳定"
        else:
            trend = "-"
        
        row += f" {trend} |"
        print(row)
    
    # 详细的引用率表格（按有效报告计算）
    print("\n## 有效报告引用百分率（排除空报告和缺失报告）\n")
    print("| 模型 | 32k | 64k | 128k | 256k |")
    print("|------|-----|-----|------|------|")
    
    for model in models:
        row = f"| {model} |"
        
        for size in context_sizes:
            stats = results[model][size]
            total_cited = 0
            total_required = 0
            valid_count = 0
            
            for detail in stats["details"]:
                if detail["status"] == "ok":
                    total_cited += detail["cited"]
                    total_required += detail["total"]
                    valid_count += 1
            
            if total_required > 0:
                percentage = (total_cited / total_required) * 100
                row += f" {percentage:.1f}% ({valid_count}个) |"
            else:
                row += " N/A |"
        
        print(row)
    
    # 按 Case 输出详细表格（每个 case 一行，显示所有模型和上下文大小）
    print("\n## 按 Case 详细引用情况\n")
    print("每个单元格显示: 引用数/总数 (✅=完整引用, ⚠️=部分引用, ❌=无引用/空/缺失)\n")
    
    # 表头
    header = "| Case |"
    separator = "|------|"
    for size in context_sizes:
        for model in models:
            short_model = model.replace("claude35_sonnet", "c35").replace("claude37_sonnet", "c37").replace("Qwen3-235B-A22B", "Qwen3").replace("gpt-4.1", "gpt41")
            header += f" {short_model}_{size} |"
            separator += "--------|"
    print(header)
    print(separator)
    
    # 每个 case 一行
    for case in cases:
        row = f"| {case} |"
        for size in context_sizes:
            for model in models:
                # 找到对应的 detail
                detail = None
                for d in results[model][size]["details"]:
                    if d["case"] == case:
                        detail = d
                        break
                
                if detail is None:
                    row += " - |"
                elif detail["status"] == "missing":
                    row += " ❌缺失 |"
                elif detail["status"] == "empty":
                    row += " ❌空 |"
                else:
                    cited = detail["cited"]
                    total = detail["total"]
                    if cited == total:
                        row += f" ✅{cited}/{total} |"
                    elif cited > 0:
                        row += f" ⚠️{cited}/{total} |"
                    else:
                        row += f" ❌{cited}/{total} |"
        print(row)
    
    # 输出详细信息（按模型分组）
    print("\n## 按模型详细引用情况\n")
    
    for model in models:
        print(f"### {model}\n")
        for size in context_sizes:
            stats = results[model][size]
            print(f"#### {size}\n")
            print("| Case | 状态 | 引用数 | 缺失的标题 |")
            print("|------|------|--------|------------|")
            
            for detail in stats["details"]:
                if detail["status"] == "missing":
                    print(f"| {detail['case']} | ❌ 缺失报告 | 0/{detail['total']} | - |")
                elif detail["status"] == "empty":
                    print(f"| {detail['case']} | ⚠️ 空报告 | 0/{detail['total']} | - |")
                else:
                    missing_str = ", ".join([t.split("-")[0][:20] + "..." for t in detail.get("missing_titles", [])]) if detail.get("missing_titles") else "无"
                    status = "✅" if detail["cited"] == detail["total"] else "⚠️"
                    print(f"| {detail['case']} | {status} | {detail['cited']}/{detail['total']} | {missing_str} |")
            print()
    
    # 输出问题 case 列表
    print("\n## 需要关注的 Case\n")
    print("以下 case 存在引用不完整的问题（排除空报告和缺失报告）：\n")
    
    for model in models:
        problem_cases = []
        for size in context_sizes:
            for detail in results[model][size]["details"]:
                if detail["status"] == "ok" and detail["cited"] < detail["total"]:
                    problem_cases.append(f"{size}/{detail['case']}: {detail['cited']}/{detail['total']}")
        
        if problem_cases:
            print(f"### {model}")
            for case in problem_cases:
                print(f"- {case}")
            print()

if __name__ == "__main__":
    main()
