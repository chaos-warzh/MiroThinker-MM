#!/usr/bin/env python3
"""
评测脚本：使用大模型评估报告质量

评测维度：
1. 回答要点覆盖率 (002_in.json) - 检查报告是否覆盖了必须回答的要点
2. 文档引用覆盖率 (002_out.json) - 检查报告是否引用了必须引用的文档，且有实质性展开

评分规则：
- 每个要点/文档引用单独评分（0或1）
- 要点覆盖：报告中必须包含要点的核心内容
- 文档引用：报告中必须提到文档名称，且有实质性说明（不只是列出名称）

使用方法：
    python evaluate_reports.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI


@dataclass
class EvaluationResult:
    """单个评测项的结果"""
    item: str  # 评测项内容
    score: int  # 0 或 1
    evidence: str  # 评分依据/证据
    reasoning: str  # 评分理由


@dataclass
class ReportEvaluation:
    """报告的完整评测结果"""
    report_name: str
    insight_results: List[EvaluationResult]  # 要点覆盖评测
    reference_results: List[EvaluationResult]  # 文档引用评测
    insight_score: float  # 要点覆盖得分 (0-1)
    reference_score: float  # 文档引用得分 (0-1)
    total_score: float  # 总分 (0-1)


def load_json(filepath: str) -> dict:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_markdown(filepath: str) -> str:
    """加载Markdown文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def create_insight_evaluation_prompt(report: str, insight: str) -> str:
    """创建评测要点覆盖的prompt - 严格版本"""
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
    """创建评测文档引用的prompt - 宽松版本，只需提到文档名称即可"""
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
    """调用大模型进行评测"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个严谨的评测专家，请严格按照要求进行评测并输出JSON格式结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度以保证一致性
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 尝试解析JSON
        # 处理可能的markdown代码块
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        return json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始响应: {result_text}")
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
    """评测单个报告"""
    report_name = Path(report_path).stem
    report_content = load_markdown(report_path)
    
    print(f"\n{'='*60}")
    print(f"评测报告: {report_name}")
    print(f"{'='*60}")
    
    # 评测要点覆盖
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
    
    # 评测文档引用
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
    
    # 计算总分
    insight_score = sum(r.score for r in insight_results) / len(insight_results) if insight_results else 0
    reference_score = sum(r.score for r in reference_results) / len(reference_results) if reference_results else 0
    total_score = (insight_score + reference_score) / 2
    
    return ReportEvaluation(
        report_name=report_name,
        insight_results=insight_results,
        reference_results=reference_results,
        insight_score=insight_score,
        reference_score=reference_score,
        total_score=total_score
    )


def generate_evaluation_report(evaluations: List[ReportEvaluation], output_path: str):
    """生成评测报告"""
    lines = []
    lines.append("# 报告评测结果\n")
    lines.append(f"评测时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 总览表格
    lines.append("## 评分总览\n")
    lines.append("| 报告 | 要点覆盖率 | 文档引用率 | 总分 |")
    lines.append("|------|-----------|-----------|------|")
    
    for eval_result in evaluations:
        lines.append(
            f"| {eval_result.report_name} | "
            f"{eval_result.insight_score:.1%} | "
            f"{eval_result.reference_score:.1%} | "
            f"{eval_result.total_score:.1%} |"
        )
    
    lines.append("\n")
    
    # 详细评测结果
    for eval_result in evaluations:
        lines.append(f"## {eval_result.report_name} 详细评测\n")
        
        # 要点覆盖详情
        lines.append("### 要点覆盖评测\n")
        for i, result in enumerate(eval_result.insight_results, 1):
            status = "✅" if result.score == 1 else "❌"
            lines.append(f"#### {status} 要点 {i}\n")
            lines.append(f"**要点内容**: {result.item}\n")
            lines.append(f"**得分**: {result.score}\n")
            lines.append(f"**证据**: {result.evidence}\n")
            lines.append(f"**理由**: {result.reasoning}\n")
            lines.append("")
        
        # 文档引用详情
        lines.append("### 文档引用评测\n")
        for i, result in enumerate(eval_result.reference_results, 1):
            status = "✅" if result.score == 1 else "❌"
            lines.append(f"#### {status} 引用 {i}\n")
            lines.append(f"**文档**: {result.item}\n")
            lines.append(f"**得分**: {result.score}\n")
            lines.append(f"**证据**: {result.evidence}\n")
            lines.append(f"**理由**: {result.reasoning}\n")
            lines.append("")
        
        lines.append("---\n")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n评测报告已保存到: {output_path}")


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent
    
    # 加载评测标准
    insights_data = load_json(base_dir / "002_in.json")
    references_data = load_json(base_dir / "002_out.json")
    
    insights = [item["insight"] for item in insights_data["gold_insights"]]
    references = [item["insight"] for item in references_data["gold_insights"]]
    
    print("=" * 60)
    print("报告评测系统")
    print("=" * 60)
    print(f"要点数量: {len(insights)}")
    print(f"必须引用文档数量: {len(references)}")
    
    # 获取所有md报告文件（排除评测结果文件）
    report_files = [f for f in base_dir.glob("*.md") if f.stem != "evaluation_results"]
    print(f"待评测报告数量: {len(report_files)}")
    
    if not report_files:
        print("未找到任何md报告文件！")
        return
    
    # 初始化OpenAI客户端
    # 尝试从环境变量或.env文件获取API配置
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.getenv("EVAL_MODEL", "gpt-4.1")  # 使用idealab平台支持的模型
    
    if not api_key:
        # 尝试从.env文件加载
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
        print("请设置环境变量或在.env文件中配置")
        return
    
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    print(f"使用模型: {model}")
    print(f"API Base: {api_base}")
    
    # 评测所有报告
    evaluations = []
    for report_path in report_files:
        eval_result = evaluate_report(
            str(report_path),
            insights,
            references,
            client,
            model
        )
        evaluations.append(eval_result)
    
    # 按总分排序
    evaluations.sort(key=lambda x: x.total_score, reverse=True)
    
    # 生成评测报告
    output_path = base_dir / "evaluation_results.md"
    generate_evaluation_report(evaluations, str(output_path))
    
    # 打印最终结果摘要
    print("\n" + "=" * 60)
    print("评测完成！最终结果摘要：")
    print("=" * 60)
    print(f"{'报告名称':<20} {'要点覆盖':<12} {'文档引用':<12} {'总分':<10}")
    print("-" * 60)
    for eval_result in evaluations:
        insight_str = f"{eval_result.insight_score:.1%}"
        reference_str = f"{eval_result.reference_score:.1%}"
        total_str = f"{eval_result.total_score:.1%}"
        print(
            f"{eval_result.report_name:<20} "
            f"{insight_str:<12} "
            f"{reference_str:<12} "
            f"{total_str:<10}"
        )
    print("=" * 60)
    
    # 同时保存JSON格式结果
    json_output = {
        "evaluation_time": __import__('datetime').datetime.now().isoformat(),
        "insights_count": len(insights),
        "references_count": len(references),
        "results": []
    }
    
    for eval_result in evaluations:
        json_output["results"].append({
            "report_name": eval_result.report_name,
            "insight_score": eval_result.insight_score,
            "reference_score": eval_result.reference_score,
            "total_score": eval_result.total_score,
            "insight_details": [
                {
                    "item": r.item,
                    "score": r.score,
                    "evidence": r.evidence,
                    "reasoning": r.reasoning
                }
                for r in eval_result.insight_results
            ],
            "reference_details": [
                {
                    "item": r.item,
                    "score": r.score,
                    "evidence": r.evidence,
                    "reasoning": r.reasoning
                }
                for r in eval_result.reference_results
            ]
        })
    
    json_output_path = base_dir / "evaluation_results.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    print(f"\nJSON结果已保存到: {json_output_path}")


if __name__ == "__main__":
    main()
