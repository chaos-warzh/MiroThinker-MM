"""
从多个模型生成的报告中提取 Insight，并生成标准 Insight 并集。

使用 LLM 来提取 insight，API 配置从 .env 文件读取。

Insight 定义：回答 Query 所必须涉及的关键信息点
- 用户文档 Insight：必须引用用户上传文档才能回答的信息
- 检索文档 Insight（Long Context）：必须从外部检索获取的信息
- 综合分析 Insight：需要综合用户文档和检索信息才能得出的分析
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# 获取 API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 提取 insight 的 prompt
EXTRACT_INSIGHT_PROMPT = """你是一个专业的 Insight 提取专家。你的任务是从给定的报告中提取关键 Insight。

## Insight 的定义

Insight 是"回答 Query 所必须涉及的关键信息点"，分为三类：

1. **用户文档 Insight**：必须引用用户上传文档（如 PPT）才能回答的信息
   - 例如：PPT 中提到的核心概念、定义、原则等

2. **检索文档 Insight（Long Context）**：必须从外部检索获取的信息
   - 例如：具体案例、最新数据、行业实践等

3. **综合分析 Insight**：需要综合用户文档和检索信息才能得出的分析
   - 例如：趋势分析、对比结论、建议等

## 提取要求

每个 Insight 应该：
- **可验证**：能明确判断报告是否包含这个信息
- **必要**：缺少这个信息，报告就不完整
- **独立**：每个是独立的信息点，不重复
- **粒度适中**：1-2句话可以描述清楚

## 输出格式

请以 JSON 格式输出，格式如下：
```json
{{
  "user_doc_insights": [
    "insight 1",
    "insight 2"
  ],
  "long_context_insights": [
    "insight 1",
    "insight 2"
  ],
  "analysis_insights": [
    "insight 1",
    "insight 2"
  ]
}}
```

## Query

{query}

## 报告内容

{report}

请提取报告中的所有关键 Insight，并按类别分类输出。
"""

# 合并 insight 的 prompt
MERGE_INSIGHT_PROMPT = """你是一个专业的 Insight 合并专家。你的任务是将多个来源的 Insight 进行去重和合并，生成一个标准的 Insight 集合。

## 合并要求

1. **去重**：语义相同或高度相似的 Insight 只保留一个
2. **保留完整性**：合并时保留信息最完整的版本
3. **分类准确**：确保每个 Insight 分类正确
4. **粒度一致**：保持 Insight 粒度的一致性

## 输入的 Insight 集合

{insights_json}

## 输出格式

请以 JSON 格式输出合并后的标准 Insight 集合：
```json
{{
  "user_doc_insights": [
    "insight 1",
    "insight 2"
  ],
  "long_context_insights": [
    "insight 1",
    "insight 2"
  ],
  "analysis_insights": [
    "insight 1",
    "insight 2"
  ]
}}
```

请输出合并后的标准 Insight 集合。
"""


def get_llm_client():
    """创建 OpenAI 客户端"""
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def call_llm(client: OpenAI, prompt: str, model: str = "gpt-51-1113-global") -> str:
    """调用 LLM"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def read_markdown_file(filepath: str) -> str:
    """读取 markdown 文件内容"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def extract_query_from_report(content: str) -> str:
    """从报告中提取 Query"""
    lines = content.split('\n')
    in_query_section = False
    query_lines = []
    
    for line in lines:
        if line.strip() == '## Query':
            in_query_section = True
            continue
        if in_query_section:
            if line.startswith('## '):
                break
            query_lines.append(line)
    
    return '\n'.join(query_lines).strip()


def extract_report_content(content: str) -> str:
    """从 markdown 中提取报告内容"""
    lines = content.split('\n')
    in_report_section = False
    report_lines = []
    
    for line in lines:
        if line.strip() == '## Report':
            in_report_section = True
            continue
        if in_report_section:
            if line.startswith('## 参考文献') or line.startswith('## References'):
                break
            report_lines.append(line)
    
    return '\n'.join(report_lines).strip()


def parse_json_from_response(response: str) -> dict:
    """从 LLM 响应中解析 JSON"""
    # 尝试找到 JSON 块
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试直接解析
        json_str = response
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 尝试修复常见问题
        json_str = json_str.strip()
        if not json_str.startswith('{'):
            start = json_str.find('{')
            if start != -1:
                json_str = json_str[start:]
        if not json_str.endswith('}'):
            end = json_str.rfind('}')
            if end != -1:
                json_str = json_str[:end+1]
        return json.loads(json_str)


def extract_insights_from_report(client: OpenAI, content: str, filename: str) -> dict:
    """使用 LLM 从报告中提取 Insight"""
    query = extract_query_from_report(content)
    report = extract_report_content(content)
    
    prompt = EXTRACT_INSIGHT_PROMPT.format(query=query, report=report)
    
    print(f"正在从 {filename} 提取 Insight...")
    response = call_llm(client, prompt)
    
    try:
        insights = parse_json_from_response(response)
        print(f"  - 用户文档 Insight: {len(insights.get('user_doc_insights', []))} 个")
        print(f"  - 检索文档 Insight: {len(insights.get('long_context_insights', []))} 个")
        print(f"  - 综合分析 Insight: {len(insights.get('analysis_insights', []))} 个")
        return insights
    except Exception as e:
        print(f"  - 解析失败: {e}")
        return {
            "user_doc_insights": [],
            "long_context_insights": [],
            "analysis_insights": []
        }


def merge_all_insights(client: OpenAI, all_insights: List[dict]) -> dict:
    """使用 LLM 合并所有 Insight"""
    # 先简单合并
    merged = {
        "user_doc_insights": [],
        "long_context_insights": [],
        "analysis_insights": []
    }
    
    for insights in all_insights:
        merged["user_doc_insights"].extend(insights.get("user_doc_insights", []))
        merged["long_context_insights"].extend(insights.get("long_context_insights", []))
        merged["analysis_insights"].extend(insights.get("analysis_insights", []))
    
    # 使用 LLM 去重和合并
    prompt = MERGE_INSIGHT_PROMPT.format(insights_json=json.dumps(merged, ensure_ascii=False, indent=2))
    
    print("\n正在合并和去重 Insight...")
    response = call_llm(client, prompt)
    
    try:
        final_insights = parse_json_from_response(response)
        return final_insights
    except Exception as e:
        print(f"合并失败: {e}")
        # 返回简单去重的结果
        return {
            "user_doc_insights": list(set(merged["user_doc_insights"])),
            "long_context_insights": list(set(merged["long_context_insights"])),
            "analysis_insights": list(set(merged["analysis_insights"]))
        }


def main():
    """主函数"""
    # 获取当前目录
    current_dir = Path(__file__).parent
    
    # 获取所有 markdown 文件
    md_files = list(current_dir.glob("*.md"))
    print(f"找到 {len(md_files)} 个 markdown 文件")
    
    # 创建 LLM 客户端
    client = get_llm_client()
    
    # 从每个文件提取 insight
    all_insights = []
    insights_by_file = {}
    
    for md_file in md_files:
        content = read_markdown_file(str(md_file))
        insights = extract_insights_from_report(client, content, md_file.name)
        all_insights.append(insights)
        insights_by_file[md_file.name] = insights
    
    # 保存每个文件的 insight
    with open(current_dir / "insights_by_file.json", 'w', encoding='utf-8') as f:
        json.dump(insights_by_file, f, ensure_ascii=False, indent=2)
    print(f"\n每个文件的 Insight 已保存到 insights_by_file.json")
    
    # 合并所有 insight
    final_insights = merge_all_insights(client, all_insights)
    
    # 保存最终的标准 insight 集合
    with open(current_dir / "gold_insights.json", 'w', encoding='utf-8') as f:
        json.dump(final_insights, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("标准 Insight 集合统计：")
    print(f"  - 用户文档 Insight: {len(final_insights.get('user_doc_insights', []))} 个")
    print(f"  - 检索文档 Insight: {len(final_insights.get('long_context_insights', []))} 个")
    print(f"  - 综合分析 Insight: {len(final_insights.get('analysis_insights', []))} 个")
    print(f"  - 总计: {sum(len(v) for v in final_insights.values())} 个")
    print("=" * 50)
    print(f"\n标准 Insight 集合已保存到 gold_insights.json")
    
    # 生成 markdown 格式的报告
    generate_markdown_report(current_dir, final_insights, insights_by_file)


def generate_markdown_report(output_dir: Path, final_insights: dict, insights_by_file: dict):
    """生成 markdown 格式的 insight 报告"""
    report_lines = [
        "# 标准 Insight 集合",
        "",
        "本文档包含从所有模型报告中提取并合并的标准 Insight 集合。",
        "",
        "## 统计信息",
        "",
        f"- 用户文档 Insight: {len(final_insights.get('user_doc_insights', []))} 个",
        f"- 检索文档 Insight: {len(final_insights.get('long_context_insights', []))} 个",
        f"- 综合分析 Insight: {len(final_insights.get('analysis_insights', []))} 个",
        f"- **总计: {sum(len(v) for v in final_insights.values())} 个**",
        "",
        "---",
        "",
        "## 用户文档 Insight",
        "",
        "这些 Insight 来自用户上传的文档（如 PPT）：",
        "",
    ]
    
    for i, insight in enumerate(final_insights.get('user_doc_insights', []), 1):
        report_lines.append(f"{i}. {insight}")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 检索文档 Insight（Long Context）",
        "",
        "这些 Insight 来自外部检索的文档：",
        "",
    ])
    
    for i, insight in enumerate(final_insights.get('long_context_insights', []), 1):
        report_lines.append(f"{i}. {insight}")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 综合分析 Insight",
        "",
        "这些 Insight 需要综合用户文档和检索信息才能得出：",
        "",
    ])
    
    for i, insight in enumerate(final_insights.get('analysis_insights', []), 1):
        report_lines.append(f"{i}. {insight}")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 各模型报告的 Insight 覆盖情况",
        "",
    ])
    
    for filename, insights in insights_by_file.items():
        total = (
            len(insights.get('user_doc_insights', [])) +
            len(insights.get('long_context_insights', [])) +
            len(insights.get('analysis_insights', []))
        )
        report_lines.append(f"- **{filename}**: {total} 个 Insight")
        report_lines.append(f"  - 用户文档: {len(insights.get('user_doc_insights', []))}")
        report_lines.append(f"  - 检索文档: {len(insights.get('long_context_insights', []))}")
        report_lines.append(f"  - 综合分析: {len(insights.get('analysis_insights', []))}")
    
    # 写入文件
    with open(output_dir / "gold_insights.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Markdown 报告已保存到 gold_insights.md")


if __name__ == "__main__":
    main()
