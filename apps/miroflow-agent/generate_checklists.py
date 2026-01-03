#!/usr/bin/env python3
"""
从 Query 中提取 Checklist 脚本

根据 query 的要求，提取出需要满足的 checklist 项目。
不固定数量，根据 query 的实际要求来提取。

Usage:
    # 生成 checklist（指定输出目录）
    uv run python generate_checklists.py generate \
        --query-file deep_research_bench_query.jsonl \
        --output-dir checklists/batch2_checklists
    
    # 列出已生成的 checklist
    uv run python generate_checklists.py list \
        --checklist-dir checklists/batch2_checklists \
        --verbose
    
    # 导出为 Markdown
    uv run python generate_checklists.py export \
        --checklist-dir checklists/batch2_checklists \
        --output checklists/batch2_checklists/README.md
    
    # 强制重新生成
    uv run python generate_checklists.py generate \
        --query-file deep_research_bench_query.jsonl \
        --output-dir checklists/batch2_checklists \
        --force
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件
load_dotenv()


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class Config:
    """配置 - 使用阿里云 API"""
    api_key: str
    base_url: str
    model_name: str = "gpt-51-1113-global"
    temperature: float = 0.1
    
    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"))


@dataclass
class ChecklistItem:
    """单个 Checklist 项"""
    id: int
    requirement: str
    category: str = "content"  # content, format, style, etc.


# =============================================================================
# LLM 客户端
# =============================================================================

class LLMClient:
    """LLM 客户端"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.call_count = 0
    
    def call(self, system: str, user: str) -> Optional[dict]:
        self.call_count += 1
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=self.config.temperature,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"  ⚠️ LLM 调用错误: {e}")
            return None


# =============================================================================
# Checklist 管理器
# =============================================================================

class ChecklistManager:
    """Checklist 管理器 - 按 case_id 存储在子文件夹中"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_case_dir(self, case_id: str) -> Path:
        """获取 case 目录"""
        return self.base_dir / case_id
    
    def get_checklist_path(self, case_id: str) -> Path:
        """获取 checklist 文件路径"""
        return self.get_case_dir(case_id) / "checklist.json"
    
    def load(self, case_id: str) -> Optional[List[ChecklistItem]]:
        """加载 checklist"""
        checklist_path = self.get_checklist_path(case_id)
        if not checklist_path.exists():
            return None
        
        try:
            data = json.loads(checklist_path.read_text(encoding='utf-8'))
            items = []
            for item_data in data.get('checklist', []):
                items.append(ChecklistItem(
                    id=item_data['id'],
                    requirement=item_data['requirement'],
                    category=item_data.get('category', 'content')
                ))
            return items if items else None
        except Exception:
            return None
    
    def save(self, case_id: str, checklist: List[ChecklistItem], query: str) -> None:
        """保存 checklist"""
        case_dir = self.get_case_dir(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        
        checklist_path = self.get_checklist_path(case_id)
        data = {
            'case_id': case_id,
            'query': query,
            'generated_at': datetime.now().isoformat(),
            'checklist': [
                {
                    'id': item.id,
                    'requirement': item.requirement,
                    'category': item.category
                }
                for item in checklist
            ]
        }
        checklist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


# =============================================================================
# 生成 Checklist
# =============================================================================

def generate_checklists(args):
    """从 query 中提取 checklist"""
    config = Config.from_env()
    
    # 加载 query 文件
    query_path = Path(args.query_file)
    if not query_path.exists():
        print(f"❌ query 文件不存在: {query_path}")
        return
    
    # 从 query 文件路径提取数据集名称
    # 例如: datasets_batch2/query.jsonl -> datasets_batch2
    dataset_name = query_path.parent.name if query_path.parent.name else query_path.stem
    
    # 构建输出目录: checklists/{数据集名称}/
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    queries = []
    for line in query_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            data = json.loads(line)
            qid = data.get('id') or data.get('number') or data.get('task')
            if qid:
                qid_str = f"{int(qid):03d}" if str(qid).isdigit() else str(qid)
                queries.append({'case_id': qid_str, 'query': data.get('query', '')})
    
    print(f"📋 找到 {len(queries)} 个查询，开始提取 checklist...")
    print(f"📁 数据集: {dataset_name}")
    print(f"📁 输出目录: {output_dir}")
    print()
    
    llm = LLMClient(config)
    manager = ChecklistManager(output_dir)
    
    generated = 0
    skipped = 0
    failed = 0
    
    for i, q in enumerate(queries):
        case_id = q['case_id']
        query_text = q['query']
        
        # 检查是否已有缓存
        existing = manager.load(case_id)
        if existing and not args.force:
            print(f"  [{i+1}/{len(queries)}] Case {case_id}: ⏭️ 已存在（{len(existing)} 项）")
            skipped += 1
            continue
        
        print(f"  [{i+1}/{len(queries)}] Case {case_id}: 提取中...")
        
        # 提取 checklist
        prompt = f"""从以下查询中提取核心要求，尽量精简合并。

查询：
{query_text}

规则：
1. 只提取查询中明确提到的核心要求
2. 相关的要求合并为一条（如"分析A、B、C三个方面"合并为一条）
3. 不要过度拆分，一件事情只用一条描述
4. 通常一个查询只需要3-8条核心要求
5. 分类：content（内容）、format（格式）、style（风格）、scope（范围）

输出 JSON：
{{
    "checklist": [
        {{"id": 1, "requirement": "要求描述", "category": "content"}},
        ...
    ]
}}"""
        
        result = llm.call("你是一个专业的需求分析专家，擅长从用户查询中提取明确的要求。", prompt)
        
        if not result or "checklist" not in result:
            print(f"    ❌ 提取失败")
            failed += 1
            continue
        
        # 解析并保存
        items = []
        for item_data in result["checklist"]:
            try:
                items.append(ChecklistItem(
                    id=item_data["id"],
                    requirement=item_data["requirement"],
                    category=item_data.get("category", "content")
                ))
            except (KeyError, ValueError):
                continue
        
        if not items:
            print(f"    ❌ 解析失败")
            failed += 1
            continue
        
        manager.save(case_id, items, query_text)
        print(f"    ✅ 提取成功（{len(items)} 项）")
        generated += 1
    
    print()
    print("=" * 50)
    print(f"📊 提取完成:")
    print(f"   ✅ 新生成: {generated}")
    print(f"   ⏭️ 已跳过: {skipped}")
    print(f"   ❌ 失败: {failed}")
    print(f"   🔢 LLM 调用: {llm.call_count} 次")
    print("=" * 50)


# =============================================================================
# 列出 Checklist
# =============================================================================

def list_checklists(args):
    """列出已生成的 checklist"""
    checklist_dir = Path(args.checklist_dir)
    
    if not checklist_dir.exists():
        print(f"❌ 目录不存在: {checklist_dir}")
        return
    
    # 查找所有 case 目录
    case_dirs = sorted([d for d in checklist_dir.iterdir() if d.is_dir() and (d / "checklist.json").exists()])
    
    if not case_dirs:
        print("📋 没有找到任何 checklist")
        return
    
    print(f"📋 找到 {len(case_dirs)} 个 checklist:")
    print()
    
    for case_dir in case_dirs:
        checklist_path = case_dir / "checklist.json"
        try:
            data = json.loads(checklist_path.read_text(encoding='utf-8'))
            case_id = data.get('case_id', case_dir.name)
            items = data.get('checklist', [])
            generated_at = data.get('generated_at', '?')
            query = data.get('query', '')[:80]
            
            print(f"  Case {case_id}: {len(items)} 项 (生成于 {generated_at})")
            
            if args.verbose:
                print(f"    Query: {query}...")
                for item in items:
                    category_icon = {
                        "content": "📝",
                        "format": "📋",
                        "style": "🎨",
                        "scope": "🔍"
                    }.get(item.get('category', ''), "⚪")
                    req = item.get('requirement', '')
                    if len(req) > 60:
                        req = req[:60] + "..."
                    print(f"      {category_icon} [{item.get('category', 'content')}] {req}")
                print()
        except Exception as e:
            print(f"  ⚠️ {case_dir.name}: 读取失败 ({e})")


# =============================================================================
# 导出 Checklist 为 Markdown
# =============================================================================

def export_checklists(args):
    """导出 checklist 为 Markdown 格式"""
    checklist_dir = Path(args.checklist_dir)
    
    if not checklist_dir.exists():
        print(f"❌ 目录不存在: {checklist_dir}")
        return
    
    # 查找所有 case 目录
    case_dirs = sorted([d for d in checklist_dir.iterdir() if d.is_dir() and (d / "checklist.json").exists()])
    
    if not case_dirs:
        print("📋 没有找到任何 checklist")
        return
    
    output_path = Path(args.output) if args.output else checklist_dir / "README.md"
    
    lines = [
        "# Checklist 汇总",
        "",
        f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**总计**: {len(case_dirs)} 个案例",
        "",
        "---",
        "",
    ]
    
    for case_dir in case_dirs:
        checklist_path = case_dir / "checklist.json"
        try:
            data = json.loads(checklist_path.read_text(encoding='utf-8'))
            case_id = data.get('case_id', case_dir.name)
            items = data.get('checklist', [])
            query = data.get('query', '')
            
            lines.extend([
                f"## Case {case_id}",
                "",
                f"**Query**: {query}",
                "",
                "| # | 分类 | 要求 |",
                "|---|------|------|",
            ])
            
            for item in items:
                category_icon = {
                    "content": "📝",
                    "format": "📋",
                    "style": "🎨",
                    "scope": "🔍"
                }.get(item.get('category', ''), "⚪")
                lines.append(
                    f"| {item.get('id', '?')} | {category_icon} {item.get('category', 'content')} | "
                    f"{item.get('requirement', '')} |"
                )
            
            lines.extend(["", "---", ""])
            
        except Exception as e:
            lines.append(f"⚠️ Case {case_dir.name}: 读取失败 ({e})")
            lines.append("")
    
    output_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"✅ 已导出到: {output_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从 Query 中提取 Checklist")
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 生成子命令
    gen_parser = subparsers.add_parser('generate', help='从 query 中提取 checklist')
    gen_parser.add_argument("--query-file", type=str, required=True, help="query.jsonl 文件路径")
    gen_parser.add_argument("--output-dir", type=str, required=True, help="checklist 输出目录（如 checklists/batch2_checklists）")
    gen_parser.add_argument("--force", action="store_true", help="强制重新生成已存在的 checklist")
    
    # 列出子命令
    list_parser = subparsers.add_parser('list', help='列出已生成的 checklist')
    list_parser.add_argument("--checklist-dir", type=str, required=True, help="checklist 目录")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    # 导出子命令
    export_parser = subparsers.add_parser('export', help='导出 checklist 为 Markdown')
    export_parser.add_argument("--checklist-dir", type=str, required=True, help="checklist 目录")
    export_parser.add_argument("--output", "-o", type=str, help="输出文件路径（默认为目录下的 README.md）")
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_checklists(args)
    elif args.command == 'list':
        list_checklists(args)
    elif args.command == 'export':
        export_checklists(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
