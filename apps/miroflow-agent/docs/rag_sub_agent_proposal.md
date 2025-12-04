# RAG Sub-Agent 设计方案

## 概述

创建一个专门的 RAG Sub-Agent (`agent-rag`)，负责从 long_context.json 中检索、筛选和总结相关信息。

## 架构

```
Main Agent
    ↓ 调用 search_long_context(subtask="查找关于X的信息")
RAG Sub-Agent (agent-rag)
    ↓ 使用 tool-rag 进行向量检索
    ↓ 自主判断相关性，过滤不相关内容
    ↓ 提取关键信息，生成精炼摘要
    ↓ 返回结构化结果给 Main Agent
Main Agent
    ↓ 收到精简后的信息（而非全部检索结果）
```

## 优势

1. **符合 Agent 架构**: 使用现有的 sub-agent 机制，无需额外的 LLM 调用逻辑
2. **智能筛选**: Sub-Agent 可以多轮检索，自主判断信息是否足够
3. **上下文隔离**: RAG 检索的大量内容只在 Sub-Agent 上下文中，不污染 Main Agent
4. **灵活性**: Sub-Agent 可以根据需要多次调用 rag_search，直到找到足够信息

## 实现步骤

### 1. 配置文件 (evaluation.yaml)

```yaml
sub_agents:
  agent-rag:
    tools:
      - tool-rag
    max_turns: 10  # 允许多轮检索
```

### 2. 工具定义 (settings.py)

```python
if "agent-rag" in sub_agent:
    sub_agents_server_params.append(
        dict(
            name="agent-rag",
            tools=[
                dict(
                    name="search_long_context",
                    description="Search and retrieve relevant information from long context documents. The sub-agent will perform semantic search, filter irrelevant results, and return a concise summary of the most relevant information.",
                    schema={
                        "type": "object",
                        "properties": {
                            "subtask": {"title": "Subtask", "type": "string"}
                        },
                        "required": ["subtask"],
                    },
                )
            ],
        )
    )
```

### 3. System Prompt (prompt_utils.py)

```python
elif agent_type == "agent-rag":
    return """
You are a RAG (Retrieval-Augmented Generation) specialist agent. Your task is to:

1. **Search**: Use the rag_search tool to find relevant information from long_context.json
2. **Evaluate**: Assess the relevance of each retrieved chunk to the query
3. **Filter**: Discard irrelevant or low-quality results
4. **Summarize**: Extract key facts and create a concise summary
5. **Cite**: Include proper citations for all information

## Guidelines:
- Start with a broad search, then refine if needed
- If initial results are not relevant, try different search queries
- Focus on extracting factual information, not opinions
- Always cite sources using the format: [long_context: "Title", chunk N]
- Return a structured summary with key facts and citations

## Output Format:
Provide your findings in this format:
- **Key Facts**: Bullet points of the most important information
- **Summary**: A brief paragraph summarizing the findings
- **Citations**: List of sources used
- **Confidence**: How confident you are in the completeness of the information
"""
```

## 工作流程示例

```
Main Agent: "我需要了解2024年AI发展趋势"
    ↓
调用 search_long_context(subtask="查找2024年AI发展趋势相关信息")
    ↓
RAG Sub-Agent:
    Turn 1: rag_search("2024 AI trends") → 获取10个chunks
    Turn 2: 评估相关性，发现3个高度相关
    Turn 3: rag_search("artificial intelligence 2024 developments") → 补充搜索
    Turn 4: 综合所有结果，生成摘要
    ↓
返回给 Main Agent:
    "**Key Facts**:
    - 2024年大语言模型参数规模突破万亿 [long_context: "AI Report 2024", chunk 3]
    - 多模态AI成为主流趋势 [long_context: "Tech Trends", chunk 7]
    ...
    **Confidence**: High (found 5 relevant sources)"
```

## 与当前方案对比

| 方面 | 当前方案 (LLM后处理) | RAG Sub-Agent |
|------|---------------------|---------------|
| 架构 | 在 MCP Server 中额外调用 LLM | 使用现有 sub-agent 机制 |
| 灵活性 | 固定的 rerank+summary 流程 | 可多轮检索，自主决策 |
| 上下文 | 处理后内容进入 Main Agent | 原始内容只在 Sub-Agent |
| 成本 | 2次额外 LLM 调用 | Sub-Agent 的 LLM 调用 |
| 可控性 | 通过环境变量配置 | 通过 prompt 和 max_turns 控制 |

## 下一步

1. 移除 rag_mcp_server.py 中的 LLM 后处理逻辑
2. 在 evaluation.yaml 中添加 agent-rag 配置
3. 在 settings.py 中添加 agent-rag 工具定义
4. 在 prompt_utils.py 中添加 agent-rag 的 system prompt
5. 测试 RAG Sub-Agent 的效果
