# RAG Rerank + Summary 方案

## 背景

当前 RAG 检索存在以下问题：
1. **干扰信息**: Embedding 相似度高不代表内容真正相关，可能检索到语义相似但实际无关的内容
2. **信息截断**: 长文本 chunk 可能被截断，关键信息可能在截断部分
3. **噪声累积**: 多次检索的结果直接拼接到上下文，噪声会累积影响最终报告质量

## 解决方案

在 RAG 检索后添加两个后处理步骤：

### 1. Rerank (重排序)

使用 LLM 对检索结果进行相关性评分，过滤掉不相关的内容。

**流程**:
```
原始检索结果 (10个 chunks)
    ↓
LLM 评分 (0-10分)
    ↓
过滤低分结果 (threshold=0.6)
    ↓
精选结果 (5个 chunks)
```

**评分标准**:
- 10分：完全相关，直接回答查询，包含关键信息
- 7-9分：高度相关，包含有用信息
- 4-6分：部分相关，信息价值有限
- 1-3分：略微相关，提到相关词汇但内容不相关
- 0分：完全不相关或干扰信息

### 2. Summary (摘要)

对每个 chunk 提取关键信息，避免截断问题。

**输出格式**:
```json
{
    "key_facts": ["事实1", "事实2"],
    "summary": "精炼的摘要内容",
    "relevance_note": "与查询的关联说明"
}
```

## 实现架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG MCP Server                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  rag_search(query, json_path)                                   │
│       ↓                                                          │
│  RAGTool.search() / diverse_search()                            │
│       ↓                                                          │
│  原始检索结果 (10 chunks)                                        │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ RAGPostProcessor (if enabled)                               ││
│  │                                                              ││
│  │  1. RAGReranker.rerank()                                    ││
│  │     - LLM 评分每个 chunk                                     ││
│  │     - 过滤低分结果                                           ││
│  │     - 返回 top_n 个高相关结果                                ││
│  │                                                              ││
│  │  2. RAGSummarizer.summarize_chunks()                        ││
│  │     - 提取每个 chunk 的关键事实                              ││
│  │     - 生成精炼摘要                                           ││
│  │     - 添加相关性说明                                         ││
│  │                                                              ││
│  │  3. format_results()                                        ││
│  │     - 格式化输出，包含引用信息                               ││
│  └─────────────────────────────────────────────────────────────┘│
│       ↓                                                          │
│  精炼结果返回给 Main Agent                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
MiroThinker-MM/libs/miroflow-tools/src/miroflow_tools/
├── tools/
│   ├── rag_tool.py          # 原有 RAG 工具
│   └── rag_rerank.py        # 新增: Rerank + Summary 模块
│       ├── RAGReranker      # LLM-based 重排序
│       ├── RAGSummarizer    # LLM-based 摘要
│       └── RAGPostProcessor # 组合处理器
└── mcp_servers/
    └── rag_mcp_server.py    # 集成 post-processor
```

## 配置选项

配置在 `src/config/settings.py` 中定义（默认值），可通过环境变量覆盖：

```python
# ==================== RAG Post-Processing Configuration ====================
# Enable LLM-based reranking to filter noise from RAG results
RAG_ENABLE_RERANK = os.environ.get("RAG_ENABLE_RERANK", "0") == "1"

# Enable LLM-based summarization to extract key facts from each chunk
RAG_ENABLE_SUMMARY = os.environ.get("RAG_ENABLE_SUMMARY", "0") == "1"

# Model for reranking (uses OPENAI_API_KEY and OPENAI_BASE_URL)
RAG_RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "gpt-4o-mini")

# Model for summarization (uses OPENAI_API_KEY and OPENAI_BASE_URL)
RAG_SUMMARY_MODEL = os.environ.get("RAG_SUMMARY_MODEL", "gpt-4o-mini")

# Maximum number of chunks to keep after reranking
RAG_RERANK_TOP_N = int(os.environ.get("RAG_RERANK_TOP_N", "5"))

# Minimum relevance score (0-1) to keep a chunk after reranking
RAG_RELEVANCE_THRESHOLD = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.6"))

# Maximum length of each chunk summary
RAG_MAX_SUMMARY_LENGTH = int(os.environ.get("RAG_MAX_SUMMARY_LENGTH", "500"))
```

## 使用方式

### 启用 Rerank + Summary

```bash
# 通过环境变量设置
export RAG_ENABLE_RERANK=1
export RAG_ENABLE_SUMMARY=1
```

### 只启用 Rerank

```bash
export RAG_ENABLE_RERANK=1
export RAG_ENABLE_SUMMARY=0
```

### 只启用 Summary

```bash
export RAG_ENABLE_RERANK=0
export RAG_ENABLE_SUMMARY=1
```

## 输出格式对比

### 原始输出 (无后处理)

```
=== RAG Search Results ===
Query: '北京旅游景点'
Results Found: 10

Result 1
Citation: [long_context: "北京旅游指南", chunk 5]
Relevance Score: 0.892
Title: 北京旅游指南
--- Content ---
故宫是明清两代的皇家宫殿，位于北京中轴线的中心...
[... 可能包含大量不相关内容 ...]
```

### 增强输出 (启用 Rerank + Summary)

```
=== RAG Search Results (Enhanced) ===
Query: '北京旅游景点'
Results Found: 5

Result 1
Citation: [long_context: "北京旅游指南", chunk 5]
Relevance Score: 0.90
Relevance Reason: 直接包含北京主要景点的详细介绍

Title: 北京旅游指南

--- Key Facts ---
• 故宫是明清两代皇家宫殿，世界文化遗产
• 占地72万平方米，有9999间房屋
• 门票价格：旺季60元，淡季40元
• 开放时间：8:30-17:00

--- Content ---
故宫是北京最著名的景点之一，建于明永乐年间...

Relevance: 该片段详细介绍了故宫的历史、规模和参观信息，与查询高度相关
```

## 性能考虑

### 额外 API 调用

| 功能 | API 调用次数 | 预估 Token |
|------|-------------|-----------|
| Rerank | 1次 | ~2000 tokens |
| Summary | 1次 | ~4000 tokens |
| 总计 | 2次 | ~6000 tokens |

### 延迟影响

- Rerank: +1-2秒
- Summary: +2-3秒
- 总计: +3-5秒

### 优化建议

1. **使用快速模型**: 默认使用 `gpt-4o-mini`，速度快且成本低
2. **批量处理**: Rerank 和 Summary 都使用批量 prompt，减少 API 调用
3. **可选启用**: 默认关闭，只在需要高质量结果时启用

## 测试

### 启用后运行测试

```bash
# 设置环境变量
export RAG_ENABLE_RERANK=1
export RAG_ENABLE_SUMMARY=1

# 运行测试
cd MiroThinker-MM/apps/miroflow-agent
uv run python run_folder_task.py --folder data/bench_case1202/001
```

### 对比测试

1. 关闭后处理，运行任务，记录结果
2. 启用后处理，运行相同任务，对比结果
3. 检查：
   - 检索结果是否更精准
   - 最终报告是否减少了干扰信息
   - 引用是否更准确

## 后续优化

1. **缓存 Rerank 结果**: 相同 query + chunks 可以缓存评分结果
2. **自适应阈值**: 根据检索结果质量动态调整阈值
3. **多模型支持**: 支持使用不同模型进行 Rerank 和 Summary
4. **流式处理**: 支持流式返回结果，减少等待时间
