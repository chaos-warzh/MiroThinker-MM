# 中间过程评测方案

## 背景分析

在当前的 miroflow-agent 架构中，存在以下特点：
1. **无干扰信息**：与老师提到的场景不同，当前设置中没有故意添加干扰信息
2. **Memory 只增不减**：信息是累积的，不会被过滤或删除
3. **多层级结构**：Main Agent → Sub-agents (browsing, coding, reading, reasoning)

## 可评测的中间过程

### 1. Sub-agent Summary 质量评测

**评测对象**：每个 sub-agent 生成的 `final_answer_text`

**评测维度**：
- **信息完整性 (Completeness)**：sub-agent 是否提取了所有相关信息
- **信息准确性 (Accuracy)**：提取的信息是否正确
- **信息相关性 (Relevance)**：提取的信息是否与 subtask 相关
- **引用正确性 (Citation Accuracy)**：引用的来源是否正确

**评测方法**：
```python
# 为每个 subtask 准备 ground truth
{
    "subtask": "搜索北京旅游景点",
    "expected_facts": [
        {"fact": "故宫是明清两代皇宫", "source_type": "web"},
        {"fact": "长城是世界文化遗产", "source_type": "web"}
    ],
    "expected_sources": ["故宫官网", "长城官网"]
}
```

### 2. 信息传递完整性评测

**评测对象**：从 sub-agent summary 到 main agent final report 的信息传递

**评测维度**：
- **信息保留率 (Retention Rate)**：sub-agent 提取的信息有多少被保留到最终报告
- **信息失真率 (Distortion Rate)**：信息在传递过程中是否发生变化
- **信息整合质量 (Integration Quality)**：多个 sub-agent 的信息是否被正确整合

**评测方法**：
```python
# 追踪信息流
{
    "sub_agent_facts": {
        "agent-browsing_1": ["fact1", "fact2"],
        "agent-browsing_2": ["fact3", "fact4"]
    },
    "final_report_facts": ["fact1", "fact2", "fact3"],  # fact4 丢失
    "retention_rate": 0.75,
    "distortion_cases": []
}
```

### 3. 工具调用效率评测

**评测对象**：`tool_calls_data` 中的工具调用记录

**评测维度**：
- **调用必要性 (Necessity)**：每次工具调用是否必要
- **调用效率 (Efficiency)**：是否有重复或冗余的调用
- **调用顺序合理性 (Order Rationality)**：调用顺序是否合理

**评测方法**：
```python
# 分析工具调用模式
{
    "total_calls": 15,
    "unique_queries": 12,
    "duplicate_calls": 3,
    "efficiency_score": 0.8,
    "unnecessary_calls": ["call_id_5", "call_id_8"]
}
```

### 4. 推理链质量评测

**评测对象**：`message_history` 中的推理过程

**评测维度**：
- **推理连贯性 (Coherence)**：推理步骤之间是否连贯
- **推理正确性 (Correctness)**：每一步推理是否正确
- **推理效率 (Efficiency)**：是否有不必要的推理步骤

**评测方法**：
```python
# 提取推理链
{
    "reasoning_steps": [
        {"step": 1, "action": "分析任务", "correct": True},
        {"step": 2, "action": "搜索信息", "correct": True},
        {"step": 3, "action": "整合结果", "correct": False, "error": "遗漏了关键信息"}
    ],
    "coherence_score": 0.9,
    "correctness_score": 0.67
}
```

### 5. 验证循环效果评测

**评测对象**：Report Validation 循环中的修改

**评测维度**：
- **问题发现率 (Issue Detection Rate)**：验证是否发现了真正的问题
- **修复成功率 (Fix Success Rate)**：发现的问题是否被正确修复
- **过度修改率 (Over-modification Rate)**：是否有不必要的修改

**评测方法**：
```python
# 对比验证前后
{
    "original_issues": ["字数不足", "缺少引用"],
    "detected_issues": ["字数不足"],  # 漏检了"缺少引用"
    "fixed_issues": ["字数不足"],
    "introduced_issues": ["添加了虚假URL"],  # 修复时引入新问题
    "detection_rate": 0.5,
    "fix_rate": 1.0,
    "regression_rate": 0.5
}
```

## 实现建议

### 方案 A：基于 Ground Truth 的评测

为每个测试用例准备详细的中间过程 ground truth：

```json
{
    "task_id": "004",
    "query": "...",
    "intermediate_ground_truth": {
        "expected_subtasks": [
            {
                "subtask_description": "搜索北京景点",
                "expected_facts": [...],
                "expected_sources": [...]
            }
        ],
        "expected_reasoning_steps": [...],
        "expected_tool_calls": [...]
    },
    "final_ground_truth": "..."
}
```

**优点**：评测精确
**缺点**：标注成本高

### 方案 B：基于 LLM 的自动评测

使用 LLM 作为评判者，评估中间过程质量：

```python
def evaluate_sub_agent_summary(subtask, summary, original_sources):
    prompt = f"""
    评估以下 sub-agent 的总结质量：
    
    子任务：{subtask}
    总结：{summary}
    原始来源：{original_sources}
    
    请从以下维度评分（1-5分）：
    1. 信息完整性
    2. 信息准确性
    3. 信息相关性
    4. 引用正确性
    
    并给出具体的问题和建议。
    """
    return llm_evaluate(prompt)
```

**优点**：自动化程度高
**缺点**：评测结果可能不稳定

### 方案 C：基于规则的自动评测

定义可量化的规则进行评测：

```python
def evaluate_information_flow():
    # 1. 提取 sub-agent 中的关键实体
    sub_agent_entities = extract_entities(sub_agent_summaries)
    
    # 2. 提取最终报告中的关键实体
    final_entities = extract_entities(final_report)
    
    # 3. 计算保留率
    retention_rate = len(sub_agent_entities & final_entities) / len(sub_agent_entities)
    
    # 4. 检查引用一致性
    citation_consistency = check_citation_consistency(sub_agent_summaries, final_report)
    
    return {
        "retention_rate": retention_rate,
        "citation_consistency": citation_consistency
    }
```

**优点**：结果稳定、可复现
**缺点**：可能遗漏一些复杂的质量问题

## 推荐实施路径

### 第一阶段：信息流追踪

1. 在 `TaskLog` 中添加信息流追踪字段
2. 记录每个 sub-agent 提取的关键信息
3. 追踪这些信息在最终报告中的出现情况

### 第二阶段：自动化评测指标

1. 实现信息保留率计算
2. 实现引用一致性检查
3. 实现工具调用效率分析

### 第三阶段：LLM 辅助评测

1. 设计评测 prompt
2. 实现 LLM 评测流程
3. 与人工评测结果对比校准

## 具体实现代码示例

### 1. 扩展 TaskLog 以支持信息流追踪

```python
@dataclass
class IntermediateEvaluation:
    """中间过程评测数据"""
    # Sub-agent 信息提取
    sub_agent_extracted_facts: Dict[str, List[str]] = field(default_factory=dict)
    
    # 信息流追踪
    fact_retention: Dict[str, bool] = field(default_factory=dict)
    
    # 工具调用分析
    tool_call_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # 推理链分析
    reasoning_chain_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # 验证循环分析
    validation_analysis: Dict[str, Any] = field(default_factory=dict)
```

### 2. 信息提取函数

```python
def extract_key_facts(text: str) -> List[Dict[str, Any]]:
    """从文本中提取关键事实"""
    # 使用 NER 或 LLM 提取关键实体和事实
    pass

def track_fact_flow(sub_agent_facts: Dict[str, List[str]], 
                    final_report: str) -> Dict[str, bool]:
    """追踪事实从 sub-agent 到最终报告的流动"""
    retention = {}
    for agent_id, facts in sub_agent_facts.items():
        for fact in facts:
            # 检查事实是否出现在最终报告中
            retention[f"{agent_id}:{fact}"] = fact in final_report
    return retention
```

### 3. 评测报告生成

```python
def generate_intermediate_evaluation_report(task_log: TaskLog) -> Dict[str, Any]:
    """生成中间过程评测报告"""
    return {
        "task_id": task_log.task_id,
        "sub_agent_performance": evaluate_sub_agents(task_log),
        "information_flow": evaluate_information_flow(task_log),
        "tool_efficiency": evaluate_tool_efficiency(task_log),
        "reasoning_quality": evaluate_reasoning_chain(task_log),
        "validation_effectiveness": evaluate_validation_loop(task_log)
    }
```

## 与老师提到的 Memory 评测的对比

老师提到的场景：
- 给 agent 很多干扰信息
- 评测 memory 是否能过滤掉干扰信息
- 最终 memory 应该只包含有用信息

当前场景的差异：
- 没有故意添加干扰信息
- Memory 是只增不减的
- 需要评测的是信息的**完整性**和**准确性**，而不是**过滤能力**

可以考虑的改进方向：
1. **添加干扰信息测试**：在测试用例中故意添加干扰信息，评测 agent 的过滤能力
2. **实现 Memory 压缩**：在 context 接近上限时，实现智能的 memory 压缩，只保留关键信息
3. **评测 Memory 质量**：即使没有干扰信息，也可以评测 memory 中信息的质量和组织方式
