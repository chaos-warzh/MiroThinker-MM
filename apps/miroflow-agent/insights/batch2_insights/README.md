# 评估脚本示例文件

本目录包含 `evaluate_with_llm.py` 脚本的示例文件，用于演示正确的文件格式。

## 文件说明

- `example_result.md`: 待评估的报告示例（Markdown格式）
- `example_gold.json`: 标准答案文件示例（包含必须的要点）
- `example_metadata.json`: 元数据文件示例（包含评估要求和配置）
- `cambodia_good_report.md`: 高质量示例报告，满足全部要求
- `cambodia_bad_report.md`: 蓄意包含幻觉、违规引用的报告
- `cambodia_gold.json`: 与上述报告配套的标准答案
- `cambodia_metadata.json`: 与上述报告配套的评估配置
- `cambodia_source/`: 事实核查所需的源文件（需通过 `--source-folder` 指定）

## 使用示例

```bash
# 在 apps/miroflow-agent 目录下运行
uv run python evaluate_with_llm.py \
    --result examples/example_result.md \
    --gold examples/example_gold.json \
    --metadata examples/example_metadata.json \
    --source-folder examples/source_documents \
    --output examples/evaluation_report.txt
```

```
uv run python evaluate_with_llm.py --result examples/example_result.md --gold examples/example_gold.json --metadata examples/example_metadata.json --source-folder examples/source_documents --output examples/evaluation_report.txt
```

### 高低质量报告对比

```bash
# 优秀报告
uv run python evaluate_with_llm.py \
    --result examples/cambodia_good_report.md \
    --gold examples/cambodia_gold.json \
    --metadata examples/cambodia_metadata.json \
    --source-folder examples/cambodia_source \
    --output examples/good_report_eval.txt

# 问题报告
uv run python evaluate_with_llm.py \
    --result examples/cambodia_bad_report.md \
    --gold examples/cambodia_gold.json \
    --metadata examples/cambodia_metadata.json \
    --source-folder examples/cambodia_source \
    --output examples/bad_report_eval.txt
```

## 注意事项

1. **结果文件** (`example_result.md`): 
   - 可以是任何Markdown格式的文本
   - 这是待评估的生成报告

2. **标准答案文件** (`example_gold.json`):
   - 必须包含 `number` 字段（案例编号）
   - 必须包含 `gold_insights` 数组，每个元素包含 `insight` 字段
   - 这些insights是评估信息召回率的依据

3. **元数据文件** (`example_metadata.json`):
   - `language_style`: 语言风格描述（可选）
   - `language_requirements`: 语言要求列表（可选）
   - `character_limit`: 字符数限制 [最小值, 最大值]（可选）
   - `word_limit`: 单词数限制 [最小值, 最大值]（可选）
   - `forbidden_words`: 禁用词列表（可选）
   - 如果某个字段未设置，对应评估会使用默认值

4. **源文档文件夹** (`source_documents/`):
   - 包含用于事实准确性验证的源文档
   - 可以是任何文本文件（.txt, .md, .json等）
   - 可选，如果不提供会使用基础检查方法

## 字段详细说明

### language_style
描述期望的语言风格，例如：
- "正式、专业、客观"
- "简洁明了"
- "学术性"
- "通俗易懂"

### language_requirements
具体的语言要求列表，例如：
- "使用第三人称"
- "避免使用第一人称"
- "使用专业术语"
- "保持客观中立"
- "避免口语化表达"

### character_limit
字符数范围（包括中文字符、英文字母、数字、标点等，去除空白字符）
- 格式: `[最小值, 最大值]`
- 示例: `[500, 1000]` 表示字符数应在500到1000之间

### word_limit
英文单词数范围（按空格分割）
- 格式: `[最小值, 最大值]`
- 示例: `[300, 600]` 表示单词数应在300到600之间

### forbidden_words
报告中不应出现的词汇列表
- 格式: 字符串数组
- 评估时会检查报告中是否包含这些词汇



### 好坏报告评估对比
（坏的报告会编造事实，违反事实，遗漏关键点，内容零散，多用口语化语句，引用的源文档未提及甚至提及相反的信息，违反语言风格，部分信息引用未注明来源等）
信息召回方面：good_report明确涵盖了吴哥窟+三道美食+干季时间等所有要点，而bad_report会编造景点搬迁、没有覆盖指定菜品、推荐雨季到访等

