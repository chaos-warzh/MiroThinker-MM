# Task Report: 001

Generated at: 2025-12-05 00:17:41

## Query

datasets/query.jsonl

## Final Report

# Deep Research相关Benchmark数据集调研报告

## 引言

随着大型语言模型（LLM）技术的快速发展，评估这些模型在复杂、开放式深度研究任务中的能力变得越来越重要。本报告旨在系统性地梳理当前主要的大模型评测基准（Benchmark），特别关注那些用于评估深度研究（Deep Research）能力的数据集。通过对DRBench论文中提及的相关工作进行分析，我们提取并结构化了12个主要的评测基准的特征，以帮助研究团队更好地理解现有评测工具的优缺点，为未来的研究方向提供参考。

## 调研方法

本报告基于ServiceNow Research等机构发表的DRBench论文中的相关工作部分，提取并分析了论文中提到的各个评测基准的关键特征 [paper.pdf, page 3]。我们按照以下字段对每个基准进行了结构化整理：
- 论文名称：基准的官方名称
- 领域/场景：基准适用的主要领域或场景
- query数量：基准包含的查询或任务数量
- 任务：基准评估的主要任务类型
- 输入：基准使用的输入数据类型
- 类型：基准评估的输出或生成类型
- 评价指标：用于评估模型性能的主要指标
- 其他：额外的重要特性，如是否提供环境、是否使用公共和私有数据等

## 各论文Benchmark信息汇总表

| 论文名称 | 领域/场景 | query数量 | 任务 | 输入 | 类型 | 评价指标 | 其他 |
|---------|----------|----------|------|-----|------|---------|------|
| Deep Research Bench | Generic | 89 | Web Research & Computer Use | Web | Answer Generation | Answer Accuracy | Public data only, provides environment |
| DeepResearch Bench | Generic | 100 | Web Research | Web | Report Generation | Insight Recall | Public data only, no environment |
| DeepResearchGym | Generic | 1,000 | Web Research | Web | Document Retrieval | Document Retrieval | Public data only, no environment |
| ResearcherBench | AI | 65 | Web Research | Web | Report Generation | Insight Recall, Factuality | Public data only, no environment |
| LiveDRBench | Generic | 100 | Web Research & Computer Use | Web | Report Generation | Insight Precision, Recall | Public data only, no environment |
| BrowseComp-Plus | Generic | 1,005 | Web Research | Web | Answer Generation | Answer Accuracy, URL Recall | Public data only, no environment |
| Mind2Web 2 | Generic | 130 | Web Research | Web | Task Completion | Partial Completion | Public data only, no environment |
| GAIA | Generic | 466 | Web Research | Web | Answer Generation | Answer Accuracy | Public data only, no environment |
| GAIA2 | Generic | 963 | Computer Use | Computer Environment | Task Completion | Action Accuracy | Public data only, provides environment |
| TheAgentCompany | Enterprise | 175 | Computer Use | Computer Environment | Task Completion | Task Completion, Efficiency | Public data only, provides environment |
| OSWorld | Generic | 369 | Computer Use | Computer Environment | Task Completion | Task Completion | Public data only, provides environment |
| DRBench | Enterprise | 114 | Deep Research | Web + Enterprise Data | Report Generation | Insight Recall | Public & local data, provides environment |

## 分析与观察

### 基准数量与规模

当前共有12个主要的深度研究相关基准 [paper.pdf, page 3]，其中：
- 平均查询数量为381.3个
- 中位数查询数量为152.5个
- 最大规模的基准是BrowseComp-Plus，包含1,005个查询
- 最小规模的基准是ResearcherBench，包含65个查询

### 领域分布

- 大多数基准（9个）面向通用（Generic）领域
- 仅有2个基准（DRBench和TheAgentCompany）专注于企业（Enterprise）场景
- 只有1个基准（ResearcherBench）专注于AI领域

### 任务类型分布

- Web Research是最常见的任务类型，有6个基准专注于此
- Computer Use次之，有3个基准
- 2个基准结合了Web Research和Computer Use
- 只有DRBench明确定位为Deep Research任务 [paper.pdf, page 3]

### 评价指标分析

最常用的评价指标包括：
- Answer Accuracy（3个基准）
- Insight Recall（3个基准）
- Task Completion（2个基准）

其他指标如Document Retrieval、Factuality、URL Recall等各有1个基准采用。

### 环境与数据特性

- 5个基准（占比41.7%）提供了交互式环境
- 只有DRBench（占比8.3%）同时使用了公共数据和本地/私有数据 [paper.pdf, page 3]
- 大多数基准（11个，占比91.7%）仅使用公共数据

## 关键发现

1. **数据来源局限性**：绝大多数基准（91.7%）仅使用公共数据，缺乏对私有/企业数据的评估，这与实际企业应用场景存在差距 [paper.pdf, page 3]。

2. **环境支持不足**：只有41.7%的基准提供交互式环境，限制了对模型在真实环境中交互能力的评估。

3. **企业场景缺乏**：仅有16.7%的基准专注于企业场景，无法全面评估模型在企业环境中的表现。

4. **评价指标多样性**：现有基准使用了11种不同的评价指标，反映了评估标准的多样性，但也增加了跨基准比较的难度。

5. **深度研究定位稀缺**：只有DRBench明确定位为Deep Research任务，表明这一领域的专门评测工具仍然稀缺 [paper.pdf, page 3]。

## 结论与建议

基于本次调研，我们发现当前深度研究评测基准存在明显的局限性，特别是在企业应用场景、私有数据使用和环境交互方面。DRBench作为首个同时使用公共和私有数据、提供企业环境的深度研究基准，填补了这一空白 [paper.pdf, page 3]。

对于未来的研究方向，我们建议：

1. 开发更多专注于企业场景的评测基准，以更好地评估模型在实际业务环境中的表现。

2. 增强对私有/企业数据与公共数据结合使用的评估，以反映真实应用场景。

3. 提供更多交互式环境支持，以评估模型的工具使用和环境适应能力。

4. 统一评价指标体系，以便于跨基准比较和综合评估。

5. 扩展深度研究任务的覆盖范围，包括更多领域和更复杂的查询类型。

## 参考资料

1. Abaskohi, A., Chen, T., Muñoz-Mármol, M., Fox, C., Ramesh, A. V., Marcotte, É., ... & Laradji, I. H. (2025). DRBench: A Realistic Benchmark for Enterprise Deep Research. arXiv:2510.00172v1 [paper.pdf].

2. Bosse, N. I., Evans, J., Gambee, R. G., Hnyk, D., Mühlbacher, P., Phillips, L., ... & Wildman, J. (2025). Deep Research Bench: Evaluating AI Web Research Agents. arXiv:2506.06287 [paper.pdf, page 3].

3. Du, M., Xu, B., Zhu, C., Wang, X., & Mao, Z. (2025). DeepResearch Bench: A comprehensive benchmark for deep research agents. arXiv:2506.11763 [paper.pdf, page 3].

4. Coelho, J., Ning, J., He, J., Mao, K., Paladugu, A., Setlur, P., ... & Martins, B. (2025). DeepResearchGym: A free, transparent, and reproducible evaluation sandbox for deep research. arXiv:2505.19253 [paper.pdf, page 3].

5. Xu, T., Lu, P., Ye, L., Hu, X., & Liu, P. (2025). ResearcherBench: Evaluating deep AI research systems on the frontiers of scientific inquiry. arXiv:2507.16280 [paper.pdf, page 3].

## Summary


============================== Final Answer ==============================
In this conversation, I was asked to create a comprehensive research report on Deep Research benchmarks based on the provided materials. I first analyzed the image to understand the required fields for the benchmark dataset, which included: paper name, domain/scenario, query count, task, input, type, evaluation metrics, and other information (excluding "dataset example" as instructed).

I then examined the paper to identify benchmarks mentioned in the related work section. From the PDF, I extracted information about 12 different benchmarks: Deep Research Bench, DeepResearch Bench, DeepResearchGym, ResearcherBench, LiveDRBench, BrowseComp-Plus, Mind2Web 2, GAIA, GAIA2, TheAgentCompany, OSWorld, and DRBench itself.

I organized this information into a structured table and analyzed the statistics, finding that most benchmarks focus on generic domains, web research tasks, and use only public data. DRBench stands out as the only benchmark that combines both public and local data in an enterprise context.

The final report includes an introduction, methodology, benchmark information summary table, detailed analysis of the benchmarks' characteristics, key findings, conclusions, and recommendations for future research directions.

\boxed{
# Deep Research相关Benchmark数据集调研报告

## 引言

随着大型语言模型（LLM）技术的快速发展，评估这些模型在复杂、开放式深度研究任务中的能力变得越来越重要。本报告旨在系统性地梳理当前主要的大模型评测基准（Benchmark），特别关注那些用于评估深度研究（Deep Research）能力的数据集。通过对DRBench论文中提及的相关工作进行分析，我们提取并结构化了12个主要的评测基准的特征，以帮助研究团队更好地理解现有评测工具的优缺点，为未来的研究方向提供参考。

## 调研方法

本报告基于ServiceNow Research等机构发表的DRBench论文中的相关工作部分，提取并分析了论文中提到的各个评测基准的关键特征 [paper.pdf, page 3]。我们按照以下字段对每个基准进行了结构化整理：
- 论文名称：基准的官方名称
- 领域/场景：基准适用的主要领域或场景
- query数量：基准包含的查询或任务数量
- 任务：基准评估的主要任务类型
- 输入：基准使用的输入数据类型
- 类型：基准评估的输出或生成类型
- 评价指标：用于评估模型性能的主要指标
- 其他：额外的重要特性，如是否提供环境、是否使用公共和私有数据等

## 各论文Benchmark信息汇总表

| 论文名称 | 领域/场景 | query数量 | 任务 | 输入 | 类型 | 评价指标 | 其他 |
|---------|----------|----------|------|-----|------|---------|------|
| Deep Research Bench | Generic | 89 | Web Research & Computer Use | Web | Answer Generation | Answer Accuracy | Public data only, provides environment |
| DeepResearch Bench | Generic | 100 | Web Research | Web | Report Generation | Insight Recall | Public data only, no environment |
| DeepResearchGym | Generic | 1,000 | Web Research | Web | Document Retrieval | Document Retrieval | Public data only, no environment |
| ResearcherBench | AI | 65 | Web Research | Web | Report Generation | Insight Recall, Factuality | Public data only, no environment |
| LiveDRBench | Generic | 100 | Web Research & Computer Use | Web | Report Generation | Insight Precision, Recall | Public data only, no environment |
| BrowseComp-Plus | Generic | 1,005 | Web Research | Web | Answer Generation | Answer Accuracy, URL Recall | Public data only, no environment |
| Mind2Web 2 | Generic | 130 | Web Research | Web | Task Completion | Partial Completion | Public data only, no environment |
| GAIA | Generic | 466 | Web Research | Web | Answer Generation | Answer Accuracy | Public data only, no environment |
| GAIA2 | Generic | 963 | Computer Use | Computer Environment | Task Completion | Action Accuracy | Public data only, provides environment |
| TheAgentCompany | Enterprise | 175 | Computer Use | Computer Environment | Task Completion | Task Completion, Efficiency | Public data only, provides environment |
| OSWorld | Generic | 369 | Computer Use | Computer Environment | Task Completion | Task Completion | Public data only, provides environment |
| DRBench | Enterprise | 114 | Deep Research | Web + Enterprise Data | Report Generation | Insight Recall | Public & local data, provides environment |

## 分析与观察

### 基准数量与规模

当前共有12个主要的深度研究相关基准 [paper.pdf, page 3]，其中：
- 平均查询数量为381.3个
- 中位数查询数量为152.5个
- 最大规模的基准是BrowseComp-Plus，包含1,005个查询
- 最小规模的基准是ResearcherBench，包含65个查询

### 领域分布

- 大多数基准（9个）面向通用（Generic）领域
- 仅有2个基准（DRBench和TheAgentCompany）专注于企业（Enterprise）场景
- 只有1个基准（ResearcherBench）专注于AI领域

### 任务类型分布

- Web Research是最常见的任务类型，有6个基准专注于此
- Computer Use次之，有3个基准
- 2个基准结合了Web Research和Computer Use
- 只有DRBench明确定位为Deep Research任务 [paper.pdf, page 3]

### 评价指标分析

最常用的评价指标包括：
- Answer Accuracy（3个基准）
- Insight Recall（3个基准）
- Task Completion（2个基准）

其他指标如Document Retrieval、Factuality、URL Recall等各有1个基准采用。

### 环境与数据特性

- 5个基准（占比41.7%）提供了交互式环境
- 只有DRBench（占比8.3%）同时使用了公共数据和本地/私有数据 [paper.pdf, page 3]
- 大多数基准（11个，占比91.7%）仅使用公共数据

## 关键发现

1. **数据来源局限性**：绝大多数基准（91.7%）仅使用公共数据，缺乏对私有/企业数据的评估，这与实际企业应用场景存在差距 [paper.pdf, page 3]。

2. **环境支持不足**：只有41.7%的基准提供交互式环境，限制了对模型在真实环境中交互能力的评估。

3. **企业场景缺乏**：仅有16.7%的基准专注于企业场景，无法全面评估模型在企业环境中的表现。

4. **评价指标多样性**：现有基准使用了11种不同的评价指标，反映了评估标准的多样性，但也增加了跨基准比较的难度。

5. **深度研究定位稀缺**：只有DRBench明确定位为Deep Research任务，表明这一领域的专门评测工具仍然稀缺 [paper.pdf, page 3]。

## 结论与建议

基于本次调研，我们发现当前深度研究评测基准存在明显的局限性，特别是在企业应用场景、私有数据使用和环境交互方面。DRBench作为首个同时使用公共和私有数据、提供企业环境的深度研究基准，填补了这一空白 [paper.pdf, page 3]。

对于未来的研究方向，我们建议：

1. 开发更多专注于企业场景的评测基准，以更好地评估模型在实际业务环境中的表现。

2. 增强对私有/企业数据与公共数据结合使用的评估，以反映真实应用场景。

3. 提供更多交互式环境支持，以评估模型的工具使用和环境适应能力。

4. 统一评价指标体系，以便于跨基准比较和综合评估。

5. 扩展深度研究任务的覆盖范围，包括更多领域和更复杂的查询类型。

## 参考资料

1. Abaskohi, A., Chen, T., Muñoz-Mármol, M., Fox, C., Ramesh, A. V., Marcotte, É., ... & Laradji, I. H. (2025). DRBench: A Realistic Benchmark for Enterprise Deep Research. arXiv:2510.00172v1 [paper.pdf].

2. Bosse, N. I., Evans, J., Gambee, R. G., Hnyk, D., Mühlbacher, P., Phillips, L., ... & Wildman, J. (2025). Deep Research Bench: Evaluating AI Web Research Agents. arXiv:2506.06287 [paper.pdf, page 3].

3. Du, M., Xu, B., Zhu, C., Wang, X., & Mao, Z. (2025). DeepResearch Bench: A comprehensive benchmark for deep research agents. arXiv:2506.11763 [paper.pdf, page 3].

4. Coelho, J., Ning, J., He, J., Mao, K., Paladugu, A., Setlur, P., ... & Martins, B. (2025). DeepResearchGym: A free, transparent, and reproducible evaluation sandbox for deep research. arXiv:2505.19253 [paper.pdf, page 3].

5. Xu, T., Lu, P., Ye, L., Hu, X., & Liu, P. (2025). ResearcherBench: Evaluating deep AI research systems on the frontiers of scientific inquiry. arXiv:2507.16280 [paper.pdf, page 3].
}

-------------------- Extracted Result --------------------
# Deep Research相关Benchmark数据集调研报告

## 引言

随着大型语言模型（LLM）技术的快速发展，评估这些模型在复杂、开放式深度研究任务中的能力变得越来越重要。本报告旨在系统性地梳理当前主要的大模型评测基准（Benchmark），特别关注那些用于评估深度研究（Deep Research）能力的数据集。通过对DRBench论文中提及的相关工作进行分析，我们提取并结构化了12个主要的评测基准的特征，以帮助研究团队更好地理解现有评测工具的优缺点，为未来的研究方向提供参考。

## 调研方法

本报告基于ServiceNow Research等机构发表的DRBench论文中的相关工作部分，提取并分析了论文中提到的各个评测基准的关键特征 [paper.pdf, page 3]。我们按照以下字段对每个基准进行了结构化整理：
- 论文名称：基准的官方名称
- 领域/场景：基准适用的主要领域或场景
- query数量：基准包含的查询或任务数量
- 任务：基准评估的主要任务类型
- 输入：基准使用的输入数据类型
- 类型：基准评估的输出或生成类型
- 评价指标：用于评估模型性能的主要指标
- 其他：额外的重要特性，如是否提供环境、是否使用公共和私有数据等

## 各论文Benchmark信息汇总表

| 论文名称 | 领域/场景 | query数量 | 任务 | 输入 | 类型 | 评价指标 | 其他 |
|---------|----------|----------|------|-----|------|---------|------|
| Deep Research Bench | Generic | 89 | Web Research & Computer Use | Web | Answer Generation | Answer Accuracy | Public data only, provides environment |
| DeepResearch Bench | Generic | 100 | Web Research | Web | Report Generation | Insight Recall | Public data only, no environment |
| DeepResearchGym | Generic | 1,000 | Web Research | Web | Document Retrieval | Document Retrieval | Public data only, no environment |
| ResearcherBench | AI | 65 | Web Research | Web | Report Generation | Insight Recall, Factuality | Public data only, no environment |
| LiveDRBench | Generic | 100 | Web Research & Computer Use | Web | Report Generation | Insight Precision, Recall | Public data only, no environment |
| BrowseComp-Plus | Generic | 1,005 | Web Research | Web | Answer Generation | Answer Accuracy, URL Recall | Public data only, no environment |
| Mind2Web 2 | Generic | 130 | Web Research | Web | Task Completion | Partial Completion | Public data only, no environment |
| GAIA | Generic | 466 | Web Research | Web | Answer Generation | Answer Accuracy | Public data only, no environment |
| GAIA2 | Generic | 963 | Computer Use | Computer Environment | Task Completion | Action Accuracy | Public data only, provides environment |
| TheAgentCompany | Enterprise | 175 | Computer Use | Computer Environment | Task Completion | Task Completion, Efficiency | Public data only, provides environment |
| OSWorld | Generic | 369 | Computer Use | Computer Environment | Task Completion | Task Completion | Public data only, provides environment |
| DRBench | Enterprise | 114 | Deep Research | Web + Enterprise Data | Report Generation | Insight Recall | Public & local data, provides environment |

## 分析与观察

### 基准数量与规模

当前共有12个主要的深度研究相关基准 [paper.pdf, page 3]，其中：
- 平均查询数量为381.3个
- 中位数查询数量为152.5个
- 最大规模的基准是BrowseComp-Plus，包含1,005个查询
- 最小规模的基准是ResearcherBench，包含65个查询

### 领域分布

- 大多数基准（9个）面向通用（Generic）领域
- 仅有2个基准（DRBench和TheAgentCompany）专注于企业（Enterprise）场景
- 只有1个基准（ResearcherBench）专注于AI领域

### 任务类型分布

- Web Research是最常见的任务类型，有6个基准专注于此
- Computer Use次之，有3个基准
- 2个基准结合了Web Research和Computer Use
- 只有DRBench明确定位为Deep Research任务 [paper.pdf, page 3]

### 评价指标分析

最常用的评价指标包括：
- Answer Accuracy（3个基准）
- Insight Recall（3个基准）
- Task Completion（2个基准）

其他指标如Document Retrieval、Factuality、URL Recall等各有1个基准采用。

### 环境与数据特性

- 5个基准（占比41.7%）提供了交互式环境
- 只有DRBench（占比8.3%）同时使用了公共数据和本地/私有数据 [paper.pdf, page 3]
- 大多数基准（11个，占比91.7%）仅使用公共数据

## 关键发现

1. **数据来源局限性**：绝大多数基准（91.7%）仅使用公共数据，缺乏对私有/企业数据的评估，这与实际企业应用场景存在差距 [paper.pdf, page 3]。

2. **环境支持不足**：只有41.7%的基准提供交互式环境，限制了对模型在真实环境中交互能力的评估。

3. **企业场景缺乏**：仅有16.7%的基准专注于企业场景，无法全面评估模型在企业环境中的表现。

4. **评价指标多样性**：现有基准使用了11种不同的评价指标，反映了评估标准的多样性，但也增加了跨基准比较的难度。

5. **深度研究定位稀缺**：只有DRBench明确定位为Deep Research任务，表明这一领域的专门评测工具仍然稀缺 [paper.pdf, page 3]。

## 结论与建议

基于本次调研，我们发现当前深度研究评测基准存在明显的局限性，特别是在企业应用场景、私有数据使用和环境交互方面。DRBench作为首个同时使用公共和私有数据、提供企业环境的深度研究基准，填补了这一空白 [paper.pdf, page 3]。

对于未来的研究方向，我们建议：

1. 开发更多专注于企业场景的评测基准，以更好地评估模型在实际业务环境中的表现。

2. 增强对私有/企业数据与公共数据结合使用的评估，以反映真实应用场景。

3. 提供更多交互式环境支持，以评估模型的工具使用和环境适应能力。

4. 统一评价指标体系，以便于跨基准比较和综合评估。

5. 扩展深度研究任务的覆盖范围，包括更多领域和更复杂的查询类型。

## 参考资料

1. Abaskohi, A., Chen, T., Muñoz-Mármol, M., Fox, C., Ramesh, A. V., Marcotte, É., ... & Laradji, I. H. (2025). DRBench: A Realistic Benchmark for Enterprise Deep Research. arXiv:2510.00172v1 [paper.pdf].

2. Bosse, N. I., Evans, J., Gambee, R. G., Hnyk, D., Mühlbacher, P., Phillips, L., ... & Wildman, J. (2025). Deep Research Bench: Evaluating AI Web Research Agents. arXiv:2506.06287 [paper.pdf, page 3].

3. Du, M., Xu, B., Zhu, C., Wang, X., & Mao, Z. (2025). DeepResearch Bench: A comprehensive benchmark for deep research agents. arXiv:2506.11763 [paper.pdf, page 3].

4. Coelho, J., Ning, J., He, J., Mao, K., Paladugu, A., Setlur, P., ... & Martins, B. (2025). DeepResearchGym: A free, transparent, and reproducible evaluation sandbox for deep research. arXiv:2505.19253 [paper.pdf, page 3].

5. Xu, T., Lu, P., Ye, L., Hu, X., & Liu, P. (2025). ResearcherBench: Evaluating deep AI research systems on the frontiers of scientific inquiry. arXiv:2507.16280 [paper.pdf, page 3].

-------------------- Token Usage --------------------
Total Input Tokens: 1283840
Total Cache Input Tokens: 0
Total Output Tokens: 14962
-----------------------------------------------------
Pricing is disabled - no cost information available
-----------------------------------------------------