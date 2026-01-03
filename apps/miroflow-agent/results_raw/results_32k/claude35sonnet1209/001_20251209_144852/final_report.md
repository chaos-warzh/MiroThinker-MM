# Final Report (After Validation)

Generated at: 2025-12-09 14:52:20

## Query

假设你是一名科研工作者，正在为某人工智能研究团队开展大模型评测基准的系统性梳理工作。请根据提供的材料，撰写一份《Deep Research相关Benchmark数据集调研报告》。
具体而言，依据图片image0中所示的字段要求，提取并结构化提供的pdf论文在相关工作部分提到的每篇论文对应的Benchmark特征，其中请忽略“数据集example”指标。
报告语言应采用正式书面语，结构清晰，包含引言、调研方法、各论文Benchmark信息汇总表、分析与观察、参考资料等部分，
最终形成一份可用于团队技术决策的标准化调研文档，字数在1500至2000之间。

## Report

# Deep Research相关Benchmark数据集调研报告

## 1. 引言

随着人工智能技术的快速发展，大模型在企业环境中的应用日益广泛。为了评估这些模型在复杂、开放式深度研究任务中的表现，我们需要一个全面而实际的基准测试。本报告旨在调研和分析最新的Deep Research相关Benchmark数据集，特别关注DRBench及其相关工作。[paper.pdf]

深度研究（Deep Research）是一项涉及高层次战略问题提出、子问题规划、相关材料检索与评估，以及生成清晰、可操作的数据源支持的摘要的复杂任务。[paper.pdf] 这种任务通常由领域专家使用搜索引擎、通信平台和业务应用程序的组合来执行，需要大量的人力投入。[paper.pdf] 因此，开发能够准确评估AI代理在这些任务中表现的基准测试至关重要。

## 2. 调研方法

本次调研主要基于以下方法：

1. 文献综述：分析DRBench论文中提到的相关工作，重点关注各Benchmark的特点和评估方法。[paper.pdf]
2. 数据提取：根据论文中提供的信息，提取各Benchmark的关键特征，包括领域、query数量、任务类型、输入形式、评价指标等。[paper.pdf]
3. 比较分析：对比不同Benchmark的特点，总结其优势和局限性，特别是在企业深度研究任务评估方面的表现。[paper.pdf]

## 3. Benchmark信息汇总表

| 论文名称 | 领域（场景） | query数量 | 任务 | 输入 | 类型 | 评价指标 | 其他 |
|---------|-------------|-----------|------|------|------|----------|------|
| DRBench [paper.pdf] | 企业 | 15 | 深度研究 | query文本 + 文档 | 报告生成 | Insight Recall, Factuality, Report Quality | 114个洞察，10个领域，商业角色设定 |
| Deep Research Bench (Bosse et al., 2025) [paper.pdf] | 通用 | 89 | 网络研究 | - | - | - | 仅网络数据 |
| DeepResearch Bench (Du et al., 2025) [paper.pdf] | 通用 | 100 | 网络研究 | - | - | - | 仅网络数据 |
| DeepResearchGym (Coelho et al., 2025) [paper.pdf] | 通用 | 1,000 | 网络研究 | - | - | Insight Recall, Factuality | 仅网络数据 |
| ResearcherBench (Xu et al., 2025b) [paper.pdf] | 通用 | 65 | 网络研究和计算机使用 | - | - | Insight Precision, Recall | 包含计算机使用任务 |
| LiveDRBench (Java et al., 2025) [paper.pdf] | 通用 | 100 | 网络研究和计算机使用 | - | - | Answer Accuracy | 包含计算机使用任务 |
| BrowseComp-Plus (Chen et al., 2025) [paper.pdf] | 通用 | 1,005 | 网络研究 | - | - | Partial Completion | 仅网络数据 |
| Mind2Web 2 (Gou et al., 2025) [paper.pdf] | 通用 | 130 | 网络研究 | - | - | Answer Accuracy | 仅网络数据 |
| GAIA (Mialon et al., 2024) [paper.pdf] | 通用 | 466 | 网络研究 | - | - | Action Accuracy | 仅网络数据 |
| GAIA2 (Andrews et al., 2025) [paper.pdf] | 通用 | 963 | 网络研究 | - | - | - | 仅网络数据 |

## 4. 分析与观察

1. 任务多样性：
   DRBench相比其他Benchmark，特别关注企业环境下的深度研究任务，这更贴近实际应用场景。[paper.pdf] 它涵盖了10个企业领域，包括销售、网络安全和合规等，体现了其在实际商业环境中的广泛适用性。[paper.pdf] 相比之下，其他Benchmark主要集中在通用网络研究或简单的计算机使用任务上，缺乏企业特定的上下文。

2. 数据来源：
   DRBench是唯一一个结合公共网络数据和企业内部数据的Benchmark，这更接近真实的企业研究环境。[paper.pdf] 它的搜索空间包括生产力软件、云文件系统、电子邮件、聊天对话和开放网络，反映了企业数据的异构性。[paper.pdf] 其他Benchmark如Deep Research Bench和DeepResearch Bench仅限于网络数据，无法全面评估AI代理在企业环境中的表现。

3. 评估指标：
   DRBench引入了更全面的评估指标，包括Insight Recall、Factuality和Report Quality，这有助于全面评估AI代理的性能。[paper.pdf] 这些指标不仅评估了代理检索相关洞察的能力，还考虑了生成报告的事实准确性和整体质量。相比之下，其他Benchmark如DeepResearchGym和ResearcherBench主要关注洞察召回率和准确性，缺乏对报告质量的评估。

4. 任务复杂度：
   DRBench的任务涉及多步骤、长期规划的研究问题，这对AI代理的能力提出了更高的要求。[paper.pdf] 例如，"我们应该如何调整产品路线图以确保符合这一标准？"这样的问题需要代理能够理解复杂的上下文，并从多个来源综合信息。相比之下，其他Benchmark如LiveDRBench和BrowseComp-Plus可能更侧重于单一步骤的任务或简单的网页浏览。

5. 真实性和可重现性：
   DRBench通过精心设计的合成管道生成任务，并进行人工验证，确保了任务的真实性和可重现性。[paper.pdf] 这种方法平衡了真实企业数据的隐私保护需求和创建标准化、可重复的基准测试的需要。其他Benchmark可能缺乏这种平衡，要么过于简化，要么难以在不同研究中重现。

6. 规模和多样性：
   虽然DRBench的query数量（15个）相对较少，但它包含了114个需要提取的洞察点，涵盖了10个不同的企业领域。[paper.pdf] 这种深度和广度的结合使得DRBench能够更全面地评估AI代理的能力。相比之下，如DeepResearchGym虽然有1,000个query，但可能缺乏DRBench那样的深度和领域多样性。

7. 企业角色设定：
   DRBench独特地将每个任务与特定的企业角色（如监管事务经理）相关联，这增加了任务的真实性和相关性。[paper.pdf] 这种设置要求AI代理不仅要检索信息，还要考虑特定角色的需求和视角，这是其他Benchmark所缺乏的。

## 5. 结论

DRBench作为一个专门针对企业深度研究任务的基准测试，在任务多样性、数据来源、评估指标和任务复杂度等方面都显示出了显著的优势。它不仅考虑了企业环境的特殊性，还通过结合公共和私有数据源，以及引入多维度的评估指标，为AI代理的评估提供了更全面和真实的场景。

然而，DRBench的query数量相对较少，这可能限制了其在某些方面的代表性。未来的研究可以考虑扩大任务规模，同时保持其深度和多样性。此外，随着企业环境和AI技术的不断发展，DRBench也需要持续更新和完善，以保持其相关性和有效性。

总的来说，DRBench为评估和推动企业环境下的深度研究AI代理的发展提供了一个强有力的工具，它的出现填补了现有基准测试在企业应用评估方面的空白，为未来的研究和开发指明了方向。

## 6. 参考资料

1. [paper.pdf] DRBench: A REALISTIC BENCHMARK FOR ENTERPRISE DEEP RESEARCH
2. [Image: image0.png] Benchmark分析字段图表