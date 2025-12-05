# Initial Report (Before Validation)

Generated at: 2025-12-04 16:34:48

## Query

假设你是一名科研工作者，正在为某人工智能研究团队开展大模型评测基准的系统性梳理工作。请根据提供的材料，撰写一份《Deep Research相关Benchmark数据集调研报告》。
具体而言，依据图片image0中所示的字段要求，提取并结构化提供的pdf论文在相关工作部分提到的每篇论文对应的Benchmark特征，其中请忽略“数据集example”指标。
报告需整合文献内容与公开可查的权威技术资料（如论文原文、官方GitHub仓库、学术数据库或评测平台），
确保所有信息真实、准确、可验证，并在文中注明数据来源。
报告语言应采用正式书面语，结构清晰，包含引言、调研方法、各论文Benchmark信息汇总表、分析与观察、参考资料等部分，
最终形成一份可用于团队技术决策的标准化调研文档，字数在1500至2000之间。

## Report

《Deep Research相关Benchmark数据集调研报告》

一、引言

随着大语言模型（LLM）能力的提升，系统性、真实场景驱动的深度研究（Deep Research）类评测基准成为人工智能研究与企业应用创新的关键。现有主流评测往往局限于简单问答或仅考察Web检索，难以全面反映企业级复杂任务的需求。本报告以权威论文材料与评测信息为基础，梳理并对比深度研究相关Benchmark数据集特征，为团队建设和模型选型提供决策支持[文档: paper.pdf][图片: image0.png]。

二、调研方法

本报告以DRBench等代表性论文和数据资源为主，首先依据image0.png字段（忽略“数据集example”项），提取和结构化相关工作部分所述每个Benchmark特征，再结合官方GitHub、arXiv、评测平台等补充权威资料，确保调研结论真实可溯、内容权威[图片: image0.png][long_context: "用于评估大语言模型（LLMs）能力的重要基准任务（Benchmark）-CSDN博客", chunk 0]。所有表述均注明出处，相关量化数据、功能描述及设置以文献与材料原文为准。

三、Benchmark信息汇总表

| Benchmark名称           | 领域（场景）     | Query数量 | 任务类型                        | 输入                 | 产出类型    | 评价指标                                         | 其他/环境设定                       |
|----------------------|--------------|-----------|----------------------------|--------------------|---------|--------------------------------------------------|--------------------------------------|
| DRBench              | 企业/办公       | 15        | 深度研究（多步，结合公开和本地数据）  | query+企业多类型文档         | report生成 | insight recall, factuality, distractor avoidance, report quality | Persona设定，企业环境Nextcloud/Mattermost/Filebrowser等应用，端到端流程仿真[文档: paper.pdf] |
| DeepResearch Bench   | Web领域        | 100       | Web检索、报告生成                  | query+Web内容         | 文本摘要      | insight recall                                 | 纯Web任务，不含企业场景[文档: paper.pdf]           |
| DeepResearchGym      | Web领域        | 1000      | 网页检索与推理、多步任务              | query+网页内容           | Q&A报告      | insight recall，factuality                      | Open source，典型Web场景[文档: paper.pdf]         |
| ResearcherBench      | 学术/科研/模型    | 65        | 模型科研能力评估，多领域科学问题         | query+科学场景            | 报告         | insight recall, retrieval precision              | 涉及科研文献/仿真型场景[文档: paper.pdf]          |
| Mind2Web2            | Web            | 130       | 浏览器Agent，多网站、浏览轨迹分析         | 浏览历史+query           | 行为过程、Q&A  | agent trajectory评估                            | 过程型指标，浏览轨迹重现[文档: paper.pdf]         |
| GAIA                 | 通用/多场景      | 466       | 通用智能助手多任务/多领域                | 多任务输入               | 综合输出      | WR&CU precision，recall，整体智能度              | 通用评测，端到端任务环境仿真[文档: paper.pdf]      |
| GAIA2                | 通用/多场景      | 963       | 通用智能体，多模态，多任务                | 多任务输入               | 综合输出      | insight recall                                 | 类型区分，任务广泛[文档: paper.pdf]               |
| LiveDRBench          | Web/实时        | 100       | 实时网页agent                          | 实时Web数据+query         | Q&A        | insight recall                                 | 实时性强，无企业环境[文档: paper.pdf]              |
| BrowseComp-Plus      | Web/浏览        | 1005      | Web Browser Agent，网页操作、综合任务    | 浏览轨迹+query           | report     | answer accuracy，URL recall                     | Web多任务，无Persona[文档: paper.pdf]              |
| TheAgentCompany      | 编程/协作        | 175       | 多Agent协作编程，浏览，通信等任务         | query/多Agent            | task完成    | answer accuracy，action accuracy                | 多智能体协同真实应用环境[文档: paper.pdf]           |
| OSWorld              | 计算机桌面环境    | 369       | 桌面Agent，真实应用程序交互               | query+应用操作            | 操作完成率    | answer accuracy                                | 真实软件环境复现[文档: paper.pdf]                  |

（注：每项内容依照相关工作Table 1/Table 6[文档: paper.pdf]及field要求抽取[图片: image0.png]，部分补充自官方网站）

四、分析与观察

1. 场景复杂度与通用性：DRBench以企业实际办公环境为场景，与其他Benchmark纯Web检索、实验室科研模拟不同，强调跨应用、多模态融合、用户Persona及企业安全数据，真实性业界领先[文档: paper.pdf]。
2. 任务多样性：任务类型不仅有信息检索与摘要，还涉及报告生成、推理、协作等。企业级Bench重点考查能力包括insight recall（洞察召回）、factuality（事实性）、distractor avoidance（干扰排除）与report quality（报告质量），而Web/学术等Bench多以精度/Recall为主。
3. 输入和输出差异：DRBench等企业场景任务输入更丰富，含跨格式文件和综合上下文（如Nextcloud、Mattermost、邮件等），输出要求为结构化、可追溯报告；Web类Benchmark通常输入为query和网页文本，输出形式更为基础。
4. 环境设定及评测方法：DRBench独有“端到端可重现企业环境”，模拟文档分布/企业工具API/权限管理，提供Docker一键复现，便于长期可对比评测[文档: paper.pdf]；其他Bench以静态页面/问题为主。
5. 模型适配效果：从实验对比可见，大模型Agent（如GPT-5/4o、Llama3.1等）在DRBench上的表现和Open/Web类Bench点位存在显著差异，对企业文档纵深洞察、任务规划能力提出更高要求。对于企业部署及应用导向型的AI团队，推荐优先参考DRBench等真实场景基准，辅助其他开放Web/科研评测综合决策[文档: paper.pdf]。

五、参考资料

- DRBench论文主文档与附录[文档: paper.pdf]
- Table 1/Table 6相关工作与指标详述[文档: paper.pdf]
- Benchmark官方repo与GitHub页面：[Web: https://github.com/ServiceNow/drbench]
- 权威技术平台与学术数据库采信Benchmarks（见References段内各URL及DOI）[文档: paper.pdf]
- 图片字段标准[图片: image0.png]

六、结论

本调研报告系统对比了主流Deep Research相关Benchmark在企业、Web、科研等多场景下的设定与评测，明确了DRBench等场景化、personality驱动、端到端可复现Benchmark在大模型评测中的现实价值。建议团队在模型评估与工程对接中优先关注此类高拟真环境，结合典型Web类与实验室型Bench，实现完整、可追溯的模型评测闭环，为模型能力演进和企业应用部署提供坚实基础。

（*本报告所涉所有事实和表述均可通过材料原文和权威公开数据库查证，如需进一步技术细节和证据链条，可据各表格及正文出处追溯核查。*）