<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/68525b342230a897a65cc1c0/87mYQ_a-4jpnMkVR4hrgm.png" width="55%" alt="MiroThinker" />
</div>
<!-- <hr> -->
<div align="center">

[![Demo](https://img.shields.io/badge/Demo-FFB300?style=for-the-badge&logo=airplayvideo&logoColor=white)](https://dr.miromind.ai/)
[![Models](https://img.shields.io/badge/Models-5EDDD2?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/collections/miromind-ai/mirothinker-v01-689301b6d0563321862d44a1)
[![Data](https://img.shields.io/badge/Data-0040A1?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1)
[![Blog](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://miromind.ai/blog/miromind-open-deep-research)

[![Github](https://img.shields.io/badge/GitHub-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MiroMindAI/MiroThinker)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/EprKHYcm)
[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://cdn-uploads.huggingface.co/production/uploads/68525b342230a897a65cc1c0/SGK70isvVpeJwk_fny9sb.png)
[![RedNote](https://img.shields.io/badge/RedNote-FF2442?style=for-the-badge&logo=revoltdotchat&logoColor=white)](https://www.xiaohongshu.com/user/profile/663098830000000003033edc)
[![Website](https://img.shields.io/badge/Website-4285F4?style=for-the-badge&logo=monster&logoColor=white)](https://miromind.ai/)

</div>

## Introduction

MiroThinker is an open-source agentic model series built on top of Qwen3. Designed for deep research and complex, long-horizon problem solving, it integrates strong capabilities in task decomposition, multi-hop reasoning, retrieval-augmented generation, code execution, web browsing, and document/file processing, making it suitable for a wide range of real-world applications.

We have released the MiroThinker-v0.1 series, including both SFT and DPO variants at parameter scales of 8B, 14B, and 32B. Notably, MiroThinker v0.1 achieves state-of-the-art performance among open-source models on the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA), a rigorous evaluation suite for advanced agentic capabilities, demonstrating its strength in long-context, decision-intensive, and real-world task scenarios.

## Performance 

### GAIA Benchmark

| **Method** | Text-103<br>Best Pass@1 | Text-103<br>Pass@1 (Avg@8) | Val-165<br>Best Pass@1 | Val-165<br>Pass@1 (Avg@8) |
| ----------------------------------------------------------------- | :--: | :--: | :--: | :--: |
| Search-o1-7B                                                      | 17.5 | -    | -    | -    |
| R1-Searcher-7B                                                    | 20.4 | -    | -    | -    |
| WebDancer-7B                                                      | 31.0 | -    | -    | -    |
| WebSailor-7B                                                      | 37.9 | -    | -    | -    |
| CK-Pro-8B                                                         | 40.3 | -    | 32.7 | -    |
| MiroThinker-8B-SFT-v0.1                                           | 44.7 | 40.1 | 34.6 | 31.8 |
| &nbsp;&nbsp;&nbsp;&nbsp;+ Commercial Tools                        | 46.6 | 42.1 | 37.6 | 33.9 |
| MiroThinker-8B-DPO-v0.1                                           | 46.6 | 44.8 | 37.0 | 35.4 |
| &nbsp;&nbsp;&nbsp;&nbsp;+ Commercial Tools                        | 50.5 | 46.7 | 38.2 | 35.9 |
|                                                                   |      |      |      |      |
| Search-o1-32B                                                     | 28.2 | -    | -    | -    |
| WebThinker-32B-RL                                                 | 48.5 | -    | -    | -    |
| WebDancer-QwQ-32B                                                 | 51.5 | -    | -    | -    |
| WebSailor-32B                                                     | 53.2 | -    | -    | -    |
| WebShaper-QwQ-32B                                                 | 53.3 | -    | -    | -    |
| WebShaper-72B                                                     | 60.1 | -    | -    | -    |
| MiroThinker-14B-SFT-v0.1                                          | 47.6 | 44.4 | 37.0 | 34.4 |
| &nbsp;&nbsp;&nbsp;&nbsp;+ Commercial Tools                        | 49.5 | 47.5 | 41.8 | 39.8 |
| MiroThinker-14B-DPO-v0.1                                          | 48.5 | 46.6 | 42.4 | 39.2 |
| &nbsp;&nbsp;&nbsp;&nbsp;+ Commercial Tools                        | 52.4 | 48.5 | 45.5 | 42.0 |
| MiroThinker-32B-SFT-v0.1                                          | 55.3 | 51.3 | 44.9 | 42.7 |
| &nbsp;&nbsp;&nbsp;&nbsp;+ Commercial Tools                        | 58.3 | 54.2 | 48.5 | 45.8 |
| <span style="white-space:nowrap;">MiroThinker-32B-DPO-v0.1</span> | 57.3 | 54.1 | 48.5 | 45.9 |
| &nbsp;&nbsp;&nbsp;&nbsp;+ Commercial Tools                        | **60.2** | **57.9** | **50.9** | **48.9** |

1. Following the practices of WebThinker, WebAgents, and CognitiveKernel, we report the Best Pass@1, the highest score across three runs, which often reflects stronger performance, though it may exhibit some variability. To provide a more stable measure, we additionally report Pass@1 (Avg@8), which offers greater consistency at the cost of slightly lower scores.

2. For consistency with prior open-source works, we evaluate GAIA-Text-103 using the WebAgents LLM-as-judge template, and report results on GAIA-Val-165 using the official GAIA scorer script.

3. By default, we use open-source tools wherever possible, except for the code tool [E2B](https://github.com/e2b-dev/E2B) and the Google search tool [Serper](https://serper.dev/). We use [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct), and [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) in our implementation. The framework can be easily extended to other open-source tools of your choice.

4. Replacing these open-source tools with commercial alternatives can yield performance gains. Commercial tools were mainly used for multimodal capabilities and certain complex reasoning subtasks. The majority of tasks, including planning, browsing, refinement, navigation, and more, were handled by our models.

### More Benchmarks

Coming soon

## Quick Start

MiroThinker-v0.1 is trained on our large-scale, high-quality trajectory and preference datasets [MiroVerse-v0.1](https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1), utilizing the efficient training framework [MiroTrain](https://github.com/MiroMindAI/MiroTrain), and enhanced with tool-use capabilities through our agentic framework [MiroFlow](https://github.com/MiroMindAI/MiroFlow). 

To promote reproducibility and benefit the community, we decided to open-source the entire suite mentioned above. For more technical details, evaluation results, and usage tutorials, please visit our [technical blog](https://miromind.ai/blog/miromind-open-deep-research).

## License

MiroThinker-v0.1 is licensed under Apache 2.0.

## Contact Us

MiroThinker is developed by the MiroMind Foundation Model Team.
If you would like to leave us a message, feel free to get in touch. 
In addition to [GitHub](https://github.com/MiroMindAI/), 
[Discord](https://discord.com/invite/EprKHYcm), 
[WeChat](https://cdn-uploads.huggingface.co/production/uploads/68525b342230a897a65cc1c0/SGK70isvVpeJwk_fny9sb.png), 
and [RedNote](https://www.xiaohongshu.com/user/profile/663098830000000003033edc), 
you can also reach us via email at talent@miromind.ai.
