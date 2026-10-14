# 第 4 章：从头开始实现 GPT 模型来生成文本

 
## 主要章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主要章节代码。

 
## 奖励材料

- [02_performance-analysis](02_performance-analysis) 包含分析主要章节中实现的 GPT 模型性能的可选代码
- [03_kv-cache](03_kv-cache) 实现 KV 缓存以加速推理过程中的文本生成
- [07_moe](07_moe) 专家混合 (MoE) 的解释和实施
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) 包含将 GPT 架构实现转换为 Llama 3.2 并从 Meta AI 加载预训练权重的分步指南（完成第 4 章后查看替代架构可能会很有趣，但您也可以在阅读第 5 章后将其保存）


 
## 注意替代方案

 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

 

- [04_gqa](04_gqa) 包含对分组查询注意 (GQA) 的介绍，大多数现代 LLM（Llama 4、gpt-oss、Qwen3、Gemma 3 等）都使用它作为常规多头注意 (MHA) 的替代方案
- [05_mla](05_mla) 包含对多头潜在注意 (MLA) 的介绍，DeepSeek V3 使用它作为常规多头注意 (MHA) 的替代方案
- [06_swa](06_swa) 包含滑动窗口注意（SWA）的介绍，由 Gemma 3 和其他人使用
- [08_deltanet](08_deltanet) 对门控 DeltaNet 作为流行的线性注意力变体的解释（用于 Qwen3-Next 和 Kimi Linear）
- [10_kv-sharing](10_kv-sharing) 包含跨层 KV 共享的介绍，Gemma 4 E2B 和 E4B 使用它来减少 KV 缓存内存


 
## 更多

在下面的视频中，我提供了一个代码会话，其中涵盖了一些章节内容作为补充材料。

<br>
<br>

[![视频链接](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)