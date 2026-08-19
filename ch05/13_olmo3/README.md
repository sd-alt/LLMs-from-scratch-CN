# Olmo 3 7B 和 32B 从头开始

此文件夹中的 [standalone-olmo3.ipynb](standalone-olmo3.ipynb) Jupyter 笔记本包含 Olmo 3 7B 和 32B 的从头开始实现，需要大约 13 GB 的 RAM 才能运行。

替代方案 [standalone-olmo3-plus-kvcache.ipynb](standalone-olmo3-plus-kv-cache.ipynb) 笔记本添加了 KV 缓存，以实现更好的运行时性能（但增加了更多代码复杂性）。要了解有关 KV 缓存的更多信息，请参阅我的 [从头开始在 LLM 中理解和编码 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 文章。

下面是与Qwen3作为参考模型的并排比较；如果您对Qwen3 0.6B独立笔记本感兴趣，可以在这里找到[](../11_qwen3)。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-7B.webp?1">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-32B.webp?1">

Olmo 3 也有不同的风格，如下所示（架构相同，只是训练管道不同）：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-pipeline.webp?1">


 
## Olmo 3 与 Qwen3 相比如何

本节重点关注架构，而不是训练细节，与 Qwen3 进行简要比较。


7B型号：

1. 从上图可以看出，Olmo 3架构与Qwen3比较相似。然而，值得注意的是，这本质上很可能是受到 Olmo 2 前身的启发，而不是 Qwen3。

2）与 Olmo 2 类似，Olmo 3 仍然使用 post-norm 风格而不是 pre-norm，因为他们在 Olmo 2 论文中发现它可以稳定训练。

3）有趣的是，7B模型仍然使用类似于Olmo 2的多头注意力。
然而，为了提高效率并减少 KV 缓存大小，他们现在使用滑动窗口注意力（例如，类似于 Gemma 3）。

接下来是32B型号：

4) 总体而言，它是相同的架构，但只是进行了扩展。此外，比例（例如，从输入到前馈层中的中间大小等）与 Qwen3 中的比例大致匹配。

5) 我的猜测是，由于词汇量较小，该架构最初比 Qwen3 稍小，然后他们将中间大小扩展从 Qwen3 中的 5 倍扩大到 Olmo 3 中的 5.4，以获得 32B 模型进行直接比较。

6) 另外，请注意 32B 模型（最后！）使用分组查询注意力。





<br>要了解有关架构差异的更多信息并阅读与其他架构的比较，请参阅我的 [大 LLM 架构比较：从 DeepSeek-V3 到 Kimi K2：现代 LLM 架构设计](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) 文章。