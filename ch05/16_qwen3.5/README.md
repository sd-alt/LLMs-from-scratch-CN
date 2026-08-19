# Qwen3.5 0.8B 从头开始

此文件夹包含 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 的从头开始样式实现。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/03.webp">

Qwen3.5 基于 Qwen3-Next 架构，我在我的 [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms) 文章的 [2.（线性）注意力混合](https://magazine.sebastianraschka.com/i/177848019/2-linear-attention-hybrids) 部分对此进行了更详细的描述

<a href="https://magazine.sebastianraschka.com/p/beyond-standard-llms"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/02.webp" width="500px"></a>

请注意，Qwen3.5 交替使用 `linear_attention` 和 `full_attention` 层。
这些笔记本保持完整模型流程的可读性，同时重用 [qwen3_5_transformers.py](qwen3_5_transformers.py) 中的线性注意构建块，其中包含来自 Hugging Face 的 Apache 2.0 版开源许可下的线性注意代码。

 
## 文件

- [qwen3.5.ipynb](qwen3.5.ipynb)：主要Qwen3.5 0.8B笔记本实现。
- [qwen3.5-plus-kv-cache.ipynb](qwen3.5-plus-kv-cache.ipynb)：与 KV 缓存解码相同的模型，以提高效率。
- [qwen3_5_transformers.py](qwen3_5_transformers.py)：来自 Hugging Face Transformers 的一些辅助组件，用于 Qwen3.5 线性注意力。