# 滑动窗口注意力（SWA）

此奖励材料说明了使用滑动窗口注意力 (SWA) 相对于常规多头注意力 (MHA) 时所节省的内存。



 
## 简介

什么是滑动窗口注意力（SWA）？如果我们将常规自注意力视为“全局”注意力机制，因为每个序列元素都可以访问其他每个序列元素，那么我们可以将 SWA 视为“局部”注意力，因为这里我们限制当前查询位置周围的上下文大小。如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

如上图所示，每个令牌只关注其位置周围的固定大小的本地窗口，而不是关注所有先前的令牌。这种局部注意力大大降低了 KV 缓存的大小。

在本介绍的其余部分中，我们将在 [Gemma 3](https://arxiv.org/abs/2503.19786) 的背景下讨论 SWA，它是在 [../../ch05/12_gemma3](../../ch05/12_gemma3) 中从头开始实现的。

滑动窗口注意力最初是在 [LongFormer 2020 年论文](https://arxiv.org/abs/2004.05150) 中介绍的，但我们关注 Google 的 Gemma 模型的原因是它们是非常好的开放权重模型，表明滑动窗口注意力确实是最近有能力的模型中的一种可行方法。

[Gemma 2](https://arxiv.org/abs/2408.00118) 使用了一种混合方法，以 1:1 的比例组合了局部（滑动窗口）和全局注意力层。每个令牌可以参与 4 k 个令牌的上下文窗口。这种 1:1 混合的原因是它在效率和全局上下文建模之间取得了平衡，因为仅使用局部注意力的大语言模型可能过于严格。

[Gemma 3](https://arxiv.org/abs/2503.19786) 随后将设计进一步提高了效率。它使用滑动窗口和全注意力层之间的 5:1 比例，这意味着每 5 个局部注意力层就有一个全局层。此外，滑动窗口大小从 Gemma 2 中的 4096 个令牌减少到 Gemma 3 中的 1024 个令牌。

有趣的是，Gemma 3 技术报告中的消融研究表明，这些变化对整体模型质量仅产生很小的影响。换句话说，通过滑动窗口注意力实现的大量内存和计算节省，同时建模性能的损失最小。



 
## 滑动窗口注意力（SWA）的内存节省

内存节省主要体现在 KV 存储上。我们可以通过以下公式计算 KV 存储大小：

字节 ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

当使用SWA时，我们用窗口大小W替换上面的序列长度（seqlen）。因此，当使用滑动窗口注意力时，我们将KV缓存大小减少“W / seqlen”因子。 （请注意，为了简单起见，这假设每一层都使用滑动窗口注意力。）


您可以使用此文件夹中的 [memory_estimator_swa.py](memory_estimator_swa.py) 脚本将其应用于不同的模型配置，以查看通过使用 SWA 而非 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
GQA + SWA (Ratio: 5:1) : 0.78 GB
```

请注意，Gemma 3 将 SWA 与 GQA 结合使用。

下图进一步显示了使用 SWA 而非 MHA 时针对不同上下文长度所节省的成本：

 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" />

 

您可以通过以下方式重现该图：

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


 
## SWA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_swa.py](gpt_with_kv_swa.py) 脚本提供了在 GPT 模型实现的上下文中比较 MHA 和 SWA 内存使用情况的实践示例。

请注意，SWA 还可以与 MLA 和 GQA 结合使用（如前所述），但为了简单起见，这里没有这样做。

请注意，该模型未经训练，因此会生成无意义的文本。但是，您可以使用它作为第 5-7 章中标准 GPT 模型的直接替代品并对其进行训练。

此外，此实现使用[另一个奖励部分](../03_kv-cache)中解释的 KV 缓存，因此内存节省更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # like Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

我们没有看到上图中如此大的节省的原因有两个：

1. 我使用较小的配置让模型在合理的时间内完成生成。
2. 更重要的是，我们在这里看的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（但这是一个单独分析的主题）。
