# 分组查询注意力（GQA）

此奖励材料说明了使用分组查询注意力 (GQA) 相对于常规多头注意力 (MHA) 时所节省的内存。

 
## 简介

近年来，分组查询注意力（GQA）已成为多头注意力（MHA）的计算和参数效率更高的新标准替代品。请注意，这并不是什么新鲜事，可以追溯到 2023 年 [GQA：从多头检查点](https://arxiv.org/abs/2305.13245) 训练通用多查询变压器模型。甚至老式 Llama 2 系列中的较大型号也使用了它。

以下是 GQA 的简短摘要。与 MHA 不同，MHA 中每个头也有自己的一组键和值，为了减少内存使用，GQA 将多个头分组以共享相同的键和值投影。

例如，如下图所示，如果有 3 个键值组和 6 个注意力头，则头 1 和 2 共享一组键和值，而头 3 和 4 以及头 5 和 6 分别共享另一组键和值。

 

![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

 

这种键和值的共享减少了键和值计算的总数，从而降低了内存使用量并提高了效率。

因此，总而言之，GQA 背后的核心思想是通过在多个查询头之间共享键和值头来减少它们的数量。这 (1) 减少了模型的参数数量，(2) 减少了推理过程中键和值张量的内存带宽使用，因为需要在 KV 缓存中存储和检索的键和值更少。

虽然 GQA 主要是 MHA 的计算效率解决方法，但消融研究（例如 [原始 GQA 论文](https://arxiv.org/abs/2305.13245) 和 [Llama 2 论文](https://arxiv.org/abs/2307.09288) 中的研究）表明，它在 LLM 建模性能方面的表现与标准 MHA 相当。

然而，这假设键值组的数量是经过仔细选择的。在所有注意力头共享一个键值组（称为多查询注意力）的极端情况下，内存使用量会大幅减少，但建模性能可能会受到影响。 （并且，在另一个极端，如果我们将键值组的数量设置为等于查询头的数量，我们又回到了标准的多头注意力。）

 
## GQA 内存节省内存的节省主要体现在KV存储上。我们可以通过以下公式计算KV存储大小：

字节 ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

您可以使用此文件夹中的 [memory_estimator_gqa.py](memory_estimator_gqa.py) 脚本将此脚本应用于不同的模型配置，以查看通过使用 GQA 而非 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

下图进一步显示了使用 GQA 而非 MHA 时所节省的成本，其中不同的键值组大小与上下文长度有关：

 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />

 

您可以通过 `uv run plot_memory_estimates_gqa.py` 重现该图。

 
## GQA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) 脚本提供了在 GPT 模型实现的上下文中比较 MHA 和 GQA 内存使用情况的实践示例。

请注意，GQA 也用于 [Llama 3](../../ch05/07_gpt_to_llama)、[Gemma 3](../../ch05/12_gemma3) 和 [Qwen3](../../ch05/11_qwen3) 奖励材料中。但是，为了简单起见，此文件夹中的代码脚本修改了传统上不使用 GQA 的 GPT 架构。

请注意，该模型未经训练，因此会生成无意义的文本。但是，您可以使用它作为第 5-7 章中标准 GPT 模型的直接替代品并对其进行训练。

此外，此实现使用 [另一个奖励部分](../03_kv-cache) 中解释的 KV 缓存，因此内存节省更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

我们没有看到上图中如此大的节省的原因有两个：

1.我使用较小的配置让模型在合理的时间内完成生成。
2.更重要的是，我们在这里看的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（但这是一个单独分析的主题）。