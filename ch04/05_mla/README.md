# 多头潜在注意力（MLA）

此奖励材料说明了使用多头潜在注意力 (MLA) 相对于常规多头注意力 (MHA) 时所节省的内存。

 
## 简介

在 [../04_gqa](../04_gqa) 中，我们讨论了分组查询注意（GQA）作为 MHA 的计算效率解决方法。消融研究（例如 [原始 GQA 论文](https://arxiv.org/abs/2305.13245) 和 [Llama 2 论文](https://arxiv.org/abs/2307.09288) 中的研究）表明，它在 LLM 建模性能方面与标准 MHA 相当。

现在，[DeepSeek V2、V3 和 R1](https://arxiv.org/abs/2412.19437) 中使用的多头潜在注意力 (MLA) 提供了一种不同的内存节省策略，该策略也与 KV 缓存配合得很好。 MLA 不像 GQA 那样共享键和值头，而是将键和值张量压缩到低维空间，然后将它们存储在 KV 缓存中。

在推理时，这些压缩的张量在使用之前会被投影回其原始大小，如下图所示。这增加了额外的矩阵乘法，但减少了内存使用。

 

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

 

（顺便说一句，查询也被压缩，但仅限于训练期间，而不是推理期间。）

顺便说一句，如前所述，MLA 在 DeepSeek V3 中并不是什么新鲜事，因为它的 [DeepSeek V2 前身](https://arxiv.org/abs/2405.04434) 也使用（甚至引入）了它。此外，V2 论文包含一些有趣的消融研究，可以解释为什么 DeepSeek 团队选择 MLA 而不是 GQA（见下图）。

 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

 

如上图所示，GQA 的性能似乎比 MHA 差，而 MLA 的建模性能比 MHA 更好，这可能就是 DeepSeek 团队选择 MLA 而不是 GQA 的原因。 （看到 MLA 和 GQA 之间的“每个令牌的 KV 缓存”节省比较也会很有趣！）

总结本节，在我们继续讨论下一个架构组件之前，MLA 是一个巧妙的技巧，可以减少 KV 缓存的使用，同时在建模性能方面甚至稍微优于 MHA。

 
## MLA 内存节省

内存的节省主要体现在KV存储上。我们可以通过以下公式计算KV存储大小：

字节 ≈ batch_size × seqlen × n_layers × Latent_dim × bytes_per_elem

相比之下，MHA KV 高速缓存的计算如下：字节 ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × bytes_per_elem

这意味着，在 MLA 中，我们将“embed_dim × 2 (K,V)”减少为“latent_dim”，因为我们只存储压缩的潜在表示，而不是完整的键和值向量，如上图所示。



您可以使用此文件夹中的 [memory_estimator_mla.py](memory_estimator_mla.py) 脚本将此脚本应用于不同的模型配置，以查看通过使用 MLA 而非 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

请注意，上面的压缩 (`--emb_dim 2048 -> latent_dim 1024`) 实现了与 GQA 类似的保存。在实践中，压缩是一个需要仔细研究的超参数，因为选择太小的 `latent_dim` 会对建模性能产生负面影响（类似于在 GQA 中选择太多的 `n_kv_groups`）。

下图进一步显示了使用 MLA 而非 MHA 时所节省的成本，其中不同的 `latent_dim` 值作为上下文长度的函数：

 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

 

您可以通过 `uv run plot_memory_estimates_mla.py` 重现该图。



 
## MLA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_mla.py](gpt_with_kv_mla.py) 脚本提供了在 GPT 模型实现的上下文中比较 MHA 和 MLA 内存使用情况的实践示例。

此处，MLA 代码的灵感来自 [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) 实现。

注意，MLA也可以与[GQA](../04_gqa)结合使用，但为了简单起见，我在这里没有这样做。 （目前，我还不知道有哪个著名的大语言模型在这样做。）

另请注意，该模型未经训练，因此会生成无意义的文本。但是，您可以使用它作为第 5-7 章中标准 GPT 模型的直接替代品并对其进行训练。

最后，此实现使用 [另一个奖励部分](../03_kv-cache) 中解释的 KV 缓存，因此内存节省更加明显。

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

我们没有看到上图中如此大的节省的原因有两个：

1.我使用较小的配置让模型在合理的时间内完成生成。
2.更重要的是，我们在这里看的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（但这是一个单独分析的主题）。