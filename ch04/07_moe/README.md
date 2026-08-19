# 专家组合 (MoE)

此奖励材料说明了使用专家混合 (MoE) 层而不是常规前馈 (FFN) 层时的内存节省（每个令牌）。



 
## 简介

MoE 的核心思想是用多个专家层替换变压器块中的每个前馈模块，其中每个专家层也是一个前馈模块。这意味着我们用多个前馈块替换单个前馈块，如下图所示。



 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

变压器块内的前馈块（如上图中的深灰色块所示）通常包含大量模型的总参数。 （请注意，变压器模块以及前馈模块在 LLM 中重复多次；在 DeepSeek-V3 中重复了 61 次。）

因此，用“多个”前馈模块替换“单个”前馈模块（如 MoE 设置中所做的那样）会大大增加模型的总参数数量。然而，关键的技巧是我们不会为每个代币使用（“激活”）所有专家。相反，路由器只为每个令牌选择一小部分专家。

由于一次只有少数专家处于活动状态，MoE 模块通常被称为“稀疏”模块，与始终使用完整参数集的“密集”模块形成鲜明对比。然而，MoE 提供的大量参数增加了 LLM 的容量，这意味着它可以在训练过程中占用更多知识。不过，稀疏性使推理保持高效，因为我们不会同时使用所有参数。

例如，DeepSeek-V3 每个 MoE 模块有 256 个专家，总共 6710 亿个参数。然而在推理过程中，一次只有 9 个专家处于活动状态（1 个共享专家加上路由器选择的 8 个专家）。这意味着每个令牌推理步骤仅使用 370 亿个参数，而不是全部 6710 亿个参数。

DeepSeek-V3 的 MoE 设计的一个显着特点是使用共享专家。这是一位对每个代币始终活跃的专家。这个想法并不新鲜，已经在 [2022 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 和 [2024 DeepSeek MoE](https://arxiv.org/abs/2401.06066) 论文中引入。

 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE shared expert" width="500px" />

（带注释的图来自 [DeepSeekMoE：走向混合专家语言模型](https://arxiv.org/abs/2401.06066) 论文中的终极专家专业化。）

 [DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596) 中首次指出了拥有共享专家的好处，他们发现与没有共享专家相比，共享专家可以提高整体建模性能。这可能是因为常见或重复的模式不必由多个专家单独学习，这为他们提供了更多学习更专业模式的空间。

 
## 混合专家 (MoE) 内存节省

MoE 模型中的内存节省主要来自激活存储和计算的减少。在常规（密集）前馈层（FFN）中，每个标记都会激活完整的中间维度。

相比之下，MoE 层仅通过每个令牌的一小部分专家（例如 `num_experts` 中的 `top_k`）路由每个令牌。

当使用 MoE 层时，每个代币只有 `top_k` 专家处于活动状态，因此有效内存（和计算）相对于相同总容量的密集 FFN 大约缩放 `top_k / num_experts` 倍。


您可以使用此文件夹中的 [memory_estimator_moe.py](memory_estimator_moe.py) 脚本将此脚本应用于不同的模型配置，以查看使用 MoE 而不是 FFN 可以节省多少内存（请注意，这是针对单个变压器块，要获得总节省量，请乘以模型中变压器块的数量）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

因此，根据上面的结果，我们可以看到，如果我们的 FFN 的输入/输出维度 (`emb_dim`) 为 7,168，中间大小 (`hidden_dim`) 为 14,336，则该层中的参数约为 308M，并且所有这些参数在前向传递中都处于活动状态。

现在，如果我们使用总参数数量大致相同（约 308M）的 MoE 层，其中有 8 个专家，其中 2 个专家处于活动状态，则每次前向传递中只有约 77M 参数处于活动状态。

此外，在专家数量恒定的情况下，我们拥有的专家越多，活动参数的数量就越少，“节省”就越大：

 

 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />



 

您可以通过以下方式重现该图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


 
## 教育部代码示例此文件夹中的 [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) 和 [gpt_with_kv_moe.py](gpt_with_kv_moe.py) 脚本提供了在 GPT 模型实现的上下文中比较常规 FFN 和 MoE 内存使用情况的实践示例。请注意，两个脚本都使用 [SwiGLU](https://arxiv.org/abs/2002.05202) 前馈模块，如本页第一张图所示（GPT-2 传统上使用 GELU）。

**注意：该模型未经训练，因此会生成无意义的文本。您可以在奖励材料中找到经过培训的 MoE：[../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb)。**



首先，让我们使用常规 FFN 运行模型：


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

为了与 MoE 进行公平比较，我们必须缩小专家规模。例如，我们使用 32 位专家，我们必须设置 `--hidden_dim 32768/32`：


```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

我们可以看到，密集前馈层在大约 0.76 毫秒内处理一个令牌，并使用大约 0.19 MB 的激活（峰值接近 0.75 MB），

稀疏 MoE 层仅保留约 0.04 MB 的内存（峰值为 0.11）。然而，这是以大约两倍的计算时间为代价的。 （这会增加路由开销，而且我的实现也可能不是最有效的。）

在这两种情况下，总体生成的 GPU 内存峰值仍然在 11.5 GB 左右，因为两个版本都加载相同数量的权重参数并具有相同的 KV 缓存大小，这在这里占主导地位。

无论哪种方式，我们都可以看到这里的权衡，MoE 将 FFN 内存减少了大约 4-5 倍，同时前馈计算时间大致加倍。

请注意，如果我们一次性处理更多令牌，例如批量大小大于 1（由于代码简单，这里我们没有批量），节省的费用会更加明显。