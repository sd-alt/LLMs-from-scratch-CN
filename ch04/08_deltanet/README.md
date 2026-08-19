# 用于线性注意力的门控 DeltaNet

最近，[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 和 [Kimi Linear](https://arxiv.org/abs/2510.26692) 提出了混合变压器，该变压器实现了注意力机制的替代方案，该机制相对于上下文长度线性而不是二次缩放。

Qwen3-Next 和 Kimi Linear 都使用 3:1 的比率，这意味着每三个使用线性 Gated DeltaNet 变体的 Transformer 块，就有一个块使用了完全注意力，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next versus Kimi Linear">



 

## 介绍和概述

门控 DeltaNet 是一种线性注意力变体，其灵感来自循环神经网络，包括 [门控 Delta 网络：使用 Delta 规则改进 Mamba2](https://arxiv.org/abs/2412.06464) 论文中的门控机制。从某种意义上说，Gated DeltaNet 是具有 Mamba 式门控的 DeltaNet，DeltaNet 是一种线性注意力机制。

Kimi Linear通过Kimi Delta Attention（KDA）机制修改了Qwen3-Next的线性注意力机制，本质上是Gated DeltaNet的细化。 Qwen3-Next 应用标量门（每个注意力头一个值）来控制记忆衰减率，而 Kimi Linear 将其替换为每个特征维度的通道门控。作者表示，这可以更好地控制记忆，进而改善长上下文推理。

此外，对于完整的注意力层，Kimi Linear 用多头潜在注意力 (MLA) 取代了 Qwen3-Next 的门控注意力层（本质上是具有输出门控的标准多头注意力层）。这与我们之前在 DeepSeek V3/R1 部分讨论的 MLA 机制相同，但有一个额外的门。 （回顾一下，MLA 压缩键/值空间以减少 KV 缓存大小。）

Kimi Linear 中的 MLA 不使用门，这是故意的，以便作者可以更直接地将架构与标准 MLA 进行比较，但是，他们 [表示](https://x.com/yzhang_cs/status/1984631714464088563) 计划在将来添加它。

由于我们已经在 [../05_mla](../05_mla) 中实现了 MLA，因此本奖励材料重点关注门控 DeltaNet 方面。


 
## 门控注意力在我们讨论门控 DeltaNet 本身之前，我们先简单谈谈门。正如上图 Qwen3-Next 架构的上半部分所示，Qwen3-Next 使用了“门控注意力”。这本质上是带有附加 s 形门的常规全注意力。

这个门控是一个简单的修改，出于说明目的，我将其添加到下面第 3 章的 `MultiHeadAttention` 代码中：

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Add gate
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### NEW: Add gate
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### NEW: Add gate
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```



正如我们所看到的，在像往常一样计算注意力之后，模型使用来自同一输入的单独门控信号，应用 sigmoid 使其保持在 0 和 1 之间，并将其与注意力输出相乘。这允许模型动态地放大或缩小某些特征。 Qwen3-Next 开发人员 [声明](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 这有助于提高训练稳定性：

> [...]注意力输出门控机制有助于消除注意力池和大规模激活等问题，确保整个模型的数值稳定性。


 
## 门控 DeltaNet

现在，什么是门控 DeltaNet？ Gated DeltaNet（*Gated Delta Network* 的缩写）是 Qwen3-Next 的线性注意力层，旨在作为标准 softmax 注意力的替代方案。它是从前面提到的 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文中采用的。

门控 DeltaNet 最初是作为 Mamba2 的改进版本提出的，它将 Mamba2 的门控衰减机制与 delta 规则相结合。

Mamba 是一个状态空间模型（Transformer 的替代品），这是一个值得将来单独讨论的大话题。

Delta 规则部分是指计算新值和预测值之间的差异 (delta, Δ)，以更新用作记忆状态的隐藏状态（稍后会详细介绍）。

（旁注：阅读经典机器学习文献的读者可以认为这类似于受生物学启发的赫布学习：“一起发射的细胞连接在一起。”它基本上是感知器更新规则和基于梯度下降的学习的先驱，但没有监督。）

门控 DeltaNet 的门与前面讨论的门控注意力中的门类似，不同之处在于它使用 SiLU 而不是逻辑 sigmoid 激活，如下所示。 （与标准 sigmoid 相比，SiLU 的选择可能会提高梯度流量和稳定性。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="Gated DeltaNet" width=500px>然而，如上图所示，Gated DeltaNet 中的“gate”还指的是几个额外的门：

- `α`（衰减门）控制内存随时间衰减或重置的速度，
- `β`（更新门）控制新输入修改状态的强度。

在代码中，上面描述的门控 DeltaNet 的简化版本（没有卷积混合）可以按如下方式实现（该代码受到 Qwen3 团队的 [官方实现](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835) 的启发）。

（请注意，某些实现将衰减门称为 `gk`（步骤 k 的门），其中 `exp(gk)` 与论文的 $\alpha_t$ 匹配。为了保持这种关系明确，下面的代码片段将对数空间门 `alpha_log` 与指数衰减 `alpha` 分开。）


```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Gates for delta rule and output gating
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # Note: The decay gate alpha corresponds to
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # We could implement this as
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # but the bias is separate for interpretability and
        # to mimic the official implementation

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### NEW: Compute delta rule gates
        beta = torch.sigmoid(self.W_beta(x))
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # NEW

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)

        ####################################################
        ### NEW: QKNorm-like normalization for delta rule
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### NEW: Gated delta rule update
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### NEW: Apply RMSNorm and SiLU gate
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

（请注意，为了简单起见，我省略了 Qwen3-Next 和 Kimi Linear 使用的卷积混合，以保持代码更具可读性并专注于循环方面。）

因此，正如我们在上面所看到的，标准（或门控）注意力有很多差异。

在门控注意力中，模型计算所有令牌之间的正常注意力（每个令牌都参与或查看每个其他令牌）。然后，在获得注意力输出后，门（sigmoid）决定保留多少输出。结论是，它仍然是常规的缩放点积注意力，与上下文长度呈二次方缩放。

作为复习，缩放点生产注意力计算为 softmax(QKᵀ)V，其中 Q 和 K 是 *n*×*d* 矩阵，其中 *n* 是输入标记的数量，*d* 是嵌入维度。因此 QKᵀ 会产生一个注意力 *n*×n* 矩阵，该矩阵乘以 *n*×d* 维值矩阵 V：

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="Quadratic attention" width=500px />

在门控 DeltaNet 中，没有 *n*×n* 注意力矩阵。相反，模型会一一处理令牌。它保留一个运行内存（状态），随着每个新令牌的进入而更新。这就是实现的方式，其中 `S` 是每个时间步 *t* 循环更新的状态。

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

门控制着记忆如何改变：

- α (`alpha`) 调节旧记忆的遗忘量（衰减）。

- β (`beta`) 调节当前令牌在时间步 *t* 更新内存的数量。（上面的代码片段中没有显示最终的输出门，它类似于门控注意力；它控制保留多少输出。）

因此，从某种意义上说，门控 DeltaNet 中的状态更新类似于循环神经网络 (RNN) 的工作原理。优点是它可以线性缩放（通过 for 循环），而不是与上下文长度成二次方缩放。

这种循环状态更新的缺点是，与常规（或门控）注意力相比，它牺牲了来自完全成对注意力的全局上下文建模能力。

门控 DeltaNet 在某种程度上仍然可以捕获上下文，但它必须克服内存 (*S*) 瓶颈。该内存是固定大小的，因此效率更高，但它将过去的上下文压缩为类似于 RNN 的单个隐藏状态。

这就是为什么 Qwen3-Next 和 Kimi Linear 架构没有用 DeltaNet 层替换所有注意力层，而是使用前面提到的 3:1 比例。

 
## DeltaNet 内存节省

在上一节中，我们讨论了 DeltaNet 相对于完全注意力机制的优势，即相对于上下文长度的线性计算复杂性（而不是二次计算复杂性）。

除了线性计算复杂性之外，DeltaNet 的另一大优势是节省内存，因为 DeltaNet 模块不会增加 KV 缓存。 （有关 KV 缓存的更多信息，请参阅 [../03_kv-cache](../03_kv-cache)）。相反，如前所述，它们保持固定大小的循环状态，因此内存随着上下文长度保持不变。

对于常规的多头注意力（MHA）层，我们可以计算 KV 缓存大小如下：

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

（存在 2 乘数是因为我们在缓存中存储了键和值。）

对于上面实现的简化 DeltaNet 版本，我们有：


```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

请注意，`KV_cache_DeltaNet` 内存大小没有上下文长度 (`n_tokens`) 依赖性。此外，我们只有存储的内存状态 S，而不是单独的键和值，因此 `2 × bytes` 就变成了 `bytes`。但是，请注意，这里现在有一个二次 `d_head × d_head`。这来自国家：

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

但这通常无需担心，因为头部尺寸通常相对较小。例如，Qwen3-Next 中的值为 128。带有卷积混合的完整版本有点复杂，包括内核大小等，但上面的公式应该说明了门控 DeltaNet 背后的主要趋势和动机。

我们可以通过以下帮助程序脚本可视化不同上下文长度的内存估计和节省：

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

请注意，上面将 `head_dim` 计算为 `emb_dim / n_heads`。即 2048 / 16 = 128。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="Gated DeltaNet scaling" width=500px>