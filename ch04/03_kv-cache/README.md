# 奖励材料：KV 缓存



**该文件夹实现了在GPT模型上添加KV缓存。**

 
## 概述

简而言之，KV 缓存存储中间键 (K) 和值 (V) 计算，以便在推理过程中重用，从而在生成响应时显着加快速度。缺点是增加了代码的一些复杂性，增加了内存占用，并且在训练时无法使用。然而，在部署 LLM 时，推理速度的提升通常值得在代码复杂性和内存方面进行权衡。

 
## 它是如何工作的

想象一下大语言模型正在生成一些文本。具体来说，假设 LLM 收到以下提示：“时间过得真快”。

下图显示了使用第 3 章中修改后的图形进行的底层注意力分数计算的摘录，其中突出显示了键和值向量：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

现在，正如我们在第 2 章和第 4 章中了解到的，LLM 一次生成一个单词（或标记）。假设LLM生成了“快”这个词，这样下一轮的提示就变成了“时间过得快”。下图对此进行了说明：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

我们可以看到，根据比较前两张图，前两个标记的键和值向量完全相同，并且在每个下一个标记文本生成轮中重新计算它们是浪费的。

所以，KV缓存的想法是实现一种缓存机制，存储之前生成的键和值向量以供重用，这有助于我们避免不必要的重新计算。

 

## KV缓存实现

实现 KV 缓存的方法有很多，主要思想是我们只计算每个生成步骤中新生成的令牌的键和值张量。

我选择了一种强调代码可读性的简单方案。我认为最简单的方法是滚动浏览代码更改以查看其实现方式。

该文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：取自第3章和第4章的独立代码，用于实现LLM并运行简单的文本生成功能
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上面相同，但进行了必要的更改以实现 KV 缓存。

你可以

a.打开 [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 文件并查找标记新更改的 `# NEW` 部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b.通过您选择的文件比较工具检查两个代码文件以比较更改：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

为了总结实现细节，这里有一个简短的演练。

 

### 1. 注册缓存缓冲区

在 `MultiHeadAttention` 构造函数中，我们添加两个缓冲区：`cache_k` 和 `cache_v`，它们将跨步骤保存连接的键和值：

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```

 

### 2. 带有 `use_cache` 标志的前向传递

接下来，我们扩展 `MultiHeadAttention` 类的 `forward` 方法以接受 `use_cache` 参数。将新的令牌块投影到 `keys_new`、`values_new` 和 `queries` 后，我们要么初始化 kv 缓存，要么追加到我们的缓存中：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

 


### 3.清除缓存

生成文本时，在独立序列之间（例如文本生成调用），我们必须重置两个缓冲区，因此我们还在 `MultiHeadAttention` 类中添加了一个缓存重置方法：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

 

### 4.在完整模型中传播`use_cache`

对 `MultiHeadAttention` 类进行更改后，我们现在修改 `GPTModel` 类。首先，我们向讲师添加代币索引的位置跟踪：

```python
self.current_pos = 0
```

然后，我们用显式循环替换单行块调用，将 `use_cache` 传递给每个转换器块：

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述更改还需要对 `TransformerBlock` 类进行小修改以接受 `use_cache` 参数：
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，为了方便起见，我们向 `GPTModel` 添加模型级重置，以一次性清除所有块缓存：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

 

### 5. 在生成中使用缓存

通过对 `GPTModel`、`TransformerBlock` 和 `MultiHeadAttention` 的更改，最后，我们在简单的文本生成函数中使用 KV 缓存：

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

请注意，我们仅通过 `logits = model(next_idx, use_cache=True)` 向模型提供 c) 中的新令牌。如果没有缓存，我们会向模型提供整个输入 `logits = model(idx[:, -ctx_len:], use_cache=False)`，因为它没有存储的键和值可供重用。

 

## 简单的性能比较

在概念层面上介绍了 KV 缓存之后，最大的问题是它在一个小示例的实践中实际表现如何。为了尝试实现，我们可以将上述两个代码文件作为 Python 脚本运行，这将运行小型 124 M 参数 LLM 来生成 200 个新令牌（以 4 个令牌提示符“Hello，I am”开始）：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在配备M4芯片（CPU）的Mac Mini上，结果如下：

|                        |令牌/秒 |
| ---------------------- | ---------- |
| `gpt_ch04.py` | 27 |
| `gpt_with_kv_cache.py` | 144 |

因此，正如我们所看到的，我们已经通过 124 M 的小型参数模型和较短的 200 个令牌序列长度获得了大约 5 倍的加速。 （请注意，此实现针对代码可读性进行了优化，并未针对 CUDA 或 MPS 运行时速度进行优化，这需要预先分配张量，而不是恢复和连接它们。）

**注意：** 在这两种情况下，模型都会生成“乱码”，即文本如下所示：

> 输出文本：你好，我是Featureiman Byeswickattribute 争论记录器诺曼底康普顿类似孔ITVEGIN ministriesysics Kle 功能recountricionchangingVirgin embarrassedgl ...

这是因为我们还没有训练模型。下一章将训练模型，您可以在训练后的模型上使用 KV 缓存（但是，KV 缓存仅在推理过程中使用）来生成连贯的文本。在这里，我们使用未经训练的模型来保持代码简单（r）。

但更重要的是，`gpt_ch04.py` 和 `gpt_with_kv_cache.py` 实现都会生成完全相同的文本。这告诉我们 KV 缓存的实现是正确的——很容易出现索引错误，从而导致不同的结果。


 

## KV缓存优缺点

随着序列长度的增加，KV 缓存的优点和缺点在以下方面变得更加明显：

- [好] **计算效率提高**：如果没有缓存，步骤 *t* 的注意力必须将新查询与 *t* 先前的键进行比较，因此累积工作量呈二次方扩展，O(n²)。使用缓存，每个键和值都会计算一次，然后重复使用，从而将每步的总复杂度降低到线性 O(n)。

- [坏] **内存使用量线性增加**：每个新令牌都会附加到 KV 缓存。对于长序列和较大的 LLM，累积的 KV 缓存会变得更大，这可能会消耗大量甚至令人望而却步的 (GPU) 内存。作为一种解决方法，我们可以截断 KV 缓存，但这会增加更多的复杂性（但同样，在部署 LLM 时，这可能是值得的。）



 
## 优化KV缓存实现

虽然我上面对 KV 缓存的概念实现有助于提高清晰度，并且主要面向代码可读性和教育目的，但将其部署在现实场景中（尤其是较大的模型和较长的序列长度）需要更仔细的优化。

 
### 扩展缓存时的常见陷阱

- **内存碎片和重复分配**：如前所示，通过 `torch.cat` 连续连接张量，由于频繁的内存分配和重新分配而导致性能瓶颈。

- **内存使用量线性增长**：如果没有适当的处理，KV 缓存大小对于非常长的序列来说变得不切实际。

 
#### 技巧 1：预分配内存

我们可以根据预期的最大序列长度预先分配足够大的张量，而不是重复连接张量。这确保了内存使用的一致性并减少了开销。在伪代码中，这可能如下所示：

```python
# Example pre-allocation for keys and values
max_seq_len = 1024  # maximum expected sequence length
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

在推理过程中，我们可以简单地写入这些预先分配的张量的切片。

 
#### 技巧 2：通过滑动窗口截断缓存

为了避免耗尽 GPU 内存，我们可以实现带有动态截断的滑动窗口方法。通过滑动窗口，我们仅维护缓存中的最后一个 `window_size` 令牌：


```python
# Sliding window cache implementation
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

 
#### 实践中的优化

您可以在 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 文件中找到这些优化。


在配备 M4 芯片 (CPU) 的 Mac Mini 上，生成 200 个令牌，窗口大小等于下面的上下文长度（以保证结果相同），代码运行时间比较如下：

|                                  |令牌/秒 |
| -------------------------------- | ---------- |
| `gpt_ch04.py` | 27 |
| `gpt_with_kv_cache.py` | 144 |
| `gpt_with_kv_cache_optimized.py` | 166 |

不幸的是，速度优势在 CUDA 设备上消失了，因为这是一个小型模型，并且设备传输和通信超过了该小型模型的 KV 缓存的优势。


 
## 其他资源

1. [Qwen3 从头开始 KV 缓存基准](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 从头开始 KV 缓存基准](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [从头开始理解和编码 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- 本自述文件的更详细的撰写