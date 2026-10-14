# DeepSeek 稀疏注意力 (DSA)

该奖励材料实现了 [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 中引入的 DeepSeek 稀疏注意力 (DSA) 机制，并首次在实验性 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 版本中发布。

以下概述遵循 [从 DeepSeek V3 到 V3.2：架构、稀疏注意力和 RL 更新](https://magazine.sebastianraschka.com/p/technical-deepseek) 中的 DSA 讨论。

 
## 简介

标准因果自注意力关注每个查询的所有先前标记，产生 O(L²) 计算和 O(L) KV 缓存增长（序列长度为 L）。

[滑动窗口注意力（SWA）](../06_swa) 已经表明，将注意力限制在固定的本地窗口上可以大大降低这种成本。在 SWA 中，每个查询令牌仅关注附近先前令牌的本地范围。

 

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/09.png" alt="Sliding window attention" width="800px" />

*图 1. 滑动窗口注意力将每个查询标记限制到固定的本地上下文窗口。*

 

DSA 使用相同的广泛思想，即仅关注先前令牌的子集。然而，它用学习的选择机制代替了固定窗口。对于每个查询标记，该模型会对候选过去的标记进行评分，并仅保留最相关的标记。

 

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/10.png" alt="DeepSeek Sparse Attention selected-token pattern" width="800px" />

*图 2.DeepSeek 稀疏注意力为每个查询标记选择过去标记的学习子集。*

 

### 架构概述

DSA 在标准注意力的基础上添加了两个组件。

**1.闪电索引器**

对于每个查询标记 $t$ 和每个候选过去标记 $s$，索引器计算标量相关性得分。此实现使参考代码中的比例因子变得明确：

$$I_{t,s} = \sum_{j=1}^{H_I} \frac{w_{t,j}}{\sqrt{H_I}} \cdot \text{ReLU}\left(\frac{q_{t,j} \cdot k_s}{\sqrt{d_I}}\right)$$

其中：
- $H_I$ 是轻量级索引头的数量，
- $q_{t,j}$ 是令牌 $t$ 和头 $j$ 的索引器查询向量，
- $k_s$ 是过去令牌 $s$ 的共享索引器键向量，
- $w_{t,j}$ 是由 $1 / \sqrt{H_I}$ 缩放的学习的每头门。

ReLU 将负点积贡献归零，并且各个索引头的门控总和汇总为每个过去标记的单个相关性得分。

在完整的 DeepSeek 模型中，索引器使用来自多头潜在注意力 (MLA) 的压缩标记表示。该文件夹使 GPT 实现更加简单，并根据常规隐藏状态计算索引器查询和键。

**2.代币选择器**计算完所有索引器分数后，仅保留前 K 个最高分数位置。所有其他位置都在标准 softmax *之前*被屏蔽为 −∞，因此该模型实际上只关注 $k \ll L$ 令牌。

索引器中的 ReLU 并不是最终稀疏性的来源。由于分数是在多个索引头上求和的，因此大多数最终分数仍然可以是非零的。令牌选择器通过仅保留前 K 个位置来创建稀疏模式。

在融合生产实现中，这可以将注意力计算从 O(L²) 降低到 O(L·k)。这里的实现保留了标准的密集注意力得分矩阵，并在 softmax 之前应用 DSA 选择的 top-K 掩码。这使得选择逻辑易于检查，但它不提供融合内核计算节省。

下图总结了流程。闪电索引器对候选标记进行评分，选择器保留前 K 个位置，并且生成的掩码限制通常的注意力 softmax。

 

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/11.png" alt="DeepSeek Sparse Attention flowchart" width="700px" />

*图 3.DSA 首先对候选标记进行评分，然后保留前 K 个标记作为最终的注意力掩码。*

 
## 实施

`gpt_with_kv_dsa.py` 提供：

|班级 |描述 |
|---|---|
| `LightningIndexer` |用于过去标记相关性的轻量级多头记分器。 |
| `MultiHeadAttentionWithDSA` |带 DSA 稀疏掩码的标准 MHA + 可选 KV 缓存。 |
| `GPTModel` | `MultiHeadAttentionWithDSA` 中的 GPT 样式模型交换。 |

该实现遵循此存储库中其他奖励材料的风格，并且可以作为独立脚本运行。它的目的是使 DSA 机制可以在小型 GPT 样式模型中进行检查。它没有实现 DeepSeek 的完整 MLA 堆栈、融合稀疏内核或特定于部署的优化。

 
## 用法

```bash
uv run gpt_with_kv_dsa.py \
  --emb_dim 768 \
  --n_heads 12 \
  --n_layers 12 \
  --max_new_tokens 200 \
  --index_n_heads 4 \
  --index_head_dim 64 \
  --topk 64
```

关键论点：

|论证|默认|描述 |
|---|---|---|
| `--index_n_heads` | 4 |轻型分度器头的数量 (H_I)。 |
| `--index_head_dim` | 64 | 64每个分度头的尺寸。 |
| `--topk` | 64 | 64每个查询涉及的令牌数量 (k)。对于短序列，以序列长度为上限。 |

 
## 与 DeepSeek V3.2 的关系

完整的 DeepSeek-V3.2 模型使用多头潜在注意力（MLA，请参阅 [../05_mla](../05_mla)）和 DSA，并且索引器查询源自共享的压缩潜在表示而不是原始输入。 DeepSeek-V3.2 使用与 DeepSeek-V3.2-Exp 相同的架构，其中首次引入并测试了 DSA。

这里再现了关键选择的想法。一个廉价的学习点积评分器将每个查询限制在注意力 softmax 之前最相关的标记。

下面报告的推理成本比较是说明 DSA 在长上下文部署中为何重要的有用背景。节省的成本取决于生产内核和服务基础设施，因此不应将此数字视为此文件夹中教学实施的基准。

 

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/19.png" alt="Inference cost comparison for DeepSeek Sparse Attention" width="800px" />

*图 4. DeepSeek 报告的 DSA 在长上下文服务中节省的推理成本，来自 [DeepSeek V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)。*

 
## 参考文献

- DeepSeek V3.2技术报告：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp 模型卡和参考代码：https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka 的“从 DeepSeek V3 到 V3.2：架构、稀疏注意力和 RL 更新”：https://magazine.sebastianraschka.com/p/technical-deepseek