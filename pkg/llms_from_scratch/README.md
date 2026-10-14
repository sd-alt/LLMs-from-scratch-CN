# `llms-from-scratch` PyPI 包

这个可选的 PyPI 包可让您方便地从“从头开始构建大型语言模型”一书中的各个章节导入代码。

 
## 安装

 
### 来自 PyPI

从官方 [Python 包索引](https://pypi.org/project/llms-from-scratch/) (PyPI) 安装 `llms-from-scratch` 包：

```bash
pip install llms-from-scratch
```

> **注意：** 如果您使用的是 [`uv`](https://github.com/astral-sh/uv)，请将 `pip` 替换为 `uv pip` 或使用 `uv add`：

```bash
uv add llms-from-scratch
```



 
### 从 GitHub 进行可编辑安装

如果您想修改代码并在开发过程中反映这些更改：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -e .
```

> **注意：** 对于 `uv`，请使用：

```bash
uv add --editable . --dev
```



 
## 使用包

安装后，您可以使用以下命令从任何章节导入代码：

```python
from llms_from_scratch.ch02 import GPTDatasetV1, create_dataloader_v1

from llms_from_scratch.ch03 import (
    SelfAttention_v1,
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    PyTorchMultiHeadAttention # Bonus: Faster variant using PyTorch's scaled_dot_product_attention
)

from llms_from_scratch.ch04 import (
    LayerNorm,
    GELU,
    FeedForward,
    TransformerBlock,
    GPTModel,
    GPTModelFast # Bonus: Faster variant using PyTorch's scaled_dot_product_attention
    generate_text_simple
)

from llms_from_scratch.ch05 import (
    generate,
    train_model_simple,
    evaluate_model,
    generate_and_print_sample,
    assign,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader,
    plot_losses,
    download_and_load_gpt2
)

from llms_from_scratch.ch06 import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split,
    SpamDataset,
    calc_accuracy_loader,
    evaluate_model,
    train_classifier_simple,
    plot_values,
    classify_review
)

from llms_from_scratch.ch07 import (
    download_and_load_file,
    format_input,
    InstructionDataset,
    custom_collate_fn,
    check_if_running,
    query_model,
    generate_model_scores
)

	
from llms_from_scratch.appendix_a import NeuralNetwork, ToyDataset

from llms_from_scratch.appendix_d import find_highest_gradient, train_model
```



 

### GPT-2 KV 缓存变体（奖励材料）

```python
from llms_from_scratch.kv_cache.gpt2 import GPTModel
from llms_from_scratch.kv_cache.generate import generate_text_simple
```

有关 KV 缓存的更多信息，请参阅 [KV 缓存 README](../../ch04/03_kv-cache)。



 

### 骆驼 3（奖励材料）

```python
from llms_from_scratch.llama3 import (
		load_weights_into_llama,
  	Llama3Model,
    Llama3ModelFast,
    Llama3Tokenizer,
    ChatFormat,
    clean_text
)

# KV cache drop-in replacements
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple
```

关于`llms_from_scratch.llama3`的使用信息，请参阅[此奖励部分](../../ch05/07_gpt_to_llama/README.md)。

有关 KV 缓存的更多信息，请参阅 [KV 缓存 README](../../ch04/03_kv-cache)。


 
### Qwen3（奖励材料）

```python
from llms_from_scratch.qwen3 import (
    load_weights_into_qwen,
    Qwen3Model,
    Qwen3Tokenizer,
)

# KV cache drop-in replacements
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import (
    generate_text_simple,
    generate_text_simple_stream
)

# KV cache drop-in replacements with batched inference support
from llms_from_scratch.kv_cache_batched.generate import (
    generate_text_simple,
    generate_text_simple_stream
)
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```

有关`llms_from_scratch.qwen3`的使用信息，请参阅[此奖金部分](../../ch05/11_qwen3/README.md)。

有关 KV 缓存的更多信息，请参阅 [KV 缓存 README](../../ch04/03_kv-cache)。