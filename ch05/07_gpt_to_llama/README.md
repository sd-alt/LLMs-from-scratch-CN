# 将 GPT 转换为 Llama



该文件夹包含用于将 GPT 实现从第 4 章和第 5 章转换为 Meta AI 的 Llama 架构的代码，建议阅读顺序如下：

- [conversion-gpt-to-llama2.ipynb](converting-gpt-to-llama2.ipynb)：包含逐步将 GPT 转换为 Llama 2 7B 的代码，并从 Meta AI 加载预训练权重
- [conversion-llama2-to-llama3.ipynb](converting-llama2-to-llama3.ipynb)：包含将 Llama 2 模型转换为 Llama 3、Llama 3.1 和 Llama 3.2 的代码
- [独立-llama32.ipynb](standalone-llama32.ipynb)：实现Llama 3.2的独立笔记本

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/gpt-and-all-llamas.webp">


 
### 通过 `llms-from-scratch` 包使用 Llama 3.2

为了轻松使用 Llama 3.2 1B 和 3B 模型，您还可以使用基于 [pkg/llms_from_scratch](../../pkg/llms_from_scratch) 的此存储库中的源代码的 `llms-from-scratch` PyPI 包。

 
#### 1）安装

```bash
pip install llms_from_scratch blobfile
```

（请注意，加载分词器需要 `blobfile`。）

 
#### 2) 模型和文本生成设置

指定要使用的模型：

```python
MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"
```

可由用户定义的基本文本生成设置。请注意，对于文本生成示例，建议的 8192 个令牌上下文大小需要大约 3 GB VRAM。

```python
# Text generation settings
if "instruct" in MODEL_FILE:
    PROMPT = "What do llamas eat?"
else:
    PROMPT = "Llamas eat"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

 
#### 3）权重下载和加载

这会根据上面的模型选择自动下载权重文件：

```python
import os
import requests

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(MODEL_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded to {MODEL_FILE}")
```

然后按如下方式加载模型权重：

```python
import torch
from llms_from_scratch.llama3 import Llama3Model

if "1B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
elif "3B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
else:
    raise ValueError("Incorrect model file name")

model = Llama3Model(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

 
#### 4) 初始化分词器

以下代码下载并初始化分词器：

```python
from llms_from_scratch.llama3 import Llama3Tokenizer, ChatFormat, clean_text

TOKENIZER_FILE = "tokenizer.model"

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

if not os.path.exists(TOKENIZER_FILE):
    urllib.request.urlretrieve(url, TOKENIZER_FILE)
    print(f"Downloaded to {TOKENIZER_FILE}")
    
tokenizer = Llama3Tokenizer("tokenizer.model")

if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)
```

 
#### 5) 生成文本

最后，我们可以通过以下代码生成文本：

```python
import time

from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

torch.manual_seed(123)

start = time.time()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=TOP_K,
    temperature=TEMPERATURE
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = token_ids_to_text(token_ids, tokenizer)

if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

print("\n\nOutput text:\n\n", output_text)
```

使用 Llama 3.2 1B Instruct 模型时，输出应类似于下图所示：

```
Time: 3.17 sec
50 tokens/sec
Max memory allocated: 2.91 GB


Output text:

 Llamas are herbivores, which means they primarily eat plants. Their diet consists mainly of:

1. Grasses: Llamas love to graze on various types of grasses, including tall grasses and grassy meadows.
2. Hay: Llamas also eat hay, which is a dry, compressed form of grass or other plants.
3. Alfalfa: Alfalfa is a legume that is commonly used as a hay substitute in llama feed.
4. Other plants: Llamas will also eat other plants, such as clover, dandelions, and wild grasses.

It's worth noting that the specific diet of llamas can vary depending on factors such as the breed,
```

 
#### 专业技巧 1：使用 FlashAttention 加速推理

您可以使用 `Llama3ModelFast` 作为直接替代品，而不是使用 `Llama3Model`。有关更多信息，我鼓励您检查 [pkg/llms_from_scratch/llama3.py](../../pkg/llms_from_scratch/llama3.py) 代码。

`Llama3ModelFast` 使用 PyTorch 的 `scaled_dot_product` 函数替换了 `GroupedQueryAttention` 模块中的从头开始缩放的点积代码，该函数在 Ampere GPU 或更高版本上使用 `FlashAttention`。

下表显示了 A100 上的性能比较：|                 |令牌/秒 |内存|
| ---------------- | ---------- | -------- |
|骆驼3模型 | 42 | 42 2.91 GB | 2.91 GB
|骆驼3模型快速 | 54 | 54 2.91 GB | 2.91 GB

 
#### 专业技巧 2：通过编译加速推理


要获得高达 4 倍的加速，请替换

```python
model.to(device)
```

与

```python
model = torch.compile(model)
model.to(device)
```

注意：编译时会产生大量的多分钟前期成本，并且加速在第一次 `generate` 调用后生效。

下表显示了 A100 上后续 `generate` 调用的性能比较：

|                 |令牌/秒 |内存|
| ---------------- | ---------- | -------- |
|骆驼3模型 | 170 | 170 3.12 GB | 3.12 GB
|骆驼3模型快速 | 177 | 177 3.61 GB | 3.61 GB

 
#### 专业技巧 3：通过编译加速推理

在 CPU 上运行模型时，您可以使用 KV 缓存 `Llama3Model` 直接替换显着提高推理性能。 （请参阅我的 [从头开始在 LLM 中理解和编码 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 文章，了解有关 KV 缓存的更多信息。）

```python
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Llama3Model(LLAMA32_CONFIG)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
)
```

请注意，峰值内存使用量仅针对 Nvidia CUDA 设备列出，因为它更容易计算。然而，其他设备上的内存使用情况可能相似，因为它使用类似的精度格式，并且 KV 缓存存储导致生成的 150 个令牌文本的内存使用量甚至更低（但是，不同的设备可能以不同的方式实现矩阵乘法，并可能导致不同的峰值内存需求；并且 KV 缓存内存可能会因较长的上下文长度而大幅增加）。|型号|模式|硬件|令牌/秒 | GPU 内存 (VRAM) |
| ----------- | ----------------- | ---------------- | ---------- | ----------------- |
|骆驼3模型|常规| Mac Mini M4 CPU | 1 | - |
|骆驼3模型|常规编译| Mac Mini M4 CPU | 1 | - |
|骆驼3模型| KV缓存| Mac Mini M4 CPU | 68 | 68 - |
|骆驼3模型| KV缓存编译| Mac Mini M4 CPU | 86 | 86 - |
|             |                   |                 |            |                   |
|骆驼3模型|常规| Mac Mini M4 GPU | 15 | 15 - |
|骆驼3模型|常规编译| Mac Mini M4 GPU |错误| - |
|骆驼3模型| KV缓存| Mac Mini M4 GPU | 62 | 62 - |
|骆驼3模型| KV缓存编译| Mac Mini M4 GPU |错误| - |
|             |                   |                 |            |                   |
|骆驼3模型|常规| Nvidia A100 GPU | 42 | 42 2.91 GB | 2.91 GB
|骆驼3模型|常规编译| Nvidia A100 GPU | 170 | 170 3.12 GB | 3.12 GB
|骆驼3模型| KV缓存| Nvidia A100 GPU | 58 | 58 2.87 GB | 2.87 GB
|骆驼3模型| KV缓存编译| Nvidia A100 GPU | 161 | 161 3.61 GB | 3.61 GB

请注意，上述所有设置都经过测试，可产生相同的文本输出。