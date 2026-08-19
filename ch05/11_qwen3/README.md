# Qwen3 从头开始

此文件夹中的 [standalone-qwen3.ipynb](standalone-qwen3.ipynb) Jupyter 笔记本包含 Qwen3 0.6B、1.7B、4B、8B 和 32B 的从头开始实现。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">


此文件夹中的 [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb) 和 [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb) Jupyter 笔记本包含 30B-A3B Mixture-of-Experts (MoE) 的从头开始实现，包括 Thinking、指令和编码器模型变体。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-coder-flash-overview.webp?123" width="430px">

 
# Qwen3 从头开始的代码

此文件夹中的独立笔记本包含线性方式的从头开始代码：

1. [独立-qwen3.ipynb](standalone-qwen3.ipynb)：没有花里胡哨的密集Qwen3模型
2. [standalone-qwen3-plus-kvcache.ipynb](standalone-qwen3-plus-kvcache.ipynb)：同上，但带有KV缓存以提高推理效率
3. [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb)：与第一款笔记本类似，但是是 Mixture-of-Experts (MoE) 变体
4. [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb)：同上，但带有KV缓存以提高推理效率

或者，我还将代码组织到一个 Python 包 [中（这里为](../../pkg/llms_from_scratch/)）（包括单元测试和 CI），您可以按如下所述运行它。

 
# 训练

`Qwen3Model` 类的实现方式与 `GPTModel` 类类似，因此它可以用作第 5 章中的训练以及第 6 章和第 7 章中的微调的直接替代。


 
# 通过 `llms-from-scratch` 包使用 Qwen3

为了轻松使用 Qwen3 从头开始实现，您还可以使用基于 [pkg/llms_from_scratch](../../pkg/llms_from_scratch) 存储库中源代码的 `llms-from-scratch` PyPI 包。

 
#### 1）安装

```bash
pip install llms_from_scratch tokenizers
```

 
#### 2) 模型和文本生成设置

指定要使用的模型：

```python
USE_REASONING_MODEL = True
# Uses the base model if USE_REASONING_MODEL = False

USE_INSTRUCT_MODEL = False
# Uses the instruct mode (without reasoning) if 
# USE_REASONING_MODEL = True
# USE_INSTRUCT_MODEL = True
# This setting does have no effect if USE_REASONING_MODEL = False


# Use
# USE_REASONING_MODEL = True
# For Qwen3 Coder Flash model as well
```

可由用户定义的基本文本生成设置。对于 150 个令牌，0.6B 模型需要大约 1.5 GB 内存。

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

 
#### 3a) 0.6B模型的权重下载和加载

下面根据上面的模型选择（推理或基础）自动下载权重文件。请注意，本节重点介绍 0.6B 模型。如果您想使用任何较大的模型（1.7B、4B、8B 或 32B），请跳过本节并继续执行第 3b) 节。```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

然后按如下方式加载模型权重：

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);
```

 
#### 3b) 较大 Qwen 模型的重量下载和加载

如果您有兴趣使用任何较大的 Qwen 模型，例如 1.7B、4B、8B 或 32B，请使用下面的代码而不是 3a) 下的代码，这需要额外的代码依赖项：

```bash
pip install safetensors huggingface_hub
```

然后使用以下代码（对`USE_MODEL`进行适当更改以选择所需的模型尺寸）

```python
USE_MODEL = "1.7B"

if USE_MODEL == "1.7B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
elif USE_MODEL == "4B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
elif USE_MODEL == "8B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
elif USE_MODEL == "14B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
elif USE_MODEL == "32B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
elif USE_MODEL == "30B-A3B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
else:
    raise ValueError("Invalid USE_MODEL name.")
    
repo_id = f"Qwen/Qwen3-{USE_MODEL}"
local_dir = f"Qwen3-{USE_MODEL}"

if not USE_REASONING_MODEL:
  repo_id = f"{repo_id}-Base"
  local_dir = f"{local_dir}-Base"
```

现在，下载权重并将其加载到 `model` 中：

```python
from llms_from_scratch.qwen3 import (
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

with device:
    model = Qwen3Model(QWEN3_CONFIG)

weights_dict = download_from_huggingface_from_snapshots(
    repo_id=repo_id,
    local_dir=local_dir
)
load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)  # only required for the MoE models
del weights_dict  # delete weight dictionary to free up disk space
```


 

#### 4) 初始化分词器

以下代码下载并初始化分词器：

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)
```



 

#### 5) 生成文本

最后，我们可以通过以下代码生成文本：

```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```





```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
```

使用 Qwen3 0.6B 推理模型时，输出应类似于下图所示（这是在 A100 上运行的）：

```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



对于较大的模型，您可能更喜欢流式变体，它在生成每个令牌后立即打印它：

```python
from llms_from_scratch.generate import generate_text_simple_stream

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_simple_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=150,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

```
 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



 

#### 专业技巧 1：通过编译加速推理


要获得高达 4 倍的加速，请替换

```python
model.to(device)
```

与

```python
model.to(device)
model = torch.compile(model)
```

注意：编译时会产生大量的多分钟前期成本，并且加速在第一次 `generate` 调用后生效。

下表显示了 A100 上后续 `generate` 调用的性能比较：

|                          |硬件|令牌/秒 |内存|
| ------------------------ | ----------------|------------------------ | -------- |
| Qwen3Model 0.6B | Nvidia A100 GPU | 25 | 25 1.49 GB | 1.49 GB
| Qwen3Model 0.6B编译| Nvidia A100 GPU | 107 | 107 1.99 GB | 1.99 GB


 
#### 专业技巧 2：使用 KV 缓存加速推理

在 CPU 上运行模型时，您可以使用 KV 缓存 `Qwen3Model` 直接替换显着提高推理性能。 （请参阅我的 [从头开始在 LLM 中理解和编码 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 文章，了解有关 KV 缓存的更多信息。）

```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```请注意，峰值内存使用量仅针对 Nvidia CUDA 设备列出，因为它更容易计算。然而，其他设备上的内存使用情况可能相似，因为它使用类似的精度格式，并且 KV 缓存存储导致生成的 150 个令牌文本的内存使用量甚至更低（但是，不同的设备可能以不同的方式实现矩阵乘法，并可能导致不同的峰值内存需求；并且 KV 缓存内存可能会因较长的上下文长度而大幅增加）。

|型号|模式|硬件|令牌/秒 | GPU 内存 (VRAM) |
| ---------------- | ----------------- | ---------------- | ---------- | ----------------- |
| Qwen3Model 0.6B |常规| Mac Mini M4 CPU | 1 | - |
| Qwen3Model 0.6B |常规编译| Mac Mini M4 CPU | 1 | - |
| Qwen3Model 0.6B | KV缓存| Mac Mini M4 CPU | 80| - |
| Qwen3Model 0.6B | KV缓存编译| Mac Mini M4 CPU | 137 | 137 - |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B |常规| Mac Mini M4 GPU | 21 | 21 - |
| Qwen3Model 0.6B |常规编译| Mac Mini M4 GPU |错误| - |
| Qwen3Model 0.6B | KV缓存| Mac Mini M4 GPU | 28 | 28 - |
| Qwen3Model 0.6B | KV缓存编译| Mac Mini M4 GPU |错误| - |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B |常规| Nvidia A100 GPU | 26 | 26 1.49 GB | 1.49 GB
| Qwen3Model 0.6B |常规编译| Nvidia A100 GPU | 107 | 107 1.99 GB | 1.99 GB
| Qwen3Model 0.6B | KV缓存| Nvidia A100 GPU | 25 | 25 1.47 GB | 1.47 GB
| Qwen3Model 0.6B | KV缓存编译| Nvidia A100 GPU | 90 | 90 1.48 GB | 1.48 GB

请注意，上述所有设置都经过测试，可产生相同的文本输出。



 

#### 专业技巧 3：批量推理

我们可以通过批量推理进一步提高吞吐量。虽然这不是同类比较，但由于我们现在正在使用更多数量的输入序列运行​​推理，这会增加每秒的令牌吞吐量，同时会增加内存使用量。这只需要在准备提示方面进行少量代码修改。例如，考虑下面的批量提示：

```python
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
# ...

prompts = [
    "Give me a short introduction to neural networks.",
    "Give me a short introduction to machine learning.",
    "Give me a short introduction to deep learning models.",
    "Give me a short introduction to natural language processing.",
    "Give me a short introduction to generative AI systems.",
    "Give me a short introduction to transformer architectures.",
    "Give me a short introduction to supervised learning methods.",
    "Give me a short introduction to unsupervised learning.",
]

tokenized_prompts = [tokenizer.encode(p) for p in prompts]
max_len = max(len(t) for t in tokenized_prompts)
padded_token_ids = [
    t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
]
input_tensor = torch.tensor(padded_token_ids).to(device)

output_token_ids = generate_text_simple(
    model=model,
    idx=input_tensor,
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

KV 缓存版本的代码类似，只是它需要使用这些直接替换：

```python
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```


下面的实验是在批量大小为 8 的情况下运行的。

|型号|模式|硬件|批量大小|令牌/秒 | GPU 内存 (VRAM) |
| ---------------- | ----------------- | ---------------- | ---------- | ---------- | ----------------- |
| Qwen3Model 0.6B |常规| Mac Mini M4 CPU | 8 | 2 | - |
| Qwen3Model 0.6B |常规编译| Mac Mini M4 CPU | 8 | - | - |
| Qwen3Model 0.6B | KV缓存| Mac Mini M4 CPU | 8 | 92 | 92 - |
| Qwen3Model 0.6B | KV缓存编译| Mac Mini M4 CPU | 8 | 128 | 128 - |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B |常规| Mac Mini M4 GPU | 8 | 36 | 36 - |
| Qwen3Model 0.6B |常规编译| Mac Mini M4 GPU | 8 | - | - |
| Qwen3Model 0.6B | KV缓存| Mac Mini M4 GPU | 8 | 61 | 61 - |
| Qwen3Model 0.6B | KV缓存编译| Mac Mini M4 GPU | 8 | - | - |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B |常规| Nvidia A100 GPU | 8 | 184 | 184 2.19 GB | 2.19 GB
| Qwen3Model 0.6B |常规编译| Nvidia A100 GPU | 8 | 351 | 351 2.19 GB | 2.19 GB
| Qwen3Model 0.6B | KV缓存| Nvidia A100 GPU | 8 | 140 | 140 3.13 GB | 3.13 GB
| Qwen3Model 0.6B | KV缓存编译| Nvidia A100 GPU | 8 | 280 | 280 1.75 GB | 1.75 GB