# 加载预训练权重的替代方法

该文件夹包含替代权重加载策略，以防 OpenAI 无法提供权重。

- [Weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb)：（推荐）包含从我通过转换原始 TensorFlow 权重创建的 PyTorch 状态字典加载权重的代码

- [Weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb)：包含通过 `transformers` 库从 Hugging Face Model Hub 加载权重的代码

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb)：包含直接通过 `safetensors` 库从 Hugging Face 模型中心加载权重的代码（跳过 Hugging Face 变压器模型的实例化）