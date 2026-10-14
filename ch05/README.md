# 第 5 章：未标记数据的预训练

 
## 主要章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主要章节代码

 
## 奖励材料

- [02_alternative_weight_loading](02_alternative_weight_loading) 包含从替代位置加载 GPT 模型权重的代码，以防 OpenAI 无法提供模型权重
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 包含在古腾堡计划的整个图书语料库上对大语言模型进行更长时间预训练的代码
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 包含实现更复杂训练功能的代码，包括学习率调度程序和梯度裁剪
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 包含可选的超参数调整脚本
- [06_user_interface](06_user_interface) 实现交互式用户界面以与预训练的 LLM 交互
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 包含一个奖励笔记本，展示如何通过 PyTorch 的 `load_state_dict` 方法更有效地加载模型权重
- [09_extending-tokenizers](09_extending-tokenizers) 包含 GPT-2 BPE 标记生成器的从头开始实现
- [10_llm-training-speed](10_llm-training-speed) 显示了提高 LLM 训练速度的 PyTorch 性能技巧
- [18_muon](18_muon) 解释了如何将 Muon 优化器与 GPT 模型训练设置结合使用

 
## 从头开始的大语言模型架构

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

 


- [07_gpt_to_llama](07_gpt_to_llama) 包含将 GPT 架构实现转换为 Llama 3.2 并从 Meta AI 加载预训练权重的分步指南
- [11_qwen3](11_qwen3) Qwen3 0.6B 和 Qwen3 30B-A3B（专家混合）的从头开始实现，包括用于加载基础、推理和编码模型变体的预训练权重的代码
- [12_gemma3](12_gemma3) Gemma 3 270M 的从头开始实现以及 KV 缓存的替代方案，包括加载预训练权重的代码
- [13_olmo3](13_olmo3) Olmo 3 7B 和 32B（Base、Instruct 和 Think 变体）的从头实现以及 KV 缓存的替代方案，包括加载预训练权重的代码
- [17_gemma4](17_gemma4) Gemma 4 E2B 和 E4B 密集变体的从头开始实现 
## 本章的代码视频

<br>
<br>

[![视频链接](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)