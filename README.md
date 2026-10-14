# 构建大型语言模型（从头开始）

该存储库包含用于开发、预训练和微调类似 GPT 的 LLM 的代码，并且是[《构建大型语言模型（从头开始）》](https://amzn.to/4fqvn0D) 书籍的官方代码存储库。

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

在 [*构建大型语言模型（从头开始）*](http://mng.bz/orYv) 中，您将通过从头开始逐步编码来学习和理解大型语言模型 (LLM) 如何从内到外工作。在本书中，我将指导您创建自己的大语言模型，并用清晰的文本、图表和示例解释每个阶段。

本书中描述的用于出于教育目的训练和开发您自己的小型但实用的模型的方法反映了创建大型基础模型（例如 ChatGPT 背后的模型）所使用的方法。此外，本书还包含用于加载较大预训练模型的权重以进行微调的代码。

- 链接到官方 [源代码存储库](https://github.com/rasbt/LLMs-from-scratch)
- [Manning 上的书籍链接（出版商的网站）](http://mng.bz/orYv)
- [链接至 Amazon.com 上的图书页面](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

要下载此存储库的副本，请单击 [下载 ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip) 按钮或在终端中执行以下命令：

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

（如果您从 Manning 网站下载了代码包，请考虑访问 GitHub 上的官方代码存储库 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 以获取最新更新。）

<br>
<br>


# 目录

请注意，此 `README.md` 文件是 Markdown (`.md`) 文件。如果您已从 Manning 网站下载了此代码包并在本地计算机上查看它，我建议您使用 Markdown 编辑器或预览器进行正确查看。如果您还没有安装 Markdown 编辑器，[Ghostwriter](https://ghostwriter.kde.org) 是一个不错的免费选择。

您也可以在浏览器中通过 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 查看 GitHub 上的此文件和其他文件，这会自动呈现 Markdown。

<br>
<br>

> **提示：**
> 如果您正在寻求有关安装 Python 和 Python 包以及设置代码环境的指导，我建议您阅读位于 [设置](setup) 目录中的 [README.md](setup/README.md) 文件。

<br>
<br>

[![代码测试 Linux](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![代码测试 Windows](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![代码测试 macOS](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)

- [故障排除指南](./troubleshooting.md)

|章节标题 |主代码（用于快速访问）|所有代码+补充|
|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| [设置建议](setup) <br/>[如何最好地阅读本书](https://sebastianraschka.com/blog/2025/reading-books.html) | - | - |
|第 1 章：理解大型语言模型 |没有代码 | - |
|第 2 章：使用文本数据 | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb)（摘要）<br/>- [练习解决方案.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb) | [./ch02](./ch02) |
|第 3 章：编码注意力机制 | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb)（摘要） <br/>- [练习解决方案.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03) |
|第 4 章：从头开始实施 GPT 模型 | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py)（摘要）<br/>- [运动解决方案.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04) |
|第 5 章：未标记数据的预训练 | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py)（摘要） <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py)（摘要） <br/>- [练习解决方案.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05) |
|第 6 章：文本分类微调 | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb) <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py) <br/>- [练习解决方案.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06) |
|第 7 章：按照说明进行微调 | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py)（摘要）<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py)（摘要）<br/>- [练习-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07) |
|附录 A：PyTorch 简介 | - [代码-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [代码-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [练习解决方案.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./附录-A](./appendix-A) |
|附录 B：参考资料和延伸阅读 |没有代码 | [./附录-B](./appendix-B) |
|附录 C：练习解决方案 | - [锻炼解决方案列表](appendix-C) | [./附录-C](./appendix-C) |
|附录 D：在训练循环中添加附加功能 | - [附录-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb) | [./附录-D](./appendix-D) |
|附录 E：使用 LoRA 进行参数高效的微调 | - [附录-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb) | [./附录-E](./appendix-E) |

<br>
 

下面的思维模型总结了本书所涵盖的内容。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">


<br>
 

## 先决条件

最重要的先决条件是扎实的 Python 编程基础。
有了这些知识，您将为探索大语言模型的迷人世界做好充分准备
并理解本书中提出的概念和代码示例。

如果您对深度神经网络有一定的经验，您可能会发现某些概念更熟悉，因为大语言模型是建立在这些架构之上的。

本书使用PyTorch从头开始实现代码，无需使用任何外部LLM库。虽然精通 PyTorch 不是先决条件，但熟悉 PyTorch 基础知识肯定有用。如果您是 PyTorch 的新手，附录 A 提供了 PyTorch 的简要介绍。或者，您也可以找到我的书《[PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/)》，这本书有助于了解基础知识。



<br>
 

## 硬件要求

本书主要章节中的代码旨在在合理的时间内在传统笔记本电脑上运行，并且不需要专门的硬件。这种方法确保了广泛的受众可以参与这些材料。此外，代码会自动使用 GPU（如果可用）。 （请参阅 [设置](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) 文档以获取其他建议。）


 
## 视频课程

[一个 17 小时 15 分钟的配套视频课程](https://www.manning.com/livevideo/master-and-build-large-language-models)，我在其中编写了本书每一章的代码。本课程被组织成反映本书结构的章节，因此它可以用作本书的独立替代品或补充性代码资源。

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>


 


## 配套书/续集

[*构建推理模型（从头开始）*](https://mng.bz/lZ5B) 虽然是一本独立的书，但可以被视为 *构建大型语言模型（从头开始）* 的续集。

它从预训练模型开始，实施不同的推理方法，包括推理时间缩放、强化学习和蒸馏，以提高模型的推理能力。与*构建大型语言模型（从头开始）*类似，[*构建推理模型（从头开始）*](https://mng.bz/lZ5B) 采用从头开始实施这些方法的实践方法。

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="120px"></a>

- 亚马逊链接（待定）
- [曼宁链接](https://mng.bz/lZ5B)
- [GitHub 存储库](https://github.com/rasbt/reasoning-from-scratch)

<br>

 
## 练习

本书的每一章都包含几个练习。附录 C 中总结了这些解决方案，相应的代码笔记本可在该存储库的主要章节文件夹中找到（例如，[./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)）。

除了代码练习之外，您还可以从 Manning 网站下载一份免费的 170 页 PDF，标题为 [测试自己构建大型语言模型（从头开始）](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch)。每章包含大约 30 个测验问题和解决方案，可帮助您测试您的理解情况。

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>

 
## 奖励材料

几个文件夹包含可选材料，作为对感兴趣读者的奖励：
- **设置**
  - [Python 设置提示](setup/01_optional-python-setup-preferences)
  - [安装本书中使用的 Python 包和库](setup/02_installing-python-libraries)
  - [Docker 环境搭建指南](setup/03_optional-docker-environment)

- **第 2 章：使用文本数据**
  - [字节对编码 (BPE) 标记器从头开始](ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb)
  - [比较各种字节对编码 (BPE) 实现](ch02/02_bonus_bytepair-encoder)
  - [了解嵌入层和线性层之间的区别](ch02/03_bonus_embedding-vs-matmul)
  - [数据加载器直觉与简单数字](ch02/04_bonus_dataloader-intuition)

- **第 3 章：编码注意力机制**
  - [比较高效的多头注意力实现](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [了解 PyTorch 缓冲区](ch03/03_understanding-buffers/understanding-buffers.ipynb)

- **第 4 章：从头开始实现 GPT 模型**
  - [FLOPs 分析](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV 缓存](ch04/03_kv-cache)
  - [注意替代方案](ch04/#attention-alternatives)
    - [分组查询注意](ch04/04_gqa)
    - [多头潜在注意力](ch04/05_mla)
    - [推拉窗注意](ch04/06_swa)
    - [门控 DeltaNet](ch04/08_deltanet)
    - [DeepSeek 稀疏注意力 (DSA)](ch04/09_dsa)
    - [跨层KV共享](ch04/10_kv-sharing)
  - [混合专家 (MoE)](ch04/07_moe)

- **第 5 章：未标记数据的预训练**
  - [替代重量加载方法](ch05/02_alternative_weight_loading/)
  - [在古腾堡项目数据集](ch05/03_bonus_pretraining_on_gutenberg) 上预训练 GPT
  - [在训练循环中添加附加功能](ch05/04_learning_rate_schedulers)
  - [优化预训练的超参数](ch05/05_bonus_hparam_tuning)
  - [构建用户界面以与预训练的 LLM](ch05/06_user_interface) 交互
  - [将 GPT 转换为 Llama](ch05/07_gpt_to_llama)
  - [高效内存模型权重加载](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [使用新代币扩展 Tiktoken BPE Tokenizer](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [加快 LLM 培训速度的 PyTorch 性能技巧](ch05/10_llm-training-speed)
  - [大语言模型架构](ch05/#llm-architectures-from-scratch)
    - [骆驼 3.2 从头开始](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
    - [Qwen3 密集和专家混合 (MoE) 从头开始](ch05/11_qwen3/)
    - [杰玛 3 从头开始](ch05/12_gemma3/)
    - [Olmo 3 从头开始](ch05/13_olmo3/)
    - [从头开始的小绫](ch05/15_tiny-aya/)
    - [Qwen3.5 从头开始](ch05/16_qwen3.5/)
    - [Gemma 4 E2B 和 E4B 从头开始](ch05/17_gemma4/)
  - [第 5 章与其他大语言模型作为直接替代（例如 Llama 3、Qwen 3）](ch05/14_ch05_with_other_llms/)
- **第 6 章：分类微调**
  - [额外实验微调不同层并使用更大的模型](ch06/02_bonus_additional-experiments)
  - [在 50k IMDb 电影评论数据集](ch06/03_bonus_imdb-classification) 上微调不同模型
  - [构建用户界面以与基于 GPT 的垃圾邮件分类器交互](ch06/04_user_interface)
- **第 7 章：按照说明进行微调**
  - [用于查找附近重复项并创建被动语态条目的数据集实用程序](ch07/02_dataset-utilities)
  - [使用 OpenAI API 和 Ollama 评估指令响应](ch07/03_model-evaluation)
  - [生成用于指令微调的数据集](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [改进指令微调的数据集](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [使用 Llama 3.1 70B 和 Ollama](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb) 生成偏好数据集
  - [LLM对齐的直接偏好优化（DPO）](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [构建用户界面以与指令微调 GPT 模型](ch07/06_user_interface) 交互

来自 [从头开始推理](https://github.com/rasbt/reasoning-from-scratch) 存储库的更多奖励材料：

- **Qwen3（从头开始）基础知识**
  - [Qwen3 源代码演练](https://github.com/rasbt/reasoning-from-scratch/blob/main/chC/01_main-chapter-code/chC_main.ipynb)
  - [优化Qwen3](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM)

- **评估**
  - [基于验证器的评估 (MATH-500)](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03)
  - [多项选择评估 (MMLU)](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/02_mmlu)
  - [LLM排行榜评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/03_leaderboards)
  - [大语言模型法官评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/04_llm-judge)
- **推理缩放**
  - [自洽](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch04/01_main-chapter-code/ch04_main.ipynb)
  - [自我完善](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch05/01_main-chapter-code/ch05_main.ipynb)

- **强化学习** (RL)
  - [RLVR 与 GRPO 从头开始](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/01_main-chapter-code/ch06_main.ipynb)


<br>
 

## 问题、反馈以及对此存储库的贡献


我欢迎各种反馈，最好通过 [曼宁论坛](https://livebook.manning.com/forum?product=raschka&page=1) 或 [GitHub 讨论](https://github.com/rasbt/LLMs-from-scratch/discussions) 进行分享。同样，如果您有任何疑问或只是想向其他人传达想法，也请随时在论坛中发布。

请注意，由于此存储库包含与印刷书籍相对应的代码，因此我目前无法接受扩展主要章节代码内容的贡献，因为它会引入与实体书的偏差。保持一致有助于确保每个人都有顺畅的体验。


 
## 引文

如果您发现这本书或代码对您的研究有用，请考虑引用它。

芝加哥风格的引文：

> 拉斯卡，塞巴斯蒂安。 *构建大型语言模型（从头开始）*。曼宁，2024 年。ISBN：978-1633437166。

BibTeX 条目：

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```