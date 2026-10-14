# 故障排除指南

此页面收集了在阅读本书时遇到的常见问题和设置技巧。

 
## 笔记本图像加载问题

章节笔记本使用 `https://sebastianraschka.com/images/LLMs-from-scratch-images/...` 上托管的 Markdown 图像链接。这使得存储库下载大小易于管理，但这也意味着图像取决于图像主机和网络连接。

如果 `.ipynb` 笔记本中的图像未呈现：

- 直接在浏览器中打开图像 URL 之一，例如 [https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp)。
- 如果 URL 也无法在浏览器中加载，则问题可能是临时网站、DNS、VPN、代理、防火墙或本地网络问题，而不是笔记本电脑问题。
- 我建议在不同的设备或网络上仔细检查 URL（例如，尝试在手机上打开图像）；如果图像在您的手机上加载正常，则可能表明您的计算机上存在 VPN 或防火墙问题。
- 如果图像也无法在您的手机上加载，请随时打开 GitHub [Issue](https://github.com/rasbt/LLMs-from-scratch/issues) 来帮助我进一步调试。

 
## 更新存储库时保留个人笔记本更改

如果您想在修改笔记本的同时接收存储库更新，请首先分叉存储库，然后克隆您的分叉。主要书籍笔记本与印刷书籍保持同步，通常不会更改，除非进行了关键修复。大多数存储库更新都会添加奖励材料。

笔记本文件是 JSON 文件，因此 Git 差异和合并冲突可能难以阅读。为了避免不必要的冲突，我建议将您的实验与跟踪的笔记本分开：

- 在更改之前复制笔记本，例如从 `ch02.ipynb` 到 `ch02_experiments.ipynb`。
- 将您的草稿笔记本保存在单独的文件夹或您自己的分支中。
- 使用 `upstream` 远程从原始存储库获取更新，然后仅在需要这些更新时合并或变基。

要创建一个分叉并克隆它：

1. 打开[https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
2. 单击 GitHub 右上角的 **Fork** 按钮。
3. 克隆您的 fork，将 `YOUR-USERNAME` 替换为您的 GitHub 用户名：

```bash
git clone https://github.com/YOUR-USERNAME/LLMs-from-scratch.git
cd LLMs-from-scratch
```

然后将原始存储库添加为 `upstream`，以便您可以获取未来的更新：

```bash
git remote add upstream https://github.com/rasbt/LLMs-from-scratch.git
git fetch upstream
git merge upstream/main
```

如果您确实需要合并已编辑的笔记本，请考虑安装 [`nbdime`](https://nbdime.readthedocs.io/) 以获得笔记本感知的差异和合并工具：

```bash
pip install nbdime
nbdime config-git --enable
```

有关更多上下文，请参阅 [#1015](https://github.com/rasbt/LLMs-from-scratch/issues/1015)。

 
## Apple Silicon 和 MPS 支持

某些笔记本和脚本在可用时使用 `cuda`，否则回退到 `cpu`，而不选择 Apple 的 `mps` 后端。在许多地方，`mps` 支持的省略是有意的，因为早期的 PyTorch/MPS 版本在几个示例中产生了不稳定或不同的结果，特别是在训练和微调期间。

如果您使用的是 Apple Silicon Mac，并看到不同的损失、急剧的损失峰值、生成的文本质量不佳或结果与本书不匹配，请首先在 `cpu` 上重新运行示例。为了通过书籍匹配行为进行更快的训练，我建议在本地 NVIDIA GPU 或云 GPU 上使用 `cuda`。

较新的 PyTorch 版本可能会改进 MPS 行为，如果仔细验证结果，您可以在本地尝试 `mps`。但是，如果您自己将 `mps` 支持添加到脚本中，请记住特定于 CUDA 的选项（例如 `pin_memory=True`、`torch.compile` 和 DDP/多 GPU 代码）可能需要单独的防护。

有关更多上下文，请参阅 [#977](https://github.com/rasbt/LLMs-from-scratch/issues/977)、[#625](https://github.com/rasbt/LLMs-from-scratch/discussions/625)、[#644](https://github.com/rasbt/LLMs-from-scratch/discussions/644)、[#442](https://github.com/rasbt/LLMs-from-scratch/discussions/442) 和 [#846](https://github.com/rasbt/LLMs-from-scratch/issues/846)。

 
## 其他问题

对于其他问题，请随时打开新的 GitHub [Issue](https://github.com/rasbt/LLMs-from-scratch/issues)。