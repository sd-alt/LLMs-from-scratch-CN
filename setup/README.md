# 可选设置说明


本文档列出了设置计算机和使用此存储库中的代码的不同方法。我建议从上到下浏览不同的部分，然后决定哪种方法最适合您的需求。

 

## 快速入门

如果您的计算机上已经安装了 Python，那么最快的入门方法是通过从此代码存储库的根目录执行以下 pip 安装命令来安装 [../requirements.txt](../requirements.txt) 文件中的软件包要求：

```bash
pip install -r requirements.txt
```

<br>

> **注意：** 如果您在 Google Colab 上运行任何笔记本并想要安装依赖项，只需在笔记本顶部的新单元中运行以下代码：
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`
> 或者，克隆存储库后，您可以从项目根目录使用 `uv pip install --group bonus` 安装所有奖励材料的依赖项。如果您不想在稍后查看可选的奖励材料时单独安装它们，这非常有用。



在下面的视频中，我分享了我个人在计算机上设置Python环境的方法：

<br>
<br>

[![视频链接](https://img.youtube.com/vi/yAcWnfsZhzo/0.jpg)](https://www.youtube.com/watch?v=yAcWnfsZhzo)


 
# 本地设置

本节提供了在本地运行本书中的代码的建议。请注意，本书主要章节中的代码旨在在合理的时间内在传统笔记本电脑上运行，并且不需要专门的硬件。我在 M3 MacBook Air 笔记本电脑上测试了所有主要章节。此外，如果您的笔记本电脑或台式计算机具有 NVIDIA GPU，代码将自动利用它。

 
## 设置Python

如果您的计算机上尚未设置 Python，我已在以下目录中写了有关我个人的 Python 设置首选项的信息：

- [01_可选-python-setup-首选项](./01_optional-python-setup-preferences)
- [02_安装 python 库](./02_installing-python-libraries)

下面的*使用 DevContainers* 部分概述了在计算机上安装项目依赖项的替代方法。

 

## 使用 Docker DevContainers作为上面*设置 Python* 部分的替代方案，如果您更喜欢隔离项目依赖项和配置的开发设置，那么使用 Docker 是一个非常有效的解决方案。这种方法消除了手动安装软件包和库的需要，并确保了一致的开发环境。您可以找到有关设置 Docker 和使用 DevContainer 的更多说明：

- [03_可选-docker-环境](03_optional-docker-environment)

 

## Visual Studio 代码编辑器

代码编辑器有很多不错的选择。我的首选是流行的开源 [Visual Studio Code (VSCode)](https://code.visualstudio.com) 编辑器，它可以通过许多有用的插件和扩展轻松增强（有关更多信息，请参阅下面的 *VSCode 扩展* 部分）。 macOS、Linux 和 Windows 的下载说明可以在 [VSCode 主网站](https://code.visualstudio.com) 上找到。

 

## VSCode 扩展

如果您使用 Visual Studio Code (VSCode) 作为主要代码编辑器，则可以在 `.vscode` 子文件夹中找到推荐的扩展。这些扩展提供了对此存储库有用的增强功能和工具。

要安装它们，请在 VSCode 中打开此“setup”文件夹（文件 -> 打开文件夹...），然后单击右下角弹出菜单中的“安装”按钮。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/vs-code-extensions.webp?1" alt="1" width="700">

或者，您可以将 `.vscode` 扩展文件夹移动到此 GitHub 存储库的根目录中：

```bash
mv setup/.vscode ./
```

然后，每次打开 `LLMs-from-scratch` 主文件夹时，VSCode 都会自动检查系统上是否已安装推荐的扩展。

 

# 云资源

本节介绍运行本书中提供的代码的云替代方案。

虽然代码可以在没有专用 GPU 的传统笔记本电脑和台式机上运行，但配备 NVIDIA GPU 的云平台可以显着提高代码的运行时间，特别是在第 5 章到第 7 章中。

 

## 使用 Lightning Studio

为了在云端获得流畅的开发体验，我推荐[Lightning AI Studio](https://lightning.ai/)平台，它允许用户设置持久环境并在云端CPU和GPU上使用VSCode和Jupyter Lab。

启动新的 Studio 后，您可以打开终端并执行以下设置步骤来克隆存储库并安装依赖项：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -r requirements.txt
```（与 Google Colab 相比，这些只需要执行一次，因为 Lightning AI Studio 环境是持久的，即使您在 CPU 和 GPU 机器之间切换也是如此。）

然后，导航到要运行的 Python 脚本或 Jupyter Notebook。或者，您还可以轻松连接 GPU 以加速代码的运行时间，例如，当您在第 5 章中预训练 LLM 或在第 6 章和第 7 章中对其进行微调时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

 

## 使用 Google Colab

要在云中使用 Google Colab 环境，请转到 [https://colab.research.google.com/](https://colab.research.google.com/) 并从 GitHub 菜单中打开相应的章节笔记本，或者将笔记本拖到*上传*字段中，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


另请确保将相关文件（笔记本从中导入的数据集文件和 .py 文件）上传到 Colab 环境，如下所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


您可以选择通过更改*运行时*在 GPU 上运行代码，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


 

# 有问题吗？

如果您有任何疑问，请随时通过此 GitHub 存储库中的 [讨论](https://github.com/rasbt/LLMs-from-scratch/discussions) 论坛进行联系。