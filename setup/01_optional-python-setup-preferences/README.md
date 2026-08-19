# Python 设置技巧



有多种方法可以安装 Python 并设置计算环境。在这里，我分享一下我的个人喜好。

<br>

> **注：**
> 如果您正在 Google Colab 上运行任何笔记本并想要安装依赖项，只需在笔记本顶部的新单元中运行以下代码并跳过本教程的其余部分：
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

下面的其余部分描述了如何管理本地计算机上的 Python 环境和包。

我一直是 [Conda](https://anaconda.org/anaconda/conda) 和 [pip](https://pypi.org/project/pip/) 的长期用户，但最近，[uv](https://github.com/astral-sh/uv) 软件包获得了巨大的关注，因为它提供了一种更快、更有效的方式来安装软件包和解决依赖关系。

我建议从*选项 1：使用 uv* 开始，因为它是 2025 年更现代的方法。如果您遇到*选项 1* 的问题，请考虑*选项 2：使用 Conda*。

在本教程中，我使用的是运行 macOS 的计算机，但此工作流程对于 Linux 计算机类似，也可能适用于其他操作系统。


 
# 选项 1：使用 uv

本节将指导您通过 `uv pip` 接口使用 `uv` 完成 Python 设置和软件包安装过程。对于大多数以前使用过 pip 的 Python 用户来说，`uv pip` 界面可能比原生 `uv` 命令更熟悉。 
> **注：**
> 还有其他方法可以安装 Python 并使用 `uv`。例如，您可以直接通过 `uv` 安装 Python，并使用 `uv add` 而不是 `uv pip install` 来实现更快的包管理。
>
> 如果您是 macOS 或 Linux 用户并且更喜欢本机 `uv` 命令，请参阅 [./native-uv.md 教程](./native-uv.md)。我还建议查看官方 [`uv` 文档](https://docs.astral.sh/uv/)。
>
> `uv add` 语法也适用于 Windows 用户。但是，我发现 `pyproject.toml` 中的某些依赖项会在 Windows 上导致问题。因此，对于 Windows 用户，我推荐使用 `pixi`，它具有与 `uv add` 类似的 `pixi add` 工作流程。有关更多信息，请参阅 [./native-pixi.md 教程](./native-pixi.md)。
>
> 虽然 `uv add` 和 `pixi add` 提供了额外的速度优势，但我认为 `uv pip` 稍微更加用户友好，使其成为初学者的一个很好的起点。但是，如果您不熟悉 Python 包管理，本机 `uv` 接口也是一个从头开始学习它的好机会。这也是我现在使用 `uv` 的方式，但我意识到，如果您来自 `pip` 和 `conda`，进入的门槛会更高一些。




 
## 1.安装Python（如果没有安装）

如果您之前没有在系统上手动安装过 Python，我强烈建议您这样做。这有助于防止与操作系统的内置 Python 安装发生潜在冲突，从而导致出现问题。

但是，即使您之前已经在系统上安装了 Python，也可以通过在终端中执行以下代码来检查是否安装了现代版本的 Python（我推荐 3.10 或更高版本）：

```bash
python --version
```
如果它返回 3.10 或更高版本，则无需采取进一步操作。

 
> **注：**
> 如果 `python --version` 指示未安装 Python 版本，您可能还需要检查 `python3 --version`，因为您的系统可能配置为使用 `python3` 命令。

 
> **注：**
> 我建议安装比最新版本至少早 2 个版本的 Python 版本，以确保 PyTorch 兼容性。例如，如果最新版本是 Python 3.13，我建议安装版本 3.10 或 3.11。否则，如果 Python 未安装或者版本较旧，您可以按照如下所述为您的操作系统安装它。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-not-found.png" width="500" height="auto" alt="No Python Found">

<br>

**Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

<br>

**macOS**

如果您使用 Homebrew，请使用以下命令安装 Python：

```bash
brew install python@3.10
```

或者，从官方网站下载并运行安装程序：[https://www.python.org/downloads/](https://www.python.org/downloads/)。


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-version.png" width="700" height="auto" alt="Python version">

<br>

**Windows**

从官方网站下载并运行安装程序：[https://www.python.org/downloads/](https://www.python.org/downloads/)。


 

## 2.创建虚拟环境

我强烈建议在单独的虚拟环境中安装 Python 包，以避免修改操作系统可能依赖的系统范围的包。要在当前文件夹中创建虚拟环境，请执行以下三个步骤。

<br>

**1.安装紫外线**

```bash
pip install uv
```

<br>

**2.创建虚拟环境**

```bash
uv venv --python=python3.10
```

<br>

**3.激活虚拟环境**

```bash
source .venv/bin/activate
```

 
> **注：**
> 如果您使用的是 Windows，则可能需要将上面的命令替换为 `source .venv/Scripts/activate` 或 `.venv/Scripts/activate`。



请注意，每次启动新的终端会话时都需要激活虚拟环境。例如，如果您重新启动终端或计算机并希望第二天继续处理该项目，只需在项目文件夹中运行 `source .venv/bin/activate` 即可重新激活您的虚拟环境。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-1.png" width="600" height="auto" alt="Venv activated">

或者，您可以通过执行命令 `deactivate` 来停用环境。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-2.png" width="800" height="auto" alt="Venv deactivated">

 
## 3. 安装软件包

激活虚拟环境后，您可以使用 `uv` 安装 Python 包。例如：

```bash
uv pip install packaging
```

要从 `requirements.txt` 文件（例如位于此 GitHub 存储库顶层的文件）安装所有必需的包，请运行以下命令，假设该文件与终端会话位于同一目录中：

```bash
uv pip install -r requirements.txt
```


或者，直接从存储库安装最新的依赖项：

```bash
uv pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-install.png" width="700" height="auto" alt="Uv install">

 

> **注：**
> 如果由于某些依赖关系（例如，如果您使用的是 Windows），您在使用上述以下命令时遇到问题，您可以随时回退到使用常规 pip：
> `pip install -r requirements.txt`
> 或
> `pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

 > **奖励材料的可选依赖项：**
> 要包含整个奖励材料中使用的可选依赖项，请从项目根目录安装 `bonus` 依赖项组：
> `uv pip install --group bonus`
> 如果您不想在稍后查看可选的奖励材料时单独安装它们，这非常有用。

<br>

**完成设置**

就是这样！您的环境现在应该已准备好运行存储库中的代码。

或者，您可以通过执行此存储库中的 `python_environment_check.py` 脚本来运行环境检查：

```bash
python setup/02_installing-python-libraries/python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/env-check.png" width="700" height="auto" alt="Environment check">

如果您遇到特定软件包的任何问题，请尝试使用以下命令重新安装它们：

```bash
uv pip install packagename
```

（这里，`packagename` 是一个占位符名称，需要替换为您遇到问题的包名称。）

如果问题仍然存在，请考虑 [在 GitHub 上打开讨论](https://github.com/rasbt/LLMs-from-scratch/discussions) 或完成下面的*选项 2：使用 Conda* 部分。

<br>

**开始使用代码**

一切设置完毕后，您就可以开始使用代码文件了。例如，通过运行以下命令启动 [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/)：

```bash
jupyter lab
```

 
> **注：**
> 如果您在使用 jupyter lab 命令时遇到问题，您还可以使用虚拟环境中的完整路径启动它。例如，在 Linux/macOS 上使用 `.venv/bin/jupyter lab` 或在 Windows 上使用 `.venv\Scripts\jupyter-lab`。

 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/jupyter.png" width="900" height="auto" alt="Uv install">

 
<br>
<br>
 

# 选项 2：使用 Conda



本节将指导您通过 [miniforge](https://github.com/conda-forge/miniforge) 使用 [`conda`](https://www.google.com/search?client=safari&rls=en&q=conda&ie=UTF-8&oe=UTF-8) 完成 Python 设置和软件包安装过程。

在本教程中，我使用的是运行 macOS 的计算机，但此工作流程对于 Linux 计算机类似，也可能适用于其他操作系统。


 
## 1.下载并安装Miniforge

从 GitHub 存储库 [（此处](https://github.com/conda-forge/miniforge)）下载 miniforge。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/download.png" alt="download" width="600px">

根据您的操作系统，这应该下载 `.sh`（macOS、Linux）或 `.exe` 文件（Windows）。

对于 `.sh` 文件，打开命令行终端并执行以下命令

```bash
sh ~/Desktop/Miniforge3-MacOSX-arm64.sh
```

其中 `Desktop/` 是 Miniforge 安装程序下载到的文件夹。在您的计算机上，您可能必须将其替换为 `Downloads/`。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/miniforge-install.png" alt="miniforge-install" width="600px">接下来，逐步执行下载说明，并按“Enter”键确认。


 
## 2.创建新的虚拟环境

安装成功完成后，我建议创建一个名为 `LLMs` 的新虚拟环境，您可以通过执行

```bash
conda create -n LLMs python=3.10
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/new-env.png" alt="new-env" width="600px">

> 许多科学计算库并不立即支持最新版本的Python。因此，在安装 PyTorch 时，建议使用早一两个版本的 Python 版本。例如，如果Python的最新版本是3.13，则建议使用Python 3.10或3.11。

接下来，激活新的虚拟环境（每次打开新的终端窗口或选项卡时都必须执行此操作）：

```bash
conda activate LLMs
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/activate-env.png" alt="activate-env" width="600px">


 
## 可选：设置终端样式

如果您想将终端设计得与我的类似，以便可以看到哪个虚拟环境处于活动状态，请查看 [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) 项目。

 
## 3. 安装新的 Python 库



要安装新的 Python 库，您现在可以使用 `conda` 软件包安装程序。例如，您可以安装 [JupyterLab](https://jupyter.org/install) 和 [水印](https://github.com/rasbt/watermark)，如下所示：

```bash
conda install jupyterlab watermark
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/conda-install.png" alt="conda-install" width="600px">



您还可以仍然使用 `pip` 来安装库。默认情况下，`pip` 应链接到新的 `LLms` conda 环境：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/check-pip.png" alt="check-pip" width="600px">

 
## 4. 安装 PyTorch

PyTorch 可以像使用 pip 的任何其他 Python 库或包一样安装。例如：

```bash
pip install torch
```

然而，由于 PyTorch 是一个具有 CPU 和 GPU 兼容代码的综合库，因此安装可能需要额外的设置和解释（有关更多信息，请参阅本书中的*A.1.3 安装 PyTorch*）。

强烈建议您查阅 PyTorch 官方网站 [https://pytorch.org](https://pytorch.org) 上的安装指南菜单。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/pytorch-installer.jpg" width="600px">

 
## 5.安装本书中使用的Python包和库

有关如何安装所需库的说明，请参阅本书中使用的 [安装 Python 包和库](../02_installing-python-libraries/README.md) 文档。

<br>

---




有疑问吗？请随时通过 [论坛](https://github.com/rasbt/LLMs-from-scratch/discussions) 与我们联系。