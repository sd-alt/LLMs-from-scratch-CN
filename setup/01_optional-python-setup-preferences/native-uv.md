# 原生 uv Python 和包管理

对于那些更喜欢 `uv` 的本机命令而不是 `uv pip` 界面的人来说，本教程是 [README.md](./README.md) 文档中*选项 1：使用 uv* 的替代方案。虽然 `uv pip` 比纯 `pip` 更快，但 `uv` 的本机接口甚至比 `uv pip` 更快，因为它的开销更少，并且不必处理对 PyPy 包依赖项管理的遗留支持。

下表提供了不同依赖项和包管理方法的速度比较。速度比较特指安装过程中的包依赖性解析，而不是已安装包的运行时性能。请注意，该项目的软件包安装是一次性过程，因此根据整体便利性（而不仅仅是安装速度）来选择首选方法是合理的。


|命令|速度比较|
|------------------------|------------------|
| `conda install <pkg>` |最慢（基线）|
| `pip install <pkg>` |比上面快 2-10 倍 |
| `uv pip install <pkg>`|比上面快 5-10 倍 |
| `uv add <pkg>` |比上面快 2-5 倍 |

本教程重点介绍 `uv add`。


否则，与 [README.md](./README.md) 中的*选项 1：使用 uv* 类似，本教程将指导您使用 `uv` 完成 Python 设置和包安装过程。

在本教程中，我使用的是运行 macOS 的计算机，但此工作流程对于 Linux 计算机类似，也可能适用于其他操作系统。


 
## 1.安装uv

Uv 可以按如下方式安装，具体取决于您的操作系统。

<br>

**macOS 和 Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

<br>

**Windows**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

 

> **注：**
> 更多安装选项请参考官方[uv文档](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)。

 
## 2.安装Python包和依赖项

要从 `pyproject.toml` 文件（例如位于此 GitHub 存储库顶层的文件）安装所有必需的包，请运行以下命令，假设该文件与终端会话位于同一目录中：

```bash
uv sync --dev --python 3.11
```> **注：**
> 如果您的系统上没有可用的 Python 3.11，uv 将为您下载并安装它。
> 我建议使用比最新版本至少早 1-3 个版本的 Python 版本，以确保 PyTorch 兼容性。例如，如果最新版本是Python 3.13，我建议使用版本3.10、3.11、3.12。您可以通过访问 [python.org](https://www.python.org/downloads/) 找到最新的 Python 版本。

> **注：**
> 如果由于某些依赖关系（例如，如果您使用的是 Windows），您在使用上述以下命令时遇到问题，您可以随时回退到常规 pip：
> `uv add pip`
> `uv run python -m pip install -U -r requirements.txt`


请注意，上面的 `uv sync` 命令将通过 `.venv` 子文件夹创建一个单独的虚拟环境。 （如果您想删除虚拟环境并从头开始，只需删除 `.venv` 文件夹即可。）

您可以通过 `uv add` 安装 `pyproject.toml` 中未指定的新软件包，例如：

```bash
uv add packaging
```

您可以通过 `uv remove` 删除软件包，例如，

```bash
uv remove packaging
```



 
## 3.运行Python代码

<br>

您的环境现在应该已准备好运行存储库中的代码。

或者，您可以通过执行此存储库中的 `python_environment_check.py` 脚本来运行环境检查：

```bash
uv run python setup/02_installing-python-libraries/python_environment_check.py
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-run-check.png?1" width="700" height="auto" alt="Uv install">


<br>

**启动 JupyterLab**

您可以通过以下方式启动 JupyterLab 实例：

```bash
uv run jupyter lab
```

**跳过 `uv run` 命令**

如果您觉得输入 `uv run` 很麻烦，您可以按照如下所述手动激活虚拟环境。

在 macOS/Linux 上：

```bash
source .venv/bin/activate
```

在 Windows 上（PowerShell）：

```bash
.venv\Scripts\activate
```

然后，您可以通过运行脚本

```bash
python script.py
```

并通过以下方式启动 JupyterLab

```bash
jupyter lab
```

 
> **注：**
> 如果您在使用 jupyter lab 命令时遇到问题，您还可以使用虚拟环境中的完整路径启动它。例如，在 Linux/macOS 上使用 `.venv/bin/jupyter lab` 或在 Windows 上使用 `.venv\Scripts\jupyter-lab`。

 


 

## 可选：手动管理虚拟环境

或者，您仍然可以使用 `uv pip install` 直接从存储库安装依赖项。但请注意，这不会像 `uv add` 那样在 `uv.lock` 文件中记录依赖项。此外，它还需要手动创建和激活虚拟环境：

<br>

**1.创建新的虚拟环境**运行以下命令手动创建一个新的虚拟环境，该环境将通过新的 `.venv` 子文件夹保存：

```bash
uv venv --python=python3.10
```

<br>

**2.激活虚拟环境**

接下来，我们需要激活这个新的虚拟环境。

在 macOS/Linux 上：

```bash
source .venv/bin/activate
```

在 Windows 上（PowerShell）：

```bash
.venv\Scripts\activate
```

<br>

**3.安装依赖项**

最后，我们可以使用 `uv pip` 接口从远程位置安装依赖项：

```bash
uv pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```



---

有疑问吗？请随时访问 [论坛](https://github.com/rasbt/LLMs-from-scratch/discussions)。