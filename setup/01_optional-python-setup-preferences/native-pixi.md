# 原生 pixi Python 和包管理

对于那些更喜欢 `pixi` 的本机命令而不是传统环境和包管理器（如 `conda` 和 `pip`）的人来说，本教程是 [`./native-uv.md`](native-uv.md) 文档的替代方案。

请注意，pixi 在底层使用 `uv add`，如 [`./native-uv.md`](native-uv.md) 中所述。

Pixi 和 uv 都是 Python 的现代包和环境管理工具，但 pixi 是一个多语言包管理器，不仅设计用于管理 Python，还设计用于管理其他语言（类似于 conda），而 uv 是针对超快速依赖项解析和包安装而优化的特定于 Python 的工具。

如果有人需要支持多种语言（不仅仅是 Python）的多语言包管理器，或者更喜欢类似于 conda 的声明式环境管理方法，那么他们可能会选择 pixi 而不是 uv。如需了解更多信息，请访问 [pixi 官方文档](https://pixi.sh/latest/)。

在本教程中，我使用的是运行 macOS 的计算机，但此工作流程对于 Linux 计算机类似，也可能适用于其他操作系统。

 
## 1.安装pixi

Pixi 可以按如下方式安装，具体取决于您的操作系统。

<br>

**macOS 和 Linux**

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

或

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

<br>

**Windows**

从官方 [文档](https://pixi.sh/latest/installation/#__tabbed_1_2) 下载安装程序或运行列出的 PowerShell 命令。



> **注：**
> 更多安装选项请参考[pixi官方文档](https://pixi.sh/latest/)。


 
## 1.安装Python

您可以使用 pixi 安装 Python：

```bash
pixi add python=3.10
```

> **注：**
> 我建议安装比最新版本至少早 2 个版本的 Python 版本，以确保 PyTorch 兼容性。例如，如果最新版本是 Python 3.13，我建议安装版本 3.10 或 3.11。您可以通过访问 [python.org](https://www.python.org) 找到最新的 Python 版本。

 
## 3.安装Python包和依赖项

要从 `pixi.toml` 文件（例如位于此 GitHub 存储库顶层的文件）安装所有必需的包，请运行以下命令，假设该文件与终端会话位于同一目录中：

```bash
pixi install
```

> **注：**
> 如果您遇到依赖项问题（例如，如果您使用的是 Windows），您可以随时回退到 pip：`pixi run pip install -U -r requirements.txt`

默认情况下，`pixi install` 将创建一个特定于项目的单独虚拟环境。

您可以通过 `pixi add` 安装 `pixi.toml` 中未指定的新软件包，例如：

```bash
pixi add packaging
```

您可以通过 `pixi remove` 删除软件包，例如，

```bash
pixi remove packaging
```

 
## 4. 运行Python代码

您的环境现在应该已准备好运行存储库中的代码。

或者，您可以通过执行此存储库中的 `python_environment_check.py` 脚本来运行环境检查：

```bash
pixi run python setup/02_installing-python-libraries/python_environment_check.py
```

<br>

**启动 JupyterLab**

您可以通过以下方式启动 JupyterLab 实例：

```bash
pixi run jupyter lab
```


---

有疑问吗？请随时通过 [论坛](https://github.com/rasbt/LLMs-from-scratch/discussions) 与我们联系。