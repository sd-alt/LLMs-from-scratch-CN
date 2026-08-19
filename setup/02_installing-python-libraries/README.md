# 安装本书中使用的Python包和库

本文档提供了有关仔细检查已安装的 Python 版本和包的更多信息。 （有关安装 Python 和 Python 包的更多信息，请参阅 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 文件夹。）

我在本书中使用了此处列出的 [](https://github.com/rasbt/LLMs-from-scratch/blob/main/requirements.txt) 的以下库。这些库的新版本也可能兼容。但是，如果您的代码遇到任何问题，您可以尝试这些库版本作为后备。



> **注：**
> 如果您按照 [选项 1：使用 uv](../01_optional-python-setup-preferences/README.md) 中所述使用 `uv`，则可以在以下命令中通过 `uv pip` 替换 `pip`。例如，`pip install -r requirements.txt` 变为 `uv pip install -r requirements.txt`



为了最方便地安装这些需求，您可以使用此代码存储库根目录中的 `requirements.txt` 文件并执行以下命令：

```bash
pip install -r requirements.txt
```

或者，您可以通过 GitHub URL 安装它，如下所示：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt
```


然后，完成安装后，请检查所有软件包是否已安装并且是否是最新的

```bash
python python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_1.jpg" width="600px">

还建议通过运行此目录中的 `python_environment_check.ipynb` 来检查 JupyterLab 中的版本，理想情况下，这应该会给出与上面相同的结果。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_2.jpg" width="500px">

如果您看到以下问题，则您的 JupyterLab 实例可能连接到了错误的 conda 环境：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/jupyter-issues.jpg" width="450px">

在这种情况下，您可能需要使用 `watermark` 来检查是否使用 `--conda` 标志在正确的 conda 环境中打开了 JupyterLab 实例：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/watermark.jpg" width="350px">


 
## 安装 PyTorch

PyTorch 可以像使用 pip 的任何其他 Python 库或包一样安装。例如：

```bash
pip install torch
```

然而，由于 PyTorch 是一个具有 CPU 和 GPU 兼容代码的综合库，因此安装可能需要额外的设置和解释（有关更多信息，请参阅本书中的*A.1.3 安装 PyTorch*）。

强烈建议您查阅 PyTorch 官方网站 [https://pytorch.org](https://pytorch.org) 上的安装指南菜单。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/pytorch-installer.jpg" width="600px">

<br>



 
## JupyterLab 提示如果您在 JupyterLab 而不是 VSCode 中查看笔记本代码，请注意 JupyterLab（默认设置）在最新版本中存在滚动错误。我的建议是转到“设置”->“设置编辑器”并将“窗口模式”更改为“无”（如下图所示），这似乎可以解决该问题。


![Jupyter 故障 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter 故障 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

<br>

---




有疑问吗？请随时通过 [论坛](https://github.com/rasbt/LLMs-from-scratch/discussions) 与我们联系。