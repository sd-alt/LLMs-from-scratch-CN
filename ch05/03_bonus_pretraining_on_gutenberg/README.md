# 在古腾堡项目数据集上预训练 GPT

此目录中的代码包含用于在 Project Gutenberg 提供的免费书籍上训练小型 GPT 模型的代码。

正如古腾堡计划网站所述，“绝大多数古腾堡计划电子书在美国属于公共领域。”

请阅读 [古腾堡计划权限、许可和其他常见请求](https://www.gutenberg.org/policy/permission.html) 页面，了解有关使用古腾堡计划提供的资源的更多信息。

 
## 如何使用此代码

 

### 1) 下载数据集

在本节中，我们使用 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub 存储库中的代码从古腾堡项目下载书籍。

截至撰写本文时，这将需要大约 50 GB 的磁盘空间并需要大约 10-15 小时，但这可能更多地取决于古腾堡计划自那时以来的增长程度。

 
#### 适用于 Linux 和 macOS 用户的下载说明


Linux和macOS用户可以按照以下步骤下载数据集（如果您是Windows用户，请参阅下面的注释）：

1. 将 `03_bonus_pretraining_on_gutenberg` 文件夹设置为工作目录，以将 `gutenberg` 存储库本地克隆到此文件夹中（这是运行提供的脚本 `prepare_dataset.py` 和 `pretraining_simple.py` 所必需的）。例如，当位于 `LLMs-from-scratch` 存储库的文件夹中时，通过以下方式导航到 *03_bonus_pretraining_on_gutenberg* 文件夹：
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 克隆 `gutenberg` 存储库：
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 导航到本地克隆的 `gutenberg` 存储库的文件夹：
```bash
cd gutenberg
```

4. 从 `gutenberg` 存储库文件夹安装 *requirements.txt* 中定义的所需包：
```bash
pip install -r requirements.txt
```

5. 下载数据：
```bash
python get_data.py
```

6. 返回`03_bonus_pretraining_on_gutenberg`文件夹
```bash
cd ..
```

 
#### 针对 Windows 用户的特殊说明

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 代码与 Linux 和 macOS 兼容。但是，Windows 用户必须进行一些小的调整，例如将 `shell=True` 添加到 `subprocess` 调用并替换 `rsync`。

或者，在 Windows 上运行此代码的更简单方法是使用“Windows Subsystem for Linux”(WSL) 功能，该功能允许用户在 Windows 中使用 Ubuntu 运行 Linux 环境。更多信息请阅读[微软官方安装说明](https://learn.microsoft.com/en-us/windows/wsl/install)和[教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。使用 WSL 时，请确保已安装 Python 3（通过 `python3 --version` 检查，或者使用 `sudo apt-get install -y python3.10` 来安装 Python 3.10）并在那里安装以下软件包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **注：**
> 有关如何设置 Python 和安装软件包的说明，请参阅 [可选 Python 设置首选项](../../setup/01_optional-python-setup-preferences/README.md) 和 [安装 Python 库](../../setup/02_installing-python-libraries/README.md)。
>
> （可选）此存储库提供了运行 Ubuntu 的 Docker 映像。有关如何使用提供的 Docker 映像运行容器的说明，请参阅 [可选 Docker 环境](../../setup/03_optional-docker-environment/README.md)。

 
### 2) 准备数据集

接下来，运行 `prepare_dataset.py` 脚本，该脚本将（截至撰写本文时为 60,173 个）文本文件连接成更少的较大文件，以便可以更有效地传输和访问它们：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **提示：**
> 请注意，生成的文件以纯文本格式存储，并且为了简单起见未进行预标记。但是，如果您计划更频繁地使用数据集或训练多个时期，您可能需要更新代码以预先标记化的形式存储数据集，以节省计算时间。有关更多信息，请参阅本页底部的*设计决策和改进*。

> **提示：**
> 您可以选择较小的文件大小，例如 50 MB。这将产生更多文件，但可能有助于在少量文件上更快地进行预训练以进行测试。


 
### 3) 运行预训练脚本

您可以按如下方式运行预训练脚本。请注意，出于说明目的，附加命令行参数显示为默认值：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出将按以下方式格式化：> 文件总数：3
> 标记文件 1（共 3 个）：data_small/combined_1.txt
> 培训...
> Ep 1（步骤 0）：列车损失 9.694，Val 损失 9.724
> Ep 1（步骤 100）：列车损失 6.672，Val 损失 6.683
> Ep 1（步骤 200）：列车损失 6.543，Val 损失 6.434
> Ep 1（步骤 300）：列车损失 5.772，Val 损失 6.313
> Ep 1（步骤 400）：列车损失 5.547，Val 损失 6.249
> Ep 1（步骤 500）：列车损失 6.182，Val 损失 6.155
> Ep 1（步骤 600）：列车损失 5.742，Val 损失 6.122
> Ep 1（步骤 700）：列车损失 6.309，Val 损失 5.984
> Ep 1（步骤 800）：列车损失 5.435，Val 损失 5.975
> Ep 1（步骤 900）：列车损失 5.582，Val 损失 5.935
> ...
> Ep 1（步骤 31900）：列车损失 3.664，Val 损失 3.946
> Ep 1（步骤 32000）：列车损失 3.493，Val 损失 3.939
> Ep 1（步骤 32100）：列车损失 3.940，Val 损失 3.961
> 已保存 model_checkpoints/model_pg_32188.pth
> 书籍处理时间 3h 46m 55s
> 总耗时 3h 46m 55s
> 剩余书籍预计到达时间：7 小时 33 分钟 50 秒
> 对文件 2（共 3 个）进行标记：data_small/combined_2.txt
> 培训...
> Ep 1（步骤 32200）：列车损失 2.982，Val 损失 4.094
> Ep 1（步骤 32300）：列车损失 3.920，Val 损失 4.097
> ...


 
> **提示：**
> 实际上，如果您使用的是 macOS 或 Linux，除了在终端上打印日志输出外，我建议使用 `tee` 命令将日志输出保存到 `log.txt` 文件中：

```bash
python -u pretraining_simple.py | tee log.txt
```

 
> **警告：**
> 请注意，在 V100 GPU 上，对 `gutenberg_preprocessed` 文件夹中约 500 Mb 文本文件之一进行训练大约需要 4 小时。
> 该文件夹包含 47 个文件，需要大约 200 小时（超过 1 周）才能完成。您可能想在较少数量的文件上运行它。


 
## 设计决策和改进

请注意，此代码的重点是为了教育目的而保持简单和最小化。可以通过以下方式改进代码，以提高建模性能和训练效率：

1. 修改 `prepare_dataset.py` 脚本以从每个书籍文件中删除古腾堡样板文本。
2. 更新数据准备和加载实用程序以对数据集进行预标记并将其保存为标记化形式，以便每次调用预训练脚本时不必重新标记化。
3. 通过添加 [附录 D：向训练循环添加附加功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中介绍的功能来更新 `train_model_simple` 脚本，即余弦衰减、线性预热和梯度裁剪。
4. 更新预训练脚本以保存优化器状态（请参阅第 5 章中的 *5.4 在 PyTorch* 中加载和保存权重部分；[ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)），并添加用于加载现有模型和优化器检查点的选项，并在训练运行中断时继续训练。
5. 添加更高级的记录器（例如权重和偏差）以实时查看损失和验证曲线
6. 添加分布式数据并行性 (DDP) 并在多个 GPU 上训练模型（请参阅附录 A 中的*A.9.3 使用多个 GPU 进行训练*部分；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。
7. 将 `previous_chapter.py` 脚本中从头开始的 `MultiheadAttention` 类替换为 [高效多头注意力实现](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) 奖励部分中实现的高效 `MHAPyTorchScaledDotProduct` 类，该部分通过 PyTorch 的 `nn.functional.scaled_dot_product_attention` 函数使用 Flash Attention。
8. 通过[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)（`model = torch.compile`）或[雷霆](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）优化模型来加速训练。
9. 实施梯度低阶投影（GaLore）以进一步加快预训练过程。只需将 `AdamW` 优化器替换为 [GaLore Python 库](https://github.com/jiaweizzhao/GaLore) 中提供的 `GaLoreAdamW` 即可实现此目的。