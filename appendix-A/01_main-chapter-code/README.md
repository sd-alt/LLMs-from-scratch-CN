# 附录 A：PyTorch 简介

### 主要章节代码

- [code-part1.ipynb](code-part1.ipynb) 包含本章中出现的所有 A.1 至 A.8 节代码
- [code-part2.ipynb](code-part2.ipynb) 包含本章中出现的所有 A.9 节 GPU 代码
- [DDP-script.py](DDP-script.py) 包含演示多 GPU 使用的脚本（请注意，Jupyter Notebooks 仅支持单 GPU，因此这是一个脚本，而不是笔记本）。您可以将其运行为 `python DDP-script.py`。如果您的机器有超过 2 个 GPU，请以 `CUDA_VISIBLE_DEVIVES=0,1 python DDP-script.py` 运行。
- [exercise-solutions.ipynb](exercise-solutions.ipynb)包含本章的练习解决方案

### 可选代码

- [DDP-script-torchrun.py](DDP-script-torchrun.py) 是 `DDP-script.py` 脚本的可选版本，它通过 PyTorch `torchrun` 命令运行，而不是通过 `multiprocessing.spawn` 自己生成和管理多个进程。 `torchrun`命令的优点是自动处理分布式初始化，包括多节点协调，这稍微简化了设置过程。您可以通过 `torchrun --nproc_per_node=2 DDP-script-torchrun.py` 使用此脚本