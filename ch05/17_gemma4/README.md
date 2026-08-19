# 杰玛 4

该目录包含一个独立的纯文本 Gemma 4 笔记本，它是根据 Gemma 3 参考笔记本构建的，并针对密集的 `google/gemma-4-E2B` 和 `google/gemma-4-E4B` 检查点进行了调整。

- [独立-gemma4.ipynb](./standalone-gemma4.ipynb)在纯PyTorch中实现共享Gemma 4密集架构，并通过`CHOOSE_MODEL`在E2B和E4B参考配置之间切换。