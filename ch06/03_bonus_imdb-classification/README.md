# 对 50k IMDb 电影评论的情绪进行分类的附加实验

## 概述

此文件夹包含其他实验，用于将第 6 章中的（解码器式）GPT-2 (2018) 模型与编码器式 LLM 进行比较，例如 [BERT (2018)](https://arxiv.org/abs/1810.04805)、[RoBERTa (2019)](https://arxiv.org/abs/1907.11692) 和 [ModernBERT (2024)](https://arxiv.org/abs/2412.13663)。我们没有使用第 6 章中的小型 SPAM 数据集，而是使用 IMDb 中的 50k 电影评论数据集（[数据集源](https://ai.stanford.edu/~amaas/data/sentiment/)）和二元分类目标，预测评论者是否喜欢这部电影。这是一个平衡的数据集，因此随机预测应产生 50% 的准确率。





|         |型号|测试精度 |
| -------- | ------------------------------------------- | ------------- |
| **1.1** | 124M GPT-2 基线 | 91.88% |
| **1.2** | 124M GPT-2 基线（带 Muon）| 92.40% |
| **2** | 340M 伯特 | 90.89% |
| **3** | 66M 蒸馏伯特 | 91.40% |
| **4** | 355M 罗伯塔 | 92.95% |
| **5** | 304M DeBERTa-v3 | 94.69% |
| **6** | 149M ModernBERT 基础 | 93.79% |
| **7** | 395M ModernBERT 大号 | 95.07% |
| **8** |逻辑回归基线 | 88.85% |






 
## 第 1 步：安装依赖项

通过安装额外的依赖项

```bash
pip install -r requirements-extra.txt
```

 
## 第 2 步：下载数据集

这些代码使用 IMDb（[数据集来源](https://ai.stanford.edu/~amaas/data/sentiment/)）的 50k 电影评论来预测电影评论是正面还是负面。

运行以下代码创建`train.csv`、`validation.csv`和`test.csv`数据集：

```bash
python download_prepare_dataset.py
```


 
## 步骤 3：运行模型

 
### 1) 124M GPT-2 基线

第6章使用的124M GPT-2模型，从预训练权重开始，并对所有权重进行微调：

```bash
python train_gpt.py --trainable_layers "all" --num_epochs 1
```

```
Ep 1 (Step 000000): Train loss 3.706, Val loss 3.853
Ep 1 (Step 000050): Train loss 0.682, Val loss 0.706
...
Ep 1 (Step 004300): Train loss 0.199, Val loss 0.285
Ep 1 (Step 004350): Train loss 0.188, Val loss 0.208
Training accuracy: 95.62% | Validation accuracy: 95.00%
Training completed in 9.48 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.64%
Validation accuracy: 92.32%
Test accuracy: 91.88%
```

<br>

替代的 [train_gpt_muon.py](train_gpt_muon.py) 脚本使用 PyTorch 的新 Muon 优化器针对非嵌入层运行相同的代码。有关 Muon 的更多信息，请参阅原始 [论文](https://arxiv.org/abs/2502.16982) 和 [../../ch05/18_muon](../../ch05/18_muon)。


```bash
python train_gpt_muon.py --trainable_layers "all" --num_epochs 1
```

```
Ep 1 (Step 000000): Train loss 2.659, Val loss 3.237
Ep 1 (Step 000050): Train loss 0.919, Val loss 0.799
...
Training accuracy: 98.12% | Validation accuracy: 91.88%
Training completed in 23.01 minutes.

Evaluating on the full datasets ...

Training accuracy: 97.45%
Validation accuracy: 92.52%
Test accuracy: 92.40%
```观察：Muon 似乎优化得更好/更快，但这也导致了这里训练集的过度拟合。

PS：训练时间无法直接比较，因为这是在不同的 GPU 上运行的。

<br>

---

<br>

 
### 2) 340M BERT


340M参数编码器式[BERT](https://arxiv.org/abs/1810.04805)型号：

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "bert"
```

```
Ep 1 (Step 000000): Train loss 0.848, Val loss 0.775
Ep 1 (Step 000050): Train loss 0.655, Val loss 0.682
...
Ep 1 (Step 004300): Train loss 0.146, Val loss 0.318
Ep 1 (Step 004350): Train loss 0.204, Val loss 0.217
Training accuracy: 92.50% | Validation accuracy: 88.75%
Training completed in 7.65 minutes.

Evaluating on the full datasets ...

Training accuracy: 94.35%
Validation accuracy: 90.74%
Test accuracy: 90.89%
```

<br>

---

<br>

 
### 3) 66M DistilBERT

66M 参数编码器式 [DistilBERT](https://arxiv.org/abs/1910.01108) 模型（从 340M 参数 BERT 模型中提取），从预训练权重开始，仅训练最后一个 Transformer 块加上输出层：



```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "distilbert"
```

```
Ep 1 (Step 000000): Train loss 0.693, Val loss 0.688
Ep 1 (Step 000050): Train loss 0.452, Val loss 0.460
...
Ep 1 (Step 004300): Train loss 0.179, Val loss 0.272
Ep 1 (Step 004350): Train loss 0.199, Val loss 0.182
Training accuracy: 95.62% | Validation accuracy: 91.25%
Training completed in 4.26 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.30%
Validation accuracy: 91.12%
Test accuracy: 91.40%
```
<br>

---

<br>

 
### 4) 355M 罗伯塔

355M 参数编码器式 [RoBERTa](https://arxiv.org/abs/1907.11692) 模型，从预训练权重开始，仅训练最后一个 Transformer 块加上输出层：


```bash
python train_bert_hf.py --trainable_layers "last_block" --num_epochs 1 --model "roberta"
```

```
Ep 1 (Step 000000): Train loss 0.695, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.670, Val loss 0.690
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.098
Ep 1 (Step 004350): Train loss 0.170, Val loss 0.086
Training accuracy: 98.12% | Validation accuracy: 96.88%
Training completed in 11.22 minutes.

Evaluating on the full datasets ...

Training accuracy: 96.23%
Validation accuracy: 94.52%
Test accuracy: 94.69%
```

<br>

---

<br>

 
### 5) 304M DeBERTa-v3

304M 参数编码器式 [DeBERTa-v3](https://arxiv.org/abs/2111.09543) 型号。 DeBERTa-v3 通过解开注意力和改进的位置编码对早期版本进行了改进。


```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "deberta-v3-base"
```

```
Ep 1 (Step 000000): Train loss 0.689, Val loss 0.694
Ep 1 (Step 000050): Train loss 0.673, Val loss 0.683
...
Ep 1 (Step 004300): Train loss 0.126, Val loss 0.149
Ep 1 (Step 004350): Train loss 0.211, Val loss 0.138
Training accuracy: 92.50% | Validation accuracy: 94.38%
Training completed in 7.20 minutes.

Evaluating on the full datasets ...

Training accuracy: 93.44%
Validation accuracy: 93.02%
Test accuracy: 92.95%
```

<br>

---

<br>



 
### 6) 149M ModernBERT 基础

[ModernBERT (2024)](https://arxiv.org/abs/2412.13663) 是 BERT 的优化重新实现，它结合了并行残差连接和门控线性单元 (GLU) 等架构改进，以提高效率和性能。它保持了 BERT 最初的预训练目标，同时在现代硬件上实现了更快的推理和更好的可扩展性。

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-base"
```



```
Ep 1 (Step 000000): Train loss 0.699, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.564, Val loss 0.606
...
Ep 1 (Step 004300): Train loss 0.086, Val loss 0.168
Ep 1 (Step 004350): Train loss 0.160, Val loss 0.131
Training accuracy: 95.62% | Validation accuracy: 93.75%
Training completed in 10.27 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.72%
Validation accuracy: 94.00%
Test accuracy: 93.79%
```

<br>

---

<br>


 
### 7) 395M ModernBERT 大号

与上面相同，但使用更大的 ModernBERT 变体。

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-large"
```



```
Ep 1 (Step 000000): Train loss 0.666, Val loss 0.662
Ep 1 (Step 000050): Train loss 0.548, Val loss 0.556
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.115
Ep 1 (Step 004350): Train loss 0.154, Val loss 0.116
Training accuracy: 96.88% | Validation accuracy: 95.62%
Training completed in 27.69 minutes.

Evaluating on the full datasets ...

Training accuracy: 97.04%
Validation accuracy: 95.30%
Test accuracy: 95.07%
```





<br>

---

<br>

 
### 8) 逻辑回归基线

将 scikit-learn [逻辑回归](https://sebastianraschka.com/blog/2022/losses-learned-part1.html) 分类器作为基线：


```bash
python train_sklearn_logreg.py
```

```
Dummy classifier:
Training Accuracy: 50.01%
Validation Accuracy: 50.14%
Test Accuracy: 49.91%


Logistic regression classifier:
Training Accuracy: 99.80%
Validation Accuracy: 88.62%
Test Accuracy: 88.85%
```