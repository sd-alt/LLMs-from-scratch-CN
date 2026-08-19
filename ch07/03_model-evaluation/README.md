# 第 7 章：微调以遵循说明

该文件夹包含可用于模型评估的实用程序代码。



 
## 使用 OpenAI API 评估指令响应


- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb) 笔记本使用 OpenAI 的 GPT-4 来评估指令微调模型生成的响应。它适用于以下格式的 JSON 文件：

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- The target given in the test set
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- Response by an LLM
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- Response by a 2nd LLM
},
```

 
## 使用 Ollama 本地评估指令响应

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb) 笔记本提供了上述笔记本的替代方案，利用通过 Ollama 本地下载的 Llama 3 模型。