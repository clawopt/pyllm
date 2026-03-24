# Hugging Face Transformers 教程

Transformers 是 Hugging Face 提供的预训练模型库。

## 安装

```bash
pip install transformers torch
```

## 快速开始

```python
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
result = classifier('I love this!')
print(result)
```

## 下一步

继续学习：
- [Ollama教程](/pages/llm/ollama/)
- [LangChain教程](/pages/llm/langchain/)
