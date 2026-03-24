# Ollama 教程

Ollama 是一个在本地运行大语言模型的工具。

## 安装

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 快速开始

```bash
ollama serve
ollama run llama2
```

## Python API

```python
import ollama

response = ollama.chat(model='llama2', messages=[
    {'role': 'user', 'content': 'Hello!'}
])
print(response['message']['content'])
```

## 下一步

继续学习：
- [vLLM教程](/pages/llm/vllm/)
- [LangChain教程](/pages/llm/langchain/)
