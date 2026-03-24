# LangChain 教程

LangChain 是构建大语言模型应用的强大框架。

## 安装

```bash
pip install langchain langchain-openai
```

## 快速开始

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
response = llm.invoke("Hello!")
print(response.content)
```

## 下一步

继续学习：
- [PG Vector教程](/pages/database/pgvector/)
