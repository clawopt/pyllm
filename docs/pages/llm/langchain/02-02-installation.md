---
title: 安装 LangChain 核心库与生态工具
description: pip 安装 LangChain、OpenAI SDK、环境变量配置、依赖版本验证
---
# 安装 LangChain

环境就绪之后，这一节我们正式安装 LangChain 以及它运行所必需的依赖。整个过程只需要几条命令。

## 核心安装：一条命令搞定

LangChain 的核心包和最常用的 OpenAI 集成包可以一次性安装：

```bash
pip install "langchain[all]" langchain-openai langchain-community
```

这条命令做了什么？

- **`langchain[all]`**：安装 LangChain 核心本身，`[all]` 表示同时安装所有可选依赖（比如文档加载器、向量存储集成、各种工具封装）
- **`langchain-openai`**：LangChain 官方提供的 OpenAI 模型接入层——这是目前最常用的 LLM 后端
- **`langchain-community`**：社区维护的第三方集成（搜索工具、数据库连接器等）

安装过程通常需要 1-3 分钟，取决于你的网络速度。如果遇到下载慢的情况，可以换用国内镜像源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "langchain[all]" langchain-openai langchain-community
```

## 安装完成后的验证

安装完毕后，用以下命令确认一切正常：

```python
import langchain
import langchain_openai

print(f"LangChain 版本: {langchain.__version__}")
print("✅ LangChain 及其依赖安装成功！")
```

运行后你应该看到类似 `LangChain 版本: 0.3.x / 0.4.x / 1.x`（具体版本取决于你安装时的最新版）的输出。

## 你实际安装了什么

上面的 `pip install` 看起来只是一条命令，但它背后会安装一大堆依赖包。了解这些依赖分别负责什么，对后续调试问题很有帮助：

| 包名 | 作用 | 你什么时候会用到 |
|------|------|----------------|
| **langchain-core** | LangChain 的核心抽象（Runnable、管道符、序列化等） | **每一章都会用到**——这是框架的骨架 |
| **langchain-openai** | OpenAI API 的封装（ChatOpenAI、OpenAI Embeddings 等） | 第 2-8 章（几乎所有 LLM 调用都通过它） |
| **langchain-community** | 社区贡献的工具集成 | 第 7 章（Agent 工具）、第 4 章（RAG 检索） |
| **pydantic** | 数据验证和解析（OutputParser 底层依赖） | 第 3 章（结构化输出解析） |
| **tenacity** | 异步重试（中间件和网络请求的容错） | 生产部署时用到 |
| **chromadb** | 轻量级向量数据库（RAG 开发时常用） | 第 4 章（本地 RAG 原型） |

你不需要现在就记住每一个包的作用——在后续章节中用到的时候我会逐一解释。

## 关于 API Key 的说明

LangChain 本身是免费开源的，但你要调用的大模型（如 GPT-4o）不是免费的——你需要一个 **API Key** 来认证身份和计费。

```python
# 这是下一节要用的东西，先在这里提一下
import os

# 方式一：直接硬编码（仅适合快速测试，不要用于生产代码）
# os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# 方式二（推荐）：从环境变量读取（更安全）
# api_key = os.getenv("OPENAI_API_KEY")
```

关于 API Key 有几个重要的安全原则：

1. **永远不要把 API Key 写进代码里提交到 Git** —— 这是最常见的安全事故之一
2. 使用 `.env` 文件管理 Key，并确保 `.env` 已加入 `.gitignore`
3. 如果你不小心泄露了 Key，立即去对应平台的后台撤销重新生成

我们会在下一节详细演示如何正确地配置和使用 API Key。
