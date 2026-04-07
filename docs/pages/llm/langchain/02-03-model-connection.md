---
title: 模型接入实战
description: 接入 OpenAI GPT 系列模型、环境变量配置、API Key 安全管理、本地模型（Ollama）的接入方式
---
# 模型接入：让 LangChain 对话大模型

LangChain 本身不包含任何大模型——它只是一个"接线员"，负责把你的请求转发给真正的 LLM。所以在开始写应用之前，你需要先决定：**我要用哪个模型？** 以及 **怎么让 LangChain 连上它？**

这一节我们覆盖三种最常见的接入方式：OpenAI 云端 API、阿里云百炼（国内常用）、以及本地运行的 Ollama。

## 方式一：OpenAI —— 最主流的选择

OpenAI 的 GPT-4o / GPT-4o-mini 是目前生态最成熟、文档最全的 LLM 后端。接入它需要两步：**获取 API Key** 和 **在代码中配置**。

### 第一步：获取 API Key

1. 访问 [platform.openai.com/api-keys](https://platform.openai.com/api-keys) 并登录你的 OpenAI 账号
2. 点击 "Create new secret key"
3. 复制生成的 `sk-...` 开头的字符串

### 第二步：配置到项目中

创建一个 `.env` 文件（注意这个文件应该被加入 `.gitignore`）：

```bash
# 在项目根目录下执行
touch .env
```

写入以下内容：

```properties
# .env 文件 —— 存放敏感信息，不要提交到 Git
OPENAI_API_KEY=sk-your-key-here
```

然后在代码中读取：

```python
import os
from dotenv import load_dotenv

# 加载 .env 文件中的变量到 os.environ
load_dotenv()

# 从环境变量读取 API Key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

print(f"API Key 已加载: {api_key[:8]}...")
```

运行后你应该看到 `API Key 已加载: sk-xxxxxxx...`——说明 Key 已成功从 `.env` 文件读入。

### 第三步：创建 ChatOpenAI 实例并测试

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",      # 用 mini 版本就够了，便宜且快
    temperature=0,              # 温度设为 0 让输出更确定
)

response = llm.invoke("用一句话介绍你自己")
print(response.content)
```

如果一切正常，你会看到类似这样的输出：

```
我是 Claude，由 Anthropic 开发的人工智能助手。
```

> **为什么用 gpt-4o-mini 而不是 gpt-4o？** 因为对于学习和开发来说，mini 版本已经完全够用了，而且它的价格只有完整版的 1/10，速度还更快。等你要做生产部署时再切换到 gpt-4o 不迟。另外，`temperature=0` 是为了让模型的输出更确定、更可复现——调试时你希望同样的输入得到同样的输出，而不是每次都有随机变化。

## 方式二：阿里云百炼 —— 国内替代方案

如果你在国内，访问 OpenAI API 可能会遇到网络不稳定或无法访问的问题。阿里云百炼（DashScope）提供了一个兼容 OpenAI SDK 的国产替代方案：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 百炼的 API Key 格式和 OpenAI 相同
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
# 或者直接设置 base_url 指向百炼的网关
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

llm = ChatOpenAI(
    model="qwen-plus",          # 通义千问系列
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL"),
)

response = llm.invoke("你好，你是谁？")
print(response.content)
```

关键区别只有两处：
- `model` 改为百炼支持的模型名（如 `qwen-plus`、`qwen-max` 等）
- `base_url` 指向百炼的兼容接口

其他代码（包括 LangChain 的 Chain/Agent 语法）**完全不需要改动**——这就是使用标准化接口的好处。

## 方式三：Ollama —— 完全本地运行

如果你不想依赖任何云端 API（出于隐私、成本或网络原因），可以用 [Ollama](https://ollama.com/) 在本地运行开源模型：

```bash
# 1. 安装 Ollama（macOS 可用 Homebrew）
brew install ollama

# 2. 拉取一个模型（以 Qwen2.5-7B-Instruct 为例，约 4.7GB）
ollama pull qwen2.5:7b-instruct

# 3. 启动 Ollama 服务（默认监听 localhost:11434）
ollama serve
```

Ollama 启动后，它在本地暴露了一个兼容 OpenAI API 的接口——这意味着你可以用几乎相同的代码来调用它：

```python
from langchain_openai import ChatOpenAI

# Ollama 的本地地址和端口
local_llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    model="qwen2.5:7b-instruct",
    temperature=0,
)

response = local_llm.invoke("用一句话解释什么是 RAG")
print(response.content)
```

输出可能类似：

```
RAG（Retrieval-Augmented Generation，检索增强生成）是一种 AI 架构模式，
通过先从外部知识库检索相关信息，再将检索结果作为上下文提供给大语言模型...
```

### Ollama 的适用场景与局限

| 维度 | 云端 API | Ollama 本地 |
|------|---------|-----------|
| 成本 | 按 token 计费 | 免费（但需要硬件） |
| 延迟 | 取决于网络 | 几乎为零 |
| 隐私性 | 数据经过第三方 | 数据完全在本地 |
| 模型能力 | 最强（GPT-4o 级别） | 取决于本地硬件 |
| 适用场景 | 生产环境、快速原型 | 隐私要求高、离线开发 |

对于学习阶段来说，三种方式任选其一即可。本教程后续章节主要以 **OpenAI + gpt-4o-mini** 为例进行演示——因为它最稳定、文档最多、遇到问题时最容易找到解决方案。当你理解了核心概念之后，切换到百炼或 Ollama 只需要改几行配置。
