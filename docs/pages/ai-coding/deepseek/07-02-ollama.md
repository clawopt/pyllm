# 7.2 Ollama 本地部署

> **一条命令下载，一条命令运行——Ollama 让本地部署 DeepSeek 变得跟安装 App 一样简单。**

---

## 这一节在讲什么？

上一节我们了解了蒸馏模型的选择策略，这一节我们动手——用 Ollama 在本地部署 DeepSeek 蒸馏模型。Ollama 是最简单的本地 LLM 部署工具——一条命令下载模型，一条命令运行，自动使用 GPU 加速，提供兼容 OpenAI 格式的 REST API。

---

## 安装 Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install | bash

# macOS Homebrew
brew install ollama

# 验证安装
ollama --version
```

---

## 下载和运行模型

```bash
# 下载推理模型 7B（推荐）
ollama pull deepseek-r1:7b

# 下载代码模型 6.7B
ollama pull deepseek-coder:6.7b

# 运行模型
ollama run deepseek-r1:7b

# 运行后直接在终端对话
>>> 证明根号2是无理数
>>> 用 Python 写一个快速排序
```

---

## API 服务

Ollama 启动后自动提供 REST API，兼容 OpenAI 格式：

```bash
# 启动 API 服务（默认端口 11434）
ollama serve
```

```python
# 用 OpenAI SDK 调用本地 Ollama
from openai import OpenAI

client = OpenAI(
    api_key="ollama",  # Ollama 不需要真实 Key
    base_url="http://localhost:11434/v1"
)

response = client.chat.completions.create(
    model="deepseek-r1:7b",
    messages=[{"role": "user", "content": "写一个快速排序"}]
)

print(response.choices[0].message.content)
```

---

## 硬件要求

| 模型 | 最低 VRAM | 推荐 VRAM | 推荐显卡 | CPU 可跑？ |
|------|----------|----------|---------|-----------|
| deepseek-r1:1.5b | 2GB | 4GB | 集成显卡 | ✅ 可以 |
| deepseek-r1:7b | 6GB | 8GB | RTX 4060 | ⚠️ 很慢 |
| deepseek-coder:6.7b | 6GB | 8GB | RTX 4060 | ⚠️ 很慢 |
| deepseek-r1:14b | 10GB | 12GB | RTX 4070 | ❌ |
| deepseek-r1:32b | 20GB | 24GB | RTX 4090 | ❌ |
| deepseek-r1:70b | 40GB+ | 80GB+ | 2×RTX 4090 | ❌ |

---

## 常见误区

**误区一：本地模型能替代云端 API**

7B 蒸馏版在简单任务上表现不错，但复杂任务仍有差距。最佳策略是"混合使用"——日常简单任务用本地模型，复杂任务切云端 API。

**误区二：Ollama 只能跑 DeepSeek**

Ollama 支持多种模型——Llama、Qwen、Mistral、Gemma 等。你可以根据任务类型选择最合适的模型。

**误区三：CPU 推理跟 GPU 差不多**

差很多。CPU 推理速度通常只有 3-10 token/s，GPU 推理可达 30-50 token/s。如果你有独立 GPU，确保 Ollama 使用了 GPU：

```bash
# 检查 Ollama 是否使用了 GPU
ollama run deepseek-r1:7b "Hello"
# 如果输出中显示 "using GPU"，说明 GPU 加速已启用
```

**误区四：下载模型需要翻墙**

Ollama 的模型托管在 HuggingFace 上，国内可能需要代理。你也可以从 ModelScope（国内镜像）下载模型。

---

## 小结

这一节我们学习了用 Ollama 部署 DeepSeek 蒸馏模型：`ollama pull` 下载模型，`ollama run` 运行对话，`ollama serve` 启动 API 服务。Ollama 自动使用 GPU 加速，提供兼容 OpenAI 格式的 API。7B 模型需要 8GB VRAM，RTX 4060 即可运行。下一节我们学习 vLLM 生产级部署。
