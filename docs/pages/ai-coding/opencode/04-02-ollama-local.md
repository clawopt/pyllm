# 4.2 Ollama 本地模型

> **你的代码不想发给任何云服务？Ollama 让 AI 完全运行在你的本机——零成本、零泄露、零依赖。**

---

## 这一节在讲什么？

对于很多开发者来说，把代码发送到云端 LLM 是一个隐私顾虑——特别是处理公司内部项目、安全敏感代码、个人项目时。Ollama 解决了这个问题：它让你在本地运行 LLM，代码完全不出本机。这一节我们详细讲解 Ollama 的安装、模型选择、OpenCode 集成配置，以及本地模型的局限性和适用场景。

---

## 为什么用本地模型

在深入配置之前，先理解本地模型的三个核心价值：

**隐私保护**：代码不离开你的电脑。对于公司内部项目、安全敏感代码、个人隐私数据，这是硬性需求。即使 LLM 提供商承诺不使用你的数据训练模型，"数据不出本机"仍然是最强的隐私保证。

**零成本**：本地模型不按 token 计费——你只需要一台有足够内存的电脑。对于高频使用的开发者，这能省下不少 API 费用。

**离线可用**：不需要网络连接。在飞机上、在客户现场、在网络不稳定的环境里，你仍然可以使用 AI 编程助手。

但本地模型也有明显的局限——我们会在后面详细讨论。

---

## 安装 Ollama

**macOS / Linux**：

```bash
curl -fsSL https://ollama.ai/install | bash
```

**macOS 也可以用 Homebrew**：

```bash
brew install ollama
```

**Linux 手动安装**：

```bash
# 下载并安装
curl -fsSL https://ollama.ai/install.sh | sh

# 启动 Ollama 服务
ollama serve
```

**Windows**：从 [ollama.ai](https://ollama.ai) 下载安装包。

安装完成后验证：

```bash
ollama --version
# 输出：ollama version is 0.x.x
```

---

## 下载模型

Ollama 的模型是本地存储的——你需要先下载模型，才能使用。不同大小的模型对硬件要求不同：

| 模型 | 大小 | 最低内存 | 推荐内存 | 代码能力 |
|------|------|---------|---------|---------|
| llama3.2:1b | 0.5GB | 4GB | 8GB | ⭐ |
| llama3.2:3b | 2GB | 8GB | 16GB | ⭐⭐ |
| deepseek-coder:6.7b | 4GB | 8GB | 16GB | ⭐⭐⭐ |
| codestral:7b | 4GB | 8GB | 16GB | ⭐⭐⭐ |
| llama3.1:8b | 5GB | 16GB | 32GB | ⭐⭐⭐ |
| deepseek-coder:33b | 19GB | 32GB | 64GB | ⭐⭐⭐⭐ |
| llama3.1:70b | 40GB | 64GB | 128GB | ⭐⭐⭐⭐ |

下载模型：

```bash
# 通用小模型（2GB，快速）
ollama pull llama3.2:3b

# 代码生成专用（4GB，推荐）
ollama pull deepseek-coder:6.7b

# 代码补全专用（4GB）
ollama pull codestral:7b

# 更强的通用模型（5GB）
ollama pull llama3.1:8b
```

下载速度取决于你的网络——Ollama 的模型托管在 HuggingFace 上，国内可能需要代理。

查看已下载的模型：

```bash
ollama list
# 输出：
# NAME                    ID              SIZE    MODIFIED
# deepseek-coder:6.7b     xxxxxxxxxxxx    3.8 GB  2 hours ago
# llama3.2:3b             xxxxxxxxxxxx    2.0 GB  1 day ago
```

---

## 配置 OpenCode 使用 Ollama

Ollama 默认运行在 `http://localhost:11434`，OpenCode 会自动检测。你只需要在配置文件中指定 Ollama 模型：

```json
{
  "providers": {
    "ollama": {}
  },
  "agents": {
    "coder": {
      "model": "ollama/deepseek-coder:6.7b",
      "maxTokens": 8192
    }
  }
}
```

如果你的 Ollama 运行在非默认端口，可以通过环境变量指定：

```bash
export LOCAL_ENDPOINT="http://localhost:11434"
```

启动 OpenCode 后，AI 会使用你配置的 Ollama 模型。所有请求都发送到本地，不经过任何云服务。

---

## 混合模型策略

本地模型和云端模型各有优劣，最好的策略是"混合使用"——日常简单任务用本地模型，复杂任务切云端模型：

```json
{
  "agents": {
    "coder": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "maxTokens": 16384
    }
  }
}
```

在 TUI 中，你可以随时用 `/models` 命令切换：

```
# 简单问答 → 切到 Ollama
/models → 选择 ollama/deepseek-coder:6.7b

# 复杂任务 → 切到 Claude
/models → 选择 anthropic/claude-sonnet-4-20250514
```

或者用非交互模式按需选择：

```bash
# 简单任务用本地模型
opencode run -m ollama/deepseek-coder:6.7b "Add comments to @src/utils.ts"

# 复杂任务用云端模型
opencode run -m anthropic/claude-sonnet-4-20250514 "Refactor the authentication module"
```

---

## 本地模型的局限

理解本地模型的局限，才能合理使用它：

**1. 推理能力不如云端大模型**

7B 参数的本地模型在代码生成质量上跟 Claude Sonnet 有明显差距。它更适合辅助任务——代码补全、简单修改、快速问答——而不是复杂任务——架构设计、大型重构、跨文件修改。

**2. 长上下文处理能力有限**

本地模型的上下文窗口通常只有 4K~8K token（云端模型可达 128K~1M）。这意味着 AI 能"看到"的代码量有限，对于大型文件或跨文件分析，本地模型会力不从心。

**3. 生成速度取决于硬件**

本地模型的生成速度完全取决于你的硬件——CPU 推理很慢（可能只有 5~10 token/s），GPU 推理快得多（可达 30~50 token/s）。如果你没有独立 GPU，本地模型的响应速度可能比云端 API 还慢。

**4. 模型需要手动更新**

云端模型的改进是自动的——OpenAI 和 Anthropic 会持续更新模型。但本地模型需要你手动 `ollama pull` 更新。如果你一直用旧版本，可能会错过重要的改进。

---

## 推荐的本地模型选择

根据你的硬件条件，推荐以下配置：

**8GB 内存（入门级）**：

```bash
ollama pull llama3.2:3b
# 配置：ollama/llama3.2:3b
# 适合：简单问答、代码解释
# 不适合：代码生成、复杂修改
```

**16GB 内存（主流级）**：

```bash
ollama pull deepseek-coder:6.7b
# 配置：ollama/deepseek-coder:6.7b
# 适合：代码生成、简单修改、代码解释
# 不适合：复杂架构、跨文件重构
```

**32GB+ 内存（专业级）**：

```bash
ollama pull deepseek-coder:33b
# 配置：ollama/deepseek-coder:33b
# 适合：代码生成、中等复杂度修改
# 接近云端模型的质量，但推理速度较慢
```

**有独立 GPU（NVIDIA 8GB+ VRAM）**：

```bash
# GPU 加速推理，速度大幅提升
ollama pull deepseek-coder:6.7b
# Ollama 会自动使用 GPU
```

---

## 常见误区

**误区一：本地模型能替代 Claude/GPT-4**

目前还不能。7B 参数的本地模型在代码生成质量上跟 Claude Sonnet 有明显差距。本地模型适合辅助任务，复杂任务还是需要云端大模型。如果你非常在意隐私，可以日常用本地模型 + 复杂任务切云端。

**误区二：模型越大越好**

不是。大模型需要更多内存和更长的推理时间。如果你的硬件只有 8GB 内存，强行运行 33B 模型会导致系统卡顿甚至崩溃。选择模型时，先看你的硬件条件，再看任务需求。

**误区三：Ollama 只能跑 Llama**

不是。Ollama 支持多种模型——Llama、DeepSeek Coder、Codestral、Mistral、Qwen 等。你可以根据任务类型选择最合适的模型。代码生成推荐 DeepSeek Coder，通用推荐 Llama。

**误区四：本地模型不需要网络**

下载模型时需要网络。模型文件通常有几 GB，首次下载需要稳定的网络连接。下载完成后，使用时确实不需要网络。所以建议提前下载好需要的模型，不要等到离线环境才想起来。

---

## 小结

这一节我们详细讲解了 Ollama 本地模型的安装、配置和使用。本地模型的三个核心价值是隐私保护、零成本、离线可用，但推理能力、上下文窗口和生成速度都不如云端大模型。最佳策略是"混合使用"——日常简单任务用本地模型，复杂任务切云端模型。下一节我们学习 Auto Compact 和上下文管理，这是长对话场景下的关键能力。
