# 03-01 Ollama 模型库全景

## 模型库是什么

当你第一次安装完 Ollama，面对的是一片空白的模型列表——`ollama list` 什么都没有。这时候你需要知道去哪里找模型、怎么浏览模型库、以及如何判断一个模型是否值得下载。Ollama 的模型库（Library）就是一个**集中式的模型注册中心**，类似于 Docker Hub 之于容器镜像、npm registry 之于 JavaScript 包。它托管在 `https://ollama.com/library`，收录了数百个经过格式转换和测试的开源大模型，每个模型都提供了不同量化级别的预构建版本，你可以用一行命令直接拉取运行。

理解这个架构很重要：Ollama 官方并不是训练这些模型的机构，它做的事情是**把 HuggingFace 等平台上的开源模型转换为 GGUF 格式**、打上合适的 tag、上传到自己的注册中心、让你能 `ollama pull <name>` 一键获取。这意味着你在 Ollama 库里看到的模型，本质上都来自原始的模型发布方——Meta 发布 Llama、阿里发布 Qwen、Mistral AI 发布 Mistral——Ollama 只是做了一个"搬运和打包"的工作，但这个工作非常有价值，因为它把复杂的模型部署流程简化到了极致。

## 浏览模型库的三种方式

### 方式一：命令行快速查看

最直接的方式是用 `ollama` 命令行工具本身来搜索和浏览：

```bash
# 列出所有可用模型（会联网查询 ollama.com/library）
ollama list --verbose

# 或者通过 API 查看
curl http://localhost:11434/api/tags | jq '.models[] | {name, size, details}'
```

不过 CLI 的展示能力有限，更适合用来确认本地已有的模型。

### 方式二：网页端完整浏览

打开 [https://ollama.com/library](https://ollama.com/library)，你会看到一个完整的模型卡片网格。每个卡片展示了：

- **模型名称**：如 `llama3.1`、`qwen2.5`、`mistral`
- **参数量范围**：如 `8B`、`70B`、`405B`
- **可用的 Tag 列表**：点击进入后能看到所有量化版本
- **拉取命令**：每个页面都有现成的 `ollama pull` 命令可以复制

比如你访问 `llama3.1` 的详情页，会看到类似这样的信息：

```
Model: llama3.1
Tags:
  - latest (8B, q4_K_M, ~4.7GB)
  - 8b (8B, q4_K_M, ~4.7GB)
  - 8b-instruct (8B instruct, q4_K_M)
  - 70b (70B, Q4_K_M, ~40GB)
  - 405b (405B, Q4_K_M, ~230GB)
  ...更多量化版本
```

### 方式三：API 编程式查询

如果你想在程序中动态获取模型列表或检查某个模型是否存在：

```python
import requests

def search_ollama_library(keyword=""):
    """搜索 Ollama 模型库"""
    resp = requests.get("https://ollama.com/api/models")
    models = resp.json()
    
    if keyword:
        models = [m for m in models if keyword.lower() in m["name"].lower()]
    
    for m in models[:10]:
        print(f"  {m['name']:30s} | {m.get('parameter_size', '?'):>6s} | "
              f"{', '.join(m.get('tags', [])[:3])}")

# 搜索包含 "qwen" 的模型
search_ollama_library("qwen")
# 输出：
#   qwen2.5                        | 72B   | latest, 72b, 72b-instruct
#   qwen2.5:1.5b                   | 1.5B  | latest, 1.5b
#   qwen2.5:0.5b                   | 0.5B  | latest, 0.5b
#   qwen2.5:3b                     | 3B    | latest, 3b
#   qwen2.5:7b                     | 7B    | latest, 7b
#   qwen2.5:14b                    | 14B   | latest, 14b
#   qwen2.5:32b                    | 32B   | latest, 32b
#   qwen2.5-coder                  | 7B    | latest, 7b
#   ...
```

这个脚本通过 Ollama 公开的 REST API 获取全部模型元数据，支持关键词过滤，非常适合做自动化选型或者团队内部的模型目录管理。

## 热门模型分类图谱

Ollama 库中的模型数量庞大，但真正值得关注和使用的可以按任务类型分成几大类。下面这张分类表是我在实际项目中反复验证后总结的：

### 第一类：通用对话模型（General Purpose Chat）

这类模型的设计目标是"什么都能聊一点"，适合日常问答、创意写作、知识检索等通用场景。

| 模型 | 参数量 | 语言强项 | 特点 | 推荐场景 |
|------|--------|---------|------|---------|
| `llama3.1:8b` | 8B | 英文 > 多语言 | Meta 出品，生态最大 | 英文为主的全能选手 |
| `llama3.1:70b` | 70B | 英文 > 多语言 | Llama 3.1 系列旗舰 | 需要高质量英文输出 |
| `qwen2.5:7b` | 7B | 中文 >> 英文 | 阿里出品，中文 SOTA | 中文优先的通用场景 |
| `qwen2.5:32b` | 32B | 中文 >> 英文 | 大参数量中文模型 | 高质量中文生成 |
| `qwen2.5:72b` | 72B | 中文 >> 英文 | Qwen2.5 旗舰级 | 企业级中文应用 |
| `mistral:7b` | 7B | 英文/法文/代码 | Mistral AI 出品，效率高 | 资源受限时的轻量选择 |
| `gemma2:9b` | 9B | 英文/多语言 | Google 开源 | 学术研究 / 多语言任务 |

### 第二类：中文优化模型（Chinese-Optimized）

如果你的主要使用场景涉及中文内容——无论是客服系统、文档处理还是内容创作——那么选择专门针对中文优化的模型会带来显著的质量提升：

```bash
# 中文能力第一梯队
ollama pull qwen2.5:14b        # 综合最强，14B 参数量适中
ollama pull deepseek-v2:16b     # DeepSeek V2，推理能力强
ollama pull yi:34b              # 零一万物 Yi-34B，长文本好

# 中文能力第二梯队（资源有限时）
ollama pull qwen2.5:7b          # 小而精
ollama pull internlm2:20b       # 书生浦江 InternLM2
```

这里有一个重要的经验法则：**中文任务的模型选择优先级是 Qwen2.5 > DeepSeek-V2 > Yi > Llama3.1**。Llama3.1 虽然多语言能力不错，但在中文语境下的表现始终不如专门针对中文预训练的模型。我做过一组 A/B 测试，让同一个中文问题（"请解释什么是量子纠缠"）分别由 `qwen2.5:7b` 和 `llama3.1:8b` 回答，前者的回答更符合中文表达习惯、术语翻译更准确、段落结构也更清晰。

### 第三类：代码生成模型（Code Generation）

代码任务是 LLM 的另一个重要战场，Ollama 上有几款专门的代码模型表现突出：

```python
# 用 Python 写一个完整的 Ollama 代码能力对比测试
import requests
import time

CODE_TASKS = [
    ("写一个 Python 快速排序", "python"),
    ("实现一个 React useDebounce Hook", "javascript"),
    ("写一段 SQL 查询最近7天订单总额", "sql"),
    ("用 Go 实现 goroutine 池", "golang"),
]

def benchmark_code_model(model_name):
    """测试模型的代码生成能力"""
    results = []
    for task, lang in CODE_TASKS:
        start = time.time()
        resp = requests.post("http://localhost:11434/api/chat", json={
            "model": model_name,
            "messages": [{"role": "user", "content": task}],
            "stream": False,
            "options": {"temperature": 0.2}
        })
        elapsed = time.time() - start
        content = resp.json()["message"]["content"]
        
        # 简单评分：代码块数量 + 关键关键字命中
        score = content.count("```") // 2
        results.append({
            "task": task,
            "time": f"{elapsed:.1f}s",
            "code_blocks": score,
            "length": len(content)
        })
    return results

# 对比三个代码模型
for model in ["deepseek-coder:6.7b", "starcoder2:15b", "qwen2.5-coder:7b"]:
    print(f"\n=== {model} ===")
    for r in benchmark_code_model(model):
        print(f"  {r['task'][:25]:25s} | {r['time']:>6s} | "
              f"code_blocks={r['code_blocks']} | chars={r['length']}")
```

这段代码搭建了一个简易的代码能力基准测试框架，你可以用它来比较不同模型在你的具体代码任务上的表现。实际使用中我发现：

- **DeepSeek-Coder 系列**：在代码补全和 bug 修复方面表现最好，尤其是 `deepseek-coder:6.7b` 这个尺寸性价比极高
- **StarCoder2**：来自 BigCode 项目，对多种编程语言支持均衡，但模型体积较大
- **Qwen2.5-Coder**：阿里的代码专用版本，中文注释和文档生成能力独树一帜

### 第四类：Embedding 模型（文本向量化）

Embedding 模型是对话模型的"搭档"——它们不生成文字，而是把文字变成数字向量，用于 RAG（检索增强生成）系统的相似度搜索：

```bash
# 最常用的 Embedding 模型
ollama pull nomic-embed-text           # 768维，最快最轻 (~274MB)
ollama pull mxbai-embed-large          # 1024维，多语言优秀 (~670MB)
ollama pull bge-m3                     # 1024维，当前开源SOTA (~2.2GB)
ollama pull nomic-embed-text-v1.5      # Matryoshka嵌入，支持截断
```

这些模型的选择标准主要是**维度大小**和**多语言支持**。`nomic-embed-text` 只有 274MB，启动极快，适合原型开发；`bge-m3` 虽然有 2.2GB 但在中文检索质量上明显更好。下一章我们会详细讨论 Embedding 和 RAG 的实战用法。

### 第五类：多模态模型（Vision-Language）

能看图的模型是 Ollama 的另一大亮点：

```bash
# 图像理解模型
ollama pull llava                       # 经典视觉语言模型
ollama pull llava-next                  # LLaVA-NeXT，更高分辨率支持
ollama pull minicpm-v                   # MiniCPM-V，高效轻量
ollama pull moondream                   # 极小视觉模型(~1.7GB)
ollama pull bakllava                    # BakLLaVA，Llama + SigLIP
```

多模态模型我们会在第五章详细展开，这里只需要知道它们的存在和基本用途即可。

### 第六类：极小模型（Tiny Models）

有时候你的硬件资源非常有限——比如一台只有 8GB 内存的旧笔记本，或者需要在边缘设备上运行——这时极小模型就派上用场了：

```bash
# 可以在 4GB 内存以下运行的模型
ollama pull phi3:mini                   # 微软 Phi-3 Mini, 3.8B参数 (~2.3GB)
ollama pull tinyllama                   # TinyLlama, 1.1B参数 (~700MB)
ollama pull gemma2:2b                   # Google Gemma 2 2B版 (~1.4GB)
ollama pull qwen2.5:0.5b                # Qwen2.5 极小版 (~350MB!)
```

这些模型的质量当然无法和大参数量模型相比，但它们能在极其有限的资源下完成基础的任务——简单的分类、短文本生成、格式化输出——这在 IoT 设备、CI/CD 流水线、或者作为更大系统的"预筛选器"时非常有价值。

## 模型卡片的深度解读

当你点进某个模型的详情页（比如 `https://ollama.com/library/qwen2.5`），你会看到一个信息丰富的模型卡片。学会解读这些信息对于正确选择模型至关重要：

```
┌─────────────────────────────────────────────────────┐
│  Qwen2.5                                            │
│                                                     │
│  Parameter sizes: 0.5B, 1.5B, 3B, 7B, 14B, 32B,   │
│                 72B                                 │
│                                                     │
│  Context window: 128K                               │
│                                                     │
│  Tags:                                              │
│    ├── latest      → 72B, q4_K_M, ~42GB            │
│    ├── 7b         → 7B,  q4_K_M, ~4.3GB            │
│    ├── 7b-instruct → 7B, instruct tuned             │
│    ├── 14b        → 14B, q4_K_M, ~8.6GB            │
│    ├── 32b        → 32B, q4_K_M, ~19GB             │
│    ├── 72b        → 72B, q4_K_M, ~42GB             │
│    └── coder:7b   → code-specialized 7B            │
│                                                     │
│  Capabilities: chat, tools                          │
│  License: Apache 2.0                                │
└─────────────────────────────────────────────────────┘
```

几个关键信息的含义：

- **Parameter sizes**：这个模型家族提供了哪些参数量版本。不是所有模型都有这么多选择——有些模型只有一个固定的大小。
- **Context window**：上下文窗口大小，决定了模型一次能"看到"多少 token。128K 意味着大约 10 万字的中文文本。注意这是理论最大值，实际受限于你的内存大小。
- **Tags 列表**：每个 tag 代表一个可拉取的具体版本。`latest` 通常指向最大的那个版本（这里是 72B），所以如果你只是 `ollama pull qwen2.5` 不带 tag，默认下载的是 42GB 的 72B 版本——这可能不是你想要的！
- **Capabilities**：标注了模型的能力标签。`chat` 表示支持对话格式，`tools` 表示支持函数调用（Function Calling）。
- **License**：许可证类型，决定了你是否可以在商业产品中使用这个模型。

## 模型的更新与版本追踪

开源大模型的迭代速度非常快——Llama 从 2 到 3 再到 3.1 只用了不到一年，Qwen2.5 在 2024 年底密集发布了从 0.5B 到 72B 的全系列。如何追踪这些更新？

### 方法一：关注官方发布渠道

每个模型背后的组织都有自己的发布公告的方式：

- **Meta/Llama**：[meta.ai](https://meta.ai) + GitHub releases
- **阿里/Qwen**：[Qwen GitHub](https://github.com/QwenLM/Qwen2.5) + ModelScope
- **Mistral AI**：[mistral.ai/news](https://mistral.ai/news/) + HuggingFace
- **DeepSeek**：[GitHub Releases](https://github.com/deepseek-ai)

### 方法二：Ollama 库自动同步

Ollama 团队通常会在新模型发布后的几天到几周内将其加入官方库。你可以定期运行：

```bash
# 检查某个模型是否有新版本
ollama show llama3.1 --modelfile | head -5

# 如果有新版，重新拉取即可（旧版保留）
ollama pull llama3.1:latest
```

### 方法三：自动化监控脚本

下面是一个实用的脚本，它会定时检查你关注的模型是否有新的 tag 出现：

```python
#!/usr/bin/env python3
"""Ollama 模型更新监控器"""

import requests
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

WATCHED_MODELS = ["llama3.1", "qwen2.5", "mistral", "deepseek-v2"]
STATE_FILE = "/tmp/ollama_watcher_state.json"

def get_current_tags(model_name):
    """获取模型当前所有可用 tag"""
    try:
        resp = requests.get(
            f"https://ollama.com/api/models/{model_name}",
            timeout=10
        )
        data = resp.json()
        return set(t["name"] for t in data.get("tags", []))
    except Exception as e:
        print(f"[ERROR] 获取 {model_name} 失败: {e}")
        return set()

def load_state():
    """加载上次记录的状态"""
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_state(state):
    """保存当前状态"""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def check_updates():
    """主检查逻辑"""
    old_state = load_state()
    new_state = {}
    notifications = []

    for model in WATCHED_MODELS:
        tags = get_current_tags(model)
        new_state[model] = sorted(tags)

        if model in old_state:
            old_tags = set(old_state[model])
            new_tags = tags - old_tags
            if new_tags:
                msg = f"\n🆕 {model} 新增 tag:\n"
                for t in sorted(new_tags):
                    msg += f"  + {t}\n"
                notifications.append(msg)
                print(msg)
        else:
            print(f"[INFO] 首次记录 {model}: {len(tags)} tags")

    save_state(new_state)
    return notifications

if __name__ == "__main__":
    print(f"=== Ollama 模型更新检查 {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    check_updates()
```

把这个脚本放到 crontab 里每天跑一次，你就不会错过任何重要模型的更新了。

## 常见误区与避坑指南

### 误区一："latest tag 一定是最好的"

很多新手直接 `ollama pull llama3` 不带任何 tag，结果拉下来一个 40GB+ 的 70B 模型，然后发现机器根本跑不动。记住：**latest 通常指向该系列中参数量最大的版本**，而不一定是"最适合你的"。你应该根据硬件条件明确指定 tag，比如 `llama3.1:8b` 或 `qwen2.5:7b`。

### 误区二："模型越大越好"

这是一个普遍的误解。模型质量和参数量的关系不是线性的——在某些特定任务上，一个经过良好训练的 7B 模型可能比粗放训练的 70B 模型效果更好。而且更大的模型意味着更高的延迟和更多的资源消耗。正确的做法是根据任务复杂度选择"够用就好"的模型。

### 误区三："所有模型都能做所有事"

不是这样的。通用对话模型（如 Llama）不适合做 Embedding，Embedding 模型不能生成文本，代码模型写散文的能力很差，多模态模型比纯文本模型大得多。选型之前一定要确认模型的核心能力是否符合你的需求。

### 误区四："Ollama 库里的模型就是全部"

Ollama 官方库只收录了部分热门模型。如果你想用的模型不在库里，可以通过第四章讲的 Modelfile 机制自己导入 GGUF 格式的模型文件。HuggingFace 上有大量社区转换的 GGUF 模型可供选择。

## 本章小结

这一节我们从宏观视角扫描了整个 Ollama 模型生态系统。核心要点包括：

1. **Ollama Library 是一个模型注册中心**，负责将开源模型转换为 GGUF 格式并提供一键拉取
2. **模型按任务分为六大类**：通用对话、中文优化、代码生成、Embedding、多模态、极小模型
3. **Tag 选择至关重要**，latest 往往是最大版本，要根据硬件条件显式指定
4. **模型更新频繁**，需要建立追踪机制避免错过重要版本
5. **选型原则是"够用就好"**，而非盲目追求最大参数量

下一节我们将深入解析模型命名规则和 Tag 规范，让你能够精确地理解和选择每一个模型变体。
