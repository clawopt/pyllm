# 01-03 第一个模型：运行与对话

## 从 `ollama run` 开始你的第一次本地 AI 对话

如果你已经完成了上一节的安装和环境检查，现在终端里输入下面这行命令：

```bash
ollama run qwen2.5:1.5b
```

然后你会看到类似这样的输出：

```
pulling manifest...
pulling 672d8b0a...
pulling vfdecode...
pulling spec...
verifying sha256: fxxxxx...
success
>>> 
```

这个 `>>>` 提示符就是 Ollama 的交互式聊天界面——和你在 ChatGPT 网页上看到的那种对话框一样，只不过现在是在你自己的终端里。在 `>>>` 后面输入任何文字，回车，模型就会回复。

```
>>> 你好
你好！我是通义千问大模型，很高兴为您服务。有什么我可以帮助您的吗？
```

恭喜，你已经成功在你的本地电脑上运行了一个拥有 15 亿参数的大语言模型。不管你是用 Mac、Linux 还是 Docker 完成的安装，这一刻的体验应该是相似的：**没有网络延迟，没有 API Key 配置，没有任何中间商——你和 AI 之间的连接是纯粹的点对点。**

## 交互式聊天的基本操作

进入 `>>>` 模式后，你可以像平常聊天一样持续对话。Ollama 会自动维护上下文——也就是说它能"记住"你们之前说了什么。

```
>>> 请用 Python 写一个快速排序
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x >= pivot]
    return quicksort(left) + [pivot] + quicksort(right)

>>> 这个函数的时间复杂度是多少？
快速排序的平均时间复杂度是 O(n log n)，最坏情况下（数组已排序或逆序）会退化到 O(n²)。不过在实际应用中，通过随机选择 pivot 或三数取中策略，可以有效地避免最坏情况。
```

注意到了吗？模型不仅正确地执行了你要求的任务（写代码），还能理解后续的问题并给出有深度的回答——这说明它确实在维护对话的上下文。

### 内置命令：`/help` 是你的好帮手

在 `>>>` 提示符下，除了正常输入文本外，你还可以使用一系列以 `/` 开头的内置命令。输入 `/help` 可以查看完整列表：

```
>>> /help

Available Commands:
  /set <parameter>     Set a parameter.
  /show-modelfile     Show Modelfile of this model.
  /save <name>        Save the current session to an alias.
  /load <name>        Load a previously saved session.
  /clear              Clear conversation history.
  /bye                Exit the session.
  /?                  Show help about using LLMs.
  /short             Switch to a shorter response mode.
  /verbose           Switch to a verbose response mode.
  /color <on|off>       Toggle color in response.
  /multiline          Toggle multiline input mode.
  /echo               Echo input back to terminal.
```

这些内置命令在日常使用中非常实用，我们逐个介绍最有用的几个。

### `/set`：实时调整模型行为

这是你最常会用到的命令。它允许你在不退出当前对话的情况下动态修改模型的推理参数：

```
>>> /set temperature 2
Temperature set to 2

>>> 写一首关于春天的诗
春风拂柳绿如烟，
桃花映水面红艳。
燕子归来寻旧巢，
满园春色醉心田。
```

对比一下把 temperature 设回 0.1 的效果：

```
>>> /set temperature 0.1
>>> 再写一首关于春天的诗
春暖花开燕归来,
绿草如茵映楼台。
细雨绵绵润万物,
生机盎然百花开。
```

可以明显感受到区别：temperature 高时（2.0），模型的输出更有创造性但也更不可控；temperature 低时（0.1），输出更保守、更确定但也更"无趣"。这个参数的具体含义和调优技巧我们会在第四章深入讨论，这里先记住一个经验值：**日常对话用 0.7 左右，创意写作用 1.0-1.5，需要精确事实时用 0.1-0.3**。

`/set` 支持的所有参数包括：

| 参数 | 默认值 | 作用 | 典型调整场景 |
|------|--------|------|-----------|
| `temperature` | 0.8 | 控制随机性 | 创意↑ / 精确↓ |
| `top_k` | 40 | 只从 top-k 个 token 中采样 | 减少幻觉 |
| `top_p` | 0.9 | 核采样累积概率阈值 | 平衡质量和多样性 |
| `num_ctx` | 4096 | 上下文窗口长度 | 长文档分析时加大 |
| `num_predict` | -1 (-1=不限制) | 最大生成长度 | 防止无限生成 |
| `repeat_penalty` | 1.1 | 重复惩罚系数 | 避免循环重复 |

### `/save` 和 `/load`：保存和恢复对话状态

这是一个经常被忽略但极其有用的功能组合。当你进行了一段很有价值的对话（比如让模型帮你 review 了一整份代码），你可以先把当前状态保存下来：

```
>>> /save code-review
Session 'code-review' saved
```

之后任何时候，即使重启了 Ollama 或切换了其他模型，你也可以恢复之前的对话：

```
>>> /load code-review
Session 'code-review' loaded
>>> 我们之前在讨论什么？
我们在审查一段 Python 快速排序的实现。
```

`/save` 保存的是当前的完整对话历史和系统提示词设置，`/load` 则完全恢复到保存时的状态。这在以下场景特别有用：
- **长时间运行的复杂任务**（中途保存进度）
- **对比不同参数的效果**（同一问题，不同 temperature 的回答差异）
- **分享给同事复现你的对话**

### `/clear` 和 `/bye`：结束与重置

```
>>> /clear
Conversation cleared.

>>> # 之前的对话上下文已清空，模型不再"记得"之前的内容
```

`/clear` 清空对话历史但不退出；`/bye` 直接退出交互模式回到 shell。

## 非交互式模式：单次问答

不是所有场景都需要交互式对话。很多时候你只是想问一个问题拿个答案就走——比如在脚本里调用、或者在管道中处理。

```bash
# 方式一：直接跟在命令后面
ollama run qwen2.5:1.5b "什么是递归？用简单的语言解释"
# 模型会直接输出回答然后退出，不会进入 >>> 模式

# 方式二：通过管道传入 prompt
echo "将以下 JSON 转为 YAML 格式:
{\"name\": \"张三\", \"age\": 30, \"skills\": [\"Python\", \"Go\"]}" | ollama run qwen2.5:1.5b

# 方式三：从文件读取 prompt
ollama run qwen2.5:1.5b "$(cat my_prompt.txt)"
```

非交互式模式的典型用途：
- **Shell 脚本中的一键调用**：`summarize=$(ollama run model "总结这段文字")`
- **Git Hook 集成**：commit message 自动生成
- **cron 定时任务**：每日报告生成
- **Makefile / CI Pipeline**：自动化测试或文档生成

## 模型响应的结构化信息

当你通过 API 使用 Ollama 时（下一章会详细讲），每次请求的响应中除了 `message.content`（模型的文字回答）外，还包含丰富的元数据。即使在终端里，你也可以通过一些方式观察到这些信息。

开启 verbose 模式可以看到更多细节：

```
>>> /verbose
Verbose mode activated.

>>> 什么是机器学习？
machine learning is a subset of artificial intelligence (AI)
that enables computers to learn from data without being explicitly
programmed. Instead of writing rules manually, you provide
the computer with examples (training data), and it discovers patterns
and makes predictions or decisions on its own.

total duration:       1.234s
prompt eval count:     12 tokens
prompt per second:     9.73
eval count:            45 tokens
eval per second:      36.46
```

这里有几个值得关注的指标：

- **total_duration**：总耗时。首次运行某个模型时会包含模型加载时间（可能要几秒到十几秒），后续请求会快很多。
- **prompt eval count**：输入的 token 数量。Ollama 内部会把你的输入 tokenize 后送入模型，这个数字告诉你实际消耗了多少输入 token。
- **eval count**：模型生成的 token 数量。
- **prompt/eval per second**：**这是衡量模型推理速度的核心指标**。对于 1.5B 这样的小模型在 CPU 上通常能达到 30-50 tokens/s（约等于人打字的速度），7B 模型则降到 10-20 tokens/s，如果用了 GPU 加速可以达到 50-150+ tokens/s。

这些指标在你做性能优化时会是重要的基准线。

## 多轮对话中的上下文管理

理解 Ollama 如何处理多轮对话，有助于你更好地控制它的行为和资源消耗。

```
┌─────────────────────────────────────────────────────────────┐
│                 Ollama 对话的内部结构                     │
│                                                             │
│  User: "你好"                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Prompt = [SYSTEM] + [User:你好]                    │   │
│  │         ↓                                          │   │
│  │ Model → "你好！我是..."                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                      ↑                                  │
│              上下文缓存 (KV Cache)                       │
│                                                             │
│  User: "你叫什么名字？"                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Prompt = [SYSTEM] + [User:你好] + [Asst:...]        │   │
│  │              + [User:你叫什么名字？]                   │   │
│  │         ↓                                          │   │
│  │ Model → "我是 Qwen，由阿里云开发的大语言模型..."     │   │
│  └─────────────────────────────────────────────────────┘   │
│                      ↑                                  │
│              上下文缓存变大...                              │
│                                                             │
│  ...继续对话...                                              │
│                      ↑                                  │
│              上下文缓存越来越大 → 内存占用增加 → 变慢       │
└─────────────────────────────────────────────────────────────┘
```

关键洞察：**每轮对话，Ollama 都要把完整的对话历史重新送给模型**（加上新的问题）。这意味着：

1. **对话越长，每轮的输入 token 数越多，推理越慢**
2. **上下文有上限**（默认 4096 tokens，可通过 `/set num_ctx` 调整）
3. **超过上限后，最早的内容会被"遗忘"**

这就是为什么长对话后模型可能会"忘记"最开始你说的话——不是因为模型不好使，而是因为那些内容已经被推出了上下文窗口。

### 一个实际的内存增长观察实验

```python
import subprocess
import time
import json


def measure_context_growth():
    """观察多轮对话中上下文的变化"""
    
    import urllib.request
    import urllib.error
    
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    
    messages = [{"role": "user", "content": "记住我的名字叫小明"}]
    
    print("=== 上下文长度随对话轮次变化 ===\n")
    print(f"{'轮次':<6} {'输入tokens':<12} {'输出tokens':<12} {'总token':<10}")
    print("-" * 50)
    
    follow_ups = [
        "我叫什么？",
        "我刚才让你记住的是什么？",
        "请用三种不同的方式介绍你自己",
        "把上面三个问题的答案整合成一段自我介绍",
        "在这段介绍中，你提到了哪些关键词？",
    ]
    
    for i, question in enumerate(follow_ups, 1):
        messages.append({"role": "user", "content": question})
        
        data = {
            "model": "qwen2.5:1.5b",
            "messages": messages,
            "stream": False,
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
        
        msg = result["message"]
        
        input_tokens = result.get("prompt_eval_count", 0)
        output_tokens = result.get("eval_count", 0)
        total = input_tokens + output_tokens
        
        print(f"{i:<6}{input_tokens:<12}{output_tokens:<12}{total:<10}")


# 运行前确保 Ollama 服务正在运行
measure_context_growth()
```

运行这个脚本你会清晰地看到：随着对话轮次增加，每轮的 token 数在稳步增长。当总 token 数接近模型的上下文窗口上限时，就需要考虑 `/clear` 重置或者开始新对话了。

## 常见对话问题与调试技巧

### 问题 1：模型回答太慢

**可能原因及排查顺序**：

```bash
# 1. 先看是不是首次加载该模型（首次会有几秒下载/加载时间）
# 观察是否显示 "pulling ..." 字样

# 2. 检查是不是 CPU 模式（无 GPU 加速）
# 启动日志中有 "using cpu" 还是 "using cuda/metal"
ollama run qwen2.5:7b "hi"
# 如果是 CPU 模式且模型较大，慢是正常的

# 3. 检查模型大小是否匹配硬件
# 7B 模型在 8GB 内存 Mac 上跑肯定会比 1.5B 慢很多
ollama list
# 查看 SIZE 列确认模型体积

# 4. 尝试减少上下文长度
/set num_ctx 2048  # 缩短上下文窗口能显著加速
```

### 问题 2：模型回答质量不理想

**这不是 Ollama 的问题，而是模型本身的能力边界。** 但有几个常见的误操作会让情况变得更糟：

```
❌ 错误做法：问题太模糊
>>> 帮我想个办法
# 模型只能瞎猜你的意图

✅ 正确做法：具体明确的问题
>>> 我有一个 Python 列表去重问题，列表里有字典和元组，用什么方法最高效？
# 模型能给出精确的代码方案
```

```
❌ 错误做法：一次塞太多要求
>>> 用中文写一篇 2000 字的论文，主题是量子计算，要包含引言/方法/实验/结论/参考文献，格式为 IEEE
# 小模型被要求完成超出其能力范围的任务

✅ 正确做法：分步引导
>>> 先帮我列一下量子计算的 5 个核心概念
>>> 好，基于这些概念，写一段 300 字的量子计算简介
>>> 最后，基于上面的内容，扩展为一篇完整的论文大纲
```

```
❌ 错误做法：temperature 设置不当
/set temperature 0.01  # 太低→模型几乎只选最高概率的词→输出呆板
/set temperature 3    # 太高→输出混乱不连贯

✅ 正确做法：根据任务类型选择
# 日常对话: 0.7
# 创意写作: 1.2
# 代码生成: 0.4（准确性优先)
# 翻译: 0.8
```

### 问题 3：模型"幻觉"（一本正经地说错话）

这是所有大模型（不管本地还是云端）的固有问题，但在小模型上会更明显。应对策略：

```python
# 在 SYSTEM 提示中加入约束（通过 Modelfile 或 /set system）
system_prompt = """
你是一个诚实的助手。如果不知道答案，请直接说"我不知道"，
不要编造信息。对于不确定的事实性信息，请标注"我不确定"。
"""

# 在提问时提供更多上下文（RAG 思路）
# 不要问: "这家公司是做什么的？"
# 要问: "根据以下公司描述[粘贴原文]，这家公司的主营业务是什么？"
```

### 问题 4：中文模型输出英文或乱码

```bash
# 确认你用的是中文优化的模型
ollama list
# 应该能看到类似 qwen2.5 / yi / deepseek-v2 / internlm 这些

# 如果模型名不带版本 tag，Ollama 会自动拉取 latest
# 而 latest 版本通常是英文权重微调的
# 解决: 明确指定中文友好的 tag
ollama run qwen2.5:7b "你好"  # ✅ 中文模型
ollama run llama3:latest "你好"  # ⚠️ 可能偏英文
```

---

到这里，你应该已经可以在终端里和本地大模型流畅地对话了。下一节我们将学习如何管理多个模型——拉取、删除、查看详情，以及理解 Ollama 存储模型文件的方式。
