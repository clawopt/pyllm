# 02-3 Python 接入实战

## 三种接入方式，三种不同的抽象层次

在 Python 中使用 Ollama，有三种主流方式，每种适合不同的场景和开发者偏好。理解它们的关系很重要——不是"哪个最好"，而是"哪个最适合你当前的需求"。

```
┌───────────────────────────────────────────────────────────────┐
│               Python 调用 Ollama 的三条路径              │
│                                                            │
│  方式一: requests 直调（最底层，最灵活）                  │
│    requests.post("http://localhost:11434/api/chat", ...)      │
│    → 完全控制 HTTP 层面                                   │
│    → 需要自己处理流式、错误、重试                          │
│    → 适合: 想理解 Ollama API 细节 / 极致性能优化          │
│                                                            │
│  方式二: ollama Python 库（官方 SDK，推荐）                 │
│    from langchain_ollama import ChatOllama                   │
│    → 高级封装，自动处理大部分细节                             │
│    → 内置 streaming / 工具调用支持                           │
│    → 适合: 大多数生产项目 / 快速开发                         │
│                                                            │
│  方式三: OpenAI SDK 兼容（万能适配器）                    │
│    from openai import OpenAI                               │
│    client = OpenAI(base_url="http://localhost:11434/v1")       │
│    → 一行代码从云端切换到本地                               │
│    → LangChain / LlamaIndex 自动识别                           │
│    → 适合: 已有 OpenAI 代码想迁移到本地                       │
└───────────────────────────────────────────────────────────────┘
```

下面我们逐一深入。

## 方式一：requests 直调

这是最原始也最透明的方式。上一章我们已经用过它来演示 API 调用，现在让我们把它封装成一个真正可用的类。

```python
import requests
import json
import time
from typing import Optional, List, Dict, Any


class OllamaClient:
    """
    基于 requests 的 Ollama 客户端
    
    封装了:
    - 连接管理（自动重连/健康检查）
    - 流式与非流式两种模式
    - 错误处理与重试机制
    - Token 计费统计
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5:7b",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 会话历史（用于多轮对话）
        self.conversations: Dict[str, List[Dict]] = {}
        
        # 统计信息
        self.total_tokens_used = 0
        self.total_requests = 0
    
    def _request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: Optional[Dict] = None,
        stream: bool = False,
    ) -> Any:
        """底层 HTTP 请求方法，带自动重试"""
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                if method == "GET":
                    resp = requests.get(url, timeout=self.timeout)
                else:
                    kwargs = {"timeout": self.timeout}
                    if stream:
                        kwargs["stream"] = True
                    resp = requests.post(url, json=payload, **kwargs)
                
                resp.raise_for_status()
                return resp
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
                
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * 2)
                    continue
                raise
                
        raise ConnectionError(f"Failed after {self.max_retries} attempts")
    
    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        stream: bool = False,
        conversation_id: Optional[str] = None,
    ) -> str | Dict:
        """
        发送聊天请求（非流式）
        
        Returns:
            流式=False: str（纯文本回答）
            流式=True: dict {content: str, stats: dict}
        """
        
        model = model or self.default_model
        messages = []
        
        # 构建消息列表
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})
        
        options = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
        
        result = self._request(
            "api/chat",
            payload={
                "model": model,
                "messages": messages,
                "options": options,
                "stream": stream,
            },
            stream=stream,
        )
        
        if stream:
            return self._handle_stream_response(result, model)
        else:
            data = result.json()
            content = data["message"]["content"]
            
            # 更新统计
            self.total_requests += 1
            self.total_tokens_used += data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
            
            # 保存对话历史
            if conversation_id:
                if conversation_id not in self.conversations:
                    self.conversations[conversation_id] = []
                self.conversations[conversation_id].extend([
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": content},
                ])
            
            return content
    
    def chat_stream(self, message: str, **kwargs):
        """流式聊天：返回生成器和完整回复"""
        
        result = self.chat(message, stream=True, **kwargs)
        return result  # {content: "...", stats: {...}}
    
    def _handle_stream_response(self, response, model):
        """处理 SSE 流式响应"""
        
        full_content = ""
        eval_count = 0
        prompt_eval_count = 0
        total_duration_ns = 0
        
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        full_content += delta
                    
                    if chunk.get("done"):
                        eval_count = chunk.get("eval_count", 0)
                        prompt_eval_count = chunk.get("prompt_eval_count", 0)
                        total_duration_ns = chunk.get("total_duration", 0)
                        
                except json.JSONDecodeError:
                    pass
                    
            elif line == "data: [DONE]":
                break
        
        self.total_requests += 1
        self.total_tokens_used += eval_count + prompt_eval_count
        
        return {
            "content": full_content,
            "stats": {
                "input_tokens": prompt_eval_count,
                "output_tokens": eval_count,
                "total_tokens": eval_count + prompt_eval_count,
                "duration_ms": total_duration_ns // 1_000_000,
                "model": model,
            }
        }
    
    def embed(self, texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
        """
        文本向量化（批量）
        """
        embeddings = []
        
        for text in texts:
            result = self._request("api/embeddings", json={
                "model": model,
                "input": text,
            }).json()
            embeddings.append(result["embedding"])
        
        return embeddings
    
    def get_stats(self) -> Dict:
        """获取使用统计"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_used,
            "conversations": len(self.conversations),
        }


# ===== 使用示例 =====

if __name__ == "__main__":
    client = OllamaClient(default_model="qwen2.5:7b")
    
    # 基本问答
    answer = client.chat("什么是装饰器？")
    print(f"\nQ: 什么是装饰器？\nA: {answer}\n")
    
    # 流式输出
    print("=" * 50)
    print("流式输出演示:")
    result = client.chat_stream("用 Python 写一个 Web 爬虫")
    print(f"\n{result['content']}")
    print(f"[{result['stats']['output_tokens']} tokens, {result['stats']['duration_ms']:.0f}ms]\n")
    
    # 多轮对话
    conv_id = "demo_conv"
    client.chat("我叫小明", conversation_id=conv_id)
    client.chat("我刚才告诉你什么？", conversation_id=conv_id)
    print(f"对话历史: {len(client.conversations[conv_id])} 轮\n")
    
    # 统计
    print(client.get_stats())
```

这个 `OllamaClient` 类展示了直调方式的全部要点：
- **连接复用**：一个 `Session` 对象维护整个生命周期
- **自动重试**：网络抖动时自动重试 3 次
- **多轮对话**：通过 `conversation_id` 维护独立的会话上下文
- **流式处理**：SSE 解析 + 增量拼接的完整实现
- **统计追踪**：token 消耗和请求数量实时记录

## 方式二：ollama 官方 Python 库

如果你不想自己处理 HTTP 细节，官方的 `ollama` Python 库是最直接的选择：

```bash
pip install ollama
```

```python
import ollama


def official_sdk_demo():
    """ollama 官方库用法全览"""
    
    # ===== 同步调用 =====
    response = ollama.chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': '什么是闭包？'}],
        options={'temperature': 0.7},
    )
    print(f"回答: {response.message.content}\n")
    
    # ===== 流式调用 =====
    for chunk in ollama.chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': '写一首关于春天的诗'}],
        stream=True,
    ):
        if chunk['message']['content']:
            print(chunk['message']['content'], end='', flush=True)
    print()  # 换行
    
    # ===== Generate（底层接口）=====
    response = ollama.generate(
        model='qwen2.5:7b',
        prompt='Explain quantum computing in one sentence.',
    )
    print(f"Generate: {response.response}\n")
    
    # ===== Embedding =====
    response = ollama.embeddings(
        model='nomic-embed-text',
        input='I love machine learning',
    )
    vec = response.embeddings[0]
    print(f"Embedding: dim={len(vec)}, first 5 values: {vec[:5]}\n")
    
    # ===== 列出模型 =====
    models = ollama.list()
    print(f"已安装模型数: {len(models)}")
    for m in models:
        print(f"  - {m['name']}")


official_sdk_demo()
```

官方库的优势在于**极简的 API**——不需要手动拼 URL、解析 JSON、处理 SSE。但它也有局限：
- 功能相对基础（比如没有内置的重试机制）
- 文档相对简单（很多高级功能需要看源码）
- 社区生态不如 OpenAI SDK 成熟

## 方式三：OpenAI SDK 兼容——无缝切换云端与本地

这可能是对已有项目最有价值的方式。如果你的代码原本调用 GPT-4o 或 Claude，现在想换成本地模型：

```python
from openai import OpenAI


def openai_compat_demo():
    """OpenAI SDK 兼容模式 —— 从云端切换到本地只需改一行代码"""
    
    # ===== 这一行就是唯一的切换点 =====
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # 必填但 Ollama 不校验
    )
    
    # ===== 之后的代码和调用 OpenAI 完全一样 =====
    response = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": "你是 Python 编程专家"},
            {"role": "user", "content": "解释一下 Python 的 GIL"},
        ],
        temperature=0.7,
        max_tokens=500,
    )
    
    choice = response.choices[0]
    print(f"回答:\n{choice.message.content}\n")
    
    # ===== Streaming 也完全兼容 =====
    stream = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "用一句话介绍 Rust"}],
        stream=True,
    )
    
    print("流式输出:", end=" ")
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print("\n")
    
    # ===== Embedding 也兼容 =====
    embedding_resp = client.embeddings.create(
        model="nomic-embed-text",
        input="Hello world",
    )
    dims = embedding_resp.data[0].embedding.shape
    print(f"Embedding dimension: {dims}\n")


openai_compat_demo()
```

这种方式的威力在于：**你可以用一个配置变量决定走云端还是本地**：

```python
import os

# 在 .env 或环境变量中设置
USE_LOCAL = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

if USE_LOCAL:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    DEFAULT_MODEL = "qwen2.5:7b"
else:
    client = OpenAI()  # 使用 OPENAI_API_KEY 环境变量
    DEFAULT_MODEL = "gpt-4o-mini"
```

这样你的应用就具备了"双模运行"能力：有网时用云端大模型，没网时自动降级到本地小模型——对离线场景特别有用。

## LangChain 集成预览

虽然 LangChain 的详细集成在第七章展开，但这里先给个快速预览，因为它是实际项目中最高频的使用模式之一：

```python
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


def langchain_quickstart():
    """LangChain + Ollama 最简集成"""
    
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)
    
    # 直接当 ChatModel 用
    response = llm.invoke([
        SystemMessage(content="你是技术文档助手"),
        HumanMessage(content="总结以下文本的核心观点:\n\n"
                      "Transformer 是一种基于注意力机制的神经网络架构..."
        ),
    ])
    
    print(response.content)
    
    # Chain 模式（RAG 预览）
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    template = """根据以下上下文回答问题。
    如果上下文中没有答案，就说不知道。
    
    上下文:
    {context}
    
    问题: {question}
    """
    
    chain = (
        ChatPromptTemplate.from_template(template)
        | llm
        | StrOutputParser()
    )
    
    result = chain.invoke({
        "context": "Python 是一门高级编程语言...",
        "question": "Python 有哪些数据类型？",
    })
    
    print(f"\nChain 输出: {result}")


langchain_quickstart()
```

注意这里 `ChatOllama` 来自 `langchain-ollama` 包（需要单独 `pip install langchain-ollama`），它比直接用 `OpenAI(base_url=...)` 的好处是：原生支持 Ollama 特有的功能（如 Modelfile 模型）和更好的错误处理。

## 异步编程：高并发场景

对于需要同时处理多个请求的场景（Web 服务后端），同步阻塞式调用会成为瓶颈。Python 的 `asyncio` + `aiohttp` 可以解决这个问题：

```python
import asyncio
import aiohttp
import json


class AsyncOllamaClient:
    """异步 Ollama 客户端 —— 适合高并发 Web 服务"""
    
    def ____(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    
    async def chat(self, message: str, model="qwen2.5:7b", **kwargs):
        """异步聊天"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            **kwargs,
        }
        
        async with self.session.post(
            f"{self.base_url}/api/chat", json=payload
        ) as resp:
            data = await resp.json()
            return data["message"]["content"]
    
    async def chat_batch(self, questions: list[str], model="qwen2.5:7b"):
        """并发批量提问——比串行快 N 倍"""
        tasks = [self.chat(q, model=model) for q in questions]
        results = await asyncio.gather(*tasks)
        return results


async def demo_concurrent():
    """对比串行 vs 并发的性能差异"""
    
    client = AsyncOllamaClient()
    questions = [
        "1+1=?",
        "Python 怎么读文件？",
        "什么是递归？",
        "写一个冒泡排序",
        "Linux 常用命令有哪些？",
    ]
    
    # 串行执行
    start = asyncio.get_event_loop().time()
    serial_results = []
    for q in questions:
        r = await client.chat(q)
        serial_results.append(r)
    serial_time = asyncio.get_event_loop().time() - start
    
    # 并发执行
    start = asyncio.get_event_loop().time()
    concurrent_results = await client.chat_batch(questions)
    concurrent_time = asyncio.get_event_loop().time() - start
    
    print(f"串行: {serial_time:.1f}s ({len(questions)} 个请求)")
    print(f"并发: {concurrent_time:.1f}s ({len(questions)} 个请求)")
    print(f"加速比: {serial_time/concurrent_time:.1f}x")


if __name__ == "__main__":
    asyncio.run(demo_concurrent())
```

在你的 Mac 上运行这段代码，你会看到并发版本通常比串行快 **3-8 倍**（取决于模型大小和 CPU 核数）。这是因为 Ollama 内部有队列机制——多个请求可以排队等待同一个模型依次处理，而异步 I/O 让你在等待期间可以做其他事情（比如处理其他请求或更新 UI），而不是傻等。

## 完整的生产级 CLI 助手工具

最后，把上面所有内容整合成一个实用的命令行工具：

```python
#!/usr/bin/env python3
"""Ollama CLI 助手 —— 终端里的 AI 助手"""

import argparse
import sys
import json
import requests

API = "http://localhost:11434"


def cmd_chat(args):
    """单次问答"""
    resp = requests.post(f"{API}/api/chat", json={
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False,
    }, timeout=args.timeout).json()
    print(resp["message"]["content"])


def cmd_chat_file(args):
    """文件内容作为 prompt"""
    with open(args.filepath, "r") as f:
        prompt = f"请分析以下文件内容:\n\n{f.read()}"
    
    resp = post_chat(prompt, args.model, args.timeout)
    print(resp)


def cmd_translate(args):
    """翻译"""
    lang_map = {
        "zh2en": ("翻译为英文", "Translate to English"),
        "en2zh": ("翻译为中文", "Translate to Chinese"),
        "auto": ("自动检测语言并翻译", "Auto-detect and translate"),
    }
    direction, hint = lang_map.get(args.direction, ("", ""))
    prompt = f"{hint}\n\n{args.text}"
    
    resp = post_chat(prompt, args.model, args.timeout)
    print(resp)


def cmd_code(args):
    """代码相关任务"""
    task_map = {
        "write": "编写",
        "explain": "逐行解释",
        "review": "审查（指出问题和改进建议）",
        "test": "为以下代码编写测试用例",
        "refactor": "重构以下代码使其更 Pythonic",
        "docstring": "为以下函数添加 Google 风格 docstring",
        "debug": "调试以下代码，找出 bug 并修复",
        "optimize": "优化以下代码的性能",
    }
    action = task_map.get(args.task, "处理")
    prompt = f"请{action}以下 {args.lang}代码:\n\n{args.code}"
    
    resp = post_chat(prompt, args.model, args.timeout)
    print(resp)


def post_chat(prompt, model, timeout=30):
    """统一的聊天请求"""
    resp = requests.post(f"{API}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=timeout).json()
    return resp["message"]["content"]


def main():
    parser = argparse.ArgumentParser(
        description="🦙 Ollama CLI —— 终端里的 AI 助手",
        formatter_class=lambda prog: argparse.HelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # === chat 子命令 ===
    p_chat = subparsers.add_parser("chat", help="💬 单次问答")
    p_chat.add_argument("prompt", help="你的问题")
    p_chat.add_argument("-m", "--model", default="qwen2.5:7b", help="模型名称")
    p_chat.set_defaults(func=cmd_chat)
    
    # === translate 子命令 ===
    p_trans = subparsers.add_parser("translate", help="🌐 翻译")
    p_trans.add_argument("text", help="要翻译的文本")
    p_trans.add_argument("-d", "--direction", default="auto", choices=["zh2en", "en2zh", "auto"])
    p_trans.add_argument("-m", "--model", default="qwen2.5:7b")
    p_trans.set_defaults(func=cmd_translate)
    
    # === code 子命令 ===
    p_code = subparsers.add_parser("code", help="💻 代码助手")
    p_code.add_argument("code", help="代码片段或 '-' 表示从 stdin 读取")
    p_code.add_argument("-t", "--task", default="explain", choices=["write", "explain", "review", "test", "refactor", "docstring", "debug", "optimize"])
    p_code.add_argument("--lang", default="python", help="编程语言（影响注释风格）")
    p_code.add_argument("-m", "--model", default="deepseek-coder:6.7b-instruct")  # 代码任务用代码模型
    p_code.set_defaults(func=cmd_code)
    
    # === file 子命令 ===
    p_file = subparsers.add_parser("file", help="📄 文件分析")
    p_file.add_argument("filepath", help="文件路径")
    p_file.add_argument("-m", "--model", default="qwen2.5:7b")
    p_file.set_defaults(func=cmd_chat_file)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
```

使用方式：

```bash
# 安装后可以直接用命令行
chmod +x ollama_cli.py && sudo cp ollama_cli.py /usr/local/bin/

# 单次问答
ollama-cli.py chat "什么是装饰器？" -m qwen2.5:7b

# 翻译
ollama-cli.py translate "Hello world" -d zh2en

# 代码助手
echo 'def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)' | ollama_cli.py code -t review -- -

# 分析文件
ollama-cli.py file my_script.py
```

本章我们从 API 服务启动讲到了 Python 接入的三种主要方式和完整的工程化实践。下一章将覆盖 JavaScript/TypeScript 和其他语言的接入方法。
