---
title: 异步编程：提升应用的并发能力
description: ainvoke/astream/abatch 异步方法、async/await 模式、FastAPI 集成、并发性能对比
---
# 异步编程：提升应用的并发能力

上一节我们学习了流式输出——它让单个请求的"感知延迟"大幅降低。但如果你需要**同时处理多个请求**呢？比如一个 Web 服务同时来了 10 个用户提问，如果用同步方式串行处理，第 10 个用户要等前 9 个全部完成才能开始，体验极差。

这一节我们将学习 **异步（Async）编程**——让应用能够同时处理多个请求，充分利用 I/O 等待时间。

## 为什么需要异步

LLM 调用是一个典型的 **I/O 密集型操作**：

```
同步调用的时间分解（以 GPT-4o-mini 为例）：
├─ 网络发送请求 ────────── ~50ms
├─ 服务端排队等待 ───────── ~200ms  
├─ 模型生成响应 ────────── ~2500ms
├─ 网络接收响应 ────────── ~50ms
└─ 总计 ─────────────────── ~2800ms

关键：在这 2800ms 中，你的程序在"干等"，CPU 完全空闲！
```

如果能利用这段等待时间同时处理其他请求，吞吐量就能提升数倍。

## LangChain 的异步 API

LangChain 为几乎所有组件提供了异步版本的方法。命名规则很简单：在同步方法名前加 `a` 前缀：

| 同步方法 | 异步方法 | 说明 |
|---------|---------|------|
| `invoke()` | `ainvoke()` | 单次调用 |
| `batch()` | `abatch()` | 批量调用 |
| `stream()` | `astream()` | 流式输出 |

### ainvoke()：单次异步调用

```python
import asyncio
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def main():
    # 发起请求后不阻塞，可以去做别的事
    task1 = chat.ainvoke("说一句话")
    task2 = chat.ainvoke("讲个笑话")
    
    # 同时等待两个结果
    result1, result2 = await asyncio.gather(task1, task2)
    
    print(f"回答1: {result1.content}")
    print(f"回答2: {result2.content}")

asyncio.run(main())
```

`ainvoke()` 返回的是一个协程对象（Coroutine），不会阻塞当前线程。只有当你 `await` 它时才会真正等待结果。

### abatch()：批量异步调用

```python
import asyncio
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

questions = ["你好", "什么是 RAG？", "写个排序函数"]

async def sequential():
    """串行处理 — 总耗时 = 所有请求之和"""
    results = []
    for q in questions:
        r = await chat.ainvoke(q)
        results.append(r.content)
    return results

async def concurrent():
    """并行处理 — 总耗时 ≈ 最慢的那个请求"""
    tasks = [chat.ainvoke(q) for q in questions]
    results = await asyncio.gather(*tasks)
    return [r.content for r in results]

# 对比两种模式
import time

start = time.time()
r1 = await sequential()
print(f"串行: {time.time()-start:.2f}s")

start = time.time()
r2 = await concurrent()
print(f"并行: {time.time()-start:.2f}s")
```

典型输出：

```
串行: 8.52s     # 3 个请求串行
并行: 3.21s     # 3 个请求同时进行，总时间 ≈ 最慢的 1 个
```

**加速比约 2.7 倍**——对于 3 个请求就有这么明显的提升，请求数越多效果越显著。

### astream()：异步流式输出

```python
import asyncio
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def stream_demo():
    async for chunk in chat.astream("用三句话介绍 Python"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

asyncio.run(stream_demo())
```

`astream()` 是一个**异步生成器（AsyncGenerator）**，可以用 `async for` 来逐 chunk 消费。

## Chain 中的异步使用

Chain 的管道操作符 `|` 在异步模式下同样适用：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("解释{topic}")
parser = StrOutputParser()

chain = prompt | chat | parser

# 异步版本 — 用法完全一致
result = await chain.ainvoke({"topic": "装饰器"})
print(result)

# 异步批量 + 流式
async for chunk in chain.astream({"topic": "RAG"}):
    print(chunk, end="", flush=True)
```

LCEL 的美妙之处就在这里——**不管底层是同步还是异步，你写的代码几乎一样**。

## FastAPI 集成实战

FastAPI 是 Python 最流行的异步 Web 框架，和 LangChain 的异步 API 配合得天衣无缝：

```python
"""
fastapi_chat.py — 异步流式聊天服务
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="流式聊天 API")
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("简洁回答: {question}")


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """非流式接口"""
    result = await chat.ainvoke(request.question)
    return {"reply": result.content}


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """SSE 流式接口"""

    async def generate():
        async for chunk in chat.astream(request.question):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/chat/batch")
async def chat_batch_endpoint(requests: list[ChatRequest]):
    """批量问答接口"""
    tasks = [chat.ainvoke(r.question) for r in requests]
    results = await asyncio.gather(*tasks)
    return [
        {"question": req.question, "reply": r.content}
        for req, r in zip(requests, results)
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

启动服务：

```bash
pip install fastapi uvicorn python-multipart
uvicorn fastapi_chat:app --reload
```

测试三个端点：

```bash
# 非流式
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 RAG？"}'

# SSE 流式
curl -N http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "讲个笑话"}'

# 批量
curl -X POST http://localhost:8000/chat/batch \
  -H "Content-Type: application/json" \
  -d '[{"question": "你好"}, {"question": "Python"}, {"question": "再见"}]'
```

三个端点分别对应三种不同的使用场景：
- `/chat`：简单集成，不需要实时反馈
- `/chat/stream`：前端需要逐字展示的场景
- `/chat/batch`：后台批处理的场景（如离线分析任务）

## 并发控制：限制最大并发数

无限制的并发可能导致 API 速率超限或服务器过载。LangChain 提供了内置的并发控制机制：

```python
import asyncio
from langchain_community.chat_models import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter

chat = InMemoryRateLimiter(
    ChatOpenAI(model="gpt-4o-mini"),
    max_requests_per_minute=20   # 每分钟最多 20 次请求
)

# 超过限制时会自动等待直到配额释放
results = await abatch([chat.ainvoke(f"问题{i}") for i in range(25)])
```

当第 21-25 个请求到达时，它们会自动排队等待，而不是直接触发 API 报错。这对于生产环境中的速率保护非常重要。

## 同步 vs 异步的选择指南

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| CLI 脚本 / 一次性工具 | 同步 (`invoke`) | 简单直接，不需要并发 |
| Web 服务 (FastAPI) | 异步 (`ainvoke`) | 不阻塞事件循环，支持多用户 |
| 批量数据处理 | 异步 (`abatch`) | 并发执行，速度快 |
| 流式输出 (Web) | 异步 (`astream`) | FastAPI 原生支持 SSE |
| Jupyter Notebook | 同步即可 | 单线程环境，异步无优势 |

下一节我们将学习 **中间件（Middleware）**——一种能在 Chain 的输入和输出之间插入横切逻辑（如日志、限流、审核）的强大模式。
