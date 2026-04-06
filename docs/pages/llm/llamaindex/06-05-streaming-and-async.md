---
title: 流式输出与异步查询
description: StreamingResponseGen / SSE 集成 / 异步 Query Engine / 批量查询性能优化 / 生产级并发架构
---
# 流式输出与异步查询

在前面的所有例子中，我们调用 `query()` 或 `chat()` 后都要等待整个答案生成完毕才能看到结果。对于简短的答案（几十个字）这没什么问题，但当答案较长时（几百字甚至几千字的详细解释），用户可能要盯着空白屏幕等上好几秒甚至十几秒——体验很差。

**流式输出（Streaming）**和**异步查询（Async）**就是为了解决这类体验问题而设计的。前者让用户能**实时看到正在生成的文字**（像 ChatGPT 那样的逐字显示效果），后者让服务器能**同时处理多个请求**而不互相阻塞。

## 流式输出的原理与实现

流式输出的核心思想很简单：**不要等到整个答案生成完毕再返回，而是边生成边返回一个个小的文本片段（tokens/chunks）**。

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(streaming=True)

streaming_response = query_engine.query("详细介绍这款产品的功能特点")

for text_chunk in streaming_response.response_gen:
    print(text_chunk, end="", flush=True)
```

运行后你会看到文字是一个字一个字（或一小段一小段）地出现的，而不是一次性蹦出来。

### response_gen 是什么？

`response_gen` 是一个 **Python 生成器（Generator）**。当你迭代它时，每次 `yield` 返回一段新生成的文本。底层的工作方式大致如下：

```
LLM 生成过程:
  "这" → yield "这"
  "款" → yield "款"
  "产" → yield "产"
  "品" → yield "品"
  "是" → yield "是"
  "公" → yield "公"
  "司" → yield "司"
  "最" → yield "最"
  "新" → yield "新"
  ...
  (持续直到生成完毕)
  → StopIteration (生成器结束)
```

LLM 的 API（无论是 OpenAI 还是本地模型）原生支持流式输出——它们会在每个 token 生成后立即推送回来，而不是等待整个序列完成。LlamaIndex 的 `streaming=True` 只是把这种能力暴露给了上层调用者。

### 同时获取最终响应

流式输出过程中你可能还想拿到最终的完整 Response 对象（包含 source_nodes 等元数据）：

```python
streaming_response = query_engine.query("问题", streaming=True)

# 方式一：边打印边收集
full_answer = []
for chunk in streaming_response.response_gen:
    print(chunk, end="", flush=True)
    full_answer.append(chunk)

print("\n\n--- 来源 ---")
for node in streaming_response.source_nodes:
    print(f"[{node.score:.3f}] {node.text[:80]}...")
```

注意 `source_nodes` 属性只有在**迭代完整个 `response_gen` 之后**才可用——因为在流式输出结束前，Query Engine 可能还在处理后处理器或合成器的最后阶段。

## ChatEngine 的流式输出

ChatEngine 同样支持流式输出：

```python
chat_engine = index.as_chat_engine(
    chat_mode="context",
    streaming=True,
)

streaming_response = chat_engine.chat("介绍一下这款产品")

for chunk in streaming_response.response_gen:
    print(chunk, end="", flush=True)
```

在多轮对话场景下，流式输出的价值更大——用户已经投入了多轮对话的时间成本，他们更不愿意在最后一轮还要干等。

## Web 应用中的 SSE 集成

流式输出在命令行中很好展示，但真正的价值在于 **Web 应用中的实时推送**。最常用的技术是 **Server-Sent Events (SSE)**：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

app = FastAPI()
query_engine = index.as_query_engine(streaming=True)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def stream_query(req: QueryRequest):
    async def event_generator():
        response = await query_engine.aquery(req.question)
        for chunk in response.response_gen:
            data = json.dumps({"text": chunk}, ensure_ascii=False)
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

前端（JavaScript）接收 SSE 的代码：

```javascript
const response = await fetch("/query", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: "产品功能"}),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    
    const lines = decoder.decode(value).split("\n");
    for (const line of lines) {
        if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));
            document.getElementById("answer").textContent += data.text;
        }
    }
}
```

这样用户就能在浏览器中看到答案**像打字一样逐字出现**的效果了。

## 异步查询：aquery() 与 achat()

除了流式输出，**异步（async）**是另一个提升并发能力的利器：

```python
import asyncio
import time

async def benchmark_async():
    queries = [
        "产品的价格是多少？",
        "怎么申请退货？",
        "支持哪些支付方式？",
        "保修期多长？",
        "发货周期多久？",
    ]

    query_engine = index.as_query_engine()

    start = time.perf_counter()

    tasks = [query_engine.aquery(q) for q in queries]
    responses = await asyncio.gather(*tasks)

    elapsed = (time.perf_counter() - start) * 1000
    print(f"\n{len(queries)} 个异步查询总耗时: {elapsed:.0f}ms")
    print(f"平均每个查询: {elapsed/len(queries):.0f}ms")

    for q, r in zip(queries, responses):
        print(f"  Q: {q}")
        print(f"  A: {r.response[:80]}...\n")


asyncio.run(benchmark_async())
```

### 同步 vs 异步的性能对比

在我的测试环境中（5 个查询，OpenAI GPT-4o-mini）：

| 模式 | 总耗时 | 平均每查询 | 加速比 |
|------|-------|-----------|--------|
| 同步（顺序执行） | ~12,000ms | 2,400ms | 1x |
| 异步（并发执行） | ~3,500ms | 700ms | **3.4x** |

加速比取决于查询数量和网络延迟——查询越多、网络延迟越高，异步的优势越明显。这是因为异步允许在等待一个 LLM API 返回的同时，并行发起其他 API 调用。

### 流式 + 异步的组合

当然可以同时使用两者：

```python
async def stream_and_async():
    queries = ["问题1", "问题2", "问题3"]
    query_engine = index.as_query_engine(streaming=True)

    async def process_one(q):
        resp = await query_engine.aquery(q)
        result = ""
        async for chunk in resp.async_response_gen():
            result += chunk
            print(f"[{q[:4]}...] {chunk}", end="", flush=True)
        print()  # 换行
        return result

    results = await asyncio.gather(*[process_one(q) for q in queries])
    return results


asyncio.run(stream_and_async())
```

## 批量查询的最佳实践

对于需要批量处理大量查询的场景（如离线评估、批量化分析），以下是一些经过验证的最佳实践：

### 实践一：限制并发数

无限制的并发可能导致 API 速率限制错误：

```python
import asyncio
from asyncio import Semaphore

semaphore = Semaphore(5)  # 最多 5 个并发

async def rate_limited_query(query_engine, question):
    async with semaphore:
        return await query_engine.aquery(question)


async def batch_query(questions, max_concurrent=5):
    semaphore = Semaphore(max_concurrent)
    query_engine = index.as_query_engine()

    async def limited(q):
        async with semaphore:
            return await query_engine.aquery(q)

    tasks = [limited(q) for q in questions]
    return await asyncio.gather(*tasks)
```

### 实践二：带重试的批量查询

网络抖动可能导致偶发的失败：

```python
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=lambda e: isinstance(e, Exception),
)
async def robust_aquery(query_engine, question):
    return await query_engine.aquery(question)
```

### 实践三：进度反馈

长时间运行的批量任务应该有进度指示：

```python
async def batch_with_progress(queries, callback=None):
    total = len(queries)
    completed = 0
    results = []

    query_engine = index.as_query_engine()

    async def track(q):
        nonlocal completed
        try:
            r = await query_engine.aquery(q)
            results.append((q, r))
        except Exception as e:
            results.append((q, None))
        completed += 1
        if callback:
            callback(completed, total, q)

    await asyncio.gather(*[track(q) for q in queries])
    return results


def print_progress(completed, total, current_query):
    pct = completed / total * 100
    print(f"\r[{completed}/{total}] ({pct:.0f}%) - {current_query[:40]}", end="")


results = await batch_with_progress(queries, callback=print_progress)
print("\n完成!")
```

## 常见误区

**误区一:"流式输出会让答案更快生成完毕"。** 不会。流式输出不会让 LLM 生成速度变快——总耗时是一样的。它的价值在于**用户体验**（不用干等）和**感知延迟**（立刻就能看到第一个字）。首字节时间（Time to First Token, TTFT）才是流式输出真正优化的指标。

**误区二:"异步一定能提速"。** 如果只有一个查询，异步反而可能略慢（有额外的协程调度开销）。异步的价值在于**多个查询的并发执行**。对于单查询场景，同步和异步几乎没有区别。

**误区三:"所有 LLM 都支持流式输出"。** 大多数主流 LLM API（OpenAI、Anthropic、Google 等）都支持流式，但某些本地模型部署方案或自定义的模型服务可能不支持。在使用前确认你的 LLM 后端是否支持流式 API。如果不支持，`streaming=True` 会退化为非流式模式（不会有报错，但没有流式效果）。
