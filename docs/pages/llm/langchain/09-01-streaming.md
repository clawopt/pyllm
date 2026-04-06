---
title: 流式输出的价值与实现
description: stream() vs invoke() 的区别、Token 级流式输出、流式在 Chain 中的传递、SSE/WebSocket 实时推送
---
# 流式输出的价值与实现

在前面的所有章节中，我们调用 LLM 时主要使用 `invoke()`——它的工作方式是**等待模型生成完整响应后一次性返回**。这种方式简单直接，但在实际产品中有一个致命的体验问题：用户可能要盯着空白屏幕等好几秒甚至十几秒才能看到任何输出。

这一节我们将学习 **流式输出（Streaming）**——让模型的生成过程像打字一样逐字（或逐 token）地展示出来，大幅提升用户感知的响应速度。

## 流式 vs 非流式：直观对比

先用一个简单的例子感受差异：

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 方式一：非流式 — 等待全部完成
print("=== 非流式 ===")
response = chat.invoke("用三句话介绍 Python")
print(response.content)
```

运行效果：用户发送请求后，屏幕空白了约 **3-5 秒**，然后整段文字突然一起出现。

```python
# 方式二：流式 — 边生成边显示
print("\n=== 流式 ===")
for chunk in chat.stream("用三句话介绍 Python"):
    print(chunk.content, end="", flush=True)
print()
```

运行效果：文字像打字机一样一个字一个字地出现。虽然总耗时可能和非流式差不多，但**用户在第一个 token 产生后（约 0.3-0.5 秒）就能看到内容**，体感上快得多。

### 为什么流式"感觉更快"

这涉及到人类心理学的 **感知延迟 vs 实际延迟**：

| 模式 | 首字节延迟 | 总耗时 | 用户感知 |
|------|-----------|--------|---------|
| `invoke()` | ~3000ms | ~3500ms | "好慢！" |
| `stream()` | ~300ms | ~3500ms | "挺快的！" |

首字节延迟（Time to First Token, TTFT）从 3 秒降到了 0.3 秒——**快了 10 倍**。这就是为什么几乎所有 AI 产品（ChatGPT、Claude、文心一言）都使用流式输出。

## stream() 的基本用法

### 单次流式调用

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

stream = chat.stream("解释什么是装饰器")

for chunk in stream:
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()
```

每个 `chunk` 是一个 `AIMessageChunk` 对象：
- `chunk.content`：当前这一小段文本
- `chunk.response_metadata`：元信息（如 finish_reason）

### 流式输出的数据结构

```python
stream = chat.stream("说一句话")

for i, chunk in enumerate(stream):
    print(f"[{i}] type={type(chunk).__name__}, "
          f"content='{chunk.content}', "
          f"len={len(chunk.content) if chunk.content else 0}")
```

典型输出：

```
[0] type=AIMessageChunk, content='你', len=1
[1] type=AIMessageChunk, content='好', len=1
[2] type=AIMessageChunk, content='！', len=1
[3] type=AIMessageChunk, content='有', len=2
[4] type=AIMessageChunk, content='什', len=2
[5] type=AIMessageChunk, content='么', len=2
[6] type=AIMessageChunk, content='可以', len=3
[7] type=AIMessageChunk, content='帮', len=2
[8] type=AIMessageChunk, content='你的', len=3
[9] type=AIMessageChunk, content='？', len=1
```

注意观察：
- 每个 chunk 的 `content` 可能是 1 个字符、几个字符、甚至空字符串（有些 chunk 只包含元数据）
- 中文字符通常被分成更小的 chunk（因为中文的 token 化比较特殊）

## 在 Chain 中使用流式

`stream()` 不仅对单独的 LLM 调用有效，在整个 Chain 管道中也能正常工作：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("用{style}的风格解释{topic}")
parser = StrOutputParser()

chain = prompt | chat | parser

# Chain 的 stream() 同样返回迭代器
for chunk in chain.stream({"style": "幽默", "topic": "量子计算"}):
    print(chunk, end="", flush=True)
print()
```

关键点：**Chain 的 `stream()` 会自动将流式效果传递到管道中的每一个组件**。`prompt` 不产生流式输出（它是同步的），但 `chat` 会逐 token 返回，`parser` 对每个 chunk 做 parse 操作。

### astream()：异步流式

对于异步应用（如 Web 服务器），LangChain 还提供了异步版本：

```python
import asyncio
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def main():
    async for chunk in chat.astream("讲个笑话"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

asyncio.run(main())
```

`astream()` 是 `stream()` 的异步版本，底层使用 `async for` 来消费异步迭代器。在 FastAPI 等 Web 框架中使用 `astream()` 可以避免阻塞事件循环。

## 流式输出的实际价值场景

### 场景一：CLI 交互程序

命令行工具中流式输出能让用户知道"程序还在工作"：

```python
def chat_loop():
    chat = ChatOpenAI(model="gpt-4o-mini")
    
    while True:
        question = input("\n你: ").strip()
        if not question or question.lower() == "退出":
            break
        
        print("\n助手: ", end="", flush=True)
        for chunk in chat.stream(question):
            print(chunk.content, end="", flush=True)
        print()
```

没有流式的话，用户每次提问后都要面对几秒的空白屏幕，体验很差。

### 场景二：Web 应用实时推送

在 Web 应用中，流式输出可以通过 SSE（Server-Sent Events）或 WebSocket 实时推送到浏览器：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

app = FastAPI()
chat = ChatOpenAI(model="gpt-4o-mini")

@app.post("/chat")
async def chat_endpoint(request: dict):
    question = request.get("message")

    async def generate():
        async for chunk in chat.astream(question):
            if chunk.content:
                yield f"data: {json.dumps({'content': chunk.content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

前端 JavaScript：

```javascript
const response = await fetch("/chat", {
    method: "POST",
    body: JSON.stringify({message: userQuestion})
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    const data = decoder.decode(value);
    // 解析 SSE 数据并追加到页面
}
```

这样用户就能在网页上看到文字逐渐出现的效果——和 ChatGPT 的体验一致。

### 场景三：长文本生成

当模型需要生成很长的内容（如完整的代码文件、长篇报告）时，流式输出不仅是体验优化，更是**功能必需**——如果等待 30 秒才一次性返回，用户很可能以为程序卡死了而刷新页面。

## 流式的限制和注意事项

**注意一：流式输出不能中途取消**

一旦开始 `stream()` 循环，你必须消费完整个迭代器。目前 LangChain 的标准流式接口不支持中途取消（cancel）。如果你需要这个能力，需要自己实现超时或取消机制。

**注意二：流式 + OutputParser 的兼容性**

大多数 Output Parser 都支持流式模式，但自定义解析器需要注意——它必须能处理不完整的输入（因为每个 chunk 只是最终答案的一部分）：

```python
class MyCustomParser(BaseOutputParser):
    def parse(self, text):
        return text.upper()   # 最终答案转大写
    
    def parse_partial(self, text):
        # 流式模式下也会调用此方法
        # 注意：text 此时可能是不完整的！
        return text          # 流式时不做转换，原样输出
```

如果你的 parser 在流式模式下做了不可逆的操作（比如把半截断的 JSON 尝试解析），就会报错。

**注意三：流式增加的复杂度**

流式输出的错误处理比非流式复杂得多——你需要区分"连接断开"、"模型出错"、"解析失败"等各种情况，每种情况的恢复策略都不同。

到这里，我们已经掌握了流式输出的基础用法。下一节我们将学习另一个性能利器——**异步编程**，它能让你的应用同时处理多个并发请求。
