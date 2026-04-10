# LangChain + vLLM

> **白板时间**：你已经有一个跑在 vLLM 上的 LLM 服务了。现在你想用它构建一个 RAG（检索增强生成）应用——需要文档加载、文本分块、向量检索、Prompt 模板、对话记忆、工具调用……手写这些代码要几百行。但如果你用 **LangChain**，这些能力都是现成的组件，你只需要把它们"串"起来。而 vLLM 作为 OpenAI 兼容的后端，可以无缝接入 LangChain 的整个生态系统。

## 一、两种集成方式

### 1.1 方式一：ChatOpenAI 兼容模式（推荐）

这是最简单的方式——利用 vLLM 的 OpenAI 兼容性：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-vllm",     # vLLM 不验证 API key
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
    max_tokens=1024,
)

response = llm.invoke("用Python实现一个二分查找算法")
print(response.content)
```

**优点**：
- ✅ 零学习成本——所有 LangChain 教程/示例直接可用
- ✅ 支持 Chain / Agent / Tool / RAG / Memory / Streaming 全部功能
- ✅ 异步支持 (`AsyncOpenAI`)
- ✅ 切换模型只需改 `base_url` 和 `model`

### 1.2 方式二：vLLM 原生集成

```python
from langchain_community.llms import VLLM

llm = VLLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    max_new_tokens=512,
    top_p=0.9,
    temperature=0.7,
    vllm_kwargs={"tensor_parallel_size": 1},
)

response = llm.invoke("解释 PagedAttention 的原理")
print(response)
```

**适用场景**：
- 离线批量推理（不需要 HTTP 开销）
- 更细粒度的 vLLM 参数控制
- 需要在同一个进程中同时做 Embedding 和 LLM 推理

## 二、完整 RAG Pipeline

### 2.1 端到端 RAG 示例

比如下面的程序构建了一个完整的本地 RAG 系统：

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


def build_rag_pipeline(docs_dir: str = "./documents"):
    """构建完整的 RAG Pipeline"""
    
    # ===== Step 1: LLM (vLLM 后端) =====
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.3,
        streaming=True,
    )
    
    # ===== Step 2: Embeddings (vLLM 或其他) =====
    embeddings = OpenAIEmbeddings(
        base_url="http://localhost:8001/v1",   # 如果有 embedding 模型
        api_key="not-needed",
        model="BAAI/bge-m3",
    )
    
    # ===== Step 3: 加载文档 =====
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    documents = loader.load()
    
    print(f"[加载] {len(documents)} 个文档")
    
    # ===== Step 4: 文本分块 =====
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    splits = text_splitter.split_documents(documents)
    
    print(f"[分块] {len(splits)} 个文本块")
    
    # ===== Step 5: 向量化存储 =====
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # ===== Step 6: 构建 Prompt =====
    prompt_template = """
    你是一个专业的技术助手。请基于以下参考文档回答用户的问题。
    如果你不知道答案，请诚实地说不知道。
    
    参考文档:
    {context}
    
    用户问题: {question}
    
    请用中文回答:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # ===== Step 7: 组装 Chain =====
    rag_chain = create_retrieval_chain(llm, retriever, prompt)
    
    return rag_chain


def query_rag(rag_chain, question: str):
    """查询 RAG"""
    
    response = rag_chain.invoke({"question": question})
    
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"{'='*60}")
    print(f"回答: {response['answer']}")
    print(f"\n参考来源:")
    for i, doc in enumerate(response["context"]):
        print(f"  [{i+1}] {doc.page_content[:120]}...")


# 使用示例
if __name__ == "__main__":
    rag = build_rag_pipeline("./docs")
    
    questions = [
        "PagedAttention 是什么？它解决了什么问题？",
        "Continuous Batching 和 Static Batching 有什么区别？",
        "如何选择合适的量化方案？",
    ]
    
    for q in questions:
        query_rag(rag, q)
```

## 三、Agent + Tool Calling

### 3.1 ReAct Agent 模式

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
import json
import requests


# 定义工具函数
@tool
def search_web(query: str) -> str:
    """搜索网络获取实时信息"""
    return f"搜索结果: 关于 '{query}' 的最新信息是...（模拟结果）"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况"""
    weather_data = {
        "北京": "晴, 22°C, 湿度 45%",
        "上海": "多云, 26°C, 湿度 72%",
        "深圳": "阵雨, 28°C, 湿度 85%",
    }
    return weather_data.get(city, f"{city}: 暂无数据")


# 创建 Agent
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.0,
)

tools = [search_web, calculate, get_weather]

agent = create_react_agent(llm, tools, prompt="")
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行 Agent
result = agent_executor.invoke({
    "input": "北京和深圳现在的天气怎么样？另外帮我算一下 2^20 是多少"
})

print(f"\n[最终回答]\n{result['output']}")
```

### 3.2 带 Function Calling 的 Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional


class SearchQuery(BaseModel):
    """搜索查询参数"""
    query: str = Field(description="搜索关键词")
    category: Optional[str] = Field(default=None, description="分类过滤")


@tool(args_schema=SearchQuery)
def search_database(query: str, category: str = None) -> str:
    """在知识库中搜索信息"""
    
    mock_db = [
        {"title": "PagedAttention 原理", "category": "技术", 
         "content": "PagedAttention 将 KV Cache 切分为固定大小的 Block..."},
        {"title": "vLLM 安装指南", "category": "教程",
         "content": "pip install vllm 即可完成安装..."},
        {"title": "性能调优最佳实践", "category": "运维",
         "content": "建议使用 AWQ INT4 量化 + Prefix Caching..."},
    ]
    
    results = []
    for item in mock_db:
        if query.lower() in item["title"].lower() or query.lower() in item["content"].lower():
            if not category or item["category"] == category:
                results.append(item)
    
    if not results:
        return f"未找到与 '{query}' 相关的结果"
    
    return json.dumps(results[:3], ensure_ascii=False, indent=2)


llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="Qwen/Qwen2.5-7B-Instruct",
)

llm_with_tools = llm.bind_tools([search_database])

response = llm_with_tools.invoke(
    "查找关于性能优化的文档",
)

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"调用工具: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")
        
        result = search_database(**tool_call["args"])
        print(f"结果: {result}")
```

## 四、Streaming 输出

### 4.1 流式 Token 输出

```python
from langchain_openai import ChatOpenAI
import sys

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="Qwen/Qwen2.5-7B-Instruct",
    streaming=True,
)

prompt = "写一首关于春天的七言绝句"

print("[流式输出] ")
for chunk in llm.stream([{"role": "user", "content": prompt}]):
    content = chunk.content
    if content:
        print(content, end="", flush=True)

print("\n\n[完成]")
```

### 4.2 astream_events 完整事件流

```python
async def streaming_demo():
    """完整的事件流演示"""
    
    from langchain_openai import AsyncChatOpenAI
    
    llm = AsyncChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="Qwen/Qwen2.5-7B-Instruct",
        streaming=True,
    )
    
    async for event in llm.astream_events(
        "列举人工智能的十大应用领域",
        version="v1",
        include_names=["on_chat_model_start", "on_chat_model_stream"],
    ):
        event_type = event["event"]
        
        if event_type == "on_chat_model_start":
            model_name = event["data"]["input"]["model"]
            print(f"[启动] 模型: {model_name}")
        
        elif event_type == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)


import asyncio
asyncio.run(streaming_demo())
```

## 五、Memory（对话记忆）

### 5.1 对话历史管理

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
    BaseChatMessageHistoryStore,
)


class SessionManager:
    """多会话记忆管理器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
            model="Qwen/Qwen2.5-7B-Instruct",
        )
        self.sessions: dict[str, InMemoryChatMessageHistory] = {}
    
    def get_session(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()
        return self.sessions[session_id]
    
    async def chat(self, session_id: str, message: str):
        """带记忆的对话"""
        
        history = self.get_session(session_id)
        
        await history.aadd_message(HumanMessage(content=message))
        
        response = await self.llm.ainvoke(
            history.messages + [SystemMessage(content="你是有用的助手。")]
        )
        
        await history.add_message(AIMessage(content=response.content))
        
        return {
            "reply": response.content,
            "history_length": len(history.messages),
        }


manager = SessionManager()

r1 = manager.chat("user_001", "我叫小明，我喜欢编程")
print(f"A: {r1.reply()}")

r2 = manager.chat("user_001", "我叫什么名字？")
print(f"A: {r2.reply()}")  # 应该记住用户叫小明
```

---

## 六、总结

本节完成了 LangChain 与 vLLM 的全面集成：

| 能力 | 代码 | 说明 |
|------|------|------|
| **基本调用** | `ChatOpenAI(base_url=vllm)` | 一行切换到 vLLM |
| **RAG Pipeline** | Loader → Splitter → VectorStore → Retriever → Chain | 完整检索增强生成 |
| **Agent + Tools** | `create_react_agent()` + `@tool` | 自主规划 + 工具调用 |
| **Function Calling** | `bind_tools()` + Pydantic schema | 结构化工具参数 |
| **Streaming** | `.stream()` / `.astream_events()` | 逐 token 流式输出 |
| **Memory** | `InMemoryChatMessageHistory` | 多轮对话历史管理 |

**核心要点回顾**：

1. **ChatOpenAI 兼容模式是首选**——零学习成本，所有 LangChain 代码直接复用
2. **vLLM 同时充当 LLM 和 Embedding 提供者**——一个服务搞定 RAG 全栈
3. **Agent 模式让 LLM 具备了自主行动能力**——vLLM + LangChain = 可靠的 Agent 基础设施
4. **Streaming 在交互场景下必不可少**——TTFT < 500ms 的用户体验关键
5. **Memory 让对话从单次变成持续**——结合 vLLM 的 Prefix Cache 效果更好

下一节我们将学习 **LlamaIndex + vLLM**——另一个强大的 LLM 应用框架。
