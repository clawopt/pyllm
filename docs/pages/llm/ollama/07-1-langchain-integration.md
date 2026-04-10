# 07-1 LangChain + Ollama

## 为什么需要 LangChain

到上一章为止，我们已经能够用原生 Python + requests 库直接调用 Ollama API 来构建 RAG 系统。这种方式的好处是**完全可控、零依赖**——你清楚每一行代码在做什么。但当项目变得复杂——需要支持多种 LLM 后端切换、需要链式调用多个工具、需要管理复杂的 Prompt 模板、需要做 Agent 智能体时——纯手写代码的维护成本会急剧上升。

LangChain 的价值在于它提供了一套**标准化的抽象层和组件库**，让你可以用声明式的方式组装 LLM 应用，而不必关心底层的 HTTP 调用细节。当你的后端从 Ollama 切换到 OpenAI、或者从本地切换到云端时，只需要改一行配置。

```
┌─────────────────────────────────────────────────────────────┐
│              LangChain 在架构中的位置                         │
│                                                             │
│  你的应用代码                                                │
│  ┌───────────────────────────────────────────────┐         │
│  │  chain.invoke("分析这个文档")                   │         │
│  └───────────────────┬───────────────────────────┘         │
│                      │                                      │
│                      ▼                                      │
│  ┌───────────────────────────────────────────────┐         │
│  │            LangChain 抽象层                     │         │
│  │                                               │         │
│  │  ChatModel │ Embeddings │ DocumentLoader      │         │
│  │  TextSplitter │ VectorStore │ Retriever        │         │
│  │  Chain │ Agent │ Tool │ Memory               │         │
│  └───────────────────┬───────────────────────────┘         │
│                      │                                      │
│          ┌───────────┼───────────┐                          │
│          ▼           ▼           ▼                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                  │
│  │  Ollama   │ │  OpenAI   │ │ Anthropic │                 │
│  │ (本地)    │ │ (云端)    │ │ (云端)    │                 │
│  └──────────┘ └──────────┘ └──────────┘                  │
│                                                             │
│  💡 核心价值: 改变底层实现不需要修改上层代码                │
└─────────────────────────────────────────────────────────────┘
```

## 三种接入方式

### 方式一：ChatOllama（官方推荐）

```bash
# 安装 LangChain Ollama 集成包
pip install langchain-ollama
```

```python
#!/usr/bin/env python3
"""LangChain + Ollama 完整集成指南 - 方式一: ChatOllama"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


def basic_chat():
    """基础对话"""
    
    # 初始化模型
    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.7,
        base_url="http://localhost:11434"  # 默认值，可省略
    )
    
    # 简单调用
    response = llm.invoke([HumanMessage(content="什么是 RAG？")])
    print(f"回答: {response.content}")
    
    # 流式输出
    print("\n流式输出:")
    for chunk in llm.stream([HumanMessage(content="写一首关于 Python 的诗")]):
        print(chunk.content, end="", flush=True)
    print()


def structured_chat():
    """结构化多轮对话"""
    
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)
    
    messages = [
        SystemMessage(content="你是一个专业的 Python 编程助手。回答要简洁且包含代码示例。"),
        HumanMessage(content="如何用 Python 读取 JSON 文件？"),
    ]
    
    response = llm.invoke(messages)
    print(response.content)


def with_options():
    """带完整参数控制"""
    
    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.2,
        top_p=0.9,
        num_ctx=8192,
        repeat_penalty=1.15,
        seed=42,
        # Ollama 特有参数也可以传
        num_predict=2048,       # 最大生成长度
        stop=["</think>", "```"],  # 停止序列
    )
    
    response = llm.invoke([
        HumanMessage(content="解释什么是量子计算")
    ])
    print(response.content)


if __name__ == "__main__":
    basic_chat()
    structured_chat()
    with_options()
```

### 方式二：ChatOpenAI 兼容模式（最灵活）

这是我最推荐的方式，因为 **OpenAI SDK 是整个 AI 生态的事实标准接口**。几乎所有框架都支持 OpenAI 格式，使用这种接入方式可以获得最大的生态兼容性：

```python
#!/usr/bin/env python3
"""方式二: 通过 OpenAI SDK 兼容层访问 Ollama"""

from openai import OpenAI

# 关键：将 base_url 指向 Ollama 的 OpenAI 兼容端点
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama 不需要真实 key，但不能为空
)


def chat_completion():
    """标准的 OpenAI Chat Completion 接口"""
    
    response = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": "你是技术专家"},
            {"role": "user", "content": "解释微服务架构的优缺点"}
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    
    print(response.choices[0].message.content)


def streaming_chat():
    """流式输出"""
    
    stream = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "写一个快速排序算法"}],
        stream=True,
        temperature=0.2,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def embedding():
    """Embedding 接口（通过 OpenAI 兼容端点）"""
    
    response = client.embeddings.create(
        model="nomic-embed-text",
        input="这是一段测试文本"
    )
    
    embedding = response.data[0].embedding
    print(f"维度: {len(embedding)}")
    print(f"前5个值: {embedding[:5]}")


# === LangChain 集成：使用 ChatOpenAI 指向 Ollama ===
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def langchain_with_ollama_via_openai():
    """在 LangChain 中通过 OpenAI 兼容层使用 Ollama"""
    
    # LLM
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5:7b",
        temperature=0.3,
    )
    
    # Embedding
    embeddings = OpenAIEmbeddings(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="nomic-embed-text"
    )
    
    # 使用方式和调用真正的 OpenAI API 完全一致！
    response = llm.invoke("什么是向量数据库？")
    print(response.content)
    
    # 测试 Embedding
    vec = embeddings.embed_query("测试文本")
    print(f"\nEmbedding 维度: {len(vec)}")


if __name__ == "__main__":
    chat_completion()
    streaming_chat()
    embedding()
    langchain_with_ollama_via_openai()
```

### 方式三：自定义 LLM 类（完全控制）

当你需要对 Ollama 的调用行为做深度定制时：

```python
#!/usr/bin/env python3
"""方式三: 自定义 LangChain LLM 类"""

from langchain_core.language_models.llms import LLM
from langchain_core.outputs import LLMResult, Generation
from typing import Optional, List, Any
import requests


class CustomOllamaLLM(LLM):
    """自定义 Ollama LLM 实现"""
    
    model_name: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    num_ctx: int = 4096
    
    @property
    def _llm_type(self) -> str:
        return "custom-ollama"
    
    def _call(self, prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Any = None,
              **kwargs) -> str:
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            }
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        
        return resp.json()["response"]
    
    def _generate(self, prompts: List[str],
                  stop: Optional[List[str]] = None,
                  run_manager: Any = None,
                  **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


# 使用
if __name__ == "__main__":
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    llm = CustomOllamaLLM(model_name="qwen2.5:7b", temperature=0.5)
    
    chain = (
        PromptTemplate.from_template("用一句话解释: {topic}") 
        | llm 
        | StrOutputParser()
    )
    
    result = chain.invoke({"topic": "区块链技术"})
    print(result)
```

## 完整 Chain 示例：RAG Pipeline

```python
#!/usr/bin/env python3
"""
完整的 LangChain RAG Pipeline — 基于 Ollama
展示: 文档加载 → 分块 → Embedding → 向量存储 → 检索 → 生成
"""

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_rag_pipeline(docs_path="./docs"):
    """构建完整的 RAG 处理管道"""
    
    # === 组件初始化 ===
    
    # LLM
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)
    
    # Embedding
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 文档加载
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"📄 加载了 {len(documents)} 个文档")
    
    # 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )
    splits = text_splitter.split_documents(documents)
    print(f"✂️ 分块后共 {len(splits)} 个文本块")
    
    # 向量库
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_langchain"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # === 构建 Chain ===
    
    template = """基于以下参考文档回答用户问题。
如果文档中没有相关信息，请明确说明。

参考文档:
{context}

用户问题: {question}
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


if __name__ == "__main__":
    pipeline = build_rag_pipeline("./my_docs")
    
    while True:
        question = input("\n❓ 输入问题 (q退出): ").strip()
        if question.lower() in ["q", "quit"]:
            break
        
        if not question:
            continue
        
        print(f"\n🤔 思考中...")
        answer = pipeline.invoke(question)
        print(f"\n💡 回答:\n{answer}")
```

## Agent 模式：Tool Calling

Ollama 中支持函数调用（Function Calling）的模型（如 Qwen2.5:7b-tools、Llama 3.1）可以与 LangChain 的 Agent 能力结合：

```python
#!/usr/bin/env python3
"""LangChain Agent + Ollama Tool Calling"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气"""
    return f"{city}今天晴朗，温度 25°C，适合户外活动。"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def search_docs(query: str) -> str:
    """在内部知识库中搜索文档"""
    return f"找到与 '{query}' 相关的 3 篇文档：\n1. Ollama 部署指南\n2. Docker 配置说明\n3. API 参考手册"


def run_agent():
    """运行 Tool Calling Agent"""
    
    # 注意：需要支持 tools 的模型版本
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.1)
    
    tools = [get_weather, calculate, search_docs]
    
    agent = create_tool_calling_agent(llm, tools, prompt=None)
    agent_executor = AgentExecutor(agent=tools + [agent], verbose=True)
    
    queries = [
        "北京今天天气怎么样？",
        "计算 (15 + 27) * 3",
        "帮我找一下 Ollama 部署相关的文档",
        "如果北京天气好的话，帮我算一下 100 除以 3"
    ]
    
    for q in queries:
        print(f"\n{'='*50}")
        print(f"用户: {q}")
        result = agent_executor.invoke({"input": q})
        print(f"助手: {result['output']}")


if __name__ == "__main__":
    run_agent()
```

## Streaming 输出详解

```python
#!/usr/bin/env python3
"""LangChain Streaming 与 Ollama"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import asyncio


async def streaming_demo():
    """异步流式输出演示"""
    
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.8)
    
    prompt = ChatPromptTemplate.from_template("写一个关于{topic}的故事")
    chain = prompt | llm
    
    print("📖 开始生成故事...\n")
    
    async for chunk in chain.astream({"topic": "一只学会编程的猫"}):
        content = chunk.content
        if content:
            print(content, end="", flush=True)
    
    print("\n\n✅ 完成!")


def streaming_with_astream_events():
    """更细粒度的 astream_events 流式处理"""
    
    llm = ChatOllama(model="qwen2.5:7b")
    
    print("🔍 使用 astream_events 监听详细事件:\n")
    
    async for event in llm.astream_events(
        "用 Python 写一个斐波那契数列生成器",
        version="v1",
    ):
        kind = event["event"]
        
        if kind == "on_chat_model_start":
            print("  [开始] 模型开始推理...")
        elif kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)
        elif kind == "on_chat_model_end":
            output_tokens = event["data"]["output"].usage_metadata.get("output_tokens", "?")
            print(f"\n  [完成] 生成了约 {output_tokens} tokens")


if __name__ == "__main__":
    import asyncio
    asyncio.run(streaming_demo())
    print()
    asyncio.run(streaming_with_astream_events())
```

## 多模型路由：根据复杂度选择模型

```python
#!/usr/bin/env python3
"""智能模型路由: 简单问题用小模型，复杂问题用大模型"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch


class SmartRouter:
    """基于问题复杂度的智能路由器"""
    
    def __init__(self):
        # 小模型：快速、便宜
        self.small_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.3)
        
        # 大模型：慢但质量高
        self.large_llm = ChatOllama(model="qwen2.5:32b", temperature=0.3)
        
        # 分类器模型（用于判断复杂度）
        self.classifier = ChatOllama(
            model="qwen2.5:3b", 
            temperature=0.01,
            format="json"  # 强制 JSON 输出
        )
    
    def classify_complexity(self, question: str) -> str:
        """判断问题的复杂度"""
        
        prompt = f"""判断以下问题的复杂度。
只返回JSON: {{"complexity": "simple|medium|complex"}}
问题: {question}"""
        
        response = self.classifier.invoke(prompt).content
        
        import json
        try:
            data = json.loads(response)
            return data.get("complexity", "medium")
        except:
            return "medium"
    
    def invoke(self, question: str) -> str:
        """根据复杂度路由到不同模型"""
        
        complexity = self.classify_complexity(question)
        print(f"📊 问题复杂度: {complexity}")
        
        if complexity == "simple":
            print("→ 使用小模型 (qwen2.5:1.5b)")
            llm = self.small_llm
        elif complexity == "medium":
            print("→ 使用中等模型 (qwen2.5:7b)")
            llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)
        else:
            print("→ 使用大模型 (qwen2.5:32b)")
            llm = self.large_llm
        
        return llm.invoke(question).content


if __name__ == "__main__":
    router = SmartRouter()
    
    questions = [
        "你好",
        "Python 中怎么反转一个列表？",
        "请设计一个分布式系统的 CAP 定理权衡方案，并给出具体的工程实践建议"
    ]
    
    for q in questions:
        print(f"\n❓ {q}")
        answer = router.invoke(q)
        print(f"💡 {answer[:200]}{'...' if len(answer)>200 else ''}\n")
```

## 本章小结

这一节全面介绍了 LangChain 与 Ollama 的三种集成方式：

1. **ChatOllama**（`langchain-ollama` 包）：官方封装，开箱即用，最简单
2. **ChatOpenAI 兼容模式**：利用 `base_url="http://localhost:11434/v1"` 将 Ollama 伪装成 OpenAI，获得最大的生态兼容性——**强烈推荐**
3. **自定义 LLM 类**：继承 `langchain_core.language_models.LLM`，完全控制调用逻辑
4. **完整 RAG Pipeline** 展示了 LangChain 的声明式链式编程威力
5. **Agent + Tool Calling** 让支持工具调用的 Ollama 模型具备执行能力
6. **Streaming 支持**包括 `astream()` 和 `astream_events()` 两种粒度
7. **多模型路由**可以根据问题复杂度自动选择合适的模型，平衡速度和质量

下一节我们将学习 LlamaIndex 框架与 Ollama 的集成。
