# 5.3 DeepSeek + RAG 实战

> **RAG 让 AI 从"通才"变成"专家"——先检索你的私有数据，再生成精准回答。**

---

## 这一节在讲什么？

DeepSeek 的知识有截止日期——它不知道你公司的内部文档、最新的项目代码、客户的业务数据。RAG（Retrieval-Augmented Generation）解决了这个问题——先从你的私有数据中检索相关内容，再把检索结果作为上下文传给 DeepSeek，让 AI 基于你的数据生成回答。这一节我们用 DeepSeek + Chroma 构建一个完整的 RAG 系统。

---

## RAG 架构

```
RAG 的工作流程：

  1. 离线阶段：把文档切分成片段 → 生成向量 → 存入向量数据库
  2. 在线阶段：
     用户问题 → 向量化 → 在向量数据库中检索 top-k 相关片段
     → 把检索结果作为上下文 → 发送给 DeepSeek → 生成回答
```

---

## 完整的 RAG 实现

```python
import chromadb
from openai import OpenAI

client = OpenAI(api_key="sk-xxxxxxxx", base_url="https://api.deepseek.com")

# 1. 初始化 Chroma 向量数据库
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("documents")

# 2. 文档切分和索引
documents = [
    "DeepSeek-V3 是通用模型，擅长代码生成和内容创作，价格 $0.27/M input token。",
    "DeepSeek-R1 是推理模型，擅长数学推理和逻辑分析，价格 $0.55/M input token。",
    "DeepSeek 的 API 完全兼容 OpenAI 格式，只需改 base_url 即可迁移。",
    "DeepSeek 支持 Function Calling，最多 128 个工具定义。",
    "DeepSeek-R1 的思考过程通过 reasoning_content 字段返回。"
]

for i, doc in enumerate(documents):
    collection.add(
        ids=[f"doc_{i}"],
        documents=[doc]
    )

# 3. RAG 查询函数
def rag_query(question: str) -> str:
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": f"根据以下上下文回答用户问题。如果上下文中没有相关信息，请说'我不知道'。\n\n上下文：\n{context}"
            },
            {"role": "user", "content": question}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content

# 4. 测试
print(rag_query("DeepSeek-R1 的价格是多少？"))
print(rag_query("DeepSeek 支持多少个工具定义？"))
```

---

## Function Calling + RAG

更高级的做法是用 Function Calling 封装检索逻辑——让 Agent 自主决定何时检索：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "在知识库中搜索相关文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    }
]

def search_docs(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=3)
    return "\n".join(results["documents"][0])

# 在 Agent 循环中使用
# Agent 会根据用户问题自主决定是否调用 search_docs
```

---

## 常见误区

**误区一：RAG 检索的文档越多越好**

不是。检索太多文档会占用上下文窗口，降低 AI 的回答质量。建议只检索 top-3 到 top-5 最相关的片段。

**误区二：RAG 能解决所有知识更新问题**

RAG 能解决"私有数据"的问题，但不能解决"推理能力"的问题。如果你的问题需要深度推理，RAG 检索到的文档只是"参考资料"，AI 仍然需要正确理解和推理。

**误区三：Chroma 的默认 embedding 就够用**

Chroma 的默认 embedding 模型是 `all-MiniLM-L6-v2`，对英文效果好但对中文效果一般。如果你的文档主要是中文，建议使用支持中文的 embedding 模型。

**误区四：RAG 不需要维护**

需要。当你的文档更新时，需要重新索引；当数据量增大时，需要优化检索策略。RAG 不是"一劳永逸"的——它需要持续维护。

---

## 小结

这一节我们用 DeepSeek + Chroma 构建了一个 RAG 系统：文档切分 → 向量化 → 存入 Chroma → 检索相关片段 → 作为上下文传给 DeepSeek → 生成回答。更高级的做法是用 Function Calling 封装检索逻辑，让 Agent 自主决定何时检索。下一章我们进入代码生成与调试实战。
