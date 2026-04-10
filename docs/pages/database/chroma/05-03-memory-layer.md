# 5.3 对话历史管理与 Memory Layer

> **没有记忆的 RAG 系统就像金鱼——每次对话都从零开始，永远无法理解"它指的是什么"**

---

## 这一节在讲什么？

在前面两节中，我们构建的 RAG 系统是"无状态"的——每次查询都是独立的，系统不记得你之前问过什么、说过什么。但真实的对话不是这样的。当用户问"它的退款政策是什么？"时，"它"指代的是上一轮对话中提到的某个产品；当用户问"还有更详细的吗？"时，"更详细的"是相对于上一轮回答的深入。这些指代和省略在自然对话中无处不在，而要理解它们，系统必须拥有"记忆"。

这一节我们要讲的是如何用 Chroma 构建 RAG 系统的 Memory Layer——包括短期记忆（当前会话的对话历史）和长期记忆（跨会话的用户偏好和知识），以及如何让检索系统同时利用文档库和记忆库来生成更准确的回答。

---

## 为什么 RAG 系统需要记忆

让我们通过一个具体的对话场景来理解记忆的必要性：

```
用户: 苹果公司最新一季的营收是多少？
系统: 根据文档，苹果公司2024年Q3营收为948亿美元。

用户: 它的利润率呢？                    ← "它"指代"苹果公司"
系统: ???                               ← 无记忆的系统不知道"它"是谁

用户: 比上一季度呢？                    ← "上一季度"指代 Q2
系统: ???                               ← 无记忆的系统不知道在比什么

用户: 我只关心服务业务的部分            ← 用户偏好：关注服务业务
系统: ???                               ← 无记忆的系统下次还是会返回硬件数据
```

没有记忆的 RAG 系统在多轮对话中会反复要求用户澄清指代，体验非常差。而有了记忆之后，系统可以自动解析指代、保持上下文连贯、甚至记住用户的偏好来优化后续的检索结果。

---

## 记忆的两种类型

```
┌─────────────────────────────────────────────────────────────────┐
│  RAG 系统的记忆架构                                             │
│                                                                 │
│  ┌─────────────────────────────────────┐                        │
│  │  短期记忆 (Short-term Memory)        │                        │
│  │  - 当前会话的对话历史                 │                        │
│  │  - 生命周期：一次会话                 │                        │
│  │  - 存储：内存 / Chroma Collection    │                        │
│  │  - 用途：指代消解、上下文理解         │                        │
│  └─────────────────────────────────────┘                        │
│                                                                 │
│  ┌─────────────────────────────────────┐                        │
│  │  长期记忆 (Long-term Memory)         │                        │
│  │  - 跨会话的用户偏好和历史交互         │                        │
│  │  - 生命周期：永久（或 TTL 过期）      │                        │
│  │  - 存储：Chroma 持久化 Collection    │                        │
│  │  - 用途：个性化检索、用户画像         │                        │
│  └─────────────────────────────────────┘                        │
│                                                                 │
│  查询时：                                                       │
│  User Question                                                  │
│    ↓                                                            │
│  [短期记忆] → 改写查询（指代消解）                               │
│    ↓                                                            │
│  [文档库 + 长期记忆] → 混合检索                                  │
│    ↓                                                            │
│  组装 Prompt → LLM 生成回答                                     │
│    ↓                                                            │
│  更新短期记忆（追加当前轮对话）                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 短期记忆：对话历史管理

短期记忆存储当前会话的对话历史，用于指代消解和上下文理解。最简单的实现方式是把对话历史作为列表保存在内存中：

```python
class ConversationMemory:
    """短期记忆：管理当前会话的对话历史"""

    def __init__(self, max_turns: int = 10):
        self.history = []
        self.max_turns = max_turns

    def add_user_message(self, message: str):
        """添加用户消息"""
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant_message(self, message: str):
        """添加助手消息"""
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def get_history(self, last_n: int = None) -> list:
        """获取最近 N 轮对话历史"""
        if last_n:
            return self.history[-last_n * 2:]  # 每轮包含 user + assistant
        return self.history

    def get_context_string(self, last_n: int = None) -> str:
        """获取格式化的对话历史文本"""
        history = self.get_history(last_n)
        lines = []
        for msg in history:
            prefix = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def _trim(self):
        """保留最近 max_turns 轮对话"""
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def clear(self):
        """清空对话历史"""
        self.history = []


# 使用
memory = ConversationMemory(max_turns=5)
memory.add_user_message("苹果公司最新一季的营收是多少？")
memory.add_assistant_message("苹果公司2024年Q3营收为948亿美元。")
memory.add_user_message("它的利润率呢？")

print(memory.get_context_string())
# 用户: 苹果公司最新一季的营收是多少？
# 助手: 苹果公司2024年Q3营收为948亿美元。
# 用户: 它的利润率呢？
```

### 长期记忆：用户偏好与历史交互

长期记忆需要持久化存储——即使会话结束、程序重启，记忆仍然存在。Chroma 的持久化 Collection 天然适合这个场景。我们可以创建一个专门的 Memory Collection，存储用户的偏好、历史查询摘要、重要交互记录等：

```python
import chromadb
from chromadb.utils import embedding_functions
import hashlib
import time
import json


class LongTermMemory:
    """长期记忆：跨会话的用户偏好和历史交互"""

    def __init__(self, user_id: str, persist_dir: str = "./memory_db"):
        self.user_id = user_id
        self.client = chromadb.Client(settings=chromadb.Settings(
            is_persistent=True,
            persist_directory=persist_dir
        ))

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name=f"memory_{user_id}",
            embedding_function=ef
        )

    def save_preference(self, key: str, value: str):
        """保存用户偏好"""
        self.collection.upsert(
            documents=[f"用户偏好 {key}: {value}"],
            ids=[f"pref_{key}"],
            metadatas=[{
                "type": "preference",
                "key": key,
                "value": value,
                "updated_at": int(time.time())
            }]
        )

    def save_interaction(self, question: str, answer: str, sources: list = None):
        """保存一次交互记录"""
        interaction_id = hashlib.md5(
            f"{self.user_id}::{time.time()}".encode()
        ).hexdigest()[:16]

        summary = f"用户问了: {question}。回答摘要: {answer[:100]}"
        self.collection.upsert(
            documents=[summary],
            ids=[interaction_id],
            metadatas=[{
                "type": "interaction",
                "question": question[:200],
                "sources": json.dumps(sources or []),
                "created_at": int(time.time())
            }]
        )

    def search_memory(self, query: str, n_results: int = 3):
        """搜索相关记忆"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def get_preferences(self):
        """获取所有用户偏好"""
        results = self.collection.get(
            where={"type": "preference"},
            include=["metadatas"]
        )
        prefs = {}
        for meta in results['metadatas']:
            prefs[meta["key"]] = meta["value"]
        return prefs

    def cleanup_old_memories(self, max_age_days: int = 90):
        """清理过期记忆（TTL 策略）"""
        cutoff = int(time.time()) - max_age_days * 24 * 3600
        self.collection.delete(
            where={
                "created_at": {"$lt": cutoff},
                "type": "interaction"
            }
        )


# 使用
ltm = LongTermMemory(user_id="user_123")

# 保存偏好
ltm.save_preference("关注领域", "服务业务和财务数据")
ltm.save_preference("语言", "中文")

# 保存交互
ltm.save_interaction(
    question="苹果公司的服务业务收入是多少？",
    answer="苹果公司2024年Q3服务业务收入为240亿美元",
    sources=["finance_report.pdf"]
)

# 搜索记忆
memory_results = ltm.search_memory("苹果公司收入")
for i in range(len(memory_results['ids'][0])):
    print(f"  [{memory_results['distances'][0][i]:.4f}] {memory_results['documents'][0][i][:60]}...")
```

---

## 将记忆集成到 RAG 查询流程

现在我们把短期记忆和长期记忆都集成到 RAG 查询流程中：

```python
class MemoryAugmentedRAG:
    """带记忆的 RAG 系统"""

    def __init__(self, user_id: str, persist_dir: str = "./rag_memory_db"):
        self.client = chromadb.Client(settings=chromadb.Settings(
            is_persistent=True,
            persist_directory=persist_dir
        ))

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 文档库
        self.doc_collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=ef
        )

        # 记忆库
        self.memory_collection = self.client.get_or_create_collection(
            name=f"memory_{user_id}",
            embedding_function=ef
        )

        # 短期记忆
        self.short_term_memory = ConversationMemory(max_turns=5)
        self.user_id = user_id

    def ask(self, question: str, llm_generate=None):
        """带记忆的 RAG 查询"""
        # Step 1: 利用短期记忆改写查询
        history_context = self.short_term_memory.get_context_string(last_n=3)
        if history_context:
            rewritten_query = self._rewrite_with_context(question, history_context)
        else:
            rewritten_query = question

        # Step 2: 同时检索文档库和记忆库
        doc_results = self.doc_collection.query(
            query_texts=[rewritten_query],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        memory_results = self.memory_collection.query(
            query_texts=[rewritten_query],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        # Step 3: 合并检索结果
        all_contexts = []

        for i in range(len(doc_results['ids'][0])):
            if doc_results['distances'][0][i] <= 1.2:
                all_contexts.append({
                    "type": "document",
                    "content": doc_results['documents'][0][i],
                    "source": doc_results['metadatas'][0][i].get("source", "unknown"),
                    "distance": doc_results['distances'][0][i]
                })

        for i in range(len(memory_results['ids'][0])):
            if memory_results['distances'][0][i] <= 1.0:
                all_contexts.append({
                    "type": "memory",
                    "content": memory_results['documents'][0][i],
                    "distance": memory_results['distances'][0][i]
                })

        # Step 4: 组装 prompt
        context_parts = []
        for ctx in all_contexts:
            if ctx["type"] == "document":
                context_parts.append(f"[文档来源: {ctx['source']}]\n{ctx['content']}")
            else:
                context_parts.append(f"[历史记忆]\n{ctx['content']}")

        context_text = "\n\n".join(context_parts) if context_parts else "无相关信息"

        prompt = f"""你是一个专业的文档问答助手，拥有对话历史和用户记忆。

对话历史：
{history_context or "（这是新对话的开始）"}

参考信息：
{context_text}

当前问题：{question}

请基于参考信息和对话历史回答问题。注意：
1. 如果问题中有代词（它、这个、那个等），请根据对话历史理解指代
2. 优先使用文档来源的信息，历史记忆作为补充
3. 如果没有相关信息，请回答"我没有找到相关信息"

回答："""

        # Step 5: 生成回答
        if llm_generate:
            answer = llm_generate(prompt)
        else:
            answer = f"[LLM 生成需要配置] 基于检索到的 {len(all_contexts)} 条上下文"

        # Step 6: 更新记忆
        self.short_term_memory.add_user_message(question)
        self.short_term_memory.add_assistant_message(answer)

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "n_doc_contexts": sum(1 for c in all_contexts if c["type"] == "document"),
            "n_memory_contexts": sum(1 for c in all_contexts if c["type"] == "memory")
        }

    def _rewrite_with_context(self, question: str, history: str) -> str:
        """基于对话历史改写查询（简单实现）"""
        # 生产环境应该用 LLM 做查询改写
        # 这里用简单策略：把最近一轮的实体信息拼接到查询中
        return f"{question} (上下文: {history[-200:]})"
```

---

## TTL 清理策略

长期记忆不能无限增长——过期的交互记录会占用存储空间、降低检索质量、甚至引入过时信息。TTL（Time-To-Live）策略是解决这个问题的标准方法：为每条记忆设置过期时间，定期清理过期数据。

```python
def cleanup_expired_memories(collection, max_age_days: int = 90):
    """清理过期记忆"""
    cutoff = int(time.time()) - max_age_days * 24 * 3600

    count_before = collection.count()
    collection.delete(where={"created_at": {"$lt": cutoff}})
    count_after = collection.count()

    removed = count_before - count_after
    print(f"🗑️ 清理完成: 删除 {removed} 条过期记忆 (>{max_age_days}天)")
    return removed


# 建议在定时任务中执行
# 每天凌晨清理一次
# cron: 0 2 * * * python -c "from memory import cleanup_expired_memories; ..."
```

---

## 常见误区

### 误区 1：把所有对话历史都塞进 prompt

对话历史越长，prompt 越长，LLM 的推理成本越高，而且过长的历史会稀释关键信息的注意力。建议只保留最近 5~10 轮对话，更早的历史用摘要替代。

### 误区 2：长期记忆和文档库混在同一个 Collection

记忆和文档是不同类型的数据——记忆是用户特定的、有时效性的，文档是公共的、相对稳定的。把它们混在一起会导致检索时互相干扰。建议用独立的 Collection 分别存储。

### 误区 3：忽略记忆的时效性

用户三个月前问的"最新财报"和今天问的"最新财报"可能指完全不同的内容。长期记忆必须有 TTL 机制，过期的信息应该被清理或标记为"可能过时"。

---

## 本章小结

记忆是让 RAG 系统从"单轮问答"进化为"多轮对话"的关键能力。核心要点回顾：第一，短期记忆存储当前会话的对话历史，用于指代消解和上下文理解，通常保存在内存中，会话结束即消失；第二，长期记忆存储跨会话的用户偏好和历史交互，使用 Chroma 持久化 Collection 存储，支持语义检索；第三，查询时同时检索文档库和记忆库，合并结果后组装 prompt；第四，对话历史不宜过长，5~10 轮为宜，更早的历史用摘要替代；第五，长期记忆需要 TTL 清理策略，防止过期信息干扰检索质量。

下一章我们将进入生产部署与优化——如何让 Chroma 从开发原型升级为生产基础设施。
