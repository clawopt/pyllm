# 1.3 环境安装与第一个 Hello World

> **从 pip install 到跑通第一个查询——让 Chroma 在你的机器上真正活起来**

---

## 安装：三条路，选一条就行

Chroma 的设计哲学是**零配置启动**——你不需要装 PostgreSQL、不需要配置集群、甚至不需要写一行配置文件。它有三种安装方式，按使用场景递进：

### 方式一：pip 直接安装（推荐新手）

这是最简单的方式，适合本地开发、原型验证、Jupyter Notebook 实验：

```bash
pip install chromadb
```

就这么一行。它会拉取 Chroma 的核心依赖（onnxruntime、posthog、fastapi 等），整个过程通常在 1~2 分钟内完成。

如果你需要 GPU 加速的 embedding 推理（比如用 sentence-transformers 做大规模向量化），可以额外装：

```bash
pip install chromadb[dev]        # 开发依赖（测试、lint 工具）
pip install chromadb[embeddings] # 额外的 embedding provider 支持
```

### 方式二：Docker 容器化（推荐生产/团队协作）

当你需要把 Chroma 作为独立服务运行、或者团队成员共享同一个数据库实例时，Docker 是最干净的选择：

```bash
docker pull chromadb/chroma:latest
docker run -p 8000:8000 chromadb/chroma:latest
```

这会启动一个 Chroma Server，监听在 `http://localhost:8000`，提供 HTTP API 和 gRPC 接口。客户端代码只需要改一行连接地址：

```python
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
```

Docker 方式的优势在于：
- **环境隔离**：不会污染宿主机的 Python 环境
- **版本一致性**：团队成员用同一个镜像，避免"我这边能跑"的问题
- **易于部署**：直接推到任何支持 Docker 的服务器上

### 方式三：从源码编译（贡献者/深度定制）

如果你想修改 Chroma 源码、或者需要最新的未发布特性：

```bash
git clone https://github.com/chroma-core/chroma.git
cd chroma
poetry install          # 用 poetry 管理 Python 依赖
make docker-compose-up   # 启动完整开发环境（含前端 UI）
```

这种方式会给你一个完整的开发环境，包括 Chroma 的 Web UI（默认在 `http://localhost:3000`），可以可视化地浏览 Collection 和执行查询。

---

## 第一个 Hello World：五步走通全流程

好，假设你已经 `pip install chromadb` 成功了。让我们用最少的代码走一遍完整的 CRUD 流程。打开你的 Python REPL 或新建一个 `hello_chroma.py`：

```python
import chromadb

# Step 1: 创建 Client —— 这是所有操作的入口
client = chromadb.Client()

# Step 2: 创建 Collection —— 类似于 SQL 的 CREATE TABLE
collection = client.create_collection(name="my_first_collection")

# Step 3: 添加文档 —— 类似于 INSERT INTO
collection.add(
    documents=[
        "Chroma 是一个开源的向量数据库",
        "向量数据库用于存储和检索高维向量",
        "RAG 是检索增强生成的缩写"
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Step 4: 执行查询 —— 类似于 SELECT ... WHERE similarity()
results = collection.query(
    query_texts=["什么是 RAG？"],
    n_results=2
)

# Step 5: 打印结果
print(results)
```

运行这段代码，你会看到类似这样的输出：

```
{
  'ids': [['doc3', 'doc1']],
  'distances': [[0.523, 0.789]],
  'documents': [['RAG 是检索增强生成的缩写', 'Chroma 是一个开源的向量数据库']],
  'metadatas': [None],
  'embeddings': None,
  'data': None,
  'uris': None,
  'included': ['metadatas', 'documents', 'distances']
}
```

这个输出看起来有点复杂，但别担心——我们逐字段拆解。

---

## 结果解析：每个字段的含义是什么？

`collection.query()` 返回的是一个字典，里面包含多个列表。理解这些字段是使用 Chroma 的基本功：

### `ids` — 文档 ID 列表

```python
results['ids']  # [['doc3', 'doc1']]
```

返回的是与查询文本**最相似的文档 ID**，按相似度降序排列。注意这里是一个嵌套列表——外层对应 `query_texts` 的每条查询（我们只传了一条），内层是匹配到的文档 ID。

**为什么 doc3 排第一？** 因为我们问的是"什么是 RAG？"，而 doc3 的内容正好包含"RAG"这个词组。Chroma 内部的 embedding 模型捕捉到了这种语义关联。

### `distances` — 距离分数

```python
results['distances']  # [[0.523, 0.789]]
```

这是查询向量与每个结果向量之间的**距离值**。距离越小表示越相似。具体含义取决于 Collection 创建时指定的 distance metric：

| Distance Metric | 距离越小意味着 | 典型范围 |
|----------------|--------------|---------|
| `cosine`（默认） | 余弦距离小 → 相似度高 | [0, 2] |
| `l2` | 欧氏距离小 → 向量接近 | [0, +∞) |
| `ip` | 内积大（但 Chroma 返回的是距离化的 IP） | 取决于向量模长 |

对于 cosine 距离来说，0 表示完全相同，2 表示完全相反（方向相反的单位向量）。我们的例子中 0.523 说明 doc3 与查询有较强的语义关联，0.789 说明 doc1 也有一定相关性但弱一些。

### `documents` — 文档原文

```python
results['documents']  # [['RAG 是检索增强生成的缩写', 'Chroma 是一个开源的向量数据库']]
```

这是添加时传入的原始文本内容。顺序与 `ids` 和 `distances` 一一对应——`results['documents'][0][0]` 对应 `results['ids'][0][0]` 和 `results['distances'][0][0]`。

### `metadatas` — 元数据

```python
results['metadatas']  # [None]
```

因为我们添加文档时没有传 `metadatas` 参数，所以这里是 `None`。如果传入了 metadata（后面第 2.3 节会详细讲），这里会返回对应的字典列表。

### `embeddings` — 向量数据

```python
results['embeddings']  # None
```

默认不返回 embedding 向量本身，因为它们通常很大（比如 384 维或 768 维浮点数数组）。如果你确实需要拿到向量（比如做下游分析），可以通过 `include` 参数显式请求：

```python
results = collection.query(
    query_texts=["什么是 RAG？"],
    n_results=2,
    include=["documents", "distances", "embeddings"]  # 显式请求 embeddings
)
print(results['embeddings'])  # 现在有值了：[[[0.123, -0.456, ...], [...]]]
```

### `include` 参数：控制返回哪些字段

这是 Chroma 最常用的参数之一。它让你精确控制返回值的体积，避免传输不必要的数据：

```python
# 只要 ID 和距离（最轻量）
results = collection.query(
    query_texts=["test"],
    n_results=5,
    include=["ids", "distances"]
)

# 只要文档内容和元数据（用于展示给用户）
results = collection.query(
    query_texts=["test"],
    n_results=5,
    include=["documents", "metadatas"]
)

# 全都要（调试时用）
results = collection.query(
    query_texts=["test"],
    n_results=5,
    include=["ids", "documents", "embeddings", "metadatas", "distances"]
)
```

可选项包括：`["ids", "documents", "embeddings", "metadatas", "distances"]`。默认情况下 `include=["metadatas", "documents", "distances"]`。

---

## `persist` 参数：数据到底存哪了？

你可能注意到上面的代码中，当我们创建 Client 时没有指定任何路径：

```python
client = chromadb.Client()  # 内存模式！
```

这意味着**所有数据都存在内存里**——程序退出后数据就丢了。这对于快速实验很方便，但对于实际应用显然不行。

要让数据持久化到磁盘，只需加一个参数：

```python
import chromadb

# 持久化模式：数据写入 ./chroma_db 目录
client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory="./chroma_db",
        is_persistent=True
    )
)

collection = client.get_or_create_collection(name="my_collection")
collection.add(
    documents=["这条数据会被保存到磁盘"],
    ids=["persistent_doc"]
)
```

程序结束后，去当前目录下看一眼：

```
./chroma_db/
├── chroma.sqlite3           # SQLite 数据库（存储 id、metadata、collection 信息）
├── xxxxxxxx-xxxx-...-blob   # Blob 文件（存储实际的向量数据和文档文本）
└── ...                      # 其他内部文件
```

**关键行为说明**：

1. **首次创建**：如果 `persist_directory` 指定的目录不存在，Chroma 会自动创建
2. **重复启动**：如果目录已存在且里面有数据，Chroma 会加载已有数据（热启动）
3. **WAL 机制**：Chroma 使用 Write-Ahead Logging 保证崩溃安全——即使程序异常退出，已提交的数据也不会丢失
4. **跨进程安全**：基于 SQLite WAL 模式，支持多进程并发读，但写操作会串行化

### 冷启动 vs 热启动的性能差异

```python
# 第一次运行（冷启动）：需要创建索引文件，较慢
client = chromadb.Client(settings=chromadb.Settings(persist_directory="./db"))
collection = client.create_collection(name="cold_start")
# ... 添加 10000 条数据 ...

# 第二次运行（热启动）：直接加载已有索引，快很多
client = chromadb.Client(settings=chromadb.Settings(persist_directory="./db"))
collection = client.get_collection(name="cold_start")  # 秒级加载
```

典型性能参考（10K 条 384 维向量）：
- 冷启动 + 首次写入：约 2~5 秒
- 热启动（从磁盘加载）：约 0.5~1 秒
- 单次查询：< 10ms

---

## 完整可运行的示例：带 Metadata 的增删查

让我们把前面学的知识整合成一个更完整的示例，涵盖 CRUD 全流程：

```python
import chromadb
import time

# ====== 初始化 ======
client = chromadb.Client(settings=chromadb.Settings(
    is_persistent=True,
    persist_directory="./my_chroma_demo"
))

# ====== Create / Get Collection ======
collection = client.get_or_create_collection(
    name="tech_docs",
    metadata={"description": "技术文档集合"}
)
print(f"Collection 名称: {collection.name}")
print(f"当前文档数: {collection.count()}")

# ====== Add（带 Metadata）======
collection.add(
    documents=[
        "Python 是一种解释型、通用的高级编程语言",
        "PyTorch 是 Facebook AI Research 开发的深度学习框架",
        "Chroma 是一个开源的嵌入式向量数据库"
    ],
    ids=["py_intro", "pt_intro", "ch_intro"],
    metadatas=[
        {"category": "language", "version": "3.11"},
        {"category": "framework", "company": "Meta"},
        {"category": "database", "license": "Apache-2.0"}
    ]
)
print(f"\n添加后文档数: {collection.count()}")

# ====== Query（带 Where 过滤）======
print("\n--- 查询 1: 纯语义搜索 ---")
results = collection.query(
    query_texts=["深度学习用什么框架？"],
    n_results=2,
    include=["ids", "documents", "distances", "metadatas"]
)
for i, doc_id in enumerate(results['ids'][0]):
    print(f"  [{i}] ID={doc_id}, 距离={results['distances'][0][i]:.4f}")
    print(f"      内容: {results['documents'][0][i]}")
    print(f"      分类: {results['metadatas'][0][i]}")

print("\n--- 查询 2: 语义搜索 + Metadata 过滤 ---")
results = collection.query(
    query_texts=["编程语言有哪些？"],
    where={"category": "language"},  # 只找 category=language 的文档
    n_results=3,
    include=["ids", "documents", "metadatas"]
)
for i, doc_id in enumerate(results['ids'][0]):
    print(f"  [{i}] {results['documents'][0][i]}")

# ====== Get（通过 ID 精确获取）======
print("\n--- Get: 通过 ID 获取 ---")
get_result = collection.get(
    ids=["pt_intro", "ch_intro"],
    include=["documents", "metadatas"]
)
for i, doc_id in enumerate(get_result['ids']):
    print(f"  {doc_id}: {get_result['documents'][i]} | {get_result['metadatas'][i]}")

# ====== Update（更新文档内容和 Metadata）======
collection.update(
    ids=["py_intro"],
    documents=["Python 是一种广泛使用的动态类型编程语言，由 Guido van Rossum 于 1991 年创建"],
    metadatas=[{"category": "language", "version": "3.12", "creator": "Guido"}]
)
print("\n更新后重新查询:")
updated = collection.get(ids=["py_intro"], include=["documents", "metadatas"])
print(f"  {updated['documents'][0]}")
print(f"  Metadata: {updated['metadatas'][0]}")

# ====== Delete ======
collection.delete(ids=["ch_intro"])
print(f"\n删除后文档数: {collection.count()}")

# ====== 验证持久化 ======
print("\n✅ 数据已持久化到 ./my_chroma_demo 目录")
print("下次启动 client 时数据仍然存在！")
```

运行这段代码，你会看到完整的生命周期输出：

```
Collection 名称: tech_docs
当前文档数: 0

添加后文档数: 3

--- 查询 1: 纯语义搜索 ---
  [0] ID=pt_intro, 距离=0.4872
      内容: PyTorch 是 Facebook AI Research 开发的深度学习框架
      分类: {'category': 'framework', 'company': 'Meta'}
  [1] ID=py_intro, 距离=0.8123
      内容: Python 是一种解释型、通用的高级编程语言
      分类: {'category': 'language', 'version': '3.11'}

--- 查询 2: 语义搜索 + Metadata 过滤 ---
  [0] Python 是一种解释型、通用的高级编程语言

--- Get: 通过 ID 获取 ---
  pt_intro: PyTorch 是 Facebook AI Research 开发的深度学习框架 | {'category': 'framework', 'company': 'Meta'}
  ch_intro: Chroma 是一个开源的嵌入式向量数据库

更新后重新查询:
  Python 是一种广泛使用的动态类型编程语言，由 Guido van Rossum 于 1991 年创建
  Metadata: {'category': 'language', 'version': '3.12', 'creator': 'Guido'}

删除后文档数: 2

✅ 数据已持久化到 ./my_chroma_demo 目录
下次启动 client 时数据仍然存在！
```

---

## 常见陷阱与排查

### 陷阱 1：忘记设置 `persist_directory`，数据神秘消失

**症状**：程序跑得好好的，重启后发现 Collection 是空的。

**原因**：使用了默认的内存模式 `chromadb.Client()`，没有传入 `Settings(persist_directory=...)`。

**修复**：

```python
# ❌ 错误：数据只在内存里
client = chromadb.Client()

# ✅ 正确：数据持久化到磁盘
client = chromadb.Client(settings=chromadb.Settings(
    is_persistent=True,
    persist_directory="./my_data"
))
```

### 陷阱 2：ID 重复导致报错

**症状**：`collection.add()` 抛出 `ValueError: Duplicate id detected`。

**原因**：Chroma 默认不允许重复 ID。如果你尝试用已有的 ID 再次 add，会报错。

**解决方案**：

```python
# 方案 A：用 upsert 替代 add（存在则更新，不存在则插入）
collection.upsert(
    documents=["更新后的内容"],
    ids=["existing_id"]  # 如果 existing_id 已存在，自动更新
)

# 方案 B：先 delete 再 add
collection.delete(ids=["existing_id"])
collection.add(documents=["新内容"], ids=["existing_id"])
```

### 陷阱 3：Distance Metric 不匹配导致结果奇怪

**症状**：同样的数据，换了台机器或者重建 Collection 后，查询结果的排序变了。

**原因**：Collection 的 distance metric 在创建后不可更改。如果你之前用的是 `cosine`，后来不小心用了 `l2` 创建同名 Collection（先删旧的再建的），排序逻辑就完全不同了。

**检查方法**：

```python
print(collection.metadata)  # 查看 Collection 的元信息，包含 distance metric
```

### 陷阱 4：中文文本的 encoding 问题

**症状**：添加中文文档后查询正常，但 `documents` 字段显示乱码。

**原因**：Chroma 内部使用 UTF-8 编码，但如果你的终端/日志系统不是 UTF-8，显示可能有问题。**数据本身没有损坏**，只是展示层的问题。

**验证方法**：

```python
results = collection.query(query_texts=["测试"], n_results=1)
doc = results['documents'][0][0]
assert isinstance(doc, str)  # 确认是正常的 Python string
print(doc.encode('utf-8').decode('utf-8'))  # 强制 UTF-8 输出
```

---

## 本章小结

到这里，你已经完成了 Chroma 的"第一次握手"：

| 操作 | API | 类比 SQL |
|------|-----|----------|
| 连接数据库 | `chromadb.Client(settings=...)` | `psql -h host dbname` |
| 创建表 | `client.create_collection(name=...)` | `CREATE TABLE` |
| 插入数据 | `collection.add(documents=..., ids=...)` | `INSERT INTO` |
| 查询 | `collection.query(query_texts=..., n_results=...)` | `SELECT ... ORDER BY similarity()` |
| 精确查找 | `collection.get(ids=...)` | `SELECT * WHERE id=?` |
| 更新 | `collection.update(ids=..., documents=...)` | `UPDATE SET ... WHERE id=?` |
| 删除 | `collection.delete(ids=...)` | `DELETE WHERE id=?` |

**核心要点回顾**：

1. **三种安装方式**：pip（开发）、Docker（生产/团队）、源码（贡献者）
2. **`include` 参数控制返回字段**：默认返回 `metadatas/documents/distances`，可以按需精简
3. **`persist_directory` 决定数据生命**：不设则内存模式（重启丢失），设了则持久化到磁盘
4. **ID 唯一性约束**：重复 ID 会报错，用 `upsert` 解决"存在则更新"需求
5. **distance metric 不可变**：创建 Collection 时选定，之后不能改

下一章我们将深入 CRUD 的每一个操作细节——批量添加的性能优化、分页获取、Metadata 过滤的高级用法，以及如何管理多个 Collection。
