# 2.3 向量 ID 与结果映射

> **FAISS 的搜索结果只有整数 ID——你需要自己把它映射回原始数据**

---

## 这一节在讲什么？

在 Milvus 中，搜索结果可以直接返回文档内容、来源、分类等字段——因为 Milvus 同时存储了向量数据和标量数据。FAISS 只存储向量，搜索结果只返回整数 ID 和距离值——你需要自己维护一个映射表，把 ID 映射回原始文档内容。这一节我们要聊 FAISS 的 ID 机制、IndexIDMap 的用法，以及如何设计映射方案。

---

## 默认 ID 行为

FAISS 默认用向量的添加顺序作为 ID——第一条向量 ID=0，第二条 ID=1，以此类推：

```python
import faiss
import numpy as np

d = 128
index = faiss.IndexFlatL2(d)

# 添加 3 批向量
index.add(np.random.rand(1000, d).astype('float32'))  # ID: 0~999
index.add(np.random.rand(2000, d).astype('float32'))  # ID: 1000~2999
index.add(np.random.rand(500, d).astype('float32'))   # ID: 3000~3499

query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
print(f"最近邻 ID: {indices}")  # 如 [[423 1892 3100 56 2789]]
```

这种顺序 ID 的问题在于——它跟你的业务数据没有关联。你还需要一个额外的数据结构来把 ID 映射回文档内容。

---

## IndexIDMap：自定义 ID

`IndexIDMap` 是 FAISS 提供的包装器——它允许你给向量指定任意 int64 ID：

```python
# 用 IndexIDMap 包装原始 Index
base_index = faiss.IndexFlatL2(d)
index = faiss.IndexIDMap(base_index)

# 添加向量时指定自定义 ID
vectors = np.random.rand(10000, d).astype('float32')
ids = np.array([1001, 1002, 1003, ..., 11000])  # 自定义 ID

index.add_with_ids(vectors, ids)

# 搜索结果中的 indices 就是自定义 ID
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
print(f"最近邻 ID: {indices}")  # 如 [[1042 7823 5291 2100 9438]]
```

`IndexIDMap` 的实现原理很简单——它内部维护一个从自定义 ID 到内部顺序 ID 的映射表。搜索时，它先在底层 Index 中搜索得到内部 ID，然后转换回自定义 ID。

### 常见误区：IndexFlatL2 不支持 add_with_ids

不是所有 Index 都直接支持 `add_with_ids()`——`IndexFlatL2` 和 `IndexHNSWFlat` 等基础 Index 不支持。你需要用 `IndexIDMap` 包装后才能使用自定义 ID：

```python
# ❌ 错误：IndexFlatL2 不支持 add_with_ids
index = faiss.IndexFlatL2(d)
index.add_with_ids(vectors, ids)  # 报错！

# ✅ 正确：用 IndexIDMap 包装
index = faiss.IndexIDMap(faiss.IndexFlatL2(d))
index.add_with_ids(vectors, ids)
```

---

## ID 映射方案：把搜索结果映射回原始数据

无论你用默认 ID 还是自定义 ID，FAISS 的搜索结果都只有 ID 和距离——你需要自己把 ID 映射回原始文档内容。常见的映射方案有两种：

### 方案1：Python 列表/字典映射

```python
# 用列表存储文档内容——ID 就是列表索引
documents = [
    {"content": "AI breakthrough in 2024", "source": "tech_news"},
    {"content": "Climate change report", "source": "science_daily"},
    # ... 10000 条
]

# 搜索后用 ID 映射回文档
distances, indices = index.search(query, k=5)
for i, idx in enumerate(indices[0]):
    doc = documents[idx]
    print(f"排名 {i+1}: {doc['content']} (距离: {distances[0][i]:.4f})")
```

### 方案2：数据库映射

```python
import psycopg2

# FAISS 搜索得到 ID → 用 ID 从 pgvector 查询完整数据
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

distances, indices = index.search(query, k=5)
for idx in indices[0]:
    cur.execute("SELECT content, source FROM documents WHERE id = %s", (int(idx),))
    row = cur.fetchone()
    print(f"ID={idx}: {row[0]} (来源: {row[1]})")
```

方案2 就是第 6 章要讲的 FAISS + pgvector 混合方案的基础——FAISS 负责高性能向量搜索，pgvector 负责存储和标量过滤。

---

## 常见误区：FAISS 的 ID 只支持 int64

FAISS 的 ID 只能是 int64 整数——不支持字符串 ID。如果你的业务 ID 是字符串（如 UUID、URL 哈希），你需要自己维护一个字符串到 int64 的映射表：

```python
# 字符串 ID → int64 映射
str_to_int = {}
int_to_str = {}

def get_or_create_int_id(str_id):
    if str_id not in str_to_int:
        int_id = len(str_to_int)
        str_to_int[str_id] = int_id
        int_to_str[int_id] = str_id
    return str_to_int[str_id]

# 使用
str_id = "doc_abc123"
int_id = get_or_create_int_id(str_id)
index.add_with_ids(vector, np.array([int_id]))

# 搜索后转换回字符串 ID
distances, indices = index.search(query, k=5)
for idx in indices[0]:
    original_id = int_to_str[int(idx)]
    print(f"原始 ID: {original_id}")
```

---

## 小结

这一节我们聊了 FAISS 的向量 ID 机制：默认用添加顺序作为 ID，`IndexIDMap` 支持自定义 int64 ID，搜索结果需要你自己映射回原始数据。FAISS 不支持字符串 ID——如果业务需要字符串 ID，需要自己维护映射表。下一节开始我们进入第 3 章，深入 FAISS 的基础索引类型。
