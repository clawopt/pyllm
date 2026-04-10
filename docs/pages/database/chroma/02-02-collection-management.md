# 2.2 Collection 管理

> **Collection 是 Chroma 的逻辑容器——理解它的生命周期、元信息和隔离策略**

---

## 什么是 Collection？

如果你有 SQL 背景，可以把 Collection 理解为**一张表（Table）**；如果你用过 MongoDB，它类似于一个 **Collection**（名字的由来）；如果你用过 Elasticsearch，它对应一个 **Index**。

Collection 是 Chroma 中组织文档的逻辑单元。每个 Collection 有以下固有属性：

| 属性 | 说明 | 可变性 |
|------|------|--------|
| `name` | Collection 的唯一标识名 | 创建后不可更改 |
| `distance metric` | 向量距离度量方式（cosine/l2/ip） | 创建后**不可更改** |
| `metadata` | 用户自定义的描述信息 | 可更新 |

**为什么 distance metric 不可变？** 因为 HNSW 索引的结构依赖于距离度量。一旦索引构建完成，切换度量意味着需要重建整个索引——Chroma 选择在 API 层面直接禁止这个操作，避免用户误操作导致数据不一致。

---

## 创建 Collection

### 基础创建

```python
import chromadb

client = chromadb.Client()

# 最简创建
collection = client.create_collection(name="documents")
```

### 指定 Distance Metric

这是创建 Collection 时最重要的决策：

```python
# 余弦相似度（默认，推荐用于大多数 NLP 场景）
col_cosine = client.create_collection(
    name="nlp_docs",
    metadata={"hnsw:space": "cosine"}
)

# 欧氏距离（适合已经归一化的向量）
col_l2 = client.create_collection(
    name="image_features",
    metadata={"hnsw:space": "l2"}
)

# 内积（最快，但要求输入向量已归一化）
col_ip = client.create_collection(
    name "fast_search",
    metadata={"hnsw:space": "ip"}  # 注意：实际代码中这里不应该有语法错误
)
```

**三种度量的选择指南**：

```
你的 embedding 模型输出什么？
│
├─ 输出 L2 归一化的单位向量（如 sentence-transformers 大多数模型）
│   → 用 cosine 或 ip 都可以（结果等价）
│   → 推荐 cosine（语义更直观：0=相同, 1=正交, 2=相反）
│
├─ 输出未归一化的向量（如某些自定义模型）
│   → 用 l2（欧氏距离对模长敏感，能区分"方向相同但强度不同"的情况）
│
└─ 不确定？
    → 用 cosine（最安全的选择，兼容性最好）
```

### get_or_create_collection：幂等的集合获取

在实际开发中，你经常需要确保某个 Collection 存在——如果不存在就创建，如果存在就直接用。`get_or_create_collection()` 就是为此设计的：

```python
# 第一次调用：不存在则创建
collection = client.get_or_create_collection(name="user_data")
print(collection.count())  # 0

# 第二次调用：已存在则直接返回（不会清空数据！）
collection2 = client.get_or_create_collection(name="user_data")
print(collection2.count())  # 仍然是 0（同一个对象）

# ⚠️ 注意：这个方法不会检查或修改 distance metric
# 如果之前用 cosine 创建的，后来想改成 l2——做不到！必须先删再建
```

**与 create_collection 的区别**：

| 方法 | 集合已存在时 | 集合不存在时 |
|------|-------------|-------------|
| `create_collection()` | 抛出 `ValueError: Collection already exists` | 创建新集合 |
| `get_or_create_collection()` | 返回已有集合 | 创建新集合 |

---

## 查看和列举 Collection

### 列举所有 Collection

```python
collections = client.list_collections()
for col in collections:
    print(f"名称: {col.name}, 文档数: {col.count()}")
```

### 获取单个 Collection

```python
# 通过名称获取
collection = client.get_collection(name="documents")

# 获取不存在的 Collection 会报错
try:
    col = client.get_collection(name="nonexistent")
except Exception as e:
    print(f"❌ 错误: {e}")
```

### 查看 Collection 元信息

```python
collection = client.get_collection(name="my_collection")

# 名称
print(f"名称: {collection.name}")

# 文档数量
print(f"文档数: {collection.count()}")

# 距离度量（从 metadata 中提取）
print(f"距离度量: {collection.metadata}")

# 完整的 metadata 字典（包含用户自定义信息）
if collection.metadata:
    for key, value in collection.metadata.items():
        print(f"  {key}: {value}")
```

典型输出：

```
名称: my_collection
文档数: 1523
距离度量: {'hnsw:space': 'cosine', 'description': '用户知识库'}
  hnsw:space: cosine
  description: 用户知识库
```

### 获取 Collection 的向量维度

Chroma 没有直接暴露维度字段，但你可以通过查询一条数据的 embedding 来推断：

```python
def get_embedding_dimension(collection):
    """获取 Collection 中向量的维度"""
    result = collection.get(limit=1, include=["embeddings"])
    if result['embeddings'] and result['embeddings'][0]:
        return len(result['embeddings'][0][0])
    return None  # Collection 为空

dim = get_embedding_dimension(collection)
print(f"向量维度: {dim}")  # 例如 384, 768, 1536 等
```

---

## 删除 Collection

```python
# 删除指定 Collection（及其所有数据和索引！）
client.delete_collection(name="old_collection")

# 尝试删除不存在的 Collection
try:
    client.delete_collection(name="nonexistent")
except Exception as e:
    print(f"❌ 错误: {e}")
```

**⚠️ 警告**：删除 Collection 是**不可逆操作**——所有数据、索引、metadata 全部清除。执行前请确认。

安全删除模式：

```python
def safe_delete_collection(client, collection_name):
    """安全删除 Collection（先确认）"""
    try:
        col = client.get_collection(collection_name)
        count = col.count()
        print(f"⚠️ 即将删除 '{collection_name}' (包含 {count} 条文档)")
        print("确认删除？(生产环境建议加二次确认机制)")
        client.delete_collection(collection_name)
        print(f"✅ '{collection_name}' 已删除")
        return True
    except Exception as e:
        print(f"❌ 删除失败: {e}")
        return False
```

---

## 多 Collection 场景：隔离策略

为什么你需要多个 Collection？在实际项目中，不同类型的数据通常有不同的特征：

| 使用场景 | 推荐方案 | 原因 |
|----------|----------|------|
| 用户文档 + 系统文档 | 两个 Collection | 距离度量可能不同，搜索范围需隔离 |
| 多租户 SaaS | 每个租户一个 Collection | 数据隔离 + 权限控制简单 |
| 不同语言的内容 | 按语言分 Collection | 避免跨语言干扰检索质量 |
| 不同 embedding 模型 | 按模型分 Collection | 维度可能不同，无法混存 |
| 开发/测试/生产环境 | 用不同的 persist_directory 或 Collection 名前缀 | 环境隔离 |

### 示例：多租户知识库

```python
import chromadb

client = chromadb.Client(settings=chromadb.Settings(
    is_persistent=True,
    persist_directory="./multi_tenant_db"
))

class TenantKnowledgeBase:
    """多租户知识库管理器"""

    def __init__(self, client):
        self.client = client
        self._collections = {}

    def get_tenant_collection(self, tenant_id):
        """获取或创建租户专属 Collection"""
        if tenant_id not in self._collections:
            collection_name = f"tenant_{tenant_id}_kb"
            self._collections[tenant_id] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "tenant_id": tenant_id,
                    "created_at": "2025-01-15"
                }
            )
        return self._collections[tenant_id]

    def add_document(self, tenant_id, doc_id, text, metadata=None):
        """为指定租户添加文档"""
        col = self.get_tenant_collection(tenant_id)
        meta = metadata or {}
        meta["tenant_id"] = tenant_id
        col.add(documents=[text], ids=[doc_id], metadatas=[meta])

    def search(self, tenant_id, query, n_results=5):
        """在指定租户的范围内搜索"""
        col = self.get_tenant_collection(tenant_id)
        results = col.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def list_tenants(self):
        """列出所有有数据的租户"""
        collections = self.client.list_collections()
        tenants = []
        for col in collections:
            if col.name.startswith("tenant_") and col.count() > 0:
                tid = col.metadata.get("tenant_id", "unknown")
                tenants.append({
                    "tenant_id": tid,
                    "collection": col.name,
                    "document_count": col.count()
                })
        return tenants


# ====== 使用示例 ======
kb = TenantKnowledgeBase(client)

# 租户 A 添加文档
kb.add_document("tenant_a", "a_001", "租户A的产品手册v1", {"category": "manual"})
kb.add_document("tenant_a", "a_002", "租户A的API文档", {"category": "api"})

# 租户 B 添加文档
kb.add_document("tenant_b", "b_001", "租户B的销售策略", {"category": "sales"})
kb.add_document("tenant_b", "b_002", "租户B的技术架构", {"category": "tech"})

# 各自搜索（天然隔离！）
print("\n🔍 租户 A 搜索 '产品':")
results_a = kb.search("tenant_a", "产品")
for i, doc in enumerate(results_a['documents'][0]):
    print(f"  [{i}] {doc[:60]}... (距离: {results_a['distances'][0][i]:.4f})")

print("\n🔍 租户 B 搜索 '技术':")
results_b = kb.search("tenant_b", "技术")
for i, doc in enumerate(results_b['documents'][0]):
    print(f"  [{i}] {doc[:60]}... (距离: {results_b['distances'][0][i]:.4f})")

# 列出所有活跃租户
print("\n📋 活跃租户:")
for t in kb.list_tenants():
    print(f"  {t['tenant_id']}: {t['document_count']} 条文档 ({t['collection']})")
```

输出：

```
🔍 租户 A 搜索 '产品':
  [0] 租户A的产品手册v1... (距离: 0.3456)

🔍 租户 B 搜索 '技术':
  [0] 租户B的技术架构... (距离: 0.4123)

📋 活跃租户:
  tenant_a: 2 条文档 (tenant_tenant_a_kb)
  tenant_b: 2 条文档 (tenant_tenant_b_kb)
```

### 示例：按语言隔离的多语言知识库

```python
LANGUAGES = {
    "zh": {"name": "中文知识库", "model_hint": "paraphrase-multilingual-MiniLM-L12-v2"},
    "en": {"name": "English Knowledge Base", "model_hint": "all-MiniLM-L6-v2"},
    "ja": {"name": "日本語ナレッジベース", "model_hint": "paraphrase-multilingual-MiniLM-L12-v2"},
}

def get_lang_collection(client, lang_code):
    """获取语言特定的 Collection"""
    if lang_code not in LANGUAGES:
        raise ValueError(f"不支持的语言: {lang_code}")
    config = LANGUAGES[lang_code]
    return client.get_or_create_collection(
        name=f"kb_{lang_code}",
        metadata={
            "hnsw:space": "cosine",
            "language": lang_code,
            "display_name": config["name"]
        }
    )

# 使用
zh_col = get_lang_collection(client, "zh")
en_col = get_lang_collection(client, "en")

zh_col.add(documents=["人工智能是计算机科学的一个分支"], ids=["zh_001"])
en_col.add(documents=["AI is a branch of computer science"], ids=["en_001"])

# 中文搜索只命中中文文档
r = zh_col.query(query_texts=["机器学习"], n_results=2)
print(r['documents'])  # 只包含中文结果
```

---

## Collection 命名规范与最佳实践

### 命名规则

- 只允许小写字母、数字、下划线
- 必须以字母开头
- 长度建议 3~63 个字符
- 不能包含特殊字符（空格、连字符、点号等）

```python
# ✅ 合法的名称
"users"
"order_history_v2"
"tenant_42_docs"

# ❌ 非法的名称
"My Collection"       # 包含空格和大写
"data-set"            # 包含连字符
"123data"             # 以数字开头
"a.b.c"               # 包含点号
```

### 推荐命名模式

```python
# 模式 1：功能前缀
"kb_products"         # 产品知识库
"kb_faq"              # FAQ 库
"mem_conversations"   # 对话记忆

# 模式 2：环境前缀（同一 persist_directory 下）
"dev_user_docs"
"staging_user_docs"
"prod_user_docs"

# 模式 3：带版本号
"embedding_v1"        # 第一版 embedding
"embedding_v2"        # 升级后的新 embedding（新的 Collection）
```

---

## 常见陷阱

### 陷阱 1：Distance Metric 选错导致排序异常

**症状**：同样的数据，新建 Collection 后查询结果的顺序完全变了。

**原因**：不小心用了不同的 `hnsw:space`。比如之前是 `cosine`，重新创建了同名 Collection 但设成了 `l2`。

**排查**：

```python
col = client.get_collection(name="my_collection")
print(col.metadata)  # 检查 hnsw:space 的值
```

### 陷阱 2：Collection 名字冲突

**症状**：`create_collection()` 报 `ValueError: Collection <name> already exists`。

**原因**：程序重启后重复调用 `create_collection()`。

**修复**：

```python
# 方案 A：始终用 get_or_create_collection
col = client.get_or_create_collection(name="my_col")

# 方案 B：先尝试 get，失败再 create
try:
    col = client.get_collection(name="my_col")
except Exception:
    col = client.create_collection(name="my_col")
```

### 陷阱 3：误删 Collection 导致数据丢失

**预防措施**：

```python
# 生产环境中不要随意使用 delete_collection
# 建议封装一层权限控制
class SafeCollectionManager:
    def __init__(self, client, allowed_prefixes=None):
        self.client = client
        self.allowed_prefixes = allowed_prefixes or []

    def delete_collection(self, name):
        if self.allowed_prefixes:
            if not any(name.startswith(p) for p in self.allowed_prefixes):
                raise PermissionError(f"不允许删除 Collection: {name}")
        confirm = input(f"确认删除 '{name}'? (yes/no): ")
        if confirm.lower() == "yes":
            self.client.delete_collection(name)
            print(f"✅ 已删除 {name}")
        else:
            print("❌ 操作取消")
```

### 陷阱 4：持久化模式下 Collection 的"幽灵数据"

**症状**：明明调了 `delete_collection()`，但下次启动 Client 时 Collection 又出现了。

**原因**：`delete_collection()` 在持久化模式下会从内存中移除并标记删除，但如果程序异常退出（没来得及 flush WAL），磁盘上的数据文件可能残留。

**解决**：正常退出程序让 Chroma 完成 checkpoint。如果遇到残留，手动清理 `.chroma` 目录下的相关文件（**谨慎操作**）。

---

## 本章小结

| 操作 | 方法 | 关键点 |
|------|------|--------|
| 创建 | `create_collection(name, metadata)` | metadata 中设置 `hnsw:space` 决定距离度量 |
| 幂等获取 | `get_or_create_collection(name)` | 存在则返回，不存在则创建 |
| 获取已有 | `get_collection(name)` | 不存在则抛异常 |
| 列举全部 | `list_collections()` | 返回 Collection 对象列表 |
| 删除 | `delete_collection(name)` | 不可逆！数据和索引全删 |
| 查看属性 | `.name / .count() / .metadata` | 维度需通过 embedding 推断 |

**核心要点**：

1. **Distance Metric 一旦选定不可更改**——这是设计决策，因为 HNSW 索引结构依赖它
2. **`get_or_create_collection` 是开发中最常用的方法**——避免重复创建错误
3. **多 Collection 是数据隔离的标准做法**——多租户、多语言、多环境各用独立 Collection
4. **命名遵循小写+下划线规范**——避免特殊字符导致的隐蔽 bug
5. **删除操作不可逆**——生产环境务必加确认机制

下一节我们将深入 Metadata 设计最佳实践——如何规划 metadata schema、RAG 场景中的典型字段、以及那些会让性能急剧下降的反模式。
