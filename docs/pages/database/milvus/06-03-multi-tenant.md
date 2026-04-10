# 6.3 多租户与权限隔离

> **Milvus 的 Partition 是多租户隔离的最佳方案——物理隔离比逻辑过滤快得多**

---

## 这一节在讲什么？

如果你的 RAG 系统需要服务多个租户（比如一个 SaaS 产品，每个客户有自己的文档库），数据隔离就是一个必须解决的问题——租户 A 不能搜到租户 B 的数据。在 pgvector 中，你只能用 WHERE 条件做逻辑隔离（`WHERE tenant_id = 'A'`）；在 Milvus 中，你可以用 Partition 做物理隔离——每个租户的数据在独立的分区中，搜索时只扫描自己的分区。这一节我们要聊 Milvus 多租户的三种方案、Partition Key 的用法，以及 RBAC 权限控制。

---

## 多租户的三种方案

### 方案1：字段过滤（最简单，性能最差）

所有租户的数据存在同一个 Collection 中，搜索时用 `tenant_id` 过滤：

```python
# 插入时带上 tenant_id
client.insert(
    collection_name="documents",
    data=[
        {"tenant_id": "tenant_A", "content": "doc1", "embedding": [0.1] * 768},
        {"tenant_id": "tenant_B", "content": "doc2", "embedding": [0.2] * 768},
    ]
)

# 搜索时按 tenant_id 过滤
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='tenant_id == "tenant_A"'  # 只搜索租户 A 的数据
)
```

优点：最简单，不需要额外配置。缺点：每次搜索都要扫描全量数据做过滤（即使有标量索引，性能也不如物理隔离），而且如果过滤条件写错了，租户 A 可能搜到租户 B 的数据——这是安全事故。

### 方案2：Partition 隔离（推荐）

每个租户一个 Partition，搜索时只搜索对应租户的分区：

```python
# 用 Partition Key 自动分区
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields=fields, num_partitions=256)
client.create_collection(collection_name="documents", schema=schema)

# 插入时不需要指定分区——Milvus 根据 tenant_id 自动路由
client.insert(
    collection_name="documents",
    data=[
        {"tenant_id": "tenant_A", "content": "doc1", "embedding": [0.1] * 768},
        {"tenant_id": "tenant_B", "content": "doc2", "embedding": [0.2] * 768},
    ]
)

# 搜索时按 tenant_id 过滤——自动路由到对应分区
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='tenant_id == "tenant_A"'  # 自动路由到 tenant_A 的分区
)
```

优点：物理隔离，搜索只扫描对应分区的数据，性能好；即使过滤条件写错了，也不会搜到其他租户的数据（因为其他分区的数据根本不在搜索范围内）。缺点：分区数量有限（建议 64~512），如果租户数量超过分区数，多个租户会共享一个分区。

### 方案3：Collection 隔离

每个租户一个 Collection：

```python
# 为每个租户创建独立的 Collection
for tenant_id in ["tenant_A", "tenant_B", "tenant_C"]:
    client.create_collection(
        collection_name=f"docs_{tenant_id}",
        dimension=768,
        metric_type="COSINE"
    )

# 搜索时只搜索对应租户的 Collection
results = client.search(
    collection_name="docs_tenant_A",
    data=[[0.1] * 768],
    limit=5
)
```

优点：完全隔离，每个租户的 Schema、索引、数据都独立。缺点：管理开销大——每个 Collection 都有元数据开销，如果租户数量成千上万，协调器会被压垮。只适合租户数量少（< 100）且每个租户数据量大的场景。

---

## 三种方案的对比

| 维度 | 字段过滤 | Partition 隔离 | Collection 隔离 |
|------|---------|--------------|----------------|
| 隔离级别 | 逻辑隔离 | 物理隔离 | 完全隔离 |
| 搜索性能 | 差（全量扫描+过滤） | 好（只扫描对应分区） | 最好（只扫描对应 Collection） |
| 数据安全 | 低（过滤条件可能写错） | 高（物理隔离） | 最高（完全隔离） |
| 管理开销 | 低 | 低 | 高 |
| 适合租户数 | 不限 | < 10000 | < 100 |
| 推荐程度 | 不推荐 | ✅ 推荐 | 特殊场景 |

---

## RBAC 权限控制（Milvus 2.5+）

Milvus 2.5+ 支持 RBAC（Role-Based Access Control）——你可以创建用户、角色，并为角色分配权限，控制谁能访问哪些 Collection：

```python
# 创建用户
client.create_user(user_name="tenant_a_user", password="secure_password")

# 创建角色
client.create_role(role_name="tenant_a_role")

# 为角色授予权限——只能访问 docs_tenant_A Collection
client.grant_privilege(
    role_name="tenant_a_role",
    object_type="Collection",
    privilege="Search",
    object_name="docs_tenant_A"
)

# 把角色分配给用户
client.grant_role(user_name="tenant_a_user", role_name="tenant_a_role")

# 用该用户连接
tenant_client = MilvusClient(
    uri="http://localhost:19530",
    token="tenant_a_user:secure_password"
)

# 只能搜索 docs_tenant_A
results = tenant_client.search(
    collection_name="docs_tenant_A",
    data=[[0.1] * 768],
    limit=5
)

# 搜索 docs_tenant_B 会被拒绝
# results = tenant_client.search(collection_name="docs_tenant_B", ...)  # ❌ 权限不足
```

RBAC 提供了应用层之外的安全保障——即使应用代码有 Bug，用户也无法访问没有权限的 Collection。

---

## 常见误区：用 Collection 隔离万级租户

有些团队为每个租户创建一个 Collection，当租户数量达到几千时，Milvus 的协调器开始变慢——因为每个 Collection 的元数据（Schema、索引信息、Segment 信息）都需要协调器管理。当 Collection 数量超过 10000 时，创建 Collection 的延迟可能从毫秒级飙升到秒级。正确的做法是用 Partition Key——所有租户共享一个 Collection，通过 Partition 实现物理隔离。

---

## 小结

这一节我们聊了 Milvus 多租户的三种方案：字段过滤（最简单但性能差）、Partition 隔离（推荐，物理隔离+性能好）、Collection 隔离（完全隔离但管理开销大）。对于大多数 SaaS 场景，Partition Key + RBAC 是最佳组合——Partition 实现数据隔离，RBAC 实现权限控制。下一节开始我们进入第 7 章，聊 Milvus 的生产部署与运维。
