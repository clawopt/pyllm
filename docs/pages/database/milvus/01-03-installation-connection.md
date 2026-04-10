# 1.3 安装与连接：从 Milvus Lite 到分布式集群

> **Milvus 的安装比你想的简单——pip install 就能跑，Docker 一键起集群**

---

## 这一节在讲什么？

上一节我们聊了 Milvus 的架构，那些协调器、工作节点、存储层听起来很复杂，但好消息是——你不需要一开始就部署完整的分布式集群。Milvus 提供了三种部署模式，从最简单的 Milvus Lite（一个 Python 包）到生产级的分布式集群，你可以根据当前阶段选择合适的方案。这一节我们要把三种部署模式都过一遍，让你能快速上手，同时也了解生产部署需要什么。

---

## 三种部署模式

Milvus 提供了三种部署模式，覆盖从开发到生产的全部场景：

| 模式 | 适合场景 | 数据规模 | 运维成本 | 性能 |
|------|---------|---------|---------|------|
| Milvus Lite | 开发、测试、学习 | < 100 万 | 零 | 够用 |
| Milvus Standalone | 中小规模生产 | < 1000 万 | 低 | 好 |
| Milvus Cluster | 大规模生产 | 1000 万 ~ 10 亿+ | 高 | 极好 |

### Milvus Lite：pip install 就能跑

Milvus Lite 是 Milvus 的轻量版本，它把整个向量数据库打包成了一个 Python 包——不需要 Docker、不需要服务器、不需要任何外部依赖，`pip install` 就能用。它的数据存储在本地文件中，适合开发、测试和学习。

```bash
pip install pymilvus
```

```python
from pymilvus import MilvusClient

# 连接本地文件——零配置，数据存储在 ./milvus_demo.db
client = MilvusClient("./milvus_demo.db")

# 创建 Collection
client.create_collection(
    collection_name="demo",
    dimension=3,
    metric_type="COSINE"
)

# 插入数据
client.insert(
    collection_name="demo",
    data=[
        {"id": 1, "vector": [0.1, 0.2, 0.3], "text": "hello"},
        {"id": 2, "vector": [0.4, 0.5, 0.6], "text": "world"},
        {"id": 3, "vector": [0.7, 0.8, 0.9], "text": "milvus"},
    ]
)

# 搜索
results = client.search(
    collection_name="demo",
    data=[[0.1, 0.2, 0.3]],
    limit=2,
    output_fields=["text"]
)

for result in results[0]:
    print(f"ID: {result['id']}, Distance: {result['distance']:.4f}, Text: {result['entity']['text']}")
```

Milvus Lite 的核心优势是**零配置**——你不需要关心 Docker、不需要关心端口、不需要关心组件配置，直接写 Python 代码就行。它的局限性也很明显：不支持分布式、不支持多用户并发写入、数据量大了性能会下降。但对于学习和开发来说，它已经完全够用了。

### Milvus Standalone：Docker 一键启动

当你需要更真实的测试环境，或者小规模生产部署时，Milvus Standalone 是更好的选择——它把所有组件（协调器、工作节点、存储）打包在一个 Docker 容器里，用 Docker Compose 一键启动。

```yaml
# docker-compose.yml
version: '3.5'

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379

  minio:
    image: minio/minio:latest
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data

  milvus:
    image: milvusdb/milvus:v2.5-latest
    container_name: milvus-standalone
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - milvus_data:/var/lib/milvus
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

```bash
# 启动
docker compose up -d

# 查看状态
docker compose ps

# 停止
docker compose down
```

启动后，你就可以用 Python 连接了：

```python
from pymilvus import MilvusClient

# 连接 Docker 中的 Milvus
client = MilvusClient(uri="http://localhost:19530")

# 后续操作跟 Milvus Lite 完全一样
client.create_collection(
    collection_name="demo",
    dimension=768,
    metric_type="COSINE"
)
```

注意连接方式的区别：Milvus Lite 用本地文件路径（`"./milvus_demo.db"`），Milvus Standalone 用网络地址（`"http://localhost:19530"`）。19530 是 Milvus 的默认 gRPC 端口。

### Milvus Cluster：生产级分布式部署

当你需要处理亿级数据、高并发搜索时，就需要 Milvus Cluster——每个组件独立部署，可以独立扩缩容。Cluster 的部署方式有三种：

- **Docker Compose**：最简单的集群部署方式，适合测试和中小规模生产
- **Helm + Kubernetes**：生产级部署，支持自动扩缩容和滚动升级
- **Milvus Operator**：Kubernetes Operator，声明式管理 Milvus 集群生命周期

Cluster 的部署配置比较复杂，我们在第 7 章会详细讲解。这里你只需要知道：当你需要 Cluster 时，最简单的方式是 Docker Compose——Milvus 官方提供了完整的 `docker-compose.yml` 模板。

### Zilliz Cloud：免运维的全托管方案

如果你不想自己运维 Milvus，Zilliz Cloud 是最省心的选择——它是 Milvus 的商业公司 Zilliz 提供的全托管云服务，你只需要注册账号、创建集群，就能直接使用，不需要管 etcd、MinIO、Pulsar 这些基础设施。Zilliz Cloud 提供免费层（Free Tier），适合小规模使用和测试。

```python
from pymilvus import MilvusClient

# 连接 Zilliz Cloud
client = MilvusClient(
    uri="https://your-cluster.api.zillizcloud.com",
    token="your-api-key"
)

# 后续操作跟自建 Milvus 完全一样
```

---

## MilvusClient vs ORM API

pymilvus 提供了两套 API：MilvusClient（简化 API）和 ORM API（完整 API）。对于大多数场景，MilvusClient 已经足够了：

```python
# MilvusClient API（推荐新手使用）
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
client.create_collection(collection_name="demo", dimension=768)
client.insert(collection_name="demo", data=[...])
results = client.search(collection_name="demo", data=[...], limit=5)

# ORM API（更灵活，适合高级用户）
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect("default", host="localhost", port="19530")
fields = [
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields)
collection = Collection("demo", schema)
collection.insert([embeddings])
collection.create_index(field_name="embedding", index_params={...})
collection.load()
results = collection.search(data=[...], anns_field="embedding", param={...}, limit=5)
```

MilvusClient 的优势是简单——方法名直观、参数简洁、不需要手动管理连接。ORM API 的优势是灵活——可以精细控制 Schema、索引参数、搜索参数。本教程默认使用 MilvusClient API，在需要精细控制的场景（如自定义 Schema、多向量字段）时切换到 ORM API。

---

## 连接参数详解

无论使用哪种部署模式，连接 Milvus 都需要指定 `uri` 参数：

| 部署模式 | uri 格式 | 示例 |
|---------|---------|------|
| Milvus Lite | 本地文件路径 | `"./milvus.db"` |
| Milvus Standalone | `http://host:port` | `"http://localhost:19530"` |
| Milvus Cluster | `http://host:port` | `"http://milvus:19530"` |
| Zilliz Cloud | `https://xxx.api.zillizcloud.com` | `"https://in03-xxx.api.zillizcloud.com"` |

对于需要认证的场景（Zilliz Cloud 或配置了认证的自建 Milvus），还需要 `token` 参数：

```python
# Zilliz Cloud 认证
client = MilvusClient(
    uri="https://in03-xxx.api.zillizcloud.com",
    token="your-api-key"
)

# 自建 Milvus 认证（如果配置了认证）
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
```

---

## 常见误区：Milvus Lite 能用于生产吗

Milvus Lite 的官方定位是"开发、测试和学习"——它把数据存储在本地文件中，不支持并发写入、不支持分布式、没有高可用保障。如果你把它用于生产环境，一旦服务器重启或文件损坏，数据就可能丢失。Milvus Lite 适合用在 Jupyter Notebook 里做实验、在单元测试中做 Mock、在本地开发时做调试——但不要把它当作生产数据库来用。

另一个常见误区是**在 Milvus Lite 和 Milvus Standalone 之间频繁切换**——有些同学在本地用 Milvus Lite 开发，部署时切换到 Standalone，结果发现行为不一致。这是因为 Milvus Lite 的某些功能（如一致性级别、Compaction）跟 Standalone/Cluster 有差异。建议在开发阶段就用 Docker Standalone，保持开发环境和生产环境的一致性。

---

## 小结

这一节我们覆盖了 Milvus 的三种部署模式：Milvus Lite 适合学习和开发（pip install 即可），Milvus Standalone 适合中小规模生产（Docker 一键启动），Milvus Cluster 适合大规模生产（多节点分布式），Zilliz Cloud 适合不想运维的团队。连接方式也很简单——Milvus Lite 用文件路径，其他模式用网络地址。下一节开始我们进入第 2 章，正式学习 Milvus 的核心概念——Collection、Schema、字段类型和距离度量。
