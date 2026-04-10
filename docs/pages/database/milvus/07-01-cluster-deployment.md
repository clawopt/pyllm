# 7.1 集群部署方案

> **从 Docker Compose 到 Kubernetes——Milvus 的部署方案覆盖所有规模**

---

## 这一节在讲什么？

前面六章我们一直在用 Milvus Lite 或 Milvus Standalone 做实验——一个 Python 包或一个 Docker 容器就能跑。但当你需要把 Milvus 部署到生产环境时，就需要考虑集群部署、资源规划、依赖组件配置等问题。这一节我们要聊 Milvus 的三种部署方案（Standalone / Cluster / Zilliz Cloud），以及每种方案的适用场景和配置要点。

---

## Milvus Standalone：单容器部署

Milvus Standalone 把所有组件（协调器、工作节点、存储）打包在一个 Docker 容器里，用 Docker Compose 一键启动。适合中小规模生产（< 1000 万向量）和开发测试环境。

```yaml
# docker-compose.yml — Milvus Standalone
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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

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

# 查看 Milvus 日志
docker logs milvus-standalone

# 停止
docker compose down
```

---

## Milvus Cluster：分布式集群部署

当你需要处理亿级数据、高并发搜索时，就需要 Milvus Cluster——每个组件独立部署，可以独立扩缩容。

### Docker Compose 集群部署

Milvus 官方提供了完整的 Docker Compose 集群模板，包含所有组件的独立容器：

```yaml
# docker-compose.yml — Milvus Cluster（简化版）
version: '3.5'

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379
    volumes:
      - etcd_data:/etcd

  minio:
    image: minio/minio:latest
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data
    volumes:
      - minio_data:/minio_data

  # 协调器
  rootcoord:
    image: milvusdb/milvus:v2.5-latest
    command: ["milvus", "run", "rootcoord"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000

  querycoord:
    image: milvusdb/milvus:v2.5-latest
    command: ["milvus", "run", "querycoord"]
    environment:
      ETCD_ENDPOINTS: etcd:2379

  datacoord:
    image: milvusdb/milvus:v2.5-latest
    command: ["milvus", "run", "datacoord"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000

  indexcoord:
    image: milvusdb/milvus:v2.5-latest
    command: ["milvus", "run", "indexcoord"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000

  # 工作节点
  querynode:
    image: milvusdb/milvus:v2.5-latest
    command: ["milvus", "run", "querynode"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    deploy:
      resources:
        limits:
          memory: 16G

  datanode:
    image: milvusdb/milvus:v2.5-latest
    command: ["milvus", "run", "datanode"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000

  indexnode:
    image: milvusdb/milvus:v2.5-latest
    command: ["milvus", "run", "indexnode"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000

  # 代理（API 入口）
  proxy:
    image: milvusdb/milvus:v2.5-latest
    command: ["milvus", "run", "proxy"]
    ports:
      - "19530:19530"
    environment:
      ETCD_ENDPOINTS: etcd:2379

volumes:
  etcd_data:
  minio_data:
```

### Helm + Kubernetes 部署

对于生产环境，Kubernetes + Helm 是更推荐的方式——支持自动扩缩容、滚动升级、资源限制：

```bash
# 添加 Milvus Helm 仓库
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update

# 安装 Milvus Cluster
helm install milvus milvus/milvus \
  --set cluster.enabled=true \
  --set queryNode.replicas=3 \
  --set queryNode.resources.limits.memory=16Gi \
  --set dataNode.replicas=2 \
  --set indexNode.replicas=2 \
  --set etcd.replicaCount=3 \
  --set minio.mode=standalone

# 查看状态
kubectl get pods -l app.kubernetes.io/name=milvus

# 扩容 QueryNode
kubectl scale deployment milvus-querynode --replicas=5
```

---

## 依赖组件配置

Milvus Cluster 依赖三个外部组件，它们的配置直接影响系统的稳定性和性能：

### etcd：元数据存储

etcd 存储 Milvus 的所有元数据——Collection 的 Schema、Segment 的位置、索引的元数据。如果 etcd 挂了，Milvus 就无法找到任何数据。

- 生产环境建议 3 节点集群，保证高可用
- 定期备份 etcd 数据（`etcdctl snapshot save`）
- 注意 etcd 的数据库大小限制（默认 2GB），大量 Collection/Partition 可能导致 etcd 空间不足

### MinIO / S3：对象存储

MinIO 或 S3 存储实际的向量数据文件和索引文件——这是 Milvus 的"仓库"。

- 生产环境建议使用 S3（AWS）或 OSS（阿里云），而不是自建 MinIO
- 配置生命周期策略，定期清理过期的 Segment 文件
- 注意 S3 的请求频率限制，高吞吐写入场景可能需要配置 S3 限流

### Pulsar / Kafka：消息队列

消息队列缓冲写入请求，实现组件间的异步通信。

- 小规模场景可以用 Kafka（更轻量）
- 大规模场景建议用 Pulsar（更好的多租户支持）
- 注意消息队列的积压监控——积压过多说明 DataNode 消费速度跟不上

---

## 资源规划

| 组件 | CPU | 内存 | 磁盘 | 扩缩容策略 |
|------|-----|------|------|----------|
| QueryNode | 4~8 核 | 索引大小 × 1.5 | 适中 | 搜索延迟高时扩容 |
| DataNode | 4~8 核 | 8~16 GB | 高 IOPS | 写入积压时扩容 |
| IndexNode | 8~16 核 | 16~32 GB | 高 IOPS | 索引构建慢时扩容 |
| Coordinator | 2~4 核 | 4~8 GB | 低 | 通常不需要扩容 |
| etcd | 2 核 | 4~8 GB | SSD | 3 节点固定 |
| MinIO | 2~4 核 | 4~8 GB | 大容量 | 存储空间不足时扩容 |

---

## Zilliz Cloud：免运维的全托管方案

如果你不想自己管理 etcd、MinIO、Pulsar 这些基础设施，Zilliz Cloud 是最省心的选择：

```python
from pymilvus import MilvusClient

# 连接 Zilliz Cloud——不需要管基础设施
client = MilvusClient(
    uri="https://in03-xxx.api.zillizcloud.com",
    token="your-api-key"
)

# 后续操作跟自建 Milvus 完全一样
client.create_collection(collection_name="documents", dimension=768, metric_type="COSINE")
```

Zilliz Cloud 的优势：免运维、自动备份、自动扩缩容、SLA 保障。劣势：成本比自建高、数据在第三方平台上、定制化能力有限。

---

## 常见误区：Standalone 不能用于生产

Milvus Standalone 可以用于中小规模生产——它的所有组件都在一个容器里，但功能是完整的。只要你的数据量不超过单机承载能力（约 1000 万向量），Standalone 完全够用。不需要为了"看起来更专业"而部署 Cluster——Cluster 的运维成本远高于 Standalone。

---

## 小结

这一节我们覆盖了 Milvus 的三种部署方案：Standalone 适合中小规模生产，Cluster 适合大规模生产，Zilliz Cloud 适合不想运维的团队。资源规划的核心是 QueryNode 的内存要能装下索引，DataNode 和 IndexNode 的 CPU 要够用。下一节我们聊性能监控与调优。
