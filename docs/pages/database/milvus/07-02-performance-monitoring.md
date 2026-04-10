# 7.2 性能监控与调优

> **监控不是可选项——是生产环境的必需品，没有监控就是盲人摸象**

---

## 这一节在讲什么？

上一节我们把 Milvus 部署到了生产环境，但"能跑"和"跑得快"是两回事。当搜索延迟突然飙升、写入吞吐突然下降、或者内存持续增长时，你需要一套监控体系来发现问题、定位原因、解决问题。这一节我们要聊 Milvus 的监控工具、关键指标、常见性能问题的排查方法，以及参数调优。

---

## Milvus 监控体系

Milvus 的监控基于 Prometheus + Grafana——Milvus 内置了 Prometheus 指标导出器，你只需要部署 Prometheus 来采集指标，然后用 Grafana 可视化。

```yaml
# docker-compose.yml — 追加监控服务
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin

  # Milvus 已经内置了 metrics 端点（端口 9091）
  # Prometheus 从 http://milvus:9091/metrics 采集指标
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: milvus
    static_configs:
      - targets: ['milvus:9091']  # Milvus 的 metrics 端点
```

Milvus 官方提供了 Grafana Dashboard 模板（ID: 17535），导入后可以直接看到所有关键指标的可视化面板。

---

## 关键监控指标

### 搜索性能指标

| 指标 | 含义 | 告警阈值 |
|------|------|---------|
| `milvus_proxy_search_vectors_count` | 搜索请求向量数 | - |
| `milvus_proxy_sq_latency_sum` | 搜索延迟总和 | P99 > 100ms |
| `milvus_proxy_sq_latency_count` | 搜索请求总数 | - |
| `milvus_querynode_sq_latency_sum` | QueryNode 搜索延迟 | P99 > 50ms |

### 写入性能指标

| 指标 | 含义 | 告警阈值 |
|------|------|---------|
| `milvus_proxy_insert_vectors_count` | 插入向量数 | - |
| `milvus_proxy_insert_latency_sum` | 插入延迟总和 | P99 > 500ms |
| `milvus_datanode_msgstream_consume_rate` | DataNode 消费速率 | 消费速率 < 生产速率 |

### 资源使用指标

| 指标 | 含义 | 告警阈值 |
|------|------|---------|
| `milvus_querynode_memory_used_bytes` | QueryNode 内存使用 | > 80% 总内存 |
| `milvus_querynode_segment_num` | 加载的 Segment 数量 | - |
| `process_cpu_seconds_total` | CPU 使用时间 | > 80% |

---

## 常见性能问题与排查

### 问题1：搜索延迟高

**症状**：P99 搜索延迟从 10ms 飙升到 200ms。

**排查步骤**：

1. 检查 QueryNode 内存使用率——如果接近 100%，说明索引太大装不下，需要加 QueryNode 或换量化索引
2. 检查搜索参数——ef（HNSW）或 nprobe（IVFFlat）是否设得太大
3. 检查过滤条件——是否有高选择性的标量过滤拖慢了搜索
4. 检查 QueryNode 数量——是否需要扩容

```python
# 快速诊断：检查搜索延迟
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 40}}  # 用最小 ef 测试基线延迟
)
# 如果 ef=40 时延迟就很低 → 问题在搜索参数
# 如果 ef=40 时延迟还是高 → 问题在 QueryNode 资源
```

### 问题2：写入吞吐低

**症状**：插入 10 万条数据需要几分钟。

**排查步骤**：

1. 检查消息队列积压——如果 Pulsar/Kafka 有大量未消费的消息，说明 DataNode 消费速度跟不上
2. 检查 DataNode 数量——是否需要扩容
3. 检查批量大小——是否每次只插入几条数据（应该 5000~10000 条/批）
4. 检查是否频繁 flush——每次 flush 都会触发 Segment 持久化

### 问题3：索引构建慢

**症状**：`create_index()` 调用后长时间不返回。

**排查步骤**：

1. 检查 IndexNode 的 CPU 和内存——HNSW 构建是 CPU 和内存密集型操作
2. 检查数据量——1 亿条 768 维向量的 HNSW 索引构建可能需要数小时
3. 考虑换用 IVFFlat 或 IVF_PQ——构建速度比 HNSW 快

### 问题4：内存持续增长

**症状**：QueryNode 内存使用率持续上升，不释放。

**排查步骤**：

1. 检查 Segment 数量——是否有大量小 Segment 未合并
2. 检查是否需要 Compaction——`compact()` 合并小 Segment
3. 检查删除操作——删除是逻辑删除，需要 Compaction 物理清理

```python
# 手动触发 Compaction
client.compact(collection_name="documents")
```

---

## 参数调优

### QueryNode 参数

```yaml
# Milvus 配置文件（milvus.yaml）
queryNode:
  # 每个 QueryNode 加载的 Segment 数量上限
  # 增大可以加载更多数据，但需要更多内存
  segcore:
    segmentMaxRows: 1000000

  # 搜索并发度
  # 增大可以提升 QPS，但会增加 CPU 负载
  searchConcurrencyPerCore: 4
```

### DataNode 参数

```yaml
dataNode:
  # 每个 Segment 的最大行数
  # 增大可以减少 Segment 数量，但增加 Compaction 开销
  segmentMaxRows: 1000000

  # Flush 间隔
  # 减小可以让数据更快可搜索，但增加 Segment 碎片化
  flushInterval: 1000
```

### IndexNode 参数

```yaml
indexNode:
  # 索引构建的并行度
  # 增大可以加速索引构建，但增加 CPU 和内存使用
  buildIndexParallel: 4
```

---

## 常见误区：只监控搜索延迟

搜索延迟只是"结果"——你需要监控"原因"。搜索延迟高的原因可能是 QueryNode 内存不够、索引参数不合理、标量过滤太慢、Segment 碎片化等等。如果你只看搜索延迟，就只能知道"有问题"，但不知道"问题出在哪"。完整的监控应该覆盖：搜索延迟、写入吞吐、QueryNode 内存、Segment 数量、Compaction 进度、消息队列积压。

---

## 小结

这一节我们聊了 Milvus 的性能监控与调优：Prometheus + Grafana 是标准监控方案，关键指标包括搜索延迟、写入吞吐、QueryNode 内存和 Segment 数量。常见性能问题的排查思路是：搜索延迟高 → 检查 QueryNode 内存和搜索参数；写入吞吐低 → 检查消息队列积压和批量大小；索引构建慢 → 检查 IndexNode 资源；内存持续增长 → 检查 Segment 碎片化和 Compaction。下一节是本教程的最后一节——Milvus 的局限性与替代方案。
