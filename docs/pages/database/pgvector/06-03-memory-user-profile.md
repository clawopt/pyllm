# 6.3 对话记忆与用户画像

> **pgvector 的 SQL 能力让记忆管理更优雅——用 JOIN 关联用户，用 WHERE 过期清理**

---

## 这一节在讲什么？

在 Chroma 教程中，我们用独立的 Memory Collection 存储对话历史和用户偏好。pgvector 的方案更优雅——因为你可以用 SQL 的全部能力来管理记忆：用 JOIN 关联用户表、用 WHERE 条件做 TTL 清理、用 UNION 合并文档库和记忆库的搜索结果。这一节我们要设计对话记忆和用户画像的表结构，并实现基于 pgvector 的混合记忆检索。

---

## 表设计

```sql
-- 对话历史表
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    user_id INTEGER NOT NULL,
    role VARCHAR(20) NOT NULL,      -- 'user' 或 'assistant'
    content TEXT NOT NULL,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 用户偏好表（长期记忆）
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    preference_key VARCHAR(100) NOT NULL,
    preference_value TEXT NOT NULL,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, preference_key)
);

-- 索引
CREATE INDEX idx_conv_session ON conversations (session_id);
CREATE INDEX idx_conv_user ON conversations (user_id);
CREATE INDEX idx_conv_embedding ON conversations USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_pref_user ON user_preferences (user_id);
CREATE INDEX idx_pref_embedding ON user_preferences USING hnsw (embedding vector_cosine_ops);
```

---

## 混合记忆检索

pgvector 的记忆检索比 Chroma 更强大——你可以用 UNION 同时搜索文档库和记忆库，用 JOIN 关联用户权限：

```python
def search_with_memory(cur, query_embedding, user_id, top_k=5):
    """混合检索：文档库 + 对话记忆 + 用户偏好"""
    cur.execute("""
        -- 文档库搜索
        SELECT id, content, source, embedding <=> %s AS distance, 'document' AS type
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s

        UNION ALL

        -- 对话记忆搜索
        SELECT id, content, session_id::text, embedding <=> %s AS distance, 'memory' AS type
        FROM conversations
        WHERE user_id = %s
        ORDER BY embedding <=> %s
        LIMIT %s

        UNION ALL

        -- 用户偏好搜索
        SELECT id, preference_value, preference_key, embedding <=> %s AS distance, 'preference' AS type
        FROM user_preferences
        WHERE user_id = %s
        ORDER BY embedding <=> %s
        LIMIT %s

        ORDER BY distance
        LIMIT %s
    """, (
        query_embedding, query_embedding, top_k,
        query_embedding, user_id, query_embedding, 3,
        query_embedding, user_id, query_embedding, 2,
        top_k
    ))
    return cur.fetchall()
```

---

## TTL 清理：利用 PostgreSQL 的定时任务

PostgreSQL 的 `pg_cron` 扩展可以定时执行 SQL，自动清理过期记忆：

```sql
-- 安装 pg_cron 扩展
CREATE EXTENSION pg_cron;

-- 每天凌晨 2 点清理 90 天前的对话记录
SELECT cron.schedule(
    'cleanup-old-conversations',
    '0 2 * * *',
    $$DELETE FROM conversations WHERE created_at < NOW() - INTERVAL '90 days'$$
);

-- 查看已配置的定时任务
SELECT * FROM cron.job;
```

如果你使用的是不支持 pg_cron 的云托管 PostgreSQL，可以用应用层的定时任务替代：

```python
def cleanup_old_memories(cur, max_age_days=90):
    """清理过期记忆"""
    cur.execute(
        "DELETE FROM conversations WHERE created_at < NOW() - INTERVAL '%s days'",
        (str(max_age_days),)
    )
    removed = cur.rowcount
    conn.commit()
    print(f"🗑️ 清理了 {removed} 条过期对话记录")
```

---

## 常见误区

### 误区 1：把所有记忆都存在同一个表中

对话历史、用户偏好、文档数据有不同的生命周期和查询模式，应该分表存储。对话历史需要 TTL 清理，用户偏好是永久的，文档数据按业务需求管理。

### 误区 2：UNION 查询性能差

UNION ALL（不去重）的性能通常很好，因为每个子查询都独立使用自己的索引。UNION（去重）会额外排序，性能较差。在记忆检索场景中，不同类型的数据不会重复，用 UNION ALL 即可。

---

## 本章小结

pgvector 的 SQL 能力让记忆管理更优雅。核心要点回顾：第一，对话历史和用户偏好分表存储，各有独立的索引；第二，UNION ALL 同时搜索文档库和记忆库，按距离统一排序；第三，pg_cron 扩展实现自动 TTL 清理，云托管环境用应用层定时任务替代；第四，JOIN 关联用户表实现权限控制。

下一章我们将进入生产部署与进阶主题——Docker Compose 部署、性能调优、监控运维、以及 pgvector 的局限性和替代方案。
