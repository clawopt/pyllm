---
title: 数据库读写 SQL
description: read_sql/to_sql 参数详解、SQLite/PostgreSQL 实战、大模型场景下的数据库 I/O 最佳实践
---
# 数据库与 SQL 交互

在 LLM 开发的实际工作流中，你的原始数据往往不是以文件形式躺在磁盘上的，而是存在数据库里——产品数据库里存着用户对话日志，标注平台的数据库里存着人工审核结果，评估系统的数据库里存着模型打分记录。Pandas 的 `read_sql()` 和 `to_sql()` 就是连接"数据库世界"和"Pandas 数据处理世界"的桥梁。

这一节我们不罗列所有参数，而是从实际的 LLM 开发场景出发，讲清楚：什么时候该用数据库 I/O、怎么高效地读取大量数据、怎么把清洗后的数据写回去，以及生产环境中的最佳实践。

## 什么时候需要 Pandas 连数据库

先想清楚一个问题：既然数据库本身就能做聚合、过滤、排序，为什么还要把数据读到 Pandas 里来处理？答案在于**分工不同**。数据库擅长的是基于索引的精确查询和预聚合统计，而 Pandas 擅长的是复杂的行级数据变换、文本处理、多步骤的 ETL 流水线。

在 LLM 开发中，典型的数据流是这样的：

```
产品 PostgreSQL ──read_sql──→ Pandas 清洗/变换 ──to_sql/导出──→ 训练文件(JSONL)
        ↑                                                            │
        │  用户对话日志                                              ↓
        │  标注结果回写                                    模型训练/SFT
        │  质量分数更新
```

具体来说，你会在以下场景频繁使用 Pandas 的数据库功能：

- **从产品数据库拉取原始语料**：用 SQL 做初步筛选（时间范围、用户群体），然后把结果拉到 Pandas 做深度清洗
- **合并多源数据**：从不同的表 JOIN 出来之后，在 Pandas 里做特征工程
- **回写处理结果**：清洗后的质量评分、去重标记等写回数据库供下游系统消费
- **本地缓存中间结果**：用 SQLite 做轻量级的本地持久化缓存

## read_sql()：从数据库获取数据

### 基本用法与两种连接方式

Pandas 的数据库操作支持两种连接方式：一种是 Python 标准库的 `sqlite3` 直连，另一种是 SQLAlchemy 的 engine 对象。两者的选择取决于你的场景：

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect(':memory:')

conn.execute("""
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    prompt TEXT,
    response TEXT,
    quality_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.executemany("""
INSERT INTO conversations (user_id, prompt, response, quality_score)
VALUES ('u_001', '什么是AI', 'AI是...', 4.5),
       ('u_002', '解释Python', 'Python是...', 3.8),
       ('u_003', '如何学习LLM', '建议从...', 4.2)
""")

df = pd.read_sql_query(
    "SELECT * FROM conversations WHERE quality_score >= 4.0",
    conn
)
print(df)
```

输出：

```
   id user_id       prompt   response  quality_score           created_at
0   1   u_001      什么是AI     AI是...            4.5  2025-01-15 10:30:00
2   3   u_003  如何学习LLM   建议从...            4.2  2025-01-15 10:32:00
```

对于 SQLite 这种嵌入式数据库，直接用 `sqlite3.connect()` 就够了。但如果你连的是 PostgreSQL 或 MySQL，**强烈建议使用 SQLAlchemy 的 engine**：

```python
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://user:password@localhost:5432/llm_db')

df = pd.read_sql(
    text("SELECT * FROM conversations WHERE created_at > :start_date"),
    engine,
    params={'start_date': '2025-01-01'}
)
```

为什么推荐 SQLAlchemy？三个原因。第一是**连接池管理**——每次查询不需要新建连接，引擎会自动复用已有连接。第二是**参数化查询**——用 `:param_name` 占位符而不是字符串拼接，彻底杜绝 SQL 注入风险。第三是**跨数据库兼容性**——同样的代码只需改连接字符串就能切换 PostgreSQL / MySQL / Oracle。

### 复杂查询：把重活交给数据库

一个重要的性能原则是：**能用 SQL 完成的聚合和过滤，就不要拉到 Pandas 里再做**。数据库有索引优化、查询计划器、列式存储等优势，对于大规模数据的初筛和预聚合比 Pandas 高效得多。比如你需要按日统计对话数量和质量分布：

```python
query = """
WITH daily_stats AS (
    SELECT 
        DATE(created_at) AS date,
        COUNT(*) AS conversation_count,
        AVG(quality_score) AS avg_quality,
        SUM(CASE WHEN quality_score >= 4.0 THEN 1 ELSE 0 END) AS high_quality_count
    FROM conversations
    GROUP BY DATE(created_at)
),
top_users AS (
    SELECT 
        user_id,
        COUNT(*) AS total_conversations,
        AVG(quality_score) AS user_avg_score,
        RANK() OVER (ORDER BY AVG(quality_score) DESC) AS quality_rank
    FROM conversations
    GROUP BY user_id
)
SELECT 
    d.*,
    t.user_id AS top_user_that_day,
    t.user_avg_score
FROM daily_stats d
CROSS JOIN LATERAL (
    SELECT * FROM top_users WHERE quality_rank = 1 LIMIT 1
) t
ORDER BY d.date DESC
LIMIT 30
"""

df = pd.read_sql(query, conn)
print(df.head())
```

这种包含 CTE（Common Table Expression）、窗口函数（RANK）、LATERAL JOIN 的复杂查询，让数据库一次算完返回精简的结果集，而不是把几百万行原始数据全拉到 Pandas 里再 groupby。**好的做法是：SQL 负责粗筛和预聚合，Pandas 负责精细的数据变换和特征工程**。

## to_sql()：把数据写回数据库

### 基本写入与 if_exists 策略

```python
import pandas as pd

df = pd.DataFrame({
    'chunk_id': ['ch001', 'ch002', 'ch003'],
    'doc_source': ['api.md', 'guide.md', 'paper.pdf'],
    'token_count': [256, 512, 384],
    'embedding_model': ['text-embedding-3-small'] * 3,
})

df.to_sql('kb_chunks', conn, if_exists='replace', index=False)
```

`if_exists` 参数控制目标表已存在时的行为，这是最容易出错的参数之一，三种选项的含义必须搞清楚：

- `'fail'`（默认）：表已存在则抛出异常——防止意外覆盖
- `'replace'`：删除旧表、创建新表、写入数据——**慎用**，会丢失表中已有的其他数据
- `'append'`：在现有表后面追加行——最常用，但要确保列结构一致

```python
df_new = pd.DataFrame({
    'chunk_id': ['ch004'], 'doc_source': ['blog.md'],
    'token_count': [200], 'embedding_model': ['bge-large-zh'],
})

df_new.to_sql('kb_chunks', conn, if_exists='append', index=False)
```

### 批量写入的性能陷阱

`to_sql()` 默认的实现方式是逐行生成 INSERT 语句然后逐条执行。对于几千行数据这没问题，但如果你要写入几十万甚至几百万行，默认方式慢到不可接受：

```python
import pandas as pd
import numpy as np
import time

n = 500_000
large_df = pd.DataFrame({
    'id': range(n),
    'value': np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C'], n),
})

start = time.time()
large_df.to_sql('test_default', conn, if_exists='replace', index=False, method=None)
t_default = time.time() - start

start = time.time()
large_df.to_sql('test_multi', conn, if_exists='replace', index=False, method='multi')
t_multi = time.time() - start

print(f"逐行 INSERT: {t_default:.1f}s")
print(f"批量 multi:  {t_multi:.1f}s")
print(f"加速: {t_default/t_multi:.0f}x")
```

典型输出：

```
逐行 INSERT: 180.3s
批量 multi:   12.8s
加速: 14x
```

14 倍的差异只来自一个参数：`method='multi'`。它的作用是把多条 VALUES 合并成一条 INSERT 语句（默认每批 1000 行），大幅减少网络往返次数。如果数据量更大（百万级行），还可以配合 `chunksize` 参数进一步控制每批的大小：

```python
df.to_sql('big_table', engine, if_exists='append', index=False,
          method='multi', chunksize=10000)
```

对于 PostgreSQL 用户还有一个终极武器：COPY 协议。它绕过 SQL 解析器直接从文件批量导入，速度比 INSERT 快一个数量级。做法是先把 DataFrame 写成 CSV，再通过 COPY 导入：

```python
import tempfile, os

csv_path = os.path.join(tempfile.gettempdir(), 'bulk_import.csv')
large_df.to_csv(csv_path, index=False, header=False)

with engine.begin() as conn:
    conn.execute(text(f"COPY big_table FROM '{csv_path}' WITH (FORMAT CSV)"))
```

## 生产级操作模板：DataWarehouse 类

把上面的知识整合起来，这里给出一个在生产环境中可以直接使用的数据库交互类。它封装了连接管理、参数化查询、批量写入、错误处理等常见需求：

```python
import pandas as pd
from sqlalchemy import create_engine, text
from contextlib import contextmanager


class DataWarehouse:
    """LLM 数据仓库交互封装"""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    
    @contextmanager
    def _connection(self):
        with self.engine.connect() as conn:
            yield conn
    
    def extract_conversations(self, start_date: str, end_date: str,
                               min_quality: float = 3.0,
                               limit: int = None) -> pd.DataFrame:
        sql = text("""
            SELECT c.id, u.username, c.prompt, c.response,
                   c.quality_score, c.created_at, c.token_count
            FROM conversations c
            JOIN users u ON c.user_id = u.id
            WHERE c.created_at BETWEEN :start AND :end
              AND c.quality_score >= :min_q
            ORDER BY c.created_at DESC
        """)
        params = {'start': start_date, 'end': end_date, 'min_q': min_quality}
        
        with self._connection() as conn:
            result = conn.execute(sql, params)
            df = pd.DataFrame(result.all(), columns=result.keys())
        
        if limit and len(df) > limit:
            df = df.head(limit)
        return df
    
    def write_cleaned_data(self, df: pd.DataFrame, table_name: str):
        df.to_sql(
            table_name,
            self.engine,
            if_exists='replace',
            index=False,
            chunksize=50000,
            method='multi',
        )
    
    def append_evaluation_results(self, results_df: pd.DataFrame):
        results_df.to_sql(
            'evaluation_results',
            self.engine,
            if_exists='append',
            index=False,
            chunksize=10000,
            method='multi',
        )


dw = DataWarehouse("postgresql://user:pass@localhost:5432/llm_db")

conversations = dw.extract_conversations(
    start_date='2025-01-01',
    end_date='2025-03-31',
    min_quality=4.0,
)
print(f"拉取到 {len(conversations):,} 条高质量对话")

dw.write_cleaned_data(conversations, 'sft_train_v3')
```

这个类的设计有几个值得注意的点。第一，用 `create_engine` 而不是裸连接——自带连接池和自动重连。第二，SQL 查询用 `text()` 包装 + 字典传参——防止 SQL 注入。第三，`to_sql()` 统一指定了 `method='multi'` 和 `chunksize`——避免逐行插入的性能陷阱。第四，用上下文管理器 (`@contextmanager`) 管理连接生命周期——确保连接一定被归还到连接池。

## SQLite 作为本地缓存层

不是所有的数据库操作都需要连远程服务器。SQLite 是一个嵌入式的、无需安装的、零配置的数据库引擎，非常适合做本地数据缓存。在 LLM 数据处理流程中，我经常用它来缓存那些计算代价高但不常变动的中间结果：

```python
import pandas as pd
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime, timedelta


class LocalCache:
    """基于 SQLite 的本地计算结果缓存"""
    
    def __init__(self, db_path: str = '.cache/local.db'):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""CREATE TABLE IF EXISTS cache (
            key TEXT PRIMARY KEY,
            data BLOB,
            cached_at TIMESTAMP
        )""")
    
    def get_or_compute(self, key: str, compute_fn, ttl_hours: int = 24):
        cached = pd.read_sql(
            "SELECT data, cached_at FROM cache WHERE key=?",
            self.conn, params=[key]
        )
        
        if len(cached) > 0:
            cached_at = pd.to_datetime(cached.iloc[0]['cached_at'])
            if datetime.now() - cached_at < timedelta(hours=ttl_hours):
                return pickle.loads(cached.iloc[0]['data'])
        
        result = compute_fn()
        
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, data, cached_at) VALUES (?, ?, ?)",
            (key, pickle.dumps(result), pd.Timestamp.now())
        )
        self.conn.commit()
        return result
    
    def close(self):
        self.conn.close()


cache = LocalCache()

df = cache.get_or_compute(
    'expensive_aggregation_v2',
    lambda: pd.DataFrame({'a': range(100000)}).groupby(lambda x: x % 10).sum(),
    ttl_hours=6,
)
print(df.head())
cache.close()
```

这个缓存类的逻辑很直观：先查 SQLite 看有没有缓存结果，如果有且未过期就直接返回；如果没有或已过期就重新计算、序列化后存入 SQLite。下次同样的查询就会命中缓存。对于那种需要跑几分钟的复杂聚合操作，缓存能节省大量重复计算时间。

## 常见问题速查

最后快速列出几个数据库 I/O 中最高频的问题和解决方案：

**时区不一致**。数据库存的通常是 UTC 时间，但你可能需要北京时间来分析：

```python
df['created_at'] = pd.to_datetime(df['created_at'])
df['beijing_time'] = df['created_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
```

**大数据量导出超时**。不要试图一次性 to_sql 几百万行，分批追加：

```python
for i in range(0, len(large_df), 50000):
    batch = large_df.iloc[i:i+50000]
    batch.to_sql('target', engine, if_exists='append',
                 index=False, method='multi', chunksize=10000)
```

**内存型 DataFrame 直接入库用于测试**。开发阶段不需要真的连数据库，SQLite 的 `:memory:` 模式让你可以在内存中创建临时数据库跑通整个流程：

```python
conn = sqlite3.connect(':memory:')
df.to_sql('test_table', conn)
result = pd.read_sql("SELECT * FROM test_table WHERE score > 3", conn)
```

到这里，文本文件（CSV/JSONL）和数据库两种数据来源的读写方法都覆盖了。下一节我们要看的是 Parquet、Feather 这些更适合大规模数据的高性能格式。
