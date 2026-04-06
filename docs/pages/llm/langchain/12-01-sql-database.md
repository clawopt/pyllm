---
title: 连接数据库（SQL 数据库）
description: LangChain SQL 工具链、数据库连接管理、Schema 感知查询、安全沙箱与只读模式
---
# 连接数据库（SQL 数据库）

前两个项目——智能客服系统（第 10 章）和代码分析助手（第 11 章）——分别面向客服场景和开发者场景。本章我们要进入第三个领域：**数据分析**。

想象这样一个场景：你的公司有一个 PostgreSQL 数据库，里面存着几百万条销售记录、用户行为日志、产品库存数据。业务人员（不是工程师）想问："上个月华东地区的销售额同比变化如何？""哪个品类的退货率最高？"传统做法是找数据分析师写 SQL、跑查询、做图表，周期长、成本高。如果有一个 AI 助手能直接连上数据库，把自然语言问题转成 SQL 查询、执行后返回结果甚至生成可视化图表——这就是 **Text-to-SQL Agent** 要做的事情。

## 为什么 Text-to-SQL 是刚需

先理解一下这个问题的本质。SQL 是一种**声明式查询语言**——你告诉数据库"要什么"，它自己决定"怎么拿"。但写好一条 SQL 需要知道：
- 数据库里有哪些表
- 表之间怎么关联（外键关系）
- 每个字段的含义和数据类型
- 聚合函数、窗口函数等高级语法的用法

对于不熟悉 SQL 的业务人员来说，这些门槛太高了。而 LLM 天然具备将自然语言映射到结构化查询的能力——只要它能正确理解数据库的 Schema（表结构），就能生成合理的 SQL。

### 一个直观的例子

```
用户自然语言: "上个月销售额最高的 10 个产品是什么？"

AI 生成的 SQL:
SELECT product_name, SUM(amount) as total_sales
FROM orders
WHERE order_date >= '2025-03-01' AND order_date < '2025-04-01'
GROUP BY product_name
ORDER BY total_sales DESC
LIMIT 10;

执行结果:
| product_name     | total_sales |
|------------------|-------------|
| 无线蓝牙耳机 Pro | 128,500.00  |
| 机械键盘 K95      | 98,200.00   ...
```

用户不需要知道 `GROUP BY`、`ORDER BY DESC`、日期范围过滤这些细节——LLM 帮他们处理好了。

## LangChain 的 SQL 工具链概览

LangChain 提供了一套完整的 SQL 工具链，核心组件如下：

| 组件 | 作用 | 所在包 |
|------|------|--------|
| `SQLDatabase` | 封装数据库连接，提供 Schema 查询接口 | `langchain_community.utilities` |
| `create_sql_query_chain` | 将自然语言转为 SQL 的 Chain | `langchain_chains` |
| `QuerySQLDataBaseTool` | 执行 SQL 并返回结果的 Tool | `langchain_community.tools` |
| `create_react_agent` | 组装 ReAct Agent 的工厂函数 | `langchain.agents` |

整个数据流是：

```
自然语言问题 → [SQL 生成] → SQL 语句 → [SQL 执行] → 查询结果 → [结果解释/可视化]
   (LLM)        (Chain)                    (Tool)              (LLM)
```

## 第一步：建立数据库连接

在开始之前，我们需要一个可以操作的数据库。为了教程的自包含性，我们用 Python 内建的 `sqlite3` 创建一个示例电商数据库：

```python
import sqlite3
import os

DB_PATH = "./ecommerce.db"

def create_sample_database(db_path: str = DB_PATH):
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        cost REAL NOT NULL,
        stock INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        city TEXT NOT NULL,
        region TEXT NOT NULL,
        registered_at TEXT DEFAULT (datetime('now')),
        tier TEXT DEFAULT 'normal'
    );

    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL DEFAULT 1,
        unit_price REAL NOT NULL,
        total_amount REAL NOT NULL,
        status TEXT DEFAULT 'completed',
        order_date TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (customer_id) REFERENCES customers(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );

    CREATE INDEX idx_orders_customer ON orders(customer_id);
    CREATE INDEX idx_orders_date ON orders(order_date);
    CREATE INDEX idx_orders_status ON orders(status);
    """)

    cursor.executescript("""
    INSERT INTO products (name, category, price, cost, stock) VALUES
        ('无线蓝牙耳机 Pro', '电子产品', 299.00, 120.00, 1500),
        ('机械键盘 K95', '电子产品', 499.00, 180.00, 800),
        ('人体工学椅 M300', '家具', 1299.00, 550.00, 200),
        ('27寸 4K 显示器', '电子产品', 2499.00, 1100.00, 350),
        ('智能台灯 L1', '家居', 199.00, 65.00, 2000),
        ('运动跑鞋 X7', '服装', 599.00, 200.00, 600),
        ('保温杯 500ml', '家居', 89.00, 25.00, 3000),
        ('无线充电板', '电子产品', 149.00, 45.00, 1200),
        ('真皮钱包', '配饰', 299.00, 80.00, 900),
        ('瑜伽垫 TPE', '运动', 129.00, 35.00, 2500);

    INSERT INTO customers (name, email, city, region, tier) VALUES
        ('张三', 'zhangsan@example.com', '上海', '华东', 'vip'),
        ('李四', 'lisi@example.com', '北京', '华北', 'normal'),
        ('王五', 'wangwu@example.com', '广州', '华南', 'vip'),
        ('赵六', 'zhaoliu@example.com', '深圳', '华南', 'normal'),
        ('孙七', 'sunqi@example.com', '杭州', '华东', 'normal'),
        ('周八', 'zhouba@example.com', '成都', '西南', 'vip'),
        ('吴九', 'wujiu@example.com', '武汉', '华中', 'normal'),
        ('郑十', 'zhengshi@example.com', '南京', '华东', 'normal');
    """)

    import random
    from datetime import datetime, timedelta

    order_data = []
    base_date = datetime(2025, 1, 1)
    statuses = ['completed', 'completed', 'completed', 'completed', 'completed',
                'shipped', 'cancelled', 'returned']

    for i in range(500):
        cust_id = random.randint(1, 8)
        prod_id = random.randint(1, 10)
        qty = random.choice([1, 1, 1, 2, 2, 3, 5])
        cursor.execute("SELECT price FROM products WHERE id = ?", (prod_id,))
        unit_price = cursor.fetchone()[0]

        order_date = base_date + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(8, 20),
            minutes=random.randint(0, 59)
        )
        status = random.choices(statuses, weights=[50, 15, 10, 10, 5, 5, 3, 2])[0]

        order_data.append((
            cust_id, prod_id, qty, unit_price,
            round(qty * unit_price, 2), status, order_date.strftime('%Y-%m-%d %H:%M')
        ))

    cursor.executemany("""
        INSERT INTO orders (customer_id, product_id, quantity, unit_price,
                           total_amount, status, order_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, order_data)

    conn.commit()
    print(f"✅ 示例数据库创建完成: {db_path}")
    print(f"   产品: {cursor.execute('SELECT COUNT(*) FROM products').fetchone()[0]} 条")
    print(f"   客户: {cursor.execute('SELECT COUNT(*) FROM customers').fetchone()[0]} 条")
    print(f"   订单: {cursor.execute('SELECT COUNT(*) FROM orders').fetchone()[0]} 条")

    conn.close()

create_sample_database()
```

输出：

```
✅ 示例数据库创建完成: ./ecommerce.db
   产品: 10 条
   客户: 8 条
   订单: 500 条
```

这个示例数据库模拟了一个小型电商公司的核心数据：10 个产品、8 个客户、500 条订单记录（跨越 2025 年 Q1），涵盖了不同地区、不同状态（已完成/已发货/已取消/已退回）。

## 用 LangChain 连接数据库

有了数据库之后，用 LangChain 的 `SQLDatabase` 来封装连接：

```python
from langchain_community.utilities import SQLDatabase

def connect_database(db_path: str = DB_PATH) -> SQLDatabase:
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    return db

db = connect_database()
print(f"数据库类型: {db.dialect}")
print(f"表列表: {db.get_usable_table_names()}")
```

输出：

```
数据库类型: sqlite
表列表: ['customers', 'orders', 'products']
```

`SQLDatabase` 是一个轻量级的封装层，它提供了几个关键能力：

### 能力一：获取 Schema 信息

```python
schema_info = db.get_table_info()
print(schema_info)
```

输出：

```
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    cost REAL NOT NULL,
    stock INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
)

/*
3 rows from products table:
id  name               category  price  cost  stock  created_at
1   无线蓝牙耳机 Pro     电子产品  299.0  120.0  1500   ...
2   机械键盘 K95         电子产品  499.0  180.0  800    ...
*/

CREATE TABLE customers (
...
```

注意 `get_table_info()` 不仅返回了建表语句（DDL），还包含了每个表的**前几行样例数据**。这对 LLM 理解字段含义极其重要——光看 `price REAL` 不知道价格的大致范围，但看到样例数据 `299.0, 499.0` 就立刻明白了。

### 能力二：执行 SQL 查询

```python
result = db.run("SELECT category, COUNT(*) as cnt, AVG(price) as avg_price "
                "FROM products GROUP BY category ORDER BY cnt DESC")
print(result)
```

输出：

```
[(('电子产品', 4, 861.5),), (('家居', 2, 144.0),), (('家具', 1, 1299.0),), ...]
```

返回值是一个嵌套元组结构，后续需要做格式化处理才能展示给用户。

### 能力三：安全的只读模式

这是生产环境最重要的考量：**绝不允许 AI 生成的 SQL 包含写入操作（INSERT/UPDATE/DELETE/DROP）**。

```python
from langchain_community.utilities import SQLDatabase

db_safe = SQLDatabase.from_uri(
    f"sqlite:///{DB_PATH}",
    sample_rows_in_table_info=3,
    view_support=True,
)
```

`SQLDatabase` 本身不提供内置的读写权限控制——它只是一个连接封装。真正的安全防护需要在两个层面实现：

**第一层：工具层面限制**。LangChain 提供的 `QuerySQLDataBaseTool` 默认只允许 SELECT 查询：

```python
from langchain_community.tools import QuerySQLDataBaseTool

safe_query_tool = QuerySQLDataBaseTool(db=db_safe, name="sql_query",
    description="执行 SQL SELECT 查询并返回结果。只能用于读取数据，不能修改数据。"
)
```

**第二层：SQL 注入检测与拦截**。在执行前对生成的 SQL 做正则检查：

```python
import re

FORBIDDEN_SQL_PATTERNS = [
    r'\bDROP\b', r'\bDELETE\b', r'\bINSERT\b', r'\bUPDATE\b',
    r'\bALTER\b', r'\bCREATE\b', r'\bTRUNCATE\b',
    r';\s*--', r';\s*select',  # 多语句注入
]

def validate_sql(sql: str) -> tuple[bool, str]:
    sql_lower = sql.lower().strip()
    if not sql_lower.startswith("select"):
        return False, "只允许 SELECT 查询"

    for pattern in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, sql_lower):
            return False, f"检测到危险操作: {pattern}"

    return True, ""
```

## Schema 感知的查询优化

LLM 生成 SQL 的质量高度依赖于它对 Schema 的理解程度。一个常见的错误是：LLM 猜测了错误的列名或表关联方式。我们可以通过增强 Schema 信息来改善这个问题。

### 自定义 Schema 描述

默认的 Schema 只包含 DDL 和样例数据。我们可以添加业务语义注释来帮助 LLM 更准确地生成查询：

```python
class EnhancedSQLDatabase(SQLDatabase):
    def get_table_info(self, table_names=None):
        base_info = super().get_table_info(table_names)

        annotations = {
            "products": """
-- 业务说明: 商品主表
-- name: 商品名称（中文）
-- category: 商品分类（电子产品/家居/家具/服装/配饰/运动）
-- price: 零售价（元）
-- cost: 进货成本（元）
-- stock: 当前库存数量
-- 利润率 = (price - cost) / price""",
            "customers": """
-- 业务说明: 客户信息表
-- region: 大区（华北/华东/华南/西南/华中）
-- tier: 客户等级（normal/vip）
-- city: 所在城市""",
            "orders": """
-- 业务说明: 订单主表
-- status: completed=已完成 shipped=已发货 cancelled=已取消 returned=已退回
-- total_amount = quantity * unit_price
-- order_date 格式: YYYY-MM-DD HH:MM"""
        }

        annotated = base_info
        for table_name, note in annotations.items():
            if not table_names or table_name in table_names:
                annotated = annotated.replace(
                    f"CREATE TABLE {table_name}",
                    f"CREATE TABLE {table_name}\n{note}"
                )

        return annotated


db_enhanced = EnhancedSQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
schema_with_notes = db_enhanced.get_table_info(["products"])
print(schema_with_notes[:400])
```

输出：

```
CREATE TABLE products
-- 业务说明: 商品主表
-- name: 商品名称（中文）
-- category: 商品分类（电子产品/家居/家具/服装/配饰/运动）
-- price: 零售价（元）
-- cost: 进货成本（元）
-- stock: 当前库存数量
-- 利润率 = (price - cost) / price
(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    ...
```

这些注释会在后续传给 LLM 作为 prompt 的一部分，显著提升 SQL 生成的准确性。比如当用户问"利润率最高的产品"时，LLM 从注释中知道了 `利润率 = (price - cost) / price`，就能正确写出 `(price - cost) / price * 100 AS profit_margin` 这样的表达式。

## 支持多种数据库方言

上面的例子用的是 SQLite（开发测试方便）。在生产环境中，你可能使用 PostgreSQL、MySQL 或其他数据库。LangChain 的 `SQLDatabase.from_uri()` 通过不同的 URI scheme 自动适配：

```python
connections = {
    "sqlite":  "sqlite:///./ecommerce.db",
    "postgresql": "postgresql://user:pass@localhost:5432/mydb",
    "mysql": "mysql+pymysql://user:pass@localhost:3306/mydb",
}

def get_db(dialect: str = "sqlite", **kwargs) -> SQLDatabase:
    uri_template = connections.get(dialect)
    if not uri_template:
        raise ValueError(f"不支持的数据库类型: {dialect}")
    return SQLDatabase.from_uri(uri_template, **kwargs)
```

对于 PostgreSQL 用户，还有一个特别有用的功能：**自动发现表之间的外键关系**。SQLite 不强制外键约束（除非显式开启），但 PostgreSQL 会维护完整的外键元数据，`SQLDatabase` 可以利用这些信息来帮助 LLM 正确地 JOIN 表：

```python
pg_db = SQLDatabase.from_uri("postgresql://user:pass@localhost:5432/ecommerce")
print(pg_db.get_usable_foreign_keys())
# 输出: [('orders', 'customer_id', 'customers', 'id'), ...]
```

这在外键关系复杂的数据库中非常有价值——LLM 不需要猜测 `orders.customer_id` 应该 JOIN 到哪个表的哪个字段。

## 常见误区

**误区一：让 AI 直接操作生产数据库**。无论 prompt 写得多安全，都不能保证 LLM 100% 不会生成危险的 SQL。正确的做法是：**连接只读副本（Read Replica）** 或设置数据库用户为 `SELECT-only` 权限。

**误区二：Schema 太大导致 Token 爆炸**。如果你的数据库有 100 张表、每张表 50 个字段，完整的 Schema 可能有几万个 token——这会严重挤占 LLM 的上下文窗口。解决方案是用 `include_tables` 或 `custom_table_info` 只暴露必要的表和字段。

**误区三：忽略 SQL 执行超时**。LLM 可能生成没有 WHERE 子句的全表扫描查询（比如 `SELECT * FROM orders`），在大表上可能执行几分钟。应该设置查询超时并在工具层面对结果集大小做限制。
