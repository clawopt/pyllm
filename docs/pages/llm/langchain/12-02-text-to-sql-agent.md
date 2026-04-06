---
title: 设计 Text-to-SQL 代理
description: SQL 生成 Chain 原理、ReAct Agent 组装、Few-Shot 示例注入、查询结果自然语言解释、错误恢复机制
---
# 设计 Text-to-SQL 代理

上一节我们完成了数据库连接和 Schema 管理的准备工作。现在进入核心环节：**让 LLM 把用户的自然语言问题翻译成可执行的 SQL 查询**。

这不是一个简单的"文本到文本"翻译任务——它需要 LLM 同时理解三个层面的信息：**用户意图（自然语言）+ 数据库结构（Schema）+ SQL 语法规则**。任何一环出错，生成的 SQL 就可能返回错误结果或直接报错。

## 从 Chain 到 Agent 的演进

LangChain 提供了两种实现 Text-to-SQL 的方式，理解它们的区别很重要：

### 方式一：纯 Chain 模式

`create_sql_query_chain` 是一个两步 Chain：LLM 生成 SQL → 执行并返回原始结果。简单但能力有限：

```python
from langchain.chains import create_sql_query_chain

sql_chain = create_sql_query_chain(
    llm=get_llm(),
    db=db_enhanced,
)

result = sql_chain.invoke({"question": "销售额最高的前5个产品是什么？"})
print(f"生成的 SQL:\n{result}")
```

输出：

```sql
SELECT p.name, SUM(o.total_amount) as total_sales
FROM products p
JOIN orders o ON p.id = o.product_id
WHERE o.status = 'completed'
GROUP BY p.name
ORDER BY total_sales DESC
LIMIT 5
```

Chain 模式的问题在于：
- 只能生成一次 SQL，如果报错没有自动修复能力
- 返回的是原始数据元组，用户看不懂
- 无法处理多轮追问（"按地区拆分呢？"）

### 方式二：ReAct Agent 模式（推荐）

Agent 模式下，SQL 生成只是工具调用链中的一步。Agent 可以：
1. 先查看 Schema 了解表结构
2. 生成 SQL 并执行
3. 如果报错 → 分析错误原因 → 修改 SQL 重试
4. 对执行结果做进一步分析或可视化

```
用户: "上个月各地区的销售情况怎么样？"
         │
    ▼ Agent 开始推理
Thought: 我需要查询订单表，按客户所在地区分组统计销售额。
       让我先确认一下 customers 表和 orders 表的结构。
Action: sql_db_schema  → 获取表结构
Observation: (Schema 信息...)
         │
    ▼ 了解结构后生成 SQL
Action: sql_db_query   → SELECT c.region, SUM(o.total_amount)...
Observation: ((('华东', 285430.0),), (('华北', 198200.0),), ...)
         │
    ▼ 结果看起来正确，组织回答
Final Answer: 根据数据库中的订单记录...
```

## 构建完整的 Text-to-SQL Agent

### 第一步：定义工具集

Agent 需要三种工具：查 Schema、执行查询、验证 SQL 安全性：

```python
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.tools.sql_database.tool import InfoSQLDatabaseTool, ListSQLDatabaseTool
from langchain_core.tools import tool

db = connect_database()
db_enhanced = EnhancedSQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

list_tables_tool = ListSQLDatabaseTool(db=db_enhanced,
    description="列出数据库中所有可用的表名。"
)

schema_info_tool = InfoSQLDatabaseTool(db=db_enhanced,
    description="获取指定表的完整 Schema 信息（字段名、类型、样例数据和业务注释）。"
    "输入格式: 用逗号分隔的表名列表，如 'products,orders'。"
)

@tool
def safe_sql_query(sql: str) -> str:
    """执行安全的只读 SQL 查询。
    输入必须是合法的 SELECT 语句。系统会自动校验安全性。
    """
    is_safe, error_msg = validate_sql(sql)
    if not is_safe:
        return f"❌ SQL 安全检查未通过: {error_msg}"

    try:
        result = db.run(sql)
        return f"✅ 查询成功\n\n{result}"
    except Exception as e:
        return f"❌ 执行失败: {type(e).__name__}: {e}"


tools = [list_tables_tool, schema_info_tool, safe_sql_query]
```

### 第二步：设计 System Prompt

Text-to-SQL Agent 的 prompt 是决定质量的关键因素。一个好的 prompt 需要明确告诉 LLM：

```python
TEXT_TO_SQL_SYSTEM = """你是一个数据分析助手，能够将自然语言问题转换为 SQL 查询，
从关系型数据库中提取数据并进行分析。

## 数据库概况
你连接的是一个电商数据库，包含以下核心业务表:
- **products**: 商品信息（名称、分类、价格、成本、库存）
- **customers**: 客户信息（姓名、城市、大区、等级）
- **orders**: 订单信息（关联客户和商品，含金额、状态、日期）

## 工作流程
1. 当收到问题时，先用 schema_info_tool 查看相关表的完整结构
2. 基于表结构和业务注释，编写正确的 SQL 查询
3. 使用 safe_sql_query 执行查询
4. 如果查询报错，分析错误信息并修正 SQL 后重试
5. 将查询结果用清晰的中文解释给用户

## SQL 编写规范
- 所有 JOIN 必须基于已知的外键关系
- 金额字段保留 2 位小数 (ROUND)
- 日期使用 'YYYY-MM-DD' 格式
- GROUP BY 时同时 SELECT 的非聚合字段必须都在 GROUP BY 中
- 对于涉及金额的计算，注意区分 price（单价）和 total_amount（总价）
- 订单状态: completed(已完成)/shipped(已发货)/cancelled(已取消)/returned(已退回)
- 客户等级: normal/vip

## 错误处理
- 如果 SQL 语法错误，仔细阅读错误提示并修正
- 如果查询结果为空，考虑是否日期范围、条件过滤过严等问题
- 不要编造数据——如果数据库中没有相关信息，如实告知用户"""
```

### 第三步：创建 Agent

```python
from langchain.agents import create_react_agent

agent = create_react_agent(
    llm=get_llm(),
    tools=tools,
    prompt=TEXT_TO_SQL_SYSTEM,
)
```

### 测试基本查询

```python
result = agent.invoke({
    "input": "我们总共有多少个产品？分品类各多少个？"
})
print(result["output"])
```

Agent 内部过程（简化）：

```
Thought: 用户想知道产品的总数和各品类的数量分布。
       我先查看 products 表的 Schema。
Action: [schema_info] input="products"
Observation: CREATE TABLE products (... name, category, price ...)

Thought: 表结构清楚了。我需要用 COUNT(*) 统计总数，
       以及 GROUP BY category 统计各品类数量。
Action: [safe_sql_query] input="SELECT category, COUNT(*) as count FROM products GROUP BY category ORDER BY count DESC"
Observation: ✅ 查询成功 ((('电子产品', 4),), (('家居', 2),), ...)

Final Answer: 数据库中共有 **10 个产品**，按品类分布如下：

| 品类 | 数量 |
|------|------|
| 电子产品 | 4 |
| 家居 | 2 |
| 家具 | 1 |
| 服装 | 1 |
| 配饰 | 1 |
| 运动 | 1 |
```

### 测试复杂的多表 JOIN 查询

```python
result = agent.invoke({
    "input": "2025年3月份每个地区的销售额是多少？哪个地区最高？"
})
print(result["output"])
```

这个查询涉及三件事：
1. 日期范围过滤（2025年3月）
2. 多表 JOIN（orders → customers 获取 region）
3. 分组聚合 + 排序取最大值

Agent 会依次调用 `schema_info` 查看 `orders` 和 `customers` 的关联方式，然后生成类似这样的 SQL：

```sql
SELECT
    c.region AS 地区,
    ROUND(SUM(o.total_amount), 2) AS 销售额,
    COUNT(DISTINCT o.id) AS 订单数
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date >= '2025-03-01'
  AND o.order_date < '2025-04-01'
  AND o.status = 'completed'
GROUP BY c.region
ORDER BY 销售额 DESC
```

## Few-Shot 示例：提升 SQL 质量

对于复杂的查询模式，纯靠 prompt 描述往往不够——LLM 可能不知道你的数据库有哪些特殊的约定。**Few-Shot（少样本示例）** 是最有效的解决方案：在 prompt 中提供几个"问题→正确SQL"的对作为参考。

```python
FEW_SHOT_EXAMPLES = """
## 参考示例

### 示例 1
问题: 上个月销售额最高的3个产品是什么？
SQL:
SELECT p.name AS 产品名, SUM(o.total_amount) AS 总销售额
FROM products p
JOIN orders o ON p.id = o.product_id
WHERE o.status = 'completed'
  AND o.order_date >= date('now', '-30 days')
GROUP BY p.name
ORDER BY 总销售额 DESC
LIMIT 3;

### 示例 2
问题: VIP客户的平均消费金额比普通客户高多少？
SQL:
SELECT
    CASE WHEN c.tier = 'vip' THEN 'VIP' ELSE '普通' END AS 客户类型,
    ROUND(AVG(total), 2) AS 平均消费额
FROM (
    SELECT c.tier, SUM(o.total_amount) as total
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.id, c.tier
) sub
GROUP BY 客户类型;

### 示例 3
问题: 哪些产品的库存低于平均库存水平？
SQL:
SELECT name, category, stock
FROM products
WHERE stock < (SELECT AVG(stock) FROM products)
ORDER BY stock ASC;
"""

TEXT_TO_SQL_SYSTEM_WITH_EXAMPLES = TEXT_TO_SQL_SYSTEM + "\n\n" + FEW_SHOT_EXAMPLES
```

把 Few-Shot 示例加入 system prompt 后重新创建 Agent，SQL 生成的准确率会有显著提升——特别是在以下场景：
- 特定的日期格式约定（SQLite 用 `date('now', '-30 days')` vs PostgreSQL 用 `NOW() - INTERVAL '30 days'`）
- 复杂的子查询模式
- CASE WHEN 条件聚合
- 业务字段的特殊含义（如 `status` 的枚举值）

## 查询结果的解释与格式化

原始 SQL 返回的数据是元组列表，对用户来说不友好。Agent 的最后一步是把数据转化为人类可读的形式：

### 自动表格化

```python
def format_query_result(raw_result: str) -> str:
    lines = raw_result.strip().split("\n")
    data_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("(("):
            import ast
            try:
                data = ast.literal_eval(line.replace(",)", ")"))
                data_lines.append(data)
            except Exception:
                pass

    if not data_lines:
        return raw_result

    if len(data_lines) == 1 and len(data_lines[0]) == 1:
        val = data_lines[0][0]
        return str(val)

    headers = None
    rows = []
    for row in data_lines:
        if isinstance(row[0], tuple):
            if headers is None:
                headers = [f"列{i+1}" for i in range(len(row))]
            rows.append(list(row))
        else:
            rows.append(list(row))

    col_widths = []
    for col_idx in range(len(rows[0])):
        max_len = max(len(str(r[col_idx])) for r in rows)
        col_widths.append(max(max_len, 10))

    output = ""
    if headers:
        header_row = " | ".join(h.center(col_widths[i]) for i, h in enumerate(headers))
        output += header_row + "\n"
        output += "-" * len(header_row) + "\n"

    for row in rows:
        row_str = " | ".join(str(v).center(col_widths[i]) for i, v in enumerate(row))
        output += row_str + "\n"

    return output
```

测试一下效果：

```
原始结果: ((('华东', 285430.0),), (('华北', 198200.0),), (('华南', 156780.0),))

格式化后:
  地区     |  销售额    
-----------|------------
 华东      | 285430.0  
 华北      | 198200.0  
 华南      | 156780.0  
```

### 自然语言总结

更进一步的增强：让 LLM 不仅展示数据，还给出**洞察性总结**：

```python
SUMMARIZE_PROMPT = """根据以下 SQL 查询结果，用中文写一段简洁的分析总结。
要求数据准确，突出关键发现，如果有异常值或值得关注的趋势请指出。

用户问题: {question}
查询结果:
{formatted_result}"""

summarize_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "你是一个数据分析师，擅长从数字中发现洞察。"),
        ("human", SUMMARIZE_PROMPT),
    ])
    | get_llm()
    | StrOutputParser()
)
```

当用户问"各地区的销售情况"时，最终输出会变成：

```
📊 各地区销售情况（2025年Q1）

| 地区 | 销售额(¥) | 占比 | 订单数 |
|------|----------|------|--------|
| 华东 | 285,430 | 38.2% | 89 |
| 华北 | 198,200 | 26.5% | 67 |
| 华南 | 156,780 | 21.0% | 54 |
| 西南 | 68,450  | 9.2%  | 23 |
| 华中 | 39,120  | 5.2%  | 14 |

📌 关键发现:
• **华东地区以 ¥285,430 领跑**，贡献了超过三分之一的销售额，主要来自上海和杭州两个城市的 VIP 客户
• **西南地区虽然订单数最少，但客单价（¥2,976）位居第二**，说明该区域客户购买力强
• 五个地区的销售分布呈现明显的阶梯状，华东>华北>华南>西南>华中
```

这就是从"能查数据"到"能讲出洞察"的关键一步。下一节我们将在此基础上增加**可视化图表**能力，让数据不仅能被阅读，还能被看见。
