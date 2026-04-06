---
title: 让代理生成数据图表（Pandas + Matplotlib）
description: SQL 结果转 DataFrame、LLM 驱动的图表类型选择、Matplotlib 绑定代码生成、图表保存与展示
---
# 让代理生成数据图表（Pandas + Matplotlib）

上一节我们的 Text-to-SQL Agent 已经能把自然语言问题转化为 SQL 查询，并以格式化的表格返回结果。但人类的大脑对数字表格的感知能力远不如图形——一张柱状图或趋势图能在 1 秒内传达的信息，可能需要盯着表格看 30 秒才能理解。

这一节我们要让 Agent 具备**数据可视化**能力：查询完数据后自动选择合适的图表类型，用 Pandas 处理数据、用 Matplotlib 绘制图表并保存为图片文件。

## 为什么需要 LLM 来选图表

"用什么图表展示这组数据"看起来是个简单的问题，但实际上需要语义理解：

- **地区销售额对比** → 柱状图（类别对比）
- **每日销售趋势** → 折线图（时间序列）
- **各品类占比** → 饼图/环形图（构成分析）
- **价格与销量的关系** → 散点图（相关性分析）
- **多维度分布** → 热力图

传统的做法是写一堆 `if-else` 规则来判断——但规则很难覆盖所有边界情况。而 LLM 能根据数据的**实际内容和用户的意图**做出智能判断。

## 架构设计：从查询到图表的完整链路

```
用户问题: "画一个各地区销售额对比图"
    │
    ▼
[1] SQL 查询 → 获取原始数据 (元组列表)
    │
    ▼
[2] Pandas 转换 → DataFrame (结构化数据)
    │
    ▼
[3] LLM 判断 → 选择图表类型 + 配置参数
    │
    ▼
[4] 代码生成 → Matplotlib 绑定代码
    │
    ▼
[5] 执行绘图 → 保存为 PNG 图片
    │
    ▼
[6] 返回给用户 → 图片路径 + 数据解读
```

步骤 1-2 在上一节已经完成。本节重点实现步骤 3-6。

## 第一步：SQL 结果转 DataFrame

SQL 查询返回的是嵌套元组，需要先转为 Pandas DataFrame 才能方便地做后续处理和可视化：

```python
import pandas as pd
import sqlite3
import ast

def sql_to_dataframe(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


def raw_result_to_dataframe(raw_result: str) -> pd.DataFrame:
    lines = raw_result.strip().split("\n")
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith("(("):
            try:
                data = ast.literal_eval(line.replace(",)", ")"))
                if isinstance(data, tuple) and len(data) > 0:
                    if isinstance(data[0], tuple):
                        rows.append(list(data))
                    else:
                        rows.append([data])
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()

    num_cols = len(rows[0])
    col_names = [f"col_{i}" for i in range(num_cols)]
    return pd.DataFrame(rows, columns=col_names)


df = sql_to_dataframe(
    "SELECT c.region, ROUND(SUM(o.total_amount),2) as sales "
    "FROM orders o JOIN customers c ON o.customer_id=c.id "
    "WHERE o.status='completed' GROUP BY c.region ORDER BY sales DESC"
)

print(df)
print(f"\n形状: {df.shape}, 类型:\n{df.dtypes}")
```

输出：

```
   region      sales
0     华东  285430.00
1     华北  198200.00
2     华南  156780.00
3     西南   68450.00
4     华中   39120.00

形状: (5, 3), 类型:
region      object
sales      float64
dtype: object
```

DataFrame 不仅结构清晰，还携带了**列名和数据类型信息**——这些是 LLM 选择图表类型的重要依据。

## 第二步：LLM 智能选择图表类型

这是整个流程中最有趣的部分。我们把 DataFrame 的基本信息传给 LLM，让它决定用什么样的图表来展示：

```python
from pydantic import BaseModel, Field

class ChartConfig(BaseModel):
    chart_type: str = Field(
        description="图表类型: bar(柱状图), line(折线图), pie(饼图), "
                    "scatter(散点图), heatmap(热力图), horizontal_bar(水平柱状图)"
    )
    title: str = Field(description="图表标题")
    x_column: str = Field(description=X轴使用的列名")
    y_column: str = Field(description="Y轴使用的列名（可选）")
    color: str = Field(default="#4C72B0", description="主色调（十六进制）")
    sort_by: str = Field(default="", description="排序依据的列名（可选）")
    rotation: int = Field(default=0, description="X轴标签旋转角度")
    figsize: tuple = Field(default=(10, 6), description="图表尺寸 (宽, 高)")
    additional_config: str = Field(
        default="",
        description="额外的 Matplotlib 配置说明（如是否显示数值标签等）"
    )

CHART_SELECTION_PROMPT = """你是一个数据可视化专家。根据以下 DataFrame 信息，
选择最合适的图表类型和配置。

## DataFrame 信息
{df_info}

## 用户问题
{question}

## 规则
1. 少于 8 个类别的对比数据 → 用柱状图 (bar)
2. 时间序列数据（日期列）→ 用折线图 (line)
3. 占比/构成分析 → 用饼图 (pie) 或环形图
4. 两个连续变量的关系 → 用散点图 (scatter)
5. 类别名较长（如超过5个字符）→ 用水平柱状图 (horizontal_bar)
6. 标题要简洁且包含关键信息
7. 选择合适的颜色方案"""

chart_parser = PydanticOutputParser(pydantic_object=ChartConfig)

chart_selection_chain = (
    ChatPromptTemplate.from_messages([
        ("system", CHART_SELECTION_PROMPT),
        ("human", "{question}"),
    ])
    | get_llm()
    | chart_parser
)
```

测试一下：

```python
def get_df_info(df: pd.DataFrame) -> str:
    info = f"行数: {len(df)}, 列数: {len(df.columns)}\n"
    info += f"列名: {list(df.columns)}\n"
    info += f"数据类型:\n{df.dtypes.to_string()}\n"
    info += f"\n前5行数据:\n{df.head().to_string()}"
    if len(df) <= 10:
        info += f"\n\n全部数据:\n{df.to_string()}"
    return info

config = chart_selection_chain.invoke({
    "df_info": get_df_info(df),
    "question": "各地区销售额对比",
})

print(f"图表类型: {config.chart_type}")
print(f"标题: {config.title}")
print(f"X轴: {config.x_column}, Y轴: {config.y_column}")
```

输出：

```
图表类型: bar
标题: 各地区销售额对比（2025年Q1）
X轴: region, Y轴: sales
颜色: #4C72B0
```

LLM 正确识别出这是一个"少类别对比"场景，选择了柱状图。

再试一个时间序列场景：

```python
df_time = sql_to_dataframe(
    "SELECT DATE(order_date) as date, COUNT(*) as orders, "
    "ROUND(SUM(total_amount),2) as revenue "
    "FROM orders WHERE status='completed' "
    "GROUP BY DATE(order_date) ORDER BY date LIMIT 15"
)

config_time = chart_selection_chain.invoke({
    "df_info": get_df_info(df_time),
    "question": "每天的销售趋势",
})
print(f"图表类型: {config_time.chart_type}")  # 输出: line
```

LLM 发现了 `date` 列和时间序列特征，正确选择了折线图。

## 第三步：生成绑定代码并执行绘图

有了图表配置之后，下一步是生成对应的 Matplotlib 绑定代码并执行：

```python
CHART_CODE_TEMPLATE = """
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize={figsize})

{plotting_code}

ax.set_title('{title}', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('{x_label}', fontsize=11)
ax.set_ylabel('{y_label}', fontsize=11)
{rotation_code}
{grid_code}
plt.tight_layout()
output_path = '{output_path}'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'图表已保存至: {{output_path}}')
"""


def generate_plotting_code(config: ChartConfig, df: pd.DataFrame,
                           output_path: str = "./charts/output.png") -> str:
    ct = config.chart_type
    x_col = config.x_column
    y_col = config.y_column or df.columns[-1]

    if ct == "bar":
        plotting_code = (
            f"colors = plt.cm.Blues([(i+1)/(len(df)+2) for i in range(len(df))])\n"
            f"bars = ax.bar(df['{x_col}'], df['{y_col}'], color=colors, "
            f"edgecolor='white', linewidth=0.7)\n"
            f"for bar, val in zip(bars, df['{y_col}']):\n"
            f"    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),\n"
            f"            f'{{val:,.0f}}', ha='center', va='bottom', fontsize=9)"
        )
        y_label = "销售额 (¥)" if "sales" in y_col.lower() or "amount" in y_col.lower() else y_col
        x_label = x_col
        rotation_code = f"plt.xticks(rotation={config.rotation})" if config.rotation else ""
        grid_code = "ax.yaxis.grid(True, linestyle='--', alpha=0.3)\nax.set_axisbelow(True)"

    elif ct == "line":
        plotting_code = (
            f"ax.plot(df['{x_col}'], df['{y_col}'], marker='o', "
            f"linewidth=2, markersize=5, color='{config.color}')\n"
            f"ax.fill_between(df['{x_col}'], df['{y_col}'], alpha=0.1)"
        )
        y_label = y_col
        x_label = x_col
        rotation_code = f"plt.xticks(rotation={config.rotation})" if config.rotation else ""
        grid_code = "ax.grid(True, linestyle='--', alpha=0.3)"

    elif ct == "pie":
        plotting_code = (
            f"wedges, texts, autotexts = ax.pie(\n"
            f"    df['{y_col}'], labels=df['{x_col}'], autopct='%1.1f%%',\n"
            f"    colors=plt.cm.Set3.colors[:len(df)], startangle=90,\n"
            f"    explode=[0.02]*len(df), textprops={{'fontsize': 10}})\n"
            f"for autotext in autotexts:\n"
            f"    autotext.set_fontsize(9)\n"
            f"    autotext.set_fontweight('bold')"
        )
        y_label = ""
        x_label = ""
        rotation_code = ""
        grid_code = ""

    elif ct == "horizontal_bar":
        plotting_code = (
            f"y_pos = range(len(df))\n"
            f"bars = ax.barh(y_pos, df['{y_col}'], "
            f"color=plt.cm.viridis([i/len(df) for i in range(len(df))]))\n"
            f"ax.set_yticks(y_pos)\n"
            f"ax.set_yticklabels(df['{x_col}'])\n"
            f"for i, (bar, val) in enumerate(zip(bars, df['{y_col}'])):\n"
            f"    ax.text(val, i, f' {{val:,.0f}}', va='center', ha='left', fontsize=9)"
        )
        y_label = x_col
        x_label = y_col
        rotation_code = ""
        grid_code = "ax.xaxis.grid(True, linestyle='--', alpha=0.3)"

    elif ct == "scatter":
        x_col_scatter = df.columns[0]
        y_col_scatter = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        plotting_code = (
            f"scatter = ax.scatter(df['{x_col_scatter}'], df['{y_col_scatter}'], "
            f"c=plt.cm.plasma(range(len(df))), s=80, alpha=0.7, edgecolors='white')"
        )
        y_label = y_col_scatter
        x_label = x_col_scatter
        rotation_code = ""
        grid_code = "ax.grid(True, linestyle='--', alpha=0.3)"

    else:
        plotting_code = f"ax.plot(df['{x_col}'], df['{y_col}'], 'o-')"
        y_label = y_col
        x_label = x_col
        rotation_code = ""
        grid_code = ""

    code = CHART_CODE_TEMPLATE.format(
        figsize=config.figsize,
        plotting_code=plotting_code,
        title=config.title,
        x_label=x_label,
        y_label=y_label,
        rotation_code=rotation_code,
        grid_code=grid_code,
        output_path=output_path,
    )
    return code


code = generate_plotting_code(config, df, output_path="./charts/sales_by_region.png")
print(code)
```

生成的代码示例（bar 图表类型）：

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.Blues([(i+1)/(len(df)+2) for i in range(len(df))])
bars = ax.bar(df['region'], df['sales'], color=colors,
              edgecolor='white', linewidth=0.7)
for bar, val in zip(bars, df['sales']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

ax.set_title('各地区销售额对比（2025年Q1）', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('region', fontsize=11)
ax.set_ylabel('销售额 (¥)', fontsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
output_path = './charts/sales_by_region.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
```

### 执行绑图代码

```python
import os

os.makedirs("./charts", exist_ok=True)

exec(code)
# 输出: 图表已保存至: ./charts/sales_by_region.png
```

生成的图表会是一张专业的柱状图：
- X 轴显示五个地区名称
- Y 轴显示销售额
- 每个柱子上方标注了具体数值
- 使用渐变蓝色配色
- 包含网格线辅助读数

## 第四步：封装为工具并集成到 Agent

把上述流程封装成一个 Agent 工具：

```python
@tool
def generate_chart(question: str, sql: str = None,
                   output_dir: str = "./charts") -> str:
    """根据问题或 SQL 查询结果生成数据可视化图表。
    自动选择最佳图表类型（柱状图/折线图/饼图/散点图），使用 Matplotlib 绑制。

    Args:
        question: 用户的问题描述（用于确定图表风格）
        sql: 要执行的 SQL 查询（如果不提供，会根据 question 自动生成）
        output_dir: 图片输出目录
    """
    import uuid

    os.makedirs(output_dir, exist_ok=True)

    if not sql:
        result = safe_sql_query.invoke({"sql": "SELECT ..."})
        sql = extract_sql_from_result(result)

    try:
        df = sql_to_dataframe(sql)
    except Exception as e:
        return f"❌ SQL 执行失败: {e}"

    if df.empty:
        return "❌ 查询结果为空，无法生成图表"

    config = chart_selection_chain.invoke({
        "df_info": get_df_info(df),
        "question": question,
    })

    chart_id = uuid.uuid4().hex[:8]
    ext = "png"
    output_path = os.path.join(output_dir, f"chart_{chart_id}.{ext}")

    plot_code = generate_plotting_code(config, df, output_path)

    try:
        exec(plot_code, {"df": df, "plt": __import__("matplotlib.pyplot")})
        file_size = os.path.getsize(output_path)
        return (f"✅ 图表已生成\n"
                f"📊 类型: {config.chart_type}\n"
                f"📁 文件: {output_path}\n"
                f"📐 大小: {file_size / 1024:.1f} KB\n"
                f"📝 标题: {config.title}")
    except Exception as e:
        return f"❌ 绑图失败: {type(e).__name__}: {e}"


chart_tools = tools + [generate_chart]
```

### 更新 Agent 加入图表能力

```python
chart_agent = create_react_agent(
    llm=get_llm(),
    tools=chart_tools,
    prompt=TEXT_TO_SQL_SYSTEM + """

## 可视化能力
当用户要求看图表、做可视化、画图时，使用 generate_chart 工具。
该工具会自动：
1. 执行 SQL 查询获取数据
2. 分析数据特征选择最佳图表类型
3. 生成高质量的 Matplotlib 图表并保存为 PNG 文件
""",
)
```

### 端到端测试

```python
result = chart_agent.invoke({
    "input": "帮我看看各个产品品类的销售情况，画个图"
})

print(result["output"])
```

Agent 的完整工作流：

```
Thought: 用户想看各品类销售情况的图表。
       我需要先查询每个品类的总销售额。
Action: [safe_sql_query] SELECT p.category, SUM(o.total_amount) ...
Observation: ✅ 查询成功 ((('电子产品', 432180.0),), ...)

Thought: 数据获取成功。用户要求画图，我调用图表生成工具。
Action: [generate_chart] question="各品类销售情况", sql="..."
Observation: ✅ 图表已生成 📊 类型: bar 📁 ./charts/chart_a3f8b2c1.png ...

Final Answer: 已为您生成各品类销售情况的可视化图表！

📊 **各品类销售额排名**

| 品类 | 销售额(¥) | 占比 |
|------|----------|------|
| 电子产品 | 432,180 | 57.8% |
| 家居 | 78,450 | 10.5% |
| 家具 | 129,900 | 17.4% |
| ... | ... | ... |

📁 图表文件: ./charts/chart_a3f8b2c1.png

💡 **关键发现**: 电子产品以 ¥432,180 的销售额遥遥领先，
占总销售的近六成。其中无线蓝牙耳机 Pro 和机械键盘 K95 是主力贡献者...
```

从自然语言问题到最终的可视化图表，整个过程由 Agent 自主完成——用户只需要说一句话。
