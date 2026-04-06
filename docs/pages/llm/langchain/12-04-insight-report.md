---
title: 输出洞察报告
description: 多维度数据分析报告生成、Agent 自主规划分析路径、完整 DataAnalysisAgent 系统组装、CLI 与 FastAPI 部署
---
# 输出洞察报告

前三节我们实现了数据分析 Agent 的三大核心能力：**连接数据库**（12.1）、**Text-to-SQL 查询**（12.2）、**数据可视化**（12.3）。现在是时候把它们整合起来，并加上最后一个关键能力——**自动生成结构化的数据洞察报告**。

## 从"回答问题"到"产出报告"

前面三节的能力都是"问答式"的：用户问一个问题，Agent 回答一个答案（或一张图）。但在真实的业务场景中，用户的需求往往更宏观：

> "帮我做一份 Q1 销售分析报告"

这不是一个 SQL 查询能解决的问题。一份好的分析报告需要：
1. **整体概览**：总销售额、订单量、同比/环比变化
2. **维度拆解**：按地区、品类、客户等级、时间趋势
3. **异常发现**：哪些指标异常？可能的原因是什么？
4. **行动建议**：基于数据给出可执行的建议

这需要一个能**自主规划和执行多步分析任务**的 Agent——它不是被动地一次回答一个查询，而是主动地设计一套完整的分析方案并逐步执行。

## 报告生成架构

```
用户需求: "做一份 Q1 销售分析报告"
         │
    ▼ Agent 规划阶段
Thought: 一份完整的销售报告需要以下维度:
       1) 整体业绩概览 (总销售额、订单数、客单价)
       2) 地区维度分析 (各区域销售排名)
       3) 品类维度分析 (哪些品类卖得好)
       4) 时间趋势 (月度/周度变化)
       5) Top 产品排行
       6) 关键发现与建议
       
       我将依次执行这些分析，最后汇总成报告。
         │
    ▼ 执行阶段（多轮工具调用）
Step 1: SQL → 总体指标 → 格式化数据
Step 2: SQL → 地区分布 → 生成柱状图
Step 3: SQL → 品类分布 → 生成饼图
Step 4: SQL → 月度趋势 → 生成折线图
Step 5: SQL → Top10产品 → 生成水平柱状图
         │
    ▼ 汇总阶段
LLM 综合所有数据和图表 → 生成 Markdown 报告
```

## 实现报告生成器

### 报告模板设计

```python
from datetime import datetime
import json

class ReportTemplate:
    @staticmethod
    def sales_analysis_report() -> dict:
        return {
            "title": "销售数据分析报告",
            "sections": [
                {
                    "id": "overview",
                    "title": "一、整体业绩概览",
                    "queries": [
                        {
                            "sql": ("SELECT COUNT(DISTINCT id) as total_orders, "
                                   "ROUND(SUM(total_amount),2) as total_revenue, "
                                   "ROUND(AVG(total_amount),2) as avg_order_value, "
                                   "COUNT(DISTINCT customer_id) as unique_customers "
                                   "FROM orders WHERE status='completed'"),
                            "label": "核心指标",
                            "chart": None,
                        },
                        {
                            "sql": ("SELECT "
                                   "COUNT(CASE WHEN status='completed' THEN 1 END) as completed, "
                                   "COUNT(CASE WHEN status='shipped' THEN 1 END) as shipped, "
                                   "COUNT(CASE WHEN status='cancelled' THEN 1 END) as cancelled, "
                                   "COUNT(CASE WHEN status='returned' THEN 1 END) as returned "
                                   "FROM orders"),
                            "label": "订单状态分布",
                            "chart": "pie",
                        },
                    ],
                },
                {
                    "id": "regional",
                    "title": "二、地区维度分析",
                    "queries": [
                        {
                            "sql": ("SELECT c.region, "
                                   "ROUND(SUM(o.total_amount),2) as sales, "
                                   "COUNT(o.id) as orders, "
                                   "ROUND(AVG(o.total_amount),2) as avg_order, "
                                   "COUNT(DISTINCT o.customer_id) as customers "
                                   "FROM orders o "
                                   "JOIN customers c ON o.customer_id=c.id "
                                   "WHERE o.status='completed' "
                                   "GROUP BY c.region ORDER BY sales DESC"),
                            "label": "各地区销售对比",
                            "chart": "bar",
                        },
                    ],
                },
                {
                    "id": "category",
                    "title": "三、品类维度分析",
                    "queries": [
                        {
                            "sql": ("SELECT p.category, "
                                   "ROUND(SUM(o.total_amount),2) as revenue, "
                                   "COUNT(o.id) as units_sold, "
                                   "ROUND(SUM(o.total_amount)/COUNT(o.id),2) as avg_price, "
                                   "ROUND((SUM(o.total_amount) / "
                                   "(SELECT SUM(total_amount) FROM orders WHERE status='completed')) * 100, 1) "
                                   "as pct_of_total "
                                   "FROM orders o "
                                   "JOIN products p ON o.product_id=p.id "
                                   "WHERE o.status='completed' "
                                   "GROUP BY p.category ORDER BY revenue DESC"),
                            "label": "品类销售表现",
                            "chart": "horizontal_bar",
                        },
                        {
                            "sql": ("SELECT p.name, p.category, "
                                   "SUM(o.quantity) as units, "
                                   "ROUND(SUM(o.total_amount),2) as revenue "
                                   "FROM orders o JOIN products p ON o.product_id=p.id "
                                   "WHERE o.status='completed' "
                                   "GROUP BY p.name ORDER BY revenue DESC LIMIT 8"),
                            "label": "Top 8 畅销产品",
                            "chart": "bar",
                        },
                    ],
                },
                {
                    "id": "trend",
                    "title": "四、时间趋势分析",
                    "queries": [
                        {
                            "sql": ("SELECT strftime('%Y-%m', order_date) as month, "
                                   "COUNT(*) as orders, "
                                   "ROUND(SUM(total_amount),2) as revenue, "
                                   "ROUND(AVG(total_amount),2) as aov "
                                   "FROM orders WHERE status='completed' "
                                   "GROUP BY month ORDER BY month"),
                            "label": "月度销售趋势",
                            "chart": "line",
                        },
                    ],
                },
                {
                    "id": "insights",
                    "title": "五、关键发现与建议",
                    "queries": [],
                    "analysis_only": True,
                },
            ],
        }
```

### 报告执行引擎

```python
class ReportEngine:
    def __init__(self, db, output_dir: str = "./reports"):
        self.db = db
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/charts", exist_ok=True)

    def execute_report(self, template: dict) -> dict:
        report_data = {"title": template["title"], "sections": []}

        for section in template["sections"]:
            section_result = {
                "title": section["title"],
                "data_items": [],
                "charts": [],
            }

            if not section.get("analysis_only"):
                for q in section["queries"]:
                    item = self._execute_query(q, section["id"])
                    section_result["data_items"].append(item)

                    if q.get("chart") and item.get("df") is not None and len(item["df"]) > 0:
                        chart_path = self._generate_chart_for_item(item, q["chart"])
                        if chart_path:
                            section_result["charts"].append({
                                "path": chart_path,
                                "type": q["chart"],
                                "label": q["label"],
                            })

            report_data["sections"].append(section_result)

        return report_data

    def _execute_query(self, query_config: dict, section_id: str) -> dict:
        try:
            df = sql_to_dataframe(query_config["sql"])
            return {
                "label": query_config["label"],
                "df": df,
                "raw": df.to_dict(orient="records") if not df.empty else [],
                "shape": df.shape if not df.empty else (0, 0),
            }
        except Exception as e:
            return {"label": query_config["label"], "error": str(e), "df": None}

    def _generate_chart_for_item(self, item: dict, chart_type: str) -> str:
        df = item.get("df")
        if df is None or df.empty:
            return None

        config = chart_selection_chain.invoke({
            "df_info": get_df_info(df),
            "question": item["label"],
        })

        chart_path = f"{self.output_dir}/charts/{item['label']}_{uuid.uuid4().hex[:6]}.png"
        plot_code = generate_plotting_code(config, df, chart_path)

        try:
            exec(plot_code, {"df": df})
            return chart_path
        except Exception:
            return None


engine = ReportEngine(db=db_enhanced)
template = ReportTemplate.sales_analysis_report()
report_data = engine.execute_report(template)
print(f"报告生成完成，共 {len(report_data['sections'])} 个章节")
for sec in report_data["sections"]:
    charts_count = len(sec.get("charts", []))
    data_count = len(sec.get("data_items", []))
    print(f"  {sec['title']}: {data_count} 组数据, {charts_count} 张图表")
```

输出：

```
报告生成完成，共 5 个章节
  一、整体业绩概览: 2 组数据, 1 张图表
  二、地区维度分析: 1 组数据, 1 张图表
  三、品类维度分析: 2 组数据, 2 张图表
  四、时间趋势分析: 1 组数据, 1 张图表
  五、关键发现与建议: 0 组数据, 0 张图表
```

### LLM 驱动的洞察撰写

有了结构化数据之后，最后一步是让 LLM 把数字变成有洞见的文字：

```python
REPORT_WRITER_PROMPT = """你是一位资深的数据分析师和商业顾问。
请根据以下结构化的数据分析结果，撰写一份专业、清晰、有洞察力的 Markdown 分析报告。

## 要求
1. 用中文撰写
2. 数据必须准确引用提供的数值，不要编造
3. 每个章节先呈现关键数据，再给出你的分析和解读
4. 在"关键发现与建议"章节中，至少提出 3 条可执行的 actionable 建议
5. 如果发现异常数据或值得注意的趋势，要明确指出
6. 使用表格展示对比数据
7. 语气专业但不晦涩，适合向管理层汇报

## 报告数据
{report_json}

现在开始撰写报告:"""

def generate_final_report(report_data: dict) -> str:
    report_json = json.dumps(report_data, ensure_ascii=False, default=str, indent=2)

    writer_chain = (
        ChatPromptTemplate.from_messages([
            ("system", REPORT_WRITER_PROMPT),
            ("human", "{report_json}"),
        ])
        | get_llm(temperature=0.3)
        | StrOutputParser()
    )

    report_md = writer_chain.invoke({"report_json": report_json})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{engine.output_dir}/sales_report_{timestamp}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n📄 报告已保存至: {report_path}")
    return report_md


final_report = generate_final_report(report_data)
print(final_report[:2000])
```

生成的报告示例：

```markdown
# 销售数据分析报告

> **生成时间**: 2025-04-06 | **数据范围**: 2025年Q1 (1月-3月)
> **数据来源**: 电商数据库 | **分析引擎**: AI Data Analyst v1.0

---

## 一、整体业绩概览

### 核心经营指标

| 指标 | 数值 |
|------|------|
| 总订单数 | 423 笔 |
| 总销售额 | ¥748,980.00 |
| 平均客单价 (AOV) | ¥1,770.55 |
| 购买客户数 | 8 人 |

### 订单状态分布

| 状态 | 数量 | 占比 |
|------|------|------|
| 已完成 | 423 | 84.6% |
| 已发货 | 52 | 10.4% |
| 已取消 | 16 | 3.2% |
| 已退回 | 9 | 1.8% |

**📊 [图表: 订单状态饼图](./charts/订单状态饼图_abc123.png)**

> **初步观察**: 订单完成率达到 84.6%，处于健康水平。
> 取消率 3.2% 和退货率 1.8% 均在可控范围内，
> 但仍有优化空间。

---

## 二、地区维度分析

| 地区 | 销售额(¥) | 订单数 | 客单价 | 客户数 |
|------|----------|--------|--------|--------|
| 华东 | 285,430 | 89 | 3,207 | 3 |
| 华北 | 198,200 | 67 | 2,957 | 2 |
| 华南 | 156,780 | 54 | 2,903 | 2 |
| 西南 | 68,450 | 23 | 2,976 | 1 |
| 华中 | 39,120 | 14 | 2,794 | 1 |

**📊 [图表: 各地区销售额柱状图](./charts/各地区销售对比_def456.png)**

> **关键发现**:
> 1. **华东地区以 ¥285,430 领跑**，贡献了总销售额的 38.1%，
>    主要得益于上海和杭州两个城市的 VIP 客户高频购买
> 2. **西南地区虽然订单最少（23笔），但客单价 ¥2,976 高居第一**，
>    说明该区域客户购买力强但渗透不足，是潜在的增长点
> 3. 五个地区的销售呈明显的阶梯状分布，头部集中效应明显

---

## 五、关键发现与建议

### 🎯 可执行建议

**建议 1: 重点突破西南市场**
- 西南地区客单价最高（¥2,976）但客户数仅 1 人
- 行动：针对成都/重庆市场开展定向营销，目标 Q2 新增 3-5 名活跃客户
- 预期收益：按当前客单价估算，每新增 1 名客户 ≈ ¥9,000/季

**建议 2: 电子产品线优化**
- 电子产品占总销售的 57.8%，过度依赖单一品类风险较高
- 行动：提升家居和运动品类的曝光和促销力度
- 目标：将电子产品占比从 58% 降至 50% 以下

**建议 3: 降低订单取消率**
- 当前取消率 3.2%（16/500），主要原因为库存不足和价格变动
- 行动：优化库存预警机制，对热销产品设置安全库存线
- 目标：Q2 将取消率降至 2% 以下
```

## 完整系统组装：DataAnalysisAgent

把所有能力封装为统一的入口类：

```python
class DataAnalysisAgent:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.db = EnhancedSQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.agent = None
        self.engine = None
        self._initialized = False

    def initialize(self):
        db_conn = connect_database(self.db_path)
        safe_tool = QuerySQLDataBaseTool(
            db=self.db, name="sql_db_query",
            description="执行只读 SQL 查询"
        )
        schema_tool = InfoSQLDatabaseTool(db=self.db)
        list_tool = ListSQLDatabaseTool(db=self.db)
        all_tools = [list_tool, schema_tool, safe_tool, generate_chart]

        self.agent = create_react_agent(
            llm=get_llm(),
            tools=all_tools,
            prompt=TEXT_TO_SQL_SYSTEM_WITH_EXAMPLES + CHART_AGENT_ADDITION,
        )

        self.engine = ReportEngine(db=self.db)
        self._initialized = True
        return self

    def ask(self, question: str) -> str:
        result = self.agent.invoke({"input": question})
        return result["output"]

    def visualize(self, question: str) -> dict:
        result = self.agent.invoke({"input": f"画图: {question}"})
        return {"response": result["output"]}

    def generate_report(self, report_type: str = "sales") -> str:
        templates = {"sales": ReportTemplate.sales_analysis_report}
        template = templates.get(report_type)
        if not template:
            return f"不支持的报告类型: {report_type}"

        data = self.engine.execute_report(template)
        return generate_final_report(data)

    def chat(self, message: str, history: list = None) -> str:
        if any(kw in message.lower() for kw in ["报告", "report", "分析报告"]):
            return self.generate_report("sales")

        if any(kw in message.lower() for kw in ["画", "图", "chart", "可视化"]):
            return self.visualize(message)

        return self.ask(message)


def run_cli():
    print("=" * 60)
    print("   📊 数据分析助手 (Data Analysis Agent)")
    print("   输入自然语言问题即可查询数据库")
    print("   特殊命令:")
    print("   /report   — 生成完整分析报告")
    print("   /chart    — 强制生成图表")
    print("   /tables   — 查看所有表")
    print("   /quit     — 退出")
    print("=" * 60)

    agent = DataAnalysisAgent()
    agent.initialize()

    while True:
        try:
            user_input = input("\n📊 > ").strip()
            if user_input.lower() in ("quit", "/quit", "exit"):
                break
            if user_input == "/tables":
                print(f"表: {agent.db.get_usable_table_names()}")
                continue
            if user_input == "/report":
                print("\n🔄 正在生成分析报告，请稍候...")
                report = agent.generate_report("sales")
                print(f"\n{report[:1500]}...")
                continue

            response = agent.chat(user_input)
            print(f"\n{response}")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    run_cli()
```

### CLI 交互演示

```
============================================================
   📊 数据分析助手 (Data Analysis Agent)
============================================================

📊 > 我们总共有多少客户？

根据数据库中的信息，目前共有 **8 位注册客户**。

客户分布如下：
| 大区 | 客户数 | VIP 数 |
|------|-------|--------|
| 华东 | 3 | 1 |
| 华北 | 2 | 0 |
| 华南 | 2 | 1 |
| 西南 | 1 | 1 |
| 华中 | 1 | 0 |

📊 > 帮我看看上个月的销售趋势，画个图

🔍 正在查询数据...

✅ 图表已生成
📊 类型: line
📁 ./charts/chart_e7f2a91b.png
📝 标题: 月度销售趋势（2025年Q1）

从折线图可以看出：
• 1月份销售额起步于约 ¥180K
• 2月份有明显增长，达到峰值约 ¥290K
• 3月份略有回落至约 ¥278K
• 整体呈上升趋势，Q2 有望继续增长

📊 > 帮我生成一份完整的销售分析报告

🔄 正在生成分析报告，请稍候...

📄 报告已保存至: ./reports/sales_report_20250406_143022.md

# 销售数据分析报告
> **生成时间**: 2025-04-06 ...
[完整报告内容...]
```

## 项目总结

本章构建了一个端到端的**数据分析 Agent 系统**，能力矩阵如下：

| 能力 | 实现方式 | 章节 |
|------|---------|------|
| 数据库连接与 Schema 管理 | `SQLDatabase` + `EnhancedSQLDatabase` | 12.1 |
| 自然语言转 SQL | ReAct Agent + Few-Shot 示例 | 12.2 |
| 查询结果解释与格式化 | 自动表格化 + 自然语言总结 Chain | 12.2 |
| 智能图表生成 | LLM 选图类型 + Matplotlib 绑定代码执行 | 12.3 |
| 多维分析报告 | `ReportEngine` + LLM 洞察撰写 | 12.4 |

这个项目展示了 LangChain 在数据分析领域的完整应用链路：**从非结构化的用户需求出发，经过 Schema 理解、SQL 生成、安全执行、数据处理、智能选图、代码生成执行，最终产出包含数据表格、可视化图表和文字洞察的结构化报告**。整个过程由 Agent 自主规划和驱动，人类只需要用自然语言描述需求。
