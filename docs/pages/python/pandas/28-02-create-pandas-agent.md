# 创建 Pandas DataFrame Agent


#### Agent 的核心原理

LangChain 的 `create_pandas_dataframe_agent` 是一个开箱即用的 Agent，它的工作流程是：

```
用户问题 (自然语言)
    ↓
LLM 理解意图 → 生成 Pandas 代码
    ↓
在沙箱中执行代码 → 获取结果
    ↓
LLM 将结果翻译为自然语言回答
    ↓
返回给用户
```

关键点：**LLM 不直接操作数据，它只负责"理解意图→写代码→解释结果"**。实际的数据操作由 Python 执行。

---

#### 基础 Agent 创建

##### 最简方式：一行代码

```python
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df_eval,
    verbose=True,
    allow_dangerous_code=True,
    agent_type="tool-calling",
)

response = agent.invoke("这个数据集有多少行？每列都是什么类型？")
print(response["output"])
```

输出：
```
> Entering new AgentExecutor chain...
Invoking: `python_repl_ast` with `df.shape` and `df.dtypes`

(500, 11)
model               object
task                 object
score              float64
...
<final answer>
这个数据集有 **500 行、11 列**。
列包括: model（模型名称）、task（任务类型）、score（得分）、latency_ms（延迟毫秒数）、tokens_used（token用量）、category（难度）、language（语言）、timestamp（时间戳）、cost_usd（成本）、quality_grade（质量等级）。
</final answer>
```

---

#### 自定义工具增强 Agent

内置 Agent 功能有限，我们通过自定义工具来大幅扩展能力：

```python
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from lang_core.prompts import ChatPromptTemplate, MessagesPlaceholder


@tool
def describe_dataframe() -> str:
    """获取 DataFrame 的完整描述信息，包括形状、列信息、统计摘要等"""
    buf = []
    buf.append(f"📊 数据集概览:\n")
    buf.append(f"行数: {len(df_eval):,}")
    buf.append(f"列数: {len(df_eval.columns)}\n")

    buf.append(f"列信息:")
    for col in df_eval.columns:
        dtype = df_eval[col].dtype
        non_null = df_eval[col].notna().sum()
        unique = df_eval[col].nunique()
        buf.append(f"  - {col}: {dtype} | 非空:{non_null} | 唯一值:{unique}")

    buf.append(f"\n数值列统计:")
    numeric_cols = df_eval.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df_eval[numeric_cols].describe().round(2)
        buf.append(stats.to_string())

    return "\n".join(buf)


@tool
def list_columns() -> str:
    """列出所有可用的列名及其简要说明"""
    column_info = {
        "model": "模型名称 (gpt-4, claude-3.5-sonnet 等)",
        "task": "任务类型 (reasoning, coding, math 等)",
        "score": "评估分数 (0-1)",
        "latency_ms": "响应延迟 (毫秒)",
        "tokens_used": "消耗的 token 数",
        "category": "难度等级 (easy/medium/hard)",
        "language": "语言 (zh/en/code)",
        "timestamp": "测试时间",
        "cost_usd": "API 调用成本 (美元)",
        "quality_grade": "质量等级 (A/B/C/D)",
    }

    lines = ["可用列名及说明:"]
    for col, desc in column_info.items():
        lines.append(f"  • {col}: {desc}")
    return "\n".join(lines)


@tool
def query_data(python_code: str) -> str:
    """用 Pandas 查询数据。输入合法的 Python/Pandas 代码，代码中可以用 'df' 变量访问 DataFrame。
    返回查询结果的字符串表示。只能用于读取数据，不能修改数据。"""
    local_vars = {"df": df_eval, "pd": pd, "np": np}

    try:
        result = eval(python_code, {"__builtins__": {}}, local_vars)

        if isinstance(result, pd.DataFrame):
            if len(result) <= 20:
                return result.to_string(index=False)
            else:
                head = result.head(20).to_string(index=False)
                tail = f"\n... (共 {len(result)} 行，仅显示前20行)"
                return head + tail
        elif isinstance(result, (int, float, str)):
            return str(result)
        elif isinstance(result, pd.Series):
            return result.to_string()
        else:
            return str(result)
    except Exception as e:
        return f"❌ 执行错误: {type(e).__name__}: {e}"


@tool
def get_top_n(group_by: str, metric: str = "score", n: int = 5,
              ascending: bool = False) -> str:
    """按指定列分组并取 Top N 结果。
    group_by: 分组列名
    metric: 排序指标列名
    n: 返回前几名
    ascending: 是否升序 (默认 False=降序)"""

    try:
        grouped = df_eval.groupby(group_by, observed=True)[metric].mean().sort_values(
            ascending=ascending
        ).head(n)

        lines = [f"Top {n} ({group_by} 按 {metric} {'升' if ascending else '降'}序):\n"]
        for val, score in grouped.items():
            lines.append(f"  {val}: {score:.3f}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 错误: {e}"
```

---

#### 构建完整 Agent 系统

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

TOOLS = [describe_dataframe, list_columns, query_data, get_top_n]

SYSTEM_PROMPT = """你是一个专业的数据分析助手。你可以使用以下工具来分析一个包含 LLM 模型评估数据的 DataFrame (变量名为 df)。

重要规则：
1. 当用户询问数据基本信息时，先调用 describe_dataframe 或 list_columns
2. 当用户需要具体查询时，使用 query_data 工具传入 Pandas 代码
3. 当用户问排名时，优先使用 get_top_n 工具
4. 回答要用中文，数字保留合理的小数位数
5. 如果发现异常值或有趣的趋势，主动指出
6. 不要编造数据——严格基于查询结果回答

数据背景：这是一个 LLM 模型多任务评估数据集，
包含 gpt-4、claude-3.5-sonnet 等模型在 reasoning/coding/math 等任务上的表现。
"""


def build_analysis_agent(llm: ChatOpenAI) -> AgentExecutor:

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return executor


agent_executor = build_analysis_agent(llm)
print("✅ Agent 已创建！")
print(f"   可用工具: {[t.name for t in TOOLS]}")
```

输出：
```
✅ Agent 已创建！
   可用工具: ['describe_dataframe', 'list_columns', 'query_data', 'get_top_n']
```

##### 测试基本功能

```python
test_queries = [
    "这个数据集的基本情况是什么？",
    "哪个模型的平均分最高？",
    "reasoning 任务上各模型的表现如何？",
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"{'─'*60}")

    result = agent_executor.invoke({
        "input": q,
        "chat_history": [],
    })

    print(f"\nA: {result['output']}")
```

输出：
```
============================================================
Q: 这个数据集的基本情况是什么？
──────────────────────────────────────────────────────────
> Entering new AgentExecutor chain...
Invoking: `describe_dataframe` with {}

📊 数据集概览:
行数: 500
列数: 11
...

A: 这个数据集包含 **500 条记录**和 **11 个字段**，是关于多个 LLM 模型在不同任务上的评估数据。涵盖 5 个模型（gpt-4、claude-3.5-sonnet 等）、8 种任务类型（reasoning、coding、math 等）。平均得分为 0.69，平均延迟约 789ms，平均消耗约 501 tokens。难度分布为 easy 40%、medium 35%、hard 25%。

============================================================
Q: 哪个模型的平均分最高？
──────────────────────────────────────────────────────────
> Entering new AgentExecutor chain...
Invoking: `get_top_n` with `{"group_by":"model","metric":"score","n":5}`

Top 5 (model 按 score 降序):
  gpt-4: 0.700
  claude-3.5-sonnet: 0.693
  gpt-4o-mini: 0.689
  llama-3-70b: 0.687
  qwen2.5-72b: 0.685

A: 从数据来看，**gpt-4** 的平均分最高（0.700），但差距非常小——前 5 名的差距仅在 0.015 分以内。这说明在这个评估集中，各模型的整体表现相当接近。值得注意的是 claude-3.5-sonnet 以 0.693 分紧随其后。

============================================================
Q: reasoning 任务上各模型的表现如何？
──────────────────────────────────────────────────────────
> Entering new AgentExecutor chain...
Invoking: `query_data` with `df[df.task=='reasoning'].groupby('model')['score'].agg(['count','mean','std']).round(3).sort_values('mean', ascending=False)`

          count   mean   std
model
gpt-4         62  0.723  0.138
claude-3...   58  0.712  0.141
gpt-4o-mini   64  0.698  0.145
llama-3-70b   61  0.690  0.139
qwen2.5-72b   63  0.687  0.140

A: 在 reasoning（推理）任务上，排名如下：

🥇 **gpt-4**: 均分 0.723（62 条样本）
🥈 **claude-3.5-sonnet**: 均分 0.712（58 条）
🥉 **gpt-4o-mini**: 均分 0.698（64 条）

有意思的是，gpt-4 在推理任务上的优势比整体优势更明显（0.723 vs 整体 0.700），说明它在复杂推理场景下的能力更为突出。
```

Agent 能够正确选择工具、执行查询、并将结果转化为自然语言的洞察性回答。
