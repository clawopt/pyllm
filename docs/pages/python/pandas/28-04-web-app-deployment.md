# 部署为简单的 Web 应用


#### 从 Agent 到 Web 服务

Agent 已经可以交互式使用了，但要分享给团队或非技术用户，需要一个 Web 界面。本节用 **Streamlit** 快速搭建一个数据分析 Web 应用。

##### 架构设计

```
用户浏览器 (Streamlit UI)
    ↓ HTTP 请求
Streamlit 后端
    ↓ 调用
LangChain Pandas Agent
    ↓ 执行
Pandas DataFrame
    ↓ 返回
渲染为表格/图表/文字
```

---

#### Streamlit 应用实现

```python
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.set_page_config(
    page_title="🐼 Pandas AI 数据分析助手",
    page_icon="🤖",
    layout="wide",
)

st.title("🐼 Pandas AI 数据分析助手")
st.caption("用自然语言探索你的数据 — 由 LangChain + Pandas 驱动")

with st.sidebar:
    st.header("⚙️ 配置")

    api_key = st.text_input("OpenAI API Key", type="password",
                             help="输入你的 OpenAI API Key")

    model_option = st.selectbox(
        "选择模型",
        ["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet"],
        index=0,
        help="gpt-4o-mini 性价比最高，gpt-4 能力最强"
    )

    temperature = st.slider("Temperature (创造性)", 0.0, 1.0, 0.0)

    st.divider()
    st.markdown("""
    **使用提示:**
    - 问"数据长什么样？"看概览
    - 问"哪个模型最好？"做对比
    - 问"有什么异常？"找离群点
    - 支持多轮追问！
    """)


@st.cache_data
def load_sample_data():
    import numpy as np
    np.random.seed(42)
    N = 500

    df = pd.DataFrame({
        "model": np.random.choice(
            ["gpt-4","gpt-4o-mini","claude-3.5-sonnet","llama-3-70b","qwen2.5-72b"], N
        ),
        "task": np.random.choice(
            ["reasoning","coding","math","translation","summarization",
             "extraction","classification","creative_writing"], N
        ),
        "score": np.clip(np.random.normal(0.7, 0.15, N), 0, 1).round(3),
        "latency_ms": np.random.exponential(800, N).astype(int),
        "tokens_used": np.random.poisson(500, N),
        "category": np.random.choice(["easy","medium","hard"], N, p=[0.4,0.35,0.25]),
        "language": np.random.choice(["zh","en","code"], N, p=[0.3,0.4,0.3]),
    })

    price_map = {
        "gpt-4": 0.03, "gpt-4o-mini": 0.00015,
        "claude-3.5-sonnet": 0.003, "llama-3-70b": 0.001,
        "qwen2.5-72b": 0.0008,
    }
    df["cost_usd"] = (df["tokens_used"].astype(float) / 1000 *
                       df["model"].map(price_map)).round(6)
    return df


if not api_key:
    st.warning("⚠️ 请在左侧边栏输入 API Key 以启用 AI 分析功能")
    st.info("即使没有 API Key，你也可以预览下方的基础数据统计。")

df = load_sample_data()

col1, col2, col3 = st.columns(3)
col1.metric("总记录数", f"{len(df):,}")
col2.metric("模型数量", str(df["model"].nunique()))
col3.metric("平均得分", f"{df['score'].mean():.3f}")

with st.expander("📊 数据预览 (前10行)"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("📈 快速统计"):
    tab1, tab2 = st.tabs(["按模型", "按任务"])

    with tab1:
        model_stats = df.groupby("model").agg(
            count=("score", "count"),
            mean_score=("score", "mean"),
            avg_latency=("latency_ms", "mean"),
            total_cost=("cost_usd", "sum"),
        ).round(3).sort_values("mean_score", ascending=False)
        st.dataframe(model_stats, use_container_width=True)

    with tab2:
        task_stats = df.groupby("task")["score"].agg(["count","mean","std"]).round(3)
        task_stats.columns = ["样本数", "均分", "标准差"]
        st.dataframe(task_stats.sort_values("均分", ascending=False), use_container_width=True)


st.divider()
st.header("💬 AI 对话分析")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("问一个关于数据的问题..."):
    if not api_key:
        st.error("请先在侧边栏配置 API Key")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    llm = ChatOpenAI(
        model=model_option,
        temperature=temperature,
        api_key=api_key,
        request_timeout=60,
    )

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        max_iterations=5,
    )

    with st.spinner("🤖 正在分析..."):
        try:
            response = agent.invoke(prompt)
            answer = response.get("output", "抱歉，无法生成回答。")
        except Exception as e:
            answer = f"❌ 出错了: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)


with st.sidebar:
    st.divider()
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages.clear()
        st.rerun()

    st.caption(f"当前对话轮次: {len(st.session_state.messages)//2}")
```

---

#### 部署方式

##### 本地运行

```bash
pip install streamlit streamlit-chat

streamlit run pandas_agent_app.py --server.port 8501
```

访问 `http://localhost:8501` 即可看到界面。

##### Streamlit Cloud 一键部署（免费）

```yaml
name: pandas-ai-agent
python: "3.11"

files:
  - pandas_agent_app.py
  - requirements.txt

streamlit>=1.30
langchain>=0.3
langchain-openai>=0.2
langchain-experimental>=0.3
pandas>=2.0
numpy>=1.24
python-dotenv>=1.0
```

将代码推送到 GitHub 仓库后，在 [share.streamlit.io](https://share.streamlit.io) 连接仓库即可自动部署。

##### Docker 部署（生产环境）

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pandas_agent_app.py .

EXPOSE 8501

HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "pandas_agent_app.py",
           "--server.address=0.0.0.0",
           "--server.port=8501",
           "--server.enableCORS=false",
           "--server.enableXsrfProtection=false"]
```

```bash
docker build -t pandas-ai-agent .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="sk-proj-xxx" \
  pandas-ai-agent
```

---

#### 高级功能扩展

##### 上传自定义 CSV/Excel 文件

```python
uploaded_file = st.file_uploader(
    "📁 上传你的数据文件 (CSV / Excel)",
    type=["csv", "xlsx", "parquet"],
    help="支持 CSV、Excel (.xlsx)、Parquet 格式"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            user_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            user_df = pd.read_excel(uploaded_file)
        else:
            user_df = pd.read_parquet(uploaded_file)

        st.success(f"✅ 已加载: {uploaded_file.name} ({len(user_df):,} 行 × {len(user_df.columns)} 列)")

        with st.expander("查看列信息"):
            col_info = pd.DataFrame({
                "列名": user_df.columns,
                "类型": user_df.dtypes.astype(str).values,
                "非空数": user_df.notna().sum().values,
                "唯一值": user_df.nunique().values,
            })
            st.dataframe(col_info, use_container_width=True)

        global df
        df = user_df

        if "agent_cache" in st.session_state:
            del st.session_state["agent_cache"]

    except Exception as e:
        st.error(f"❌ 文件读取失败: {e}")
```

##### 查询结果可视化增强

```python
def render_query_result(answer: str):
    lines = answer.strip().split("\n")

    in_table = False
    table_rows = []
    other_lines = []

    for line in lines:
        stripped = line.strip()
        if "|" in stripped and "---" not in stripped:
            in_table = True
            cells = [c.strip() for c in stripped.split("|")]
            cells = [c for c in cells if c]
            table_rows.append(cells)
        elif "---" in stripped:
            continue
        else:
            if in_table and table_rows:
                try:
                    table_df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
                    st.dataframe(table_df, use_container_width=True)
                except Exception:
                    st.code("\n".join(table_rows))
                table_rows = []
                in_table = False
            other_lines.append(line)

    if table_rows:
        try:
            table_df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
            st.dataframe(table_df, use_container_width=True)
        except Exception:
            pass

    if other_lines:
        text = "\n".join(other_lines)
        st.markdown(text)
```

##### 使用量监控面板

```python
with st.sidebar:
    if "usage_stats" not in st.session_state:
        st.session_state.usage_stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }

    st.divider()
    st.subheader("📊 使用统计")

    stats = st.session_state.usage_stats
    col_a, col_b = st.columns(2)
    col_a.metric("查询次数", f"{stats['total_queries']}")
    col_b.metric("累计成本", f"${stats['total_cost_usd']:.4f}")

    if stats["total_queries"] > 0:
        st.caption(f"平均每次成本: ${stats['total_cost_usd']/stats['total_queries']:.6f}")

    if st.button("重置统计"):
        st.session_state.usage_stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }
        st.success("✅ 统计已重置")
```

---

#### 完整应用截图效果描述

启动后的应用包含以下区域：

| 区域 | 内容 |
|------|------|
| **顶部标题栏** | 🐼 Pandas AI 数据分析助手 + 副标题 |
| **左侧边栏** | API Key 输入、模型选择、温度调节、使用提示、统计面板、清空按钮 |
| **主区域上部** | 三个指标卡片（总记录数、模型数、平均分）+ 数据预览折叠区 + 快速统计 Tab |
| **主区域下部** | 聊天界面：消息历史 + 输入框 |
| **文件上传** | 支持 CSV/Excel/Parquet 自定义数据 |

用户打开网页 → 输入 API Key → 可以立即开始用自然语言提问分析数据。整个应用从零到可用只需要 **~150 行 Python 代码**。
