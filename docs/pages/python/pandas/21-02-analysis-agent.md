---
title: 构建数据分析 Agent
description: 自定义 Tool 开发 / 多工具协作 / Agent 记忆管理 / 错误处理与重试 / 完整 Agent 示例
---
# 数据分析 Agent 构建


## 完整的数据分析 Agent 实现

```python
import pandas as pd
import numpy as np
import json
from typing import Optional, Type
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel


class QueryResult(BaseModel):
    """Tool 返回的结构化结果"""
    answer: str
    data_summary: str = ""
    row_count: int = 0
    visualization_data: Optional[dict] = None


class FilterParams(BaseModel):
    """过滤参数"""
    column: str
    operator: str = "gt"  # gt/lt/ge/le/eq/ne/in/notin
    value: float | str


class AggregationParams(BaseModel):
    """聚合参数"""
    group_by: str
    metric: str = "mean"
    columns: list[str] = []



class DataFrameInfoTool(StructuredTool):
    """获取 DataFrame 基本信息"""

    name: str = "dataframe_info"
    description: str = ("获取当前数据集的基本信息：行数、列名、数据类型、"
                       "缺失值统计。不需要额外参数。")
    df: pd.DataFrame

    args_schema: Type[BaseModel] = type(None)

    def _run(self) -> QueryResult:
        info = []
        info.append(f"📊 行数: {len(self.df):,}")
        info.append(f"📋 列数: {len(self.df.columns)}")
        info.append(f"\n列详情:")

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            nulls = self.df[col].isna().sum()
            unique = self.df[col].nunique()
            sample_val = self.df[col].dropna().iloc[0] if nulls < len(self.df) else 'N/A'
            info.append(f"  • {col}: {dtype} (空值:{nulls}, 唯一:{unique}) "
                       f"示例: {str(sample_val)[:50]}")

        return QueryResult(
            answer="\n".join(info),
            row_count=len(self.df),
        )


class DataFilterTool(StructuredTool):
    """过滤数据"""

    name: str = "filter_data"
    description: str = ("根据条件过滤 DataFrame。参数: column(列名), operator(gt/lt/eq/"
                       "ne/in/notin), value(值)。示例: column=MMLU, operator=gt, value=85")
    df: pd.DataFrame
    args_schema: Type[BaseModel] = FilterParams

    def _run(self, column: str, operator: str,
              value: float | str) -> QueryResult:
        if column not in self.df.columns:
            return QueryResult(answer=f"❌ 列 '{column}' 不存在。可用列: {list(self.df.columns)}")

        ops_map = {
            'gt': lambda s, v: s > v,
            'lt': lambda s, v: s < v,
            'ge': lambda s, v: s >= v,
            'le': lambda s, v: s <= v,
            'eq': lambda s, v: s == v,
            'ne': lambda s, v: s != v,
            'in': lambda s, v: s.isin(v.split(',') if isinstance(v, str) else [v]),
            'notin': lambda s, v: ~s.isin(v.split(',') if isinstance(v, str) else [v]),
        }

        op_fn = ops_map.get(operator)
        if not op_fn:
            return QueryResult(
                answer=f"❌ 不支持的操作符 '{operator}'。支持: {list(ops_map.keys())}"
            )

        mask = op_fn(self.df[column], value)
        result = self.df[mask].reset_index(drop=True)

        preview = result.head(10).to_string(index=False)
        return QueryResult(
            answer=f"✅ 过滤后剩余 {len(result)} 条记录\n{preview}",
            row_count=len(result),
            data_summary=f"{len(result)}/{len(self.df)} ({len(result)/len(self.df)*100:.1f}%)",
        )


class DataAggregationTool(StructuredTool):
    """分组聚合"""

    name: str = "aggregate"
    description: str = ("对数据进行分组聚合分析。参数: group_by(分组列), metric(avg/sum/count/"
                       "min/max/std), columns(要聚合的列，逗号分隔)")
    df: pd.DataFrame
    args_schema: Type[BaseModel] = AggregationParams

    def _run(self, group_by: str, metric: str = "mean",
              columns: list[str] = []) -> QueryResult:
        if group_by not in self.df.columns:
            return QueryResult(answer=f"❌ 列 '{group_by}' 不存在")

        if not columns:
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            columns = [c for c in numeric_cols if c != group_by]

        agg_funcs = {
            'mean': 'mean', 'sum': 'sum', 'count': 'count',
            'min': 'min', 'max': 'max', 'std': 'std'
        }
        func = agg_funcs.get(metric, 'mean')

        try:
            grouped = self.df.groupby(group_by)[columns].agg(func).round(2)
            result_str = grouped.to_markdown()

            viz_data = {}
            for col in columns[:5]:
                viz_data[col] = grouped[col].to_dict() if col in grouped.columns else {}

            return QueryResult(
                answer=f"✅ 按 `{group_name}` 分组，使用 `{metric}` 聚合:\n{result_str}",
                data_summary=f"{len(grouped)} 个组",
                visualization_data=viz_data,
            )
        except Exception as e:
            return QueryResult(answer=f"❌ 聚合失败: {e}")


class TopNTool(StructuredTool):
    """Top-N 查询"""

    name: str = "top_n"
    description: str = ("查询排名前 N 的记录。参数: n(数量), by(排序列名), ascending("
                       "True升序/False降序)")
    df: pd.DataFrame

    class Args(BaseModel):
        n: int = 5
        by: str = ""
        ascending: bool = False

    args_schema: Type[BaseModel] = Args

    def _run(self, n: int = 5, by: str = "",
              ascending: bool = False) -> QueryResult:
        if not by or by not in self.df.columns:
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            by = numeric_cols[0] if numeric_cols else self.df.columns[0]

        result = self.df.nlargest(n, by) if not ascending \
                 else self.df.nsmallest(n, by)

        return QueryResult(
            answer=f"✅ Top-{n} (按 {by} {'升' if ascending else '降'}序):\n"
                    f"{result.to_string(index=False)}",
            row_count=len(result),
        )



def build_analysis_agent(df, llm=None):
    """构建数据分析 Agent"""

    tools = [
        DataFrameInfoTool(df=df),
        DataFilterTool(df=df),
        DataAggregationTool(df=df),
        TopNTool(df=df),
    ]

    if llm is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = Agent(tools=tools, llm=llm, verbose=True)
    return agent


np.random.seed(42)
sample_df = pd.DataFrame({
    "model": ["GPT-4o"]*20 + ["Claude-3.5-Sonnet"]*15 + ["Llama-3.1-70B"]*10 +
             ["Qwen2.5-72B"]*8 + ["DeepSeek-V3"]*7,
    "task": np.random.choice(["chat", "code", "math", "reasoning"], 60),
    "score": np.random.uniform(65, 95, 60).round(2),
    "latency_ms": np.random.randint(150, 3000, 60),
    "tokens": np.random.randint(500, 8000, 60),
    "cost_usd": np.random.uniform(0.05, 3.00, 60).round(4),
})

print("=== 数据分析 Agent ===")
print(f"数据集: {len(sample_df):,} 行 × {len(sample_df.columns)} 列")
print(f"\n可用工具:")
print("  1. dataframe_info - 数据概览")
print("  2. filter_data   - 条件过滤")
print("  3. aggregate     - 分组聚合")
print("  4. top_n          - 排名查询")
```
