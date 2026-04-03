---
title: MCP 协议简介与 Pandas 集成
description: MCP (Model Context Protocol) 概述 / MCP Server 架构 / Pandas 作为 MCP 数据源 / 工具注册与调用
---
# MCP 协议与数据服务


## 什么是 MCP

**MCP（Model Context Protocol）** 是 Anthropic 提出的开放协议，用于标准化 LLM 与外部数据源/工具之间的通信。

```
┌──────────┐     MCP      ┌─────────────┐     MCP      ┌──────────┐
│  LLM     │ ◄────────────► │ MCP Server  │ ◄────────────► │ Pandas   │
│ (Client) │              │ (数据/工具)  │              │ DataFrame│
└──────────┘              └─────────────┘              └──────────┘
```

## 为什么 MCP 需要 Pandas

MCP Server 需要一个**结构化的数据管理层**来：
- 存储和管理可查询的数据集
- 执行数据操作（过滤、聚合、排序）
- 管理工具调用的状态和日志
- 缓存常用查询结果

→ **Pandas 是最自然的选择**

## MCP Server + Pandas 基础架构

```python
import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List


class PandasMCPServer:
    """基于 Pandas 的 MCP Server 实现"""

    def __init__(self):
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.tool_registry = {}
        self.call_log = []

    def register_dataset(self, name: str, df: pd.DataFrame,
                         description: str = ""):
        """注册数据集"""
        self.datasets[name] = {
            'data': df.copy(),
            'description': description,
            'registered_at': pd.Timestamp.now(),
        }
        print(f"✓ 注册数据集 '{name}': {len(df):,} 行 × {len(df.columns)} 列")
        return self

    def list_datasets(self) -> List[Dict]:
        """列出所有可用数据集"""
        result = []
        for name, info in self.datasets.items():
            result.append({
                'name': name,
                'rows': len(info['data']),
                'columns': list(info['data'].columns),
                'description': info.get('description', ''),
            })
        return result

    def query_dataset(self, dataset_name: str, query: str) -> Dict[str, Any]:
        """执行数据查询"""
        if dataset_name not in self.datasets:
            return {
                'success': False,
                'error': f"数据集 '{dataset_name}' 不存在",
                'available': list(self.datasets.keys()),
            }

        df = self.datasets[dataset_name]['data']
        self._log_call('query', dataset_name, query)

        try:
            q_lower = query.lower().strip()

            if q_lower.startswith("select "):
                cols_str = q_lower.replace("select ", "").split(" from")[0].strip()
                requested_cols = [c.strip() for c in cols_str.split(',')]
                available_cols = [c for c in requested_cols if c in df.columns]
                result_df = df[available_cols] if available_cols else df.head(20)
                return {
                    'success': True,
                    'result_type': 'table',
                    'columns': list(result_df.columns),
                    'rows': len(result_df),
                    'preview': result_df.head(10).to_dict(orient='records'),
                    'total_rows': len(result_df),
                }

            elif "where" in q_lower or "filter" in q_lower or "条件" in q_lower:
                return self._handle_filter(df, q_lower)

            elif "group" in q_lower or "aggregate" in q_lower or "分组" in q_lower:
                return self._handle_aggregate(df, q_lower)

            elif "sort" in q_lower or "order" in q_lower or "排序" in q_lower:
                return self._handle_sort(df, q_lower)

            elif "top" in q_lower or "前" in q_lower or "limit" in q_lower:
                n = int(''.join(filter(str.isdigit, q_lower))[:3]) or 10
                result_df = df.head(n)
                return {
                    'success': True,
                    'result_type': 'table',
                    'preview': result_df.to_dict(orient='records'),
                    'rows': len(result_df),
                }

            else:
                desc = df.describe().round(2).to_dict()
                return {
                    'success': True,
                    'result_type': 'summary',
                    'statistics': desc,
                    'total_rows': len(df),
                    'columns': list(df.columns),
                }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _handle_filter(self, df, query):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        for col in numeric_cols:
            if col.lower() in query:
                if ">" in query and any(d in query for d in ['>', '大于', 'gt']):
                    try:
                        val = float(''.join(c for c in query if c.isdigit() or c == '.'))
                        filtered = df[df[col] > val]
                        return {
                            'success': True,
                            'result_type': 'filtered',
                            'original_rows': len(df),
                            'filtered_rows': len(filtered),
                            'preview': filtered.head(10).to_dict(orient='records'),
                        }
                    except ValueError:
                        pass
                elif "<" in query or "小于" in query or "lt" in query:
                    try:
                        val = float(''.join(c for c in query if c.isdigit() or c == '.'))
                        filtered = df[df[col] < val]
                        return {
                            'success': True,
                            'result_type': 'filtered',
                            'original_rows': len(df),
                            'filtered_rows': len(filtered),
                            'preview': filtered.head(10).to_dict(orient='records'),
                        }
                    except ValueError:
                        pass

        return {'success': False, 'error': f'无法解析过滤条件: {query}'}

    def _handle_aggregate(self, df, query):
        group_col_candidates = [c for c in df.columns
                               if df[c].nunique() < 50]
        group_col = None
        for gc in group_col_candidates:
            if gc.lower() in query:
                group_col = gc
                break

        if not group_col:
            group_col = group_col_candidates[0] if group_col_candidates else df.columns[0]

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        agg_cols = [c for c in numeric_cols if c != group_col]

        grouped = df.groupby(group_col)[agg_cols].mean().round(2)
        return {
            'success': True,
            'result_type': 'aggregated',
            'group_by': group_col,
            'groups': len(grouped),
            'preview': grouped.reset_index().head(15).to_dict(orient='records'),
        }

    def _handle_sort(self, df, query):
        sort_col = None
        ascending = True

        for col in df.select_dtypes(include=['number']).columns:
            if col.lower() in query:
                sort_col = col
                break

        if not sort_col:
            return {'success': False, 'error': '未找到排序列'}

        if "降序" in query or "desc" in query or "从大到小" in query:
            ascending = False

        sorted_df = df.sort_values(sort_col, ascending=ascending)
        return {
            'success': True,
            'result_type': 'sorted',
            'sorted_by': sort_col,
            'ascending': ascending,
            'preview': sorted_df.head(15).to_dict(orient='records'),
        }

    def _log_call(self, tool, target, query):
        self.call_log.append({
            'tool': tool,
            'target': target,
            'query': str(query)[:100],
            'timestamp': pd.Timestamp.now(),
        })

    def get_status(self):
        """获取 Server 状态"""
        total_queries = len(self.call_log)
        return {
            'datasets_count': len(self.datasets),
            'total_data_rows': sum(len(v['data']) for v in self.datasets.values()),
            'total_queries': total_queries,
            'datasets': {k: len(v['data']) for k, v in self.datasets.items()},
        }


server = PandasMCPServer()

server.register_dataset(
    "model_benchmarks",
    pd.DataFrame({
        "model": ["GPT-4o", "Claude", "Llama", "Qwen", "DeepSeek"] * 5,
        "benchmark": ["MMLU", "HumanEval", "MATH", "GPQA", "BBH"] * 5,
        "score": np.random.uniform(70, 95, 25).round(1),
        "latency_ms": np.random.randint(200, 2000, 25),
    }),
    "LLM 模型基准测试数据"
)

print("\n=== MCP Server 状态 ===")
print(json.dumps(server.get_status(), indent=2, ensure_ascii=False))

print(f"\n=== 查询示例 ===")
r1 = server.query_dataset("model_benchmarks", "找出 score > 85 的记录")
print(json.dumps(r1, indent=2, ensure_ascii=False)[:500])

r2 = server.query_dataset("model_benchmarks", "按 model 分组统计平均分")
print(json.dumps(r2, indent=2, ensure_ascii=False)[:500])
```
