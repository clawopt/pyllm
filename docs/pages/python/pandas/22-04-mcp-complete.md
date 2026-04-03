---
title: MCP 完整实战：Pandas MCP Server
description: 完整的 Pandas MCP Server 实现 / 多数据源管理 / 权限控制 / 日志审计 / 与 LLM Client 集成
---
# 生产级 MCP Server 实现


## 完整的 PandasMCPServer

```python
import pandas as pd
import numpy as np
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional


class PandasMCPServer:
    """生产级 Pandas MCP Server"""

    def __init__(self, name="pandas-mcp-server", max_memory_mb=512):
        self.name = name
        self.datasets: Dict[str, Dict] = {}
        self.audit_log = pd.DataFrame(columns=[
            'timestamp', 'client_id', 'action', 'target',
            'params', 'result', 'duration_ms', 'success'
        ])
        self.max_memory = max_memory_mb * 1024 * 1024


    def add_dataset(self, name: str, df: pd.DataFrame,
                    owner: str = "default", tags=None,
                    description: str = "") -> Dict:
        """添加数据集"""
        mem_usage = int(df.memory_usage(deep=True).sum())
        current_total = sum(
            d['memory_bytes'] for d in self.datasets.values()
        )

        if current_total + mem_usage > self.max_memory:
            return {
                'success': False,
                'error': f'内存超限 (当前 {current_total/1024/1024:.0f}MB + '
                       f'新增 {mem_usage/1024/1024:.1f}MB > {self.max_memory/1024/1024:.0f}MB)'
            }

        self.datasets[name] = {
            'data': df.copy(),
            'owner': owner,
            'tags': tags or [],
            'description': description,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'row_count': len(df),
            'col_count': len(df.columns),
            'memory_bytes': mem_usage,
            'checksum': self._hash_df(df),
            'access_count': 0,
        }

        self._audit('add_dataset', name, {'rows': len(df), 'cols': len(df.columns)}, True)
        return {
            'success': True,
            'name': name,
            'rows': len(df),
            'columns': list(df.columns),
            'memory_kb': round(mem_usage / 1024, 1),
        }

    def remove_dataset(self, name: str) -> Dict:
        """移除数据集"""
        if name not in self.datasets:
            return {'success': False, 'error': f'数据集不存在: {name}'}

        del self.datasets[name]
        self._audit('remove_dataset', name, {}, True)
        return {'success': True}

    def list_datasets(self, owner=None) -> List[Dict]:
        """列出数据集"""
        result = []
        for name, d in self.datasets.items():
            if owner and d['owner'] != owner:
                continue
            result.append({
                'name': name,
                'rows': d['row_count'],
                'columns': d['col_count'],
                'owner': d['owner'],
                'tags': d.get('tags', []),
                'description': d.get('description', ''),
                'accessed': d['access_count'],
                'created': str(d['created_at'])[:19],
            })
        return result


    def query(self, dataset: str, operation: str, params: Dict = None) -> Dict:
        """统一查询入口"""
        start = time.time()

        try:
            if dataset not in self.datasets:
                result = {'success': False, 'error': f'数据集不存在: {dataset}'}
                self._audit('query', dataset, {'op': operation}, False)
                return result

            ds = self.datasets[dataset]
            ds['access_count'] += 1
            df = ds['data']

            op_map = {
                'read': self._op_read,
                'filter': self._op_filter,
                'aggregate': self._op_aggregate,
                'sort': self._op_sort,
                'describe': self._op_describe,
                'head': self._op_head,
                'sample': self._op_sample,
            }

            handler = op_map.get(operation.lower())
            if not handler:
                result = {
                    'success': False,
                    'error': f'不支持的操作: {operation}',
                    'supported': list(op_map.keys())
                }
            else:
                result = handler(df, params or {})

            duration = (time.time() - start) * 1000
            result['duration_ms'] = round(duration, 2)
            self._audit('query', dataset, {'op': operation},
                        result.get('success', False), duration)
            return result

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._audit('query_error', dataset, {'op': operation}, False, duration)
            return {'success': False, 'error': str(e)}

    def _op_read(self, df, p):
        limit = p.get('limit', 100)
        cols = p.get('columns')
        if cols:
            available = [c for c in cols if c in df.columns]
            df = df[available]
        return {
            'success': True, 'total_rows': len(df),
            'returned_rows': min(len(df), limit),
            'data': df.head(limit).to_dict(orient='records')
        }

    def _op_filter(self, df, p):
        condition = p.get('condition')
        if not condition or isinstance(condition, str):
            return {'success': True, 'filtered': 0, 'data': []}
        filtered = self._apply_conditions(df, condition)
        return {
            'success': True,
            'original_rows': len(df),
            'filtered_rows': len(filtered),
            'data': filtered.head(50).to_dict(orient='records')
        }

    def _op_aggregate(self, df, p):
        group_col = p.get('group_by') or (df.select_dtypes(include=['object']).columns.tolist()[0])
        agg_func = p.get('func', 'mean')
        numeric_cols = [c for c in df.select_dtypes(include=['number']).columns if c != group_col]

        if group_col in df.columns and numeric_cols:
            grouped = df.groupby(group_col)[numeric_cols].agg(agg_func).round(2).reset_index()
            return {'success': True, 'groups': len(grouped), 'data': grouped.to_dict(orient='records')}
        return {'success': True, 'groups': 0, 'data': df.describe().round(2).to_dict()}

    def _op_sort(self, df, p):
        by = p.get('by') or (df.select_dtypes(include=['number']).columns.tolist()[0])
        ascending = p.get('ascending', False)
        if by in df.columns:
            sorted_df = df.sort_values(by, ascending=ascending).head(p.get('limit', 20))
            return {'success': True, 'sorted_by': by, 'data': sorted_df.to_dict(orient='records')}
        return {'success': False, 'error': f'列不存在: {by}'}

    def _op_describe(self, df, p):
        info = []
        for col in df.columns:
            info.append({
                'column': col, 'dtype': str(df[col].dtype),
                'nulls': int(df[col].isna().sum()),
                'unique': int(df[col].nunique()),
                **dict(df[col].describe().round(2)) if df[col].dtype in ['float64', 'int64'] else {},
            })
        return {'success': True, 'columns': info}

    def _op_head(self, df, p): return self._op_read(df, p)
    def _op_sample(self, df, p):
        n = p.get('n', 10)
        sampled = df.sample(min(n, len(df)), random_state=42)
        return {'success': True, 'method': 'random', 'n': n, 'data': sampled.to_dict(orient='records')}

    @staticmethod
    def _apply_conditions(df, cond_str):
        import re
        for pattern, fn in [
            (r'(\w+)\s*>\s*(\d+\.?\d*)', lambda m, d: d[m.group(1)] > float(m.group(2))),
            (r'(\w+)\s*<\s*(\d+\.?\d*)', lambda m, d: d[m.group(1)] < float(m.group(2))),
            (r'(\w+)\s*=\s*(\d+\.?\d*)', lambda m, d: d[m.group(1)] == float(m.group(2))),
            (r'(\w+)\s*contains\s*(.+)', lambda m, d: d[m.group(1)].str.contains(m.group(2), case=False)),
        ]:
            match = re.match(pattern, cond_str.strip())
            if match:
                return fn(match, df)
        return df


    def _audit(self, action, target, params, success, duration=0):
        entry = {
            'timestamp': datetime.now(),
            'client_id': 'system',
            'action': action,
            'target': target,
            'params': json.dumps(params, ensure_ascii=False)[:200],
            'result': str(success)[:50],
            'duration_ms': round(duration, 2),
            'success': success,
        }
        new_row = pd.DataFrame([entry])
        self.audit_log = pd.concat([self.audit_log, new_row], ignore_index=True)

    def get_audit_summary(self):
        """获取审计摘要"""
        if len(self.audit_log) == 0:
            return "暂无操作记录"

        total = len(self.audit_log)
        success_rate = self.audit_log['success'].mean() * 100
        avg_duration = self.audit_log['duration_ms'].mean()
        action_dist = self.audit_log.groupby('action').size().to_dict()

        return {
            'total_operations': total,
            'success_rate': round(success_rate, 1),
            'avg_duration_ms': round(avg_duration, 1),
            'by_action': action_dist,
            'datasets_managed': len(self.datasets),
            'total_rows': sum(d['row_count'] for d in self.datasets.values()),
        }

    def get_status(self):
        """服务器状态"""
        total_mem = sum(d['memory_bytes'] for d in self.datasets.values())
        return {
            'name': self.name,
            'datasets': len(self.datasets),
            'total_rows': sum(d['row_count'] for d in self.datasets.values()),
            'memory_used_mb': round(total_mem / 1024 / 1024, 2),
            'memory_limit_mb': round(self.max_memory / 1024 / 1024, 0),
            'uptime_operations': len(self.audit_log),
        }

    @staticmethod
    def _hash_df(df):
        return hashlib.md5(pd.util.hash_pandas_object(df)).hexdigest()[:12]


server = PandasMCPServer(max_memory_mb=256)

server.add_dataset("models", pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama", "Qwen", "DeepSeek"] * 10,
    "MMLU": np.random.uniform(80, 95, 50).round(1),
    "price": np.random.uniform(0.14, 18.0, 50).round(2),
}), owner="team_a", tags=["eval", "benchmark"])

print(f"\n=== Server 状态 ===")
for k, v in server.get_status().items():
    print(f"  {k}: {v}")

print(f"\n=== 查询示例 ===")
r1 = server.query("models", "read", {"limit": 5})
print(f"read: {r1['success']} ({r1.get('returned_rows', '?')} rows)")

r2 = server.query("models", "filter", {"condition": "price < 1.0"})
print(f"filter: {r2['success']} ({r2.get('filtered_rows', '?')} rows)")

r3 = server.query("models", "aggregate", {"group_by": "model", "func": "mean"})
print(f"aggregate: {r3['success']} ({r3.get('groups', '?')} groups)")

print(f"\n=== 审计摘要 ===")
for k, v in server.get_audit_summary().items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for sk, sv in v.items():
            print(f"    {sk}: {sv}")
    else:
        print(f"  {k}: {v}")
```
