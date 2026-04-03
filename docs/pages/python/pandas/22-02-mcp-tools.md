---
title: MCP Tool 定义与注册
description: MCP 工具 Schema / 参数验证 / 错误处理 / 资源管理 / Pandas DataFrame 作为 MCP Resource
---
# MCP 工具定义与路由


## MCP 工具的标准化结构

```python
import pandas as pd
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class MCPToolSchema(BaseModel):
    """MCP 工具标准 Schema"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: str  # 处理函数名



class BuiltinTools:
    """MCP Server 内置工具集"""

    @staticmethod
    def get_read_tool(server):
        """读取数据集"""
        return {
            "name": "read_data",
            "description": "读取指定数据集的内容，支持指定行数限制",
            "input_schema": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "数据集名称"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回的最大行数 (默认100)",
                        "default": 100,
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要返回的列 (可选，默认全部)"
                    }
                },
                "required": ["dataset"]
            }
        }

    @staticmethod
    def get_query_tool(server):
        """查询/过滤数据"""
        return {
            "name": "query_data",
            "description": ("对数据集执行查询：过滤、排序、聚合等。"
                          "支持自然语言或结构化条件"),
            "input_schema": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string"
                    },
                    "filter": {
                        "type": "string",
                        "description": "过滤条件，如 'score > 85'"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "排序列名"
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "排序方向"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "分组列"
                    },
                    "aggregate": {
                        "type": "string",
                        "enum": ["mean", "sum", "count", "min", "max", "std"],
                        "description": "聚合函数"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20
                    }
                },
                "required": ["dataset"]
            }
        }

    @staticmethod
    def get_describe_tool(server):
        """数据集描述"""
        return {
            "name": "describe_dataset",
            "description": "获取数据集的结构信息：列名、类型、统计量等",
            "input_schema": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string"},
                    "include_stats": {
                        "type": "boolean",
                        "default": True,
                        "description": "是否包含统计摘要"
                    }
                },
                "required": ["dataset"]
            }
        }

    @staticmethod
    def get_list_datasets_tool(server):
        """列出所有数据集"""
        return {
            "name": "list_datasets",
            "description": "列出所有已注册的数据集及其基本信息",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }



class MCPToolRouter:
    """MCP 工具调用路由器"""

    def __init__(self, server):
        self.server = server
        self.tools = {}
        self._register_builtin()

    def _register_builtin(self):
        builtin = [
            BuiltinTools.get_read_tool(self.server),
            BuiltinTools.get_query_tool(self.server),
            BuiltinTools.get_describe_tool(self.server),
            BuiltinTools.get_list_datasets_tool(self.server),
        ]
        for tool in builtin:
            self.tools[tool['name']] = tool

    def register_custom(self, name, description, input_schema, handler):
        """注册自定义工具"""
        self.tools[name] = {
            'name': name,
            'description': description,
            'input_schema': input_schema,
            'handler': handler,
        }

    def list_tools(self) -> List[Dict]:
        """列出所有可用工具"""
        return [{'name': t['name'], 'description': t['description']}
                for t in self.tools.values()]

    def invoke(self, tool_name: str, arguments: Dict) -> Dict[str, Any]:
        """执行工具调用"""

        if tool_name not in self.tools:
            return {
                'success': False,
                'error': f"未知工具: {tool_name}",
                'available_tools': list(self.tools.keys()),
            }

        tool = self.tools[tool_name]

        try:
            if tool_name == 'read_data':
                return self._handle_read(arguments)
            elif tool_name == 'query_data':
                return self._handle_query(arguments)
            elif tool_name == 'describe_dataset':
                return self._handle_describe(arguments)
            elif tool_name == 'list_datasets':
                return self._handle_list()
            else:
                return {'success': False, 'error': f'未实现: {tool_name}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _handle_read(self, args):
        dataset = args.get('dataset')
        limit = args.get('limit', 100)
        columns = args.get('columns')

        if dataset not in self.server.datasets:
            return {'success': False, 'error': f'数据集不存在: {dataset}'}

        df = self.server.datasets[dataset]['data']
        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available]

        result_df = df.head(limit)
        return {
            'success': True,
            'total_rows': len(df),
            'returned_rows': len(result_df),
            'columns': list(result_df.columns),
            'data': result_df.to_dict(orient='records'),
        }

    def _handle_query(self, args):
        dataset = args.get('dataset')
        filter_cond = args.get('filter')
        sort_by = args.get('sort_by')
        sort_order = args.get('sort_order', 'desc')
        group_by = args.get('group_by')
        aggregate = args.get('aggregate')
        limit = args.get('limit', 20)

        if dataset not in self.server.datasets:
            return {'success': False, 'error': f'数据集不存在: {dataset}'}

        df = self.server.datasets[dataset]['data'].copy()

        if filter_cond and isinstance(filter_cond, str):
            df = self._apply_filter(df, filter_cond)

        if sort_by and sort_by in df.columns:
            ascending = (sort_order != 'desc')
            df = df.sort_values(sort_by, ascending=ascending)

        if group_by and group_by in df.columns:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            agg_cols = [c for c in numeric_cols if c != group_by]
            if agg_cols:
                func_map = {'mean': 'mean', 'sum': 'sum', 'count': 'count',
                           'min': 'min', 'max': 'max', 'std': 'std'}
                func = func_map.get(aggregate or 'mean', 'mean')
                df = df.groupby(group_by)[agg_cols].agg(func).round(2).reset_index()

        result_df = df.head(int(limit))
        return {
            'success': True,
            'rows_before_filter': len(self.server.datasets[dataset]['data']),
            'rows_after_processing': len(df),
            'returned_rows': len(result_df),
            'data': result_df.to_dict(orient='records') if len(result_df) < 200
                   else f"(数据过大，返回前{limit}条)",
        }

    def _handle_describe(self, args):
        dataset = args.get('dataset')
        include_stats = args.get('include_stats', True)

        if dataset not in self.server.datasets:
            return {'success': False, 'error': f'数据集不存在: {dataset}'}

        df = self.server.datasets[dataset]['data']
        info = {
            'name': dataset,
            'rows': len(df),
            'columns': [],
            'dtypes': {},
        }

        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isna().sum()),
                'unique_count': int(df[col].nunique()),
            }
            info['columns'].append(col_info)
            info['dtypes'][col] = str(df[col].dtype)

        if include_stats:
            numeric = df.describe().round(2)
            info['statistics'] = numeric.to_dict()

        return {'success': True, **info}

    def _handle_list(self):
        datasets = []
        for name, data in self.server.datasets.items():
            datasets.append({
                'name': name,
                'rows': len(data['data']),
                'columns': list(data['data'].columns),
                'description': data.get('description', ''),
            })
        return {'success': True, 'datasets': datasets}

    def _apply_filter(self, df, condition):
        try:
            import re
            cond = condition.strip()

            for op_pattern, op_func in [
                (r'(\w+)\s*>\s*(\d+\.?\d*)',
                 lambda m: df[m.group(1)] > float(m.group(2))),
                (r'(\w+)\s*<\s*(\d+\.?\d*)',
                 lambda m: df[m.group(1)] < float(m.group(2))),
                (r'(\w+)\s*=\s*(\d+\.?\d*)',
                 lambda m: df[m.group(1)] == float(m.group(2))),
                (r'(\w+)\s*!=\s*(\d+\.?\d*)',
                 lambda m: df[m.group(1)] != float(m.group(2))),
                (r'(\w+)\s*contains\s*(.+)',
                 lambda m: df[m.group(1)].str.contains(m.group(2), case=False)),
            ]:
                match = re.match(op_pattern, cond)
                if match:
                    return df[op_func(match)]

            return df
        except Exception:
            return df


from pandas_mcp_server import PandasMCPServer
server = PandasMCPServer()
server.register_dataset("eval", pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama"] * 10,
    "score": np.random.uniform(75, 95, 30).round(1),
}))

router = MCPToolRouter(server)

print("=== 可用工具 ===")
for t in router.list_tools():
    print(f"  • {t['name']}: {t['description'][:50]}")

print(f"\n=== 列出数据集 ===")
print(json.dumps(router.invoke('list_datasets', {}), indent=2))

print(f"\n=== 描述数据集 ===")
print(json.dumps(router.invoke('describe_dataset', {'dataset': 'eval'}), indent=2))
```
