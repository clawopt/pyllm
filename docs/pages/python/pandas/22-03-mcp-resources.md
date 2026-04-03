---
title: MCP 资源管理与数据同步
description: Resource 定义 / 数据集版本管理 / 变更追踪 / 增量更新 / 导入导出
---
# MCP 资源管理与版本


## MCP Resource 概念

在 MCP 中，**Resource** 是 LLM 可以引用的数据实体。Pandas DataFrame 是最自然的 Resource 类型：

```python
import pandas as pd
from datetime import datetime


class MCPResourceManager:
    """MCP 资源管理器"""

    def __init__(self):
        self.resources = {}
        self.versions = {}

    def register_resource(self, uri: str, df: pd.DataFrame,
                         name: str, mime_type: str = "application/dataframe",
                         description: str = ""):
        """注册一个资源"""
        resource_id = f"pandas://{uri}"

        self.resources[resource_id] = {
            'id': resource_id,
            'uri': uri,
            'name': name,
            'mime_type': mime_type,
            'description': description,
            'data': df.copy(),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'version': 1,
            'row_count': len(df),
            'column_count': len(df.columns),
            'size_bytes': int(df.memory_usage(deep=True).sum()),
        }

        self.versions[resource_id] = [{
            'version': 1,
            'timestamp': datetime.now(),
            'row_count': len(df),
            'checksum': pd.util.hash_pandas_object(df) if hasattr(pd.util, 'hash_pandas_object') else hash(str(df.shape)),
        }]

        print(f"✓ 注册资源: {name} ({len(df):,} 行)")
        return resource_id

    def list_resources(self):
        """列出所有资源"""
        result = []
        for rid, r in self.resources.items():
            result.append({
                'uri': r['uri'],
                'name': r['name'],
                'mime_type': r['mime_type'],
                'rows': r['row_count'],
                'columns': r['column_count'],
                'version': r['version'],
                'description': r.get('description', ''),
            })
        return result

    def get_resource(self, uri: str):
        """获取资源内容"""
        rid = f"pandas://{uri}"
        if rid not in self.resources:
            return None

        r = self.resources[rid]
        return {
            'id': rid,
            'name': r['name'],
            'data': r['data'].head(100).to_dict(orient='records'),
            'total_rows': r['row_count'],
            'columns': list(r['data'].columns),
            'version': r['version'],
        }

    def update_resource(self, uri: str, new_df: pd.DataFrame):
        """更新资源（版本递增）"""
        rid = f"pandas://{uri}'
        if rid not in self.resources:
            return self.register_resource(uri, new_df, name=uri)

        old_r = self.resources[rid]
        old_r['data'] = new_df.copy()
        old_r['updated_at'] = datetime.now()
        old_r['version'] += 1
        old_r['row_count'] = len(new_df)
        old_r['column_count'] = len(new_df.columns)
        old_r['size_bytes'] = int(new_df.memory_usage(deep=True).sum())

        self.versions[rid].append({
            'version': old_r['version'],
            'timestamp': datetime.now(),
            'row_count': len(new_df),
        })

        print(f"✓ 更新资源 {uri} → v{old_r['version']}")
        return old_r

    def get_version_history(self, uri: str):
        """获取资源的版本历史"""
        rid = f"pandas://{uri}'
        return self.versions.get(rid, [])

    def diff_versions(self, uri: str, v1=None, v2=None):
        """比较两个版本的差异"""
        history = self.get_version_history(uri)
        if len(history) < 2:
            return {'error': '需要至少2个版本才能比较'}

        idx1 = (v1 or 1) - 1
        idx2 = (v2 or len(history)) - 1

        if idx1 < 0 or idx2 >= len(history):
            return {'error': f'版本范围无效 (共{len(history)}个版本)'}

        h1, h2 = history[idx1], history[idx2]
        return {
            'v1': h1['version'], 'v1_time': str(h1['timestamp']),
            'v1_rows': h1['row_count'],
            'v2': h2['version'], 'v2_time': str(h2['timestamp']),
            'v2_rows': h2['row_count'],
            'delta_rows': h2['row_count'] - h1['row_count'],
            'total_versions': len(history),
        }


mgr = MCPResourceManager()

eval_data = pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama", "Qwen", "DeepSeek"],
    "MMLU": [88.7, 89.2, 84.5, 83.5, 86.8],
    "price": [12.50, 18.00, 0.87, 1.47, 0.42],
})

rid = mgr.register_resource("eval/benchmarks", eval_data, "模型评估数据")

print("\n=== 资源列表 ===")
for r in mgr.list_resources():
    print(f"  {r['name']}: {r['rows']} 行, v{r['version']}")

new_eval = eval_data.copy()
new_eval.loc[new_eval['model'] == 'GPT-4o', 'MMLU'] = 89.1
mgr.update_resource("eval/benchmarks", new_eval)

new_eval.loc[new_eval['model'] == 'Claude', 'MMLU'] = 90.0
mgr.update_resource("eval/benchmarks", new_eval)

print("\n=== 版本历史 ===")
for v in mgr.get_version_history("eval/benchmarks"):
    print(f"  v{v['version']}: {str(v['timestamp'])[:19]} | {v['row_count']} 行")

print("\n=== 版本差异 ===")
diff = mgr.diff_versions("eval/benchmarks")
print(diff)
```
