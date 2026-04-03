---
title: 合并与拼接综合实战
description: SFT 多源数据整合 / 评估报告多表关联 / 增量数据更新 / 完整 ETL 流水线
---
# 拼接场景实战


## 场景一：SFT 数据集多源整合

```python
import pandas as pd
import numpy as np

class SFTDataIntegrator:
    """SFT 数据集多源整合器"""

    def __init__(self):
        self.sources = {}
        self.mappings = {}

    def register_source(self, name, df, col_map=None):
        """注册数据源，可选列名映射"""
        if col_map:
            df = df.rename(columns=col_map)
        self.sources[name] = df
        return self

    def integrate(self, id_col='id', how='outer'):
        """整合所有数据源"""
        if not self.sources:
            return pd.DataFrame()

        names = list(self.sources.keys())
        result = self.sources[names[0]].copy()
        result['_source'] = names[0]

        for name in names[1:]:
            source_df = self.sources[name].copy()
            source_df['_source'] = name

            common_cols = set(result.columns) & set(source_df.columns) - {'_source'}
            if id_col and id_col in common_cols:
                result = pd.merge(
                    result, source_df,
                    on=id_col,
                    how=how,
                    suffixes=('', f'_{name}')
                )
            else:
                result = pd.concat([result, source_df], ignore_index=True)

        return result


np.random.seed(42)
integrator = SFTDataIntegrator()

integrator.register_source('crawl', pd.DataFrame({
    'instruction': [f'问题 {i}' for i in range(50)],
    'response': [f'回答 {i}' for i in range(50)],
    'url': [f'https://example.com/{i}' for i in range(50)],
}))

integrator.register_source('human', pd.DataFrame({
    'question': [f'人工问题 {i}' for i in range(30)],
    'answer': [f'人工回答 {i}' for i in range(30)],
    'annotator': np.random.choice(['A', 'B', 'C'], 30),
    'quality_score': np.random.uniform(0.6, 1.0, 30).round(2),
}), col_map={'question': 'instruction', 'answer': 'response'})

integrator.register_source('synthetic', pd.DataFrame({
    'prompt': [f'合成提示 {i}' for i in range(20)],
    'completion': [f'合成补全 {i}' for i in range(20)],
    'generator_model': ['gpt-4o']*10 + ['claude']*10,
}), col_map={'prompt': 'instruction', 'completion': 'response'})

integrated = integrator.integrate(how='outer')
print(f"=== 整合结果 ===")
print(f"总条目: {len(integrated)}")
print(f"列: {list(integrated.columns)}")
print(f"\n来源分布:")
print(integrated['_source'].value_counts())
```

## 场景二：增量数据更新（Upsert 模式）

```python
import pandas as pd
import numpy as np

class IncrementalUpdater:
    """增量更新器：有则更新，无则插入"""

    @staticmethod
    def upsert(existing_df, new_df, on_col):
        """执行 upsert 操作"""
        existing_keys = set(existing_df[on_col])
        new_keys = set(new_df[on_col])

        to_update_mask = new_df[on_col].isin(existing_keys)
        to_insert_mask = ~to_update_mask

        updated = existing_df.copy()
        rows_to_update = new_df[to_update_mask].copy()

        for _, row in rows_to_update.iterrows():
            key_val = row[on_col]
            mask = updated[on_col] == key_val
            for col in row.index:
                if col != on_col and col in updated.columns:
                    updated.loc[mask, col] = row[col]

        new_rows = new_df[to_insert_mask].copy()
        if len(new_rows) > 0:
            updated = pd.concat([updated, new_rows], ignore_index=True)

        return updated


existing = pd.DataFrame({
    'model_id': ['m001', 'm002', 'm003'],
    'model_name': ['GPT-4o', 'Claude', 'Llama'],
    'MMLU': [88.7, 89.2, 84.5],
})

incoming = pd.DataFrame({
    'model_id': ['m002', 'm003', 'm004', 'm005'],
    'model_name': ['Claude-3.5', 'Llama-3.1', 'Qwen', 'DeepSeek'],
    'MMLU': [89.8, 86.2, 83.5, 87.3],
})

result = IncrementalUpdater.upsert(existing, incoming, on_col='model_id')
print("=== Upsert 结果 ===")
print(result.to_string(index=False))
```

## 场景三：完整 ETL 合并流水线

```python
import pandas as pd
import numpy as np

class ETLPipeline:
    """LLM 数据 ETL 流水线"""

    def __init__(self):
        self.raw_data = {}
        self.transformed = None

    def extract(self, name, path_or_df):
        """抽取数据"""
        if isinstance(path_or_df, pd.DataFrame):
            self.raw_data[name] = path_or_df
        else:
            self.raw_data[name] = pd.read_json(path_or_df, lines=True)
        return self

    def transform_merge(self, key='model'):
        """合并所有来源"""
        dfs = list(self.raw_data.values())

        unified_cols = set()
        for df in dfs:
            unified_cols.update(df.columns)

        normalized = []
        for name, df in self.raw_data.items():
            df = df.copy()
            df['_etl_source'] = name
            for col in unified_cols:
                if col not in df.columns:
                    df[col] = np.nan
            normalized.append(df[sorted(unified_cols | {'_etl_source'})])

        self.transformed = pd.concat(normalized, ignore_index=True)
        return self

    def validate(self):
        """数据质量校验"""
        issues = []
        total = len(self.transformed)

        if total == 0:
            issues.append("合并后数据为空")

        null_counts = self.transformed.isnull().sum()
        high_null = null_counts[null_counts > total * 0.5]
        if len(high_null) > 0:
            issues.append(f"高缺失率列: {dict(high_null)}")

        dupes = self.transformed.duplicated().sum()
        if dupes > 0:
            issues.append(f"重复行: {dupes}")

        print(f"ETL 校验: {len(issues)} 个问题" if issues else "ETL 校验通过 ✅")
        for issue in issues:
            print(f"  ⚠️ {issue}")

        return self

    def load_summary(self):
        """输出汇总"""
        if self.transformed is None or len(self.transformed) == 0:
            print("无数据")
            return

        print(f"\n=== ETL 结果摘要 ===")
        print(f"总行数: {len(self.transformed)}")
        print(f"总列数: {len(self.transformed.columns)}")
        print(f"来源数: {self.transformed['_etl_source'].nunique()}")
        print(f"各来源规模:")
        print(self.transformed['_etl_source'].value_counts().to_string())


pipeline = ETLPipeline()
pipeline.extract('eval_scores', pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'] * 10,
    'score': np.random.uniform(80, 95, 30),
}))
pipeline.extract('cost_data', pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'] * 5,
    'price': np.random.uniform(0.14, 3.0, 20),
}))
pipeline.extract('perf_data', pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'] * 7,
    'latency_ms': np.random.randint(200, 1500, 21),
}))

(pipeline
 .transform_merge(key='model')
 .validate()
 .load_summary())
```
