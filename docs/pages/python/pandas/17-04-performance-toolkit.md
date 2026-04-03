---
title: 性能优化综合实战与工具
description: 性能分析工具 / 基准测试框架 / 优化检查清单 / LLM 数据管道性能调优
---
# 性能优化工具集


## Pandas 性能分析工具

### memory_usage()：逐列内存分析

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'id_int64': range(100_000),
    'score_float64': np.random.randn(100_000),
    'name_object': [f'item_{i}' for i in range(100_000)],
    'cat_string': pd.Categorical(np.random.choice(['A', 'B', 'C'], 100_000)),
})

mem = df.memory_usage(deep=True)
print("=== 逐列内存使用 ===")
for col in mem.index:
    print(f"  {col:<18s}: {mem[col] / 1024:>10.1f} KB")

total = mem.sum() / 1024 / 1024
print(f"\n总计: {total:.2f} MB")
```

### profiling：代码级性能剖析

```python
import pandas as pd
import numpy as np

n = 50_000
df = pd.DataFrame({'a': np.random.randn(n), 'b': np.random.randn(n)})

def slow_operation():
    result = []
    for i in range(len(df)):
        x = df.iloc[i]['a']
        y = df.iloc[i]['b']
        if x > 0:
            result.append(x * y + (x ** 2))
        else:
            result.append(abs(x * y))
    return result

```

## 优化检查清单

```python
class OptimizationChecklist:
    """Pandas 性能优化检查清单"""

    CHECKS = [
        ('是否使用了 iterrows()', '改用 itertuples() 或向量化'),
        ('字符串列是否为 object 类型', '转为 category 或 string[pyarrow]'),
        ('整数列是否为 int64', '用 downcast="integer" 缩减'),
        ('浮点列是否为 float64', '用 downcast="float" 缩减到 float32'),
        ('groupby 的键是否为 object', '转为 category 可加速 5-10x'),
        ('读取大文件时是否指定 dtype', '减少类型推断开销'),
        ('是否可以用 .str 向量化替代 apply(str)', '.str 快 3-15x'),
        ('条件逻辑是否用了 apply+if-else', '改用 np.where / np.select / case_when'),
        ('多条件过滤是否用 query()', '复杂查询可能更快'),
        ('输出格式是否需要 Parquet', '比 CSV 快 5-10x，文件小 75%'),
        ('大数据集是否分块处理', '避免 OOM'),
        ('是否有重复计算', '缓存中间结果'),
    ]

    @classmethod
    def run_check(cls, df):
        print("=" * 55)
        print("Pandas 性能优化检查清单")
        print("=" * 55)

        issues = []
        for i, (check, suggestion) in enumerate(cls.CHECKS, 1):
            status = "⚠️"
            detail = ""

            if 'iterrows' in check:
                status = "✅ PASS"
            elif 'object' in check and 'string' in check:
                obj_cols = [c for c in df.columns if df[c].dtype == 'object']
                if obj_cols:
                    detail = f" ({len(obj_cols)} 列: {obj_cols[:3]}...)"
                    status = "⚠️ ISSUE"
                else:
                    status = "✅ PASS"
            elif 'int64' in check:
                int64_cols = [c for c in df.columns
                              if str(df[c].dtype).startswith('int64')]
                if int64_cols:
                    detail = f" ({len(int64_cols)} 列)"
                    status = "⚠️ ISSUE"
                else:
                    status = "✅ PASS"
            elif 'float64' in check:
                f64_cols = [c for c in df.columns
                            if str(df[c].dtype).startswith('float64')]
                if f64_cols:
                    detail = f" ({len(f64_cols)} 列)"
                    status = "⚠️ ISSUE"
                else:
                    status = "✅ PASS"
            else:
                status = "ℹ️  INFO"

            print(f"  [{i:02d}] {status} {check}")
            if detail:
                print(f"       → {suggestion}{detail}")

        return issues


np.random.seed(42)
sample_df = pd.DataFrame({
    'model': ['GPT-4o'] * 25000 + ['Claude'] * 25000,
    'task': np.random.choice(['chat', 'code'], 50000),
    'score': np.random.uniform(60, 98, 50000),
    'latency': np.random.randint(100, 5000, 50000),
})

OptimizationChecklist.run_check(sample_df)
```

## LLM 场景：端到端性能调优案例

```python
import pandas as pd
import numpy as np

class LLMPipelineProfiler:
    """LLM 数据处理流水线性能分析器"""

    def __init__(self, pipeline_name="SFT Pipeline"):
        self.name = pipeline_name
        self.stages = []

    def time_stage(self, stage_name, func, *args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self.stages.append({
            'stage': stage_name,
            'time_sec': round(elapsed, 4),
            'rows_in': len(args[0]) if args and hasattr(args[0], '__len__') else '?',
            'rows_out': len(result) if hasattr(result, '__len__') else '?',
        })
        return result

    def report(self):
        total = sum(s['time_sec'] for s in self.stages)
        print(f"\n{'='*55}")
        print(f"流水线性能报告: {self.name}")
        print(f"{'='*55}")
        print(f"{'阶段':<25s} {'耗时(s)':>8s} {'输入行':>8s} {'输出行':>8s} {'占比':>6s}")
        print("-"*60)

        for s in self.stages:
            pct = s['time_sec'] / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"{s['stage']:<25s} {s['time_sec']:>8.4f} "
                  f"{str(s['rows_in']):>8s} {str(s['rows_out']):>8s} "
                  f"{pct:>5.1f}% {bar}")

        print("-"*60)
        print(f"{'TOTAL':<25s} {total:>8.4f}s")
        return self.stages


profiler = LLMPipelineProfiler("SFT Data Processing Pipeline")

np.random.seed(42)
raw_data = pd.DataFrame({
    'instruction': [f'问题{i}' for i in range(200_000)],
    'response': [f'回答{i}' for i in range(200_000)],
    'source': np.random.choice(['web', 'wiki', 'book'], 200_000),
})

def step_load(raw):
    return raw.copy()

def step_clean(df):
    return df[df['instruction'].str.len() > 3].reset_index(drop=True)

def step_feature(df):
    df = df.copy()
    df['instr_len'] = df['instruction'].str.len()
    df['resp_len'] = df['response'].str.len()
    return df

def step_optimize(df):
    df['source'] = df['source'].astype('category')
    return df


cleaned = profiler.time_stage('1. 加载数据', step_load, raw_data)
cleaned = profiler.time_stage('2. 清洗数据', step_clean, cleaned)
featured = profiler.time_stage('3. 特征工程', step_feature, cleaned)
optimized = profiler.time_stage('4. 内存优化', step_optimize, featured)

profiler.report()
```
