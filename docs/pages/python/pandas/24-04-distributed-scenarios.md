---
title: 分布式实战场景
description: 千万级语料的分布式清洗、分布式特征工程、从 Pandas 到 Dask 的迁移案例
---
# 分布式实战：千万级语料处理

```python
import dask.dataframe as dd

ddf = dd.read_json('conversations_*.jsonl', lines=True,
                    blocksize='128MB')

clean = ddf[
    (ddf['prompt'].str.len() > 5) &
    (ddf['response'].str.len() > 10) &
    (ddf['quality'] >= 4.0)
]

summary = clean.groupby('source').agg(
    count=('conversation_id', 'count'),
    avg_quality=('quality', 'mean'),
)

result = summary.compute()
print(result.sort_values('count', ascending=False))
```

这个例子展示了从 Pandas 到 Dask 的典型迁移路径：代码结构几乎不变，只是把 `pd` 换成 `dd`，最后加一个 `.compute()`。**对于已有 Pandas 代码库的团队来说，这是升级成本最低的方案**。
