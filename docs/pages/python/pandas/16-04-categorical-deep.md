---
title: Category 类型深度解析
description: category 的内部存储机制、有序类别 CategoricalDtype、性能对比与适用场景
---
# Category 类型：低基数列的终极优化方案

我们在 04-04 和 07-03 节都简要介绍过 `category` 类型。这一节深入它的内部机制，让你不仅知道"怎么用"，还知道"为什么这么快"。

## 内部机制回顾

```
object 存储:  ["api", "web", "export", "api", "web", ...] × N
              → 每个元素是一个独立的 Python 字符串对象
              → 内存 ≈ N × 50 bytes

category 存储:
  整数 codes:   [0, 1, 2, 0, 1, ...] × N (每行只存一个整数)
  字典表:      {0→"api", 1→"web", 2→"export"} (只存一份)
              → 内存 ≈ N × 1 byte + 字典表 ~200 bytes
```

核心思想：**把唯一值提取到字典表里，原列只存整数索引**。这就是为什么 100 万行的字符串列转成 category 后能从 50MB 降到 1MB——因为实际只有几个不同的值在重复出现。

## 有序类别

```python
import pandas as pd

df = pd.DataFrame({
    'tier': pd.Categorical(
        ['B', 'A', 'C', 'A', 'B'],
        categories=['D', 'C', 'B', 'A'],  # 从差到好排序
        ordered=True
    ),
})

print(df['tier'].min())   # 'D'
print(df['tier'] > 'B')   # [False, True, False, True, False]
```

设置 `ordered=True` 后，category 列支持比较运算（`<`, `>`）——这在"质量等级"这类有天然顺序的分类数据中非常有用。你可以直接写 `df[df['tier'] > 'B']` 来筛选 B 级及以上的数据，而不需要先做数值编码。
