---
title: 计算性能优化
description: 向量化 vs apply、eval() 表达式引擎、numba 加速、避免循环的性能原则
---
# 计算性能：让代码跑得更快

内存优化解决的是"能不能装下"的问题，而计算性能优化解决的是"跑完要多久"的问题。在 LLM 数据处理中，一个低效的数据清洗脚本可能从几分钟变成几小时。

## 性能对比：向量化 vs 循环 vs eval()

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
s = pd.Series(np.random.randn(n))

start = time.time()
r_apply = s.apply(lambda x: x**2 + 1)
t_apply = time.time() - start

start = time.time()
r_vec = s ** 2 + 1
t_vec = time.time() - start

start = time.time()
r_eval = pd.eval("s ** 2 + 1")
t_eval = time.time() - start

print(f"apply(): {t_apply:.3f}s | 向量化: {t_vec:.4f}s ({t_apply/t_vec:.0f}x) | eval(): {t_eval:.4f}s")
```

典型结果：向量化比 apply 快 50-200 倍，`pd.eval()` 使用 numexpr 引擎加速，介于两者之间。**规则很简单：能用内置运算符就不要用 apply，能用 eval 就不要手写循环**。
