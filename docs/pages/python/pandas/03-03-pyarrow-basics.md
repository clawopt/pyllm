---
title: PyArrow 后端基础
description: 什么是 PyArrow、为什么它更快更省内存、什么时候该用它
---
# PyArrow 后端：让字符串处理更快更省内存

上一节我们提到 Pandas 3.0 默认使用 PyArrow 作为字符串后端。这一节简单解释一下：**PyArrow 是什么？为什么它更好？**

## 一句话理解

> **PyArrow = 一种更紧凑的数据存储格式，专门为分析型计算优化**

想象你有 100 万行对话数据，每行有一段文本。用传统方式（object 类型）存储时，每行文本都是一个独立的 Python 字符串对象——就像 100 万个散落的盒子，每个盒子都有自己的标签和包装。而 PyArrow 的做法是：把所有文本连续地存放在一个大仓库里，只记录每行的起始位置和长度。

结果就是：**同样的数据，PyArrow 占用的内存可能只有传统方式的 30%-60%**。

## 对 LLM 开发者的意义

大模型语料本质上就是**海量文本**——prompt、response、文档内容……全是字符串。所以 PyArrow 的内存优势在 LLM 场景下被放大了：

| 数据量 | 传统 object | PyArrow | 节省 |
|--------|------------|---------|------|
| 10 万行 | ~15 MB | ~6 MB | ~60% |
| 100 万行 | ~150 MB | ~58 MB | ~61% |
| 1000 万行 | ~1.5 GB | ~0.58 GB | ~61% |

数字越大，节省越明显。

## 你需要做什么吗？

**如果你用的是 Pandas 3.0+，什么都不用做**——PyArrow 已经是默认选项了。

如果你想确认或手动切换：

```python
import pandas as pd

# 查看当前后端
print(pd.options.dtype_backend)

# 手动启用
pd.options.dtype_backend = 'pyarrow'

# 创建 DataFrame —— 字符串列自动使用 Arrow 格式
df = pd.DataFrame({'text': ['hello', 'world', '中文测试']})
print(df.dtypes)
```

## 什么时候不需要关心 PyArrow

- 你的数据量很小（几万行以内）
- 你的数据全是数值型（几乎没有字符串列）
- 你在使用某些还不兼容 Arrow 的第三方库

在这些情况下，用默认配置就行，不用额外操心。
