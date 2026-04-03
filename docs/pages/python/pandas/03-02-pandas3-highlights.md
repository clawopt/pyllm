---
title: Pandas 3.0 要点速览
description: Copy-on-Write、PyArrow 后端、统一缺失值——新手需要知道的三个核心变化
---
# Pandas 3.0 新特性速览

如果你是第一次学 Pandas，不需要深入了解 Pandas 3.0 的每一个细节。但有几个**行为层面的变化**值得你提前知道——它们会影响你写代码的方式。

## 变化一：Copy-on-Write 默认启用

这是 Pandas 3.0 最重要也最影响日常编码的变化。

### 它解决什么问题

以前 Pandas 有一个臭名昭著的警告叫 `SettingWithCopyWarning`——当你对 DataFrame 的一个"切片"做修改时，Pandas 不确定你是在改原数据还是改副本，于是给你发个警告。

Pandas 3.0 通过 **Copy-on-Write（CoW）机制** 彻底解决了这个问题。简单来说：

```python
import pandas as pd

# 创建数据
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# 取出部分行
subset = df[df['a'] > 1]

# 修改 subset —— 只影响 subset，不会意外改动 df
subset['c'] = 99

print("原始 df:")
print(df)
print("\n修改后的 subset:")
print(subset)
```

输出：

```
原始 df:
   a  b
0  1  4
1  2  5
2  3 6

修改后的 subset:
   a  b   c
1  2  5  99
2  3  6  99
```

注意 `df` 完全没有被影响——这就是 CoW 的效果。

### 对你的影响

| 写法 | 3.0 下是否安全 |
|------|:---:|
| `df['new_col'] = value` | ✅ 安全 |
| `df.loc[条件, 'col'] = value` | ✅ 安全 |
| `subset = df[条件]; subset[x] = y` | ⚠️ 只修改 subset，不影响 df |
| 链式索引 `df['a'][mask] = x` | ❌ 会报错（本就是反模式） |

## 变化二：字符串默认使用 PyArrow 后端

Pandas 3.0 中，字符串列默认使用 Apache Arrow 格式存储，而不是传统的 Python object 格式。这对 LLM 开发者来说是个好消息——**内存占用大幅降低**。

你不需要手动配置任何东西，它自动生效。如果你想确认当前后端类型：

```python
import pandas as pd
df = pd.DataFrame({'text': ['hello', 'world', '你好世界']})
print(df.dtypes)
```

你会看到类似 `string[pyarrow]` 的类型名——这就说明已经在用 Arrow 后端了。

## 变化三：统一的缺失值表示

以前不同类型的列遇到空值时表现不一致：
- 整数列遇到 `None` → 自动变成浮点数（float64）
- 布尔列遇到 `None` → 变成 object 类型

Pandas 3.0 统一了这些行为——所有类型的列都能正确保留自己的类型，缺失值统一用 `<NA>` 表示。

```python
import pandas as pd

df = pd.DataFrame({
    '整数': [1, None, 3],
    '小数': [1.5, None, 3.5],
    '文本': ['hello', None, 'world'],
})

print(df.dtypes)
print(df)
```

注意看：**整数列仍然是整型**（不会偷偷变 float），所有缺失值都显示为 `<NA>` 而非混乱的 `NaN` / `None` 混合。

## 总结：作为新手你只需要记住

1. **CoW 让赋值操作更安全**——不用担心意外修改原始数据
2. **字符串默认更省内存**——不用额外配置
3. **缺失值处理更一致**——不再有奇怪的隐式类型转换

这三个变化都是**向好的方向改进**，你只需要正常写代码就能享受它们的好处。
