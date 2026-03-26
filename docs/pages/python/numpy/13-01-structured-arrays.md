# 创建结构化数组

结构化数组是 NumPy 中一种强大的数据类型，它允许你在一个数组中存储异构的、命名字段的数据。普通 NumPy 数组要求所有元素类型相同，但结构化数组的每个元素可以包含多个不同类型的字段，每个字段都有唯一的名称。在处理语言模型数据时，结构化数组非常适合存储 token_ids、attention_mask、token_type_ids 等相关联的数据，将它们打包在一起便于管理和传递。理解结构化数组的创建和使用方法，可以让你的数据处理代码更加清晰和高效。

## 为什么使用结构化数组

在 LLM 数据处理中，我们经常需要处理这样一组相关联的数据：
- token_ids：token 的整数 ID
- attention_mask：注意力掩码（1 表示有效，0 表示填充）
- token_type_ids：句子段 ID（用于 BERT 等模型）
- special_tokens_mask：特殊 token 掩码

普通做法是分别用独立的数组存储这些数据，但结构化数组可以将它们打包成一个紧凑的数据结构，减少函数传递参数的数量，也便于批量处理。

```python
import numpy as np

# 普通做法：多个独立数组
token_ids = np.array([464, 197, 35, 2057])
attention_mask = np.array([1, 1, 1, 1])
token_type_ids = np.array([0, 0, 0, 0])

# 结构化数组做法：将相关数据打包
data = np.array([(464, 1, 0), (197, 1, 0), (35, 1, 0), (2057, 1, 0)],
                dtype=[('token_id', np.int32),
                       ('attention_mask', np.int8),
                       ('token_type_id', np.int8)])
print(f"结构化数组:\n{data}")
```

## 定义结构化数据类型

创建结构化数组首先需要定义其数据类型（dtype）。dtype 描述了数组中每个字段的名称和数据类型：

```python
# 定义 dtype：字段名 -> 数据类型
token_dtype = np.dtype([
    ('token_id', np.int32),      # token ID
    ('attention_mask', np.int8),  # 注意力掩码
    ('token_type_id', np.int8),  # token 类型 ID
    ('special_token_mask', np.int8)  # 特殊 token 掩码
])

# 查看 dtype
print(f"Token 数据类型:\n{token_dtype}")
print(f"字段名: {token_dtype.names}")  # ('token_id', 'attention_mask', ...)
```

## 创建结构化数组

有多种方式创建结构化数组：

### 方法一：使用 dtype 参数

```python
# 创建结构化数组
tokens = np.array([(464, 1, 0, 0), (197, 1, 0, 0), (35, 1, 0, 0), (2057, 1, 0, 0)],
                  dtype=[('token_id', np.int32),
                         ('attention_mask', np.int8),
                         ('token_type_id', np.int8),
                         ('special_token_mask', np.int8)])

print(f"数组形状: {tokens.shape}")
print(f"数组 dtype: {tokens.dtype}")
print(f"数组内容:\n{tokens}")
```

### 方法二：使用字典 dtype

字段名到数据类型的字典也可以定义 dtype：

```python
# 简化的字典 dtype（字段名会自动生成）
dict_dtype = np.dtype({
    'names': ['token_id', 'attention_mask', 'token_type_id'],
    'formats': ['i4', 'i1', 'i1']  # i4=int32, i1=int8
})

tokens = np.zeros(4, dtype=dict_dtype)
tokens['token_id'] = [464, 197, 35, 2057]
tokens['attention_mask'] = [1, 1, 1, 1]
tokens['token_type_id'] = [0, 0, 0, 0]

print(f"字典 dtype 创建的数组:\n{tokens}")
```

### 方法三：使用字段名简写

对于只需要字段名和类型的基本情况：

```python
# 使用简化格式：'字段名:类型'，多个字段用逗号分隔
simple_dtype = 'token_id:i4, attention_mask:i1, token_type_id:i1'

tokens = np.array([(464, 1, 0), (197, 1, 0), (35, 1, 0), (2057, 1, 0)],
                  dtype=simple_dtype)
print(f"简化格式创建:\n{tokens}")
```

## 支持的数据类型

结构化数组支持多种 NumPy 数据类型：

```python
# 常见的数据类型
example_dtype = np.dtype([
    ('int8_field', np.int8),
    ('int16_field', np.int16),
    ('int32_field', np.int32),
    ('int64_field', np.int64),
    ('float32_field', np.float32),
    ('float64_field', np.float64),
    ('bool_field', np.bool_),
    ('bytes_field', 'S10'),  # 固定长度字节串
])

arr = np.zeros(3, dtype=example_dtype)
print(f"支持的类型示例:\n{arr.dtype}")
```

## 在LLM场景中的应用

### 存储训练样本

结构化数组非常适合存储单个训练样本的所有相关数据：

```python
# 定义训练样本的结构
sample_dtype = np.dtype([
    ('input_ids', 'i4', (128,)),      # 输入 token IDs，最多 128 个
    ('attention_mask', 'i1', (128,)), # 注意力掩码
    ('token_type_ids', 'i1', (128,)), # token 类型 ID
    ('position_ids', 'i2', (128,)),   # 位置 ID
    ('labels', 'i4', (128,)),         # 标签（用于语言建模）
])

# 创建训练样本数组
num_samples = 1000
dataset = np.zeros(num_samples, dtype=sample_dtype)

# 填充数据
np.random.seed(42)
for i in range(num_samples):
    seq_len = np.random.randint(32, 128)
    dataset[i]['input_ids'] = np.random.randint(0, 50257, size=128)
    dataset[i]['attention_mask'] = np.concatenate([
        np.ones(seq_len),
        np.zeros(128 - seq_len)
    ]).astype(np.int8)
    dataset[i]['labels'] = np.concatenate([
        dataset[i]['input_ids'][1:],
        np.zeros(1)
    ]).astype(np.int32)

print(f"数据集形状: {dataset.shape}")
print(f"数据集 dtype: {dataset.dtype}")
print(f"第一个样本 input_ids 前10个: {dataset[0]['input_ids'][:10]}")
```

### 批量处理

结构化数组可以方便地进行批量处理：

```python
def create_batch_from_samples(samples, batch_size):
    """从样本数组中提取 batch

    返回结构化数组的视图（不复制数据）
    """
    return samples[:batch_size]

# 示例
batch = create_batch_from_samples(dataset, batch_size=32)

print(f"Batch 形状: {batch.shape}")
print(f"Batch 共享内存: {np.shares_memory(dataset, batch)}")
print(f"Batch 第一条的 input_ids 前5个: {batch[0]['input_ids'][:5]}")
```

## 内存布局

结构化数组的内存布局非常紧凑，相关数据紧密存储在一起：

```python
# 对比：普通做法 vs 结构化数组
seq_len = 128
batch_size = 32

# 普通做法：多个独立数组
token_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
attention_mask = np.zeros((batch_size, seq_len), dtype=np.int8)
token_type_ids = np.zeros((batch_size, seq_len), dtype=np.int8)

total_memory_normal = token_ids.nbytes + attention_mask.nbytes + token_type_ids.nbytes
print(f"普通做法内存: {total_memory_normal / 1024:.1f} KB")

# 结构化数组
batch_dtype = np.dtype([
    ('token_ids', np.int32, (seq_len,)),
    ('attention_mask', np.int8, (seq_len,)),
    ('token_type_ids', np.int8, (seq_len,))
])
batch_structured = np.zeros(batch_size, dtype=batch_dtype)

total_memory_structured = batch_structured.nbytes
print(f"结构化数组内存: {total_memory_structured / 1024:.1f} KB")
print(f"内存节省: {(1 - total_memory_structured / total_memory_normal) * 100:.1f}%")
```

## 常见误区

**误区一：忘记字段名必须唯一**

字段名必须是唯一的字符串：

```python
# 错误：重复字段名
try:
    dup_dtype = np.dtype([('field', np.int32), ('field', np.int32)])
except Exception as e:
    print(f"错误: {e}")

# 正确
correct_dtype = np.dtype([('field1', np.int32), ('field2', np.int32)])
```

**误区二：混淆 dtype 的字段名和变量名**

结构化数组的字段名是 dtype 的一部分，不是变量名：

```python
arr = np.zeros(3, dtype=[('a', np.int32), ('b', np.float64)])

# 访问字段用字符串索引
print(arr['a'])  # 正确
# print(arr.a)  # 错误！NumPy 结构化数组不支持属性访问
```

**误区三：假设结构化数组支持所有普通数组操作**

某些操作在结构化数组上行为不同：

```python
arr = np.array([(1, 2.0), (3, 4.0)], dtype=[('x', np.int32), ('y', np.float64)])

# reshape 可能不按预期工作
try:
    reshaped = arr.reshape(1, 2)  # 可能产生意外结果
except Exception as e:
    print(f"reshape 可能有问题: {e}")
```

## API 总结

| 创建方式 | 示例 |
|---------|------|
| 元组列表 | `np.array([(1, 2), (3, 4)], dtype=[('a', np.int32), ('b', np.int32)])` |
| 字典 dtype | `np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'i4']})` |
| 简化字符串 | `np.dtype('a:i4, b:i4')` |
| 嵌套结构 | `np.dtype([('x', np.int32), ('y', [('a', np.int32), ('b', np.int32)])])` |

结构化数组是处理 LLM 数据的利器，它将相关联的字段打包在一起，既节省内存，又便于管理。掌握结构化数组的创建和使用，能让你的数据处理代码更加优雅高效。
