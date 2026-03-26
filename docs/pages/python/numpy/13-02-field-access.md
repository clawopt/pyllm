# 字段访问与操作

结构化数组的强大之处在于可以通过字段名直接访问和操作数据，而不需要使用索引来管理多个独立数组。NumPy 提供了丰富的字段访问语法，包括按名称访问字段、使用视图进行 dtype 重解释、以及批量字段操作等。掌握这些技巧可以让你高效地处理 LLM 的批次数据，快速构建 mini-batch 并进行特征工程。

## 按字段名访问

结构化数组的核心特性是可以用字段名访问数据：

```python
import numpy as np

# 创建结构化数组
samples = np.array([(464, 1, 0), (197, 1, 0), (35, 1, 0), (2057, 1, 0)],
                   dtype=[('token_id', np.int32),
                          ('attention_mask', np.int8),
                          ('token_type_id', np.int8)])

# 访问单个字段
print(f"所有 token_ids: {samples['token_id']}")
print(f"所有 attention_masks: {samples['attention_mask']}")

# 访问单个元素的一个字段
print(f"第0个样本的 token_id: {samples[0]['token_id']}")
print(f"第1个样本的 token_id: {samples[1]['token_id']}")
```

## 多字段同时访问

可以使用字段名列表同时访问多个字段：

```python
# 访问多个字段（返回视图或副本取决于具体情况）
tokens_and_mask = samples[['token_id', 'attention_mask']]
print(f"token_id 和 attention_mask:\n{tokens_and_mask}")
```

## 字段赋值

可以直接对字段赋值：

```python
# 修改字段值
samples['token_id'] = [100, 200, 300, 400]
print(f"修改后的 token_ids: {samples['token_id']}")

# 使用标量广播赋值
samples['attention_mask'] = 1  # 所有 attention_mask 设为 1
print(f"修改后的 attention_masks: {samples['attention_mask']}")

# 修改单个元素
samples[0]['token_id'] = 999
print(f"第0个样本修改后: {samples[0]}")
```

## 使用视图重解释字段

通过 `view` 方法可以将结构化数组的字段重新解释为另一种 dtype，但要注意字段的内存对齐：

```python
# 创建数据
arr = np.zeros(4, dtype=[('x', np.int32), ('y', np.float64)])
arr['x'] = [1, 2, 3, 4]
arr['y'] = [1.5, 2.5, 3.5, 4.5]

# 尝试将字段重新解释为字节数组
# 需要确保对齐方式兼容
# print(arr.view(np.int8))  # 可能报错，需要正确的字段对齐
```

## 批量构建 Batch

在 LLM 数据处理中，经常需要从样本数组中快速构建 batch：

```python
def build_batch_from_samples(samples, batch_indices):
    """从样本数组构建 batch

    参数:
        samples: 结构化数组样本
        batch_indices: batch 索引数组
    返回:
        batch 结构化数组（视图）
    """
    return samples[batch_indices]

# 示例
sample_dtype = np.dtype([
    ('input_ids', 'i4', (128,)),
    ('attention_mask', 'i1', (128,)),
    ('token_type_ids', 'i1', (128,)),
])

# 创建样本
num_samples = 1000
samples = np.zeros(num_samples, dtype=sample_dtype)
np.random.seed(42)
samples['input_ids'] = np.random.randint(0, 50257, size=(num_samples, 128))
samples['attention_mask'] = 1

# 构建 batch
batch_indices = np.array([0, 5, 10, 15])
batch = build_batch_from_samples(samples, batch_indices)

print(f"Batch 形状: {batch.shape}")
print(f"Batch input_ids 形状: {batch['input_ids'].shape}")
print(f"共享内存: {np.shares_memory(samples, batch)}")
```

## 字段过滤与条件筛选

可以像普通数组一样使用布尔索引过滤结构化数组：

```python
# 创建样本数据
samples = np.array([
    (464, 1, 0), (197, 1, 0), (35, 1, 0), (50256, 1, 1), (2057, 0, 0)
], dtype=[('token_id', np.int32), ('attention_mask', np.int8), ('is_special', np.int8)])

# 过滤：attention_mask == 1
mask_filter = samples['attention_mask'] == 1
filtered = samples[mask_filter]
print(f"attention_mask == 1 的样本:\n{filtered}")

# 组合条件
special_filter = (samples['attention_mask'] == 1) & (samples['is_special'] == 0)
filtered_no_special = samples[special_filter]
print(f"非特殊 token 且有效:\n{filtered_no_special}")
```

## 字段排序

结构化数组可以按某个字段排序：

```python
# 按 token_id 排序
sorted_samples = samples[np.argsort(samples['token_id'])]
print(f"按 token_id 排序:\n{sorted_samples}")

# 按多个字段排序
# 先按 attention_mask 降序，再按 token_id 升序
indices = np.lexsort((samples['token_id'], -samples['attention_mask']))
sorted_multi = samples[indices]
print(f"多字段排序:\n{sorted_multi}")
```

## 字段统计

对字段进行统计操作：

```python
# 创建数值字段
data = np.array([
    (100, 3.5), (200, 2.5), (300, 4.5), (400, 1.5)
], dtype=[('id', np.int32), ('score', np.float32)])

# 字段统计
print(f"score 均值: {data['score'].mean():.2f}")
print(f"score 标准差: {data['score'].std():.2f}")
print(f"score 最大: {data['score'].max():.2f}")
print(f"score 最小: {data['score'].min():.2f}")
```

## 在LLM场景中的应用

### 动态填充（Dynamic Padding）

在批处理时，不同样本的序列长度不同，可以使用结构化数组高效处理：

```python
def compute_batch_padding(data, max_length=None):
    """计算 batch 的填充掩码和有效长度

    返回:
        attention_mask: 有效位置为 1，填充位置为 0
        valid_length: 每个样本的有效长度
    """
    # 找到最大有效长度
    if max_length is None:
        valid_lengths = data['attention_mask'].sum(axis=1)
        max_length = valid_lengths.max()
    else:
        valid_lengths = np.minimum(data['attention_mask'].sum(axis=1), max_length)

    # 创建 attention_mask
    batch_size = len(data)
    attention_mask = np.zeros((batch_size, max_length), dtype=np.int8)
    for i in range(batch_size):
        attention_mask[i, :valid_lengths[i]] = 1

    return attention_mask, valid_lengths

# 示例
seq_lens = [45, 67, 23, 89, 56]
batch_data = np.zeros(5, dtype=[('input_ids', 'i4', (128,)), ('attention_mask', 'i1', (128,))])
for i, seq_len in enumerate(seq_lens):
    batch_data[i]['input_ids'][:seq_len] = np.random.randint(0, 50257, seq_len)
    batch_data[i]['attention_mask'][:seq_len] = 1

attention_mask, valid_lengths = compute_batch_padding(batch_data)
print(f"有效长度: {valid_lengths}")
print(f"Attention mask 形状: {attention_mask.shape}")
```

### 字段级别的数值操作

对结构化数组的字段进行数值操作：

```python
# 假设 token_ids 需要偏移（用于下一个 token 预测）
samples = np.array([
    ([464, 197, 35], [197, 35, 2057]),
], dtype=[('input_ids', 'i4', (3,)), ('labels', 'i4', (3,))])

# labels 是 input_ids 的偏移版本
# 这里演示如何批量设置
samples['labels'] = np.roll(samples['input_ids'], shift=-1)
samples['labels'][:, -1] = -100  # padding token 用于忽略

print(f"Input IDs: {samples['input_ids']}")
print(f"Labels: {samples['labels']}")
```

## 常见误区

**误区一：混淆字段访问和普通索引**

字段访问使用字符串索引：

```python
arr = np.zeros(3, dtype=[('x', np.int32), ('y', np.int32)])

# 错误：像普通索引一样访问
# arr[0] 返回整个记录，不是字段

# 正确：字符串索引
print(arr['x'])  # 返回所有 x 字段
print(arr[0]['x'])  # 返回第0个元素的 x 字段
```

**误区二：多字段访问返回的可能是副本**

使用字段名列表访问多个字段时，返回的是副本还是视图取决于具体情况：

```python
arr = np.zeros(3, dtype=[('x', np.int32), ('y', np.int32)])
arr['x'] = [1, 2, 3]
arr['y'] = [4, 5, 6]

# 单字段访问返回视图
single = arr['x']
single[0] = 999
print(f"arr['x'][0] 被修改后: {arr['x'][0]}")  # 999，共享内存

# 多字段访问返回副本
multi = arr[['x', 'y']]
multi[0]['x'] = 888
print(f"arr['x'][0] 未被修改: {arr['x'][0]}")  # 仍然是 999
```

**误区三：忽略 dtype 的内存对齐**

结构化数组的字段可能有内存对齐要求，这可能影响 view 操作：

```python
# dtype 的对齐可能影响视图创建
# 某些操作要求字段紧凑排列
```

## 使用 structured_to_unstructured

NumPy 提供了 `np.frombuffer` 和视图技巧在结构化数组和普通数组之间转换：

```python
# 将结构化数组转换为普通数组（副本）
samples = np.array([
    (464, 1, 0), (197, 1, 0), (35, 1, 0)
], dtype=[('token_id', np.int32), ('attention_mask', np.int8), ('token_type_id', np.int8)])

# 方法：手动堆叠字段
converted = np.stack([samples['token_id'], samples['attention_mask'], samples['token_type_id']], axis=1)
print(f"转换后形状: {converted.shape}")
print(f"转换后 dtype: {converted.dtype}")
```

## API 总结

| 操作 | 示例 | 说明 |
|------|------|------|
| 单字段访问 | `arr['field']` | 返回字段视图 |
| 多字段访问 | `arr[['field1', 'field2']]` | 返回副本 |
| 单元素字段 | `arr[0]['field']` | 返回标量 |
| 字段赋值 | `arr['field'] = value` | 广播赋值 |
| 布尔过滤 | `arr[arr['field'] > 0]` | 按条件过滤 |
| 字段排序 | `arr[np.argsort(arr['field'])]` | 按字段排序 |
| 字段统计 | `arr['field'].mean()` | 统计操作 |

字段访问是结构化数组的核心操作，掌握这些技巧可以让你的 LLM 数据处理代码更加简洁和高效。
