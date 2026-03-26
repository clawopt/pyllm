# 视图与副本

在 NumPy 中，理解和区分视图（view）和副本（copy）是高效内存使用的关键。视图是与原始数组共享底层数据的数组，对视图的修改会影响原始数组；副本则是完全独立的数据拷贝，修改副本不会影响原始数组。在处理大型语言模型时，如词嵌入矩阵等大型数据，正确区分和使用视图与副本可以显著减少内存占用，提高数据处理效率。

## 什么是视图

视图是共享同一块底层数据的数组。当你创建一个视图时，NumPy 不会复制数据，而是创建一个具有不同 shape、stride 或 dtype 但指向相同内存区域的对象。许多 NumPy 操作会返回视图而不是副本：

```python
import numpy as np

# 创建视图的操作
arr = np.arange(12).reshape(3, 4)
print(f"原始数组:\n{arr}")

# reshape 返回视图（如果可能）
view = arr.reshape(4, 3)
print(f"\n视图 shape={view.shape}:")
print(view)

# 修改视图会影响原始数组
view[0, 0] = 100
print(f"\n修改视图后，原始数组:\n{arr}")
```

## 什么是副本

副本是完全独立的数据拷贝。对副本的修改不会影响原始数组。显式复制数据需要使用 `copy()` 方法或 `np.copy()` 函数：

```python
# 使用 .copy() 方法
arr = np.arange(12).reshape(3, 4)
copy_of_arr = arr.copy()

# 修改副本不影响原始数组
copy_of_arr[0, 0] = 999
print(f"原始数组[0,0]: {arr[0, 0]}")  # 仍然是 0
print(f"副本[0,0]: {copy_of_arr[0, 0]}")  # 是 999
```

## 哪些操作返回视图，哪些返回副本

理解哪些操作返回视图、哪些返回副本对于性能优化很重要：

**返回视图的操作**：
- `reshape`：改变形状但不复制数据（如果内存布局兼容）
- `transpose` / `.T`：转置数组
- `squeeze`：移除长度为 1 的维度
- `expand_dims`：增加维度
- 切片操作：`arr[::2]` 等

**返回副本的操作**：
- `arr.copy()`：显式复制
- `np.concatenate`：拼接数组
- `np.vstack` / `np.hstack`：堆叠数组
- `np.array(obj)`：创建新数组时
- 整数数组索引（如 `arr[[0, 1, 2]]`）
- 布尔索引（如 `arr[mask]`）

```python
arr = np.arange(12).reshape(3, 4)

# 切片操作返回视图
slice_view = arr[:, :2]
slice_view[0, 0] = 999
print(f"切片修改后 arr[0,0]: {arr[0, 0]}")  # 被修改了

# 整数索引返回副本
int_indexed = arr[[0, 1, 2]]
int_indexed[0, 0] = 888
print(f"整数索引修改后 arr[0,0]: {arr[0, 0]}")  # 未被修改
```

## 使用 `np.shares_memory` 检查共享

可以使用 `np.shares_memory()` 函数检查两个数组是否共享底层数据：

```python
arr = np.arange(12).reshape(3, 4)

# reshape 可能返回视图
view = arr.reshape(4, 3)
print(f"reshape: 共享内存? {np.shares_memory(arr, view)}")

# 切片返回视图
slice_view = arr[:, :2]
print(f"切片: 共享内存? {np.shares_memory(arr, slice_view)}")

# copy 返回副本
copied = arr.copy()
print(f"copy: 共享内存? {np.shares_memory(arr, copied)}")
```

## 在LLM场景中的应用

### 词嵌入矩阵处理

在处理词嵌入矩阵时，理解视图与副本的区别尤为重要。词嵌入矩阵通常很大（如 GPT-2 的嵌入矩阵形状是 50257 × 768），复制这样的矩阵会消耗大量内存。

```python
def get_embedding_submatrix(embeddings, token_ids, copy=False):
    """获取子嵌入矩阵

    参数:
        embeddings: 完整的嵌入矩阵 (vocab_size, embedding_dim)
        token_ids: 需要提取的 token id 列表
        copy: 是否复制数据
    返回:
        子嵌入矩阵 (len(token_ids), embedding_dim)
    """
    sub_embeddings = embeddings[token_ids]

    if copy:
        return sub_embeddings.copy()

    return sub_embeddings  # 返回视图，共享内存

# 示例
np.random.seed(42)
embeddings = np.random.randn(50257, 768)
token_ids = [0, 1, 2, 3, 4]

# 获取视图（不复制）
sub_view = get_embedding_submatrix(embeddings, token_ids, copy=False)
print(f"子矩阵形状: {sub_view.shape}")
print(f"共享内存: {np.shares_memory(embeddings, sub_view)}")

# 获取副本（复制）
sub_copy = get_embedding_submatrix(embeddings, token_ids, copy=True)
print(f"副本共享内存: {np.shares_memory(embeddings, sub_copy)}")
```

### Batch 处理中的视图使用

在训练数据加载时，通常需要从大数组中提取 batch。使用视图可以避免复制数据：

```python
def create_batch_view(data, batch_indices):
    """创建 batch 视图（不复制数据）

    参数:
        data: 完整数据 (n_samples, ...)
        batch_indices: batch 的索引数组
    返回:
        batch 数据（视图）
    """
    return data[batch_indices]

# 示例
np.random.seed(42)
all_data = np.random.randn(100000, 768)  # 大型数据集
all_labels = np.random.randint(0, 10, 100000)

# 模拟数据加载器
for epoch in range(3):
    indices = np.random.permutation(100000)[:32]
    batch_x = create_batch_view(all_data, indices)
    batch_y = all_labels[indices]  # 整数索引返回副本

    # 修改不会影响原始数据
    if epoch == 0:
        print(f"batch_x 共享内存: {np.shares_memory(all_data, batch_x)}")
        print(f"batch_y 共享内存: {np.shares_memory(all_labels, batch_y)}")
```

### 数据预处理流水线

在数据预处理时，避免不必要的复制可以显著减少内存占用：

```python
def preprocess_batch(data, normalize=True, mean=None, std=None):
    """预处理数据

    注意：这个函数会返回副本，因为进行了数值操作
    """
    result = data.astype(np.float32)  # 总是创建副本

    if normalize:
        if mean is None:
            mean = result.mean(axis=0)
        if std is None:
            std = result.std(axis=0) + 1e-8
        result = (result - mean) / std  # 这会创建新的数组

    return result  # 返回的是副本，不是视图

# 示例
data = np.random.randint(0, 255, (32, 768))
processed = preprocess_batch(data)

print(f"原始数据 dtype: {data.dtype}")
print(f"处理后 dtype: {processed.dtype}")
print(f"共享内存: {np.shares_memory(data, processed)}")  # False
```

## 何时使用副本，何时使用视图

**使用视图的场景**：
- 只需要读取数据，不需要修改
- 需要用不同 shape/stride 访问相同数据
- 希望节省内存

**使用副本的场景**：
- 需要修改数据而不影响原始数据
- 需要改变 dtype 或布局
- 需要传递给不可信代码或外部库

```python
# 需要修改时使用副本
def normalize_in_place(data, target):
    """将数据归一化到目标范围（修改副本）"""
    result = data.copy()
    min_val = result.min()
    max_val = result.max()
    result = (result - min_val) / (max_val - min_val)
    result = result * (target[1] - target[0]) + target[0]
    return result

# 只需要读取时使用视图
def compute_batch_stats(data, indices):
    """计算 batch 的统计量（使用视图）"""
    batch = data[indices]  # 视图
    return {
        'mean': batch.mean(axis=0),
        'std': batch.std(axis=0),
        'min': batch.min(axis=0),
        'max': batch.max(axis=0)
    }
```

## 常见误区

**误区一：以为切片总是安全的**

虽然切片返回视图而不是副本，但在修改数据时需要特别小心。如果不确定是否共享内存，先用 `np.shares_memory()` 检查。

```python
arr = np.arange(12).reshape(3, 4)
view = arr[:, :2]
print(f"切片视图共享内存: {np.shares_memory(arr, view)}")

# 但是：
# arr[view] = value  # 这是修改原始数组
```

**误区二：忘记整数索引返回副本**

当使用整数数组索引时，NumPy 会返回副本。这在大多数情况下是好的，但如果你在循环中频繁进行这种操作，会累积大量内存副本。

```python
# 大小为 1 的数组索引会返回视图
arr = np.arange(12).reshape(3, 4)
single_row = arr[0]  # 返回视图（一维数组）
print(f"单行共享内存: {np.shares_memory(arr, single_row)}")

# 但数组索引返回副本
rows = arr[[0, 1]]  # 返回副本
print(f"多行共享内存: {np.shares_memory(arr, rows)}")
```

**误区三：忽略 dtype 转换会创建副本**

即使 shape 相同，改变 dtype 通常会创建副本：

```python
arr = np.arange(12, dtype=np.float32)
view_as_float64 = arr.view(np.float64)  # 注意：view 只是改变解释方式

# 但转换 dtype 会创建副本
arr_int32 = np.arange(12, dtype=np.int32)
arr_float64 = arr_int32.astype(np.float64)
print(f"astype 共享内存: {np.shares_memory(arr_int32, arr_float64)}")  # False
```

## 性能考量

在处理大型数组时，避免不必要的复制可以显著减少内存占用：

```python
import time

arr = np.random.randn(10000, 768)

# 测量复制 vs 视图的开销
start = time.time()
for _ in range(100):
    view = arr[:, :256]  # 视图
view_time = time.time() - start

start = time.time()
for _ in range(100):
    copy = arr[:, :256].copy()  # 副本
copy_time = time.time() - start

print(f"视图时间: {view_time:.4f}s")
print(f"副本时间: {copy_time:.4f}s")
print(f"副本慢 {copy_time/view_time:.1f}x")
```

## API 总结

| 函数/操作 | 返回类型 | 说明 |
|-----------|---------|------|
| `arr.reshape()` | View | 如果可能，返回视图 |
| `arr[:]` | View | 切片返回视图 |
| `arr[[0,1,2]]` | Copy | 整数索引返回副本 |
| `arr[mask]` | Copy | 布尔索引返回副本 |
| `arr.copy()` | Copy | 显式复制 |
| `np.shares_memory(a, b)` | bool | 检查是否共享内存 |

理解视图与副本的区别是高效使用 NumPy 和深度学习框架的基础。在处理大型语言模型的数据时，正确选择可以显著优化内存使用和性能。
