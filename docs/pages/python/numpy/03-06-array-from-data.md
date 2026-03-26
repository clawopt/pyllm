# 全自定义：np.array 与 np.asarray

在NumPy中，创建数组最直接的方式就是使用 `np.array` 或 `np.asarray` 从已有的数据（通常是Python列表或其他可迭代对象）来创建数组。这两个函数是NumPy中最基础也最灵活的数组创建方法，它们可以将任何序列数据转换为NumPy数组。在深度学习中，当你从文件加载数据、将tokenized的文本转换为模型输入、或将预处理后的数据整理成批次时，都需要用到这些函数。理解 `np.array` 和 `np.asarray` 的区别，以及如何正确指定dtype和数据类型，是使用NumPy的基础。

## np.array：从列表创建数组

`np.array` 是创建NumPy数组最直接的方式：

```python
import numpy as np

# 从一维列表创建
list_1d = [1, 2, 3, 4, 5]
arr_1d = np.array(list_1d)
print(f"一维数组: {arr_1d}")

# 从二维列表创建
list_2d = [[1, 2, 3], [4, 5, 6]]
arr_2d = np.array(list_2d)
print(f"二维数组:\n{arr_2d}")

# 指定数据类型
arr_float = np.array([1, 2, 3], dtype=np.float32)
print(f"指定float32: {arr_float.dtype}")
```

`np.array` 会完整地复制输入数据，创建一个新的数组。这意味着修改新数组不会影响原始列表，反之亦然。

## np.asarray：高效转换数组

`np.asarray` 与 `np.array` 类似，但行为略有不同：

```python
# 如果输入已经是 ndarray，不会复制数据
existing_arr = np.array([1, 2, 3])
new_arr = np.asarray(existing_arr)
print(f"共享内存: {np.shares_memory(existing_arr, new_arr)}")  # True

# 如果输入是列表，会创建新的数组
list_input = [1, 2, 3]
arr_from_list = np.asarray(list_input)
print(f"列表转换: {arr_from_list}")
```

`np.asarray` 不会复制输入数据（如果输入已经是ndarray），而 `np.array` 总是会复制。这是一个重要的性能优化点，特别是在处理大型数据时。

## 两者关键区别

面试中经常被问到："np.array 和 np.asarray 的区别是什么？"关键区别在于对已有ndarray的处理：

```python
# np.array 接受 copy 参数，可以强制复制
arr1 = np.array([1, 2, 3])
arr2 = np.array(arr1, copy=True)

# np.asarray 没有 copy 参数
arr3 = np.asarray(arr1)
print(f"asarray 不复制: {np.shares_memory(arr1, arr3)}")  # True

# 但可以通过其他方式强制复制
arr4 = np.asarray(arr1, dtype=arr1.dtype)  # 如果dtype相同，不会复制
arr5 = np.asarray(arr1, dtype=np.float64)  # dtype改变时会复制
print(f"dtype改变时复制: {not np.shares_memory(arr1, arr5)}")  # True
```

## dtype 的重要性

在深度学习中，正确指定dtype非常重要：

```python
# 从整数列表创建浮点数数组
int_list = [1, 2, 3, 4, 5]
arr_from_int = np.array(int_list, dtype=np.float32)
print(f"整数列表->float32: {arr_from_int.dtype}")

# 从token IDs创建嵌入查找表
token_ids = [101, 2003, 1037, 3010]  # 假设是词表中的token ID
arr_tokens = np.array(token_ids)
print(f"Token IDs dtype: {arr_tokens.dtype}")  # int64

# 在深度学习推理时，通常需要float32
arr_tokens_f32 = np.array(token_ids, dtype=np.int64)  # 保持为int64用于索引
print(f"保持int64: {arr_tokens_f32.dtype}")
```

dtype不仅影响内存占用，还影响计算精度和速度。GPU对float32的计算效率最高。

## copy 参数

`np.array` 有一个 `copy` 参数，可以控制是否复制数据：

```python
# 默认 copy=True，会复制数据
arr1 = np.array([1, 2, 3])
arr2 = np.array(arr1)
arr2[0] = 100
print(f"原始数组: {arr1}")  # 不受影响

# copy=False，如果可能就不复制
arr3 = np.array(arr1, copy=False)
arr3[0] = 100
print(f"原始数组: {arr1}")  # 被修改了！
```

## 从文件或外部数据创建数组

在深度学习中，经常需要从文件加载数据：

```python
# 模拟从CSV加载数据
csv_data = "1,2,3\n4,5,6\n7,8,9"
import io

# 使用 np.loadtxt 从文本加载
arr = np.loadtxt(io.StringIO(csv_data), delimiter=',')
print(f"从CSV加载:\n{arr}")

# 或者使用 np.genfromtxt
arr2 = np.genfromtxt(io.StringIO(csv_data), delimiter=',')
print(f"从CSV加载(genfromtxt):\n{arr2}")
```

`np.loadtxt` 适用于数值数据，`np.genfromtxt` 还可以处理缺失值。

## 嵌套列表与形状推断

`np.array` 会自动推断数组的形状：

```python
# 嵌套列表创建多维数组
nested_list = [[1, 2, 3], [4, 5, 6]]
arr = np.array(nested_list)
print(f"形状: {arr.shape}")

# 如果嵌套层级不一致，会创建对象数组
mixed_list = [[1, 2], [3, 4, 5]]  # 不规则的嵌套列表
arr_mixed = np.array(mixed_list, dtype=object)
print(f"对象数组: {arr_mixed}")
print(f"对象数组形状: {arr_mixed.shape}")
```

不规则的嵌套列表会导致创建 object 类型的数组，这会失去NumPy向量化计算的优势。

## order 参数：内存布局

`np.array` 有一个 `order` 参数，控制数组的内存布局：

```python
arr_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# C-order（行优先，默认）
arr_c = np.array(arr_2d, order='C')
print(f"C-order strides: {arr_c.strides}")

# Fortran-order（列优先）
arr_f = np.array(arr_2d, order='F')
print(f"Fortran-order strides: {arr_f.strides}")
```

C-order 意味着同一行的元素在内存中是连续的，Fortran-order 意味着同一列的元素在内存中是连续的。大多数情况下默认的C-order就够了，但如果要与Fortran编写的库交互，可能需要Fortran-order。

## ndmin 参数：最小维度

`np.array` 有一个 `ndmin` 参数，指定数组的最小维度：

```python
# 默认最小维度为1
arr1 = np.array([1, 2, 3])
print(f"ndmin=1: {arr1.ndim}")

# 强制至少二维
arr2 = np.array([1, 2, 3], ndmin=2)
print(f"ndmin=2:\n{arr2}")

# 强制至少三维
arr3 = np.array([1, 2, 3], ndmin=3)
print(f"ndmin=3: {arr3.shape}")
```

这个参数在某些需要确保数组维度的情况下很有用。

## 在深度学习中的应用

### 从tokenized数据创建输入张量

```python
# 模拟 tokenized 文本数据
token_ids_list = [
    [101, 2003, 1037, 3010, 102],  # "hello world"
    [101, 1045, 2296, 3288, 102]   # "i love nlp"
]

# 创建输入张量
input_ids = np.array(token_ids_list, dtype=np.int64)
print(f"输入张量形状: {input_ids.shape}")
print(f"输入张量dtype: {input_ids.dtype}")

# 创建注意力掩码（有效位置为1，padding为0）
attention_mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.int64)
print(f"注意力掩码:\n{attention_mask}")
```

### 从预训练嵌入创建嵌入矩阵

```python
# 假设我们有一些预训练的词向量
pretrained_embeddings = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
], dtype=np.float32)

vocab_size, embed_dim = pretrained_embeddings.shape
print(f"嵌入矩阵形状: ({vocab_size}, {embed_dim})")

# 用于模型的嵌入查找
embedding_matrix = pretrained_embeddings
token_ids = np.array([0, 1, 2])
embeddings = embedding_matrix[token_ids]
print(f"查找到的嵌入:\n{embeddings}")
```

## 常见误区与注意事项

### 误区一：修改副本影响原数组（np.array 的 copy 默认行为）

```python
original_list = [1, 2, 3]
arr = np.array(original_list)
arr[0] = 100
print(f"列表: {original_list}")  # 不受影响
print(f"数组: {arr}")
```

实际上 `np.array` 默认会复制数据，所以这个例子不会影响原始列表。但如果误以为修改数组会影响列表，可能导致逻辑错误。

### 误区二：混淆 np.asarray 和 np.array 在处理已有数组时的行为

```python
# arr1 和 arr2 共享内存（如果是ndarray输入）
arr1 = np.array([1, 2, 3])
arr2 = np.asarray(arr1)
arr2[0] = 999
print(f"arr1: {arr1}")  # 被修改了！

# 如果不确定，使用 np.copy
arr3 = np.array(arr1, copy=True)
arr3[0] = 111
print(f"arr1: {arr1}")  # 不受影响
```

### 误区三：dtype 不匹配导致的问题

```python
# 在索引时，dtype 必须匹配
embedding_matrix = np.random.randn(1000, 768).astype(np.float32)
token_ids = np.array([0, 1, 2], dtype=np.int64)

# 可以正常索引
embeddings = embedding_matrix[token_ids]
print(f"嵌入形状: {embeddings.shape}")

# 如果dtype不匹配，可能报错或得到错误结果
wrong_ids = np.array([0, 1, 2], dtype=np.float32)
try:
    embeddings_wrong = embedding_matrix[wrong_ids]
except IndexError as e:
    print(f"索引错误: {e}")
```

## 底层原理

`np.array` 的底层实现会先解析输入序列的结构，确定数组的形状和数据类型，然后分配连续的内存块，最后将数据从Python对象转换为NumPy的原生数据类型。这个过程中，数据会被完整复制一次。

`np.asarray` 的优化在于：如果输入已经是同类型的ndarray，它会直接返回输入数组的视图（view），而不复制数据。这节省了内存和时间。

## 小结

`np.array` 和 `np.asarray` 是从已有数据创建NumPy数组的两个基本函数。`np.array` 总是复制数据，但可以通过 `copy` 参数控制；`np.asarray` 在输入已经是ndarray时会尽量避免复制。正确理解和使用这两个函数，以及正确指定 dtype，是在深度学习中处理数据的基础。

面试时需要能够解释两者的区别，理解 copy 参数的作用，以及知道 dtype 在深度学习中的重要性。
