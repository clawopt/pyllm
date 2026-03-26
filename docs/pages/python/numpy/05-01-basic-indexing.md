# 基础索引

索引与切片是NumPy中最基础也最重要的数据访问操作。想象一下，当你有一个包含成千上万个token的序列时，你需要提取其中的某一部分进行处理；或者当你有一个批次的嵌入向量时，你需要获取特定位置的向量。基础索引允许你通过指定具体的索引位置来访问数组中的元素。与Python列表的索引类似，NumPy的索引从0开始，支持负数索引（从末尾倒数），还支持切片（获取一个范围的元素）。理解基础索引是掌握NumPy的第一步，也是后续学习更复杂索引方式的基础。

## 一维数组的索引

一维数组的索引和Python列表几乎完全一样：

```python
import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"原始数组: {arr}")

# 正向索引从0开始
print(f"arr[0]: {arr[0]}")  # 第一个元素
print(f"arr[3]: {arr[3]}")  # 第四个元素

# 负数索引从-1开始，表示最后一个元素
print(f"arr[-1]: {arr[-1]}")  # 最后一个元素
print(f"arr[-3]: {arr[-3]}")  # 倒数第三个元素

# 索引越界会报错
try:
    print(arr[100])
except IndexError as e:
    print(f"索引越界: {e}")
```

一维数组的索引是最简单的，每个索引对应数组中的一个元素。负数索引是一个非常有用的特性，它让你能够从数组末尾开始计数，而不需要知道数组的具体长度。

## 一维数组的切片

切片用于获取数组的一个子范围，返回的是一个新的数组：

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 切片语法：arr[start:stop:step]
print(f"arr[2:6]: {arr[2:6]}")  # 索引2,3,4,5（不包含6）
print(f"arr[::2]: {arr[::2]}")  # 每隔一个元素
print(f"arr[::-1]: {arr[::-1]}")  # 反转数组

# 省略参数使用默认值
print(f"arr[5:]: {arr[5:]}")  # 从索引5到末尾
print(f"arr[:5]: {arr[:5]}")  # 从开始到索引4

# 负数切片
print(f"arr[-5:-1]: {arr[-5:-1]}")  # 倒数第5个到倒数第2个
```

切片的三个要素是 start（起始位置，包含）、stop（结束位置，不包含）和 step（步长）。所有参数都可以省略，使用默认值：start 默认为0，stop 默认为数组末尾，step 默认为1。

## 二维数组的索引

二维数组的索引需要用逗号分隔行索引和列索引：

```python
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
print(f"二维数组:\n{matrix}")

# 获取单个元素
print(f"matrix[0, 0]: {matrix[0, 0]}")  # 第0行第0列
print(f"matrix[1, 2]: {matrix[1, 2]}")  # 第1行第2列
print(f"matrix[2, -1]: {matrix[2, -1]}")  # 第2行最后一列

# 获取一行（返回一维数组）
print(f"matrix[0]: {matrix[0]}")  # 第一行
print(f"matrix[0, :]: {matrix[0, :]}")  # 同上，显式写法

# 获取一列
print(f"matrix[:, 0]: {matrix[:, 0]}")  # 第一列
```

索引的顺序是先行后列，这与数学中的矩阵表示是一致的。

## 二维数组的切片

切片可以同时作用于行和列：

```python
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

# 切片行
print(f"matrix[1:3]:\n{matrix[1:3]}")  # 第1和第2行

# 同时切片行和列
print(f"matrix[1:3, 1:3]:\n{matrix[1:3, 1:3]}")  # 提取子矩阵

# 提取特定列
print(f"matrix[:, 0]: {matrix[:, 0]}")  # 所有行的第0列
print(f"matrix[:, 1::2]:\n{matrix[:, 1::2]}")  # 奇数列

# 复杂切片
print(f"matrix[::2, ::2]:\n{matrix[::2, ::2]}")  # 每隔一行一列
```

切片操作非常强大，可以让你精确地提取数组的任意子区域。

## 三维及更高维数组的索引

三维数组的索引需要指定三个维度：

```python
arr_3d = np.arange(24).reshape(2, 3, 4)
print(f"三维数组形状: {arr_3d.shape}")

# 获取单个元素
print(f"arr_3d[0, 1, 2]: {arr_3d[0, 1, 2]}")

# 获取一个切片
print(f"arr_3d[0, :, :]:\n{arr_3d[0, :, :]}")  # 第一个"块"的全部内容
print(f"arr_3d[:, 0, :]:\n{arr_3d[:, 0, :]}")  # 所有块的第0行

# 切片与索引混合
print(f"arr_3d[1, 0:2, 1:3]:\n{arr_3d[1, 0:2, 1:3]}")
```

索引规则是一样的：每个维度用逗号分隔，可以是整数（精确索引）或切片（范围选择）。

## 在LLM场景中的应用

### 提取句子中特定位置的token

在NLP任务中，经常需要提取句子中特定位置的token或嵌入：

```python
batch_size = 4
seq_len = 512
embed_dim = 768

# 模拟一批token嵌入
token_embeddings = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
print(f"Token嵌入形状: {token_embeddings.shape}")

# 提取第一个样本的 [CLS] token（通常在位置0）
cls_token = token_embeddings[0, 0, :]
print(f"[CLS] token形状: {cls_token.shape}")

# 提取第一个样本的最后位置token（通常是 [SEP]）
sep_token = token_embeddings[0, -1, :]
print(f"[SEP] token形状: {sep_token.shape}")

# 提取第一个样本的前5个位置
first_five = token_embeddings[0, :5, :]
print(f"前5个token形状: {first_five.shape}")
```

### 提取多头注意力的特定头

在分析多头注意力时，可能需要提取特定的注意力头：

```python
batch_size = 2
num_heads = 12
seq_len = 128
head_dim = 64

# 模拟注意力输出 (batch, num_heads, seq_len, head_dim)
attention_output = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

# 提取第一个样本的第3个头
head_3 = attention_output[0, 3, :, :]
print(f"第3个头的输出形状: {head_3.shape}")

# 提取前4个头
first_4_heads = attention_output[0, :4, :, :]
print(f"前4个头的形状: {first_4_heads.shape}")

# 提取特定位置范围的注意力
pos_range = attention_output[0, :, 10:20, :]
print(f"位置10-19的注意力形状: {pos_range.shape}")
```

### 提取序列的子范围

在处理长序列时，可能需要提取特定的子范围：

```python
seq_len = 2048
hidden_dim = 1024

# 模拟一个长序列的隐藏状态
hidden_states = np.random.randn(seq_len, hidden_dim).astype(np.float32)

# 提取前512个位置（通常包含最重要的信息）
first_part = hidden_states[:512, :]
print(f"前512位置形状: {first_part.shape}")

# 提取中间512个位置
middle_part = hidden_states[768:1280, :]
print(f"中间位置形状: {middle_part.shape}")

# 提取最后512个位置
last_part = hidden_states[-512:, :]
print(f"最后512位置形状: {last_part.shape}")
```

## 切片是视图而非副本

这是NumPy中一个极其重要的概念：数组切片返回的是视图（view），而不是副本（copy）。这意味着修改切片会影响原始数组：

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
slice_view = arr[2:6]

print(f"原始切片: {slice_view}")
print(f"切片base: {slice_view.base is arr}")  # True，共享底层内存

# 修改切片
slice_view[0] = 99
print(f"修改后原始数组: {arr}")  # arr[2] 变成了99
```

这个特性既有优点也有缺点。优点是节省内存，可以高效处理大型数组；缺点是可能不小心修改了不应该修改的数据。如果需要副本，使用 `.copy()` 方法。

## 常见误区与注意事项

### 误区一：切片是左闭右开区间

```python
arr = np.array([0, 1, 2, 3, 4])

# arr[0:3] 返回索引0, 1, 2，而不是0, 1, 2, 3
print(f"arr[0:3]: {arr[0:3]}")  # [0, 1, 2]

# 想要包含索引3，应该用 arr[0:4]
print(f"arr[0:4]: {arr[0:4]}")  # [0, 1, 2, 3]
```

这是Python的标准约定，初学者经常在这里犯错。

### 误区二：切片越界不会报错

```python
arr = np.array([0, 1, 2, 3, 4])

# 切片越界不会报错
print(f"arr[1:100]: {arr[1:100]}")  # 只返回有效的部分
print(f"arr[10:]: {arr[10:]}")  # 空数组

# 但索引越界会报错
try:
    print(arr[10])
except IndexError as e:
    print(f"索引越界错误: {e}")
```

NumPy对切片很宽容，即使越界也只会返回有效的部分。

### 误区三：多维数组索引顺序混淆

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# matrix[0, :] 是第一行
# matrix[:, 0] 是第一列
print(f"第一行: {matrix[0, :]}")  # [1, 2, 3]
print(f"第一列: {matrix[:, 0]}")  # [1, 4]
```

逗号前后的含义不同，前面是行，后面是列。

## 小结

基础索引与切片是NumPy中最基本的数据访问操作。一维数组的索引和切片与Python列表类似，但二维及多维数组需要用逗号分隔不同维度的索引。最重要的概念是：切片返回的是视图而不是副本，这在某些情况下是优势（节省内存），在另一些情况下可能导致意外修改。

在LLM场景中，基础索引广泛用于：提取特定位置的token或嵌入、分析多头注意力的特定头、提取序列的子范围等。掌握这些基础操作是处理各种深度学习数据的基础。

面试时需要能够解释切片是视图而非副本的原理，理解多维数组的索引方式，以及注意切片越界和负索引等特殊情况的处理。
