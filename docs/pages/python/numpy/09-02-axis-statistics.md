# 沿轴统计

在前一节中，我们学习了基本的统计量计算。在这一节中，让我们深入探讨如何在多维数组上沿不同的轴进行统计计算。轴（axis）的概念是 NumPy 多维数组操作的核心，理解 axis 参数对于正确进行统计计算至关重要。在深度学习中，我们经常需要在特定维度上进行统计：LayerNorm 在特征维度上计算均值和方差，Softmax 在分类维度上计算概率分布，注意力权重的归一化也需要沿特定轴进行。掌握沿轴统计的技巧，是进行正确深度学习实现的基础。

## axis 参数的基本概念

在 NumPy 中，axis 参数指定了沿着哪个维度进行操作：

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print(f"矩阵:\n{matrix}")
print(f"矩阵形状: {matrix.shape}")  # (2, 3)

# axis=0：沿着行的方向操作（对每一列）
print(f"沿 axis=0 求和: {matrix.sum(axis=0)}")  # [5, 7, 9]

# axis=1：沿着列的方向操作（对每一行）
print(f"沿 axis=1 求和: {matrix.sum(axis=1)}")  # [6, 15]
```

可以这样记忆：axis 指定了"哪个维度会被压缩"。

## 三维数组的 axis

对于更高维度的数组，axis 的含义类似：

```python
arr_3d = np.random.randn(4, 8, 16)

# axis=0：压缩第一个维度
print(f"沿 axis=0 形状: {arr_3d.sum(axis=0).shape}")  # (8, 16)

# axis=1：压缩第二个维度
print(f"沿 axis=1 形状: {arr_3d.sum(axis=1).shape}")  # (4, 16)

# axis=2：压缩第三个维度
print(f"沿 axis=2 形状: {arr_3d.sum(axis=2).shape}")  # (4, 8)
```

## keepdims 参数

`keepdims` 参数可以保持被压缩的维度，这对于后续的广播操作非常重要：

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# 默认：被压缩的维度消失
mean_default = matrix.mean(axis=0)
print(f"默认形状: {mean_default.shape}")  # (3,)

# keepdims=True：保持维度
mean_keepdims = matrix.mean(axis=0, keepdims=True)
print(f"keepdims 形状: {mean_keepdims.shape}")  # (1, 3)
```

这个区别在需要广播时非常重要：

```python
# 使用默认形状，无法直接广播
try:
    centered = matrix - mean_default  # 形状不匹配
except ValueError as e:
    print(f"广播失败: {e}")

# 使用 keepdims，可以正确广播
centered = matrix - mean_keepdims
print(f"中心化后:\n{centered}")
```

## 在LLM场景中的应用

### Softmax 归一化

Softmax 需要沿最后一个维度（分类维度）进行归一化：

```python
def softmax(logits, axis=-1):
    """Softmax 函数

    沿 axis 对 logits 进行指数归一化
    """
    exp_logits = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return exp_logits / exp_logits.sum(axis=axis, keepdims=True)

# 测试不同维度的 softmax
logits_1d = np.array([2.0, 1.0, 0.1])
print(f"1D Softmax: {softmax(logits_1d)}")

logits_2d = np.array([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 1.0]])
print(f"2D Softmax (沿 axis=-1):\n{softmax(logits_2d, axis=-1)}")

# 验证每行和为 1
print(f"每行和: {softmax(logits_2d, axis=-1).sum(axis=-1)}")
```

### 注意力权重的归一化

注意力权重需要沿某个维度进行归一化：

```python
batch_size = 4
num_heads = 12
seq_len = 512

# 模拟注意力分数
attention_scores = np.random.randn(batch_size, num_heads, seq_len, seq_len).astype(np.float32)

# 沿最后一个维度（key 维度）进行 softmax
attention_weights = softmax(attention_scores, axis=-1)

print(f"注意力权重形状: {attention_weights.shape}")
print(f"每个头的注意力权重和: {attention_weights[0, 0, 0, :].sum():.6f}")
```

### LayerNorm

LayerNorm 沿特征维度（最后一个维度）计算均值和方差：

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer Normalization

    沿最后一个维度计算均值和方差
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

batch_size = 4
seq_len = 512
hidden_dim = 768

x = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
gamma = np.ones(hidden_dim).astype(np.float32)
beta = np.zeros(hidden_dim).astype(np.float32)

normalized = layer_norm(x, gamma, beta)
print(f"LayerNorm 输出形状: {normalized.shape}")
print(f"每个位置的均值（应为~0）: {normalized[0, 0, :].mean():.6f}")
print(f"每个位置的标准差（应为~1）: {normalized[0, 0, :].std():.6f}")
```

### BatchNorm vs LayerNorm

BatchNorm 沿 batch 维度统计，LayerNorm 沿特征维度统计：

```python
def batch_norm(x, gamma, beta, eps=1e-5, momentum=0.1, running_mean=None, running_var=None):
    """Batch Normalization

    沿 batch 维度（axis=0）计算均值和方差
    用于 CNN 等场景
    """
    if running_mean is not None:
        # 推理模式：使用运行统计量
        mean = running_mean
        var = running_var
    else:
        # 训练模式：使用当前 batch 的统计量
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

# 对比 BatchNorm 和 LayerNorm 的 axis
print("BatchNorm: 沿 axis=0 (batch 维度)")
print("LayerNorm: 沿 axis=-1 (特征维度)")
```

### Cross Entropy 损失

交叉熵损失需要沿类别维度计算：

```python
def cross_entropy_loss(logits, labels):
    """交叉熵损失

    logits: (batch, num_classes) 的未归一化 logits
    labels: (batch,) 的类别索引，或 (batch, num_classes) 的 one-hot 编码
    """
    # Log-Softmax 沿类别维度计算
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True) + 1e-10)

    # 如果 labels 是索引，转为 one-hot
    if labels.ndim == 1:
        num_classes = logits.shape[-1]
        labels_onehot = np.eye(num_classes)[labels]

    # 交叉熵
    loss = -(labels_onehot * log_probs).sum(axis=-1).mean()

    return loss

# 测试
logits = np.array([[2.0, 1.0, 0.1],
                    [0.5, 2.5, 1.0]])
labels = np.array([0, 1])

loss = cross_entropy_loss(logits, labels)
print(f"交叉熵损失: {loss:.4f}")
```

## 多个轴同时统计

有时需要在多个轴上同时统计：

```python
arr = np.random.randn(4, 8, 16)

# 沿多个轴计算
print(f"沿 (0, 1) 求和形状: {arr.sum(axis=(0, 1)).shape}")  # (16,)
print(f"沿 (1, 2) 求和形状: {arr.sum(axis=(1, 2)).shape}")  # (4,)
print(f"沿所有轴求和: {arr.sum()}")  # 标量
```

## nan 安全函数

当数据中存在 NaN 时，需要使用 nan 安全版本：

```python
arr_with_nan = np.array([[1, 2, np.nan], [4, np.nan, 6]])

# 普通函数会返回 NaN
print(f"普通 mean: {arr_with_nan.mean()}")  # nan

# nan 安全版本会忽略 NaN
print(f"nanmean: {np.nanmean(arr_with_nan)}")
```

## 常见误区与注意事项

### 误区一：混淆 axis 的含义

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# axis=0：对每列操作
print(f"axis=0 (每列): {matrix.sum(axis=0)}")  # [5, 7, 9]

# axis=1：对每行操作
print(f"axis=1 (每行): {matrix.sum(axis=1)}")  # [6, 15]
```

### 误区二：忘记 keepdims

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

mean = matrix.mean(axis=0)  # shape: (3,)
print(f"mean shape: {mean.shape}")

# 直接相减会失败
try:
    centered = matrix - mean
except ValueError as e:
    print(f"形状不匹配: {e}")

# 应该使用 keepdims
mean_keepdims = matrix.mean(axis=0, keepdims=True)  # shape: (1, 3)
centered = matrix - mean_keepdims  # 正确广播
print(f"中心化成功: {centered.shape}")
```

### 误区三：对高维数组使用负索引

```python
arr_3d = np.random.randn(4, 8, 16)

# axis=-1 是最后一个轴，即 axis=2
print(f"axis=-1 等于 axis=2: {(arr_3d.sum(axis=-1).shape)}")
```

## 小结

沿轴统计是 NumPy 多维数组操作的核心。axis 参数指定了哪个维度被压缩，keepdims 参数可以保持被压缩的维度以便后续广播。在 LLM 场景中，Softmax 沿类别维度归一化，LayerNorm 沿特征维度计算均值和方差，交叉熵损失沿类别维度计算。掌握 axis 的用法对于正确实现深度学习算法至关重要。

面试时需要能够解释 axis 参数的含义，理解 keepdims 的作用，以及能够正确地在特定维度上进行统计计算。
