# 随机排列

随机排列是将一组元素打乱顺序的操作，在深度学习的数据预处理和训练过程中非常重要。当我们有一个训练数据集时，通常需要随机打乱顺序来避免模型学习到数据中的顺序相关性（如时间序列依赖），并确保每个 epoch 中模型看到的数据顺序都不同。NumPy 提供了 `np.random.shuffle` 和 `np.random.permutation` 两个函数来实现随机排列，它们各有特点，适用于不同场景。

## 基本随机排列操作

### np.random.shuffle：原地打乱

`np.random.shuffle` 直接在原数组上进行操作，不返回新数组。这对于内存效率很重要，因为不需要复制整个数组。

```python
import numpy as np

# 打乱一维数组
np.random.seed(42)
arr = np.arange(10)
print(f"打乱前: {arr}")

np.random.shuffle(arr)
print(f"打乱后: {arr}")
```

对于多维数组，`shuffle` 只在第一个轴（axis=0）上打乱，这意味着如果有一个形状为 (n_samples, n_features) 的数据矩阵，打乱会交换整个样本的行，但每行的特征保持不变。

```python
# 打乱多维数组 - 只打乱第一维（样本轴）
np.random.seed(42)
data = np.arange(20).reshape(10, 2)
print(f"原始数据:\n{data}")

np.random.shuffle(data)
print(f"\n打乱后（只打乱行）:\n{data}")
```

### np.random.permutation：返回新数组

`np.random.permutation` 不修改原数组，而是返回一个新的打乱后的数组。这在需要保留原始数据或需要多次使用打乱结果时很有用。

```python
np.random.seed(42)
arr = np.arange(10)
shuffled = np.random.permutation(arr)

print(f"原始数组: {arr}")
print(f"打乱后的副本: {shuffled}")
print(f"原始数组未变: {arr}")
```

`permutation` 同样只对第一个轴进行操作。

## 打乱训练数据

在深度学习中，每个训练 epoch 开始时通常需要打乱数据顺序。这有助于模型泛化，因为它不会学习到数据中的虚假顺序模式。

```python
def create_data_loader(data, batch_size, shuffle=True, seed=None):
    """简单的数据加载器

    参数:
        data: 训练数据 (n_samples, ...)
        batch_size: 批量大小
        shuffle: 是否打乱数据
        seed: 随机种子
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(data)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batches.append(data[batch_indices])

    return batches

# 示例数据
np.random.seed(42)
train_data = np.random.randn(1000, 20)  # 1000 个样本，20 个特征
train_labels = np.random.randint(0, 10, size=1000)

# 创建数据加载器
batches = create_data_loader(train_data, batch_size=32, shuffle=True, seed=42)

print(f"创建了 {len(batches)} 个 batch")
print(f"第一个 batch 的形状: {batches[0].shape}")
```

## 保持数据与标签一致

在实际应用中，数据和标签通常分别存储。打乱时必须确保数据和标签的对应关系不变，否则模型会学到错误的映射。

```python
def shuffle_data_and_labels(data, labels, seed=None):
    """同时打乱数据和标签，保持对应关系

    参数:
        data: 特征数据 (n_samples, ...)
        labels: 标签 (n_samples,)
        seed: 随机种子
    返回:
        shuffled_data, shuffled_labels
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    return data[indices], labels[indices]

# 示例
np.random.seed(42)
train_data = np.random.randn(1000, 20)
train_labels = np.random.randint(0, 10, size=1000)

shuffled_data, shuffled_labels = shuffle_data_and_labels(train_data, train_labels, seed=42)

# 验证对应关系
print("验证数据-标签对应关系:")
for i in range(3):
    print(f"  样本 {i}: 数据均值={shuffled_data[i].mean():.3f}, 标签={shuffled_labels[i]}")
```

## 跨epoch的打乱策略

在训练大型语言模型时，通常需要多个 epoch。每个 epoch 结束时，应该重新打乱数据，但又要确保不同 epoch 之间的打乱是可复现的。

```python
class EpochShuffler:
    """跨 epoch 的打乱管理器"""

    def __init__(self, data, labels, base_seed=42):
        self.data = data
        self.labels = labels
        self.base_seed = base_seed
        self.epoch = 0

    def get_shuffled_data(self):
        """获取当前 epoch 的打乱数据"""
        # 每个 epoch 使用不同的种子，但基于 base_seed
        seed = self.base_seed + self.epoch
        shuffled_data, shuffled_labels = shuffle_data_and_labels(
            self.data, self.labels, seed=seed
        )
        self.epoch += 1
        return shuffled_data, shuffled_labels

# 使用示例
shuffler = EpochShuffler(train_data, train_labels, base_seed=42)

for epoch in range(3):
    shuffled_data, shuffled_labels = shuffler.get_shuffled_data()
    print(f"Epoch {epoch}: 数据前3个标签 = {shuffled_labels[:3]}")
    # 每个 epoch 的打乱顺序都不同
```

## 训练集、验证集、测试集的划分

在划分数据集时，也需要使用随机排列来确保划分的随机性。常见的划分方式有简单随机划分、分层抽样等。

```python
def train_val_test_split(data, labels, train_ratio=0.8, val_ratio=0.1, seed=None):
    """划分训练集、验证集、测试集

    参数:
        data: 特征数据
        labels: 标签
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    返回:
        train_data, val_data, test_data
        train_labels, val_labels, test_labels
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (data[train_indices], data[val_indices], data[test_indices],
            labels[train_indices], labels[val_indices], labels[test_indices])

# 划分数据
np.random.seed(42)
train_x, val_x, test_x, train_y, val_y, test_y = train_val_test_split(
    train_data, train_labels, train_ratio=0.8, val_ratio=0.1, seed=42
)

print(f"训练集大小: {len(train_x)}")
print(f"验证集大小: {len(val_x)}")
print(f"测试集大小: {len(test_x)}")
```

## 时间序列数据的特殊考虑

对于时间序列数据（如文本或时间序列），简单随机打乱可能不合适，因为：
1. 时间序列数据通常有 temporal dependency，打乱可能破坏这种依赖
2. 在预测未来时，不能使用未来的信息

```python
def create_sequential_batches(data, batch_size, step_size=None):
    """创建顺序 batch（用于时间序列）

    适用于 LSTM、Transformer 等时间序列模型
    不打乱数据，保持时间连续性
    """
    if step_size is None:
        step_size = batch_size

    n_samples = len(data)
    batches = []
    for i in range(0, n_samples - batch_size + 1, step_size):
        batch = data[i:i + batch_size]
        batches.append(batch)

    return batches

# 示例：文本序列处理
text_indices = np.arange(1000)  # 伪代码：tokenized text
seq_length = 128

# 顺序创建序列样本（sliding window）
sequences = []
for i in range(0, len(text_indices) - seq_length, seq_length):
    seq = text_indices[i:i + seq_length]
    sequences.append(seq)

print(f"创建了 {len(sequences)} 个序列样本")
print(f"每个序列形状: {sequences[0].shape}")
```

## 常见误区

**误区一：打乱时忘记保持数据-标签对应**

这是最常见的错误之一。当数据被打乱后，标签必须以相同方式被打乱，否则模型会学到完全错误的映射。

**误区二：混淆 shuffle 和 permutation**

`shuffle` 是原地操作，返回 `None`；`permutation` 返回新的打乱数组，保留原数组不变。在需要保留原始数据时使用 `permutation`，在不需要保留时使用 `shuffle` 更省内存。

```python
np.random.seed(42)
arr = np.arange(10)

# shuffle 返回 None
result = np.random.shuffle(arr)
print(f"shuffle 返回: {result}")  # None

# permutation 返回新数组
result = np.random.permutation(arr)
print(f"permutation 返回: {result}")  # 打乱后的数组
```

**误区三：在每个 batch 内部打乱**

应该在 epoch 级别打乱数据，然后按顺序取 batch，而不是在每个 batch 内部再打乱。后者会导致同一 batch 内的样本过于相似，不利于梯度估计。

**误区四：对多维数组的所有轴打乱**

默认情况下，`shuffle` 和 `permutation` 只打乱第一轴。对于 (n_samples, n_features) 的矩阵，这意味着交换整行，而不是打乱行内的元素。这通常是预期的行为，但如果需要在所有轴上打乱，需要特殊处理。

```python
# 只打乱第一轴（样本）
np.random.shuffle(matrix)  # 交换行

# 打乱所有轴
flat = matrix.flatten()
np.random.shuffle(flat)
shuffled_matrix = flat.reshape(matrix.shape)
```

## API 总结

| 函数 | 操作 | 返回值 | 内存效率 |
|------|------|--------|---------|
| `np.random.shuffle(arr)` | 原地打乱 | None | 高 |
| `np.random.permutation(arr)` | 返回打乱副本 | 新数组 | 中 |

理解随机排列的特性和适用场景，对于正确预处理训练数据至关重要。在实际应用中，应该根据具体问题选择合适的打乱策略，而不是盲目打乱。
