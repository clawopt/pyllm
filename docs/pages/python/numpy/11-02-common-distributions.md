# 常见随机分布

随机分布在深度学习中扮演着核心角色，特别是在神经网络权重初始化和数据增强阶段。不同的分布具有不同的统计特性，选择合适的分布直接影响模型的收敛速度和最终性能。NumPy 的 `np.random` 模块提供了多种随机分布的生成函数，包括均匀分布、正态分布（高斯分布）、二项分布、泊松分布等。理解这些分布的特性和适用场景，能帮助你在初始化权重、生成合成数据、或实现随机增强时做出更好的选择。

## 均匀分布

均匀分布是最简单的连续分布。在区间 [a, b] 上的均匀分布，意味着在这个区间内的任何值出现的概率密度是常数。均匀分布常用于需要随机初始化的场景，以及需要生成指定范围内随机数的场景。

### np.random.uniform

```python
import numpy as np

# 生成 [0, 1) 区间的均匀分布随机数
np.random.seed(42)
samples = np.random.uniform(0, 1, (3, 4))
print(f"均匀分布样本 [0,1):\n{samples}")

# 生成 [low, high) 区间的均匀分布
samples = np.random.uniform(-1, 1, (3, 4))
print(f"\n均匀分布样本 [-1,1):\n{samples}")
```

**在LLM中的应用**：在早期的神经网络实践中，有时会使用均匀分布来初始化权重。但现代实践中更常用的是基于正态分布的初始化方法（如 Xavier 或 He 初始化），因为它们能更好地控制权重的方差。

## 正态分布（高斯分布）

正态分布是统计学中最重要的分布，自然界和社会现象中大量随机变量都近似服从正态分布。在深度学习中，正态分布是权重初始化的基础——无论是 Xavier 初始化还是 He 初始化，都假设权重来自正态分布。

### np.random.normal

```python
# 生成标准正态分布（均值=0，标准差=1）的样本
np.random.seed(42)
samples = np.random.normal(0, 1, (3, 4))
print(f"标准正态分布样本:\n{samples}")

# 生成任意均值和标准差的正态分布
samples = np.random.normal(loc=5.0, scale=2.0, size=(3, 4))
print(f"\n均值=5, 标准差=2 的正态分布样本:\n{samples}")
```

### np.random.randn（已弃用，推荐用 normal）

`np.random.randn` 是生成标准正态分布样本的旧函数，现在推荐使用 `np.random.normal(size=...)` 替代。虽然 `randn` 仍然可用，但新代码应该使用 `normal` 以获得更清晰的语义。

```python
# randn 的用法（不推荐）
old_way = np.random.randn(3, 4)

# 推荐用法
new_way = np.random.normal(size=(3, 4))
```

## 截断正态分布

在神经网络初始化中，截断正态分布（Truncated Normal Distribution）比普通正态分布更常用。截断正态分布将正态分布的值限制在某个范围内，避免生成过小的值或过大的值。这对于防止梯度消失或爆炸很有帮助。

### 手动实现截断正态分布

NumPy 没有直接提供截断正态分布的函数，但可以用其他方法实现：

```python
def truncated_normal(shape, mean=0.0, std=1.0, lo=-2.0, hi=2.0):
    """生成截断正态分布的样本

    参数:
        shape: 输出形状
        mean: 正态分布的均值
        std: 正态分布的标准差
        lo: 下界（截断点）
        hi: 上界（截断点）
    """
    samples = np.random.normal(mean, std, shape)

    # 截断到 [lo, hi] 范围
    samples = np.clip(samples, lo, hi)
    return samples

# 生成截断正态分布样本
weights = truncated_normal((100, 200), mean=0, std=0.02, lo=-0.05, hi=0.05)
print(f"截断正态分布形状: {weights.shape}")
print(f"样本范围: [{weights.min():.4f}, {weights.max():.4f}]")
```

TensorFlow 等框架在权重初始化时默认使用截断正态分布。例如，TensorFlow 的 `tf.truncated_normal_initializer` 默认截断范围是 ±2 个标准差。

## 其他常用分布

### 二项分布

二项分布描述 n 次独立试验中成功次数的分布，每次试验成功概率为 p。

```python
# 二项分布：n=10次试验，p=0.5 成功概率
np.random.seed(42)
samples = np.random.binomial(n=10, p=0.5, size=1000)
print(f"二项分布均值: {samples.mean():.2f} (理论值: {10*0.5})")
print(f"二项分布标准差: {samples.std():.2f} (理论值: {np.sqrt(10*0.5*0.5):.2f})")
```

二项分布在模拟离散事件时有用，但在深度学习中应用相对较少。

### 泊松分布

泊松分布描述单位时间内随机事件发生次数的分布，参数 λ 是平均发生率。

```python
# 泊松分布：λ=5（平均发生率）
np.random.seed(42)
samples = np.random.poisson(lam=5, size=1000)
print(f"泊松分布均值: {samples.mean():.2f} (理论值: 5)")
print(f"泊松分布方差: {samples.var():.2f} (理论值: 5)")
```

泊松分布在模拟稀疏事件或计数数据时有用，偶尔也会用于深度学习的某些变体（如 Poisson 损失用于泊松回归问题）。

## 分布参数与形状

NumPy 的随机函数都支持多种参数组合来控制输出形状：

```python
# 标量参数：输出单个值
single = np.random.normal(0, 1)
print(f"单个样本: {single}")

# 元组参数：输出指定形状的数组
matrix = np.random.normal(0, 1, (3, 4))
print(f"矩阵形状: {matrix.shape}")

# 多维形状
tensor = np.random.normal(0, 1, (2, 3, 4, 5))
print(f"张量形状: {tensor.shape}")
```

## 分布的可视化理解

```python
# 绘制不同分布的直方图（伪代码示意）
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 均匀分布
uniform_data = np.random.uniform(0, 1, 10000)
axes[0, 0].hist(uniform_data, bins=50, density=True, alpha=0.7)
axes[0, 0].set_title('均匀分布 U(0,1)')

# 正态分布
normal_data = np.random.normal(0, 1, 10000)
axes[0, 1].hist(normal_data, bins=50, density=True, alpha=0.7)
axes[0, 1].set_title('正态分布 N(0,1)')

# 标准差对正态分布的影响
normal_std1 = np.random.normal(0, 1, 10000)
normal_std2 = np.random.normal(0, 2, 10000)
axes[1, 0].hist(normal_std1, bins=50, density=True, alpha=0.5, label='σ=1')
axes[1, 0].hist(normal_std2, bins=50, density=True, alpha=0.5, label='σ=2')
axes[1, 0].legend()
axes[1, 0].set_title('不同标准差的正态分布')

# 截断正态分布
truncated = np.clip(np.random.normal(0, 1, 10000), -2, 2)
axes[1, 1].hist(truncated, bins=50, density=True, alpha=0.7)
axes[1, 1].set_title('截断正态分布 (±2σ)')
```

## 在LLM场景中的应用

### 参数初始化

在训练神经网络时，权重初始化的分布直接影响训练稳定性。太小的初始权重会导致梯度消失，太大的初始权重会导致梯度爆炸。现代初始化方法（如 Xavier 和 He 初始化）会根据地在前向传播和反向传播中信号方差的变化来选择合适的初始权重分布。

```python
def xavier_initialization(shape, rng=None):
    """Xavier/Glorot 初始化

    从 N(0, sqrt(2/(fan_in + fan_out))) 生成权重
    """
    if rng is None:
        rng = np.random.default_rng()
    fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else 1
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, shape)

def he_initialization(shape, rng=None):
    """He 初始化

    从 N(0, sqrt(2/fan_in)) 生成权重，适用于 ReLU 激活函数
    """
    if rng is None:
        rng = np.random.default_rng()
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, shape)

# 示例：初始化 Transformer 层的权重
rng = np.random.default_rng(42)
embedding_weights = xavier_initialization((50257, 768), rng)  # 词汇嵌入
attention_weights = he_initialization((768, 768), rng)       # 注意力权重

print(f"嵌入层权重形状: {embedding_weights.shape}")
print(f"嵌入层权重均值: {embedding_weights.mean():.6f}")
print(f"嵌入层权重标准差: {embedding_weights.std():.6f}")
```

### Dropout 与随机屏蔽

Dropout 是一种正则化技术，在训练时随机将部分神经元置零。虽然实际的 Dropout 实现更复杂，但基本思想是生成一个与输入形状相同的随机掩码。

```python
def dropout_mask(shape, p=0.5, rng=None):
    """生成 Dropout 掩码

    参数:
        shape: 掩码形状
        p: 置零概率
        rng: 随机数生成器
    返回:
        mask: 布尔数组，为 True 的位置保留
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.uniform(0, 1, shape) > p
    return mask

# 测试 Dropout 掩码
x = np.random.randn(32, 768)  # batch_size=32, hidden_dim=768
mask = dropout_mask(x.shape, p=0.1)
x_dropped = x * mask

print(f"原始激活值形状: {x.shape}")
print(f"保留比例: {mask.mean():.2%}")
```

## 常见误区

**误区一：混淆均匀分布的边界**

`np.random.uniform(a, b)` 生成的随机数在 [a, b) 区间内，即包含 a 但不包含 b。如果需要包含 b 的区间，可以使用 `np.random.uniform(a, b + epsilon)` 或在生成后处理。

```python
# 默认是半开区间 [a, b)
samples = np.random.uniform(0, 1, 10000)
print(f"最大值是否为1: {(samples == 1).any()}")  # False

# 如果需要包含右端点
samples_inclusive = np.random.uniform(0, 1.0000001, 10000)
```

**误区二：忘记设置种子**

在需要复现性的场景中，一定要记住设置种子。不同的随机种子会产生完全不同的随机序列，这可能导致实验结果不可复现。

**误区三：使用已弃用的 API**

`np.random.rand` 和 `np.random.randn` 是旧 API，推荐使用 `np.random.default_rng()` 创建生成器，然后用 `rng.uniform()` 或 `rng.normal()` 生成随机数。新 API 更安全，不会有全局状态污染的问题。

## API 总结

| 函数 | 描述 | 典型用途 |
|------|------|---------|
| `rng.uniform(low, high, size)` | 均匀分布 | 通用随机数生成 |
| `rng.normal(loc, scale, size)` | 正态分布 | 权重初始化 |
| `rng.binomial(n, p, size)` | 二项分布 | 离散事件模拟 |
| `rng.poisson(lam, size)` | 泊松分布 | 计数数据建模 |

理解不同分布的特性和适用场景，是掌握深度学习中随机性操作的基础。在实际应用中，应该根据具体问题选择合适的分布，而不是盲目使用某种分布。
