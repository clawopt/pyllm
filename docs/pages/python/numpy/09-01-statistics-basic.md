# 常用统计量

统计方法是数据分析的基础工具，NumPy 提供了丰富的统计函数。`mean` 计算均值，`std` 计算标准差，`var` 计算方差，这些是数据分析中最常用的统计量。在深度学习中，这些统计函数有着广泛的应用：LayerNorm 需要计算均值和方差，权重初始化需要了解数据的分布，模型评估需要计算各种指标。理解这些统计量的含义和计算方法，对于正确实现深度学习算法至关重要。

## mean：均值

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(f"均值: {arr.mean()}")
print(f"均值: {np.mean(arr)}")
```

## std：标准差

标准差是方差的平方根，衡量数据的离散程度：

```python
arr = np.array([1, 2, 3, 4, 5])

print(f"标准差: {arr.std()}")  # 默认 ddof=0（总体标准差）
print(f"样本标准差: {arr.std(ddof=1)}")  # 样本标准差
```

ddof 是自由度修正。对于样本标准差，使用 ddof=1 可以给出总体标准差的无偏估计。

## var：方差

方差是标准差的平方：

```python
print(f"方差: {arr.var()}")  # 默认 ddof=0
print(f"样本方差: {arr.var(ddof=1)}")
```

## min 和 max：最小值和最大值

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print(f"最小值: {arr.min()}")
print(f"最大值: {arr.max()}")
print(f"最小值索引: {arr.argmin()}")
print(f"最大值索引: {arr.argmax()}")
```

## ddof 参数详解

ddof（Delta Degrees of Freedom）是统计学中自由度的概念。对于样本标准差和样本方差，使用 ddof=1 可以给出总体参数的无偏估计：

```python
population = np.random.randn(10000)
sample = population[:100]  # 从总体中抽取样本

# 总体标准差
pop_std = population.std()
print(f"总体标准差: {pop_std:.6f}")

# 样本标准差（ddof=0，会低估）
sample_std_wrong = sample.std(ddof=0)
print(f"样本标准差(ddof=0): {sample_std_wrong:.6f}")

# 样本标准差（ddof=1，更接近总体）
sample_std_correct = sample.std(ddof=1)
print(f"样本标准差(ddof=1): {sample_std_correct:.6f}")
```

为什么 ddof=1 更准确？考虑样本 [2, 4, 6]，样本均值是 4。如果计算 Σ(x - x̄)² / n = 8/3 ≈ 2.67；但总体方差的估计应该是 8/2 = 4（因为用样本均值代替总体均值损失了一个自由度）。

## 在LLM场景中的应用

### LayerNorm 计算

LayerNorm 是 Transformer 中的重要组件，它在每个样本内部进行归一化：

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer Normalization

    LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

    其中：
    - μ = mean(x)
    - σ² = var(x)
    - γ 和 β 是可学习的缩放和偏移参数
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
print(f"输入形状: {x.shape}")
print(f"LayerNorm 输出形状: {normalized.shape}")
print(f"归一化后均值（每个样本每位置）: {normalized.mean(axis=-1)[0, 0]:.6f}")
print(f"归一化后方差（每个样本每位置）: {normalized.var(axis=-1)[0, 0]:.6f}")
```

注意这里使用了 `keepdims=True`，确保均值和方差可以正确广播到原始形状。

### 权重初始化分析

分析神经网络权重的分布：

```python
def analyze_weights(weights, name="权重"):
    """分析权重的统计特性"""
    print(f"{name}:")
    print(f"  形状: {weights.shape}")
    print(f"  均值: {weights.mean():.6f}")
    print(f"  标准差: {weights.std():.6f}")
    print(f"  最小值: {weights.min():.6f}")
    print(f"  最大值: {weights.max():.6f}")

# 模拟不同初始化的权重
np.random.seed(42)

# Xavier 初始化
xavier_weights = np.random.randn(768, 3072).astype(np.float32) * np.sqrt(2.0 / (768 + 3072))
analyze_weights(xavier_weights, "Xavier 初始化")

# He 初始化
he_weights = np.random.randn(768, 3072).astype(np.float32) * np.sqrt(2.0 / 768)
analyze_weights(he_weights, "He 初始化")

# 高斯初始化（不当）
bad_weights = np.random.randn(768, 3072).astype(np.float32) * 1.0
analyze_weights(bad_weights, "不当的高斯初始化")
```

### 模型激活值监控

监控神经网络中间层的激活值分布：

```python
def monitor_activations(activations, layer_name="Layer"):
    """监控激活值的分布"""
    print(f"\n{layer_name} 激活值:")
    print(f"  均值: {activations.mean():.6f}")
    print(f"  标准差: {activations.std():.6f}")
    print(f"  最小值: {activations.min():.6f}")
    print(f"  最大值: {activations.max():.6f}")
    print(f"  梯度消失比例 (< 0.01): {(np.abs(activations) < 0.01).mean():.2%}")
    print(f"  梯度爆炸比例 (> 10): {(np.abs(activations) > 10).mean():.2%}")

# 模拟各层激活值
batch_size = 32
seq_len = 512

# 模拟不同层的激活值
layer1_activations = np.random.randn(batch_size * seq_len, 768).astype(np.float32) * 0.02
monitor_activations(layer1_activations, "Embedding 层")

layer2_activations = np.random.randn(batch_size * seq_len, 768).astype(np.float32) * 1.0
monitor_activations(layer2_activations, "注意力层（可能爆炸）")

layer3_activations = np.random.randn(batch_size * seq_len, 768).astype(np.float32) * 0.1
monitor_activations(layer3_activations, "归一化层（正常）")
```

## 常见误区与注意事项

### 误区一：忘记 ddof 的含义

```python
arr = np.array([1, 2, 3, 4, 5])

# 默认 ddof=0，计算的是总体标准差
pop_std = arr.std()
print(f"总体标准差: {pop_std:.6f}")  # 1.414214

# ddof=1，计算的是样本标准差
sample_std = arr.std(ddof=1)
print(f"样本标准差: {sample_std:.6f}")  # 1.581139
```

在深度学习中，通常关心的是数据的实际分布特性，所以使用默认的 ddof=0。

### 误区二：多维数组上忘记指定 axis

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# 错误：默认会计算所有元素的均值
print(f"所有元素均值: {matrix.mean()}")  # 3.5

# 正确：沿轴计算
print(f"每列均值: {matrix.mean(axis=0)}")  # [2.5, 3.5, 4.5]
print(f"每行均值: {matrix.mean(axis=1)}")  # [2., 5.]
```

### 误区三：Variance 与 Mean Squared Error 混淆

```python
arr = np.array([1, 2, 3, 4, 5])

# Variance 是均值的偏离平方的期望
variance = arr.var()

# MSE 是相对于某个值的偏离平方的均值
mse = ((arr - 3) ** 2).mean()  # 假设真实值是 3

print(f"方差: {variance:.6f}")  # 2.0
print(f"MSE (相对3): {mse:.6f}")  # 2.0
```

## 小结

统计方法是数据分析的基础工具。NumPy 提供了 `mean`、`std`、`var`、`min`、`max` 等统计函数。ddof 参数控制自由度修正，样本标准差和样本方差应使用 ddof=1。在 LLM 场景中，这些统计函数用于 LayerNorm、权重初始化分析和模型监控等。

面试时需要能够解释 ddof 的含义和为什么样本标准差使用 ddof=1，理解 LayerNorm 的统计量计算，以及注意在多维数组上使用统计函数时指定 axis。
