# 取整与截断函数

在数值计算和数据处理中，我们经常需要将浮点数转换为整数，或者将数值限制在某个范围内。NumPy 提供了多种取整和截断函数，包括 `np.round`、`np.floor`、`np.ceil`、`np.trunc` 和 `np.clip`。这些函数在深度学习中有着广泛的应用：`np.clip` 用于梯度裁剪防止梯度爆炸，`np.round` 用于量化操作，`np.floor` 和 `np.ceil` 在某些算法中用于计算索引。理解这些函数的用法和区别，对于编写正确的数值计算代码非常重要。

## np.round：四舍五入

`np.round` 对浮点数进行四舍五入：

```python
import numpy as np

x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])

print(f"原始值: {x}")
print(f"round: {np.round(x)}")
```

注意 NumPy 使用的是"银行家舍入"（round half to even）：当要舍入的数字恰好是 .5 时，会舍入到最近的偶数。这与 Python 内置的 round 行为一致，也是 IEEE 754 标准推荐的做法。

## np.floor：向下取整

`np.floor` 返回小于或等于原始值的最大整数：

```python
x = np.array([-2.9, -1.5, -0.5, 0.5, 1.5, 2.9])

print(f"原始值: {x}")
print(f"floor: {np.floor(x)}")  # [-3., -2., -1., 0., 1., 2.]
```

## np.ceil：向上取整

`np.ceil` 返回大于或等于原始值的最小整数：

```python
print(f"ceil: {np.ceil(x)}")  # [-2., -1., 0., 1., 2., 3.]
```

## np.trunc：截断小数部分

`np.trunc` 只保留整数部分，直接丢弃小数：

```python
print(f"trunc: {np.trunc(x)}")  # [-2., -1., 0., 0., 1., 2.]
```

对于正数，trunc 和 floor 相同；对于负数，trunc 和 ceil 相同。

## np.clip：限制值域

`np.clip` 是最常用的截断函数，将数组中的值限制在 [min, max] 范围内：

```python
x = np.array([-3, -1, 0, 1, 2, 5, 10])

print(f"原始值: {x}")
print(f"clip(x, 0, 5): {np.clip(x, 0, 5)}")  # [0, 0, 0, 1, 2, 5, 5]
```

## 在LLM场景中的应用

### 梯度裁剪

梯度裁剪是防止梯度爆炸的重要技术：

```python
def gradient_clipping(gradients, max_norm=1.0):
    """梯度裁剪

    将梯度的 L2 范数裁剪到 max_norm
    """
    grad_norm = np.linalg.norm(gradients)

    if grad_norm > max_norm:
        # 裁剪：保持方向不变，但将范数限制在 max_norm
        gradients = gradients * (max_norm / grad_norm)
        print(f"梯度已裁剪: {grad_norm:.4f} -> {max_norm}")

    return gradients

# 测试
gradients = np.random.randn(1000, 768).astype(np.float32) * 5  # 模拟大梯度
print(f"裁剪前梯度范数: {np.linalg.norm(gradients):.4f}")
clipped_gradients = gradient_clipping(gradients.copy())
print(f"裁剪后梯度范数: {np.linalg.norm(clipped_gradients):.4f}")
```

### logits 温度缩放

在语言模型生成时，温度参数用于控制输出的随机性：

```python
def apply_temperature(logits, temperature=1.0):
    """应用温度缩放到 logits

    温度 > 1：更随机（分布更平坦）
    温度 < 1：更确定（分布更尖锐）
    温度 = 1：不缩放
    """
    if temperature == 0:
        # 贪婪策略：选择最大 logit
        return np.where(logits == logits.max(), 1.0, 0.0)

    # 缩放 logits
    scaled_logits = logits / temperature

    # clip 防止极端值
    scaled_logits = np.clip(scaled_logits, -100, 100)

    return scaled_logits

# 测试
logits = np.array([2.0, 1.0, 0.5, 0.1])
print(f"原始 logits: {logits}")
print(f"温度=0.5 (更确定): {apply_temperature(logits, 0.5)}")
print(f"温度=1.0 (不变): {apply_temperature(logits, 1.0)}")
print(f"温度=2.0 (更随机): {apply_temperature(logits, 2.0)}")
```

### 概率裁剪

在计算交叉熵时，为了避免 log(0) 的问题，经常需要裁剪概率：

```python
def clipped_cross_entropy(logits, labels, clip_prob=1e-7):
    """带概率裁剪的交叉熵

    将概率裁剪到 [clip_prob, 1 - clip_prob] 范围内
    """
    # Softmax
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()

    # 裁剪概率
    probs = np.clip(probs, clip_prob, 1 - clip_prob)

    # 交叉熵
    log_probs = np.log(probs)
    loss = -np.sum(labels * log_probs)

    return loss

# 测试
logits = np.array([2.0, 1.0, 0.1])
labels = np.array([1, 0, 0])

loss = clipped_cross_entropy(logits, labels)
print(f"交叉熵损失: {loss:.4f}")
```

### 模型量化

量化是将 float32 转换为低精度（如 int8）的过程：

```python
def quantize_to_int8(weights, scale_factor=None):
    """将权重量化到 int8

    Args:
        weights: 原始浮点权重
        scale_factor: 缩放因子，如果为 None 则自动计算
    """
    if scale_factor is None:
        # 计算缩放因子，使得最大值映射到 127
        scale_factor = weights.max() / 127.0

    # 缩放
    scaled = weights / scale_factor

    # 取整（四舍五入）
    quantized = np.round(scaled)

    # 裁剪到 [-128, 127]
    quantized = np.clip(quantized, -128, 127).astype(np.int8)

    return quantized, scale_factor

# 测试
weights = np.random.randn(768, 3072).astype(np.float32) * 0.02
quantized, scale = quantize_to_int8(weights)
print(f"原始权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
print(f"量化后范围: [{quantized.min()}, {quantized.max()}]")
print(f"缩放因子: {scale:.6f}")
```

## 常见误区与注意事项

### 误区一：混淆 trunc 和 floor

```python
x = np.array([-1.5, 1.5])

# trunc 只截断小数部分
print(f"trunc: {np.trunc(x)}")  # [-1., 1.]

# floor 向下取整
print(f"floor: {np.floor(x)}")  # [-2., 1.]
```

对于负数 -1.5，trunc(-1.5) = -1（更接近零），floor(-1.5) = -2（更小的整数）。

### 误区二：银行家舍入的意外结果

```python
# NumPy 使用银行家舍入
x = np.array([0.5, 1.5, 2.5, 3.5])
print(f"round: {np.round(x)}")  # [0., 2., 2., 4.]
# 注意：1.5 -> 2 (偶数), 2.5 -> 2 (偶数)
```

如果需要传统的四舍五入，需要自定义函数：

```python
def traditional_round(x):
    """传统四舍五入"""
    return np.floor(x + 0.5)

print(f"传统四舍五入: {traditional_round(x)}")  # [1., 2., 3., 4.]
```

### 误区三：clip 的原地操作

```python
arr = np.array([1, 2, 3, 4, 5])

# np.clip 默认返回新数组
clipped = np.clip(arr, 2, 4)
print(f"原始数组: {arr}")  # 不变

# 如果使用 out 参数，会原地修改
arr2 = np.array([1, 2, 3, 4, 5])
np.clip(arr2, 2, 4, out=arr2)
print(f"原地裁剪后: {arr2}")  # 被修改
```

## 底层原理

这些取整函数的底层实现各有不同：

- `floor`、`ceil`、`trunc` 通常直接调用底层 C 库的对应函数
- `round` 使用银行家舍入，需要额外的逻辑处理 .5 的情况
- `clip` 的实现相对简单，就是比较和赋值

```python
# clip 的等价实现
def clip_equivalent(x, min_val, max_val):
    """clip 的等价实现"""
    result = np.where(x < min_val, min_val, x)
    result = np.where(result > max_val, max_val, result)
    return result

x = np.array([-3, -1, 0, 1, 2, 5, 10])
print(f"等价实现: {clip_equivalent(x, 0, 5)}")
print(f"np.clip: {np.clip(x, 0, 5)}")
```

## 小结

取整和截断函数在数值计算中非常重要。`floor`、`ceil`、`trunc` 用于取整，`round` 用于四舍五入（银行家舍入），`clip` 用于限制值域。在 LLM 场景中，这些函数用于梯度裁剪、概率裁剪、温度缩放和模型量化等。

面试时需要能够解释不同取整函数的区别，理解 clip 的用法，以及注意银行家舍入的特殊行为。
