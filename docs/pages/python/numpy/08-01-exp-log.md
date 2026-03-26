# 指数与对数函数

指数函数和对数函数是数学中最重要的函数之一，在科学计算和深度学习中有着广泛的应用。想象一下，当你在计算 softmax 归一化时，需要对每个元素取指数；当你在计算交叉熵损失时，需要对概率取对数。这些操作都离不开 `np.exp` 和 `np.log` 函数。指数函数 exp(x) 的特点是其导数等于自身，这使得它在梯度计算中有一些很好的性质；对数函数 log(x) 则是 exp(x) 的反函数，可以将乘法转换为加法。理解这两个函数的特性和注意事项，对于实现各种深度学习算法至关重要。

## np.exp：指数函数

`np.exp` 计算 e 的 x 次幂：

```python
import numpy as np

x = np.array([0, 1, 2, 3, 4])

print(f"e^x: {np.exp(x)}")
print(f"e^0 = {np.exp(0)}")  # 1
print(f"e^1 = {np.exp(1):.6f}")  # 2.718282
```

## np.log：对数函数

`np.log` 计算自然对数（底数为 e）：

```python
x = np.array([1, 2, 3, 4, 5])

print(f"ln(x): {np.log(x)}")
print(f"ln(1) = {np.log(1)}")  # 0
print(f"ln(e) = {np.log(np.e)}")  # 1
```

NumPy 还提供了其他底数的对数函数：
- `np.log2(x)`：底数为 2 的对数
- `np.log10(x)`：底数为 10 的对数

```python
x = np.array([1, 2, 4, 8, 16])

print(f"log2(x): {np.log2(x)}")   # [0, 1, 2, 3, 4]
print(f"log10(x): {np.log10(x)}")  # [0, 0.301, 0.602, 0.903, 1.204]
```

## expm1 和 log1p：精确计算小值

对于非常小的 x 值，直接计算 exp(x) - 1 或 log(1 + x) 会有精度问题：

```python
x_small = 1e-10

# 直接计算会有精度损失
direct = np.exp(x_small) - 1

# 使用 expm1 更精确
expm1_result = np.expm1(x_small)

print(f"直接计算: {direct}")
print(f"expm1: {expm1_result}")
print(f"真实值: {x_small}")  # 约为 1e-10
```

同样的道理适用于 log1p：

```python
x_small = 1e-10

direct = np.log(1 + x_small)
log1p_result = np.log1p(x_small)

print(f"直接计算: {direct}")
print(f"log1p: {log1p_result}")
print(f"真实值: {x_small}")  # 约为 1e-10
```

## 在LLM场景中的应用

### Softmax 函数

Softmax 是分类任务中常用的激活函数，将 logits 转换为概率分布：

```python
def softmax(logits):
    """Softmax 函数

    softmax(x_i) = exp(x_i) / Σexp(x_j)
    """
    exp_logits = np.exp(logits - logits.max())  # 减去最大值保证数值稳定
    return exp_logits / exp_logits.sum()

# 测试
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Probabilities: {probs}")
print(f"概率之和: {probs.sum():.6f}")  # 应为 1
```

注意这里减去最大值来保证数值稳定，这是实现 softmax 时的常见技巧。

### 交叉熵损失

交叉熵损失是分类任务中最常用的损失函数之一：

```python
def cross_entropy_loss(logits, labels):
    """交叉熵损失

    Loss = -Σ y_true * log(y_pred)
    """
    probs = softmax(logits)
    # 为了数值稳定，使用 log_softmax
    log_probs = np.log(probs + 1e-10)  # 加一个小值防止 log(0)
    return -np.sum(labels * log_probs)

# 测试
logits = np.array([2.0, 1.0, 0.1])
labels = np.array([1, 0, 0])  # one-hot 编码

loss = cross_entropy_loss(logits, labels)
print(f"交叉熵损失: {loss:.4f}")
```

### 更稳定的 log_softmax 实现

在实际应用中，更常用的是 log_softmax，它可以避免先计算 softmax 再取 log 的精度损失：

```python
def log_softmax(logits):
    """Log-Softmax 函数

    log_softmax(x_i) = x_i - log(Σexp(x_j))
                     = x_i - x_max - log(Σexp(x_j - x_max))
    """
    x_max = logits.max(axis=-1, keepdims=True)
    return logits - x_max - np.log(np.exp(logits - x_max).sum(axis=-1, keepdims=True) + 1e-10)

# 测试
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.0, 1.0]])
log_probs = log_softmax(logits)
print(f"Log-Softmax:\n{log_probs}")
```

## 数值稳定性问题

当 logits 的值很大时，exp 可能会溢出：

```python
large_logits = np.array([1000, 1001, 1002])

# 直接 exp 会溢出
try:
    exp_result = np.exp(large_logits - large_logits.max())
    print(f"exp 结果: {exp_result}")
except RuntimeWarning as e:
    print(f"溢出: {e}")

# 正确做法：先减去最大值
stable_result = np.exp(large_logits - large_logits.max())
print(f"稳定计算: {stable_result}")
```

另一个常见问题是 log(0) 会变成负无穷：

```python
small_probs = np.array([1e-10, 1e-5, 0.5, 0.9])

# 直接 log(0) 会出问题
print(f"log(0): {np.log(0)}")  # -inf

# 正确做法：加一个小值
safe_log = np.log(small_probs + 1e-10)
print(f"安全的 log: {safe_log}")
```

## 常见误区与注意事项

### 误区一：忘记 softmax 的数值稳定性

```python
# 错误的实现
def softmax_unstable(logits):
    exp_logits = np.exp(logits)  # 可能溢出
    return exp_logits / exp_logits.sum()

# 正确的实现
def softmax_stable(logits):
    exp_logits = np.exp(logits - logits.max())  # 减去最大值
    return exp_logits / exp_logits.sum()
```

### 误区二：log 和 log1p 混淆

```python
x = 1e-10

# log(1 + x) 和 log1p(x) 对于小 x 差别很大
print(f"log(1+x): {np.log(1 + x)}")   # 可能精度损失
print(f"log1p(x): {np.log1p(x)}")     # 更精确
```

## 小结

指数函数和对数函数是深度学习中最重要的数学工具。Softmax 和交叉熵损失都离不开这两个函数。实现这些函数时，数值稳定性是关键考虑因素：需要减去最大值来防止 exp 溢出，需要加小值来防止 log(0)。`expm1` 和 `log1p` 对于小值计算更加精确。

面试时需要能够手写 softmax 并解释为什么要减去最大值，理解 log_softmax 的优势，以及注意 exp 和 log 的数值稳定性问题。
