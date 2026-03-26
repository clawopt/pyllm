# 随机数组

随机数组是深度学习中最重要也最常用的数组类型之一。想象一下，当你初始化一个神经网络的权重时，你需要随机生成的数值；当你在训练时应用 Dropout 策略时，需要随机生成掩码；当你进行数据增强时，需要随机变换数据。NumPy 提供了丰富的随机数组生成函数，这些函数都集中在 `np.random` 模块中。理解这些随机数组的生成方法，以及如何控制随机性，对于深度学习的实现至关重要。随机数的质量直接影响模型初始化的效果和实验的可复现性。

## np.random.rand：均匀分布随机数组

`np.random.rand` 生成的是 [0, 1) 区间上的均匀分布随机数：

```python
import numpy as np

# 生成指定形状的随机数组
arr1 = np.random.rand(5)
print(f"一维均匀分布: {arr1}")

arr2 = np.random.rand(3, 4)
print(f"二维均匀分布:\n{arr2}")

# 指定数据类型
arr3 = np.random.rand(2, 3).astype(np.float32)
print(f"float32 均匀分布: {arr3.dtype}")
```

均匀分布意味着每个值在 [0, 1) 区间内被选中的概率相等。这种分布在初始化、随机掩码生成等场景中很常用。

## np.random.randn：标准正态分布随机数组

`np.random.randn` 生成标准正态分布（均值0，标准差1）的随机数：

```python
# 生成标准正态分布
arr1 = np.random.randn(5)
print(f"一维标准正态分布: {arr1}")

arr2 = np.random.randn(3, 4)
print(f"二维标准正态分布:\n{arr2}")
```

标准正态分布的随机数在深度学习中非常重要，因为很多权重初始化方法都假设初始权重服从正态分布。

## np.random.randint：整数随机数组

`np.random.randint` 生成指定范围内的随机整数：

```python
# 生成 [low, high) 区间的随机整数
arr1 = np.random.randint(0, 10, size=5)
print(f"[0, 10) 随机整数: {arr1}")

arr2 = np.random.randint(0, 10, size=(3, 4))
print(f"形状 (3, 4) 的随机整数:\n{arr2}")

# 包含 high 的情况
arr3 = np.random.randint(0, 10, size=5)  # 默认 high 不包含
print(f"[0, 10) 随机整数: {arr3}")
```

注意 `np.random.randint` 的 `high` 参数是不包含的，即生成 [low, high) 区间的整数。

## 旧版 API vs 新版 API

NumPy 提供了两套随机数 API：旧版 API 使用 `np.random.seed()` 设置全局种子，新版 API 使用 `np.random.default_rng()` 创建独立的随机数生成器：

```python
# 旧版 API
np.random.seed(42)
arr_old = np.random.rand(5)
print(f"旧版 API: {arr_old}")

# 新版 API（推荐）
rng = np.random.default_rng(42)
arr_new = rng.random(5)
print(f"新版 API: {arr_new}")
```

新版 API 的优势是每个 Generator 实例都有独立的随机状态，不会互相影响。面试时可能会问两者的区别，推荐使用新版 API。

## 设置种子与可复现性

在科学研究和深度学习训练中，确保随机结果可复现是非常重要的：

```python
# 使用旧版 API 设置种子
np.random.seed(42)
arr1 = np.random.randn(3, 4)

np.random.seed(42)
arr2 = np.random.randn(3, 4)
print(f"相同种子，结果相同: {np.array_equal(arr1, arr2)}")

# 使用新版 API 设置种子
rng1 = np.random.default_rng(42)
arr3 = rng1.random((3, 4))

rng2 = np.random.default_rng(42)
arr4 = rng2.random((3, 4))
print(f"相同种子，结果相同: {np.array_equal(arr3, arr4)}")
```

种子设置后，相同的随机数生成操作会产生相同的结果。这对于调试和复现实验结果至关重要。

## 在深度学习中的应用

### 权重初始化

神经网络的权重初始化对训练稳定性有重要影响。不同的初始化方法使用不同分布的随机数：

```python
def xavier_uniform(fan_in, fan_out, rng=None):
    """Xavier 均匀初始化"""
    if rng is None:
        rng = np.random.default_rng()
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_out, fan_in))

def xavier_normal(fan_in, fan_out, rng=None):
    """Xavier 正态初始化"""
    if rng is None:
        rng = np.random.default_rng()
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, size=(fan_out, fan_in))

def he_normal(fan_in, fan_out, rng=None):
    """He 正态初始化（适合 ReLU）"""
    if rng is None:
        rng = np.random.default_rng()
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, size=(fan_out, fan_in))

# 初始化一个 Transformer 层的权重
fan_in = 768
fan_out = 3072

rng = np.random.default_rng(42)
W = he_normal(fan_in, fan_out, rng)
print(f"He初始化权重形状: {W.shape}")
print(f"权重均值: {W.mean():.6f}")
print(f"权重标准差: {W.std():.6f}")
```

不同的初始化方法适用于不同的激活函数。Xavier 初始化适用于 Sigmoid 和 Tanh，He 初始化适用于 ReLU。

### Dropout 掩码生成

Dropout 是防止过拟合的重要技术，需要生成随机掩码：

```python
def dropout_mask(shape, rate, rng=None):
    """生成 Dropout 掩码
    
    返回一个布尔数组，True 表示保留，False 表示丢弃
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.random(shape) > rate

batch_size = 32
seq_len = 512
hidden_dim = 768
dropout_rate = 0.1

rng = np.random.default_rng(42)
mask = dropout_mask((batch_size, seq_len, hidden_dim), dropout_rate, rng)
print(f"Dropout 掩码形状: {mask.shape}")
print(f"保留比例: {mask.mean():.4f}")  # 应接近 1 - dropout_rate
```

Dropout 掩码决定了哪些神经元被"丢弃"。训练时丢弃一部分神经元可以让模型不过度依赖任何一个神经元，从而学到更鲁棒的特征。

### 数据增强

在数据增强中，随机变换可以增加训练数据的多样性：

```python
def random_crop(images, crop_size, rng=None):
    """随机裁剪图像
    
    images: (batch, height, width, channels)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    batch_size, height, width, channels = images.shape
    crop_h, crop_w = crop_size
    
    # 随机选择裁剪的起始位置
    start_h = rng.integers(0, height - crop_h + 1, size=batch_size)
    start_w = rng.integers(0, width - crop_w + 1, size=batch_size)
    
    # 执行裁剪
    cropped = np.zeros((batch_size, crop_h, crop_w, channels))
    for i in range(batch_size):
        cropped[i] = images[i, start_h[i]:start_h[i]+crop_h, 
                               start_w[i]:start_w[i]+crop_w]
    
    return cropped

# 模拟图像数据
batch_size = 4
images = np.random.rand(batch_size, 224, 224, 3)
rng = np.random.default_rng(42)
cropped = random_crop(images, (112, 112), rng)
print(f"裁剪后形状: {cropped.shape}")
```

## 其他随机分布

NumPy 还支持多种随机分布：

```python
rng = np.random.default_rng(42)

# 均匀分布
uniform = rng.uniform(0, 1, size=10)  # [low, high)
print(f"均匀分布: {uniform}")

# 正态分布
normal = rng.normal(0, 1, size=10)  # (mean, std)
print(f"正态分布: {normal}")

# 指数分布
exponential = rng.exponential(1.0, size=10)
print(f"指数分布: {exponential}")

# Beta 分布
beta = rng.beta(0.5, 0.5, size=10)  # (alpha, beta)
print(f"Beta分布: {beta}")

# 二项分布（用于 Dropout）
binomial = rng.binomial(1, 0.5, size=10)  # (n, p)
print(f"二项分布: {binomial}")
```

## 常见误区与注意事项

### 误区一：全局种子影响后续所有随机操作

```python
# 设置种子
np.random.seed(42)
a = np.random.rand(5)

# 中间插入了其他随机操作
np.random.rand(3)  # 这会改变全局状态！

# 重新设置种子
np.random.seed(42)
b = np.random.rand(5)

print(f"a: {a}")
print(f"b: {b}")
print(f"相同吗: {np.array_equal(a, b)}")  # False!
```

解决方案是使用独立的 Generator 实例。

### 误区二：忘记 dtype 导致精度不匹配

```python
# 默认是 float64
arr_64 = np.random.randn(1000, 1000)
print(f"默认 dtype: {arr_64.dtype}")

# 深度学习中通常应该用 float32
arr_32 = np.random.randn(1000, 1000).astype(np.float32)
print(f"float32 dtype: {arr_32.dtype}")
```

float32 占用一半内存，在 GPU 上计算也更快。

### 误区三：randint 的参数顺序

```python
# 正确：low, high, size
correct = np.random.randint(0, 10, size=(3, 4))

# 容易混淆：high 不包含
# np.random.randint(10) 只会生成 0
single = np.random.randint(10)  # 生成 [0, 10) 的单个整数
print(f"单个随机整数: {single}")
```

## 底层原理

NumPy 的随机数生成器使用的是伪随机数算法。默认情况下，新版 API 使用 PCG64（Permuted Congruential Generator）算法，它基于线性同余发生器（LCG）但增加了一步排列来提高随机性质量。

PCG64 的状态空间是 64 位，周期长达 2^124，对于任何实际应用来说都已经足够了。伪随机数是用确定性算法生成的，但好的算法可以让生成的序列通过统计检验，看起来像真正的随机数。

## 多维随机数组的形状

```python
rng = np.random.default_rng(42)

# 一维
arr1d = rng.random(5)
print(f"1D: {arr1d.shape}")

# 二维
arr2d = rng.random((3, 4))
print(f"2D: {arr2d.shape}")

# 三维（常见于图像或序列数据）
arr3d = rng.random((batch_size, seq_len, hidden_dim))
print(f"3D: {arr3d.shape}")
```

注意 `np.random.rand` 和 `rng.random()` 接受参数的顺序不同：`np.random.rand(3, 4)` 是单独的参数，而 `rng.random((3, 4))` 是一个元组参数。

## 小结

随机数组是深度学习中不可或缺的工具。`np.random.rand` 生成均匀分布，`np.random.randn` 生成标准正态分布，`np.random.randint` 生成随机整数。新版 API 使用 `np.random.default_rng()` 创建独立的随机数生成器，比旧版 API 更安全。

在深度学习中，随机数组的应用无处不在：权重初始化（Xavier、He 等方法）、Dropout 掩码生成、数据增强的随机变换、学习率调度的随机搜索等。正确使用随机数组和设置种子，是确保实验可复现性的基础。

面试时需要能够解释 Xavier 和 He 初始化的原理和适用场景，理解为什么 Dropout 需要随机掩码，以及如何正确设置种子确保可复现性。
