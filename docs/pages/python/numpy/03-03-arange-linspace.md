# 等间隔数组

在NumPy中，当你需要创建一个有规律的数字序列时，`np.arange` 和 `np.linspace` 是两个最常用的函数。想象一下，你要生成一系列的学习率来测试模型性能，或者要创建位置编码中的角度值，这些时候你就需要等间隔数组。顾名思义，等间隔数组就是数组中相邻元素的差值相等的数组。这两个函数看起来相似，但用法和适用场景有显著区别： `np.arange` 类似于Python内置的 `range`，通过指定起点、终点和步长来创建数组；`np.linspace` 则是指定起点、终点和希望生成的元素个数。理解这两个函数的区别对于正确生成数值序列至关重要。

## np.arange：类似range的数组创建

`np.arange` 的用法和Python的 `range` 类似，但返回的是NumPy数组：

```python
import numpy as np

# 从0到5（不包含5），步长为1
arr1 = np.arange(5)
print(f"arange(5): {arr1}")

# 从0到10，步长为2
arr2 = np.arange(0, 10, 2)
print(f"arange(0, 10, 2): {arr2}")

# 支持浮点数步长
arr3 = np.arange(0, 1, 0.2)
print(f"arange(0, 1, 0.2): {arr3}")

# 负数步长
arr4 = np.arange(10, 0, -2)
print(f"arange(10, 0, -2): {arr4}")
```

`np.arange` 的签名是 `np.arange(start, stop, step)`，和Python的 `range` 一样，包含start但不包含stop。这点很重要，很多bug都来源于对这个行为的误解。

## np.linspace：指定元素数量的数组创建

当你需要确切数量的点时，`np.linspace` 是更好的选择：

```python
# 在0到1之间生成5个等间隔的点
arr1 = np.linspace(0, 1, 5)
print(f"linspace(0, 1, 5): {arr1}")

# 生成用于三角函数的角度值
angles = np.linspace(0, 2 * np.pi, 100)
print(f"角度数组长度: {len(angles)}")
print(f"最后一个角度: {angles[-1]:.4f}")  # 应接近 2π
```

`np.linspace` 的签名是 `np.linspace(start, stop, num)`，它会在start和stop之间生成num个等间隔的点，包括端点。这与 `np.arange` 不同，`np.arange` 的端点通常不在数组中，而 `np.linspace` 的端点一定在数组中。

## np.arange vs np.linspace 的区别

面试中经常被问到："np.arange 和 np.linspace 的区别是什么？"这里有几个关键区别：

```python
# np.arange 根据步长计算点数，可能不是整数
arr_arange = np.arange(0, 0.5, 0.1)
print(f"arange(0, 0.5, 0.1): {arr_arange}")  # [0.  0.1 0.2 0.3 0.4]

# np.linspace 保证精确的点数
arr_linspace = np.linspace(0, 0.5, 5)
print(f"linspace(0, 0.5, 5): {arr_linspace}")  # [0.   0.125 0.25  0.375 0.5  ]
```

当处理浮点数时，`np.arange` 可能会因为浮点数精度问题产生意外的结果，而 `np.linspace` 总是给出确定性的结果：

```python
# np.arange 的浮点数精度问题
arr = np.arange(0, 1, 0.1)
print(f"arange 长度: {len(arr)}")  # 可能是10，也可能是11

# np.linspace 不会有这个问题
arr2 = np.linspace(0, 1, 10)
print(f"linspace 长度: {len(arr2)}")  # 精确等于10
```

## 在深度学习中的应用

### 学习率调度

学习率调度是深度学习训练中的重要技术，需要生成一系列学习率值：

```python
import numpy as np

# 线性学习率预热
def linear_warmup(warmup_steps, base_lr):
    """线性预热"""
    return np.linspace(0, base_lr, warmup_steps)

warmup_steps = 1000
base_lr = 1e-3
lr_schedule = linear_warmup(warmup_steps, base_lr)
print(f"预热学习率（前5步）: {lr_schedule[:5]}")
print(f"预热学习率（后5步）: {lr_schedule[-5:]}")
```

预热策略在训练初期逐渐增加学习率，可以让模型在开始时有一个稳定的过渡，避免一开始的大梯度破坏已经学到的特征。

### 余弦学习率衰减

现代Transformer训练中，余弦衰减是常用的学习率调度策略：

```python
def cosine_decay(lr_max, lr_min, total_steps, warmup_steps):
    """带预热的余弦衰减"""
    # 预热阶段
    warmup = np.linspace(0, lr_max, warmup_steps)
    
    # 余弦衰减阶段
    cosine_steps = total_steps - warmup_steps
    cosine_lr = np.linspace(0, np.pi, cosine_steps)
    cosine_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(cosine_lr / np.pi))
    
    return np.concatenate([warmup, cosine_lr])

total_steps = 10000
warmup_steps = 500
lr_max = 1e-3
lr_min = 1e-5

lr_schedule = cosine_decay(lr_max, lr_min, total_steps, warmup_steps)
print(f"总步数: {len(lr_schedule)}")
print(f"第0步学习率: {lr_schedule[0]:.6f}")
print(f"第500步学习率: {lr_schedule[500]:.6f}")
print(f"第5000步学习率: {lr_schedule[5000]:.6f}")
```

### 三角函数位置编码

Transformer的原版位置编码使用正弦和余弦函数来编码位置信息：

```python
def positional_encoding(seq_len, d_model):
    """生成 sin/cos 位置编码
    
    论文 "Attention Is All You Need" 中的公式
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    div_term = np.exp(div_term)
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

seq_len = 50
d_model = 128
pe = positional_encoding(seq_len, d_model)
print(f"位置编码形状: {pe.shape}")
print(f"位置0的前10维: {pe[0, :10]}")
```

这里 `np.arange` 用于生成位置索引和频率div_term。注意 `div_term` 用到了 `np.arange(0, d_model, 2)`，这会生成 [0, 2, 4, ..., d_model-2]，用于选择偶数维度的位置。

### 生成网格坐标

在某些计算机视觉任务或注意力机制变体中，需要生成2D网格坐标：

```python
# 生成2D网格坐标
height, width = 7, 7
y_coords = np.arange(height)
x_coords = np.arange(width)

# 生成坐标对
yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
print(f"Y坐标形状: {yy.shape}")
print(f"X坐标形状: {xx.shape}")
print(f"网格中心点坐标: ({yy[height//2, width//2]}, {xx[height//2, width//2]})")
```

关于 `np.meshgrid` 的详细用法，会在下一节"重复与网格"中深入讲解。

## endpoint 参数

`np.linspace` 有一个 `endpoint` 参数，控制是否包含终止值：

```python
# 默认 include endpoint
arr1 = np.linspace(0, 10, 5)
print(f"endpoint=True: {arr1}")  # [0, 2.5, 5, 7.5, 10]

# 排除 endpoint
arr2 = np.linspace(0, 10, 5, endpoint=False)
print(f"endpoint=False: {arr2}")  # [0, 2, 4, 6, 8]
```

当你需要生成不包含终止值的序列时，这个参数很有用。比如生成时间序列时，排除终止值可以避免重复。

## 常见误区与注意事项

### 误区一：混淆 np.arange 的参数顺序

```python
# 正确顺序：start, stop, step
correct = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# 错误写法（容易搞混）
try:
    wrong = np.arange(0, 10, 2)  # 实际上这居然是正确的
except Exception as e:
    print(f"错误: {e}")

# 但如果写成这样就错了
wrong2 = np.arange(10, 0, 2)  # 空数组！因为10>0但步长为正
print(f"错误示例结果: {wrong2}")  # []
```

### 误区二：浮点数步长导致的元素个数不确定

```python
# 问题：浮点数步长可能导致意外的数组长度
arr = np.arange(0, 1, 0.1)
print(f"arange(0, 1, 0.1) 长度: {len(arr)}")  # 可能是10或11

# 解决方案：使用 np.linspace
arr2 = np.linspace(0, 1, 10)
print(f"linspace(0, 1, 10) 长度: {len(arr2)}")  # 精确等于10
```

浮点数运算中，0.1 不是一个精确的二进制小数，所以累加时可能会产生微小的误差，导致元素个数不确定。

### 误区三：忘记 np.arange 不包含终止值

```python
# 如果你想要 0, 1, 2, 3, 4
correct = np.arange(5)  # [0, 1, 2, 3, 4]

# 错误地写成
wrong = np.arange(1, 5)  # [1, 2, 3, 4]，少了0
print(f"缺少0的例子: {wrong}")
```

## 底层原理

`np.arange` 的底层实现实际上是利用了ufunc的Accumulate功能，通过不断累加步长来生成数组。这种实现方式在处理整数数组时效率很高，但对于浮点数数组，由于浮点数精度问题，累加多次后可能会累积误差。

`np.linspace` 则使用了不同的策略：它先计算步长 `step = (stop - start) / (num - 1)`（当 endpoint=True 时），然后通过 `start + i * step` 来计算每个点。这样可以确保生成的点精确地等间隔。

## retstep 参数

`np.linspace` 和 `np.arange` 都支持 `retstep` 参数，返回值会包含步长：

```python
# np.linspace 的 retstep
arr, step = np.linspace(0, 1, 5, retstep=True)
print(f"数组: {arr}, 步长: {step}")  # [0, 0.25, 0.5, 0.75, 1], 步长: 0.25

# np.arange 的 retstep
arr2, step2 = np.arange(0, 1, 0.2, retstep=True)
print(f"数组: {arr2}, 步长: {step2}")  # [0, 0.2, 0.4, 0.6, 0.8], 步长: 0.2
```

这个返回值在调试时很有用，可以验证生成的数组是否符合预期。

## 小结

`np.arange` 和 `np.linspace` 都是创建等间隔数组的函数，但适用场景不同。`np.arange` 适合已知起点、终点和步长的情况，特别是在处理整数序列时；`np.linspace` 适合已知起点、终点和希望生成的点数的情况，特别是在需要精确控制数组长度或处理浮点数序列时。

在深度学习中，这两个函数有着广泛应用：学习率调度（特别是余弦衰减）用 `np.linspace` 生成精确的衰减曲线，三角函数位置编码用 `np.arange` 生成位置索引和频率，网格坐标生成用 `np.meshgrid` 结合 `np.arange` 等等。

面试时需要能够解释两个函数的区别、各自的适用场景，以及浮点数精度问题带来的潜在bug。
