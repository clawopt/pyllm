# 随机初始化权重

神经网络的权重初始化是训练成功与否的关键因素之一。正确的初始化可以加速收敛、避免梯度消失或爆炸；错误的初始化可能导致训练困难，甚至完全无法收敛。在深度学习发展的历程中，研究者们提出了多种初始化方法，从最初的简单随机初始化，到 Xavier/Glorot 初始化，再到 He 初始化，每种方法都基于对信号传播的深入理解。NumPy 作为深度学习框架的底层计算库，理解这些初始化方法的原理和实现，对于深入理解深度学习至关重要。

## 为什么初始化很重要

神经网络的前向传播可以看作是信息的逐层传递。如果权重初始化不当，信息在传播过程中可能会被放大或衰减：

**梯度消失问题**：当权重太小（接近零）时，梯度在反向传播过程中会指数级减小，导致浅层网络几乎学不到东西。

**梯度爆炸问题**：当权重太大时，梯度会指数级增大，导致训练不稳定。

**对称性破坏**：如果所有权重初始化为相同的值，同一层中的神经元会学习到相同的特征，浪费了网络的容量。

```python
import numpy as np

# 可视化不同初始化对方差的影响
def simulate_signal_propagation(input_var, weights, n_layers=10):
    """模拟信号在多层网络中的传播

    观察每层输出的方差变化
    """
    x = input_var * np.ones(n_layers)
    for i in range(n_layers):
        x[i] = x[i-1] * weights[i] if i > 0 else x[i]
        # 每层计算方差（简化模拟）
        x[i] = np.var(x[i]) if i > 0 else x[i]
    return x

# 问题：如果权重 < 1，方差会逐层衰减
# 问题：如果权重 > 1，方差会逐层爆炸
```

## Xavier/Glorot 初始化

Xavier 初始化由 Xavier Glorot 和 Yoshua Bengio 在 2010 年提出，是目前最广泛使用的初始化方法之一。它的核心思想是：**让每层的输入和输出的方差保持一致**。

### 原理

对于一层网络 Y = WX + b，假设：
- X 是输入，有 n_in 个特征
- W 是权重，有 n_out 个输出
- 输入 X 的方差为 Var(X)

如果 W 的元素独立同分布，均值为 0，方差为 Var(W)，则输出的方差为：
```
Var(Y) = n_in * Var(W) * Var(X)
```

为了让 Var(Y) = Var(X)（输入输出方差一致），需要：
```
Var(W) = 1 / n_in
```

同时考虑前向传播和反向传播，Xavier 初始化使用：
```
Var(W) = 2 / (n_in + n_out)
```

### NumPy 实现 Xavier 初始化

```python
def xavier_initialization(fan_in, fan_out, shape=None, rng=None):
    """Xavier/Glorot 初始化

    从 N(0, sqrt(2/(fan_in + fan_out))) 生成权重

    参数:
        fan_in: 输入单元数
        fan_out: 输出单元数
        shape: 如果指定，直接使用形状；如果为 None，从 fan 计算
        rng: 随机数生成器
    返回:
        初始化后的权重数组
    """
    if rng is None:
        rng = np.random.default_rng()
    if shape is None:
        shape = (fan_out, fan_in)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, shape)

# 示例：为 Transformer 的注意力层初始化权重
np.random.seed(42)
rng = np.random.default_rng(42)

# 假设一个注意力层的权重形状
W_q = xavier_initialization(768, 768, rng=rng)  # Query 投影
W_k = xavier_initialization(768, 768, rng=rng)  # Key 投影
W_v = xavier_initialization(768, 768, rng=rng)  # Value 投影

print(f"Query 权重形状: {W_q.shape}")
print(f"权重均值: {W_q.mean():.6f}")
print(f"权重标准差: {W_q.std():.6f}")
print(f"权重范围: [{W_q.min():.4f}, {W_q.max():.4f}]")
```

### Xavier 初始化的两种形式

Xavier 初始化有两种常见形式：
1. **均匀形式**：从 U(-limit, limit) 中采样，其中 limit = sqrt(6/(fan_in+fan_out))
2. **正态形式**：从 N(0, sqrt(2/(fan_in+fan_out))) 中采样

```python
def xavier_uniform(fan_in, fan_out, shape=None, rng=None):
    """Xavier 均匀初始化"""
    if rng is None:
        rng = np.random.default_rng()
    if shape is None:
        shape = (fan_out, fan_in)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, shape)

def xavier_normal(fan_in, fan_out, shape=None, rng=None):
    """Xavier 正态初始化"""
    if rng is None:
        rng = np.random.default_rng()
    if shape is None:
        shape = (fan_out, fan_in)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, shape)
```

## He 初始化

He 初始化由 Kaiming He 等人在 2015 年提出，专门为 ReLU 激活函数设计。ReLU 会将一半的负值置零，这改变了信号的分布，导致需要更大的权重来维持方差。

### 原理

对于 ReLU 激活函数，假设输入 X 的一半是负值（被置零），输出 Y 的方差约为：
```
Var(Y) ≈ (1/2) * n_in * Var(W) * Var(X)
```

为了让 Var(Y) = Var(X)，需要：
```
Var(W) = 2 / n_in
```

因此，He 初始化的标准差是 sqrt(2/n_in)，比 Xavier 的 sqrt(2/(n_in+n_out)) 更大。

### NumPy 实现 He 初始化

```python
def he_initialization(fan_in, fan_out, shape=None, rng=None):
    """He 初始化（适用于 ReLU 激活函数）

    从 N(0, sqrt(2/fan_in)) 生成权重

    参数:
        fan_in: 输入单元数
        fan_out: 输出单元数
        shape: 如果指定，直接使用形状
        rng: 随机数生成器
    返回:
        初始化后的权重数组
    """
    if rng is None:
        rng = np.random.default_rng()
    if shape is None:
        shape = (fan_out, fan_in)
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, shape)

# 示例：为 ReLU 网络初始化
np.random.seed(42)
rng = np.random.default_rng(42)

# MLP 层：输入 512 -> 输出 256
W1 = he_initialization(512, 256, rng=rng)
W2 = he_initialization(256, 128, rng=rng)

print(f"层1权重形状: {W1.shape}")
print(f"层1权重标准差: {W1.std():.6f} (理论值: {np.sqrt(2/512):.6f})")
```

## Transformer 中的初始化策略

现代 Transformer 架构有几种常见的初始化模式：

### 残差连接的零初始化

在 Transformer 中，残差连接（skip connection）的输出会加到子层输出上。为了不改变信号在残差连接中的传播，**输入应该与输出保持相近的尺度**。一个常见做法是将残差分支的输出初始化为接近零的较小值。

```python
def residual_branch_initialization(fan_dim, rng=None):
    """残差分支初始化（输出接近零）

    用于 Projection 层和 FFN 后的 LayerNorm
    """
    if rng is None:
        rng = np.random.default_rng()
    # 初始化为较小的值
    return rng.normal(0, 0.02, (fan_dim, fan_dim))

def init_transformer_weights(fan_dim, rng=None):
    """Transformer 权重初始化

    参数:
        fan_dim: 隐藏层维度
        rng: 随机数生成器
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = {}

    # 注意力权重：使用 Xavier 初始化
    weights['W_q'] = xavier_initialization(fan_dim, fan_dim, rng=rng)
    weights['W_k'] = xavier_initialization(fan_dim, fan_dim, rng=rng)
    weights['W_v'] = xavier_initialization(fan_dim, fan_dim, rng=rng)
    weights['W_o'] = xavier_initialization(fan_dim, fan_dim, rng=rng)

    # FFN 权重：使用 Xavier 初始化
    # FFN 通常是 fan_dim -> ffn_dim -> fan_dim
    ffn_dim = fan_dim * 4
    weights['W1'] = xavier_initialization(fan_dim, ffn_dim, rng=rng)
    weights['W2'] = xavier_initialization(ffn_dim, fan_dim, rng=rng)

    # 偏置初始化为 0
    weights['q_bias'] = np.zeros(fan_dim)
    weights['k_bias'] = np.zeros(fan_dim)
    weights['v_bias'] = np.zeros(fan_dim)
    weights['o_bias'] = np.zeros(fan_dim)

    return weights

# 初始化示例
np.random.seed(42)
rng = np.random.default_rng(42)
transformer_weights = init_transformer_weights(768, rng)

print("Transformer 权重初始化示例:")
for name, weight in transformer_weights.items():
    if isinstance(weight, np.ndarray):
        print(f"  {name}: shape={weight.shape}, std={weight.std():.6f}")
    else:
        print(f"  {name}: {weight}")
```

### 嵌入层初始化

词汇嵌入通常初始化为较小的均匀分布或截断正态分布：

```python
def embedding_initialization(vocab_size, embedding_dim, rng=None):
    """词嵌入初始化

    通常使用均匀分布或截断正态分布
    """
    if rng is None:
        rng = np.random.default_rng()

    # 方法1：均匀分布
    # embed = rng.uniform(-0.1, 0.1, (vocab_size, embedding_dim))

    # 方法2：截断正态分布（更常用）
    embed = rng.normal(0, 0.02, (vocab_size, embedding_dim))
    embed = np.clip(embed, -0.1, 0.1)  # 截断

    return embed

# 示例：GPT-2 的嵌入层
np.random.seed(42)
rng = np.random.default_rng(42)
token_embeddings = embedding_initialization(50257, 768, rng)
position_embeddings = embedding_initialization(1024, 768, rng)  # 位置编码

print(f"Token 嵌入形状: {token_embeddings.shape}")
print(f"Token 嵌入范围: [{token_embeddings.min():.4f}, {token_embeddings.max():.4f}]")
```

## 偏置初始化

偏置（bias）通常初始化为零。但在某些特殊情况下：
- **ReLU 激活**：可以初始化为小的正数（如 0.01），让神经元一开始就是活跃的
- **LSTM/GRU 的遗忘门**：常初始化为 1，让初始记忆能力最强

```python
def bias_initialization(fan_out, zeros=True, positive_init=0.0):
    """偏置初始化

    参数:
        fan_out: 输出维度
        zeros: 是否初始化为 0
        positive_init: 如果不为 0，初始化为小正数
    """
    if zeros:
        return np.zeros(fan_out)
    else:
        return np.full(fan_out, positive_init)

# LSTM 遗忘门偏置初始化为 1
lstm_forget_bias = bias_initialization(512, zeros=False, positive_init=1.0)
print(f"LSTM 遗忘门偏置: {lstm_forget_bias[:5]}... (前5个)")
```

## 常见误区

**误区一：对所有层使用相同的初始化策略**

不同类型的层可能需要不同的初始化策略。例如，Xavier 适合 tanh/sigmoid，He 适合 ReLU。混用可能导致性能下降。

```python
# 错误：对 ReLU 网络使用 Xavier
# 正确：使用 He 初始化
relu_weights = he_initialization(512, 256)

# 错误：对 sigmoid 网络使用 He
# 正确：使用 Xavier
sigmoid_weights = xavier_initialization(512, 256)
```

**误区二：忽略输入维度和输出维度的差异**

对于输入维度很大但输出维度很小的层（如词汇嵌入后的第一层），初始化标准差的选择需要特别小心。

**误区三：偏置初始化为非零值**

除非有特殊原因（如 LSTM 遗忘门），偏置应该初始化为零。非零偏置会导致神经元一开始就有一个固定的激活偏移，可能影响学习。

**误区四：认为初始化只在训练开始时进行**

在迁移学习中，预训练模型的权重通常作为初始化。在微调时，有时会对部分层使用不同的初始化策略。

## 初始化方法对比

| 初始化方法 | 权重分布 | 适用激活函数 | 标准差公式 |
|-----------|---------|-------------|-----------|
| Xavier/Glorot | 均匀/正态 | tanh, sigmoid | sqrt(2/(n_in+n_out)) |
| He | 正态 | ReLU, Leaky ReLU | sqrt(2/n_in) |
| LeCun | 正态 | tanh, sigmoid | sqrt(1/n_in) |

## 验证初始化

在实践中，可以通过监控前向传播和反向传播中信号/梯度的方差来验证初始化是否正确：

```python
def check_initialization(shape, init_func, activation='relu'):
    """验证初始化后的方差传播

    参数:
        shape: (input_dim, output_dim)
        init_func: 初始化函数
        activation: 激活函数类型
    """
    W = init_func(shape[0], shape[1])

    # 模拟前向传播
    x = np.random.randn(1000, shape[0])
    y = x @ W.T

    if activation == 'relu':
        y = np.maximum(0, y)

    print(f"输入方差: {x.var():.6f}")
    print(f"输出方差: {y.var():.6f}")
    print(f"方差比 (output/input): {y.var()/x.var():.4f}")

    return y.var() / x.var()

# 验证 Xavier 和 He 初始化
print("Xavier 初始化（适用于 tanh）:")
check_initialization((512, 512), xavier_initialization, activation='tanh')

print("\nHe 初始化（适用于 ReLU）:")
check_initialization((512, 512), he_initialization, activation='relu')
```

理解权重初始化的原理对于调试神经网络和理解深度学习框架的行为非常重要。虽然现代框架已经自动处理了大多数初始化细节，但掌握这些原理能帮助你更好地设计模型架构。
