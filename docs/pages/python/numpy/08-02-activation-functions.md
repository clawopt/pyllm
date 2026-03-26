# 激活函数实现

激活函数是神经网络中引入非线性的关键组件。没有激活函数，无论神经网络有多少层，其表达能力都等价于一个线性变换。激活函数的选择对网络的训练稳定性和性能有着重要影响。在LLM时代，GELU取代ReLU成为了最流行的激活函数，而sigmoid则在各种门控机制（如LSTM、GRU）中发挥着重要作用。理解这些激活函数的数学原理和实现细节，对于深入理解神经网络的工作原理至关重要。

## Sigmoid 函数

Sigmoid 是最经典的激活函数之一，将任意实数映射到 (0, 1) 区间：

```python
import numpy as np

def sigmoid(x):
    """Sigmoid 函数

    σ(x) = 1 / (1 + e^(-x))

    特点：
    - 输出范围 (0, 1)，适合表示概率
    - 梯度在两端趋近于 0，容易产生梯度消失
    - 曾广泛使用，现在更多用于输出层和门控机制
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# 测试
x = np.array([-2, -1, 0, 1, 2])
print(f"Sigmoid: {sigmoid(x)}")
```

注意这里使用了 `np.clip` 来防止溢出。当 x 很大时，exp(-x) 会下溢为 0；当 x 很小时，exp(-x) 会溢出为 inf。

## Sigmoid 在门控机制中的应用

在 LSTM 和 GRU 等门控循环网络中，Sigmoid 用于决定有多少信息应该通过：

```python
def lstm_gate(input_data, hidden_state, weights, bias):
    """LSTM 门控单元

    LSTM 使用 sigmoid 来决定门的开闭程度
    """
    # 计算门控值（0 到 1 之间）
    gate_input = np.dot(weights, hidden_state) + input_data + bias
    gate_value = sigmoid(gate_input)
    return gate_value

# 模拟 LSTM 中的遗忘门
batch_size = 4
hidden_dim = 256

forget_gate = lstm_gate(
    input_data=np.random.randn(batch_size, hidden_dim).astype(np.float32),
    hidden_state=np.random.randn(batch_size, hidden_dim).astype(np.float32),
    weights=np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1,
    bias=np.zeros(hidden_dim).astype(np.float32)
)

print(f"遗忘门值范围: [{forget_gate.min():.4f}, {forget_gate.max():.4f}]")
print(f"遗忘门均值: {forget_gate.mean():.4f}")
```

遗忘门的值接近 1 时表示"保留旧信息"，接近 0 时表示"忘记旧信息"。

## ReLU 函数

ReLU（Rectified Linear Unit）是深度学习中最简单也最有效的激活函数：

```python
def relu(x):
    """ReLU 函数

    f(x) = max(0, x)

    特点：
    - 计算简单，梯度快速
    - 梯度在 x > 0 时恒为 1，缓解梯度消失
    - 但 x < 0 时梯度为 0，可能导致"死神经元"
    """
    return np.maximum(0, x)

x = np.array([-3, -2, -1, 0, 1, 2, 3])
print(f"ReLU: {relu(x)}")
```

ReLU 的一个问题是"死神经元"——如果一个神经元的输出总是负数，经过 ReLU 后会变成 0，并且由于梯度也是 0，这个神经元可能永远不会再被激活。

## Leaky ReLU

Leaky ReLU 是 ReLU 的改进版本，在负数区域有一个小的斜率：

```python
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU 函数

    f(x) = x if x > 0 else αx

    特点：
    - x < 0 时有一个小斜率，避免死神经元问题
    - α 通常取 0.01 或 0.02
    """
    return np.where(x > 0, x, alpha * x)

x = np.array([-3, -2, -1, 0, 1, 2, 3])
print(f"Leaky ReLU (α=0.01): {leaky_relu(x)}")
print(f"Leaky ReLU (α=0.1): {leaky_relu(x, alpha=0.1)}")
```

## GELU 函数

GELU（Gaussian Error Linear Unit）是现代 Transformer 模型中最常用的激活函数。GPT、BERT 等模型都使用 GELU：

```python
def gelu(x):
    """GELU 函数

    GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    特点：
    - Transformer 时代的主流激活函数
    - 比 ReLU 更平滑，可以产生更好的效果
    - 可以理解为一种软化的 ReLU
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

x = np.array([-3, -2, -1, 0, 1, 2, 3])
print(f"GELU: {gelu(x)}")
```

GELU 的数学推导比较复杂，但可以理解为：它根据输入的大小来决定应该让多少信号通过，类似于一种"软化"的 ReLU。

## Swish 函数

Swish 是 Google 提出的另一种激活函数：

```python
def swish(x, beta=1.0):
    """Swish 函数

    f(x) = x * sigmoid(βx)

    其中 β 是可学习的参数（或固定为 1）
    """
    return x * sigmoid(beta * x)

x = np.array([-3, -2, -1, 0, 1, 2, 3])
print(f"Swish (β=1): {swish(x)}")
print(f"Swish (β=2): {swish(x, beta=2)}")
```

## Tanh 函数

Tanh（双曲正切）是另一个常用的激活函数，将任意实数映射到 (-1, 1) 区间：

```python
x = np.array([-3, -2, -1, 0, 1, 2, 3])
print(f"Tanh: {np.tanh(x)}")
```

与 Sigmoid 不同，Tanh 的输出是以 0 为中心的，这在某些网络中可以帮助加快收敛。

## 在LLM场景中的应用

### FFN 中的 GELU

Transformer 的前馈网络（FFN）通常使用 GELU：

```python
def feed_forward_network(x, w1, b1, w2, b2):
    """Transformer FFN

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    """
    # 第一层线性变换
    hidden = x @ w1.T + b1

    # GELU 激活
    hidden = gelu(hidden)

    # 第二层线性变换
    output = hidden @ w2.T + b2

    return output

batch_size = 4
seq_len = 512
input_dim = 768
ffn_dim = 3072

# 初始化权重
np.random.seed(42)
W1 = np.random.randn(ffn_dim, input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
b1 = np.zeros(ffn_dim).astype(np.float32)
W2 = np.random.randn(input_dim, ffn_dim).astype(np.float32) * np.sqrt(2.0 / ffn_dim)
b2 = np.zeros(input_dim).astype(np.float32)

# 测试
x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
output = feed_forward_network(x, W1, b1, W2, b2)
print(f"FFN 输出形状: {output.shape}")
```

### 门控线性单元（GLU）

GLU 是一种使用门控机制的架构，在 LLaMA 等模型中广泛使用：

```python
def glu(x, weight, bias):
    """门控线性单元

    GLU(x) = sigmoid(x @ W + b) * (x @ W + b)

    实际上 GLU 使用两个不同的权重矩阵
    """
    return sigmoid(x) * x

# LLaMA 风格的 SwiGLU
def swiglu(x, w1, w2, b1, b2):
    """SwiGLU 激活

    SwiGLU(x) = Swish(x @ W1 + b1) * (x @ W2 + b2)
    """
    return swish(x @ w1.T + b1) * (x @ w2.T + b2)
```

## 激活函数的梯度

理解激活函数的梯度对于理解反向传播至关重要：

```python
def sigmoid_gradient(x):
    """Sigmoid 的梯度

    d/dx σ(x) = σ(x) * (1 - σ(x))
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu_gradient(x):
    """ReLU 的梯度

    d/dx max(0, x) = 1 if x > 0 else 0
    """
    return (x > 0).astype(x.dtype)

def gelu_gradient(x):
    """GELU 的梯度（近似）"""
    # 精确的梯度计算比较复杂，这里提供一个近似
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))) + \
           x * (1 - np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))**2) * \
           np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)
```

## 常见误区与注意事项

### 误区一：忘记对大值进行裁剪

```python
# 错误实现
def sigmoid_wrong(x):
    return 1 / (1 + np.exp(-x))

# 当 x 很大时会溢出
try:
    sigmoid_wrong(1000)
except FloatingPointError:
    print(f"溢出错误")

# 正确实现
def sigmoid_correct(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

print(f"正确实现: {sigmoid_correct(1000)}")
```

### 误区二：ReLU 的死神经元问题

```python
# 初始化不当可能导致大量死神经元
weights = np.random.randn(1000, 768).astype(np.float32) * 0.01
biases = np.zeros(1000).astype(np.float32)

# 前向传播
x = np.random.randn(32, 768).astype(np.float32)
z = x @ weights.T + biases
a = relu(z)

# 检查死神经元比例
dead_ratio = (a == 0).mean()
print(f"死神经元比例: {dead_ratio:.2%}")
```

### 误区三：混淆激活函数的使用场景

- Sigmoid：二分类输出层、门控机制
- Tanh：LSTM 的细胞状态更新、某些 RNN
- ReLU：CNN、简单的 MLP
- GELU：Transformer、LLM

## 小结

激活函数为神经网络引入非线性，是深度学习的基础组件。Sigmoid 将值映射到 (0, 1)，用于概率输出和门控；ReLU 简单高效，但可能导致死神经元；GELU 是现代 LLM 的主流选择，比 ReLU 更平滑。理解这些激活函数的特性和实现，对于深入理解神经网络至关重要。

面试时需要能够解释不同激活函数的特点和适用场景，理解 GELU 的数学公式，以及注意数值稳定性问题。
