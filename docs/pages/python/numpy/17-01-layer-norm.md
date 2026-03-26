# LayerNorm实现

LayerNorm（层归一化）是 Transformer 架构中的关键组件之一，由 Geoffrey Hinton 等人在 2016 年提出。与 BatchNorm 不同，LayerNorm 在单个样本的内部进行归一化，不依赖 batch 维度。这使得它特别适合序列模型，因为在序列任务中不同样本的长度可能不同。本篇文章详细介绍 LayerNorm 的原理和纯 NumPy 实现。

## LayerNorm 的基本原理

LayerNorm 对每一层的输入进行归一化，使其均值为 0、方差为 1，然后通过可学习的参数 γ（scale）和 β（shift）进行线性变换：

```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```

其中：
- x 是输入
- μ 是均值：μ = (1/H) Σ x_i
- σ² 是方差：σ² = (1/H) Σ (x_i - μ)²
- γ 和 β 是可学习的参数
- ε 是防止除零的小常数（如 1e-6）

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-6):
    """Layer Normalization 实现

    参数:
        x: 输入张量 (..., hidden_size)
        gamma: 缩放参数 (hidden_size,)
        beta: 偏移参数 (hidden_size,)
        eps: 防止除零的小常数
    返回:
        输出张量，形状与输入相同
    """
    # 计算均值（沿最后一个维度）
    mean = np.mean(x, axis=-1, keepdims=True)

    # 计算方差
    var = np.var(x, axis=-1, keepdims=True)

    # 归一化
    x_norm = (x - mean) / np.sqrt(var + eps)

    # 线性变换
    output = gamma * x_norm + beta

    return output

# 示例
np.random.seed(42)
batch_size = 2
seq_len = 5
hidden_size = 8

x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
gamma = np.ones(hidden_size, dtype=np.float32)
beta = np.zeros(hidden_size, dtype=np.float32)

output = layer_norm(x, gamma, beta)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"输入均值范围: [{x.mean(axis=-1).min():.4f}, {x.mean(axis=-1).max():.4f}]")
print(f"输出均值范围: [{output.mean(axis=-1).min():.4f}, {output.mean(axis=-1).max():.4f}]")
```

## LayerNorm 类实现

```python
class LayerNorm:
    """Layer Normalization 层"""

    def __init__(self, hidden_size, eps=1e-6):
        self.hidden_size = hidden_size
        self.eps = eps

        # 可学习参数：γ (gain) 和 β (bias)
        self.gamma = np.ones(hidden_size, dtype=np.float32)
        self.beta = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, x):
        """前向传播

        参数:
            x: (batch_size, seq_len, hidden_size) 或 (batch_size, hidden_size)
        返回:
            output: 与输入形状相同
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        output = self.gamma * x_norm + self.beta

        return output

    def parameters(self):
        """返回模型参数"""
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }

# 示例
np.random.seed(42)
ln = LayerNorm(hidden_size=8)

# 测试不同形状输入
x1 = np.random.randn(2, 8).astype(np.float32)  # (batch, hidden)
x2 = np.random.randn(2, 5, 8).astype(np.float32)  # (batch, seq, hidden)

out1 = ln.forward(x1)
out2 = ln.forward(x2)

print(f"LayerNorm 测试:")
print(f"输入1 形状: {x1.shape} -> 输出形状: {out1.shape}")
print(f"输入2 形状: {x2.shape} -> 输出形状: {out2.shape}")
print(f"\n参数形状: gamma={ln.gamma.shape}, beta={ln.beta.shape}")
```

## 批量归一化与层归一化的对比

```python
def batch_norm_vs_layer_norm():
    """对比 BatchNorm 和 LayerNorm"""

    # 假设我们有 batch=2, seq=3, hidden=4 的输入
    x = np.random.randn(2, 3, 4).astype(np.float32)

    # BatchNorm：在 batch 维度上归一化
    # mean 和 var 跨 batch 计算，但沿 (seq, hidden) 计算
    batch_mean = np.mean(x, axis=(0, 1), keepdims=True)  # 整个 batch 的均值
    batch_var = np.var(x, axis=(0, 1), keepdims=True)
    bn_output = (x - batch_mean) / np.sqrt(batch_var + 1e-6)

    # LayerNorm：在每个样本内部归一化
    # mean 和 var 在 (hidden,) 维度计算
    ln_output = layer_norm(x, gamma=np.ones(4), beta=np.zeros(4))

    print("=== BatchNorm vs LayerNorm ===\n")
    print(f"输入形状: {x.shape}")
    print(f"BatchNorm 输出形状: {bn_output.shape}")
    print(f"LayerNorm 输出形状: {ln_output.shape}")

    # 验证 BatchNorm 的均值
    print(f"\nBatchNorm 验证:")
    print(f"  输出均值（沿 batch 维度）: {bn_output.mean(axis=(0, 1))}")

    print(f"\nLayerNorm 验证:")
    print(f"  每个样本的输出均值（应为 0）:")
    for i in range(2):
        for j in range(3):
            print(f"    样本[{i},{j}]均值: {ln_output[i, j].mean():.6f}")

batch_norm_vs_layer_norm()
```

## 带 optional bias 的 LayerNorm（Transformer 原始实现）

在原始 Transformer 论文中，LayerNorm 是在每个子层的输出上应用的：

```python
class LayerNormWithBias:
    """带 bias 的 LayerNorm（Transformer 原始实现）"""

    def __init__(self, hidden_size, eps=1e-6):
        self.hidden_size = hidden_size
        self.eps = eps

        # γ 和 β 初始化
        self.gamma = np.ones(hidden_size, dtype=np.float32)
        self.beta = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, x):
        """前向传播

        x: (batch, seq_len, hidden_size)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + self.eps)

        output = self.gamma * x_norm + self.beta

        return output

    def __call__(self, x):
        """支持函数调用语法"""
        return self.forward(x)
```

## 在 Transformer 中的应用

在 Transformer 中，LayerNorm 出现在两个地方：
1. 每个子层输出（Post-LN）：`output = LayerNorm(x + Sublayer(x))`
2. 每个子层输入（Pre-LN）：`output = x + Sublayer(LayerNorm(x))`

```python
class PostLayerNorm:
    """Post-LayerNorm：先子层再归一化（原始 Transformer）"""

    def __init__(self, hidden_size):
        self.norm = LayerNorm(hidden_size)

    def forward(self, x, sublayer_output):
        """前向传播

        参数:
            x: 残差连接的输入
            sublayer_output: 子层的输出（如 attention 或 FFN）
        """
        # 残差连接 + LayerNorm
        return self.norm.forward(x + sublayer_output)

class PreLayerNorm:
    """Pre-LayerNorm：先归一化再子层"""

    def __init__(self, hidden_size):
        self.norm = LayerNorm(hidden_size)

    def forward(self, x, sublayer_func):
        """前向传播

        参数:
            x: 输入
            sublayer_func: 子层函数
        """
        # LayerNorm -> Sublayer -> 残差连接
        x_norm = self.norm.forward(x)
        sublayer_output = sublayer_func(x_norm)
        return x + sublayer_output

# 示例
np.random.seed(42)
post_ln = PostLayerNorm(hidden_size=8)
pre_ln = PreLayerNorm(hidden_size=8)

x = np.random.randn(2, 5, 8).astype(np.float32)
sublayer_out = np.random.randn(2, 5, 8).astype(np.float32)

post_output = post_ln.forward(x, sublayer_out)
pre_output = pre_ln.forward(x, lambda x: np.random.randn(2, 5, 8).astype(np.float32))

print(f"Post-LN 输出形状: {post_output.shape}")
print(f"Pre-LN 输出形状: {pre_output.shape}")
```

## 数值稳定性

LayerNorm 在计算方差时需要注意数值稳定性：

```python
def layer_norm_stable(x, gamma, beta, eps=1e-6):
    """数值稳定的 LayerNorm 实现

    使用 var = E[x²] - E[x]² 来计算方差，避免 (x - μ)² 的数值问题
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
    var = mean_sq - mean ** 2

    x_norm = (x - mean) / np.sqrt(var + eps)
    output = gamma * x_norm + beta

    return output

# 测试数值稳定性
x = np.array([1e8, 1e8 + 1, 1e8 + 2], dtype=np.float32).reshape(1, 3)
gamma = np.ones(1, dtype=np.float32)
beta = np.zeros(1, dtype=np.float32)

out = layer_norm(x, gamma, beta)
print(f"数值稳定性测试:")
print(f"输入: {x.flatten()}")
print(f"输出: {out.flatten()}")
print(f"输出均值（应为0）: {out.mean():.6f}")
```

## 常见误区

**误区一：混淆 BatchNorm 和 LayerNorm 的轴**

```python
x = np.random.randn(2, 3, 4).astype(np.float32)

# BatchNorm: 在特征维度上归一化，使用 batch 统计
# axis=0 对 (3,4) 维度计算统计量
bn_mean = np.mean(x, axis=(1, 2), keepdims=True)  # 错误！

# LayerNorm: 在特征维度（最后一个）上归一化
ln_mean = np.mean(x, axis=-1, keepdims=True)  # 正确！
```

**误区二：忘记 eps**

```python
# 错误：没有 eps 可能导致除零
var = np.var(x, axis=-1, keepdims=True)
x_norm = (x - mean) / np.sqrt(var)  # 当方差接近0时出问题

# 正确：添加 eps
x_norm = (x - mean) / np.sqrt(var + 1e-6)
```

**误区三：gamma 和 beta 的形状错误**

gamma 和 beta 应该与 hidden_size 匹配：

```python
# 正确：gamma 和 beta 的形状应该是 (hidden_size,)
x = np.random.randn(2, 5, 8)  # hidden_size = 8
gamma = np.ones(8)
beta = np.zeros(8)

output = layer_norm(x, gamma, beta)
```

## API 总结

| 类/函数 | 描述 |
|--------|------|
| `layer_norm(x, gamma, beta)` | LayerNorm 函数实现 |
| `LayerNorm` | LayerNorm 类 |
| `PostLayerNorm` | Post-LayerNorm 实现 |
| `PreLayerNorm` | Pre-LayerNorm 实现 |

LayerNorm 是现代深度学习模型的重要组件。理解其原理和实现，对于掌握 Transformer 架构至关重要。
