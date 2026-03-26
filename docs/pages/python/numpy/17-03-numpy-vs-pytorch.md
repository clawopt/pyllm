# NumPy 与 PyTorch 实现对比

本篇文章对比 NumPy 和 PyTorch 两种框架在实现 Transformer 核心组件时的异同。通过这个对比，你可以：
1. 理解 PyTorch 背后的 NumPy 原理
2. 学会在纯 NumPy 环境中实现神经网络组件
3. 理解两者的 API 设计差异和性能特点

## 基础操作对比

### 数组创建

```python
import numpy as np

# NumPy：直接创建
arr_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
zeros_np = np.zeros((3, 4), dtype=np.float32)
rand_np = np.random.randn(3, 4).astype(np.float32)

print("NumPy 数组创建:")
print(f"  形状: {arr_np.shape}, dtype: {arr_np.dtype}")
print(f"  zeros 形状: {zeros_np.shape}")
print(f"  random 形状: {rand_np.shape}")
```

PyTorch 等效代码：
```python
# import torch
# arr_pt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
# zeros_pt = torch.zeros(3, 4)
# rand_pt = torch.randn(3, 4)
```

### 矩阵乘法

```python
# NumPy：使用 @ 或 np.matmul
A = np.random.randn(3, 4).astype(np.float32)
B = np.random.randn(4, 5).astype(np.float32)
C_np = A @ B  # 或 np.matmul(A, B)

# einsum
D_np = np.einsum('ij,jk->ik', A, B)

print("\nNumPy 矩阵乘法:")
print(f"  A @ B 形状: {C_np.shape}")
print(f"  einsum 形状: {D_np.shape}")
```

PyTorch 等效：
```python
# A_pt = torch.randn(3, 4)
# B_pt = torch.randn(4, 5)
# C_pt = A_pt @ B_pt  # 或 torch.matmul(A_pt, B_pt)
# D_pt = torch.einsum('ij,jk->ik', A_pt, B_pt)
```

### Softmax 实现对比

```python
def numpy_softmax(x, axis=-1):
    """NumPy 实现 Softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def pytorch_softmax(x, dim=-1):
    """PyTorch 等效（伪代码）"""
    # return torch.softmax(x, dim=dim)
    pass

# 测试
x = np.random.randn(2, 5).astype(np.float32)
softmax_np = numpy_softmax(x, axis=-1)

print("\n=== Softmax 对比 ===")
print(f"输入形状: {x.shape}")
print(f"输出形状: {softmax_np.shape}")
print(f"每行和（应为1）: {softmax_np.sum(axis=-1)}")
```

## LayerNorm 实现对比

```python
class NumPyLayerNorm:
    """NumPy 实现的 LayerNorm"""

    def __init__(self, hidden_size, eps=1e-6):
        self.gamma = np.ones(hidden_size, dtype=np.float32)
        self.beta = np.zeros(hidden_size, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        """x: (batch, seq, hidden_size)"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

PyTorch 等效：
```python
# import torch.nn as nn
# torch_ln = nn.LayerNorm(normalized_shape=hidden_size, eps=eps)
#
# # PyTorch 内部自动处理：
# # 1. 计算均值和方差
# # 2. 归一化
# # 3. 应用 gamma 和 beta
# output = torch_ln(x)  # x 是 torch.Tensor
```

## 多头注意力对比

```python
class NumPyMultiHeadAttention:
    """NumPy 实现的多头注意力"""

    def __init__(self, hidden_size, num_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 初始化权重
        scale = np.sqrt(2.0 / hidden_size)
        self.W_q = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.W_k = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.W_v = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.W_o = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale

    def forward(self, x, mask=None):
        """x: (batch, seq, hidden_size)"""
        batch_size, seq_len, _ = x.shape

        # QKV 投影
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 重塑为多头
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 注意力分数
        scores = np.einsum('bhnd,bhmd->bhnm', Q, K) / np.sqrt(self.head_dim)

        # 应用掩码
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # 加权求和
        context = np.einsum('bhnm,bhmd->bhnd', attention_weights, V)

        # 合并多头
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = context @ self.W_o

        return output, attention_weights
```

PyTorch 等效：
```python
# import torch.nn as nn
# import math
#
# class PyTorchAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads
#         self.q_proj = nn.Linear(hidden_size, hidden_size)
#         self.k_proj = nn.Linear(hidden_size, hidden_size)
#         self.v_proj = nn.Linear(hidden_size, hidden_size)
#         self.out_proj = nn.Linear(hidden_size, hidden_size)
#
#     def forward(self, x, mask=None):
#         batch_size, seq_len, hidden_size = x.shape
#
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)
#
#         q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#         attention_weights = torch.softmax(scores, dim=-1)
#         context = torch.matmul(attention_weights, v)
#
#         context = context.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
#         return self.out_proj(context), attention_weights
```

## 结果对比验证

```python
def compare_numpy_pytorch():
    """对比 NumPy 和 PyTorch 的结果（伪代码）"""

    print("\n=== NumPy vs PyTorch 结果对比 ===\n")

    # 假设我们用相同的权重和输入
    np.random.seed(42)
    hidden_size = 64
    num_heads = 4
    batch_size = 2
    seq_len = 8

    # NumPy 实现
    numpy_attn = NumPyMultiHeadAttention(hidden_size, num_heads)
    x_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    output_np, weights_np = numpy_attn.forward(x_np)

    print("NumPy 实现:")
    print(f"  输出形状: {output_np.shape}")
    print(f"  输出均值: {output_np.mean():.6f}")
    print(f"  输出标准差: {output_np.std():.6f}")
    print(f"  注意力权重形状: {weights_np.shape}")
    print(f"  注意力权重和（第一样本，第一头）: {weights_np[0, 0].sum():.6f}")

    # PyTorch 等效说明
    print("\nPyTorch 实现（结果应与 NumPy 相同）:")
    print("  import torch")
    print("  torch.manual_seed(42)")
    print("  x_pt = torch.randn(batch_size, seq_len, hidden_size)")
    print("  # 使用相同的权重初始化")
    print("  # 输出形状: torch.Size([2, 8, 64])")
    print("  # 数值应与 NumPy 几乎相同（浮点误差内）")

compare_numpy_pytorch()
```

## 主要区别总结

| 特性 | NumPy | PyTorch |
|------|-------|---------|
| **数据类型** | ndarray | Tensor |
| **GPU 支持** | 不支持 | 自动支持 CUDA |
| **自动微分** | 不支持 | autograd |
| **广播** | 相同 | 相同 |
| **API 命名** | np.function | torch.function |
| **内存共享** | view/copy | view/clone |
| **设备管理** | 不适用 | .to(device) |

## 数值差异

NumPy 和 PyTorch 在相同输入和权重下应该产生相同结果，但由于浮点运算顺序和实现细节的细微差异，可能存在极小的数值差异：

```python
def check_numerical_difference():
    """检查数值差异"""
    print("\n=== 数值差异检查 ===\n")

    # 模拟相同计算的两种方式
    np.random.seed(42)
    x = np.random.randn(100, 100).astype(np.float32)

    # NumPy 计算
    result_np = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    # 另一种计算顺序（模拟可能的差异）
    exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
    result_np2 = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    max_diff = np.abs(result_np - result_np2).max()

    print(f"相同公式不同实现的数值差异: {max_diff:.10f}")
    print(f"差异来源: 浮点运算的舍入误差")

check_numerical_difference()
```

## 性能对比

```python
import time

def performance_comparison():
    """性能对比（NumPy）"""
    print("\n=== 性能对比（NumPy）===\n")

    # 创建大型矩阵
    size = 1000
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    # 矩阵乘法性能
    start = time.time()
    for _ in range(10):
        C = A @ B
    elapsed = time.time() - start

    print(f"矩阵乘法 ({size}x{size}) x 10:")
    print(f"  NumPy 耗时: {elapsed*1000:.2f}ms")
    print(f"  PyTorch CPU 应类似")
    print(f"  PyTorch GPU 应快 10-100x")

    # Softmax 性能
    x = np.random.randn(32, 512, 512).astype(np.float32)

    start = time.time()
    for _ in range(100):
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
        result = exp_x / exp_x.sum(axis=-1, keepdims=True)
    elapsed = time.time() - start

    print(f"\nSoftmax (32, 512, 512) x 100:")
    print(f"  NumPy 耗时: {elapsed*1000:.2f}ms")

performance_comparison()
```

## 互操作性

NumPy 和 PyTorch 可以相互转换：

```python
def numpy_pytorch_interop():
    """NumPy 和 PyTorch 互操作"""

    print("\n=== NumPy-PyTorch 互操作 ===\n")

    # NumPy -> PyTorch
    # import torch
    # np_array = np.random.randn(3, 4).astype(np.float32)
    # torch_tensor = torch.from_numpy(np_array)  # 共享内存
    # torch_tensor = torch.tensor(np_array)  # 复制

    # PyTorch -> NumPy
    # torch_tensor = torch.randn(3, 4)
    # np_array = torch_tensor.numpy()  # 如果在 CPU 上，共享内存

    print("NumPy 数组可以 zero-copy 转换为 PyTorch 张量")
    print("PyTorch 张量可以 zero-copy 转换为 NumPy 数组（仅 CPU）")
    print("GPU 上的 PyTorch 张量需要先复制到 CPU")

    # 示例
    np_array = np.random.randn(3, 4).astype(np.float32)
    print(f"\n原始 NumPy 数组 ID: {id(np_array)}")

    # PyTorch 等效代码
    # torch_tensor = torch.from_numpy(np_array)
    # print(f"PyTorch 张量 ID: {id(torch_tensor)}")  # 相同（共享内存）
    #
    # np_back = torch_tensor.numpy()
    # print(f"转回 NumPy ID: {id(np_back)}")  # 相同

numpy_pytorch_interop()
```

## 何时使用 NumPy vs PyTorch

**使用 NumPy 的场景**：
- 理解和学习底层原理
- 不需要 GPU 的小规模计算
- 快速原型开发
- 与纯 Python 代码集成

**使用 PyTorch 的场景**：
- 实际训练深度学习模型
- 需要 GPU 加速
- 需要自动微分
- 生产环境部署

## 总结

NumPy 和 PyTorch 在底层操作上非常相似，理解 NumPy 有助于深入理解 PyTorch 的工作原理。两者主要区别在于：
1. PyTorch 提供 GPU 支持和自动微分
2. PyTorch 的 nn.Module 提供了更高层次的抽象
3. PyTorch 的张量可以追踪计算图支持梯度反向传播

通过本系列的学习，你已经掌握了使用纯 NumPy 实现 Transformer 核心组件的能力，这为理解更复杂的深度学习框架奠定了坚实基础。
