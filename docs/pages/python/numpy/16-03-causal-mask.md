# 因果掩码

因果掩码（Causal Mask）是语言模型中确保自回归生成正确性的关键组件。在训练和推理时，因果掩码确保模型只能看到当前位置及其之前的信息，而不能"看到"未来的 token。本篇文章详细介绍因果掩码的原理、实现方式，以及如何在缩放点积注意力中应用掩码。

## 为什么需要因果掩码

在语言模型中，我们训练模型预测下一个 token。如果不添加掩码，模型在预测位置 i 的 token 时可以看到位置 i+1, i+2, ... 的真实 token，这会导致信息泄露（data leakage），模型会学到"作弊"而不是真正理解上下文。

**因果（causal）**意味着"原因在前，结果在后"。在注意力机制中，因果掩码确保位置 i 只能关注位置 0 到 i 的 token。

```python
import numpy as np

def create_causal_mask(seq_len):
    """创建因果掩码

    返回一个下三角形矩阵，True 表示需要 mask（看不到）

    参数:
        seq_len: 序列长度
    返回:
        mask: (seq_len, seq_len) 的布尔矩阵
    """
    # 方法1：使用 np.triu
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    return mask

# 示例
seq_len = 5
mask = create_causal_mask(seq_len)
print("因果掩码（下三角为 False，上三角为 True）:")
print(mask)
print()
print("可视化（* 表示被 mask，. 表示可见）:")
for i in range(seq_len):
    row = ""
    for j in range(seq_len):
        row += "* " if mask[i, j] else ". "
    print(row)
```

## 在注意力中使用因果掩码

```python
def softmax(x, axis=-1):
    """Softmax 函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def attention_with_causal_mask(Q, K, V, mask=None):
    """带因果掩码的缩放点积注意力

    参数:
        Q: (batch, heads, seq_len, head_dim)
        K: (batch, heads, seq_len, head_dim)
        V: (batch, heads, seq_len, head_dim)
        mask: (seq_len, seq_len) 或 (batch, seq_len, seq_len)
    返回:
        output: (batch, heads, seq_len, head_dim)
        attention_weights: (batch, heads, seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # 计算注意力分数
    scores = np.einsum('bhnd,bhmd->bhnm', Q, K)
    scores = scores / np.sqrt(d_k)

    # 创建因果掩码并应用
    seq_len = Q.shape[2]
    causal_mask = create_causal_mask(seq_len)  # True = mask 掉

    # 广播掩码到 batch 和 heads 维度
    causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq, seq)

    if mask is not None:
        # 如果提供了额外的掩码，合并
        mask = mask[:, np.newaxis, :, :] if len(mask.shape) == 2 else mask[:, np.newaxis, :, :]
        combined_mask = causal_mask | mask
    else:
        combined_mask = causal_mask

    # 应用掩码：被 mask 的位置设为很大的负数
    scores = np.where(combined_mask, -1e9, scores)

    # Softmax
    attention_weights = softmax(scores, axis=-1)

    # 加权求和
    output = np.einsum('bhnm,bhmd->bhnd', attention_weights, V)

    return output, attention_weights

# 示例
np.random.seed(42)
batch_size = 2
num_heads = 4
seq_len = 6
head_dim = 32

Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

output, attention_weights = attention_with_causal_mask(Q, K, V)
print(f"Q 形状: {Q.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")

# 验证：位置 0 只能关注位置 0
print(f"\n位置 0 的注意力权重（第一行，第一头）: {attention_weights[0, 0, 0]}")
# 位置 0 只能看到自己，所以第一个元素应该是 1，其他应该是 0
```

## 可视化因果掩码的效果

```python
def visualize_attention_with_mask(attention_weights, tokens, mask):
    """可视化带掩码的注意力权重

    参数:
        attention_weights: (batch, heads, seq, seq)
        tokens: token 列表
        mask: (seq, seq) 掩码
    """
    batch, heads, seq, _ = attention_weights.shape

    print(f"\n=== 带因果掩码的注意力 ===")
    print(f"序列: {tokens}")
    print(f"掩码形状: {mask.shape}")
    print()

    # 选择第一个 batch 和第一个 head
    weights = attention_weights[0, 0]  # (seq, seq)

    # 打印掩码
    print("掩码（* = 被 mask）:")
    for i in range(seq):
        row = f"{tokens[i]:>6} |"
        for j in range(seq):
            row += "* " if mask[i, j] else ". "
        print(row)

    print()
    print("注意力权重（每行和为1）:")
    for i in range(seq):
        row = f"{tokens[i]:>6} |"
        for j in range(seq):
            w = weights[i, j]
            char = '***' if w > 0.5 else ('**' if w > 0.2 else ('*' if w > 0.05 else ' '))
            row += f"{char:>3}"
        print(row)

# 示例
tokens = ['[BOS]', 'The', 'cat', 'eats', 'the', 'fish', '[EOS]']
seq_len = len(tokens)
mask = create_causal_mask(seq_len)

# 创建随机注意力权重作为示例
np.random.seed(42)
attention_weights = np.random.rand(1, 1, seq_len, seq_len)
attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)

# 应用因果约束（重新归一化可见位置）
causal_mask = ~mask  # True = 可见
for i in range(seq_len):
    visible_sum = attention_weights[0, 0, i, causal_mask[i]].sum()
    attention_weights[0, 0, i] = attention_weights[0, 0, i] * causal_mask[i] / visible_sum

visualize_attention_with_mask(attention_weights, tokens, mask)
```

## 填充掩码（Padding Mask）

在实际应用中，序列长度不同，需要 padding。使用填充掩码来忽略 padding：

```python
def create_padding_mask(seq_lens, max_len):
    """创建填充掩码

    参数:
        seq_lens: 每个序列的实际长度
        max_len: 最大长度
    返回:
        mask: (batch_size, max_len) 的布尔矩阵，True 表示 padding
    """
    batch_size = len(seq_lens)
    mask = np.zeros((batch_size, max_len), dtype=bool)

    for i, seq_len in enumerate(seq_lens):
        mask[i, seq_len:] = True  # padding 位置为 True

    return mask

def create_combined_mask(seq_lens, max_len):
    """创建组合掩码（因果 + 填充）

    参数:
        seq_lens: 每个序列的实际长度
        max_len: 最大长度
    返回:
        combined_mask: (batch, max_len, max_len)
    """
    batch_size = len(seq_lens)
    padding_mask = create_padding_mask(seq_lens, max_len)  # (batch, max_len)

    # 创建因果掩码
    causal_mask = create_causal_mask(max_len)  # (max_len, max_len)

    # 组合掩码
    # combined_mask[i, j, k] = True 表示位置 (i, j) 不能看到位置 (i, k)
    combined_mask = causal_mask[np.newaxis, :, :] | padding_mask[:, np.newaxis, :]

    return combined_mask, padding_mask

# 示例
seq_lens = [5, 3, 7]  # 三个不同长度的序列
max_len = max(seq_lens)

combined_mask, padding_mask = create_combined_mask(seq_lens, max_len)
print(f"填充掩码:\n{padding_mask}")
print(f"\n组合掩码形状: {combined_mask.shape}")
print(f"第一个序列的掩码（True=mask）:\n{combined_mask[0]}")
```

## 解码时的因果掩码

在自回归生成（解码）时，需要逐个生成 token，此时掩码会动态更新：

```python
def autoregressive_mask(current_pos, max_len):
    """创建自回归解码时的掩码

    参数:
        current_pos: 当前要预测的位置（0-indexed）
        max_len: 最大长度
    返回:
        mask: (max_len,) 的布尔向量，True 表示当前位置能看到的位置
    """
    mask = np.zeros(max_len, dtype=bool)
    mask[:current_pos + 1] = True  # 只能看到当前位置及之前
    return mask

def decode_step(attention_weights, current_pos, max_len):
    """解码步骤：获取当前位置的注意力

    参数:
        attention_weights: 完整的注意力权重 (seq, seq)
        current_pos: 当前位置
        max_len: 最大长度
    返回:
        当前步骤的注意力分布
    """
    mask = autoregressive_mask(current_pos, max_len)
    # 注意：这里只是演示，实际解码时不会计算完整的注意力矩阵
    return attention_weights[current_pos, mask]

# 示例：模拟解码过程
print("\n=== 自回归解码演示 ===")
print("预测下一个 token 时，只能看到之前的 token\n")

for pos in range(5):
    visible_positions = list(range(pos + 1))
    print(f"位置 {pos} 可见的位置: {visible_positions}")
```

## 注意力权重的重新归一化

应用因果掩码后，注意力权重需要重新归一化，确保每行和为 1：

```python
def apply_causal_mask_and_normalize(scores, causal_mask):
    """应用因果掩码并重新归一化注意力权重

    参数:
        scores: (batch, heads, seq, seq) 注意力分数
        causal_mask: (seq, seq) 布尔矩阵，True = mask
    返回:
        attention_weights: 归一化后的注意力权重
    """
    # 复制分数以避免修改原始数据
    scores = scores.copy()

    # 应用掩码
    scores[causal_mask] = -1e9

    # Softmax 归一化
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return attention_weights

# 示例
scores = np.random.randn(2, 4, 6, 6).astype(np.float32)
seq_len = 6
causal_mask = create_causal_mask(seq_len)

attention_weights = apply_causal_mask_and_normalize(scores, causal_mask)

# 验证：每行和为1（对于可见位置）
print("验证因果注意力权重（可见位置和应为1）:")
for i in range(seq_len):
    visible_sum = attention_weights[0, 0, i, ~causal_mask[i]].sum()
    print(f"  位置 {i} 的可见位置和: {visible_sum:.6f}")
```

## 常见误区

**误区一：混淆掩码的语义**

掩码的 True/False 语义可能混乱：

```python
# 两种掩码表示方法容易混淆：
# 方法1：True = mask（需要隐藏）
mask1 = np.triu(np.ones((5, 5)), k=1).astype(bool)  # 上三角为 True

# 方法2：False = mask（需要隐藏）
mask2 = np.tril(np.ones((5, 5)), k=-1).astype(bool)  # 下三角为 False

# 在代码中保持一致很重要！
```

**误区二：忘记在填充位置应用掩码**

当有 padding 时，需要同时应用因果掩码和填充掩码：

```python
# 正确的做法：组合两种掩码
combined_mask = causal_mask | padding_mask
scores = np.where(combined_mask, -1e9, scores)
```

**误区三：在推理时错误地应用因果掩码**

在推理时，解码是逐个 token 进行的，不需要完整的因果掩码矩阵：

```python
# 推理时的正确做法：逐个生成
# 当前只计算到位置 t 的注意力，不需要 t+1 及之后的信息
# 但实现上通常用一个大的掩码矩阵，每次取前 t+1 列
```

## API 总结

| 函数 | 描述 |
|------|------|
| `create_causal_mask(seq_len)` | 创建因果掩码 |
| `create_padding_mask(seq_lens, max_len)` | 创建填充掩码 |
| `create_combined_mask(seq_lens, max_len)` | 组合掩码 |
| `apply_causal_mask_and_normalize(scores, mask)` | 应用掩码并归一化 |

因果掩码是语言模型正确学习自回归特性的关键。理解其原理和实现，对于构建完整的 Transformer 模型至关重要。
