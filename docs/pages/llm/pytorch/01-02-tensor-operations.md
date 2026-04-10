# 1.2 Tensor 操作精要——LLM 中最常用的 20 个操作

> 如果你用过 NumPy，那么 PyTorch 的 Tensor 操作你会觉得似曾相识。但 LLM 场景有一些**高频操作**是普通教程很少强调的——比如 Multi-head Attention 中的 head 分割与合并、KV Cache 的索引操作、Attention Mask 的构造。这一节我们聚焦这些"LLM 必备"的 Tensor 操作。

## 创建 Tensor

### 基础创建方式

```python
import torch
import torch.nn.functional as F

# 从 Python 列表/元组创建
t1 = torch.tensor([1., 2., 3.])          # shape: (3,)
t2 = torch.tensor([[1, 2], [3, 4]])       # shape: (2, 2)

# 特殊值创建（LLM 中常用）
zeros = torch.zeros(2, 3)                 # 全零 → Attention Mask 初始化
ones = torch.ones(4)                      # 全一
randn = torch.randn(3, 4096)              # 正态随机 → 模型权重初始化
arange = torch.arange(0, 10)              # 序列 → Position IDs
full = torch.full((3, 3), -1e9)           # 填充指定值 → Causal Mask 的填充值
eye = torch.eye(4)                        # 单位矩阵
```

### 与模型相关的创建模式

```python
"""
Tensor 创建在 LLM 中的典型用法
"""

def demo_llm_tensor_creation():
    """演示 LLM 相关的 Tensor 创建模式"""

    print("=" * 60)
    print("📐 LLM 中的 Tensor 创建模式")
    print("=" * 60)

    # === 1. Embedding 权重表 ===
    vocab_size, embed_dim = 32000, 4096
    embedding_weights = torch.randn(vocab_size, embed_dim)
    print(f"\n[1] Embedding 权重表:")
    print(f"   shape: {embedding_weights.shape}")
    print(f"   含义: 每个 token 对应一个 {embed_dim} 维向量")

    # === 2. Position IDs ===
    seq_len = 128
    position_ids = torch.arange(seq_len)
    print(f"\n[2] Position IDs:")
    print(f"   shape: {position_ids.shape}")
    print(f"   内容: [0, 1, 2, ..., {seq_len-1}]")

    # === 3. Causal Mask（因果注意力掩码）===
    # 下三角矩阵：位置 i 只能看到位置 <= j 的内容
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"\n[3] Causal Mask (前8×8):")
    print(f"   {causal_mask[:8, :8].int()}")

    # === 4. Attention Padding Mask ===
    batch_size, max_len = 4, 128
    actual_lengths = torch.tensor([100, 80, 128, 50])  # 各样本的实际长度
    padding_mask = torch.arange(max_len).unsqueeze(0) < actual_lengths.unsqueeze(1)
    print(f"\n[4] Padding Mask:")
    print(f"   shape: {padding_mask.shape}")  # (4, 128)
    print(f"   第1行有效长度100: True×100 + False×28")
    print(f"   第4行有效长度50:  True×50  + False×78")

    # === 5. 学习率调度相关的 Tensor ===
    warmup_steps = 500
    total_steps = 10000
    lr_lambda = torch.linspace(0, 1, warmup_steps)  # 线性预热
    print(f"\n[5] Learning Rate Warmup:")
    print(f"   前5个值: {lr_lambda[:5]}")
    print(f"   最后5个值: {lr_lambda[-5:]}")


if __name__ == "__main__":
    demo_llm_tensor_creation()
```

## 形状变换——LLM 中最高频的操作

在 Transformer 模型中，张量的形状变换无处不在。理解 `view` / `reshape` / `permute` / `transpose` 的区别和正确用法，是阅读任何 LLM 源码的前提。

### 核心操作一览

```python
"""
形状变换操作详解 —— 以 Transformer 中的 Q/K/V 投影为例
"""

def demo_shape_manipulation():
    """演示 LLM 中的关键形状变换"""

    print("=" * 60)
    print("🔧 形状变换操作详解")
    print("=" * 60)

    # 模拟一个 batch 的输入
    batch_size = 2      # 批次大小
    seq_len = 64        # 序列长度
    n_embed = 768        # 嵌入维度（GPT-2 Small）
    num_heads = 12       # 注意力头数
    head_dim = n_embed // num_heads  # 每个头的维度 = 64

    x = torch.randn(batch_size, seq_len, n_embed)
    print(f"\n输入 x shape: {x.shape}")  # (B, T, C)

    # === 操作 1: Linear 投影 (Q/K/V) ===
    q_proj = nn.Linear(n_embed, n_embed)
    q = q_proj(x)
    print(f"[Linear] Q projection: {x.shape} → {q.shape}")

    # === 操作 2: reshape 为多头形式 ===
    # 目标: (B, T, C) → (B, T, heads, head_dim) → (B, heads, T, head_dim)
    q_multihead = q.view(batch_size, seq_len, num_heads, head_dim)
    print(f"[view] Reshape to multi-head: {q.shape} → {q_multihead.shape}")

    # === 操作 3: transpose 调整维度顺序 ===
    # Attention 计算需要: (B, heads, T, head_dim)
    q_transposed = q_multihead.transpose(1, 2)
    print(f"[transpose] Swap dims 1&2: {q_multihead.shape} → {q_transposed.shape}")

    # === 操作 4: permute 多维同时转置 ===
    # 等价于多次 transpose 的组合
    q_permuted = q_multihead.permute(0, 2, 1, 3)
    print(f"[permute] Same result: {q_multihead.shape} → {q_permuted.shape}")
    assert torch.equal(q_transposed, q_permuted)

    # === 操作 5: contiguous ===
    # 为什么需要 contiguous?
    q_t = q_multihead.transpose(1, 2)
    print(f"\n[contiguous] 解释:")
    print(f"  transpose 后内存不连续: {q_t.is_contiguous()}")
    q_c = q_t.contiguous()
    print(f"  .contiguous() 后: {q_c.is_contiguous()}")

    # === 操作 6: unsqueeze / squeeze ===
    vec = torch.randn(n_embed)
    print(f"\n[unsqueeze/squeeze]:")
    print(f"  原始向量: {vec.shape}")                    # (768,)
    vec_unsqueezed = vec.unsqueeze(0).unsqueeze(0)      # (1, 1, 768)
    print(f"  unsqueeze两次后: {vec_unsqueezed.shape}")   # (1, 1, 768) — 变成 batch=1, seq=1
    vec_squeezed = vec_unsqueezed.squeeze()
    print(f"  squeeze 回去: {vec_squeezed.shape}")         # (768,)

    # === 实战: 完整的 Multi-head Attention 形状变换链 ===
    print(f"\n{'='*60}")
    print(f"📋 完整 MHA 形状变换链 (GPT-2 Small)")
    print(f"{'='*60}")

    B, T, C, H, D = 2, 64, 768, 12, 64
    x = torch.randn(B, T, C)

    steps = [
        ("输入", x.shape),
        ("Q/K/V Linear", (B, T, C)),
        ("view → (B,T,H,D)", (B, T, H, D)),
        ("transpose→(B,H,T,D)", (B, H, T, D)),
        ("Q·K^T / √d", (B, H, T, T)),           # Attention scores
        ("softmax(mask)", (B, H, T, T)),
        ("·V", (B, H, T, D)),                     # 加权求和
        ("transpose→(B,T,H,D)", (B, T, H, D)),
        ("view→(B,T,C)", (B, T, C)),               # 恢复原始形状
        ("Output Projection", (B, T, C)),
    ]

    for name, shape in steps:
        print(f"  {name:<25s} {str(shape):>20s}")


if __name__ == "__main__":
    demo_shape_manipulation()
```

输出：

```
============================================================
🔧 形状变换详解
============================================================

输入 x shape: torch.Size([2, 64, 768])
[Linear] Q projection: torch.Size([2, 64, 768]) → torch.Size([2, 64, 768])
[view] Reshape to multi-head: torch.Size([2, 64, 768]) → torch.Size([2, 64, 12, 64])
[transpose] Swap dims 1&2: torch.Size([2, 64, 12, 64]) → torch.Size([2, 12, 64, 64])
[permute] Same result: torch.Size([2, 64, 12, 64]) → torch.Size([2, 12, 64, 64])

[contiguous] 解释:
  transpose 后内存不连续: False
  .contiguous() 后: True

[unsqueeze/squeeze]:
  原始向量: torch.Size([768])
  unsqueeze两次后: torch.Size([1, 1, 768])
  squeeze 回去: torch.Size([768])

============================================================
📋 完整 MHA 形状变换链 (GPT-2 Small)
============================================================
  输入                       torch.Size([2, 64, 768])
  Q/K/V Linear             torch.Size([2, 64, 768])
  view → (B,T,H,D)          torch.Size([2, 64, 12, 64])
  transpose→(B,H,T,D)       torch.Size([2, 12, 64, 64])
  Q·K^T / √d               torch.Size([2, 12, 64, 64])
  softmax(mask)             torch.Size([2, 12, 64, 64])
  ·V                        torch.Size([2, 12, 64, 64])
  transpose→(B,T,H,D)       torch.Size([2, 64, 12, 64])
  view→(B,T,C)              torch.Size([2, 64, 768])
  Output Projection         torch.Size([2, 64, 768])
```

### ⚠️ 常见坑：contiguous 到底什么时候需要？

`transpose` / `permute` 返回的张量在内存中不是连续存储的。某些操作（如 `view`）要求内存连续：

```python
x = torch.randn(2, 3, 4)

# ✅ 直接 view（原始张量总是 contiguous）
y = x.view(2, 12)  # OK

# ❌ transpose 后直接 view 会报错！
z = x.transpose(0, 1)  # (3, 2, 4)，但内存不连续
try:
    z.view(3, 8)  # RuntimeError!
except RuntimeError as e:
    print(f"❌ 错误: {e}")

# ✅ 先 contiguous 再 view
z_cont = z.contiguous().view(3, 8)  # OK!

# 💡 规则：
# transpose/permute 之后如果要用 view → 需要 .contiguous()
# 如果只是计算（matmul/softmax 等）→ 不需要
```

## 索引与切片——高级技巧

### Boolean Mask（布尔掩码）

这是实现 **Attention Mask** 的核心操作：

```python
def demo_boolean_mask():
    """演示 Boolean Mask 在 Attention 中的应用"""

    # 场景：batch 中有不同长度的序列
    # 需要用 mask 让 padding 位置的 attention score 变成 -∞

    batch_size, seq_len, n_embed = 2, 5, 4
    scores = torch.randn(batch_size, seq_len, seq_len)  # Attention scores

    # 假设第1个样本实际长度=3，第2个样本实际长度=4
    actual_lens = torch.tensor([3, 4])

    # 构造 mask: (batch, 1, seq_len) 广播到 (batch, seq_len, seq_len)
    positions = torch.arange(seq_len).unsqueeze(0)  # (1, 5)
    mask = positions < actual_lens.unsqueeze(1)     # (2, 5) → bool
    mask = mask.unsqueeze(1)                        # (2, 1, 5) → 可广播

    print("原始 scores (sample 0):")
    print(scores[0])

    # 应用 mask: False 位置填 -inf
    masked_scores = scores.masked_fill(~mask, float('-inf'))

    print("\nMasked scores (sample 0, 第4、5列应为 -inf):")
    print(masked_scores[0])

    # softmax 后，padding 位置的权重为 0
    attn_weights = F.softmax(masked_scores, dim=-1)
    print("\nSoftmax 后 (sample 0, 第4、5列应为 0):")
    print(attn_weights[0])


if __name__ == "__main__":
    demo_boolean_mask()
```

### Gather 操作——Embedding Lookup 的本质

```python
def demo_gather():
    """gather 是 Embedding 层的核心操作"""

    # Embedding 本质上就是一个 gather 操作:
    # 从一个大矩阵中按 index 取出对应的行

    embedding_table = torch.randn(100, 16)  # 100 个 token，每个 16 维
    token_ids = torch.tensor([5, 23, 77, 3])  # 4 个 token

    # 方式一：直接索引（最常用）
    embeddings = embedding_table[token_ids]
    print(f"直接索引: {embeddings.shape}")  # (4, 16)

    # 方式二：torch.gather（更通用）
    # gather 要求 index 的维度与 input 一致
    idx = token_ids.unsqueeze(-1).expand(-1, 16)  # (4, 16)
    embeddings_gather = torch.gather(embedding_table.unsqueeze(0).expand(4, -1, -1), 1, idx.unsqueeze(1))
    print(f"gather 结果: {embeddings_gather.shape}")  # (4, 1, 16)

    # 验证两者等价
    assert torch.allclose(embeddings, embeddings_gather.squeeze(1))


if __name__ == "__main__":
    demo_gather()
```

## 数学运算——LLM 计算的核心

### matmul (@)：Attention 得分计算

```python
def demo_attention_matmul():
    """演示 Attention 中的矩阵乘法"""

    B, H, T, D = 2, 8, 32, 64  # batch, heads, seq_len, head_dim

    Q = torch.randn(B, H, T, D)  # Query
    K = torch.randn(B, H, T, D)  # Key

    # Attention score = Q @ K^T / sqrt(d_k)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"K^T shape: {K.transpose(-2,-1).shape}")
    print(f"Attention scores shape: {attn_scores.shape}")  # (B, H, T, T)


if __name__ == "__main__":
    demo_attention_matmul()
```

### softmax：归一化

```python
def demo_softmax_tricks():
    """Softmax 在 LLM 中的使用细节"""

    scores = torch.tensor([1.0, 2.0, 3.0, 100000.0])  # 包含极大值

    # 普通 softmax（可能溢出）
    try:
        normal_softmax = F.softmax(scores, dim=-1)
        print(f"普通 softmax: {normal_softmax}")
    except Exception as e:
        print(f"普通 softmax 出错: {e}")

    # PyTorch 内部自动处理了数值稳定性
    stable_softmax = F.softmax(torch.tensor([1., 2., 3., 100.]), dim=-1)
    print(f"稳定 softmax: {stable_softmax}")
    print(f"和为 1: {stable_softmax.sum():.6f}")


if __name__ == "__main__":
    demo_softmax_tricks()
```

### LayerNorm / RMSNorm

```python
def demo_normalization():
    """LayerNorm vs RMSNorm 对比"""

    x = torch.randn(2, 4, 768)  # batch=2, seq=4, dim=768

    # LayerNorm (PyTorch 内置)
    layer_norm = nn.LayerNorm(768)
    ln_out = layer_norm(x)
    ln_mean = ln_out.mean(dim=-1)
    ln_var = ln_out.var(dim=-1)
    print(f"LayerNorm 输出均值: {ln_mean[:2]}")   # ≈ 0
    print(f"LayerNorm 输出方差: {ln_var[:2]}")      # ≈ 1

    # RMSNorm (LLaMA 使用，PyTorch 无内置，手写)
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight

    rms_norm = RMSNorm(768)
    rn_out = rms_norm(x)
    rn_rms = torch.sqrt(rn_out.mean(dim=-1) ** 2 + rn_out.var(dim=-1))
    print(f"\nRMSNorm 输出 RMS: {rn_rms[:2]}")       # ≈ 1.0


if __name__ == "__main__":
    demo_normalization()
```

### einsum：Einstein 求和约定

`einsum` 用简洁的字符串表示复杂的张量运算。在某些 LLM 实现中用于表达 Attention 计算：

```python
def demo_einsum():
    """einsum 在 Attention 中的应用"""

    B, H, T, D = 2, 8, 32, 64
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)

    # 标准 Attention 用 einsum 表示:
    # "bhtd,bhsd->bhts" 表示:
    #   b=batch, h=heads, t=target_seq, d=head_dim, s=source_seq
    #   Q(b,h,t,d) × K(b,h,s,d)^T → scores(b,h,t,s)
    scores = torch.einsum('bhtd,bhsd->bhts', Q, K) / (D ** 0.5)

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.einsum('bhts,bhsd->bhtd', attn, V)

    print(f"einsum Attention 输出: {output.shape}")  # (B, H, T, D)

    # 验证与 matmul 结果一致
    scores_matmul = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
    assert torch.allclose(scores, scores_matmul, atol=1e-5)
    print("✅ einsum 与 matmul 结果一致")


if __name__ == "__main__":
    demo_einsum()
```

## 广播机制（Broadcasting）

广播机制让不同形状的张量可以运算——这在 **Positional Encoding 加法** 和 **Residual Connection** 中大量使用：

```python
def demo_broadcasting():
    """广播机制在 LLM 中的应用"""

    # === 应用 1: Position Embedding + Token Embedding ===
    B, T, C = 2, 10, 768
    token_emb = torch.randn(B, T, C)        # (2, 10, 768)
    pos_emb = torch.randn(T, C)             # (10, 768) — 没有 batch 维度！

    # 广播: (10, 768) → (1, 10, 768) → (2, 10, 768)
    combined = token_emb + pos_emb
    print(f"Token emb: {token_emb.shape}")
    print(f"Position emb: {pos_emb.shape}")
    print(f"相加结果: {combined.shape}")  # (2, 10, 768) ✅ 自动广播

    # === 应用 2: Residual Connection + LayerNorm ===
    x = torch.randn(B, T, C)              # (2, 10, 768)
    sublayer_output = torch.randn(B, T, C)  # (2, 10, 768)

    residual = x + sublayer_output         # 逐元素相加
    normalized = nn.LayerNorm(C)(residual)
    print(f"\nResidual + Norm: {normalized.shape}")

    # === 应用 3: Causal Mask 广播 ===
    mask = torch.tril(torch.ones(T, T))   # (10, 10) — 下三角
    scores = torch.randn(B, T, T)         # (2, 10, 10)

    # mask (10,10) 广播到 (2, 10, 10)
    masked = scores.masked_fill(mask == 0, float('-inf'))
    print(f"\nCausal mask 广播: {masked.shape}")


if __name__ == "__main__":
    demo_broadcasting()
```

## 拼接与分割

```python
def demo_cat_split():
    """cat/split/chunk 在批处理中的应用"""

    # cat: 将多个序列拼成一个 batch
    seq_a = torch.randn(3, 768)
    seq_b = torch.randn(5, 768)
    seq_c = torch.randn(4, 768)

    # 动态 batching: 不同长度的序列 padding 后拼接
    padded_a = F.pad(seq_a, (0, 0, 0, 2), value=0)  # (5, 768)
    batch = torch.cat([padded_a, seq_b, seq_c], dim=0)  # (14, 768)
    print(f"cat 拼接: {[3,5,4]} → {list(batch.shape)}")

    # split: 按固定大小分割
    chunks = batch.split(5, dim=0)
    sizes = [c.shape[0] for c in chunks]
    print(f"split 分割(每块5): {sizes}")  # [5, 5, 4]

    # chunk: 均匀分割
    chunks_even = batch.chunk(3, dim=0)
    sizes_even = [c.shape[0] for c in chunks_even]
    print(f"chunk 分割(3块): {sizes_even}")  # [5, 5, 4]

    # stack: 新增维度堆叠（不同于 cat）
    stacked = torch.stack([seq_a[:1], seq_b[:1]], dim=0)
    print(f"stack 堆叠: {stacked.shape}")  # (2, 1, 768)


if __name__ == "__main__":
    demo_cat_split()
```

## 操作速查表

| 操作 | 函数 | 典型 LLM 用途 |
|-----|------|-------------|
| 改变形状 | `.view()` / `.reshape()` | MHA 的 head 分割 |
| 转置 | `.transpose()` / `.permute()` | Q/K/V 维度调整 |
| 内存连续化 | `.contiguous()` | transpose 后的 view |
| 增减维度 | `.unsqueeze()` / `.squeeze()` | 添加 batch/seq 维度 |
| 布尔掩码 | `.masked_fill()` | Attention Mask / Causal Mask |
| 索引取值 | `tensor[idx]` / `torch.gather()` | Embedding Lookup |
| 矩阵乘法 | `@` / `torch.matmul()` | Attention Q·K^T |
| 归一化 | `F.softmax()` | Attention weights |
| 归一化 | `nn.LayerNorm()` / RMSNorm | Transformer Block |
| 缩放 | `/ sqrt(d_k)` | Scaled Dot-Product Attention |
| 逐元素乘 | `*` / `torch.mul()` | SwiGLU 门控机制 |
| 求和约简 | `.sum()` / `.mean()` | Loss 计算 |
| 拼接 | `torch.cat()` | Dynamic Batching |
| 分割 | `.split()` / `.chunk()` | Batch 拆分处理 |
| Einstein 求和 | `torch.einsum()` | 简洁表达复杂 Attention |

下一节我们将深入 **Autograd 自动微分系统**——理解反向传播的本质。
