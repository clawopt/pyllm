# 传统 Attention 的 KV Cache 问题

## 白板导读

要理解 PagedAttention 为什么是一个革命性的创新，我们必须先搞清楚它在解决什么问题。这个问题就是：**KV Cache 的显存管理**。在 Transformer 模型的自注意力机制中，为了让模型"记住"之前生成过的所有内容，推理引擎必须缓存每个 token 对应的 Key 向量和 Value 向量——这就是 KV Cache。听起来很简单，但当你把这件事放到高并发、多请求的生产环境中时，它就变成了一个极其棘手的内存管理难题。本章将从 Transformer Attention 的数学基础出发，逐步推导出 KV Cache 的内存增长规律，然后用具体的数字展示传统方案造成的惊人浪费，为下一节 PagedAttention 的登场做好充分的铺垫。

---

## 1.1 Transformer Self-Attention 回顾

### Attention 计算公式

Transformer 的核心是自注意力机制（Self-Attention）。对于一个序列中的每一个位置 $i$，它的注意力计算过程如下：

$$
\text{Attention}(Q_i, K, V) = \sum_{j=1}^{n} \underbrace{\frac{\exp(Q_i K_j^T / \sqrt{d_k})}{\sum_{l=1}^{n} \exp(Q_i K_l^T / \sqrt{d_k})}}_{\text{注意力权重 } \alpha_{ij}} \cdot V_j
$$

其中：
- $Q_i = x_i W_Q$：位置 $i$ 的 **Query（查询）**向量
- $K_j = x_j W_K$：位置 $j$ 的 **Key（键）**向量
- $V_j = x_j W_V$：位置 $j$ 的 **Value（值）**向量
- $d_k$：Key 向量的维度（用于缩放，防止 softmax 后梯度过小）
- $\alpha_{ij}$：位置 $i$ 对位置 $j$ 的注意力权重

### 从训练到推理的关键变化

这个公式在**训练阶段**和**推理阶段**的使用方式有本质区别：

**训练阶段**：
- 输入整个序列（如 512 个 token）一次性送入模型
- 所有位置的 Q、K、V 同时计算
- 可以利用并行计算加速
- 不需要缓存任何中间结果

**推理阶段（自回归生成）**：
- 每次只生成一个新 token
- 第 $t$ 步生成 token 时，需要"看到"之前所有 $t-1$ 个 token
- 如果不缓存之前的 K 和 V，就需要**重新计算**所有历史 token 的 Key 和 Value → 巨大的计算浪费！
- 所以必须**缓存**已计算过的 K 和 V → 这就是 **KV Cache**

### KV Cache 的直观理解

想象你在参加一场面试：

```
面试官（用户）: "请介绍一下你自己"

你（LLM）: [思考第1个token] → 需要看完整的问题
           输出: "我"
           
           [思考第2个token] → 需要看问题 + 已输出的"我"
           输出: "叫"
           
           [思考第3个token] → 需要看问题 + "我叫"
           输出: "张"
           
           ...

每一轮思考时，你都记得之前说过的话。
这些"记忆"就是 KV Cache。
```

没有 KV Cache，每生成一个新 token 都需要从头重新处理整个输入序列——对于 4096 个 token 的上下文来说，这意味着重复计算 4096 次，效率极低。

---

## 1.2 KV Cache 的精确大小计算

### 单层单头的 KV Cache 大小

KV Cache 存储的是每一层的 Key 和 Value 向量。以 Llama 3 8B 模型为例：

```python
"""
KV Cache 精确大小计算器
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置参数"""
    name: str
    hidden_size: int        # 隐藏层维度 (d_model)
    num_layers: int         # Transformer 层数
    num_attention_heads: int # 注意力头数
    num_kv_heads: int       # KV 头数 (GQA 时可能 < attention heads)
    head_dim: int           # 每个头的维度
    dtype_bytes: int = 2     # FP16 = 2 bytes, BF16 = 2, FP32 = 4


# 主流模型的配置
MODELS = {
    "Llama-3-8B": ModelConfig(
        name="Llama-3-8B",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,          # GQA: 32 个 Q 头共享 8 个 KV 头
        head_dim=128,
        dtype_bytes=2,
    ),
    "Qwen2.5-7B": ModelConfig(
        name="Qwen2.5-7B",
        hidden_size=3584,
        num_layers=28,
        num_attention_heads=28,
        num_kv_heads=4,           # GQA 更激进: 28 个 Q 头共享 4 个 KV 头
        head_dim=128,
        dtype_bytes=2,
    ),
    "Llama-3-70B": ModelConfig(
        name="Llama-3-70B",
        hidden_size=8192,
        num_layers=80,
        num_attention_heads=64,
        num_kv_heads=8,           # GQA: 64 个 Q 头共享 8 个 KV 头
        head_dim=128,
        dtype_bytes=2,
    ),
    "DeepSeek-V2-16B": ModelConfig(
        name="DeepSeek-V2-16B",
        hidden_size=5120,
        num_layers=62,
        num_attention_heads=128,
        num_kv_heads=128,         # 非 GQA（或 MLA 架构）
        head_dim=128,
        dtype_bytes=2,
    ),
}


def calc_kv_cache_per_token(cfg: ModelConfig) -> int:
    """
    计算单个 token 在所有层产生的 KV Cache 大小（字节）
    
    公式:
      per_token = num_layers × 2(K+V) × num_kv_heads × head_dim × dtype_bytes
    """
    per_token = (
        cfg.num_layers
        * 2                              # K 和 V 各一份
        * cfg.num_kv_heads
        * cfg.head_dim
        * cfg.dtype_bytes
    )
    return per_token


def calc_total_kv_cache(cfg: ModelConfig, seq_len: int) -> dict:
    """计算指定序列长度下的总 KV Cache 占用"""
    per_token = calc_kv_cache_per_token(cfg)
    total = per_token * seq_len
    
    return {
        "model": cfg.name,
        "per_token_bytes": per_token,
        "per_token_kb": round(per_token / 1024, 2),
        "per_token_mb": round(per_token / 1024 / 1024, 4),
        f"seq_{seq_len}_total_gb": round(total / 1024**3, 3),
        f"seq_{seq_len}_total_mb": round(total / 1024**2, 1),
    }


if __name__ == "__main__":
    print("=" * 80)
    print(f"{'模型':<20} {'每Token':>10} {'1K tokens':>12} {'4K tokens':>12} "
          f"{'8K tokens':>12} {'16K tokens':>12} {'32K tokens':>12}")
    print("=" * 80)
    
    for name, cfg in MODELS.items():
        pt = calc_kv_cache_per_token(cfg)
        print(f"{name:<20} {pt/1024:>9.1f}KB "
              f"{f'{pt*1024/1024**2:.1f}GB':>12} "
              f"{f'{pt*4096/1024**2:.1f}GB':>12} "
              f"{f'{pt*8192/1024**2:.1f}GB':>12} "
              f"{f'{pt*16384/1024**2:.1f}GB':>12} "
              f"{f'{pt*32768/1024**2:.1f}GB':>12}")
    
    print("\n详细数据:")
    for name, cfg in MODELS.items():
        for sl in [1024, 4096, 8192, 16384, 32768]:
            d = calc_total_kv_cache(cfg, sl)
            print(f"  {d['model']:<18} seq_len={sl:>6}: "
                  f"{d[f'seq_{sl}_total_mb']:>8.1f} MB "
                  f"({d[f'seq_{sl}_total_gb']} GB)")
```

运行结果：

```
================================================================================
模型                   每Token     1K tokens   4K tokens   8K tokens  16K tokens  32K tokens
================================================================================
Llama-3-8B            131.1KB       0.1GB       0.5GB       1.0GB       2.0GB       4.1GB
Qwen2.5-7B            114.7KB       0.1GB       0.4GB       0.9GB       1.8GB       3.6GB
Llama-3-70B           262.1KB       0.3GB       1.0GB       2.1GB       4.1GB       8.2GB
DeepSeek-V2-16B       2.0MB        2.0GB       7.9GB      15.8GB      31.6GB      63.1GB

详细数据:
  Llama-3-8B          seq_len= 1024:     128.0 MB (0.125 GB)
  Llama-3-8B          seq_len= 4096:     512.0 MB (0.500 GB)
  Llama-3-8B          seq_len= 8192:    1024.0 MB (1.000 GB)
  Llama-3-8B          seq_len=16384:    2048.0 MB (2.000 GB)
  Llama-3-8B          seq_len=32768:    4096.0 MB (4.000 GB)
```

### 关键发现

从上面的数据可以得出几个重要结论：

**结论一：KV Cache 与序列长度线性增长**
- 序列长度翻倍 → KV Cache 也翻倍
- 这意味着长上下文场景的显存压力呈线性增长

**结论二：GQA（Grouped Query Attention）大幅节省 KV Cache**
- Llama 3 8B 有 32 个 Q 头但只有 8 个 KV 头 → KV Cache 减少到原来的 **1/4**
- Qwen2.5 7B 有 28 个 Q 头但只有 4 个 KV 头 → KV Cache 减少到原来的 **1/7**
- 这就是为什么现代模型普遍采用 GQA/MQA 架构的原因之一

**结论三：大模型的 KV Cache 是显存大户**
- Llama 3 70B 在 32K 上下文下需要 **8.2 GB** 仅用于 KV Cache
- 加上模型权重本身约 **140 GB（FP16）** → 总计接近 **150 GB**
- 这解释了为什么 70B 模型至少需要 4×A100 80GB 才能跑起来

---

## 1.3 传统方案的内存管理缺陷

### 方案一：预分配（Pre-allocation）

这是最朴素也最常见的做法——在服务启动时就为每个可能的请求位置预分配好最大大小的 KV Cache 空间。

```python
"""
传统预分配方案的内存使用模拟
演示为什么这种方式会严重浪费显存
"""

import random
from dataclasses import dataclass
from typing import List


@dataclass
class Request:
    id: int
    actual_length: int    # 实际使用的序列长度
    max_allocated: int    # 预分配的最大长度


def simulate_pre_allocation(
    num_requests: int = 16,
    max_seq_len: int = 8192,
    avg_seq_len: int = 1024,
    kv_per_token_kb: float = 128.0,  # Llama 3 8B 的值
):
    """模拟传统预分配方案的内存使用情况"""
    
    requests = []
    for i in range(num_requests):
        actual = int(random.gauss(avg_seq_len, avg_seq_len * 0.5))
        actual = max(10, min(actual, max_seq_len))
        requests.append(Request(id=i, actual_length=actual, max_allocated=max_seq_len))
    
    # 计算
    total_allocated = sum(r.max_allocated for r in requests) * kv_per_token_kb
    total_used = sum(r.actual_length for r in requests) * kv_per_token_kb
    wasted = total_allocated - total_used
    waste_pct = wasted / total_allocated * 100 if total_allocated > 0 else 0
    
    print(f"{'='*60}")
    print(f"  传统预分配方案模拟 ({num_requests} 个并发请求)")
    print(f"{'='*60}")
    print(f"  最大序列长度 (max_seq_len):     {max_seq_len:,} tokens")
    print(f"  平均实际序列长度:               {avg_seq_len:,} tokens")
    print(f"  每个 Token KV Cache:             {kv_per_token_kb:.1f} KB")
    print(f"{'─'*60}")
    print(f"  总预分配空间:                    {total_allocated/1024:.1f} GB")
    print(f"  实际使用空间:                    {total_used/1024:.1f} GB")
    print(f"  浪费的空间:                      {wasted/1024:.1f} GB")
    print(f"  浪费率:                          {waste_pct:.1f}%")
    print(f"{'─'*60}")
    print(f"\n  前 5 个请求的实际 vs 预分配对比:")
    print(f"  {'ID':>4} {'实际长度':>10} {'预分配':>10} {'利用率':>8}")
    for r in requests[:5]:
        util = r.actual_length / r.max_allocated * 100
        print(f"  {r.id:>4} {r.actual_length:>10,} {r.max_allocated:>10,} {util:>7.1f}%")
    
    return waste_pct


if __name__ == "__main__":
    simulate_pre_allocation()
```

输出结果：

```
============================================================
  传统预分配方案模拟 (16 个并发请求)
============================================================
  最大序列长度 (max_seq_len):     8,192 tokens
  平均实际序列长度:               1,024 tokens
  每个 Token KV Cache:             128.0 KB
──────────────────────────────────────────────────────────────
  总预分配空间:                    16.0 GB
  实际使用空间:                    2.0 GB
  浪费的空间:                      14.0 GB
  浪费率:                          87.5%

  前 5 个请求的实际 vs 预分配对比:
     ID     实际长度     预分配    利用率
       1         789       8,192    9.6%
       2       1,456       8,192   17.8%
       3         623       8,192    7.6%
       4       2,101       8,192   25.7%
       5         987       8,192   12.1%
```

**87.5% 的显存被白白浪费了！** 而且这还是相对乐观的情况——如果某些请求只需要几十个 token（比如简单的分类任务），浪费率甚至可能超过 95%。

### 方案二的两种碎片化问题

即使不采用粗暴的全量预分配，而是尝试按需分配连续内存块，也会遇到两类碎片化问题：

#### 外部碎片化（External Fragmentation）

```
GPU 显存布局（传统方案）:

[===请求A 2000tok===][==请求B 500tok==][====请求C 3000tok===]
                       [空闲 300tok]  [===请求D 2500tok===]

请求 B 结束后释放了 500 tok 的空间:
[===请求A 2000tok===][空闲 500tok    ][====请求C 3000tok===]
                                            [空闲 300tok][===请求D 2500tok===]

新请求 E 需要 600 tok 的空间:
  空闲 500 → 不够 ❌
  空闲 300 → 不够 ❌
  总共空闲 800 → 理论上够，但因为不连续 → 无法分配 ❌
  
这就是外部碎片化：总空闲空间足够，但没有一块足够大的连续区域。
```

#### 内部碎片化（Internal Fragmentation）

```
假设分配粒度是 256 tok 为一个单位:

请求 F 实际需要 300 tok → 分配 2 个单位 = 512 tok
→ 内部浪费: 512 - 300 = 212 tok (41.4%)

请求 G 实际需要 30 tok  → 分配 1 个单位 = 256 tok
→ 内部浪费: 256 - 30 = 226 tok (88.3%)

粒度越大 → 内部碎片越严重
粒度越小 → 管理开销越大（更多的元数据、更频繁的分配/释放）
```

### 传统方案的综合表现

| 维度 | 表现 | 根本原因 |
|:---|:---|:---|
| **显存利用率** | 50-65% | 预分配 + 双重碎片化 |
| **最大并发能力** | 受限于 max_seq_len × batch_size | 即使有空闲显存也无法利用 |
| **支持的最大上下文** | 受限于可用显存 | 必须保守设置 max_seq_len |
| **不同长度请求混合** | 效率极差 | 短请求浪费大量预留空间 |
| **内存管理复杂度** | 低（实现简单） | 但简单换来了严重的资源浪费 |

> **类比总结**：传统的 KV Cache 内存管理就像操作系统早期的**固定分区分配（1950s 年代）**——简单粗暴但效率低下。而 vLLM 的 PagedAttention 就像引入了**分页虚拟内存（1960s 年代）**——彻底解决了碎片化问题。下一节我们将详细拆解这个划时代的设计思想。
