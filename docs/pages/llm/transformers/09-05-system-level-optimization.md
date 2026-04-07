# 系统级优化：批处理、PagedAttention 与 FlashAttention

## 这一节讲什么？

前面几节我们学习了模型层面的优化（量化、蒸馏、剪枝）和推理引擎层面的加速（ONNX、TensorRT、vLLM）。但即使你用了最快的引擎和最优的模型，如果**系统层面**的配置不当，仍然无法获得最佳性能。

这一节我们将深入到推理系统的"毛细血管"级别：
1. **批处理策略**——从 Static 到 Continuous 的演进
2. **KV Cache 管理**——PagedAttention 如何解决内存碎片
3. **FlashAttention**——IO 感知的注意力计算加速
4. **内存规划与调度**——如何让每一字节显存都发挥价值

---

## 一、批处理策略演进

## 1.1 为什么批处理如此重要？

```python
def why_batching_matters():
    """解释批处理的重要性"""

    explanation = """
┌───────────────────────────────────────────────────────────────┐
│                   GPU 计算的本质                             │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  GPU 不是为单个请求设计的!                                  │
│                                                               │
│  一个 A100 GPU:                                             │
│    • 6912 个 CUDA cores (计算单元)                           │
│    • 可以同时执行数万个浮点运算                              │
│    • 但启动一次 kernel 需要固定开销 (~10μs)                │
│                                                               │
│  如果只处理一个请求:                                        │
│    • 大部分时间在等 kernel launch 和 memory transfer          │
│    • GPU 利用率可能只有 5~10%                                │
│    • 就像用卡车运一个包裹                                      │
│                                                               │
│  如果批量处理 32 个请求:                                     │
│    • 一次矩阵乘法可以同时处理所有请求                          │
│    • kernel launch 开销被分摊到 32 个请求                      │
│    • 内存带宽被充分利用 (顺序读取大块连续内存)                 │
│    • GPU 利用率可以提升到 70~90%                             │
│    • 就像用卡车满载运输                                       │
│                                                               │
│  核心公式:                                                   │
│    Throughput = batch_size / (latency_per_batch)               │
│    → 在延迟可接受的范围内最大化 batch_size                     │
└───────────────────────────────────────────────────────────────┘
"""
    print(explanation)

why_batching_matters()
```

## 1.2 三代批处理策略对比

```python
def batching_strategies_comparison():
    """三代批处理策略详细对比"""

    strategies = [
        {
            "名称": "Static Batching (静态批处理)",
            "原理": "收集固定数量的请求后统一处理",
            "流程": "[等待 N 个请求到达] → [打包成 batch] → [GPU 推理] → [返回全部结果] → [等待下一批]",
            "优点": "实现简单; 可预测的延迟",
            "缺点": "⚠️ 严重问题: 快请求被慢请求阻塞! (Head-of-Line Blocking)",
            "利用率": "~30-50% (长短请求混合时极低)",
        },
        {
            "名称": "Dynamic Batching (动态批处理)",
            "原理": "设置等待窗口, 窗口内到的请求组成 batch",
            "流程": "[等待 Δt 时间] → [窗口内到达的所有请求组成 batch] → [处理] → [重复]",
            "优点": "比 Static 更灵活; 减少空转等待",
            "缺点": "仍有空闲间隙; 窗口大小难调优",
            "利用率": "~50-70%",
        },
        {
            "名称": "Continuous Batching (连续批处理) ⭐",
            "原理": "任何请求完成立即移除, 新请求立即插入",
            "流程": "[持续运行] → [A 完成→ 移除 A, 插入新请求 E] → [B 完成→ 移除 B, 插入 F]",
            "优点": "✅ 无 Head-of-Line Blocking; ✅ GPU 始终满载; ✅ 吞吐量最大",
            "缺点": "实现复杂; 需要 PagedAttention 支持",
            "利用率": "~90%+ (vLLM/TGI 标配方案)",
        },
    ]

    print("=" * 80)
    print("三代批处理策略对比")
    print("=" * 80)
    for s in strategies:
        print(f"\n📦 {s['名称']}")
        for k, v in s.items():
            if k != "名称":
                print(f"   {k}: {v}")

batching_strategies_comparison()
```

## 1.3 Continuous Batching 图示

```python
def continuous_batching_visual():
    """Continuous Batching 工作原理图示"""

    visual = """
时间轴 ───────────────────────────────────────────────────────→

Static Batching:
  Batch 1: |████████████████████████████|░░░░░░░░░░░░░░░░░|
           ↑ A 还没完, B 必须等 A 完成!

Dynamic Batching:
  Window:  |███████████|     |█████████████████|     |███████|
           ↑ 等待窗口内收集请求

Continuous Batching (vLLM/TGI):
  Slot 1: |████████████████|                    |████████|
  Slot 2: |████████████████████████████|       |████████████|
  Slot 3: |████████|              |████████████████████████████|
  Slot 4:         |████████████████████|    |████████████████████|

  关键: 每个 slot 是独立的!
        A 完成后立即释放给新请求 E
        不需要等 B/C/D 完成!
"""
    print(visual)

continuous_batching_visual()
```

---

## 二、KV Cache 与 PagedAttention

## 2.1 KV Cache 回顾

在 Decoder-only 模型的自回归生成中，每生成一个新的 token 都需要用到之前所有 token 的 Key 和 Value。如果不缓存这些值，就需要重新计算——这就是 **KV Cache** 存在的原因。

```python
def kv_cache_memory_analysis():
    """KV Cache 显存占用分析"""

    models = [
        ("LLaMA-2-7B", 7e9, 4096, 32),
        ("LLaMA-2-13B", 13e9, 5120, 40),
        ("LLaMA-2-70B", 70e9, 8192, 64),
        ("Qwen-72B", 72e9, 8192, 64),
        ("Mixtral-8x7B", 47e9, 4096, 32),  # MoE 模型
    ]

    seq_lengths = [256, 1024, 2048, 4096, 8192, 16384]

    print("=" * 85)
    print("KV Cache 显存占用估算")
    print("=" * 85)
    
    header = f"{'模型':<18}"
    for sl in seq_lengths:
        header += f"seq={sl:<12}"
    print(f"\n{header}")
    print("-" * 95)

    for name, params, hidden, n_heads in models:
        row = f"{name:<18}"
        for sl in seq_lengths:
            # KV Cache = 2 × num_layers × num_heads × head_dim × seq_len × sizeof(fp16)
            # 近似: num_layers ≈ params / (hidden² × 3)
            num_layers_approx = int(params / (hidden ** 2 * 3))
            head_dim = hidden // n_heads
            
            kv_cache_gb = 2 * num_layers_approx * n_heads * head_dim * sl * 2 / (1024**3)
            
            if kv_cache_gb < 1:
                row += f"{kv_cache_gb*1024:.0f}MB<12"
            else:
                row += f"{kv_cache_gb:.1f}GB<12"
        print(row)

    print(f"\n💡 观察:")
    print(f"   • LLaMA-2-7B 在 seq_len=4096 时, KV Cache 约 {2*32*128*4096*2/1e9:.1f} GB")
    print(f"   • LLaMA-2-70B 在 seq_len=4096 时, KV Cache 可能超过 100 GB!")
    print(f"   • 这就是为什么长上下文场景下 PagedAttention 如此重要")

kv_cache_memory_analysis()
```

## 2.2 传统 KV Cache 的内存碎片问题

```python
def kv_cache_fragmentation():
    """KV Cache 内存碎片问题详解"""

    problem = """
传统 KV Cache 分配方式:

Request A (需要 200 tokens 的 KV Cache):
  ┌────────────────────────────────────────┐
  │████████████████████████████████████│ ← 连续分配 200 token 空间
  └────────────────────────────────────────┘

Request B (需要 150 tokens 的 KV Cache):
  ┌──────────────────────────────┐
  │████████████████████████████│ ← 连续分配 150 token 空间
  └──────────────────────────────┘

⚠️ 问题出现:
  Request A 生成了 50 个 token 后完成 (释放了 200 token 空间)
  Request C 新来, 需要 180 tokens 的 KV Cache
  
  但是! A 释放的空间可能不连续:
  ├─ 方案 1: 找一块 ≥ 180 的连续空间 → 可能找不到!
  ├─ 方案 2: 把 C 拆到两块不连续的空间 → 复杂且低效!
  └─ 方案 3: 拒绝 C 或强制 A/B/C 统一截断长度 → 浪费!

  结果: 显存中充满了无法利用的"碎片空间"
  实际可用率: 可能只有 50-60%
"""
    print(problem)

kv_cache_fragmentation()
```

## 2.3 PagedAttention 解决方案

```python
def pagedattention_solution():
    """PagedAttention 原理解释"""

    solution = """
PagedAttention (vLLM 核心创新):

核心思想: 借鉴操作系统的虚拟内存分页机制

┌───────────────────────────────────────────────────────────────┐
│                  物理内存 (Physical Memory)                 │
│                                                           │
│  Page 0 (16 tokens):  [████████████████████]              │
│  Page 1 (16 tokens):  [████████████████████]              │
│  Page 2 (16 tokens):  [████████████████████]              │
│  Page 3 (16 tokens):  [████████████████████]              │
│  ...                                                      │
│  Page N (16 tokens):  [████████████████████]              │
│                                                           │
│  所有 Page 大小相同 (如 16 tokens)                         │
│  由统一的 Block Manager 统一分配和管理                    │
├───────────────────────────────────────────────────────────────┤
│                  逻辑视图 (Logical View per Request)         │
│                                                           │
│  Request A 的 KV Cache:                                   │
│    Logical: [token_0 ... token_199] (200 tokens)             │
│    Physical: [Page_0, Page_1, ..., Page_12] (13 pages)      │
│                                                           │
│  Request B 的 KV Cache:                                   │
│    Logical: [token_0 ... token_149] (150 tokens)             │
│    Physical: [Page_20, Page_21, ..., Page_29] (10 pages)     │
│                                                           │
│  ✅ A 完成后释放 Page_0~Page_12                              │
│  ✅ 这些 Page 立即可以被 C 复用!                            │
│  ✅ C 不需要连续的物理内存!                                 │
│  ✅ 内存利用率: ~90%+                                       │
└───────────────────────────────────────────────────────────────┘

关键数据结构:
  • Block Table: 记录每个逻辑 page 对应的物理 page
  • Page Table: 类似 OS 的页表, 管理 page 映射
  • Copy-on-Write: 多个请求共享只读 page 时无需复制
"""
    print(solution)

pagedattention_solution()
```

---

## 三、FlashAttention：IO 感知的注意力加速

## 3.1 标准 Attention 的 IO 瓶颈

标准的 Scaled Dot-Product Attention 有严重的 **HBM（高带宽内存）读写瓶颈**：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

这个看似简单的公式在实际计算时涉及大量的中间矩阵存储：

```python
def attention_io_analysis():
    """标准 Attention 的 IO 分析"""

    analysis = """
标准 Attention 计算 (batch=B, heads=H, seq=N, dim=D):

Step 1: S = Q @ K^T          → 产生 N×N 矩阵 (每个 head)
        读: Q(B×N×D) + K(B×N×D) = 2BND
        写: S(B×N×N) = BN²

Step 2: P = softmax(S)       → 同样是 N×N 矩阵
        读+写: BN² (in-place softmax)

Step 3: O = P @ V           → 最终输出 N×D
        读: P(B×N×N) + V(B×N×D) = BN² + BND
        写: O(B×N×D) = BND

总计 HBM 读写量 (每个 head):
  ≈ 4BN² + 2BND

当 N 很大时 (如 N=4096):
  N² = 16M → HBM 带宽成为绝对瓶颈!
  
GPU 计算: ~10 TFLOPS (A100)
GPU HBM 带宽: ~2 TB/s
实际受限于带宽的时间 >> 受限于计算的时间!

💡 FlashAttention 的洞察:
  不需要一次性把整个 N×N 注意力矩阵存到 HBM!
  可以分块计算, 利用 SRAM (On-Chip Memory) 存储中间结果
  → 将 O(N²) 的 HBM 读写降低到 O(N²/M) (M = tile size)
"""
    print(analysis)

attention_io_analysis()
```

## 3.2 FlashAttention-2 算法概览

```python
def flash_attention_overview():
    """FlashAttention-2 算法概览"""

    overview = """
FlashAttention-2 核心思想:

将 Q/K/V 分成小块 (tiles), 只在 SRAM 中保留当前需要的块:

┌─────────────────────────────────────────────────────────────┐
│  Q (B, N, D)                                              │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┐              │
│  │Q₀   │Q₁   │Q₂   │Q₃   │Q₄   │...  │ (tile by columns)│
│  └──────┴──────┴──────┴──────┴──────┴──────┘              │
│                                                           │
│  K (B, N, D)                                              │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┐              │
│  │K₀   │K₁   │K₂   │K₃   │K₄   │...  │ (tile by rows)   │
│  └──────┴──────┴──────┴──────┴──────┴──────┘              │
│                                                           │
│  V (B, N, D)                                              │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┐              │
│  │V₀   │V₁   │V₂   │V₃   │V₄   │...  │ (tile by rows)    │
│  └──────┴──────┴──────┴──────┴──────┴──────┘              │
│                                                           │
│  计算 Loop (对每个 Q-tile):                                │
│  ┌──────────────────────────────────────────────────┐       │
│  │ 1. 加载当前 Q-tile 到 SRAM                        │       │
│  │ 2. 逐块加载 K/V tiles 到 SRAM                       │       │
│  │ 3. 在 SRAM 中计算:                               │       │
│  │    S_local = Q_tile @ K_tile^T / √d                 │       │
│  │    P_local = softmax(S_local)                       │       │
│  │    O_local = P_local @ V_tile                       │       │
│  │ 4. 将 O_local 写回 HBM (累加到输出位置)              │       │
│  └──────────────────────────────────────────────────┘       │
│                                                           │
│  关键优化 (FlashAttention-2 vs v1):                          │
│  • 并行处理 Q/K/V 的不同分块                              │
│  • 更好的 tiling 策略减少冗余 load/store                   │
│  • 支持 Attention with different Q/K lengths (varlen)   │
│  • 支持 GQA (Grouped Query Attention) - 用于 GPT/Qwen 等   │
│                                                           │
│  加速效果:                                                │
│  • 序列长度 512:  ~2x faster than standard attention       │
│  • 序列长度 4K:   ~4x faster (IO-bound 越明显)              │
│  • 显存节省:     ~5-10x less peak memory for attention      │
└─────────────────────────────────────────────────────────────┘
"""
    print(overview)

flash_attention_overview()
```

## 3.3 使用 FlashAttention

```bash
# 安装
pip install flash-attn --no-build-isolation

# PyTorch 原生支持 (PyTorch >= 2.2)
import torch.nn.functional as F

# 启用 FlashAttention (自动启用, 无需改代码!)
# 只需确保使用 AMP (自动精度) 或 bfloat16
with torch.amp.autocast(dtype=torch.bfloat16):
    attn_output = F.scaled_dot_product_attention(
        query, key, value,
        is_causal=True,  # 因果掩码
        scale=head_dim ** -0.5,
    )
```

---

## 四、系统级优化 Checklist

## 4.1 从模型到服务的完整优化链

```python
def full_optimization_pipeline():
    """完整的 LLM 推理服务优化流水线"""

    pipeline = """
╔════════════════════════════════════════════════════════════╗
║           LLM 推理服务完整优化链 (从零到生产)               ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Level 1: 模型优化                                          ║
║  ├─ ✅ 选择合适的模型规模 (不是越大越好)                   ║
║  ├─ ✅ 量化 (NF4/INT8/GPTQ/AWQ)                          ║
║  ├─ ✅ 蒸馏 (如果需要更小模型)                             ║
║  └─ ✅ 剪枝 (可选, 配合量化使用)                           ║
║                                                            ║
║  Level 2: 编译优化                                          ║
║  ├─ ✅ torch.compile() (PyTorch 2.0+)                       ║
║  ├─ ✅ ONNX 导出 + Graph Optimization                   ║
║  └─ ⚠️ TensorRT (NVIDIA GPU 专用, 收益最大但成本高)        ║
║                                                            ║
║  Level 3: 运行时优化                                        ║
║  ├─ ✅ FlashAttention-2 (必须开启!)                       ║
║  ├─ ✅ PagedAttention (vLLM/TGI 内置)                    ║
║  ├─ ✅ Continuous Batching (vLLM/TGI 内置)                 ║
║  ├─ ✅ FP16/BF16 推理 (不要用 FP32!)                       ║
║  └─ ✅ CUDA Graphs (减少 Python 开销)                      ║
║                                                            ║
║  Level 4: 服务层优化                                        ║
║  ├─ ✅ 动态批处理 + 请求调度                               ║
║  ├─ ✅ Prefix Caching (复用共享前缀的 KV Cache)           ║
║  ├─ ✅ 异步推理 + SSE Streaming                            ║
║  ├─ ✅ 负载均衡 (多实例/GPU 分流)                           ║
║  └─ ✅ 监控 & 自动扩缩容                                    ║
║                                                            ║
║  预期效果 (vs 原始 HF Transformers Eager Mode):              ║
║  • 延迟降低: 60-90%                                       ║
║  • 吞吐提升: 5-100x                                        ║
║  • 显存效率: 30% → 90%+                                   ║
║  • 成本降低: 单卡替代多卡集群                              ║
╚════════════════════════════════════════════════════════════╝
"""
    print(pipeline)

full_optimization_pipeline()
```

## 4.2 性能调优参数速查表

```python
def performance_tuning_params():
    """性能调优关键参数速查"""

    params = [
        ("max_model_len", "模型最大上下文长度", "根据需求设置, 越大占用越多显存"),
        ("max_num_seqs", "最大并发序列数", "根据 GPU 显存调整, 影响吞吐"),
        ("gpu_memory_utilization", "GPU 显存利用率", "0.9-0.95, 给系统留余量"),
        ("max_batch_size", "最大批处理大小", "vLLM 自动管理, 一般不需要手动设"),
        ("max_waiting_tokens", "连续批处理等待窗口", "5-20, 太小则利用率低, 太大则首延增加"),
        ("enable-flash-attn", "是否启用 FlashAttention", "✅ 必须开启! 默认开启"),
        ("enable-paged-attention", "是否启用 PagedAttention", "✅ 必须开启! 默认开启"),
        ("dtype", "推理精度", "float16 推荐, fp32 仅用于调试"),
        ("enforce-eager", "是否禁用 eager 模式", "False (使用 CUDA graphs)"),
        ("tensor_parallel_size", "张量并行度", "多卡时设置, 单卡不用管"),
        ("max_seq_len_to_capture", "捕获前缀的最大长度", "8192+, 越大增加内存但提高命中率"),
    ]

    print("=" * 75)
    print("vLLM/TGI 性能调优参数速查表")
    print("=" * 75)
    for param, desc, note in params:
        print(f"\n  {param:<35} {desc:<25}")
        print(f"  {'':>37}📌 {note}")

performance_tuning_params()
```

---

## 五、本章小结

这一节我们深入学习了推理系统的底层优化技术：

| 技术 | 解决的问题 | 核心机制 | 效果 |
|------|-----------|---------|------|
| **Continuous Batching** | 请求排队等待 | 即插即替换完成的请求 | 吞吐 2~5x |
| **PagedAttention** | KV Cache 内存碎片 | 操作系统式分页管理 | 显存利用率 90%+ |
| **FlashAttention-2** | Attention IO 瓶颈 | SRAM 中分块计算 | 延迟降 2~4x |
| **Prefix Caching** | 共享前缀重复计算 | 缓存并复用系统提示的 KV | 长对话加速 |
| **FP16/BF16 推理** | FP32 精度浪费 | 半精度计算 | 显存减半, 加速 2x |

**核心要点**：
1. **系统级优化 > 单点优化**——FlashAttention 再好，如果批处理策略糟糕也白搭
2. **vLLM / TGI 已经内置了大部分优化**——你只需要正确配置参数
3. **FlashAttention + PagedAttention + Continuous Batching 是现代 LLM 引擎的"三件套"**
4. **监控是最后一步也是最重要的一步**——时刻关注 GPU 利用率和 P99 延迟

至此，**第 9 章 模型压缩与推理优化**全部完成！最后一章，我们将进入 **第 10 章：多模态与前沿方向**。
