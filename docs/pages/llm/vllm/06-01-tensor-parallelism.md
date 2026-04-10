# 张量并行（Tensor Parallelism）

> **白板时间**：想象你需要搬运 1000 块砖头。一个人搬要很久，但如果叫来 3 个人一起搬——每人负责一部分，速度就能快 3 倍（理想情况下）。张量并行（TP）就是这个思想在 GPU 推理中的应用：**把一个太大的模型"切"成几块，每块 GPU 只存一部分权重、算一部分矩阵乘法，最后通过通信把结果拼起来**。这是让单机跑 70B+ 大模型的唯一可行方案。

## 一、为什么需要张量并行？

### 1.1 显存墙问题

先看一组数据：

| 模型 | 参数量 | FP16 权重大小 | KV Cache (8K ctx) | **总显存需求** |
|------|--------|-------------|-------------------|---------------|
| Qwen2.5-0.5B | 0.5B | ~1 GB | ~0.5 GB | **~2 GB** |
| Qwen2.5-7B | 7B | ~14 GB | ~6 GB | **~22 GB** |
| Qwen2.5-14B | 14B | ~28 GB | ~10 GB | **~40 GB** |
| Qwen2.5-32B | 32B | ~64 GB | ~20 GB | **~86 GB** |
| Qwen2.5-72B | 72B | ~144 GB | ~42 GB | **~190 GB** |

再看主流 GPU 的显存：

| GPU | 显存 | 能跑多大模型（FP16）？ |
|-----|------|----------------------|
| RTX 4090 | 24 GB | ≤ 7B（勉强） |
| RTX 6000 Ada | 48 GB | ≤ 14B |
| A100 80GB | 80 GB | ≤ 34B |
| H100 80GB | 80 GB | ≤ 34B |
| H200 141GB | 141 GB | ≤ 72B |

**结论**：单卡跑 70B 模型？至少需要 H200 或 2×A100。这就是 TP 存在的意义。

### 1.2 TP 的核心思想

```
┌─────────────────────────────────────────────────────┐
│                一个 Linear 层: Y = XA + b            │
│                                                     │
│  X: [batch, in_features]    权重 A: [in, out]      │
│                                                     │
│  ┌────────── Tensor Parallelism ──────────┐        │
│  │                                         │        │
│  │  GPU 0:   Y_0 = X @ A[:, :out/2]       │        │
│  │            ↑                            │        │
│  │  GPU 1:   Y_1 = X @ A[:, out/2:]       │        │
│  │            ↑                            │        │
│  │  AllReduce(Y_0 + Y_1) → 完整的 Y       │        │
│  │                                         │        │
│  └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

**关键点**：
- 每个 GPU 只存储 **1/TP_size** 的权重 → 显存需求降低 N 倍
- 每个 GPU 独立做自己的矩阵乘法 → 计算量降低 N 倍
- 但需要 **AllReduce 通信** 来合并部分结果 → 这是额外开销

## 二、TP 原理深度解析

### 2.1 Megatron-LM 式并行策略

vLLM 采用 Megatron-LM 的张量并行方案，包含两种基本模式：

#### Column-Parallel Linear（列并行）

```
输入 X 被复制（广播）到所有 GPU：
  
  GPU 0: Y₀ = X · A₀     其中 A₀ = A[:, 0:out/2]
  GPU 1: Y₁ = X · A₁     其中 A₁ = A[:, out/2:out]

输出 Y = [Y₀ || Y₁] （拼接，无需通信）
```

**特点**：输出被切分，不需要通信。

#### Row-Parallel Linear（行并行）

```
输入 X 被切分到各 GPU：

  GPU 0: Z₀ = X₀ · B₀    其中 X₀ = X[:, 0:in/2], B₀ = B[0:in/2, :]
  GPU 1: Z₁ = X₁ · B₁    其中 X₁ = X[:, in/2:], B₁ = B[in/2:, :]

输出 Z = AllReduce(Z₀ + Z₁)  （需要通信！）
```

**特点**：输入被切分，需要 AllReduce 通信求和。

### 2.2 Transformer 层中的 TP 切分

一个完整的 Transformer 层如何切分？

```python
# 伪代码：Transformer 一层的 TP 切分方式
def transformer_layer_tp(x, tp_rank, tp_world_size):
    
    # 1. Attention - QKV 投影 (Column-Parallel)
    # 权重形状: [3 * hidden, hidden] → 每个GPU拿 [3*hidden/tp, hidden]
    qkv = column_parallel_linear(x, weight_qkv)  
    
    # 2. Attention Head 分割
    # num_heads 分配到各 GPU
    q, k, v = split_qkv(qkv, tp_rank, tp_world_size)
    
    # 3. Attention 计算（各 GPU 独立计算自己负责的 head）
    attn_out = multi_head_attention(q, k, v)
    
    # 4. Attention 输出投影 (Row-Parallel)
    # 需要 AllReduce
    out = row_parallel_linear(attn_out, weight_o)  # ← AllReduce here
    
    # 5. FFN 第一层 (Column-Parallel)
    hidden = column_parallel_linear(x, weight_gate)  # gate proj
    hidden = activation(hidden)
    
    # 6. FFN 第二层 (Row-Parallel)
    # 需要 AllReduce
    output = row_parallel_linear(hidden, weight_up)   # ← AllReduce here
    
    return output
```

**通信模式总结**：

```
每个 Transformer Layer 需要 2 次 AllReduce:
  1. Attention 输出投影后 (Row-Parallel)
  2. FFN 第二层后 (Row-Parallel)

对于 L 层模型: 总共 2L 次 AllReduce / iteration
```

## 三、vLLM 中的 TP 配置与使用

### 3.1 基本用法

```bash
# 单卡（默认，TP=1）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# 双卡 TP
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000

# 四卡 TP
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000

# 八卡 TP（通常用于 100B+ 模型）
python -m vllm.entrypoints.openai.api_server \
    --model DeepSeek-V2-16B-Chat \
    --tensor-parallel-size 8 \
    --port 8000
```

### 3.2 Python 离线推理中的 TP

```python
from vllm import LLM, SamplingParams

def tp_demo():
    """张量并行离线推理"""
    
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        dtype="bfloat16",
    )
    
    outputs = llm.generate(
        ["介绍张量并行的原理"],
        SamplingParams(max_tokens=128, temperature=0.3)
    )
    
    print(outputs[0].outputs[0].text)

tp_demo()
```

**注意**：Python API 中设置 `tensor_parallel_size > 1` 时，vLLM 会自动使用所有可见 GPU。如果只想用部分 GPU，需要设置 `CUDA_VISIBLE_DEVICES`。

### 3.3 GPU 选择控制

```python
import os

# 方式1：环境变量（推荐）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 只用 GPU 0 和 1

llm = LLM(
    model="Qwen/Qwen2.5-32B-Instruct",
    tensor_parallel_size=2,  # 必须等于可见 GPU 数量
)

# 方式2：指定 GPU ID（某些版本支持）
# llm = LLM(model="...", tensor_parallel_size=2, gpu_memory_utilization=0.9)
```

## 四、通信开销分析

### 4.1 AllReduce 带宽需求

每次 AllReduce 需要传输的数据量：

$$\text{Data per AllReduce} = \text{batch} \times \text{seq\_len} \times \text{hidden\_size} \times 2 \text{ bytes}$$

以 Llama 3 8B 为例（hidden_size=4096）：

| 场景 | batch × seq_len | 数据量/次 | TP=2 带宽 | TP=4 带宽 |
|------|----------------|----------|-----------|----------|
| 小批量短文本 | 16 × 512 | 64 MB | 64 MB/s | 128 MB/s |
| 中批量中长 | 64 × 2048 | 1 GB | 1 GB/s | 2 GB/s |
| 大批量长文本 | 128 × 8192 | 8 GB | 8 GB/s | 16 GB/s |

### 4.2 不同互联技术的带宽对比

| 互联技术 | 带宽 (单向) | 延迟 | 适用场景 |
|---------|------------|------|---------|
| PCIe Gen4 x16 | ~32 GB/s | ~2 μs | 单机多卡（消费级） |
| PCIe Gen5 x16 | ~64 GB/s | ~1 μs | 单机多卡（新一代） |
| NVLink (SXM) | ~900 GB/s | ~0.05 μs | A100/H100/H200 同节点 |
| InfiniBand HDR | ~200 GB/s | ~1 μs | 跨节点 |
| 以太网 100GbE | ~12.5 GB/s | ~50 μs | 跨节点（低成本） |
| 以太网 400GbE | ~50 GB/s | ~15 μs | 跨节点（高性能） |

### 4.3 扩展效率（Scaling Efficiency）

扩展效率 = (TP=N 的实际吞吐) / (TP=1 的理论吞吐 × N)

```
理想的扩展效率曲线:
  Throughput
     ↑
  4x ├──────────● (TP=4, 效率 100%)
  3x │         ╱
  2x │     ●──╱  (TP=2, 效率 95%)
  1x ●─╱
  0 └──────────→ TP Size
     1   2   4   8

实际情况（有通信开销）:
  Throughput
     ↑
  4x ├──────●    (TP=4, 效率 ~85%)
  3x │    ╱ ╲
  2x │  ●   ╲   (TP=2, 效率 ~92%)
  1x ●╱
  0 └────────→ TP Size
```

**典型扩展效率参考值**：

| 配置 | NVLink | PCIe Gen4 | PCIe Gen5 |
|------|--------|-----------|-----------|
| TP=2 | ~96% | ~92% | ~94% |
| TP=4 | ~92% | ~82% | ~88% |
| TP=8 | ~85% | ~65% | ~78% |

## 五、TP 选型指南

### 5.1 模型-GPU 匹配表

```python
def tp_recommendation():
    """TP 大小选型建议"""
    
    recommendations = [
        {
            "model_range": "≤ 7B",
            "recommended_tp": 1,
            "min_gpu": "RTX 4090 (24GB) / A5000 (48GB)",
            "note": "单卡足够，无需 TP"
        },
        {
            "model_range": "8B - 14B",
            "recommended_tp": [1, 2],
            "min_gpu": "RTX 6000 Ada (48GB) / A100 40GB",
            "note": "FP16 需双卡或大显存；INT4 可单卡"
        },
        {
            "model_range": "20B - 35B",
            "recommended_tp": 2,
            "min_gpu": "2×A100 80GB / 2×RTX 6000 Ada 48GB",
            "note": "TP=2 是标准配置"
        },
        {
            "model_range": "40B - 80B",
            "recommended_tp": [2, 4],
            "min_gpu": "4×A100 80GB / 2×H200 141GB",
            "note": "量化后可 TP=2；FP16 通常需 TP=4"
        },
        {
            "model_range": "100B+",
            "recommended_tp": [4, 8],
            "min_gpu": "8×H100 80GB / 8×H200 141GB",
            "note": "需要高带宽互联 (NVLink)"
        },
    ]
    
    print(f"{'模型范围':<12} | {'推荐TP':<10} | {'最低硬件':<30} | {'备注'}")
    print("-" * 90)
    for rec in recommendations:
        tp_str = str(rec['recommended_tp'])
        print(f"{rec['model_range']:<12} | {tp_str:<10} | {rec['min_gpu']:<30} | {rec['note']}")

tp_recommendation()
```

### 5.2 决策流程图

```
你的模型有多大？
│
├─ ≤ 7B
│   └─ 有 ≥24GB 显存的 GPU？
│       ├─ 是 → TP=1 ✅
│       └─ 否 → 换更大 GPU 或使用量化 (AWQ/GPTQ INT4)
│
├─ 8B - 35B
│   └─ 有几张 GPU？
│       ├─ 1 张 (≥48GB) → TP=1 ✅（或 INT4 量化）
│       ├─ 2 张 (≥40GB each) → TP=2 ✅
│       └─ 更多 → TP=2 即可（更多卡不会更快）
│
├─ 40B - 80B
│   ├─ 2 张 A100 80GB → TP=2 + AWQ INT4 ✅
│   ├─ 4 张 A100 80GB → TP=4 FP16 ✅
│   └─ 2 张 H200 141GB → TP=2 FP16 ✅（最佳性价比）
│
└─ 100B+
    ├─ 4 张 H100 80GB + NVLink → TP=4 ✅
    ├─ 8 张 H100 80GB + NVLink → TP=8 ✅
    └─ 无 NVLink → 考虑 PP 或多节点部署
```

## 六、常见问题排查

### 6.1 TP 相关错误

```bash
# 错误1：GPU 数量不匹配
# RuntimeError: Expected 2 GPUs but only 1 available
# 解决：确保 CUDA_VISIBLE_DEVICES 包含足够的 GPU

# 错误2：NCCL 通信错误
# NCCL error: unhandled system error
# 解决：
#   export NCCL_DEBUG=INFO
#   检查防火墙和驱动版本一致性

# 错误3：显存分配不均
# 各 GPU 显存占用差异很大
# 解决：确保所有 GPU 型号相同、驱动一致

# 错误4：初始化超时
# Timeout waiting for RPC response from rank 1
# 解决：检查 GPU 间连接状态（nvidia-smi topo -m）
```

### 6.2 TP 健康检查脚本

比如下面的程序可以验证 TP 配置是否正确：

```python
import torch
import subprocess
import json

def tp_health_check():
    """TP 配置健康检查"""
    
    print("=" * 60)
    print("vLLM 张量并行健康检查")
    print("=" * 60)
    
    # 1. 检查 CUDA 可用性
    print(f"\n[1] CUDA 环境")
    print(f"  PyTorch CUDA 可用: {torch.cuda.is_available()}")
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  GPU 数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_mem / 1024**3
        print(f"  GPU {i}: {props.name} ({mem_gb:.0f}GB)")
    
    # 2. 检查 NCCL
    print(f"\n[2] NCCL 通信库")
    try:
        nccl_version = torch.cuda.nccl.version()
        print(f"  NCCL 版本: {nccl_version}")
    except:
        print(f"  ⚠️ 无法获取 NCCL 版本")
    
    # 3. 检查 GPU 拓扑
    print(f"\n[3] GPU 互联拓扑")
    result = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout[:500])
    else:
        print("  ⚠️ nvidia-smi topo 不可用")
    
    # 4. 检查 P2P 连通性
    if torch.cuda.device_count() >= 2:
        print(f"\n[4] P2P (Peer-to-Peer) 访问测试")
        can_p2p = torch.cuda.can_device_access_peer(0, 1)
        print(f"  GPU0 ↔ GPU1 P2P: {'✅' if can_p2p else '❌'}")
        
        if can_p2p:
            try:
                torch.cuda.set_device(0)
                a = torch.randn(100, device='cuda:0')
                torch.cuda.set_device(1)
                b = a.to('cuda:1')
                print(f"  数据传输测试: ✅ 成功")
            except Exception as e:
                print(f"  数据传输测试: ❌ {e}")
    
    # 5. TP 推荐配置
    n_gpus = torch.cuda.device_count()
    total_mem = sum(
        torch.cuda.get_device_properties(i).total_mem 
        for i in range(n_gpus)
    ) / 1024**3
    
    print(f"\n[5] 推荐 TP 配置")
    print(f"  总可用显存: {total_mem:.0f} GB ({n_gpus} GPU)")
    
    if total_mem >= 140:
        print(f"  → 可运行: 70B 模型 (TP={n_gpus}, FP16)")
    elif total_mem >= 70:
        print(f"  → 可运行: 32B 模型 (TP={n_gpus}, FP16)")
        print(f"  → 可运行: 70B 模型 (TP={n_gpus}, INT4/AWQ)")
    elif total_mem >= 30:
        print(f"  → 可运行: 7B-14B 模型 (TP=1 or TP={min(n_gpus,2)})")
    else:
        print(f"  → 建议: 使用更小的模型或量化")

tp_health_check()
```

---

## 七、总结

本节系统学习了张量并行的原理与实践：

| 主题 | 核心要点 |
|------|---------|
| **动机** | 单卡显存有限，70B 模型 FP16 需要 144GB+ 显存 |
| **核心原理** | Column-Parallel（切输出）+ Row-Parallel（切输入）+ AllReduce 通信 |
| **Megatron-LM** | Attention 层和 FFN 层交替使用 Col/Row 并行，每层 2 次 AllReduce |
| **通信开销** | ∝ hidden_size × batch × seq_len；NVLink >> PCIe |
| **扩展效率** | TP=2 约 92%，TP=4 约 82%（NVLink）；PCIE 下更低 |
| **选型原则** | 7B 以下单卡，8-35B 双卡 TP=2，40-80B 四卡 TP=4，100B+ 八卡 TP=8 |
| **关键参数** | `--tensor-parallel-size` 或 `--tp`（简写） |
| **健康检查** | NCCL 版本、P2P 连通性、GPU 拓扑、驱动一致性 |

**核心要点回顾**：

1. **TP 本质是"以通信换空间和时间"**——牺牲一些通信带宽换取能运行更大模型的能力
2. **NVLink 是 TP 的倍增器**——同样的 TP=4，NVLink 下效率 92% vs PCIe 下只有 82%
3. **不是 TP 越大越好**——TP=8 在无 NVLink 时效率可能低于 70%，反而比 TP=4 更慢
4. **TP 与量化是正交的**——可以先量化再 TP，实现 4 卡跑 70B 模型
5. **同型号 GPU + 同驱动版本**是 TP 正常工作的前提条件

下一节我们将学习 **流水线并行（Pipeline Parallelism）**，以及它与 TP 如何组合使用。
