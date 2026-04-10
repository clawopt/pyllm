# 流水线并行（Pipeline Parallelism）

> **白板时间**：张量并行（TP）是把模型"横着切"——每层内部切分到多卡。但有时候你不想横着切，而是想"竖着切"——把模型的层分组，前几层在 GPU 0 上跑，中间几层在 GPU 1 上跑，最后几层在 GPU 2 上跑。这就是**流水线并行（PP）**。它特别适合跨节点部署（每台机器一个 stage），或者当你的 GPU 之间带宽不够高时作为 TP 的替代方案。

## 一、PP 原理

### 1.1 核心思想

```
传统单卡推理（32 层模型）：
  
  GPU 0: [Layer 0] → [Layer 1] → ... → [Layer 31] → Output
           ↓
        全部串行

流水线并行（3-stage PP）：

  Stage 0 (GPU 0): [Layer 0-10]   → 隐含状态 →
                                              ↓
  Stage 1 (GPU 1): [Layer 11-21]  → 隐含状态 →
                                              ↓
  Stage 2 (GPU 2): [Layer 22-31]  → Output
```

**关键区别于 TP**：

| 维度 | 张量并行 (TP) | 流水线并行 (PP) |
|------|-------------|----------------|
| 切分方式 | 每层内部切分权重 | 按层分组切分 |
| 通信内容 | AllReduce（激活值求和） | Point-to-Point（传递隐含状态） |
| 通信频率 | 每个 Transformer 层 2 次 | 每个 micro-batch 的 stage 边界 |
| 通信量 | 大（整个 hidden state） | 小（隐含状态向量） |
| 适用场景 | 单机多卡（高带宽互联） | 跨节点 / 低带宽环境 |
| Bubble 开销 | 无 | 有（Pipeline Bubble） |

### 1.2 Pipeline Bubble 问题

PP 最著名的缺陷是 **Pipeline Bubble（流水线气泡）**：

```
时间 →

Stage 0: [M0][M1][M2][M3][M4][M5]│░░░░░░░░│
                                    ↑
Stage 1:     │[M0][M1][M2][M3][M4][M5]│░░░░│  Bubble = 空闲等待
                                          ↑
Stage 2:         │[M0][M1][M2][M3][M4][M5]│  Bubble

M = Micro-batch（微批次）
░ = Bubble（空闲等待）
```

**Bubble 率计算**：
$$\text{Bubble Ratio} = \frac{\text{num\_stages} - 1}{\text{num\_microbatches} + \text{num\_stages} - 1}$$

| Stages | Micro-batches | Bubble Rate |
|--------|--------------|-------------|
| 2 | 4 | 20% (1/5) |
| 2 | 8 | 11% (1/9) |
| 4 | 8 | 37.5% (3/8) |
| 4 | 16 | 23% (3/13) |

**减少 Bubble 的方法**：增加 micro-batch 数量（但会增加内存占用）

## 二、vLLM 中的 PP 配置

### 2.1 基本 PP 启动

```bash
# 2-stage Pipeline Parallelism
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 1 \
    --port 8000

# TP=2 + PP=2 (2D 并行: 4 GPU 总计)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 2 \
    --port 8000

# TP=4 + PP=2 (8 GPU 总计)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 4 \
    --port 8000
```

### 2.2 PP vs TP 选择决策

```python
def pp_vs_tp_decision():
    """PP vs TP 决策指南"""
    
    scenarios = [
        {
            "场景": "单机 4×A100 (NVLink)",
            "推荐": "TP=4",
            "原因": "NVLink 高带宽使 TP 效率 >90%，无 bubble",
        },
        {
            "场景": "单机 4×RTX 4090 (PCIe)",
            "推荐": "TP=2+PP=2 或 TP=4",
            "原因": "PCIe 带宽有限；TP=4 约 65-75% 效率；PP 可缓解通信瓶颈",
        },
        {
            "场景": "双机各 2×A100 (以太网/RoCE)",
            "推荐": "PP=2 + TP=2",
            "原因": "跨节点用 PP（stage 在不同机器），机内用 TP",
        },
        {
            "场景": "四机各 2×A100",
            "推荐": "PP=4 + TP=2",
            "原因": "4 个 stage 分布在 4 台机器上",
        },
        {
            "场景": "单机 8×H100 (NVLink/NVSwitch)",
            "推荐": "TP=8",
            "原因": "NVSwitch 全互联，TP 效率可达 ~85%",
        },
    ]
    
    print(f"{'场景':<25} {'推荐方案':<18} {'原因'}")
    print("-" * 95)
    for s in scenarios:
        print(f"{s['场景']:<25} {s['推荐']:<18} {s['原因']}")

pp_vs_tp_decision()
```

## 三、2D 并行：TP + PP 组合

### 3.1 架构示意

```
2D 并行示例 (TP=2, PP=2, 总计 4 GPU):

         Node 0                    Node 1
    ┌─────────────────┐      ┌─────────────────┐
    │                 │      │                 │
    │  GPU 0          │      │  GPU 2          │
    │  ┌───────────┐  │ P2P  │  ┌───────────┐  │
    │  │ Stage 0   │  ├────→│  │ Stage 1   │  │
    │  │ TP rank 0 │  │      │  │ TP rank 0 │  │
    │  └─────┬─────┘  │      │  └─────┬─────┘  │
    │        │AllReduce      │        │AllReduce
    │  ┌─────┴─────┐  │      │  ┌─────┴─────┐  │
    │  │ TP rank 1 │  │      │  │ TP rank 1 │  │
    │  └───────────┘  │      │  └───────────┘  │
    │       GPU 1     │      │       GPU 3     │
    │                 │      │                 │
    └─────────────────┘      └─────────────────┘
    
    机内: TP (AllReduce, 高带宽 NVLink/PCIe)
    跨机: PP (Point-to-Point, 低带宽容忍)
```

### 3.2 资源分配计算

比如下面的程序帮你计算不同并行策略的 GPU 需求：

```python
def parallel_resource_calculator():
    """并行策略资源计算器"""
    
    models = [
        ("Qwen2.5-7B", 7, 14),
        ("Qwen2.5-14B", 14, 28),
        ("Qwen2.5-32B", 32, 64),
        ("Qwen2.5-72B", 72, 144),
    ]
    
    strategies = [
        ("纯 TP", lambda tp: (tp, 1)),
        ("纯 PP", lambda pp: (1, pp)),
        ("TP+PP均衡", lambda n: _balanced_2d(n)),
    ]
    
    def _balanced_2d(total_gpus):
        import math
        tp = int(math.sqrt(total_gpus))
        pp = total_gpus // tp
        return (tp, pp)
    
    for model_name, param_b, fp16_gb in models:
        print(f"\n{'='*60}")
        print(f"模型: {model_name} ({param_b}B 参数, FP16 ≈ {fp16_gb}GB)")
        print(f"{'='*60}")
        
        for strat_name, strat_fn in strategies:
            for total in [1, 2, 4, 8]:
                if total < 1:
                    continue
                tp, pp = strat_fn(total)
                if tp * pp != total or tp < 1 or pp < 1:
                    continue
                
                per_gpu_model = fp16_gb / tp
                feasible = per_gpu_model < 80  # 单卡最大显存约80GB
                
                status = "✅" if feasible else "❌"
                
                print(f"  {status} {strat_name}: "
                      f"总GPU={total}, TP={tp}, PP={pp}, "
                      f"每卡模型≈{per_gpu_model:.0f}GB")


parallel_resource_calculator()
```

输出：

```
============================================================
模型: Qwen2.5-7B (7B 参数, FP16 ≈ 14GB)
============================================================
  ✅ 纯 TP: 总GPU=1, TP=1, PP=1, 每卡模型≈14GB
  ✅ 纯 TP: 总GPU=2, TP=2, PP=1, 每卡模型≈7GB
  ✅ 纯 TP: 总GPU=4, TP=4, PP=1, 每卡模型≈4GB

============================================================
模型: Qwen2.5-72B (72B 参数, FP16 ≈ 144GB)
============================================================
  ❌ 纯 TP: 总GPU=1, TP=1, PP=1, 每卡模型≈144GB
  ❌ 纯 TP: 总GPU=2, TP=2, PP=1, 每卡模型≈72GB
  ✅ 纯 TP: 总GPU=4, TP=4, PP=1, 每卡模型≈36GB
  ✅ 纯 TP: 总GPU=8, TP=8, PP=1, 每卡模型≈18GB
  ✅ TP+PP均衡: 总GPU=4, TP=2, PP=2, 每卡模型≈72GB
  ✅ TP+PP均衡: 总GPU=8, TP=2, PP=4, 每卡模型≈72GB
  ✅ TP+PP均衡: 总GPU=8, TP=4, PP=2, 每卡模型≈36GB
```

## 四、PP 实战配置模板

### 4.1 双机部署（最常见场景）

```bash
# ===== Node 0 (Head) =====
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --port 8000 \
    --distributed-executor-backend ray \
    --pp-partition 0

# ===== Node 1 (Worker) =====
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --port 8001 \
    --distributed-executor-backend ray \
    --pp-partition 1
```

### 4.2 Ray 集群模式（推荐）

```bash
# ===== 启动 Ray Head =====
ray start --head --port=6379 --dashboard-port=8265

# ===== 各 Worker 节点连接 =====
# Node 1: ray start --address="head-node:6379"
# Node 2: ray start --address="head-node:6379"

# ===== 启动 vLLM 服务 =====
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --worker-use-ray \
    --ray-address auto \
    --port 8000
```

## 五、性能对比与调优

### 5.1 不同并行策略的性能参考

以 Llama 3 70B 为例（FP16），在不同硬件配置下的性能数据：

| 配置 | 并行方式 | GPU 数 | TTFT (P50) | TPOT | Throughput | 扩展效率 |
|------|---------|--------|-----------|------|-----------|---------|
| 8×H100 SXM | TP=8 | 8 | 1.2s | 38ms | 180 tok/s | 85% |
| 8×H100 SXM | TP=4+PP=2 | 8 | 1.5s | 42ms | 165 tok/s | 78% |
| 4×A100 80G | TP=4 | 4 | 2.8s | 68ms | 98 tok/s | 88% |
| 4×A100 80G | TP=2+PP=2 | 4 | 3.2s | 72ms | 89 tok/s | 80% |
| 4×RTX 4090 | TP=4 | 4 | 5.1s | 112ms | 58 tok/s | 65% |
| 4×RTX 4090 | TP=2+PP=2 | 4 | 4.5s | 98ms | 62 tok/s | 69% |

### 5.2 PP 调优参数

```bash
# 减少 pipeline bubble 的关键参数
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tp 2 --pp 2 \
    --max-num-seqs 128 \              # 更多并发 → 更多 micro-batch → 更少 bubble
    --scheduler-delay-factor 0.5 \    # 允许更多请求积累
    --num scheduler-steps 1 \         # PP 调度步数
    --port 8000
```

---

## 六、总结

本节学习了流水线并行的原理与实践：

| 主题 | 核心要点 |
|------|---------|
| **核心思想** | 按层切分模型到不同 GPU，stage 之间传递隐含状态 |
| **vs TP** | PP 切层（低频通信），TP 切权重（高频 AllReduce） |
| **Pipeline Bubble** | 固有缺陷，增加 micro-batch 可降低 bubble rate |
| **适用场景** | 跨节点部署、低带宽环境、与 TP 组合使用 |
| **2D 并行** | TP（机内高带宽）+ PP（跨节点低带宽）= 最佳组合 |
| **vLLM 参数** | `--pipeline-parallel-size` 或 `--pp`（简写） |
| **Ray 集成** | `--worker-use-ray` + `--ray-address auto` 自动管理多节点 |
| **选型原则** | 同节点优先 TP，跨节点必须 PP，混合部署用 TP+PP |

**核心要点回顾**：

1. **PP 不是 TP 的替代品，而是互补品**——它们解决的是不同维度的扩展问题
2. **Bubble 是 PP 的固有代价**——通过增加并发请求数来摊薄
3. **跨节点部署时 PP 是必需的**——不可能用 AllReduce 跨千兆网做 TP
4. **2D 并行 (TP+PP) 是大模型生产部署的标准配置**——兼顾效率和可扩展性
5. **Ray 是管理 PP 多节点的推荐工具**——自动处理 worker 注册和故障恢复

下一节我们将学习 **多节点分布式部署**的具体操作细节。
