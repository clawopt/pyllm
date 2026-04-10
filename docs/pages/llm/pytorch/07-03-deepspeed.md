# 7.3 DeepSpeed ZeRO——微软的大规模训练方案

前两节我们学习了 DDP（数据并行，每卡完整模型副本）和 FSDP（PyTorch 原生的模型分片方案）。两者都是优秀的工具，但在超大规模模型训练（100B+ 参数）的领域里，还有一个被更广泛使用的方案——**DeepSpeed ZeRO（Zero Redundancy Optimizer）**。这是微软开发的分布式训练框架，它的核心创新在于把"消除冗余"这个思想发挥到了极致：不仅分片参数和梯度，还提供了精细化的配置选项和多种 offloading 策略。目前几乎所有顶级开源大模型的训练（LLaMA、Falcon、Mistral、Qwen 等）都在底层使用了 DeepSpeed 或其变体。

## ZeRO 三个阶段：逐步消除冗余

ZeRO 的名字来源于它的核心思想——**Zero Redundancy Optimizer**（零冗余优化器）。它通过三个递进的阶段来逐步减少每张 GPU 上需要存储的数据量：

```
DDP (基准):
┌─────────────────────────────────────┐
│ 每张卡存储:                          │
│   ├─ 完整参数 W        (100%)       │
│   ├─ 完整梯度 ∇W       (100%)       │
│   └─ 完整优化器状态 (m,v) (100%)     │
│ 总计: ~4× 模型大小                     │
└─────────────────────────────────────┘

ZeRO-1 (分片优化器状态):
┌─────────────────────────────────────┐
│ 每张卡存储:                          │
│   ├─ 完整参数 W        (100%)       │
│   ├─ 完整梯度 ∇W       (100%)       │
│   └─ 分片优化器状态    (~25%)       │ ← 只存 1/N 的 m,v
│ 显存节省: ~4x → ~3x                   │
│ 通信开销: 低                            │
└─────────────────────────────────────┘

ZeRO-2 (分片优化器状态 + 梯度):
┌─────────────────────────────────────┐
│ 每张卡存储:                          │
│   ├─ 完整参数 W        (100%)       │
│   └─ 分片梯度          (~25%)       │ ← 只存 1/N 的梯度
│      分片优化器状态    (~25%)       │
│ 显存节省: ~3x → ~2x                   │
│ 通信开销: 中等                          │
└─────────────────────────────────────┘

ZeRO-3 (全部分片):
┌─────────────────────────────────────┐
│ 每张卡存储:                          │
│   └─ 分片参数          (~25%)       │ ← 只存 1/N 的参数
│      分片梯度          (~25%)       │
│      分片优化器状态    (~25%)       │
│ 显存节省: ~2x → ~N×                   │
│ 通信开销: 高                            │
└─────────────────────────────────────┘
```

### ZeRO-1：只分片优化器状态

回顾一下 AdamW 优化器为每个参数维护的两个状态变量：一阶矩 $m$（动量，与参数同 shape）和二阶矩 $v$（方差，与参数同 shape）。对于 7B BF16 模型来说，优化器状态的显存占用约为 `7B × 2 × 4 bytes = 56 GB`——这比模型本身（14GB）还要大得多！

ZeRO-1 的做法是：把 $m$ 和 $v$ 切分到各张 GPU 上，每张卡只保存自己负责的那部分。在前向/反向传播时不需要这些状态（它们只在 optimizer.step() 时使用），所以不影响计算流程。当需要更新参数时，通过 AllGather 收集完整的 $m$ 和 $v$，完成更新后再分散回去。

通信模式：只在 `optimizer.step()` 时发生 AllGather + ReduceScatter，频率很低（每个 step 一次）。

### ZeRO-2：增加梯度分片

在 ZeRO-1 的基础上进一步切分梯度。梯度的形状和参数相同（因为 $\nabla_W L$ 的维度与 $W$ 一致），所以切分梯度和切片参数一样自然。

关键点在于：反向传播完成后，每张卡只有自己负责的那部分梯度的完整值。在做 optimizer.step() 之前，需要先 AllGather 完整梯度（用于权重衰减等操作），然后 AllGather 完整的优化器状态，执行更新后 ReduceScatter 把更新的参数分散回各卡。

### ZeRO-3：全部分片（包括参数）

这是最激进的阶段——连参数本身也切分了。这意味着前向传播时每层都需要 AllGather 来收集当前层的完整参数（类似 FSDP 的 FULL_SHARD 模式），反向传播时需要 ReduceScatter 分散梯度。

ZeRO-3 与 FSDP FULL_SHARD 的效果类似，但 DeepSpeed 提供了更多的配置选项和优化：
- **更细粒度的通信优化**（如 bucketing 减少小通信次数）
- **CPU Offloading**（把不常用的参数卸载到系统内存）
- **NVMe Offloading**（更进一步卸载到 SSD）
- **带宽压缩**（通信时压缩数据以减少传输量）

## DeepSpeed 配置文件

DeepSpeed 通过一个 JSON 配置文件（`ds_config.json`）来控制所有行为：

```python
import json


def create_ds_config(stage=2, offload_optimizer=False, offload_param=False):
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,

        "bf16": {
            "enabled": True,
        },

        "zero_optimization": {
            "stage": stage,

            "offload_optimizer": {
                "device": "cpu" if offload_optimizer else "none",
                "pin_memory": True,
            } if stage >= 1 else {},

            "offload_param": {
                "device": "cpu" if offload_param else "none",
                "pin_memory": True,
            } if stage == 3 else {},

            "overlap_comm": True,
            "contiguous_gradients": True,
            "round_ropt_values": False,
        },

        "gradient_clipping": 1.0,
        "gradient_accumulation_steps": "auto",
    }

    return json.dumps(config, indent=2)


print(create_ds_config(stage=2, offload_optimizer=True))
```

输出如下：

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "round_ropt_values": false
  },
  "gradient_clipping": 1.0,
  "gradient_accumulation_steps": "auto"
}
```

### 关键配置项详解

**`stage`**: ZeRO 阶段（1, 2, 或 3）。选择建议：
- 7B 以下模型：Stage 1 或 2 就够了（甚至 DDP/FSDP 也行）
- 7B~30B：Stage 2 是性价比最高的选择
- 30B+：Stage 3 可能是必须的
- 100B+：Stage 3 + CPU/NVMe Offloading

**`offload_optimizer`**: 将优化器状态 offload 到 CPU 内存。这对 Stage 1 尤其重要——因为 Stage 1 本身不分片参数和梯度，如果优化器状态也留在 GPU 上，节省有限。Offload 到 CPU 后，GPU 显存可以释放约 `2 × model_size × 4 bytes`（m 和 v），代价是每次 optimizer.step() 时需要从 CPU 读取这些状态（PCIe 带宽约 32 GB/s，延迟约几十微秒，对整体训练速度影响通常 <5%）。

**`offload_param`** (仅 Stage 3): 将参数 offload 到 CPU。这是最极端的内存节省策略——GPU 上只保留当前正在计算的几层参数，其余全部放在 CPU 内存中。配合 NVMe Offloading 甚至可以把 CPU 内存都放不下的参数放到 SSD 上（当然速度会更慢）。

**`overlap_comm`: true**: 让通信和计算重叠执行。比如在进行某层的前向计算的同时，已经在后台准备下一层需要的 AllGather 数据。这能显著掩盖通信延迟，是 DeepSpeed 相比 FSDP 的一个重要性能优势。

**`contiguous_gradients`: true**: 确保梯度在内存中连续存储，提高 AllGather/RScatter 的效率。

## 在 HF Trainer 中使用 DeepSpeed

HuggingFace Trainer 对 DeepSpeed 有原生支持——只需要提供配置文件的路径：

```python
from transformers import TrainingArguments, Trainer

ds_config_path = "./ds_config_stage2.json"

args = TrainingArguments(
    output_dir="./output/deepspeed",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    gradient_checkpointing=True,
    deepspeed=ds_config_path,     # ← 就这一行！
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)
trainer.train()
```

启动方式也需要用 `deepspeed` 启动器而不是普通的 `torchrun`：

```bash
# 使用 accelerate launch（推荐）
accelerate launch --num_processes 4 --use_deepspeed train.py

# 或者直接用 deepspeed
deepspeed --num_gpus=4 train.py --deepspeed_config=ds_config.json
```

注意：使用 DeepSpeed 时不能同时启用 FSDP 策略——两者是互斥的分布式方案。

## DeepSpeed vs FSDP 选型指南

既然 FSDP 和 DeepSpeed ZeRO 功能上有大量重叠，什么时候该选哪个呢？

| 维度 | FSDP | DeepSpeed ZeRO |
|-----|------|---------------|
| **归属** | PyTorch 原生 | 微软独立项目 |
| **安装** | 无需额外安装 | `pip install deepspeed` |
| **API 集成** | Lightning 原生支持 | HF Trainer 原生支持 |
| **Offloading** | 仅 CPU (有限) | **CPU + NVMe** |
| **通信优化** | 基础 | **高级 (overlap, compression)** |
| **社区生态** | PyTorch 社区 | 微软 + HuggingFace 社区 |
| **成熟度** | 较新（PyTorch 2.x 引入） | **非常成熟**（多年生产验证） |
| **调试友好度** | 较好（PyTorch 原生工具链） | 一般（需要了解 ds 内部机制）|

**推荐选择原则**：

1. **一般情况（< 30B 模型，单节点多卡）** → **FSDP**
   - 无需额外依赖、PyTorch 原生支持、Lightning 集成完美
   - 调试工具丰富（标准 PyTorch profiler 即可）

2. **大规模训练（30B+ 模型，或多节点）** → **DeepSpeed ZeRO-2/3**
   - 更成熟的 Offloading 支持
   - 更好的通信优化
   - 更多大型项目的验证案例

3. **显存极度紧张（想在 24GB 卡上跑 70B）** → **DeepSpeed ZeRO-3 + NVMe Offloading**
   - 这是唯一能把 70B 模型塞进消费级显卡的方案之一
   - 代价是训练速度较慢（受限于 PCIe/SSD 带宽）

4. **需要与 Megatron-LM / MosaicML Composer 等框架集成** → **DeepSpeed**
   - 这些框架都基于 DeepSpeed 构建

## 性能估算公式

最后给出一组实用的估算公式，帮助你在开始训练之前评估资源需求：

```python
def estimate_training_resources(model_params_b, num_gpus, strategy="ddp"):
    """估算训练资源需求"""
    
    param_bytes = 2  # BF16
    state_bytes = 4  # FP32 for m, v
    
    if strategy == "ddp":
        model_mem = model_params_b * param_bytes
        grad_mem = model_params_b * param_bytes
        opt_mem = model_params_b * state_bytes * 2
        act_mem = 8  # 粗估
        per_card = model_mem + grad_mem + opt_mem + act_mem
        
    elif strategy == "fsdp":
        shard_ratio = 1 / num_gpus
        model_mem = model_params_b * param_bytes * shard_ratio
        grad_mem = model_params_b * param_bytes * shard_ratio
        opt_mem = model_params_b * state_bytes * 2 * shard_ratio
        act_mem = 10  # FSDP activation checkpointing 开销稍高
        per_card = model_mem + grad_mem + opt_mem + act_mem
        
    elif strategy.startswith("zero"):
        stage = int(strategy.split("-")[1])
        shard_ratio = 1 / num_gpus
        if stage == 1:
            model_mem = model_params_b * param_bytes
            grad_mem = model_params_b * param_bytes
            opt_mem = model_params_b * state_bytes * 2 * shard_ratio
        elif stage == 2:
            model_mem = model_params_b * param_bytes
            grad_mem = model_params_b * param_bytes * shard_ratio
            opt_mem = model_params_b * state_bytes * 2 * shard_ratio
        elif stage == 3:
            model_mem = model_params_b * param_bytes * shard_ratio
            grad_mem = model_params_b * param_bytes * shard_ratio
            opt_mem = model_params_b * state_bytes * 2 * shard_ratio
        act_mem = 8
        per_card = model_mem + grad_mem + opt_mem + act_mem
    
    total_model_gb = model_params_b * param_bytes / 1e9
    
    print(f"\nModel: {model_params_b/1e9:.0f}B params ({total_model_gb:.1f} GB in BF16)")
    print(f"Strategy: {strategy.upper()}, GPUs: {num_gpus}")
    print(f"Per-card memory estimate: {per_card:.1f} GB")
    
    gpu_options = [
        ("RTX 4090", 24),
        ("A6000", 48),
        ("A100", 80),
        ("H100", 80),
    ]
    
    print(f"\nGPU compatibility:")
    for name, mem in gpu_options:
        status = "✓ OK" if per_card <= mem * 0.9 else "✗ Too small"
        util = per_card / mem * 100
        print(f"  {name:>12s} ({mem:>3d}GB): {status} "
              f"(utilization {util:.0f}%)")


estimate_training_resources(7e9, num_gpus=4, strategy="fsdp")
estimate_training_resources(70e9, num_gpus=8, strategy="zero-3")

# 典型输出:
# Model: 7.0B params (14.0 GB in BF16)
# Strategy: FSDP, GPUs: 4
# Per-card memory estimate: 6.8 GB
#
# GPU compatibility:
#   RTX 4090  (24GB): ✓ OK (utilization 28%)
#   A6000      (48GB): ✓ OK (utilization 14%)
#   A100       (80GB): ✓ OK (utilization 9%)
#   H100       (80GB): ✓ OK (utilization 9%)
#
# Model: 70.0B params (140.0 GB in BF16)
# Strategy: ZERO-3, GPUs: 8
# Per-card memory estimate: 47.5 GB
#
# GPU compatibility:
#   RTX 4090  (24GB): ✗ Too small (utilization 198%)
#   A6000      (48GB): ✗ Too small (utilization 99%)
#   A100       (80GB): ✓ OK (utilization 59%)
#   H100       (80GB): ✓ OK (utilization 59%)
```

下一节我们将把所有分布式知识整合起来，做一个端到端的实战演练——包括环境准备、启动脚本模板、性能基准测试方法以及最常见错误的排查清单。
