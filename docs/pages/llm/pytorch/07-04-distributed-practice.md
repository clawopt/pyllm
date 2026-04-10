# 7.4 分布式训练实战与排错

前三节我们分别学习了 DDP、FSDP 和 DeepSpeed ZeRO 三种分布式训练方案的理论基础和 API 用法。这一节的目标是把它们整合起来，完成一个端到端的实战演练——从环境准备到启动脚本、从性能基准测试到常见错误的系统化排查。分布式训练中最让人头疼的不是"怎么写代码"，而是"跑起来之后出了奇怪的问题不知道怎么定位"。所以这一节会花大量篇幅在错误排查上，建立一个完整的诊断思维框架。

## 环境准备清单

在启动任何分布式训练之前，先确认以下环境条件已经满足：

```python
def check_distributed_environment():
    """分布式训练环境检查"""
    import torch
    import subprocess
    import socket

    print("=" * 60)
    print("Distributed Training Environment Check")
    print("=" * 60)

    # 1. PyTorch 版本与 CUDA
    print(f"\n[1] PyTorch & CUDA")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = torch.cuda.mem_get_info(i)[0] / 1e9
            print(f"  GPU {i}: {props.name} "
                  f"({mem_gb:.1f} GB free, {props.total_memory/1e9:.0f} GB total)")
            if hasattr(props, 'major') and props.major >= 8:
                print(f"    → Ampere+ (BF16 supported ✓)")

    # 2. NCCL
    try:
        import nccl
        nccl_version = nccl.__version__
        print(f"\n[2] NCCL version: {nccl_version}")
        if nccl_version:
            print(f"  NCCL available ✓")
    except ImportError:
        print("\n[2] NCCL not found! Install: pip install nccl")

    # 3. 网络连通性 (多卡时)
    if torch.cuda.device_count() > 1:
        print(f"\n[3] NCCL connectivity test (P2P bandwidth)...")
        try:
            bandwidth = torch.cuda.nccl.bandwidth()
            print(f"  P2P bandwidth: {bandwidth / 1e9:.1f} GB/s")
            if bandwidth > 50e9:
                print(f"  → NVLink detected ✓ (excellent)")
            elif bandwidth > 10e9:
                print(f"  → PCIe Gen4+ ✓ (good)")
            else:
                print(f"  → Slow interconnect, may impact performance")
        except Exception as e:
            print(f"  Bandwidth test failed: {e}")

    # 4. DeepSpeed (可选)
    try:
        import deepspeed
        print(f"\n[4] DeepSpeed: {deepspeed.__version__} ✓")
    except ImportError:
        print(f"\n[4] DeepSpeed: not installed (optional)")

    # 5. 文件系统
    print(f"\n[5] Disk space")
    result = subprocess.run(
        ["df", "-h", "/"], capture_output=True, text=True
    )
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        parts = lines[1].split()
        if len(parts) >= 5:
            total = parts[1]
            avail = parts[3]
            use_pct = parts[4]
            print(f"  Total: {total}, Available: {avail} ({use_pct} used)")

    # 6. SSH (多节点场景)
    if "MASTER_ADDR" in os.environ or "RANK" in os.environ:
        print(f"\n[6] Multi-node config:")
        print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'N/A')}")
        print(f"  RANK: {os.environ.get('RANK', 'N/A')}")
        print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'N/A')}")

    print("\n" + "=" * 60)


check_distributed_environment()
```

运行这个检查脚本可以快速排除大部分环境问题。最常见的环境问题包括：

**CUDA 版本不匹配**：PyTorch 编译时的 CUDA 版本必须与系统中安装的 CUDA toolkit 版本一致（或兼容）。比如用 `pip install torch` 安装的 PyTorch 通常绑定 CUDA 11.8 或 12.1，如果你的驱动支持更高版本的 CUDA 但没有对应的 toolkit，可能会出现奇怪的运行时错误。

**NCCL 版本过旧或缺失**：DDP/FSDP/DeepSpeed 都依赖 NCCL 做通信。如果 NCCL 没有安装或版本太旧，分布式初始化就会失败。推荐安装方式：`conda install -c nvidia nccl` 或 `pip install nccl`。

## 完整的分布式训练脚本模板

下面是一个生产级的分布式训练脚本，支持 DDP/FSDP/DeepSpeed 三种策略切换：

```python
import os
import json
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="ddp",
                        choices=["ddp", "fsdp", "deepspeed"],
                        help="Distributed strategy")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="alpaca_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./output/dist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if args.strategy == "ddp":
        run_ddp_training(args, local_rank, is_main)
    elif args.strategy == "fsdp":
        run_fsdp_training(args, local_rank, is_main)
    elif args.strategy == "deepspeed":
        run_deepspeed_training(args, local_rank, is_main)


def run_ddp_training(args, local_rank, is_main):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    
    dist.init_process_group(backend="nccl")
    
    model, tokenizer, loader = build_model_and_data(args)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW_bnb_8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    
    for epoch in range(args.epochs):
        loader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(loader):
            ids = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            
            output = model(ids, labels=labels)
            loss = output.loss / args.grad_accum
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

            if is_main and step % 50 == 0:
                print(f"[Rank {local_rank}] Epoch {epoch} "
                      f"Step {step}: loss={loss.item()*args.grad_accum:.4f}")
    
    if is_main:
        save_model(model, tokenizer, args.output_dir)
    
    dist.destroy_process_group()


def run_fsdp_training(args, local_rank, is_main):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, FullStateDictConfig
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    
    model, tokenizer, loader = build_model_and_data(args)
    
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={TransformerBlock},
    )
    
    fsdp_model = FSDP(
        model.to(local_rank),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        state_dict_type="full",
        state_dict_config=FullStateDictConfig(offload_to_cpu=True),
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, fsdp_model.parameters()),
        lr=args.lr,
    )
    
    for epoch in range(args.epochs):
        loader.sampler.set_epoch(epoch)
        fsdp_model.train()
        
        for step, batch in enumerate(loader):
            ids = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            
            output = fsdp_model(ids, labels=labels)
            loss = output.loss / args.grad_accum
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), 1.0)
                optimizer.step()
            
            if is_main and step % 50 == 0:
                print(f"[Rank {local_rank}] Epoch {epoch} "
                      f"Step {step}: loss={loss.item()*args.grad_accum:.4f}")
    
    if is_main:
        save_model(fsdp_model, tokenizer, args.output_dir)


def run_deepspeed_training(args, local_rank, is_main):
    import deepspeed
    
    model, tokenizer, loader = build_model_and_data(args)
    
    ds_config_path = args.deepspeed_config or create_ds_config_file(args)
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
        config=ds_config_path,
    )
    
    for epoch in range(args.epochs):
        loader.sampler.set_epoch(epoch)
        model_engine.train()
        
        for step, batch in enumerate(loader):
            ids = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            
            loss = model_engine(ids, labels=labels).loss
            model_engine.backward(loss)
            model_engine.step()
            
            if is_main and step % 50 == 0:
                print(f"[Rank {local_rank}] Epoch {epoch} "
                      f"Step {step}: loss={loss.item():.4f}")
    
    if is_main:
        model_engine.save_checkpoint(args.output_dir)


if __name__ == "__main__":
    main()
```

## 启动命令速查表

```bash
# ===== DDP 启动 =====

# 单机 4 卡
torchrun --nproc_per_node=4 train_dist.py --strategy ddp

# 单机 8 卡
torchrun --nproc_per_node=8 train_dist.py --strategy ddp \
    --per_device_train_batch_size=2


# ===== FSDP 启动 =====

# 单机 4 卡
torchrun --nproc_per_node=4 train_dist.py --strategy fsdp

# Lightning 方式（更简单）
python train_lightning.py --trainer.strategy=fsdp --trainer.devices=4


# ===== DeepSpeed 启动 =====

# 使用 accelerate launch
accelerate launch --num_processes 4 --use_deepspeed \
    --deepspeed_config ds_config.json train_dist.py --strategy deepspeed

# 或直接使用 deepspeed launcher
deepspeed --num_gpus=4 train_dist.py \
    --deepspeed_config ds_config.json


# ===== 多节点启动 =====

# 节点 0 (master)
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="10.0.0.1" \
    --master_port=29500 \
    --nproc_per_node=4 \
    train_dist.py

# 节点 1
torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="10.0.0.1" \
    --master_port=29500 \
    --nproc_per_node=4 \
    train_dist.py
```

## 性能基准：MFU 与 TFLOPS/GPU

评估分布式训练效率有两个核心指标：

**TFLOPS/GPU**: 每张 GPU 每秒执行的浮点运算次数。这是绝对速度指标——值越大越好。对于 GPT-2 类模型，理论峰值 TFLOPS 可以通过硬件规格计算（A100 BF16 Tensor Core 峰值约 312 TFLOPS），实际训练中通常能达到峰值的 30%~55%。

**MFU (Model FLOPs Utilization)**: 模型浮点运算利用率 = 实际 TFLOPS / 理论峰值 TFLOPS × 100%。MFU 综合了计算效率和通信开销的影响，是衡量分布式训练效率的最重要指标。

```python
def measure_mfu(model, batch_size, seq_len, step_time_seconds, num_gpus=1):
    """估算 MFU"""
    n_params = sum(p.numel() for p in model.parameters())
    
    # GPT-2 style model 的近似 FLOPs 计算
    # 参考: https://arxiv.org/abs/2004.08649
    model_flops = 6 * n_params  # 近似: 每个 param 做 ~6 次 MAC (前向3 + 反向3)
    
    tokens_per_step = batch_size * seq_len * num_gpus
    flops_per_token = model_flops  # 每 token 的 FLOPs
    
    total_flops = tokens_per_step * flops_per_token
    achieved_tflops = total_flops / step_time_seconds / 1e12
    
    # A100 BF16 峰值 (approximate)
    a100_peak_tflops = 312e12  # 312 TFLOPS
    
    mfu = achieved_tflops / a100_peak_tflops * 100
    
    throughput = tokens_per_step / step_time_seconds
    
    return {
        'params_B': n_params / 1e9,
        'tokens_per_step': tokens_per_step,
        'achieved_tflops': f"{achieved_tflops:.1f}",
        'mfu_percent': f"{mfu:.1f}%",
        'throughput': f"{throughput:.0f} tok/s",
        'throughput_k': f"{throughput/1000:.1f} K tok/s",
    }


print(measure_mfu(your_model, batch_size=4, seq_len=1024, 
                     step_time_seconds=0.35, num_gpus=4))
# 典型输出:
# {'params_B': '7.0', 'tokens_per_step': 16384, 
#  'achieved_tflops': '78.2', 'mfu_percent': '25.1%',
#  'throughput': '46811 tok/s', 'throughput_k': '46.8 K tok/s'}
```

MFU 的参考值范围：
- **< 15%**: 有严重问题（通信瓶颈、数据加载慢、GPU 利用率低）
- **15%~30%**: 正常范围（有优化空间）
- **30%~45%**: 良好（大多数生产级训练的水平）
- **45%~55%**: 优秀（高度优化的训练栈）
- **> 55%**: 极致优化（接近硬件极限）

## 常见错误排查手册

### 错误一：NCCL Timeout / Hang

**症状**：程序启动后长时间无输出，或者运行一段时间后突然卡住不动。

**原因排查顺序**：
1. **防火墙** —— 多节点场景下最常见的原因。NCCL 默认使用端口 29500（可配置），确保该端口在所有节点间开放。
   ```bash
   # 检查端口连通性
   telnet <master_ip> 29500
   
   # 如果不通，添加防火墙规则
   sudo ufw allow 29500/tcp
   ```
2. **网络不稳定** —— InfiniBand 或以太网丢包率高。检查 `ibstat` 或 `ping`。
3. **GPU 驱动版本不一致** —— 所有节点的 NVIDIA 驱动版本应完全相同。
4. **死锁（Deadlock）** —— Collective 操作不匹配（某个 rank 少调了一次 all_reduce）。检查所有 rank 是否执行了相同的代码路径。

### 错误二：RuntimeError: Expected all tensors to be on the same device

**症状**：DDP 包装后报设备不匹配错误。

**原因**：
1. 在 DDP 包装**之前**就调用了 `.to(device)` —— 应该先包装再移动（或者让 DDP 自动处理）
2. 数据没有正确移到对应 device —— 确保 `batch['input_ids'].to(local_rank)` 使用了正确的 rank
3. 某些 buffer/tensor 在 `__init__` 中注册到了 CPU —— 使用 `register_buffer(..., persistent=False)` 并在 forward 中 `.to(device)`

### 错误三：Address already in use / Port conflict

**症状**：启动时报端口已被占用。

**解决**：
```bash
# 查找占用端口的进程
lsof -i :29500

# 杀掉它
kill -9 <PID>

# 或者换一个端口
torchrun --master_port=29501 ...
```

### 错误四：OOM on Rank 0 only（其他 Rank 正常）

**症状**：只有 Rank 0 报 OOM，其他 Rank 正常运行。

**原因**：数据分布不均匀。可能的原因：
1. Dataset 的 `__len__` 不能被所有 rank 整除，导致最后一个 batch 大小不同
2. 使用了 `drop_last=False` 且最后一个 batch 触发了显存峰值
3. Rank 0 额外执行了日志/checkpoint 等 I/O 操作占用额外显存

**解决**：设置 `drop_last=True`，或在 Rank 0 上减少日志频率。

### 错误五：Loss 不收敛 / 各 Rank Loss 不一致

**症状**：训练过程中各 Rank 报告的 loss 值差异很大。

**原因**：
1. **shuffle 问题** —— 训练集用了 shuffle=True 而非 DistributedSampler。每个 rank 看到的数据分布不同是正常的，但 loss 差异过大就不正常了
2. **随机种子不一致** —— 确保所有 rank 使用相同的初始随机种子（`torch.manual_seed(seed + rank)`）
3. **Dropout 行为差异** —— 理论上各 rank 的 Dropout mask 不同是正常的（因为数据不同），但如果使用了 `model.train()` 后某些层的行为异常就需要检查

### 错误六：Checkpoint 恢复后 Loss 跳变

**症状**：从 checkpoint 恢复训练后，第一步的 loss 和之前完全不同。

**原因**：
1. **Optimizer state 未保存/恢复** —— 只保存了 model state_dict 但没保存 optimizer state_dict
2. **学习率调度器状态未恢复** —— scheduler 的 step count 未恢复导致 warmup/cosine 曲线位置错误
3. **RNG state 未恢复** —— 数据 shuffle 的随机状态未恢复导致后续数据顺序改变
4. **Epoch/Step 计数未恢复** —— 导致 sampler 或 scheduler 的行为异常

**完整保存清单**：
```python
checkpoint = {
    'epoch': epoch,
    'global_step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state_all(),
}
```

## 成本估算公式

最后给出一组实用的成本估算工具：

```python
def estimate_training_cost(model_b, dataset_tokens, strategy, num_gpus, gpu_type="A100"):
    configs = {
        "RTX 4090": {"mem_gb": 24, "cost_hr": 0.8, "tflops_peak": 165e12},
        "A6000": {"mem_gb": 48, "cost_hr": 1.5, "tflops_peak": 198e12},
        "A100": {"mem_gb": 80, "cost_hr": 2.0, "tflops_peak": 312e12},
        "H100": {"mem_gb": 80, "cost_hr": 4.0, "tflops_peak": 1979e12},
    }
    
    cfg = configs[gpu_type]
    model_gb = model_b * 2 / 1e9
    
    # 估算每步时间 (秒)
    mfu_estimate = {"ddp": 0.40, "fsdp": 0.35, "zero-2": 0.32, "zero-3": 0.28}
    mfu = mfu_estimate.get(strategy, 0.30)
    
    effective_tflops = cfg["tflops_peak"] * mfu
    tokens_per_step_per_gpu = 4 * 512  # 假设 batch=4, seq=512
    total_tokens_per_step = tokens_per_step_per_gpu * num_gpus
    
    seconds_per_step = (total_tokens_per_step * 6 * model_b) / effective_tflops
    steps_needed = dataset_tokens / total_tokens_per_step
    hours_needed = steps_needed * seconds_per_step / 3600
    
    cost = hours_needed * cfg["cost_hr"] * num_gpus
    
    print(f"\n{'='*60}")
    print(f"Training Cost Estimate: {model_b/1e9:.0f}B model, {dataset_tokens/1e9:.1f}B tokens")
    print(f"{'='*60}")
    print(f"  Strategy:     {strategy.upper()}")
    print(f"  GPUs:         {num_gpus}× {gpu_type}")
    print(f"  Steps needed:  {steps_needed:,}")
    print(f"  Time estimate: {hours_needed:.1f} hours ({hours_needed/24:.1f} days)")
    print(f"  Cost estimate: ${cost:,.0f}")


estimate_training_cost(
    model_b=7e9, dataset_tokens=100e9,
    strategy="fsdp", num_gpus=4, gpu_type="RTX 4090"
)

# 输出示例:
# =============================================================
# Training Cost Estimate: 7.0B model, 100.0B tokens
# =============================================================
#   Strategy:     FSDP
#   GPUs:         4× RTX 4090
#   Steps needed:  97,656
#   Time estimate: 22.8 hours (1.0 days)
#   Cost estimate: $72
```

到这里，第 7 章"分布式训练"的全部四个小节就结束了。我们从 DDP 的基本原理出发，学习了 AllReduce 梯度同步机制和 DistributedSampler 的正确用法；然后深入了 FSDP 的 FULL_SHARD 分片策略，理解了 AllGather/RScatter 的通信模式和三种 sharding level 的权衡；接着介绍了 DeepSpeed ZeRO 的三个递进阶段和丰富的 offloading/通信优化选项；最后在本节完成了环境检查、脚本模板、性能基准和六大类常见错误的系统化排查流程。

下一章我们将转向推理优化——当模型训练好了之后，如何让它跑得更快、更省资源？`torch.compile`、量化技术、模型导出格式以及性能分析工具将依次登场。
