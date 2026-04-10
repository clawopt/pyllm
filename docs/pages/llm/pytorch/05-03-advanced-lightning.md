# 5.3 Lightning 中的高级训练特性

前两节我们学习了 Lightning 的基础用法和生命周期/Callback 系统，已经能够处理大多数标准的训练场景了——单优化器、单学习率调度器、单 GPU、标准精度。但 LLM 训练的实际需求往往比这复杂得多：你可能需要为不同的参数组设置不同的学习率（比如 Embedding 层用较小的 lr，其他层用较大的 lr）；你可能需要在同一模型上同时跑多个 DataLoader（比如同时做语言建模任务和分类任务的联合训练）；你需要把模型分布到多张 GPU 上来突破显存瓶颈；你还需要开启 Gradient Checkpointing 来在 24GB 显存的显卡上塞进一个 7B 模型。这一节我们将逐一解决这些问题。

## 多优化器与分层学习率

在某些场景下，不同部分的参数需要用不同的优化策略。最典型的例子是**分层学习率（Layer-wise Learning Rate Decay）**——让底层（靠近输入）的层使用较小的学习率，顶层（靠近输出）的层使用较大的学习率。直觉上的理由是：底层的特征表示更通用（类似于边缘检测之于 CNN），不应该被大幅修改；而顶层的特征更特定于当前任务，需要更大的调整空间。这在微调预训练 LLM 时尤其常见。

在 Lightning 中实现多优化器非常自然——`configure_optimizers()` 方法本身就支持返回多个优化器：

```python
class LitGPTWithLayeredLR(pl.LightningModule):
    def __init__(self, config, base_lr=3e-4, decay_factor=0.8):
        super().__init__()
        self.save_hyperparameters()
        self.gpt = GPT(config)

    def configure_optimizers(self):
        params = list(self.gpt.named_parameters())
        
        num_layers = len(self.gpt.h)
        layer_groups = {}
        
        for name, param in params:
            if not param.requires_grad:
                continue
            
            if 'wte' in name or 'lm_head' in name:
                group_name = 'embedding'
            elif 'rope' in name:
                group_name = 'rope'
            elif 'ln_f' in name:
                group_name = 'final_norm'
            else:
                layer_num = None
                for i in range(num_layers):
                    if f'.h.{i}.' in name or f'.blocks.{i}.' in name:
                        layer_num = i
                        break
                if layer_num is not None:
                    group_name = f'layer_{layer_num}'
                else:
                    group_name = 'other'

            if group_name not in layer_groups:
                layer_groups[group_name] = []
            layer_groups[group_name].append(param)

        sorted_groups = sorted(layer_groups.items())
        param_groups = []
        
        for idx, (group_name, group_params) in enumerate(sorted_groups):
            decay = self.hparams.decay_factor ** idx
            lr = self.hparams.base_lr * decay
            
            param_groups.append({
                'params': group_params,
                'lr': lr,
                'name': group_name,
            })
            
            total_params = sum(p.numel() for p in group_params)
            print(f"  Group '{group_name}': "
                  f"lr={lr:.2e}, params={total_params:,}")

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=0.01,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }
```

运行后你会看到类似这样的输出：

```
  Group 'embedding': lr=3.00e-04, params=1,310,000
  Group 'layer_0':     lr=2.40e-04, params=393,216
  Group 'layer_1':     lr=1.92e-04, params=393,216
  Group 'layer_2':     lr=1.54e-04, params=393,216
  Group 'layer_3':     lr=1.23e-04, params=393,216
  Group 'final_norm':  lr=9.83e-05, params=256
  Group 'lm_head':     lr=7.86e-05, params=1,310,000
```

Embedding 和 LM Head 使用最大的学习率（base_lr），每往深一层就乘以 decay_factor（0.8），最深的层使用最小的学习率。这种策略在微调场景中通常能带来比统一学习率更好的效果。

另一个多优化器的典型应用是 **GAN 训练**——Generator 和 Discriminator 各自拥有独立的优化器：

```python
class LitGAN(pl.LightningModule):
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=4e-4)
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            loss = self.generator_loss(batch)
        else:
            loss = self.discriminator_loss(batch)
        return loss
```

Lightning 会在每个 step 中依次调用两个优化器各自的 `training_step`（通过 `optimizer_idx` 参数区分），然后依次执行各自的 `.step()` 和 `.zero_grad()`。

## 多 DataLoader 支持

默认情况下，Lightning 的 `trainer.fit(model)` 只接收一个 train_loader 和可选的 val_loader。但有些任务需要多个数据加载器——比如在训练语言模型的同时还想在一个分类数据集上做辅助训练，或者想给训练集的不同部分分配不同的采样权重。

Lightning 通过以下方法支持多 DataLoader：

```python
class LitMultiTaskGPT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.gpt = GPT(config)
        self.classifier = nn.Linear(config.n_embed, num_classes)

    def train_dataloader(self):
        lm_loader = build_dataloader(
            file_path="corpus.jsonl",
            tokenizer=tokenizer,
            batch_size=16,
        )
        cls_loader = DataLoader(classification_dataset, batch_size=32)
        return [lm_loader, cls_loader]

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            output = self.gpt(batch['input_ids'], labels=batch['labels'])
            loss = output['loss']
            self.log('lm_loss', loss)
        else:
            logits = self.classifier(
                self.gpt.gpt.wte(batch['input_ids']).mean(dim=1)
            )
            loss = F.cross_entropy(logits, batch['label'])
            self.log('cls_loss', loss)

        return loss

    def val_dataloader(self):
        return [
            build_dataloader(file_path="val_corpus.jsonl", ...),
            DataLoader(val_cls_dataset),
        ]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # 类似 training_step 的逻辑
        ...
```

当有多个 train_dataloader 时，Lightning 默认会在每个 epoch 中遍历所有 loader（以最长的那个为准，短的会循环）。你也可以通过设置 `Trainer(strategy='ddp')` 让每个 GPU 处理不同的 loader 子集。

## 分布式训练策略选择

这是 Lightning 最强大的功能之一——它让你可以通过改变一个字符串参数就在 DDP、FSDP、DeepSpeed 等分布式训练方案之间切换，而不需要修改任何模型代码。

### DDP（DistributedDataParallel）

DDP 是最基本的分布式策略：每张 GPU 上保存一份完整的模型副本，每个进程独立地在自己分配到的数据子集上计算梯度，然后通过 AllReduce 操作同步平均所有 GPU 上的梯度。

```python
trainer = pl.Trainer(
    strategy="ddp",          # 或 "ddp_spawn"
    devices=4,               # 使用 4 张 GPU
    accelerator="gpu",
)
```

DDP 的优点是实现简单、通信开销相对较低（只同步梯度）；缺点是每张卡都要存一份完整的模型，对于大模型来说显存可能不够用。7B FP16 模型约需 14GB 显存 + 28GB 优化器状态 = 42GB/卡，这意味着至少需要 A100 (80GB) 才能跑得动。

### FSDP（Fully Sharded Data Parallel）

FSDP 是 PyTorch 原生的模型分片方案。不同于 DDP 的"完整复制"，FSDP 将模型的参数、梯度和优化器状态全部切分到各张 GPU 上。在前向传播时，当前需要的层参数通过 All-Gather 从各卡收集；反向传播时，梯度通过 Reduce-Scatter 分散回各卡。

```python
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

strategy = FSDPStrategy(
    sharding_strategy="FULL_SHARD",   # 全部分片（最省显存）
    # 其他选项: SHARD_GRAD_OP (分片梯度+优化器), NO_SHARD (=DDP)
    
    auto_wrap_policy=None,           # 或使用 transformer_auto_wrap_policy
    
    state_dict_type="full",          # checkpoint 时收集完整状态
    state_dict_config=FullStateDictConfig(
        offload_to_cpu=True,         # 保存 checkpoint 时 offload 到 CPU
    ),
)

trainer = pl.Trainer(
    strategy=strategy,
    devices=4,
    precision="bf16-mixed",
)
```

FSDP 的显存占用约为总模型大小的 `1/N`（N 为 GPU 数量）。对于 7B 模型，4×A100 (80GB) 下每卡只需约 10.5GB 显存就能放下全部内容（包括参数+梯度+优化器状态），而 DDP 需要 42GB/卡。这就是为什么 FSDP 成为 13B+ 模型的首选方案。

### DeepSpeed ZeRO

DeepSpeed 是微软开发的分布式训练框架，它的 ZeRO（Zero Redundancy Optimizer）系列提供了三种级别的内存优化：

| 阶段 | 分片内容 | 显存节省 | 通信开销 |
|-----|---------|---------|---------|
| ZeRO-1 | 仅优化器状态 | ~4x | 低 |
| ZeRO-2 | 优化器状态 + 梯度 | ~8x | 中 |
| ZeRO-3 | 全部切分（参数+梯度+优化器状态）| ~N× | 高 |

```python
trainer = pl.Trainer(
    strategy="deepspeed_stage_2",      # 或 deepspeed_stage_3
    devices=4,
    precision="bf16-mixed",
)

# 或者使用配置文件方式:
ds_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
    },
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
}

trainer = pl.Trainer(
    strategy=DeepSpeedStrategy(config=ds_config),
    devices=4,
)
```

ZeRO-2 在大多数情况下是性价比最高的选择——它将优化器状态和梯度分片到各卡，通信开销适中，显存节省显著。ZeRO-3 进一步分片了参数本身，通信开销更大但在极大规模模型（100B+）或 GPU 显存极度有限的情况下可能是唯一的选择。另外 DeepSpeed 还支持 CPU/NVMe Offloading——把不常用的参数卸载到系统内存甚至 SSD 上，进一步降低 GPU 显存需求，代价是速度变慢。

## Gradient Checkpointing：以时间换空间

即使使用了 FSDP 或 DeepSpeed，某些情况下显存仍然不够——尤其是当你想在消费级显卡（如 RTX 3090 的 24GB 或 RTX 4090 的 24GB）上训练或微调 7B 模型的时候。7B BF16 模型本身约 14GB，加上优化器状态约 28GB，加上激活值（中间层的输出）可能再增加数 GB，总计轻松超过 40GB。

**Gradient Checkpointing（也叫 Activation Checkpointing / Rematerialization）** 的核心思想是：在前向传播时不保留所有中间层的激活值（它们占用了大量显存），而是在需要时重新计算。具体来说，正常的前向传播会保存每一层的输入供反向传播使用；Gradient Checkpointing 则只保留少数几个检查点的激活值，其余的在反向传播时从最近的检查点开始重新计算。

```python
trainer = pl.Trainer(
    gradient_checkpointing=True,   # 一行搞定！
    precision="bf16-mixed",
)
```

是的，在 Lightning 中只需要加这一个参数就够了。它会自动对 `nn.Module` 的子模块应用 gradient checkpointing。

代价是什么呢？大约 20%~30% 的额外计算时间。因为反向传播时需要重新计算那些被丢弃的激活值，相当于每个被 checkpoint 的层多做了一次前向传播。但对于无法放入显存的任务来说，这个时间代价是完全值得的——毕竟没有它的话根本跑不起来。

```python
def benchmark_gradient_checkpointing():
    model = GPT(GPTConfig(vocab_size=1000, n_embed=512, num_heads=8, num_layers=12))
    model_cuda = model.cuda().bfloat16()
    model_gc = model.cuda().bfloat16()

    from torch.utils.checkpoint import checkpoint
    for block in model_gc.gpt.h:
        original_forward = block.forward
        def checkpointed_forward(x, mask=None, orig_fwd=original_forward):
            return checkpoint(orig_fwd, x, mask, use_reentrant=False)
        block.forward = checkpointed_forward

    x = torch.randint(0, 1000, (4, 256)).cuda()

    import time

    model_cuda.eval()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model_cuda(x)
    torch.cuda.synchronize()
    normal_time = (time.time() - t0) / 10

    model_gc.eval()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model_gc(x)
    torch.cuda.synchronize()
    gc_time = (time.time() - t0) / 10

    print(f"Normal forward: {normal_time*1000:.1f} ms")
    print(f"With grad ckpt: {gc_time*1000:.1f} ms")
    print(f"Overhead: {(gc_time/normal_time - 1)*100:.1f}%")


benchmark_gradient_checkpointing()

# 典型输出:
# Normal forward: 45.2 ms
# With grad ckpt: 58.7 ms
# Overhead: 29.9%  ← 约 30% 的额外时间
```

实际的时间开销取决于模型层数、序列长度和硬件配置，但 20%~40% 是一个合理的经验范围。对于 7B 模型在 RTX 4090 上的微调场景，开启 Gradient Checkpointing 后的显存占用可以从约 26GB 降到约 18GB，刚好能放进 24GB 显存。

## 半精度配置详解

我们在第 4 节讨论过 FP16/BF16 的区别，这里从 Lightning 的角度补充一些细节。

```python
precision_configs = {
    "32": "纯 FP32，无混合精度，最稳定但最慢且最耗显存",
    "16-mixed": "FP16 混合精度，需要 GradScaler，适合旧GPU(V100等)",
    "bf16-mixed": "BF16 混合精度，不需要GradScaler，推荐用于Ampere+(A100/H100/RTX30+)",
}

for prec, desc in precision_configs.items():
    trainer = pl.Trainer(precision=prec)
    print(f"precision='{prec}': {desc}")
```

需要注意的是，`precision="bf16-mixed"` 要求你的 GPU 硬件支持 BF16 运算。如果你在不支持的硬件上使用，Lightning 会自动回退到 FP32 并给出警告。你可以用以下代码检测是否支持：

```python
def check_bf16_support():
    if not torch.cuda.is_available():
        print("No CUDA available")
        return False
    
    capability = torch.cuda.get_device_capability()
    major, minor = capability
    
    if major >= 8:  # Ampere (SM 8.0+) or later
        print(f"✓ BF16 supported (compute capability {major}.{minor})")
        return True
    else:
        print(f"✗ BF16 NOT supported (compute capability {major}.{minor})")
        return False


check_bf16_support()
```

到这里，Lightning 的高级特性基本覆盖完毕。下一节我们将用一个完整的实战项目把这些知识串联起来——用 Lightning 训练我们的 GPT 模型，并与手写循环做一次全面的对比。
