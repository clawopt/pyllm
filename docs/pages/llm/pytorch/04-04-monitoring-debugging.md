# 4.4 训练监控与调试

经过前三节的努力，我们已经拥有了一个功能完整的训练循环：模型前向传播、损失计算、梯度裁剪、AdamW 优化器更新、余弦学习率调度、混合精度加速、梯度累积支持，以及 checkpoint 管理和早停机制。代码写好了，训练也启动了——但怎么知道它跑得对不对呢？loss 在下降固然是好事，但如果下降得不够快、或者突然飙升到 NaN、或者 train loss 和 val loss 的差距越来越大，这些现象分别意味着什么？这一节就是你的"训练诊断手册"：我们将系统性地学习如何监控训练过程、如何识别各种异常信号、以及如何根据症状定位根因并采取正确的修复措施。

## 日志方案：从最简到生产级

在讨论具体问题之前，先解决一个基础设施问题：你用什么方式记录和查看训练日志？这看起来是个小问题，但选择不当会严重影响调试效率。

**Level 0：print 大法**——最原始的方式，直接用 print 输出 loss 值：

```python
for step, batch in enumerate(loader):
    loss = train_step(model, batch)
    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")
```

优点是零依赖、立即可用；缺点是无法回溯历史（终端滚过去就看不到了）、无法可视化趋势、多进程训练时输出混乱。

**Level 1：Python logging + tqdm**——稍微好一点的标准做法：

```python
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

for step, batch in enumerate(tqdm(loader, desc="Training")):
    loss = train_step(model, batch)
    if step % 100 == 0:
        logger.info(f"Step {step}: loss={loss.item():.4f}, "
                     f"lr={scheduler.get_last_lr()[0]:.2e}")
```

这种方式同时输出到终端和日志文件，可以事后回溯。tqdm 进度条提供了实时的视觉反馈。

**Level 2：TensorBoard / W&B**——可视化方案，强烈推荐用于任何超过几小时的训练任务：

```python
import wandb

wandb.init(
    project="llm-training",
    config={
        "model": "gpt-small",
        "lr": 3e-4,
        "batch_size": 16,
        "epochs": 20,
    }
)

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        loss = train_step(model, batch)

        if step % 10 == 0:
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": compute_grad_norm(model),
                "epoch": epoch,
                "step": global_step,
            })

    val_loss = evaluate(model, val_loader)
    wandb.log({"val_loss": val_loss, "epoch": epoch})
```

Weights & Biases (W&B) 或 TensorBoard 能让你在浏览器中实时看到 loss 曲线的学习率变化图、GPU 利用率等关键指标，并且支持多个实验的对比。当你需要比较不同超参数配置的效果时，这种可视化能力是无价的。

**Level 3：JSON Lines 结构化日志**——适合离线分析和自动化处理：

```python
import json
import time

log_file = open("metrics.jsonl", "w")

for step in range(total_steps):
    t_start = time.time()
    loss = train_step(model, batch)
    elapsed = time.time() - t_start

    record = {
        "step": step,
        "epoch": step // steps_per_epoch,
        "train_loss": loss.item(),
        "lr": scheduler.get_last_lr()[0],
        "tokens_per_sec": batch_size * seq_len / elapsed,
        "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "timestamp": time.time(),
    }
    log_file.write(json.dumps(record) + "\n")
    log_file.flush()

log_file.close()

# 事后分析
import pandas as pd
df = pd.read_json("metrics.jsonl", lines=True)
print(df.describe())
```

JSONL 格式的优势在于每行是一个独立的 JSON 对象，可以用任何文本工具查看，也可以用 pandas/其他工具轻松加载分析。

## 关键监控指标：你应该关注哪些数字？

一个健康的训练过程应该持续监控以下指标：

**Train Loss** — 最基本的指标。正常情况下应该单调递减（或至少总体趋势向下），衰减速度通常呈现"快→慢→更慢"的模式（因为容易学的模式先被捕捉）。如果 train loss 完全不下降，说明优化出了根本性问题；如果下降极慢，可能是学习率太小或者数据/模型有问题。

**Validation Loss** — 比 train loss 更重要的指标，因为它反映的是模型的泛化能力而非记忆能力。val loss 应该跟随 train loss 一起下降，但通常会略高于 train loss（这是正常的）。如果 val loss 远大于 train loss（比如 2~3 倍以上），说明严重过拟合；如果 val loss 开始上升而 train loss 继续下降，这是过拟合的明确信号。

**Learning Rate** — 确认学习率调度是否按预期工作。warmup 阶段应该线性增长，cosine decay 阶段应该平滑下降。如果 lr 图出现异常跳变，检查是否有代码错误或外部干预。

**Gradient Norm** — 每步梯度的 L2 茁数。这个指标能提前预警很多问题：梯度过大（>10）可能意味着即将出现 NaN；梯度过小（<1e-6）意味着学习停滞。正常的梯度范数通常在 0.1~5 之间，具体取决于模型大小和学习率。

**Throughput (tokens/sec)** — 训练效率的直接度量。如果这个值突然下降，可能是数据加载瓶颈、GC 暂停或其他系统级问题。对于 A100 GPU，7B 模型的典型 throughput 在 10K~50K tokens/sec 范围（取决于 batch size 和序列长度）。

**GPU Utilization & Memory** — `nvidia-smi` 可以看到 GPU 利用率和显存占用。利用率长期低于 80% 说明存在瓶颈（通常是 CPU 数据加载）；显存接近上限说明 batch size 太大或有内存泄漏。

## 常见异常诊断手册

现在我们进入最实用的部分——针对每种常见的训练异常，给出症状描述、可能原因和解决方案。

### 异常一：Loss = NaN（或 Inf）

这是最令人恐慌的情况之一。程序不报错，但 loss 突然变成 NaN 之后就一直保持 NaN（因为 NaN 会通过所有算子传播）。

**排查步骤**（按优先级排序）：

1. **检查数据**：是否有异常值？比如 token ID 超出词表范围、全零序列、无穷大值等。
   ```python
   def check_data(loader):
       for i, batch in enumerate(loader):
           ids = batch['input_ids']
           labels = batch['labels']
           if (ids < 0).any() or (ids >= vocab_size).any():
               print(f"Batch {i}: invalid token IDs!")
               return False
           if (labels == -100).all():
               print(f"Batch {i}: all labels are padding!")
               return False
       return True
   ```

2. **检查学习率**：是否太大？尤其是 warmup 结束后的第一步。尝试将峰值学习率除以 10 再试。

3. **检查数值精度**：使用 FP16 时是否正确使用了 GradScaler？BF16 是否可用？某些旧 GPU 不支持 BF16 强制使用会导致未定义行为。

4. **检查 loss 函数**：`ignore_index=-100` 是否设置正确？如果 label 中没有 -100 但设置了 ignore_index，本身不会导致 NaN，但如果所有位置都被 ignore 了（全是 -100），loss 就是 0/0=NaN。

5. **开启异常检测**：PyTorch 提供了 nan 检测工具：
   ```python
   torch.autograd.set_detect_anomaly(True)
   ```
   这会在每次 autograd 操作时检查是否产生了 inf/nan，虽然会显著降低运行速度，但对定位问题极其有用。

### 异常二：Loss 不下降

模型在训练了相当长的时间后 loss 几乎不变，或者下降幅度微乎其微。

**常见原因及对策**：

| 原因 | 症状 | 解决方案 |
|-----|------|---------|
| 学习率过小 | loss 曲线几乎水平 | 增大 lr（×2~×10） |
| 未 shuffle | loss 阶梯状下降 | 确保 train_loader shuffle=True |
| 数据标签错误 | loss 不下降且稳定在一个非平凡值 | 打印几个样本检查 input_ids 和 labels |
| 模型 bug | loss 从一开始就不合理 | 用单个 batch 过拟合测试验证 |
| Embedding 层冻结 | 初期 loss 很高然后缓慢下降 | 检查 requires_grad 设置 |

其中"模型 bug"是最容易被忽视但又最常见的。下面的过拟合测试能快速排除这种可能性：

```python
def sanity_overfit_test(model, single_batch):
    """用单个 batch 过拟合来验证模型能否学习"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    x = single_batch['input_ids'].clone()
    y = single_batch['labels'].clone()
    
    losses = []
    for step in range(500):
        optimizer.zero_grad()
        out = model(x, labels=y)
        loss = out['loss']
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}")

    final_loss = losses[-1]
    initial_loss = losses[0]

    if final_loss < 0.1 * initial_loss:
        print(f"✓ Sanity check PASSED: loss {initial_loss:.4f} → {final_loss:.4f}")
        return True
    else:
        print(f"✗ Sanity check FAILED: loss barely changed")
        return False
```

如果能在这个测试中把 loss 降到接近零，说明模型架构和训练流程都是正确的，问题出在大规模数据上；如果连单个 batch 都学不好，那一定是代码有 bug 或者模型表达能力不足。

### 异常三：Val Loss >> Train Loss（过拟合）

Val loss 显著高于 train loss 且差距持续扩大。

**解决方案**：
- 增加 dropout（从 0 到 0.1~0.2）
- 增加 weight_decay（从 0.01 到 0.1）
- 增加数据量或做数据增强
- 减小模型规模
- 早停（Early Stopping）——当 val loss 连续 N 个 epoch 不再改善时停止训练

### 异常四：Loss 震荡剧烈

Loss 曲线像心电图一样剧烈波动而不是平滑下降。

**原因通常是**：batch size 太小（梯度估计噪声大）或学习率太大。解决方案：增大 batch size（或用梯度累积模拟）、降低学习率、增加 warmup 步数。

### 异常五：GPU OOM（Out of Memory）

CUDA out of memory 错误。

**按顺序尝试以下措施**：
1. 减小 `per_device_train_batch_size`
2. 启用 `gradient_checkpointing=True`（以时间换空间，节省约 30%~50% 显存）
3. 使用混合精度训练（FP16/BF16）
4. 使用梯度累积来补偿减小的 batch size
5. 如果还不够，考虑 ZeRO/FSDP 等分布式策略（第 7 章）

```python
def estimate_memory_usage(model, batch_size, seq_len, dtype=torch.float32):
    bytes_per_param = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}
    bpp = bytes_per_param[dtype]

    params = sum(p.numel() for p in model.parameters())
    model_mem = params * bpp

    activations_est = 34 * params * (batch_size * seq_len / params) ** 0.75
    grad_mem = params * bpp
    optimizer_mem = params * bpp * 2  # Adam: m + v
    states_mem = params * bpp * 2     # momentum + variance

    total = model_mem + activations_est + grad_mem + optimizer_mem + states_mem
    total_gb = total / 1e9

    print(f"Estimated memory breakdown ({dtype}):")
    print(f"  Model parameters:  {model_mem/1e9:>8.2f} GB")
    print(f"  Activations (est): {activations_est/1e9:>8.2f} GB")
    print(f"  Gradients:         {grad_mem/1e9:>8.2f} GB")
    print(f"  Optimizer states:  {optimizer_mem/1e9:>8.2f} GB")
    print(f"  {'─'*35}")
    print(f"  TOTAL:             {total_gb:>8.2f} GB")


estimate_memory_usage(your_model, batch_size=8, seq_len=2048, dtype=torch.bfloat16)
```

## Checkpoint 管理最佳实践

除了基本的保存/加载之外，还有几个 checkpoint 相关的最佳实践值得了解。

**Top-K 保留策略**：不只保留最新的 checkpoint，而是根据验证集指标保留历史最好的 K 个。这样即使某个 checkpoint 后来被发现有问题（比如评估时才发现），还可以回退到之前的版本。

```python
class TopKCheckpointer:
    def __init__(self, save_dir, k=3, metric_name='val_loss', mode='min'):
        self.save_dir = save_dir
        self.k = k
        self.metric_name = metric_name
        self.mode = mode
        self.checkpoints = []  # [(metric_value, path)]

    def maybe_save(self, model, optimizer, metrics):
        value = metrics[self.metric_name]
        path = os.path.join(self.save_dir, f"ckpt_{len(self.checkpoints)}.pt")

        should_save = (
            len(self.checkpoints) < self.k or
            (self.mode == 'min' and value < self.checkpoints[-1][0]) or
            (self.mode == 'max' and value > self.checkpoints[-1][0])
        )

        if should_save:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, path)

            self.checkpoints.append((value, path))
            self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))

            while len(self.checkpoints) > self.k:
                _, old_path = self.checkpoints.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)

            print(f"Checkpoint saved: {metric_name}={value:.4f} "
                  f"(rank {self.checkpoints.index((value, path))+1}/{self.k})")
```

**完整性校验**：加载 checkpoint 后立即做一次 forward pass 来验证参数完整性，防止保存过程中出现文件损坏或版本不匹配的问题：

```python
def safe_load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    missing, unexpected = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )

    if missing:
        logger.warning(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

    dummy_input = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        output = model(dummy_input)
    logger.info(f"Checkpoint verification passed: "
                f"output shape={output['logits'].shape}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('global_step', 0)
```

到这里，第 4 章"训练循环与优化策略"的全部四个小节就结束了。我们从最简单的 30 行训练循环出发，逐步添加了梯度裁剪、学习率调度、评估与日志、checkpoint 管理和早停机制，组装成了生产级的 Trainer 类。然后深入了 AdamW 优化器的内部机制，理解了每个超参数的含义和调优方法，以及替代方案的适用场景。接着学习了 Cosine Annealing + Warmup 学习率调度的原理和实践、梯度累积如何突破显存限制、以及 FP16/BF16 混合精度训练的完整实现。最后建立了完整的训练监控体系——从日志方案选择到关键指标解读，再到五种常见异常的系统化诊断流程。

下一章我们将进入 PyTorch Lightning 的世界——看看它是如何把我们在本章手写的几十行样板代码封装成声明式的 API，让开发者能够专注于模型逻辑本身而不是工程细节。
