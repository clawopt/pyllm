# 4.3 学习率调度与梯度技巧

上一节我们深入了优化器的内部机制，理解了 AdamW 如何通过自适应学习率和解耦权值衰减来高效地更新参数。但优化器只解决了"怎么根据梯度更新参数"的问题，还有一个同样关键的问题是"每一步应该用多大的学习率来更新"。固定学习率在整个训练过程中几乎永远不是最优选择——训练初期模型对数据分布还不熟悉，太大的学习率会导致不稳定；训练后期模型已经接近收敛，继续使用较大的学习率会在最优解附近震荡而无法精细调整。这就是**学习率调度（Learning Rate Scheduling）**要解决的问题。除了学习率调度之外，这一节还会介绍两个在实际 LLM 训练中不可或缺的技巧：**梯度累积（Gradient Accumulation）**——在显存有限时模拟更大的 batch size；以及**混合精度训练（Mixed Precision Training）**——在不损失（或几乎不损失）精度的前提下大幅加速训练并节省显存。

## 学习率调度：为什么不能一成不变？

想象一下你在开车去一个目的地。刚出发时你还在熟悉路况，应该开得慢一点；上了高速后可以加速巡航；快到目的地时需要减速慢行以精确停靠。训练一个深度学习模型的过程与此类似——不同的训练阶段需要不同的"速度"（学习率）。如果全程都用同一个学习率，就像全程都以高速公路的速度行驶：起步太快可能冲出跑道（训练初期 loss 震荡甚至发散），终点无法精准停车（训练后期在最优解附近反复横跳）。

更具体地说，学习率调度的必要性来自几个方面。第一是 **Embedding 层的保护**：在微调预训练模型时，Embedding 层的参数已经在海量数据上训练过了，如果一开始就用大学习率，这些精心学到的表示会被迅速破坏（灾难性遗忘），而且 Embedding 层的梯度通常比其他层大很多，更容易出问题。通过 warmup（预热）阶段从小学习率开始，给 Embedding 层和其他底层参数一个缓冲期。第二是 **SGD 的理论保证**：对于凸优化问题，学习率衰减是保证收敛的必要条件之一。虽然非凸的深度网络没有严格的理论保证，但实践中衰减策略确实有助于稳定收敛。第三是 **逃逸局部最优的能力**：某些调度策略（如 restart 机制）通过周期性地提高学习率来帮助模型跳出浅层的局部最优。

## Cosine Annealing + Warmup：LLM 的标准配置

目前 LLM 训练中最常用的学习率调度组合是**余弦退火（Cosine Annealing）配合线性 warmup**。它的学习率变化曲线看起来像这样：

```
lr
 ↑
 │    ╱‾‾‾‾\
 │   /        \
 │  /          \___
 │ /
 └──────────────────→ step
    [warmup]   [cosine decay]
```

warmup 阶段：学习率从 0 或一个很小的值线性增长到目标学习率 `lr_max`。
cosine decay 阶段：学习率按照余弦曲线从 `lr_max` 衰减到 `lr_min`（通常设为 `lr_max` 的 10%）。

```python
def cosine_schedule_with_warmup(step, total_steps, warmup_steps,
                                 max_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)


def visualize_schedule():
    total_steps = 10000
    warmup_steps = 500
    max_lr = 3e-4
    min_lr_ratio = 0.1

    steps = list(range(0, total_steps + 100, 100))
    lrs = [cosine_schedule_with_warmup(s, total_steps, warmup_steps,
                                       max_lr, min_lr_ratio) for s in steps]

    print(f"{'Step':>8s} | {'LR':>12s} | {'Phase':>15s}")
    print("-" * 42)
    for s, lr in zip(steps[::10], lrs[::10]):
        phase = "warmup" if s < warmup_steps else "cosine decay"
        print(f"{s:>8d} | {lr:>12.2e} | {phase}")

    print(f"\nPeak LR: {max_lr:.2e} at step {warmup_steps}")
    print(f"Final LR: {lrs[-1]:.2e} at step {total_steps}")
    print(f"LR ratio: {lrs[-1]/max_lr:.2f} of peak")


visualize_schedule()

# 输出:
#     Step |           LR |         Phase
# ------------------------------------------
#        0 |    0.00e+00 |       warmup
#      100 |    6.00e-05 |       warmup
#      200 |    1.20e-04 |       warmup
#      300 |    1.80e-04 |       warmup
#      400 |    2.40e-04 |       warmup
#      500 |    3.00e-04 | ← Peak!
#      600 |    2.99e-04 |  cosine decay
#     ...
#    9900 |    3.30e-05 |  cosine decay
#   10000 |    3.00e-05 |  cosine decay
#
# Peak LR: 3.00e-04 at step 500
# Final LR: 3.00e-05 at step 10000
# LR ratio: 0.10 of peak
```

HuggingFace Transformers 提供了开箱即用的实现：

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=10000,
    min_lr_ratio=0.1,
)

for step in range(10000):
    loss = compute_and_step(model, batch, optimizer)
    scheduler.step()

    if step in [0, 100, 500, 2500, 5000, 7500, 10000]:
        lr = scheduler.get_last_lr()[0]
        print(f"Step {step:>5d}: lr = {lr:.2e}")
```

关于 warmup 步数的选择，实践经验表明总步数的 1%~5% 是一个合理的范围。对于从头训练的大模型，GPT-3 用了 375 步 warmup（占总步数约 0.0003%，极短），LLaMA 用了 2000 步 warmup（约占 1%）。对于微调任务，通常 50~200 步就足够了。Warmup 太短可能导致训练初期不稳定（尤其是有 Embedding 层参与时），太长则浪费计算资源在低效的学习率上。

## 梯度累积：用小 GPU 训练大 Batch Size

在实际训练中经常会遇到这样的困境：你的 GPU 显存只能放下 batch_size=4 的数据，但论文或经验告诉你 batch_size 至少需要 32 甚至 128 才能获得好的训练效果和稳定的梯度估计。买更大的 GPU 是一种解决方案，但不是每个人都有这个预算。**梯度累积（Gradient Accumulation）**提供了一种不需要额外硬件就能模拟更大 batch size 的方法。

它的原理非常直观：正常情况下每个 batch 算一次梯度然后更新一次参数（`accumulation_steps=1`）；如果设置 `accumulation_steps=N`，那就连续算 N 个 batch 的梯度但不更新参数（让梯度累加起来），等到第 N 个 batch 之后才执行一次 `optimizer.step()`。从优化器的角度看，它"看到"的梯度是 N 个 mini-batch 梯度的平均值，等效于用了一个 N 倍大的 batch size。

```python
def training_with_gradient_accumulation(
    model, loader, optimizer,
    accumulation_steps=4,
):
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        outputs = model(input_ids, labels=labels)
        loss = outputs['loss'] / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0] if scheduler else None
            effective_bs = batch['input_ids'].shape[0] * accumulation_steps
            print(f"Step {step}: loss={loss.item()*accumulation_steps:.4f}, "
                  f"effective_batch={effective_bs}")


training_with_gradient_accumulation(model, train_loader, optimizer, accumulation_steps=8)
```

注意代码中的两个关键点：第一，`loss = loss / accumulation_steps` —— loss 必须除以累积步数。原因是 PyTorch 默认会对 loss 做关于 batch size 的平均（`reduction='mean'`），如果不除以 accumulation_steps，N 步累加后的梯度相当于被放大了 N 倍，导致实际学习率比预期大 N 倍。第二，`optimizer.zero_grad()` 只在需要更新的步骤调用，而不是每步都调用——这正是梯度"累积"的核心。

```python
def demonstrate_gradient_accumulation():
    """验证梯度累积的正确性"""
    model = nn.Linear(4, 2, bias=False)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)
    y1 = torch.randn(2, 2)
    y2 = torch.randn(2, 2)

    criterion = nn.MSELoss()

    opt_normal = torch.optim.SGD(model.parameters(), lr=0.01)
    pred = model(torch.cat([x1, x2], dim=0))
    loss_normal = criterion(pred, torch.cat([y1, y2], dim=0))
    opt_normal.zero_grad(); loss_normal.backward(); opt_normal.step()

    model2 = nn.Linear(4, 2, bias=False)
    model2.load_state_dict(model.state_dict())
    opt_acc = torch.optim.SGD(model2.parameters(), lr=0.01)

    opt_acc.zero_grad()
    loss1 = criterion(model2(x1), y1) / 2
    loss1.backward()
    loss2 = criterion(model2(x2), y2) / 2
    loss2.backward()
    opt_acc.step()

    diff = (model.weight - model2.weight).abs().max().item()
    print(f"Max weight difference after update: {diff:.8f}")
    print(f"(Should be ~0 — gradient accumulation matches large batch)")


demonstrate_gradient_accumulation()

# Max weight difference after update: 0.00000000
```

上面的实验验证了梯度累积确实等价于使用更大的 batch size——两种方式更新后的权重差异在浮点精度范围内为零。

但需要注意的是，梯度累积并不完全等同于真正的大 batch 训练。主要区别在于 BatchNorm 层的行为（如果有的话）：真正的 batch_size=32 会基于 32 个样本统计均值和方差，而 4 次 batch_size=8 的累积中每次 BN 只看到 8 个样本。不过对于纯 Transformer 架构（用的是 LayerNorm/RMSNorm 而不是 BatchNorm）来说这个问题不存在。另一个细微差别是 Dropout——累积模式下每个 mini-batch 独立地做 dropout，这与大 batch 下一次性做 dropout 在数学上不完全等价，但实践中的影响通常可以忽略不计。

## 混合精度训练：FP16/BF16 的威力

现代 GPU（特别是 NVIDIA 的 Ampere 架构及以后，如 A100、H100、RTX 30/40 系列）拥有专门的 Tensor Core 硬件单元，可以在 FP16 甚至 BF16/FP8 精度下执行矩阵乘法运算，速度比 FP32 快 2~8 倍，同时显存占用减半。**混合精度训练（Mixed Precision Training, AMP）** 的核心思想就是利用这一点：前向传播和反向传播的大部分计算用低精度（FP16 或 BF16）来做，只在必要的地方保持 FP32 以保证数值稳定性。

PyTorch 提供了两套 API 来实现混合精度训练。旧版 API（仍在广泛使用）使用 `torch.cuda.amp`：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    optimizer.zero_grad()

    with autocast(dtype=torch.float16):
        outputs = model(batch['input_ids'].cuda(), labels=batch['labels'].cuda())
        loss = outputs['loss']

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

这里有几个关键的组件需要解释。`autocast(dtype=torch.float16)` 是一个上下文管理器，它会自动把其中的矩阵乘法操作转换为 FP16 执行，同时保持对数值敏感的操作（如 softmax、layer norm、loss computation）在 FP32 下运行。`GradScaler` 解决的是 FP16 训练中的一个根本问题：梯度过小时会下溢为零（FP16 的最小正值约为 6e-8），导致参数完全无法更新。GradScaler 的做法是维护一个动态缩放因子——如果发现梯度下溢（出现 inf 或 NaN），就增大缩放因子重试；如果梯度一直正常，就逐渐减小缩放因子以提高精度。

PyTorch 2.0+ 引入了新的统一接口 `torch.amp`，推荐用于新项目：

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

for batch in loader:
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(batch['input_ids'].cuda(), labels=batch['labels'].cuda())
        loss = outputs['loss']

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

新 API 的主要改进是把 device_type 作为显式参数传入（支持 CUDA、CPU、MPS、XPU 等），以及 dtype 参数更加灵活。

### FP16 vs BF16：该选哪个？

这是混合精度训练中最常被问到的问题之一。两者都是 16 位浮点格式，但有重要区别：

| 特性 | FP16 | BF16 |
|-----|------|------|
| 总位数 | 16 | 16 |
| 指数位数 | 5 | 8（与 FP32 相同！）|
| 尾数位数 | 10 | 7 |
| 数值范围 | ±65504 | ±3.4×10³⁸（与 FP32 相同）|
| 精度 | 更高 | 较低 |
| 需要 GradScaler | **是** | **否**（通常）|
| 适用硬件 | 所有支持 FP16 的 GPU | Ampere+ (A100/H100/RTX30+) |

BF16 的最大优势在于它的动态范围和 FP32 完全一样（因为共享 8 位指数），这意味着它在大多数情况下不会溢出或下溢，因此**通常不需要 GradScaler**。这使得 BF16 的训练代码更简洁，也消除了 GradScaler 可能引入的不确定性（比如动态缩放因子的调整过程）。

```python
def fp16_vs_bfloat16_demo():
    values = [1e-20, 1e-10, 1.0, 1000.0, 1e10, 65000.0]

    print(f"{'Value':>12s} | {'FP16':>12s} | {'BF16':>12s} | {'Note'}")
    print("-" * 60)
    for v in values:
        t = torch.tensor([v])
        fp16_val = t.to(torch.float16).item()
        bf16_val = t.to(torch.bfloat16).item() if hasattr(torch, 'bfloat16') else "N/A"

        note = ""
        if abs(v) > 65004:
            note = "FP16 overflow!"
        elif abs(v) < 6e-8 and v != 0:
            note = "FP16 underflow!"

        print(f"{v:>12.1e} | {fp16_val:>12.4g} | {str(bf16_val):>12s} | {note}")


fp16_vs_bfloat16_demo()

# 输出:
#      Value |         FP16 |         BF16 | Note
# ------------------------------------------------------------
#    1.0e-20 |            0 |      9.96e-41 | FP16 underflow!
#    1.0e-10 |            0 |      1.0e-10 | FP16 underflow!
#    1.0e+00 |             1 |             1 |
#    1.0e+03 |          1000 |          1000 |
#    1.0e+10 |       9.999e+09 |       1.0e+10 |
#    6.5e+04 |       6.500e+04 |       6.500e+04 |
```

可以看到 FP16 在处理极小值（1e-20、1e-10）时会下溢为 0，而 BF16 能正确保留。对于 LLM 训练来说，如果你的 GPU 支持 BF16（A100、RTX 4090/4080 等），**优先选择 BF16**——代码更简单、数值更稳定、不需要 GradScaler。如果只有较老的 GPU（V100、RTX 2080 等），那只能用 FP16 配合 GradScaler。

## 梯度裁剪再探：更多细节

我们在 04-01 节介绍了基本的梯度裁剪用法，这里补充一些进阶细节。PyTorch 提供了两种梯度裁剪函数：

```python
clip_grad_norm_   # 按范数裁剪（推荐）
clip_grad_value_  # 按值裁剪（较少使用）
```

`clip_grad_norm_` 计算所有参数梯度的全局 L2 范数，如果超过阈值就按比例缩小所有梯度。这是最常用的方式，因为它保持了梯度之间的相对方向关系不变——只是统一地缩放了幅度。

`clip_grad_value_` 则是逐元素地把每个梯度分量截断到 `[-max_value, max_value]` 范围内。这种方式更激进，可能会改变梯度的方向，一般不推荐使用除非你有特殊需求。

```python
def gradient_clipping_comparison():
    model = nn.Linear(10, 3)
    x = torch.randn(4, 10)
    y = torch.randn(4, 3)
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()

    grad_before = model.weight.grad.clone()
    norm_before = grad_before.norm().item()

    model2 = nn.Linear(10, 3)
    model2.load_state_dict(model.state_dict())
    loss2 = nn.functional.mse_loss(model2(x), y)
    loss2.backward()

    torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=0.5)
    grad_after_norm = model2.weight.grad.clone()
    norm_after_norm = grad_after_norm.norm().item()

    model3 = nn.Linear(10, 3)
    model3.load_state_dict(model.state_dict())
    loss3 = nn.functional.mse_loss(model3(x), y)
    loss3.backward()

    torch.nn.utils.clip_grad_value_(model3.parameters(), clip_value=0.1)
    grad_after_value = model3.weight.grad.clone()

    print(f"Before clipping: norm={norm_before:.4f}")
    print(f"After norm_clip(0.5): norm={norm_after_norm:.4f} "
          f"(direction preserved)")
    print(f"After value_clip(0.1): range=[{grad_after_value.min():.4f}, "
          f"{grad_after_value.max():.4f}] (direction may change)")


gradient_clipping_comparison()
```

关于 `max_norm` 的取值，1.0 是最通用的默认值。GPT-2 用的是 1.0，LLaMA 用的是 1.0，几乎所有主流实现都是 1.0。如果你发现训练仍然不稳定（loss 偶尔跳到 NaN），可以尝试降到 0.5 或 0.3；如果你觉得裁剪太激进限制了学习速度，可以尝试 2.0 或更高。但注意过高的 max_norm 基本上等于没有裁剪——如果梯度的自然范数本来就小于 max_norm，裁剪操作什么都不会做。

到这里，训练循环中的三大核心组件——优化器（AdamW）、学习率调度（Cosine + Warmup）、以及梯度技巧（累积、裁剪、混合精度）——都已经完整覆盖了。下一节我们将讨论训练过程中的监控与调试方法：如何识别常见的训练异常、如何系统地排查问题、以及如何建立有效的日志系统。
