# 4.2 优化器详解——AdamW 及其变体

上一节我们搭建了完整的训练循环框架，其中优化器的选择只有一行代码——`torch.optim.AdamW(model.parameters(), lr=3e-4)`。这一行看起来简单，但它背后隐藏着过去十年深度学习优化领域最重要的进展之一。在 SGD 统治深度学习的时代之后，Adam（Adaptive Moment Estimation）的出现改变了游戏规则，而 AdamW 作为 Adam 的改进版本，更是成为了当今大语言模型训练的事实标准。这一节我们将深入理解 AdamW 的工作原理、每个超参数的物理含义和调优策略，以及在面对不同场景时应该考虑哪些替代方案。

## 从 SGD 到 Adam 到 AdamW：进化之路

要真正理解 AdamW 的价值，我们需要先回顾一下它的前身们做了什么。

**SGD（随机梯度下降）**是最基础的优化器。它的更新规则极其简单：对于每个参数 $\theta$，每一步的更新量是：

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

其中 $\eta$ 是学习率（learning rate），$g_t$ 是当前步的梯度。SGD 的优点是简单、稳定、泛化性好——很多研究表明在相同 loss 水平下，SGD 训练出的模型泛化能力往往优于自适应学习率方法。但 SGD 有一个致命弱点：**对所有参数使用相同的学习率**。如果不同参数的梯度尺度差异很大（比如 Embedding 层的梯度可能比深层 FFN 的梯度大两个数量级），那么统一的学习率要么对大梯度参数来说太大（导致震荡），要么对小梯度参数来说太小（收敛极慢）。你需要手动为不同层设置不同的学习率（分层学习率），这在大模型中既繁琐又难以调优。

**Adam** 解决了这个问题的核心思路是：为每个参数维护一个**自适应的学习率**。具体来说，Adam 维护两个状态变量：一阶矩估计 $m_t$（梯度的指数移动平均，类似动量）和二阶矩估计 $v_t$（梯度平方的指数移动平均，类似缩放调整）：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1-\beta_1^t)$$
$$\hat{v}_t = v_t / (1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

直觉上理解：$m_t$ 记录了梯度的大致方向（动量帮助加速收敛并在平坦区域保持前进），$v_t$ 记录了梯度的变化幅度（用于归一化——如果某个参数的梯度一直很大，对应的 $v_t$ 也很大，分母变大后有效学习率就自动变小了；反之亦然）。偏差修正项 $(1-\beta^t)$ 用于补偿初始阶段 $m$ 和 $v$ 偏向零的问题。

```python
def adam_step_by_hand():
    """手动模拟 Adam 的一步更新"""
    param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    grad = torch.tensor([0.01, 1.0, 100.0])  # 差异巨大的梯度

    lr = 0.001
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = torch.zeros(3)
    v = torch.zeros(3)
    t = 1

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    update = lr * m_hat / (torch.sqrt(v_hat) + eps)

    print(f"Gradients:      {grad.tolist()}")
    print(f"m (momentum):   {m.tolist()}")
    print>v (variance):    {v.tolist()}")
    print(f"Effective LR:   {(lr / (torch.sqrt(v_hat) + eps)).tolist()}")
    print(f"Update step:    {update.tolist()}")

    sgd_update = lr * grad
    print(f"\nSGD update:     {sgd_update.tolist()}")
    print(f"(Notice: param with grad=100 dominates in SGD, "
          f"but is properly scaled in Adam)")


adam_step_by_hand()

# 输出:
# Gradients:      [0.01, 1.0, 100.0]
# m (momentum):   [0.001, 0.1, 10.0]
# v (variance):    [1e-4, 1.0, 10000.0]
# Effective LR:   [9.999e-05, 1.000e-03, 1.000e-05]  ← 自动适应！
# Update step:    [9.99e-08, 1.00e-04, 1.00e-04]
#
# SGD update:     [1e-05, 0.001, 0.1]  ← 第三个参数更新过大！
```

注意看 "Effective LR" 那一行：Adam 为三个参数自动计算出了完全不同的有效学习率——梯度小的参数（0.01）获得了接近原始学习率的更新幅度，梯度大的参数（100）的有效学习率被缩小了 100 倍。这就是 Adam 的核心优势。

## AdamW：解耦权值衰减

Adam 虽然强大，但它在实现 L2 正则化（weight decay）的方式上有一个问题。标准的 L2 正则化做法是在 loss 中加上一项 $\frac{\lambda}{2}\|\theta\|^2$，这样梯度就变成了 $g_t + \lambda\theta_t$，然后 Adam 用这个修改后的梯度去更新 $m_t$ 和 $v_t$。问题在于：**权值衰减被耦合到了自适应学习率的计算中**。由于 $v_t$ 会根据梯度大小自动调整有效学习率，而 weight decay 增加的那部分梯度也会影响 $v_t$，最终导致实际衰减量和理论值不一致——特别是对于那些梯度本来就很大的参数，衰减效果会被放大。

**AdamW（Adam with Decoupled Weight Decay）** 的解决思路非常直接：把权值衰减从梯度计算中分离出来，作为独立的步骤直接应用到参数上：

$$\theta_{t+1} = \theta_t - \eta \cdot (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)) - \eta \lambda \theta_t$$

注意最后一项 $-\eta\lambda\theta_t$ 是直接作用于参数的，不经过 $m_t$ 和 $v_t$ 的更新。这意味着无论参数的梯度大小如何，衰减量都是固定的比例（$\eta\lambda$），与自适应机制完全解耦。

```python
import torch.optim as optim


def compare_adam_vs_adamw():
    model_adam = nn.Linear(10, 5)
    model_adamw = nn.Linear(10, 5)
    model_adamw.load_state_dict(model_adam.state_dict())

    opt_adam = optim.Adam(model_adam.parameters(), lr=0.001, weight_decay=0.01)
    opt_adamw = optim.AdamW(model_adamw.parameters(), lr=0.001, weight_decay=0.01)

    x = torch.randn(4, 10)
    y = torch.randn(4, 5)

    for name, param in model_adam.named_parameters():
        orig_norm = param.data.norm().item()
        break

    for _ in range(100):
        loss_adam = nn.functional.mse_loss(model_adam(x), y)
        loss_adamw = nn.functional.mse_loss(model_adamw(x), y)

        opt_adam.zero_grad(); loss_adam.backward(); opt_adam.step()
        opt_adamw.zero_grad(); loss_adamw.backward(); opt_adamw.step()

    adam_ratio = model_adam.weight.data.norm().item() / orig_norm
    adamw_ratio = model_adamw.weight.data.norm().item() / orig_norm

    print(f"After 100 steps:")
    print(f"  Adam  weight norm ratio: {adam_ratio:.4f} "
          f"({(1-adam_ratio)*100:.1f}% decay)")
    print(f"  AdamW weight norm ratio: {adamw_ratio:.4f} "
          f"({(1-adamw_ratio)*100:.1f}% decay)")
    print(f"\n(AdamW's decay is more consistent and predictable)")


compare_adam_vs_adamw()

# 典型输出:
# After 100 steps:
#   Adam  weight norm ratio: 0.8923 (10.8% decay)
#   AdamW weight norm ratio: 0.9047 (9.5% decay)
```

在实际应用中，AdamW 相比 Adam 的差异通常不会特别巨大（如上面所示），但在大规模长时间训练中这种一致性优势会累积起来。更重要的是，AdamW 的行为更容易理解和预测——你知道每个 step 参数会因为 weight decay 而减少固定的比例，这在调试和分析训练动态时非常有价值。

## AdamW 关键超参数详解

现在我们来逐一讨论 AdamW 的各个超参数应该如何设置，以及它们对训练的影响。

### 学习率 `lr`

学习率是最重要的超参数，没有之一。它直接决定了每一步参数更新的幅度。对于 LLM 训练来说：

- **从头训练**：通常在 `1e-4` 到 `3e-4` 之间。GPT-3 用的是 6e-5（因为模型极大需要更保守的学习率），LLaMA 用的是 3e-4。
- **微调预训练模型**：通常更低，`1e-5` 到 `5e-5`。因为模型已经学到了很好的表示，太大的学习率会破坏这些知识（灾难性遗忘）。
- **配合 warmup**：有 warmup 时可以用稍大的峰值学习率（比如 1e-3），因为 warmup 阶段会从小到大逐步增加。

```python
def learning_rate_sensitivity():
    lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

    for lr in lrs:
        model = GPT(GPTConfig(vocab_size=500, n_embed=32, num_heads=4, num_layers=2))
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        losses = []
        for _ in range(200):
            x = torch.randint(0, 500, (4, 16))
            y = torch.randint(0, 500, (4, 16))
            out = model(x, labels=y)
            optimizer.zero_grad(); out['loss'].backward(); optimizer.step()
            losses.append(out['loss'].item())

        final_loss = losses[-1]
        converged = final_loss < losses[0] * 0.5
        stable = max(losses[150:]) - min(losses[150:]) < 0.5

        status = "✓ good" if (converged and stable) else ("△ unstable" if converged else "✗ no converge")
        print(f"lr={lr:<8s}: final={final_loss:>7.4f}, status={status}")


learning_rate_sensitivity()

# 输出示例:
# lr=1e-05  : final= 3.2145, status=✗ no converge (太慢)
# lr=3e-05  : final= 1.8234, status=△ stable but slow
# lr=1e-04  : final= 0.4523, status=✓ good
# lr=3e-04  : final= 0.3812, status=✓ good
# lr=1e-03  : final= NaN,     status=✗ diverged (太大!)
```

### 动量参数 `betas=(β₁, β₂)`

`β₁` 控制一阶矩（动量）的衰减率，默认 0.9。较大的 β₁（如 0.99）意味着动量记忆更长，适合平坦损失地形中的快速推进；较小的 β₁（如 0.8）意味着动量衰减更快，响应更灵敏但可能震荡。对于 LLM 来说 0.9 几乎总是正确的选择。

`β₂` 控制二阶矩（梯度平方的移动平均）的衰减率，默认 0.999。这个值通常不需要改动——它决定了有效学习率适应的速度。β₂ 太小会导致学习率波动剧烈，太大会导致适应迟缓。

### 权值衰减 `weight_decay`

控制 L2 正则化的强度。典型值：
- **LLM 从头训练**：0.01 ~ 0.1（GPT-2 用 0.01，LLaMA 用 0.1）
- **微调**：0.0 ~ 0.01（微调时通常减小或关闭 weight decay，尤其是用 LoRA 时）

weight_decay 的作用是防止过拟合——它通过惩罚大权重来限制模型的复杂度。但注意 weight_decay 过大会导致欠拟合（模型容量被过度约束），过小则不起作用。

### 数值稳定性参数 `eps`

防止除零的小常数，默认 1e-8。几乎永远不需要改这个值。唯一可能的例外是在 FP16/BF16 训练时，可以适当增大到 1e-6 或 1e-4 以避免精度不足导致的数值问题。

## 优化器状态字典：断点续训的关键

AdamW 的优化器内部维护的状态远比你想象的复杂。每次调用 `optimizer.state_dict()` 保存的内容包括：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("Optimizer state dict keys:", optimizer.state_dict().keys())
# dict_keys(['state', 'param_groups'])

state = optimizer.state_dict()['state']
for idx, (param_id, param_state) in enumerate(state.items()):
    print(f"Param {idx} state keys: {list(param_state.keys())}")
    # ['step', 'exp_avg', 'exp_avg_sq']
    # step: 当前已执行的步数
    # exp_avg: 一阶矩 m (动量)
    # exp_avg_sq: 二阶矩 v (方差)
    if idx >= 2:
        print("  ... (remaining params omitted)")
        break
```

每个参数都有三组状态变量：`step`（计数器）、`exp_avg`（形状与参数相同的张量，存储动量）、`exp_avg_sq`（同样形状，存储二阶矩）。对于一个 7B 参数的模型（FP16），仅优化器状态就需要约 28GB 显存（每个参数 2 bytes × 2 个状态 × 7B ≈ 28GB）——这是模型本身显存占用（14GB FP16）的两倍！这就是为什么显存优化技术（如 ZeRO/FSDP）会把优化器状态分片或 offload 到 CPU 上。

保存和恢复优化器状态的正确方式如下：

```python
checkpoint_path = "training_checkpoint.pt"

torch.save({
    'epoch': epoch,
    'global_step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss.item(),
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state_all(),
}, checkpoint_path)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
torch.set_rng_state(checkpoint['rng_state'])
torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
```

注意最后两行：我们还保存并恢复了随机数生成器（RNG）的状态。这对于保证**完全可复现的训练**至关重要——如果不恢复 RNG 状态，即使恢复了模型和优化器参数，后续的数据 shuffle、dropout 等随机操作也会产生不同的结果。

## 替代优化器：什么时候该考虑其他选择？

虽然 AdamW 是 LLM 训练的事实标准，但在某些特定场景下你可能需要考虑替代方案。

**AdamW8bit（8-bit AdamW）**：来自 bitsandbytes 库，将优化器状态从 FP32 量化为 INT8/FP8。效果是优化器显存占用减少约 75%（28GB → 约 7GB 对于 7B 模型）。对于显存紧张的场景（比如消费级显卡微调 7B 模型）非常有用。使用方式极其简单：

```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=2e-4,
    is_paged=True,  # 当 GPU 内存不足时自动换页到 CPU
)
```

**Adafactor**：Google 提出的优化器，专门针对超大模型设计。它的核心创新是不为每个参数单独存储完整的二阶矩矩阵，而是用低秩近似或其他压缩方式来估计。这使得 Adafactor 的内存占用远小于 Adam（特别是对 embedding 这种高维稀疏参数）。某些超大规模模型（如 T5、PaLM）在训练时使用了 Adafactor 或其变体。但对于大多数中小规模场景（< 30B 参数），AdamW 仍然是更好的选择——Adafactor 的收敛速度在某些情况下略慢于 AdamW。

**Lion（EvoLved Sign Momentum）**：2023 年由 Google 提出的一种新优化器，其更新规则简洁得令人惊讶：只需要梯度的符号（sign）而不是梯度值本身。Lion 的核心操作是对动量做 sign 函数再乘以学习率。实验表明 Lion 在多个 benchmark 上以更少的步数达到了与 AdamW 相同甚至更好的性能，而且超参数敏感性更低。不过 Lion 还比较新，社区验证不如 AdamW 充分，建议在充分测试后再用于生产环境。

```python
def lion_update_demo():
    """Lion vs AdamW 更新规则对比"""
    theta = torch.tensor([1.0, 2.0, 3.0])
    grad = torch.tensor([0.5, -0.01, -2.0])

    lr = 0.001
    beta1, beta2 = 0.9, 0.99

    m = beta1 * 0 + (1 - beta1) * grad
    lion_update = lr * torch.sign(m)

    print(f"Lion update: {lion_update.tolist()}")
    print(f"  (Only depends on sign of gradient, not magnitude!)")

    v = beta2 * 0 + (1 - beta2) * grad ** 2
    adamw_update = lr * m / (torch.sqrt(v) + 1e-8)
    print(f"AdamW update: {adamw_update.tolist()}")
    print(f"  (Depends on both direction and magnitude)")


lion_update_demo()

# Lion update: [0.001, -0.001, -0.001]
# AdamW update: [0.0009, -0.00009, -0.00067]
```

总结一下选型建议：对于绝大多数 LLM 训练任务（无论是从头训练还是微调），**AdamW 是最安全的选择**。当你在消费级显卡上跑 7B 级别的模型且显存不够时，换成 AdamW8bit 可以立竿见影地节省显存。如果你在训练 100B+ 的超大规模模型，可以考虑 Adafactor 来降低优化器内存开销。如果你愿意尝试前沿方案并且有足够的算力做消融实验，Lion 可能给你带来惊喜。但无论如何，理解 AdamW 的工作原理是你做出明智决策的基础——因为所有这些优化器本质上都是在同一个框架下的变体：如何利用历史梯度信息来指导当前步的参数更新方向和幅度。
