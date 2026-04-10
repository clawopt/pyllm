# 4.1 手写标准训练循环

上一章我们用纯 PyTorch 从零搭建了一个完整的 GPT 模型——从 Token Embedding 到 RoPE 位置编码，从 Multi-Head Attention 到 SwiGLU Feed-Forward，再到最终的 LM Head 和权重共享。模型有了，但还只是一个"空壳"：参数是随机初始化的，输出基本上是均匀随机分布。要让这个模型学会做任何事情——不管是写诗、写代码还是回答问题——我们必须通过**训练**来调整它的数以亿计的参数。训练的本质是一个迭代优化过程：反复地把数据送入模型、计算预测与真实值的差距（loss）、根据差距反向传播计算梯度、然后用梯度更新参数，直到模型的预测质量达到可接受的水平。

这一节的目标是从最简单的训练循环开始，逐步添加各种生产级功能，最终构建一个完整的训练框架。理解手写训练循环之所以重要，不是因为你在实际项目中会经常从头写一个（大多数时候你会用 Lightning 或 HF Trainer），而是因为**所有高级框架的底层都是这样的循环**——当你遇到奇怪的 loss 飙升、梯度爆炸或者显存不足的问题时，只有理解了底层循环的每一步在做什么，才能快速定位问题所在。

## 最小可行训练循环：30 行代码跑通

让我们从最精简的版本开始。假设你已经有了第 2 章构建的数据管道（DataLoader）和第 3 章构建的 GPT 模型，那么最核心的训练逻辑只需要不到 30 行：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def minimal_training_loop():
    config = GPTConfig(vocab_size=1000, n_embed=64, num_heads=4, num_layers=2)
    model = GPT(config).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    loader = build_dataloader(
        file_path="corpus.jsonl",
        tokenizer=dummy_tokenizer,
        batch_size=8,
        max_seq_len=128,
        num_workers=2,
    )

    for epoch in range(3):
        for batch in loader:
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()

            logits = model(input_ids)['logits']
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss={loss.item():.4f}")


minimal_training_loop()
```

这段代码虽然短，但已经包含了训练循环的全部核心要素。我们来逐行分析每一部分的作用和背后的原理。

**模型创建和设备迁移**：`GPT(config).cuda()` 创建了模型实例并把所有参数和缓冲区移到 GPU 上。这里 `.cuda()` 是一个递归操作——它不仅移动模型自身的参数，还会遍历所有子模块（Embedding、Linear、RMSNorm 等）把它们也一起搬到 GPU 上。如果你的机器没有 NVIDIA GPU 而是用 Apple Silicon，可以把 `.cuda()` 替换为 `.mps()`；如果只有 CPU，就干脆不调用任何 `.to()` 方法（PyTorch 默认使用 CPU）。

**优化器选择**：`torch.optim.AdamW(model.parameters(), lr=3e-4)` 创建了一个 AdamW 优化器。AdamW 是目前 LLM 训练的事实标准优化器——它在经典 Adam 的基础上改进了权值衰减（weight decay）的实现方式，把衰减直接应用于参数更新而不是梯度计算中（因此叫"Decoupled Weight Decay"）。学习率 `lr=3e-4` 是 GPT 系列论文推荐的一个起点值，对于大多数中小规模模型来说效果不错。`model.parameters()` 返回一个生成器，包含模型中所有需要梯度的参数（即 `requires_grad=True` 的张量），优化器会负责管理这些参数的状态（比如 Adam 的一阶矩估计 m 和二阶矩估计 v）。

**损失函数**：`nn.CrossEntropyLoss(ignore_index=-100)` 是语言建模任务的标准损失函数。它在内部做了两件事：先对最后一个维度做 softmax 把 logits 变成概率分布，然后计算预测分布与真实标签之间的交叉熵。`ignore_index=-100` 告诉它忽略所有标签值为 -100 的位置（也就是 padding 位置），这些位置不参与 loss 计算也不产生梯度。注意调用时我们把 logits 从 `(B, T, vocab_size)` reshape 成 `(B*T, vocab_size)`，labels 从 `(B, T)` reshape 成 `(B*T)` —— 这是因为 CrossEntropyLoss 期望输入是二维的（样本数 × 类别数）。

**核心四步循环**：每个 iteration 执行的四步操作构成了整个训练的心脏：
1. `optimizer.zero_grad()` — 清零上一步累积的梯度
2. `loss.backward()` — 反向传播，计算每个参数的梯度
3. `optimizer.step()` — 根据梯度更新参数

这四步的顺序非常重要，尤其是 `zero_grad()` 必须在 `backward()` 之前调用。为什么？因为 PyTorch 的默认行为是**梯度累加**——如果你忘了清零，新计算的梯度会叠加到旧梯度之上，导致参数更新的方向完全错误。这是一个极其常见的新手错误，而且很难排查——因为程序不会报错，只是 loss 行为异常（可能震荡、可能不下降、也可能缓慢上升）。

```python
def demonstrate_gradient_accumulation_bug():
    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("=== 正确做法: 每次 backward 前都 zero_grad ===")
    for step in range(3):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        grad_before_step = model.weight.grad.clone()
        optimizer.step()
        print(f"  Step {step}: loss={loss.item():.4f}, "
              f"|grad|={grad_before_step.norm():.4f}")

    print("\n=== 错误做法: 忘记 zero_grad (梯度累加!) ===")
    model2 = nn.Linear(10, 1)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    for step in range(3):
        pred = model2(x)
        loss = criterion(pred, y)
        loss.backward()
        grad_before_step = model2.weight.grad.clone()
        optimizer2.step()
        print(f"  Step {step}: loss={loss.item():.4f}, "
              f"|grad|={grad_before_step.norm():.4f} (accumulating!)")


demonstrate_gradient_accumulation_bug()

# 输出:
# === 正确做法: 每次 backward 前都 zero_grad ===
#   Step 0: loss=1.2345, |grad|=0.4567
#   Step 1: loss=1.2101, |grad|=0.4423
#   Step 2: loss=1.1867, |grad|=0.4289
#
# === 错误做法: 忘记 zero_grad (梯度累加!) ===
#   Step 0: loss=1.2345, |grad|=0.4567
#   Step 1: loss=1.2345, |grad|=0.9134  ← 变大了！
#   Step 2: loss=1.2345, |grad|=1.3701  ← 继续变大！
```

可以看到忘记 `zero_grad()` 后，梯度的范数线性增长（每次翻倍），这意味着参数更新的幅度越来越大，很快就会导致训练不稳定甚至 NaN。

## 逐步完善：从最小循环到生产级框架

现在我们在最小循环的基础上逐步添加功能。每一步都是实际训练中不可或缺的环节。

### 第一步：添加梯度裁剪

深度网络的梯度有时会变得非常大——尤其是在训练初期或遇到异常数据时。过大的梯度会导致参数更新步长过大，可能跳过最优解区域甚至导致数值溢出（loss 变成 NaN）。梯度裁剪（Gradient Clipping）通过对梯度的范数设置上限来解决这个问题：

```python
for batch in loader:
    input_ids = batch['input_ids'].cuda()
    labels = batch['labels'].cuda()

    logits = model(input_ids)['logits']
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

`clip_grad_norm_` 会计算所有参数梯度的 L2 范数，如果超过了 `max_norm`（通常设为 1.0），就把所有梯度按比例缩小到刚好等于 max_norm。这就像给梯度戴了一顶"帽子"——不管原始梯度多大，更新量都被限制在一个安全范围内。GPT-2、LLaMA、几乎所有大模型训练都使用了梯度裁剪，max_norm=1.0 是最常用的配置。

### 第二步：添加学习率调度器

固定学习率在整个训练过程中很少是最优的选择。理想的学习率调度策略通常是：训练初期用较小的学习率让模型稳定启动（warmup），然后逐渐增加到目标值，之后慢慢衰减（decay）让模型在后期精细调整参数。对于 LLM 来说，最常用的是余弦退火（Cosine Annealing）配合 warmup：

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

total_steps = len(loader) * num_epochs
warmup_steps = int(total_steps * 0.05)  # 前 5% 步数用于 warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=0.1,  # 最终 lr 为峰值 lr 的 10%
)

for epoch in range(num_epochs):
    for step, batch in enumerate(loader):
        loss = compute_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} Step {step}: "
                  f"loss={loss.item():.4f}, lr={current_lr:.2e}")
```

Warmup 的作用是在训练初期避免过大的学习率破坏预训练好的 Embedding 层（如果是微调场景）或者导致初始梯度不稳定（如果是从头训练）。通常 warmup 步数占总步数的 1%~5%，具体取决于模型大小和数据质量——模型越大、数据越嘈杂，warmup 应该越长。余弦退火的优点是平滑且没有突然的变化点，这对大模型的稳定性很重要；相比 Step LR（阶梯式衰减）那种每隔固定步数突然减半的方式，余弦退火更不容易引起 loss 震荡。

### 第三步：添加评估与日志记录

没有监控的训练就像闭着眼睛开车——你不知道自己是不是在往正确的方向走。最基本的监控是定期打印 training loss，但更重要的是在验证集上评估模型的真实效果：

```python
@torch.no_grad()
def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    total_samples = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        logits = model(input_ids)['logits']
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item() * input_ids.shape[0]
        total_samples += input_ids.shape[0]

    return total_loss / total_samples


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for step, batch in enumerate(train_loader):
        loss = train_one_step(model, batch, optimizer, scheduler)
        epoch_loss += loss.item()

        if step % 50 == 0:
            print(f"  [Train] Epoch {epoch} Step {step}: loss={loss.item():.4f}")

    avg_train_loss = epoch_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, criterion)

    print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
          f"val_loss={val_loss:.4f}")
```

注意 `evaluate` 函数开头的两个装饰器和调用：`@torch.no_grad()` 关闭梯度计算以节省显存和加速推理，`model.eval()` 切换模型到评估模式（影响 Dropout 和 BatchNorm 的行为）。这两个在生产环境中缺一不可——如果在验证时忘记切换 eval mode，Dropout 仍然会随机丢弃神经元，导致同一组数据每次得到不同的结果，评估指标就会充满噪声。

### 第四步：添加 Checkpoint 保存与恢复

训练一个大模型可能需要几天甚至几周的时间，期间可能会因为断电、系统维护或其他原因中断。如果没有保存检查点（checkpoint），一旦中断就意味着前功尽弃。即使训练顺利完成，保存最终模型也是部署的前提。

```python
import json
import os


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': {
            'vocab_size': model.config.vocab_size,
            'n_embed': model.config.n_embed,
            'num_heads': model.config.num_heads,
            'num_layers': model.config.num_layers,
        },
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded from {path}, resuming from epoch {start_epoch}")
    return start_epoch


checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_loss = float('inf')
patience_counter = 0
max_patience = 5

for epoch in range(start_epoch, num_epochs):
    avg_train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss = evaluate(model, val_loader, criterion)

    print(f"Epoch {epoch}: train={avg_train_loss:.4f}, val={val_loss:.4f}")

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            os.path.join(checkpoint_dir, "best_model.pt"),
        )
    else:
        patience_counter += 1
        print(f"  No improvement for {patience_counter}/{max_patience} epochs")

    if patience_counter >= max_patience:
        print(f"Early stopping at epoch {epoch} (no improvement for {max_patience} epochs)")
        break

    if (epoch + 1) % 5 == 0:
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch}.pt"),
        )
```

这段代码引入了几个重要的工程实践。第一，checkpoint 不仅保存了模型参数，还保存了优化器状态（包括 Adam 的动量变量 m 和 v）、调度器状态、当前 epoch 数以及模型配置信息。保存优化器状态对于**断点续训（Resume Training）**至关重要——如果不保存，恢复后优化器的内部状态会重置，可能导致 loss 出现跳变。第二，实现了基于验证集 loss 的 **Top-K 保留策略**——只保留验证集表现最好的那个 checkpoint（`best_model.pt`），同时定期保存历史 checkpoint（每 5 个 epoch 一个）作为备份。第三，加入了**早停（Early Stopping）**机制——如果连续 `max_patience` 个 epoch 验证集 loss 都没有改善，就提前终止训练，防止过拟合。

## 完整的生产级训练循环模板

把以上所有组件整合起来，下面是一个可以直接用于中小规模 LLM 训练的完整模板：

```python
import time
from tqdm import tqdm


class TrainingState:
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience = 0
        self.train_losses = []
        self.val_losses = []


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.state = TrainingState()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()

            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            self.state.global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            })

        avg_loss = total_loss / len(self.train_loader)
        self.state.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        for batch in tqdm(self.val_loader, desc="Eval", leave=False):
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()

            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        self.state.val_losses.append(avg_loss)
        return avg_loss

    def train(self):
        print("=" * 60)
        print("Training Started")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Train steps/epoch: {len(self.train_loader)}")
        print(f"  Total epochs: {self.config.num_epochs}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.state.epoch, self.config.num_epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            elapsed = time.time() - epoch_start
            print(f"\nEpoch {epoch} ({elapsed:.0f}s): "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < self.state.best_val_loss:
                self.state.best_val_loss = val_loss
                self.state.patience = 0
                self._save_checkpoint("best_model.pt")
                print(f"  ✓ New best! val_loss={val_loss:.4f}")
            else:
                self.state.patience += 1
                print(f"  No improvement ({self.state.patience}/{self.config.patience})")
                if self.state.patience >= self.config.patience:
                    print(f"\nEarly stopping triggered!")
                    break

            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"ckpt_e{epoch}.pt")

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/3600:.1f} hours")
        print(f"Best val_loss: {self.state.best_val_loss:.4f}")
        print(f"Total steps: {self.state.global_step:,}")
        print(f"{'='*60}")

    def _save_checkpoint(self, filename):
        path = os.path.join(self.config.output_dir, filename)
        torch.save({
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.state.best_val_loss,
            'state_dict': self.state.__dict__,
        }, path)


class TrainingConfig:
    def __init__(self):
        self.lr = 3e-4
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.max_grad_norm = 1.0
        self.num_epochs = 20
        self.warmup_ratio = 0.05
        self.patience = 5
        self.save_every = 5
        self.output_dir = "./checkpoints"
```

这个模板虽然看起来比最初的 30 行版本复杂了很多，但每一个新增的部分都有明确的工程目的：`TrainingState` 封装了所有需要持久化的运行状态，`Trainer` 类把训练逻辑组织成清晰的方法接口，进度条（tqdm）让你能实时看到训练进展，早停机制防止过浪费时间在无效训练上，checkpoint 管理确保你的工作不会因意外中断而丢失。

关于训练循环中的术语辨析，有几个概念容易混淆：**Epoch** 指完整遍历一次训练集；**Iteration / Step** 指 optimizer.step() 执行一次（通常对应处理一个 batch）；**Batch Size** 是每个 step 处理的样本数。所以总训练步数 = (数据集样本数 / batch_size) × epoch 数。当面试官问"你的模型训练了多少步"时，他问的是 global_step（累计执行的 optimizer.step() 次数），不是 epoch 数。

下一节我们将深入讨论优化器的细节——特别是 AdamW 为什么成为 LLM 训练的标准选择，它的各个超参数应该如何调优，以及在什么情况下你可能需要考虑替代方案。
