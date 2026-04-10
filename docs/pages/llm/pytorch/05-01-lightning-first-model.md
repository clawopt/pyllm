# 5.1 Lightning 核心哲学与第一个模型

回顾一下我们在第 4 章做的事情：写了一个完整的训练循环，从最简单的 30 行代码开始，逐步添加了梯度裁剪、学习率调度、混合精度训练、评估循环、日志记录、checkpoint 管理、早停机制……最终组装成了一个功能齐全的 `Trainer` 类。这个类大概有 150~200 行代码，而且我们还没包括分布式训练、多 GPU 同步、FP16/BF16 自动处理、梯度 checkpointing 这些更高级的功能。如果你把这些都加上去，一个生产级的训练框架轻松就能超过 500 行。问题在于：**这 500 行代码中，真正和你的模型逻辑相关的可能只有 20~30 行（forward、loss 计算），剩下的 470 行都是通用的工程样板代码**。每次开始一个新项目都要重新写一遍这些样板代码不仅浪费时间，更重要的是——它们是 bug 的温床。设备管理写错了？分布式通信顺序不对？半精度转换漏了一层？这些错误往往不会立即报错，而是在训练了几十个小时后才以奇怪的方式显现出来。

PyTorch Lightning 的核心设计理念就是解决这个痛点。它把训练相关的所有工程细节抽象成一个统一的 `Trainer` 类，而你只需要通过继承 `LightningModule` 来定义模型的核心逻辑——前向传播怎么算、loss 怎么定义、优化器用什么。其他一切：设备管理（`.cuda()` / `.cpu()` / `.mps()`）、分布式训练（DDP/FSDP/DeepSpeed）、半精度（FP16/BF16）、梯度裁剪、学习率调度、checkpoint 保存与恢复、日志记录、早停……全部由 `Trainer` 自动处理。你用 25 行核心代码就能完成之前 200 行才能做到的事情，而且不容易出错。

## 从手写到 Lightning：一次直观的对比

让我们先看一个具体的例子来感受这种差异。下面是我们第 4 章手写的训练循环的核心部分：

```python
model = GPT(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=total_steps
)

for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            outputs = model(input_ids, labels=labels)
            val_losses.append(outputs['loss'].item())
    avg_val_loss = sum(val_losses) / len(val_losses)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pt")
```

同样的功能用 PyTorch Lightning 来实现：

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class LitGPT(pl.LightningModule):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.model = GPT(config)

    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        output = self(batch['input_ids'], batch['labels'])
        self.log('train_loss', output['loss'], prog_bar=True)
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self(batch['input_ids'], batch['labels'])
        self.log('val_loss', output['loss'], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def configure_callbacks(self):
        return [
            ModelCheckpoint(monitor='val_loss', save_top_k=3),
            EarlyStopping(monitor='val_loss', patience=5),
        ]


trainer = pl.Trainer(
    max_epochs=20,
    accelerator="auto",
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    accumulate_grad_batches=4,
)

lit_model = LitGPT(config)
trainer.fit(lit_model, train_loader, val_loader)
```

数一下有效代码行数：`LitGPT` 类大约 30 行，`Trainer` 配置大约 8 行，调用 `fit` 一行。总共不到 40 行就完成了手写版本 80+ 行的功能。但更重要的是**省略了什么**：没有 `.cuda()` 调用（Lightning 自动处理）、没有 `autocast` 上下文管理器（`precision="bf16-mixed"` 一行搞定）、没有 `scaler` 管理（Lightning 内部自动处理）、没有 `model.train()/eval()` 切换（Lightning 在正确的时机自动调用）、没有手动 checkpoint 保存（`ModelCheckpoint` callback 自动处理）、没有早停逻辑（`EarlyStopping` callback 处理）。所有这些工程细节都被封装在了 `Trainer` 的内部实现中，经过了数千个项目的验证，比你自己手写的可靠得多。

## LightningModule vs nn.Module：继承关系与新增能力

理解 Lightning 的第一步是搞清楚 `LightningModule` 和 PyTorch 原生 `nn.Module` 之间的关系。关键事实是：**`LightningModule` 继承自 `nn.Module`**。这意味着你之前学到的关于 `nn.Module` 的一切知识——`parameters()`、`state_dict()`、`.train()/.eval()`、`.to(device)`、子模块注册等——在 `LightningModule` 中完全适用，没有任何破坏性变更。

```python
import pytorch_lightning as pl

class LitGPT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = GPT(config)       # nn.Module 子模块 ✓
        self.save_hyperparameters()   # Lightning 新增

    def forward(self, x):             # 和 nn.Module 完全一样 ✓
        return self.model(x)


lit_model = LitGPT(config)

print(f"Is nn.Module? {isinstance(lit_model, nn.Module)}")
print(f"Parameters count: {sum(p.numel() for p in lit_model.parameters()):,}")
print(f"State dict keys (first 3): {list(lit_model.state_dict().keys())[:3]}")

lit_model.eval()
print(f"Training mode: {lit_model.training}")

hparams = lit_model.hparams
print(f"Hyperparameters saved: {list(hparams.keys())}")
```

除了继承 `nn.Module` 的全部能力之外，`LightningModule` 还新增了几个重要的接口方法：

| 方法 | 触发时机 | 用途 |
|-----|---------|------|
| `training_step(self, batch, batch_idx)` | 每个 training batch | 定义训练逻辑，返回 loss |
| `validation_step(self, batch, batch_idx)` | 每个 validation batch | 定义验证逻辑 |
| `test_step(self, batch, batch_idx)` | 每个 test batch | 定义测试逻辑 |
| `predict_step(self, batch, batch_idx)` | 每个 prediction batch | 定义推理逻辑 |
| `configure_optimizers(self)` | 训练开始时 | 返回优化器和学习率调度器 |
| `configure_callbacks(self)` | 训练开始时 | 返回 callbacks 列表 |

其中 `training_step` 是最核心的方法——它替代了手写循环中的"前向传播 → 计算 loss → 返回 loss"这一整块逻辑。你只需要关心"给定一个 batch，怎么计算 loss"，其余的一切（梯度计算、参数更新、学习率调整、日志记录）由 `Trainer` 接管。

## Trainer 核心参数速览

`pl.Trainer` 是 Lightning 的另一个核心组件——它是整个训练过程的编排者。虽然它的默认值已经能覆盖大多数场景，但了解每个参数的含义有助于你在需要时做出正确的配置选择。

### 基本训练控制

```python
trainer = pl.Trainer(
    max_epochs=20,           # 最大训练轮数
    max_steps=100_000,       # 最大步数（优先于 max_epochs）
    limit_train_batches=100, # 每个 epoch 只用前 100 个 batch（快速调试用）
    limit_val_batches=10,    # 验证集同理
)
```

`limit_train_batches` 和 `limit_val_batches` 在开发调试阶段极其有用——你可以只用数据集的一小部分来快速验证整个流程是否跑通，而不需要等待完整遍历整个数据集。设为 `0.1` 表示使用 10% 的数据，设为整数表示使用指定数量的 batch。

### 加速器与精度

```python
trainer = pl.Trainer(
    accelerator="gpu",          # "cpu", "gpu", "tpu", "mps", "auto"
    devices=1,                  # 使用几张卡；-1 表示全部
    precision="bf16-mixed",     # "32", "16-mixed", "bf16-mixed"
)
```

`accelerator="auto"` 会自动检测可用的硬件——有 CUDA 就用 GPU，有 Apple Silicon 就用 MPS，都没有就用 CPU。这是最推荐的设置，因为它让你的代码在不同机器上无需修改即可运行。

精度选项中 `"32"` 表示纯 FP32（不使用混合精度）；`"16-mixed"` 表示 FP16 混合精度（需要 GradScaler）；`"bf16-mixed"` 表示 BF16 混合精度（不需要 GradScaler，推荐用于 Ampere+ GPU）。

### 梯度相关

```python
trainer = pl.Trainer(
    gradient_clip_val=1.0,         # 梯度裁剪阈值
    gradient_clip_algorithm="norm", # "norm"(默认) 或 "value"
    accumulate_grad_batches=4,      # 梯度累积步数
)
```

注意这里的 `gradient_clip_val` 直接对应我们手写循环中的 `clip_grad_norm_(..., max_norm=1.0)` —— Lightning 会在每次 `optimizer.step()` 之前自动执行裁剪。

### 日志与回调

```python
from pytorch_lightning.loggers import WandbLogger

trainer = pl.Trainer(
    logger=WandbLogger(project="my-gpt"),
    enable_checkpointing=True,
    default_root_dir="./outputs",
)
```

Lightning 支持多种日志后端：TensorBoardLogger（默认）、CSVLogger、WandbLogger、CometLogger 等。切换日志后端只需改一行代码，你的 `self.log()` 调用会自动适配到新的后端。

## 第一个完整的 Lightning 项目

让我们把所有知识整合起来，用 Lightning 重写我们的 GPT 训练项目：

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger


class LitGPT(pl.LightningModule):
    """基于 Lightning 的 GPT 训练模块"""

    def __init__(
        self,
        vocab_size: int = 1000,
        n_embed: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 256,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        config = GPTConfig(
            vocab_size=vocab_size,
            n_embed=n_embed,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.gpt = GPT(config)

    def forward(self, input_ids, labels=None):
        return self.gpt(input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        output = self(batch['input_ids'], batch['labels'])
        loss = output['loss']
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch['input_ids'], batch['labels'])
        loss = output['loss']
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                monitor='val_loss',
                save_top_k=3,
                filename='gpt-{epoch:02d}-{val_loss:.4f}',
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min',
            ),
            LearningRateMonitor(logging_interval='step'),
        ]


def train_with_lightning():
    lit_model = LitGPT(
        vocab_size=1000,
        n_embed=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=256,
        lr=3e-4,
    )

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        logger=WandbLogger(project="lit-gpt-tutorial"),
        enable_checkpointing=True,
        limit_val_batches=20,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    print("\nTraining complete!")
    print(f"Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train_with_lightning()
```

这段代码的运行效果和第 4 章的手写 Trainer 完全等价，但代码量减少了约 75%，且消除了几乎所有可能的工程错误来源。当你运行它时，Lightning 会在终端输出详细的训练进度信息：

```
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
...
Epoch 0: 100%|████████████| 625/625 [01:23<00:00, 7.50it/s, v_num=1, train_loss=3.2145, lr=2.98e-05]
Validating: 100%|████████████| 20/20 [00:02<00:00, 9.87it/s, val_loss=3.1567]
Epoch 1: 100%|████████████| 625/625 [01:21<00:00, 7.69it/s, v_num=1, train_loss=2.8765, lr=8.45e-05]
Validating: 100%|████████████| 20/20 [00:02<00:00, 9.91it/s, val_loss=2.7834]
...
```

每一行都包含了当前 epoch、进度条、速度（it/s）、各种 log 指标和学习率——这些都是免费的，不需要你写一行额外的日志代码。

## 常见误区与注意事项

在使用 Lightning 的过程中，有几个新手容易踩的坑值得提前说明。

**误区一：在 `training_step` 中手动调用 `.cuda()` 或 `.to(device)`**

Lightning 会自动把 batch 数据移动到正确的设备上。如果你再手动调用 `.cuda()`，在单卡训练时不会出错（重复移动到同一设备），但在多卡或 CPU 训练时会报错或产生意外行为。正确做法是在 `training_step` 中直接使用接收到的 batch，假设它已经在正确的设备上了。

**误区二：在 `training_step` 中忘记返回 loss**

`training_step` 必须返回一个标量张量（loss），因为 `Trainer` 需要它来调用 `.backward()`。如果你返回了 None 或者一个字典但没有包含 `'loss'` 键，训练将无法进行。如果除了 loss 还想返回其他信息（比如中间层的输出用于分析），可以用字典形式返回并包含 loss 键：

```python
def training_step(self, batch, batch_idx):
    output = self(batch['input_ids'], batch['labels'])
    return {'loss': output['loss'], 'logits': output['logits']}
```

**误区三：混淆 `on_step` 和 `on_epoch` 参数**

`self.log()` 有两个重要的布尔参数：
- `on_step=True`：每一步都记录（适合 train_loss 这种每步都在变化的指标）
- `on_epoch=True`：每个 epoch 结束时自动计算平均值并记录（适合 val_loss 这种需要聚合的指标）

对于训练指标通常同时设置两个（`on_step=True, on_epoch=True`）；对于验证指标通常只设置 `on_epoch=True`（避免日志过于频繁）。

**误区四：`configure_optimizers` 中返回格式错误**

这是最常见的 Lightning 报错之一。`configure_optimizers` 支持多种返回格式，最常用的两种是：

```python
def configure_optimizers(self):
    opt = torch.optim.AdamW(self.parameters(), lr=3e-4)
    sched = get_cosine_schedule_with_warmup(opt, ...)
    
    # 格式一：简化版（只有优化器）
    return opt
    
    # 格式二：完整版（优化器 + 调度器）
    return {
        'optimizer': opt,
        'lr_scheduler': {
            'scheduler': sched,
            'interval': 'step',     # 'step' 或 'epoch'
            'frequency': 1,
        },
    }
```

注意调度器的 `interval` 参数——大多数 LLM 训练应该用 `'step'`（每步更新学习率），而不是默认的 `'epoch'`（每 epoch 更新一次）。

到这里，你已经掌握了 Lightning 的基本用法。下一节我们将深入 Lightning 的生命周期系统——了解 `Trainer` 在训练过程中究竟按什么顺序调用了哪些钩子方法，以及如何利用 Callback 系统来实现自定义的训练行为。
