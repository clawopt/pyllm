# 5.2 Lightning 生命周期与回调系统

上一节我们学会了用 Lightning 的基本 API 把手写训练循环压缩到 40 行以内，体验到了"声明式训练"的便利。但如果你想在更细粒度上控制训练过程——比如在每个 batch 开始前检查一下数据、在反向传播后记录梯度范数、在 epoch 结束时生成一些样本文本——就需要理解 Lightning 的**生命周期（Lifecycle）**和**回调（Callback）**系统了。这两个机制共同构成了 Lightning 的扩展骨架：生命周期定义了训练过程中所有可能的时间节点，回调让你可以在这些时间点上插入自定义逻辑，而无需修改 `LightningModule` 或 `Trainer` 本身的代码。

## 完整生命周期：从 prepare_data 到 teardown

当你调用 `trainer.fit(model, train_loader, val_loader)` 时，`Trainer` 内部会按照一个精心设计的顺序执行一系列操作。理解这个顺序对于编写正确的自定义逻辑至关重要——如果你的代码在错误的时间点被执行，可能会遇到变量未初始化、设备不匹配、梯度计算异常等各种问题。

下面是完整的生命周期流程图：

```
trainer.fit() 被调用
    │
    ▼
┌─ 准备阶段 ─────────────────────────────┐
│  1. on_fit_start(trainer)               │ ← Trainer 级别钩子
│  2. prepare_data()                      │ ← 数据准备（仅 rank 0）
│  3. setup(stage='fit')                  │ ← 最终设置
│                                         │
│  [以下对每个 dataloader 执行:]           │
│  4. setup_data_hook(dataloader)         │
│  5. data_loader_setup(dataloader)       │
│  6. on_pre_dataloader(dataloader)       │
└─────────────────────────────────────────┘
    │
    ▼
┌─ 训练循环 (重复 num_epochs 次) ───────────┐
│                                          │
│  on_train_epoch_start(epoch)             │
│  │                                      │
│  ├─ 对每个 training batch:               │
│  │   on_train_batch_start(batch, ...)     │
│  │   ├── on_before_backward(loss)        │
│  │   │   loss.backward()                 │
│  │   ├── on_after_backward(tensor)       │
│  │   │   optimizer.step()                │
│  │   │   optimizer.zero_grad()           │
│  │   ├── on_train_batch_end(outputs)     │
│  │   │                                  │
│  │   └── training_step(batch, idx)       │ ← 你的代码在这里！
│  │                                      │
│  on_train_epoch_end(outputs)             │
│  │                                      │
│  ├─ 验证阶段:                            │
│  │   on_validation_epoch_start()         │
│  │   ├─ 对每个 validation batch:          │
│  │   │   on_validation_batch_start(...)   │
│  │   │   validation_step(batch, idx)      │ ← 你的验证代码
│  │   │   on_validation_batch_end(...)     │
│  │   on_validation_epoch_end(outputs)    │
│  │                                      │
│  on_validation_end()                     │
│  │                                      │
│  [Checkpoint / Early Stopping 检查]       │
│                                          │
└──────────────────────────────────────────┘
    │
    ▼
on_fit_end(results)
teardown(stage='fit')
```

这个图看起来复杂，但你不需要记住每一个钩子。实际开发中真正常用的只有不到十个。让我们通过一个具体的例子来感受各个钩子的触发时机和用途。

```python
class DebugLifecycleGPT(pl.LightningModule):
    """用于演示生命周期的 GPT 模型"""

    def __init__(self, config):
        super().__init__()
        self.gpt = GPT(config)
        self.epoch_outputs = []

    def on_fit_start(self):
        print("🚀 Training is about to start!")

    def on_train_epoch_start(self):
        self.epoch_outputs = []
        print(f"\n── Epoch {self.current_epoch} starts ──")

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx == 0:
            print(f"  First batch: input shape={batch['input_ids'].shape}")

    def training_step(self, batch, batch_idx):
        output = self.gpt(batch['input_ids'], labels=batch['labels'])
        self.epoch_outputs.append(output['loss'].item())
        return output['loss']

    def on_before_backward(self, loss):
        pass

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), float('inf')
        )
        if self.global_step % 50 == 0:
            print(f"  Step {self.global_step}: grad_norm={grad_norm:.4f}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: loss={outputs['loss'].item():.4f}")

    def on_train_epoch_end(self):
        avg_loss = sum(self.epoch_outputs) / len(self.epoch_outputs)
        print(f"── Epoch {self.current_epoch} ends "
              f"(avg_loss={avg_loss:.4f}) ──")

    def on_validation_epoch_start(self):
        print("  🔍 Validation starting...")

    def validation_step(self, batch, batch_idx):
        output = self.gpt(batch['input_ids'], labels=batch['labels'])
        self.log('val_loss', output['loss'])
        return output['loss']

    def on_validation_epoch_end(self):
        print("  ✅ Validation complete")

    def on_fit_end(self):
        print("\n🎉 All done!")


model = DebugLifecycleGPT(config)
trainer = pl.Trainer(
    max_epochs=2,
    limit_train_batches=10,
    limit_val_batches=2,
    enable_checkpointing=False,
    logger=False,
)

trainer.fit(model, train_loader, val_loader)
```

运行后你会看到类似这样的输出：

```
🚀 Training is about to start!

── Epoch 0 starts ──
  First batch: input shape=torch.Size([8, 128])
  Step 0: grad_norm=12.3456
  Batch 0: loss=4.6052
  Batch 100: loss=3.8765
  ...
── Epoch 0 ends (avg_loss=3.9234) ──
  🔍 Validation starting...
  ✅ Validation complete

── Epoch 1 starts ──
  First batch: input shape=torch.Size([8, 128])
  ...
── Epoch 1 ends (avg_loss=3.1234) ──

🎉 All done!
```

从这个输出中你可以清楚地看到每个钩子的执行顺序：`on_fit_start` → `on_train_epoch_start` → `on_train_batch_start` → `training_step` → `on_after_backward` → `on_train_batch_end` → ... → `on_validation_epoch_start` → `validation_step` → ... → `on_fit_end`。

## Callback 系统：可插拔的训练行为扩展

如果说生命周期是 Lightning 的"事件系统"，那 Callback 就是这个系统的"插件机制"。Callback 允许你把自定义的逻辑封装成独立的类，然后注册到 `Trainer` 上，在特定的生命周期钩子处自动执行。Lightning 内置了大量实用的 Callback，同时你也可以轻松地编写自己的 Callback。

### 内置 Callback 一览

**ModelCheckpoint** — 自动保存最佳模型：

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',            # 监控哪个指标
    mode='min',                    # 'min'(越小越好) 或 'max'
    save_top_k=3,                  # 保留最好的 K 个
    filename='{epoch}-{val_loss:.4f}',  # 文件名格式
    save_last=True,                # 总是保存最后一个 checkpoint
    every_n_epochs=1,              # 每 N 个 epoch 保存一次
)

trainer = pl.Trainer(callbacks=[checkpoint_callback])
```

这是最常用的 Callback 之一。它会在每次验证结束后检查指标是否进入了 Top-K，如果是就保存 checkpoint；如果不在 Top-K 中就删除之前保存的旧版本以节省磁盘空间。`save_top_k=-1` 表示保留所有 checkpoint。

**EarlyStopping** — 早停机制：

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,              # 连续 N 个 epoch 无改善则停止
    mode='min',
    min_delta=0.001,        # 改善幅度小于此值视为无改善
    verbose=True,
)
```

EarlyStopping 的实现非常智能：它不会立即停止训练，而是等到当前 epoch 完全结束后再决定是否停止。这意味着即使触发了早停条件，当前的 epoch 也会正常完成，不会有数据浪费。

**LearningRateMonitor** — 记录学习率变化：

```python
from pytorch_lightning.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(callbacks=[lr_monitor])
```

这会把每步的学习率记录到日志中。配合 W&B 或 TensorBoard 使用时，你可以在界面上直接看到学习率的 warmup 和 cosine decay 曲线。

**StochasticWeightAveraging (SWA)** — 权重平均提升泛化：

```python
from pytorch_lightning.callbacks import StochasticWeightAveraging

swa = StochasticWeightAveraging(
    swa_lrs=1e-3,              # SWA 阶段使用的学习率
    swa_epoch_start=0.8,       # 在训练进度的 80% 处开始 SWA
    annealing_epochs=5,        # 退火 epoch 数
    annealing_strategy='cos',  # 余弦退火
)
```

SWA 的原理是在训练后期用一种修正的平均策略来收集多个时刻的模型权重并取平均。实验表明 SWA 通常能带来 0.2~0.5 的泛化性能提升，而且几乎不需要额外的计算开销。

### 编写自定义 Callback

内置的 Callback 虽然覆盖了大部分常见需求，但总有一些场景需要定制行为。比如你想监控梯度范数并在梯度过大时发出警告，或者想在每个 epoch 结束时从验证集采样生成一段文本来观察模型的生成质量。这些都可以通过自定义 Callback 实现。

```python
import numpy as np


class GradientMonitoring(pl.Callback):
    """自定义 Callback: 监控梯度统计"""

    def on_after_backward(self, trainer, pl_module):
        total_norm = 0.0
        param_count = 0

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1

        total_norm = total_norm ** 0.5

        if trainer.logger:
            trainer.logger.experiment.log({
                'grad_norm': total_norm,
                'global_step': trainer.global_step,
            })

        if total_norm > 100:
            print(f"⚠️ Warning: large gradient norm at step "
                  f"{trainer.global_step}: {total_norm:.2f}")


class TextGenerationCallback(pl.Callback):
    """每个 epoch 结束时生成示例文本"""

    def __init__(self, tokenizer, prompt="The future of AI", max_tokens=50):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_tokens = max_tokens

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()

        input_ids = self.tokenizer.encode(self.prompt, return_tensors='pt')
        input_ids = input_ids.to(pl_module.device)

        with torch.no_grad():
            output_ids = pl_module.gpt.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                temperature=0.8,
                top_k=40,
            )

        generated_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        if trainer.logger:
            trainer.logger.experiment.log({
                'generated_text': wandb.Text(generated_text),
                'epoch': trainer.current_epoch,
            })

        print(f"\n📝 Epoch {trainer.current_epoch} generation:")
        print(f"  {generated_text}\n")
        pl_module.train()


trainer = pl.Trainer(
    callbacks=[
        GradientMonitoring(),
        TextGenerationCallback(tokenizer=tokenizer),
        ModelCheckpoint(monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=5),
    ]
)
```

注意自定义 Callback 的方法签名——它总是接收 `(self, trainer, pl_module)` 作为前两个参数，其中 `trainer` 是当前的 `Trainer` 实例（可以访问 global_step、current_epoch 等属性），`pl_module` 是你的 `LightningModule` 实例（可以访问模型参数、调用 forward 方法等）。这种设计让 Callback 能够访问训练过程中的所有信息。

## Logger 系统：统一的日志接口

Lightning 的日志系统设计得非常优雅——你只需要调用 `self.log()`，具体的日志去向由你在创建 `Trainer` 时指定的 Logger 决定。这实现了日志逻辑和日志后端的完全解耦。

```python
def logging_demo():
    class LoggingDemoModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)

        def training_step(self, batch, batch_idx):
            loss = self.layer(batch['x']).sum()
            
            self.log('train_loss_simple', loss)
            
            self.log_dict({
                'train_loss_a': loss * 0.5,
                'train_loss_b': loss * 2.0,
            })
            
            self.log('train_loss_progbar', loss, prog_bar=True)
            self.log('train_loss_logger', loss, logger=True)
            
            return loss

    for logger_name, logger_cls in [
        ("TensorBoard", pl.loggers.TensorBoardLogger),
        ("CSV", pl.loggers.CSVLogger),
    ]:
        logger = logger_cls(save_dir=f"./logs/{logger_name.lower()}")
        model = LoggingDemoModule()
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=5,
            logger=logger,
            enable_checkpointing=False,
        )
        dummy_loader = DataLoader([
            {'x': torch.randn(4, 10)} for _ in range(10)
        ])
        trainer.fit(model, dummy_loader)
        print(f"✓ {logger_name} logging OK")


logging_demo()
```

三种 log 方式的区别：
- `self.log(name, value)` — 同时写入 progress bar 和 logger
- `self.log_dict({...})` — 批量记录多个指标
- `prog_bar=True` — 显示在终端进度条上
- `logger=True` — 写入后端（TensorBoard/W&B/CSV 等）

对于 W&B 用户还有一个高级用法：直接访问底层的 `wandb.run` 对象来记录非标量数据（如图表、图片、文本）：

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    
    self.log('train_loss', loss)
    
    if self.trainer.logger and hasattr(self.trainer.logger, 'experiment'):
        wandb_log = self.trainer.logger.experiment
        wandb_log.log({
            'custom_metric': some_complex_value,
            'attention_map': wandb.Image(attention_heatmap),
        }, commit=False)
    
    return loss
```

到这里，我们已经掌握了 Lightning 的两大扩展机制：生命周期钩子让你能在任意时间点注入自定义逻辑，Callback 系统让你能把这些逻辑封装成可复用的组件。下一节我们将进入 Lightning 的高级特性领域——多优化器/多调度器配置、多 DataLoader 支持、分布式训练策略选择，以及 Gradient Checkpointing 这些在大规模 LLM 训练中不可或缺的功能。
