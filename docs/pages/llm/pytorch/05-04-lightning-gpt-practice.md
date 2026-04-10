# 5.4 Lightning 实战：用 Lightning 训练我们的 GPT 模型

经过前三节的铺垫，我们已经掌握了 Lightning 的核心 API、生命周期系统、Callback 机制、多优化器/多 DataLoader 配置、分布式训练策略以及 Gradient Checkpointing 等高级特性。现在是时候把这些知识整合起来，用 Lightning 完整地训练我们在第 3 章手写的 GPT 模型了。这一节不仅是实战演练，更是一个全面的对比——我们将把同一个训练任务分别用手写循环和 Lightning 来实现，逐维度对比两者的代码量、功能完整度、可维护性和扩展性。这个对比会让你清楚地看到 Lightning 在什么场景下最有价值，以及在什么情况下手写循环可能更合适。

## 完整的 LitGPT 实现

下面是一个生产级的 `LitGPT` 类，集成了我们学过的所有 Lightning 特性：

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitGPT(pl.LightningModule):
    """基于 PyTorch Lightning 的 GPT 语言模型训练模块

    封装了第3章实现的 GPT 模型，提供完整的训练、验证和生成接口。
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        n_embed: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['gpt'])

        config = GPTConfig(
            vocab_size=vocab_size,
            n_embed=n_embed,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.gpt = GPT(config)

    def forward(self, input_ids, labels=None, attention_mask=None):
        return self.gpt(input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        output = self(
            batch['input_ids'],
            labels=batch['labels'],
        )
        loss = output['loss']
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(
            batch['input_ids'],
            labels=batch['labels'],
        )
        loss = output['loss']
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.logger and self.current_epoch % 2 == 0:
            self._log_sample_generation()

    def _log_sample_generation(self):
        self.eval()
        prompt_ids = torch.tensor([[1, 15, 23, 7]]).to(self.device)
        
        with torch.no_grad():
            generated = self.gpt.generate(
                prompt_ids,
                max_new_tokens=30,
                temperature=0.8,
                top_k=20,
            )

        gen_text = f"tokens: {generated[0].tolist()[:10]}..."
        
        if self.trainer.logger and hasattr(self.trainer.logger.experiment, 'log'):
            try:
                from wandb import Text as WandbText
                self.trainer.logger.experiment.log({
                    'sample_generation': WandbText(gen_text),
                    'epoch': self.current_epoch,
                }, commit=False)
            except Exception:
                pass
        
        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        total_steps = self.trainer.estimated_stepping_batches
        if total_steps is None or total_steps <= 0:
            total_steps = 10000

        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=0.1,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                filename='gpt-{epoch:02d}-{val_loss:.4f}',
                save_last=True,
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=self.hparams.get('patience', 5),
                mode='min',
                verbose=True,
            ),
            LearningRateMonitor(logging_interval='step'),
        ]

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
```

## 完整的训练启动脚本

有了 `LitGPT` 类之后，启动训练只需要一个简洁的脚本：

```python
def train_gpt_with_lightning():
    model = LitGPT(
        vocab_size=1000,
        n_embed=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=256,
        lr=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
    )

    print(f"Model params: {model.count_params()}")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        gradient_checkpointing=False,
        limit_val_batches=20,
        logger=WandbLogger(
            project="lit-gpt-tutorial",
            name="gpt-small-run",
        ),
        enable_checkpointing=True,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
        ],
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining finished!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
    
    best_result = {
        'best_val_loss': trainer.callback_metrics.get('val_loss'),
        'best_model_path': trainer.checkpoint_callback.best_model_path,
        'total_steps': trainer.global_step,
        'total_epochs': trainer.current_epoch + 1,
    }
    print(f"Results: {best_result}")
    
    return model, best_result


if __name__ == "__main__":
    model, results = train_gpt_with_lightning()
```

运行后你会看到类似这样的输出（假设你连接了 W&B）：

```
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
...
{'total': 524864, 'trainable': 524864}
Epoch 0: 100%|████████| 625/625 [01:45<00:00, 5.95it/s, v_num=1, 
  train_loss=3.2145, val_loss=3.1567, lr=2.98e-05]
Epoch 1: 100%|████████| 625/625 [01:42<00:00, 6.13it/s, v_num=1, 
  train_loss=2.8765, val_loss=2.7834, lr=8.45e-05]
...
Epoch 19: 100%|████████| 625/625 [01:40<00:00, 6.18it/s, v_num=1, 
  train_loss=0.8923, val_loss=1.1234, lr=3.02e-05]

Training finished!
Best model: ./lightning_logs/version_0/checkpoints/gpt-17-1.0823.ckpt
```

同时在 W&B 的界面上你可以看到实时的 loss 曲线、学习率变化图、生成的文本样本等所有记录的信息。

## 手写 vs Lightning：全方位对比

现在让我们做一个系统的对比。下面的表格总结了在完成相同功能的训练任务时，两种方式的差异：

| 维度 | 手写 PyTorch 循环 | PyTorch Lightning |
|-----|-------------------|-------------------|
| **代码量** | ~200 行（含样板） | ~80 行（仅核心逻辑） |
| **设备管理** | 手动 `.cuda()` / `.cpu()` | 自动检测并处理 |
| **半精度** | 手动 `autocast` + `GradScaler` | `precision="bf16-mixed"` 一行 |
| **分布式** | 需要手动写 DDP/FSDP/DeepSpeed | `strategy="fsdp"` 一行切换 |
| **梯度裁剪** | 手动调用 `clip_grad_norm_` | `gradient_clip_val=1.0` 参数 |
| **Checkpoint** | 手动 `torch.save/load` | `ModelCheckpoint` callback 自动化 |
| **早停** | 手动实现 patience 逻辑 | `EarlyStopping` callback |
| **学习率监控** | 需自己实现 | `LearningRateMonitor` 内置 |
| **日志** | 手动集成 W&B/TensorBoard | `self.log()` 统一接口 |
| **多 GPU 同步** | 需处理 sampler/rank 等 | 全自动 |
| **Gradient Checkpoint** | 需手动包装每层 | `gradient_checkpointing=True` |
| **灵活性** | 100% — 可控制每个细节 | ~90% — 极端情况需 override |
| **Bug 风险** | 高 — 样板代码容易出错 | 低 — 核心逻辑经过大量验证 |
| **上手难度** | ⭐⭐⭐⭐⭐ | ⭐⭐ |

从这个对比中可以得出几个重要的结论：

**Lightning 在以下场景下优势最明显**：
1. 多 GPU / 分布式训练——省去的 DDP/FSDP 样板代码量巨大
2. 需要频繁实验不同配置——改一行参数就能换策略
3. 团队协作——统一的接口降低了沟通成本
4. 生产环境——内置的 checkpoint 管理、日志、监控等都是开箱即用的
5. 研究 + 工程混合场景——研究阶段快速迭代，工程阶段稳定可靠

**手写循环仍然有价值的场景**：
1. 极致的性能优化——当你需要精确控制每一个 CUDA kernel 调用时
2. 自定义训练流程——当你的训练过程不符合标准的 epoch/batch 模式时（比如某些强化学习算法）
3. 学习和理解原理——正如本教程所做的，手写一次是理解底层机制的最好方式
4. 极简部署——当你需要一个没有任何依赖的最小可执行文件时

## 从 Lightning 迁移到 HF Trainer

作为本章的结束，也作为下一章（HuggingFace Trainer）的预告，值得一提的是 Lightning 和 HF Trainer 之间的迁移非常自然。两者都采用了类似的声明式设计哲学——定义模型逻辑、配置训练参数、启动 fit/train。主要的区别在于 HF Trainer 与 HuggingFace 生态系统（Tokenizer、Dataset、Model Hub、PEFT）的集成更深，而 Lightning 在通用性和自定义能力上更强。

如果你已经熟悉了 Lightning 的 API，学习 HF Trainer 几乎不需要额外的认知成本——因为它们的设计思路是同源的。下一章我们将深入 HF Trainer 的世界，看看它是如何成为 LLM 微调领域的事实标准工具的。
