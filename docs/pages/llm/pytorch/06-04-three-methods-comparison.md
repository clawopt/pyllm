# 6.4 三种训练方式终极对比

经过前面五章的学习，我们现在掌握了三种完整的大模型训练方式：

1. **纯 PyTorch 手写循环**（第 3~4 章）——从零构建模型 + 手写训练循环
2. **PyTorch Lightning**（第 5 章）——声明式 API 封装工程细节
3. **HuggingFace Trainer + PEFT**（第 6 章）——生态集成度最高的方案

这三种方式并不是互斥的替代关系，而是不同层次上的抽象：手写 PyTorch 是地基，Lightning 是在上面搭建的框架建筑，HF Trainer 则是面向 LLM 微调这个具体场景的精装公寓。理解它们各自的定位和取舍，能帮助你在实际项目中做出最合适的技术选型。这一节我们将在同一个任务上用三种方式分别实现一遍，做一次彻底的"同台竞技"。

## 对比实验设计

为了公平对比，我们需要定义一个统一的任务基准。假设任务是：在 Alpaca 数据集上微调 Qwen2.5-7B-Instruct 模型，使其能够按照指令格式生成回答。

### 任务定义

```
输入: 指令 (instruction) + 输入 (input)
输出: 模型生成的回答 (output)
```

### 统一的超参数

所有三种方式使用相同的模型、数据和学习率相关超参数：

```python
COMMON_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "data_path": "alpaca_data.jsonl",
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size_per_device": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_len": 512,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}
```

## 方式 A：纯 PyTorch 手写循环

这是我们在第 3 章和第 4 章学到的全部知识的综合应用：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time


class ManualTrainingPipeline:
    """方式A: 纯 PyTorch 手写训练循环"""

    def __init__(self, config):
        self.config = config

    def build_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base_model, lora_config)

        return self.model

    def build_data(self):
        from datasets import load_dataset

        ds = load_dataset("json", data_files=self.config["data_path"], split="train")

        def tokenize_fn(example):
            text = self.tokenizer.apply_chat_template([
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ], tokenize=False)
            return self.tokenizer(text, truncation=True, max_length=self.config["max_seq_len"])

        tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names, num_proc=8)
        from transformers import DataCollatorForLanguageModeling
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        loader = DataLoader(tokenized, batch_size=self.config["batch_size_per_device"],
                           collate_fn=collator, shuffle=True, num_workers=4)
        return loader

    def train(self):
        model = self.build_model()
        loader = self.build_data()

        optimizer = torch.optim.AdamW_bnb_8bit(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        total_steps = len(loader) * self.config["num_epochs"]
        warmup_steps = int(total_steps * self.config["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        scaler = torch.cuda.amp.GradScaler(device="cuda")

        global_step = 0
        start_time = time.time()

        for epoch in range(self.config["num_epochs"]):
            model.train()
            epoch_loss = 0

            for step, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(model.device)
                labels = batch.get('labels', input_ids.clone()).to(model.device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / self.config["gradient_accumulation_steps"]

                scaler.scale(loss).backward()

                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        self.config["max_grad_norm"],
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1

                epoch_loss += loss.item() * self.config["gradient_accumulation_steps"]

                if step % 50 == 0:
                    print(f"Epoch {epoch} Step {step}: loss={loss.item()*self.config['gradient_accumulation_steps']:.4f}, "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")

            avg_loss = epoch_loss / len(loader)
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch}: avg_loss={avg_loss:.4f}, time={elapsed/60:.1f}min")

        model.save_pretrained("./output/manual")
        print(f"✓ Training complete ({(time.time()-start_time)/60:.1f} min)")


pipeline = ManualTrainingPipeline(COMMON_CONFIG)
# pipeline.train()  # 取消注释以运行
```

统计一下有效代码行数（不含空行和注释）：约 **180 行**。其中真正和"业务逻辑"相关的只有数据准备和 loss 计算部分（约 40 行），剩下的 140 行都是工程样板代码。

## 方式 B：PyTorch Lightning

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch


class LitQLoRA(pl.LightningModule):
    """方式B: PyTorch Lightning 版本"""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config
        self.model = self._build_model()

    def _build_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"], quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(base, lora_config), tokenizer

    def forward(self, input_ids, labels=None):
        model, _ = self.model
        return model(input_ids=input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        output = self(batch['input_ids'], batch.get('labels'))
        self.log('train_loss', output.loss, prog_bar=True, on_step=True)
        return output.loss

    def configure_optimizers(self):
        model, _ = self.model
        opt = torch.optim.AdamW_bnb_8bit(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
        )
        total = self.trainer.estimated_stepping_batches or 10000
        sched = get_cosine_schedule_with_warmup(opt, int(total * self.hparams.warmup_ratio), total)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def configure_callbacks(self):
        return [ModelCheckpoint(monitor='train_loss', save_top_k=1),
                EarlyStopping(monitor='train_loss', patience=3)]


def run_lightning():
    lit = LitQLoRA(COMMON_CONFIG)
    model, tokenizer = lit.model

    trainer = pl.Trainer(
        max_epochs=COMMON_CONFIG["num_epochs"],
        accelerator="auto", devices="auto", precision="bf16-mixed",
        gradient_clip_val=COMMON_CONFIG["max_grad_norm"],
        accumulate_grad_batches=COMMON_CONFIG["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        logger=WandbLogger(project="qlora-compare"),
        enable_checkpointing=True,
    )
    trainer.fit(lit, train_loader)


# run_lightning()
```

有效代码行数：约 **90 行**。比手写版本减少了 50%，而且自动获得了设备管理、分布式支持、日志记录、checkpoint 管理、早停等功能。

## 方式 C：HuggingFace Trainer

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def run_hf_trainer():
    model_name = COMMON_CONFIG["model_name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="./output/hf-trainer",
        per_device_train_batch_size=COMMON_CONFIG["batch_size_per_device"],
        gradient_accumulation_steps=COMMON_CONFIG["gradient_accumulation_steps"],
        num_train_epochs=COMMON_CONFIG["num_epochs"],
        learning_rate=COMMON_CONFIG["learning_rate"],
        warmup_ratio=COMMON_CONFIG["warmup_ratio"],
        weight_decay=COMMON_CONFIG["weight_decay"],
        max_grad_norm=COMMON_CONFIG["max_grad_norm"],
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        report_to="wandb",
        run_name="hf-trainer-compare",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained("./output/hf-trainer/final")
    print("✓ HF Trainer training complete!")


# run_hf_trainer()
```

有效代码行数：约 **55 行**。这是最简洁的实现，而且包含了完整的 LoRA + 量化 + BF16 + Gradient Checkpointing + 8-bit 优化器功能。

## 全方位对比表

| 维度 | 手写 PyTorch | PyTorch Lightning | HF Trainer |
|-----|-------------|------------------|------------|
| **代码量** | ~180 行 | ~90 行 | **~55 行** |
| **上手难度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **⭐** |
| **设备管理** | 手动 `.cuda()` | 自动 | **自动** |
| **半精度** | 手动 `autocast`+`GradScaler` | `precision=` 参数 | **`bf16=True`** |
| **分布式** | 需要手动写 DDP/FSDP | `strategy=` 参数 | **`fsdp=` / `deepspeed=` 参数** |
| **LoRA 支持** | 需自己集成 PEFT | 需自己集成 PEFT | **内置 PEFT 集成** |
| **量化支持** | 需自己集成 BnB | 需自己集成 BnB | **内置 BnB 集成** |
| **Model Hub** | 需自己处理下载/上传 | 需自己处理 | **`from_pretrained` / `push_to_hub` 一键操作** |
| **Checkpoint** | 手动 `save/load` | `ModelCheckpoint` callback | **自动管理 + Top-K** |
| **日志** | 手动集成 W&B/TB | `self.log()` + Logger | **`report_to="wandb"` 一行配置** |
| **灵活性** | **100%** — 可控制每个细节 | ~85% — 极端情况需 override | ~60% — HF 生态内灵活 |
| **调试友好度** | **最高** — 每行都看得懂 | 高 — 有清晰的钩子系统 | 中 — 封装较深 |
| **社区生态** | 无（原生 PyTorch） | Lightning 社区 | **HF 生态（最大）** |

## 选型决策树

基于以上对比，这里给出一个实用的选型指南：

```
你要做什么？
│
├── 从零实现一个新架构 / 论文复现
│   └── 🎯 **手写 PyTorch 循环**
│       理由：需要完全的控制权，不能有抽象层遮挡细节
│
├── 研究 / 实验性项目（多任务学习 / GAN / 复杂训练流程）
│   └── 🎯 **PyTorch Lightning**
│       理由：需要灵活性但不想重复写样板代码，
│             多优化器/多 DataLoader 支持更好
│
├── LLM 微调 / 产品落地（LoRA / QLoRA）
│   ├── 快速迭代 / 个人项目 / 资源有限
│   │   └── 🎯 **HF Trainer + PEFT** （首选！）
│   │       理由：代码最少、生态最好、开箱即用的 LoRA/量化支持
│   │
│   ├── 需要自定义训练逻辑（非标准 epoch/batch 模式）
│   │   └── 🎯 **PyTorch Lightning**
│   │       理由：生命周期钩子提供更细粒度的控制
│   │
│   └── 需要极致性能或自定义 CUDA kernel
│       └── 🎯 **手写 PyTorch 循环**
│           理由：没有任何中间层的性能开销
│
├── 大规模预训练（100B+ 参数）
│   └── 🎯 **DeepSpeed / FSDP**（通过 HF Trainer 或 Lightning 使用）
│       理由：ZeRO 分片是必须的，两种框架都支持
│
└── 学习路线建议
    初学者:  Ch3(手写) → Ch4(手写循环) → Ch6(HF Trainer)
    进阶者:  Ch3 → Ch5(Lightning) → Ch6 → Ch7(分布式)
    工程师:  直接从 Ch6 开始，遇到问题时回看 Ch3-Ch5
```

## 一个重要的认知：三者不是竞争关系

最后需要强调的一点是：这三种方式的知识是**累积的**而非替代的。你用手写循环获得的对 autograd、计算图、梯度流的理解，会帮助你更有效地使用 Lightning；你用 Lightning 学到的关于生命周期和 Callback 的概念，会让你更好地理解 HF Trainer 内部的工作机制；而你在 HF Trainer 中积累的 LoRA/量化经验，在手写循环中同样适用（只是需要多写一些胶水代码）。最好的学习路径不是选择其中一种然后只学它，而是按顺序全部掌握——先手写建立直觉，再用 Lightning 提升效率，最后用 HF Trainer 解决实际问题。
