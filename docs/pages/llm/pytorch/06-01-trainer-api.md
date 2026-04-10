# 6.1 Trainer API 速成——三步微调一个模型

回顾一下我们到目前为止走过的路：第 1 章建立了 PyTorch 的基础认知，第 2 章构建了生产级的数据管道，第 3 章从零手写了一个完整的 Transformer/GPT 模型，第 4 章实现了手写的训练循环和优化策略，第 5 章用 PyTorch Lightning 把训练流程提升到了工程化水平。每一步都让我们离"真正训练一个大语言模型"更近了一步。但如果你现在去问任何一个正在做 LLM 微调的工程师"你用什么工具"，十有八九得到的回答是：**HuggingFace Transformers 的 Trainer + PEFT（LoRA）**。这不是说我们之前学的东西没有价值——恰恰相反，理解底层原理是正确使用高层工具的前提。但不得不承认的事实是：在 LLM 微调这个具体场景中，HF Trainer + PEFT 已经成为了工业界的事实标准，它的生态集成度、开箱即用的功能丰富程度以及社区支持都是其他方案难以比拟的。

这一章的目标就是让你掌握这套工具链。我们从最简单的三步上手开始，逐步深入到 LoRA 原理、量化训练、完整实战对比，最后覆盖模型评估与导出。先别被这些名词吓到——它们背后都有清晰的逻辑链条。

## 为什么 LLM 微调首选 HF Trainer？

在深入代码之前，先理解 HF Trainer 相对于我们之前学过的两种方式（手写 PyTorch 和 Lightning）的独特优势在哪里。核心原因可以归结为一点：**生态集成深度**。

当你用 HF Trainer 时，你获得的不仅仅是一个训练循环封装器，而是一个与整个 HuggingFace 生态系统无缝对接的工作流：

```
Model Hub (自动下载/上传)
    ↓
AutoModel (自动识别架构 + 权重加载)
    ↓
Trainer (统一训练接口)
    ├── 内置 PEFT/LoRA 支持
    ├── 内置 BitsAndBytes 量化支持
    ├── 内置 DeepSpeed/FSDP 分布式支持
    └── 内置 W&B/TensorBoard 日志
    ↓
push_to_hub (一键部署)
```

每一个环节都经过了数千个项目的验证，而且它们之间的配合是天衣无缝的。比如你想用 LoRA 微调一个 Qwen 模型——在 HF 体系中只需要三行配置；如果你想换成全量微调——改一行参数；如果你想加上 4-bit 量化来节省显存——再加两行配置。这种灵活性来自于 HF 团队对 LLM 训练工作流的深刻理解和大量工程投入。

## 三步上手：最快的方式微调一个 7B 模型

让我们直接看代码。下面是用 HF Trainer + PEFT 在单个 GPU 上微调 Qwen2.5-7B-Instruct 模型的最小可行示例：

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def quick_finetune():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print("[Step 1] Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Step 2] Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[Step 3] Training...")
    args = TrainingArguments(
        output_dir="./output/qwen-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="qwen-lora-demo",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8
        ),
    )

    trainer.train()

    print("Training complete!")


quick_finetune()
```

就这三步——加载模型、配置 LoRA、启动训练。运行后你会看到类似这样的输出：

```
[Step 1] Loading model and tokenizer...
[Step 2] Configuring LoRA...
trainable params: 33,554,432 || all params: 7,696,998,912 || trainable%: 0.4359%
[Step 3] Training...
{'train_runtime': 4523.12, 'train_samples_per_second': 1.98, 
 'train_steps_per_second': 0.124, 'train_loss': 2.1234, 'epoch': 3.0}
Training complete!
```

注意那个 `trainable%: 0.4359%` —— 这意味着我们在只训练不到 0.5% 参数的情况下完成了对 7B 模型的微调。这就是 LoRA 的威力所在，下一节我们会详细解释它的工作原理。

## TrainingArguments 完整参数指南

`TrainingArguments` 是 HF Trainer 的配置中心，它包含了控制训练行为的所有参数。虽然大多数参数都有合理的默认值，但理解每个参数的含义对于调试问题和优化性能至关重要。我们把参数按功能分组来介绍：

### 训练控制参数

```python
args = TrainingArguments(
    # 基本训练设置
    num_train_epochs=3,              # 总训练轮数
    max_steps=-1,                    # 最大步数（-1 表示由 epoch 决定）
    
    # Batch 相关
    per_device_train_batch_size=4,   # 每 GPU 的训练 batch size
    per_device_eval_batch_size=4,     # 每 GPU 的验证 batch size
    gradient_accumulation_steps=4,   # 梯度累积步数
    
    # 学习率相关
    learning_rate=2e-4,             # 峰值学习率
    weight_decay=0.01,               # 权值衰减
    adam_beta1=0.9,                  # Adam beta1
    adam_beta2=0.999,                # Adam beta2
    adam_epsilon=1e-8,               # Adam epsilon
    
    # 学习率调度
    lr_scheduler_type="cosine",      # "linear", "cosine", "constant", "constant_with_warmup"
    warmup_ratio=0.03,              # warmup 占总步数的比例
    warmup_steps=0,                  # 或指定具体的 warmup 步数
)
```

### 优化与精度参数

```python
args = TrainingArguments(
    # 精度
    fp16=False,                      # FP16 混合精度
    bf16=True,                       # BF16 混合精度（推荐）
    tf32=None,                       # TF32（Ampere+ 自动启用）
    
    # 梯度
    max_grad_norm=1.0,               # 梯度裁剪阈值
    
    # 内存优化
    gradient_checkpointing=True,    # 梯度检查点（节省显存 ~30-50%）
    dataloader_num_workers=4,       # 数据加载进程数
    dataloader_pin_memory=True,      # 锁定页内存
    
    # 优化器
    optim="adamw_torch",            # "adamw_hf", "adamw_torch", "adafactor", "adamw_bnb_8bit"
)
```

### 日志与保存参数

```python
args = TrainingArguments(
    # 日志
    logging_steps=10,                # 每 N 步记录一次日志
    report_to="wandb",               # "wandb", "tensorboard", "none"
    run_name="my-experiment",
    
    # Checkpoint
    save_strategy="epoch",           # "no", "steps", "epoch"
    save_steps=500,                  # save_strategy="steps" 时使用
    save_total_limit=3,              # 最多保留几个 checkpoint
    
    # 验证
    eval_strategy="steps",           # "no", "steps", "epoch"
    eval_steps=100,                  # eval_strategy="steps" 时使用
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
)
```

### 分布式参数

```python
args = TrainingArguments(
    # 分布式
    fsdp="",                        # FSDP 配置字符串或 JSON 路径
    deepspeed="",                   # DeepSpeed 配置文件路径
    ddp_find_unused_parameters=False,  # DDP: 是否查找未使用的参数
    
    # 其他
    local_rank=-1,                  # DDP rank（通常不需要手动设置）
    seed=42,                        # 随机种子
    data_seed=None,                 # 数据 shuffle 种子
)
```

这里有几个容易出错的细节值得特别说明：

**`optim` 参数的选择**：`"adamw_torch"` 使用 PyTorch 原生的 AdamW 实现（推荐），`"adamw_hf"` 使用 HuggingFace 自己的实现（行为略有不同，某些旧模型可能需要）。`"adamw_bnb_8bit"` 来自 bitsandbytes 库，用于 8-bit 优化器状态以节省显存。`"adafactor"` 用于超大模型的低内存优化器。

**`save_strategy` vs `eval_strategy`**：这两个参数独立控制 checkpoint 保存和验证执行的频率。你可以每个 epoch 保存一次 checkpoint 但每 100 步做一次验证——这在需要频繁监控 val loss 但不想频繁写磁盘的场景下很有用。

**`bf16` vs `fp16`**：和第 4 章讨论的一样，优先选择 `bf16=True`（如果你的 GPU 支持）。它会自动处理所有混合精度的细节，不需要你手动管理 GradScaler。

## DataCollator 系列

HF Trainer 提供了几种内置的 DataCollator，针对不同的任务类型做了预配置：

**DataCollatorForLanguageModeling** — 语言模型训练专用：

```python
from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,                     # False = 因果语言建模 (CLM), True = BERT式 MLM
    pad_to_multiple_of=8,          # padding 到 8 的倍数（提高 Tensor Core 效率）
    return_tensors="pt",
)
```

当 `mlm=False` 时，它基本上就是我们在第 2 章自己实现的 LLMDataCollator 的 HF 版本——做动态 padding、构造 attention_mask、处理 labels。当 `mlm=True` 时，它会随机 mask 掉一些 token 用于 BERT 式的掩码语言建模任务。

**DataCollatorWithPadding** — 最简单的 collator，只做 padding：

```python
from transformers import DataCollatorWithPadding

collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",              # "longest" 或 "max_length"
    max_length=512,                 # padding="max_length" 时使用
    return_tensors="pt",
)
```

**DataCollatorForSeq2Seq** — 序列到序列任务（翻译、摘要等）：

```python
from transformers import DataCollatorForSeq2Seq

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    max_length=128,
    label_pad_token_id=-100,
    padding=True,
)
```

对于我们的 LLM 微调任务来说，`DataCollatorForLanguageModeling(mlm=False)` 几乎总是正确的选择。它自动处理的细节包括：
- 动态 padding 到 batch 内最长序列
- 正确构造 attention_mask（区分真实 token 和 padding）
- labels 的 shift 操作（input_ids 右移一位作为 labels）
- padding 位置的 label 设为 -100（ignore_index）

这意味着你在准备数据时只需要提供原始的文本或 token IDs，不需要像在第 2 章那样手动处理 input_ids/labels 的对应关系——DataCollator 会帮你搞定一切。

到这里，你已经能够用 HF Trainer 启动一个基本的 LLM 微调任务了。但这只是冰山一角——真正让现代 LLM 微调变得实用的是 **LoRA（Low-Rank Adaptation）** 技术。下一节我们将深入 LoRA 的数学原理和工程实践，理解为什么只用 0.5% 的可训练参数就能达到接近全量微调的效果。
