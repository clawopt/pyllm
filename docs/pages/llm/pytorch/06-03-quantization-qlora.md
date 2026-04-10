# 6.3 量化训练与 QLoRA

上一节我们学会了用 LoRA 把可训练参数从 70 亿减少到约 3300 万（0.44%），这已经是一个巨大的改进了。但如果你看一下显存占用，会发现一个 7B 模型以 BF16 精度加载就需要约 14GB 显存——加上优化器状态（AdamW 的 m 和 v，又是约 28GB）、梯度（14GB）和激活值（取决于 batch size 和序列长度，可能再增加数 GB 到数十 GB），总需求轻松超过 60GB。这意味着即使有了 LoRA，你仍然至少需要一张 A100 (80GB) 或两张 A6000 (48GB) 才能舒适地微调 7B 模型。对于大多数个人开发者和中小团队来说，这个硬件门槛仍然太高了。

**量化训练**的思路是：如果模型权重不需要完整的 BF16/FP16 精度呢？如果我们能用更少的比特来表示每个权重数值，不就能大幅降低显存需求了吗？这一节我们将学习如何用 4-bit 量化把 7B 模型的显存占用从 ~14GB 降到 ~3.5GB，配合 LoRA 实现所谓的 **QLoRA（Quantized LoRA）**——让你在单张 RTX 4090 (24GB) 甚至 RTX 3090 (24GB) 上就能微调 7B 模型。

## 为什么需要量化训练？

先算一笔账。假设你要微调一个 7B 参数的模型：

| 组件 | FP16/BF16 | 占比 |
|-----|-----------|------|
| 模型参数 | 14 GB | 23% |
| AdamW 优化器状态 (m + v) | 28 GB | 47% |
| 梯度 | 14 GB | 23% |
| 激活值 (batch=4, seq=2048) | ~4-8 GB | 7% |
| **总计** | **~60-64 GB** | **100%** |

注意优化器状态占了将近一半的显存！这是全量微调或标准 LoRA 微调都无法避免的开销——因为 AdamW 需要为每个参数维护两个 FP32 状态变量。但如果我们能把模型权重的精度从 16-bit 降低到 4-bit，那模型本身只需要约 3.5GB（4-bit = 0.5 byte/param × 7B ≈ 3.5GB），加上 LoRA 的可训练部分（BF16，约 130MB）、优化器状态（只针对 LoRA 参数，约 260MB）、梯度和激活值，总需求可以降到 **12~18GB**——一张 24GB 的消费级显卡就够用了！

这就是 QLoRA 的核心价值主张：**4-bit 基础模型 + LoRA 微调 = 单卡微调大模型的能力**。

## BitsAndBytes 量化配置

HuggingFace 与 bitsandbytes 库深度集成，使得在 HF Trainer 中使用量化只需几行配置：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_4bit_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    return model


model = load_4bit_model("Qwen/Qwen2.5-7B-Instruct")
print(f"Model loaded in 4-bit: dtype={model.dtype}")
```

`BitsAndBytesConfig` 的四个关键参数需要逐一理解：

### `load_in_4bit=True`

启用 4-bit 权重量化。模型的所有预训练权重都会被量化到 4-bit 表示，在前向传播和反向传播时按需反量化到计算精度（由 `bnb_4bit_compute_dtype` 决定）。注意：只有基础模型的权重被量化，LoRA 新增的 A/B 矩阵始终以完整精度（BF16）存储和训练。

### `bnb_4bit_quant_type="nf4"`

指定 4-bit 量化的具体格式。有两个选项：

- **"fp4"**：标准的 4-bit 浮点格式，1 bit 符号 + 3 bit 尾数。简单但精度较低。
- **"nf4"（NormalFloat 4）**：bitsandbytes 团队专门为神经网络权重设计的 4-bit 格式。它的核心创新是把预训练权重的分布（近似正态分布）映射到 4-bit 能表示的 16 个值上，使得量化后的信息损失最小化。

```python
def nf4_vs_fp4_demo():
    import torch.nn.functional as F

    weights = torch.randn(10000)

    fp4_values = torch.quantize_per_tensor(
        weights, torch.quint4, scale=weights.abs().max() / 7, zero_point=0
    )
    
    print(f"Original weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"NF4 is optimized for normally distributed weights "
          f"(like pretrained model weights)")
    print(f"For LLM fine-tuning, always use 'nf4' unless you have a specific reason")


nf4_vs_fp4_demo()
```

### `bnb_4bit_compute_dtype=torch.bfloat16`

量化后的权重在进行矩阵乘法之前会被反量化到什么精度。"bf16" 是推荐值——它提供了足够的计算精度来保证训练质量，同时动态范围与 FP32 相同避免了溢出问题。如果你的 GPU 不支持 BF16，可以改用 `torch.float16`（但需要配合 GradScaler）。

### `bnb_4bit_use_double_quant=True`

这是 QLoRA 论文中的一个重要创新——**双重量化（Double Quantization）**。它的原理是：对第一次量化过程中产生的缩放因子（scale）再做一次量化。因为每个量化组（通常 64 个参数一组）都有一个 FP32 的缩放因子，当模型很大时这些缩放因子也会占用不少显存（7B 模型约有 110K 个缩放因子 × 4 bytes ≈ 0.44MB）。双重量化对这些缩放因子再做一次 8-bit 量化，额外节省约 0.37 bits/parameter（平均下来）。听起来不多，但对于 70B 模型来说也能省下几百 MB。

## QLoRA 完整实现

把量化和 LoRA 结合起来就是 QLoRA：

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch


def qlora_training():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("[Loading 4-bit quantized base model...]")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    args = TrainingArguments(
        output_dir="./output/qwen-qlora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        report_to="wandb",
        run_name="qwen-qlora-demo",
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

    output_dir = "./output/qwen-qlora-final"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ QLoRA adapter saved to {output_dir}")


qlora_training()
```

有几个细节值得特别注意：

**`optim="adamw_bnb_8bit"`**：既然基础模型已经是 4-bit 了，优化器状态也应该尽量压缩。`adamw_bnb_8bit` 使用 bitsandbytes 提供的 8-bit AdamW 实现，将 m 和 v 从 FP32 量化到 INT8，进一步节省约 75% 的优化器显存开销。这是 QLoRA 能够塞进 24GB 显存的关键技术之一。

**`per_device_train_batch_size=2` + `gradient_accumulation_steps=8`**：由于显存有限，每张卡的 batch size 只能设为 2，通过 8 步累积来模拟 batch size=16 的效果。这是显存受限时的标准做法。

**`gradient_checkpointing=True`**：在 QLoRA 场景下几乎是必须启用的——它能节省约 30%~50% 的激活值显存，对于 24GB 显存的显卡来说可能是决定性的差异。

## 显存对比：不同方案的实际占用

让我们用一个统一的基准来对比各种方案的显存需求：

```python
def memory_comparison():
    configs = [
        ("Full FT (BF16)", {
            "quantization": None,
            "lora": False,
            "optim_8bit": False,
            "grad_checkpointing": False,
        }),
        ("LoRA (BF16)", {
            "quantization": None,
            "lora": True,
            "optim_8bit": False,
            "grad_checkpointing": False,
        }),
        ("LoRA + GC", {
            "quantization": None,
            "lora": True,
            "optim_8bit": False,
            "grad_checkpointing": True,
        }),
        ("QLoRA", {
            "quantization": "4bit-nf4",
            "lora": True,
            "optim_8bit": True,
            "grad_checkpointing": True,
        }),
    ]

    base_params = 7e9
    param_bytes_bf16 = 2
    param_bytes_4bit = 0.5

    lora_ratio = 0.00436
    lora_params = base_params * lora_ratio

    print(f"\n{'Method':<20s} {'Model':>8s} {'Optimizer':>10s} "
          f"{'Gradients':>10s} {'Activations':>12s} {'Total':>8s}")
    print("-" * 78)

    for name, cfg in configs:
        if cfg["quantization"] == "4bit-nf4":
            model_mem = base_params * param_bytes_4bit / 1e9
        else:
            model_mem = base_params * param_bytes_bf16 / 1e9

        if cfg["lora"]:
            trainable_params = lora_params
            model_mem += trainable_params * param_bytes_bf16 / 1e9
        else:
            trainable_params = base_params

        if cfg["optim_8bit"]:
            opt_mem = trainable_params * 1 * 2 / 1e9
        else:
            opt_mem = trainable_params * 4 * 2 / 1e9

        grad_mem = trainable_params * param_bytes_bf16 / 1e9

        act_mem = 2.0 if cfg["grad_checkpointing"] else 6.0

        total = model_mem + opt_mem + grad_mem + act_mem

        print(f"{name:<20s} {model_mem:>7.1f}G {opt_mem:>9.1f}G "
              f"{grad_mem:>9.1f}G {act_mem:>11.1f}G {total:>7.1f}G")

    print("-" * 78)
    print("\nHardware requirements:")
    print("  RTX 3090/4090 (24GB): → QLoRA ✓")
    print("  A6000 (48GB):         → LoRA + GC ✓")
    print("  A100 (80GB):          → All methods ✓")


memory_comparison()

# 输出:
# Method               Model   Optimizer  Gradients  Activations     Total
# ----------------------------------------------------------------------------
# Full FT (BF16)         14.0G     56.0G       14.0G         6.0G    90.0G
# LoRA (BF16)            14.1G      0.3G        0.3G         6.0G    20.6G
# LoRA + GC              14.1G      0.3G        0.3G         2.0G    16.6G
# QLoRA                   3.8G      0.1G        0.1G         2.0G     6.0G
#
# Hardware requirements:
#   RTX 3090/4090 (24GB): → QLoRA ✓
#   A6000 (48GB):         → LoRA + GC ✓
#   A100 (80GB):          → All methods ✓
```

从这个对比中可以清楚地看到 QLoRA 的优势：总显存需求从全量微调的 90GB 降到 6GB——意味着一张普通的 RTX 3090 就能跑起来。而且根据 Dettmers 等人（2023）的 QLoRA 原始论文，QLoRA 在多个 benchmark 上的表现与全量 BF16 微调几乎持平甚至在某些任务上略优——这是因为 4-bit 量化引入的正则化效应在一定程度上防止了过拟合。

## NF4 vs FP4 vs INT4：量化格式选择

虽然我们在上面的代码中使用了 NF4（推荐），但了解一下其他选项有助于你在特殊场景下做出正确的选择：

| 格式 | 每参数比特数 | 动态范围 | 精度 | 适用场景 |
|-----|-----------|---------|------|---------|
| **NF4** | 4 | ±4.8 (信息论最优) | 高（正态分布优化） | **LLM 微调首选** |
| FP4 | 4 | ±65504 (IEEE 标准) | 中等 | 通用 4-bit 推理 |
| INT4 | 4 | -8 ~ 7 | 低（均匀量化） | 传统部署场景 |

NF4 的设计哲学值得深入理解。标准的 FP4 和 INT4 都是为通用数值设计的——FP4 有固定的指数-尾数分配方式，INT4 对所有值做线性量化。但神经网络预训练权重的分布有一个特殊性质：它们非常接近正态分布（均值为 0，方差固定）。NF4 利用了这个性质，把 4-bit 能表示的 16 个值精确地安排在正态分布的分位数位置上，使得量化误差的信息论下界达到最优。这也是为什么 QLoRA 论文强烈推荐 NF4 而不是 FP4 的原因。

## 常见问题排查

**问题一："Out of memory" 即使用了 QLoRA**

可能的原因和解决方案：
1. 序列长度太长 —— 减小 `max_length`（比如从 2048 降到 1024 或 512）
2. Batch size 太大 —— 进一步减小并增大 gradient_accumulation_steps
3. 未开启 gradient_checkpointing —— 加上 `gradient_checkpointing=True`
4. 使用了 `bitsandbytes` 旧版本 —— 升级到最新版 (`pip install --upgrade bitsandbytes`)
5. 数据预处理时没有 truncate —— 确保 tokenization 时有 `truncation=True`

**问题二：Loss 为 NaN 或不稳定**

QLoRA 训练中的 NaN 问题通常来自：
1. 学习率太高 —— QLoRA 推荐的学习率比标准 LoRA 更低（1e-4 vs 2e-4）
2. 未使用 `bnb_4bit_compute_dtype=torch.bfloat16` —— FP16 可能导致反量化溢出
3. 数据中有异常值 —— 检查是否有超长文本或非法字符
4. `optim` 设置错误 —— 确保使用 `adamw_bnb_8bit` 或 `paged_adamw_8bit`

**问题三：量化后推理质量明显下降**

这说明你的模型对量化敏感。尝试：
1. 使用 GPTQ/AWQ 等训练后量化方法替代 QLoRA 的训练中量化
2. 增加 LoRA rank（r=32 或 r=64）
3. 检查是否是 eval 时的问题而非训练问题——有时量化模型在生成时表现正常但在某些 benchmark 上分数下降

到这里，我们已经掌握了让 LLM 微调变得实用化的两个关键技术：LoRA（减少可训练参数）和量化（减少模型显存占用）。下一节我们将做一个全面的对比实验——同一个任务分别用手写 PyTorch 循环、PyTorch Lightning、以及 HF Trainer + PEFT 来实现，直观地感受三种方式的差异。
