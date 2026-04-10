# 6.2 PEFT 与 LoRA 微调深度实践

上一节我们用三行代码就启动了一个 7B 模型的 LoRA 微调任务，并且看到可训练参数只占总参数的 0.4359%。这个数字听起来有些不可思议——不到半百分之一的参数就能让一个预训练好的大模型适应你的特定领域或任务？这一节我们将彻底搞清楚 LoRA 到底在做什么、为什么它有效、以及如何正确地配置和使用它。理解 LoRA 不仅对微调 LLM 有实际价值，更是面试中关于高效微调方法的高频考点。

## LoRA 的核心思想：低秩分解

要理解 LoRA，我们需要回到全量微调（Full Fine-tuning）的问题所在。假设你有一个预训练好的权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$（比如 Attention 层中的 `q_proj`，对于 7B 模型来说 d=4096, k=4096），在全量微调时你需要更新这个矩阵的全部 $d \times k$ 个参数。对于一个 7B 模型来说，所有层的权重加起来有约 70 亿个参数——即使你只想让它学会一种新的对话风格或者某个垂直领域的知识，也需要更新全部 70 亿个参数。

LoRA 的观察是：**预训练模型已经学到了很好的通用表示，微调时的权重更新量 $\Delta W$ 通常是一个低秩矩阵**。也就是说，虽然 $\Delta W$ 在形状上和 $W_0$ 一样是 $d \times k$ 的，但它的"有效秩"（非零奇异值的数量）远小于 $\min(d, k)$。

基于这个观察，LoRA 把 $\Delta W$ 分解为两个小矩阵的乘积：

$$\Delta W = B A$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$ 是秩（rank）。原始权重保持冻结（不参与训练），只有 $A$ 和 $B$ 这两个小矩阵需要训练。

前向传播时的计算变为：

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

从计算量的角度看：原来的 $W_0 x$ 需要一次 $d \times k$ 的矩阵乘法；新增的 $BAx$ 先做 $k \times r$（即 $A x$），再做 $d \times r$（即 $B(Ax)$）。当 r 远小于 d 和 k 时（比如 r=16 而 d=k=4096），新增的计算量约为原计算的 $2r/\min(d,k) = 32/4096 \approx 0.8%$ —— 几乎可以忽略不计。

从参数量角度看：原来 $\Delta W$ 需要 $d \times k = 16.7M$ 个参数（以 d=k=4096 计算）；LoRA 只需 $(d + k) \times r = 8192 \times 16 = 131K$ 个参数——减少了 **127 倍**！这就是上一节看到的 "trainable%: 0.4359%" 的来源。

```python
def lora_param_count_demo():
    d, k = 4096, 4096
    full_finetune_params = d * k

    for r in [4, 8, 16, 32, 64]:
        lora_params = (d + k) * r
        ratio = lora_params / full_finetune_params * 100
        reduction = full_finetune_params / lora_params

        print(f"r={r:>3d}: LoRA params={lora_params:>8,d} "
              f"({ratio:>5.2f}%), reduction={reduction:>5.1f}×")

    print(f"\nFull fine-tuning: {full_finetune_params:,} params (100.00%)")


lora_param_count_demo()

# 输出:
# r=  4: LoRA params=   32,768 ( 0.05%), reduction= 512.0×
# r=  8: LoRA params=   65,536 ( 0.10%), reduction= 256.0×
# r= 16: LoRA params=  131,072 ( 0.20%), reduction= 128.0×
# r= 32: LoRA params=  262,144 ( 0.39%), reduction=  64.0×
# r=  64: LoRA params=  524,288 ( 0.78%), reduction=  32.0×
#
# Full fine-tuning: 16,777,216 params (100.00%)
```

## 为什么低秩假设成立？

这是理解 LoRA 最关键也最常被问到的问题：凭什么说权重更新量是低秩的？

直觉上的解释来自任务的性质。当你微调一个预训练的 LLM 时，你通常不是在教它全新的能力（比如从零开始学习注意力机制），而是在**调整它已有的知识来适应特定的分布或风格**。这种调整往往是"方向性"的——比如让模型的输出更正式一点、更偏向某个领域的术语、或者遵循某种特定的格式。这些调整可以用少数几个"方向"（由 LoRA 的 rank 决定）来表示，而不需要修改每一个独立的权重值。

更形式化的解释来自于 Aghajanyan 等人（2020）的研究工作：他们证明了预训练模型在适应下游任务时，其权重更新量确实倾向于集中在低秩子空间中。具体来说，他们发现微调后权重的 intrinsic dimension（内在维度）远小于模型本身的维度——这意味着用低秩矩阵来近似权重更新不仅在工程上方便，而且有理论支撑。

当然，低秩近似是有信息损失的——r 越小损失越大。实践中 r=8~64 是最常用的范围，需要在效果和效率之间权衡。r 太小可能无法捕捉足够的任务特异性信息导致欠拟合；r 太大则接近全量微调失去了 LoRA 的优势且可能导致过拟合。

## PEFT 库核心 API 详解

HuggingFace 的 PEFT（Parameter-Efficient Fine-Tuning）库封装了 LoRA 及其他高效微调方法（如 Prompt Tuning、Prefix Tuning、IA3 等），提供了统一的接口。下面详细讲解 `LoraConfig` 的每个参数：

```python
from peft import LoraConfig, get_peft_model, TaskType


config = LoraConfig(
    # ===== 核心参数 =====
    r=16,                           # LoRA 秩（最重要的超参数）
    lora_alpha=32,                  # 缩放因子（通常设为 2*r）
    lora_dropout=0.05,              # LoRA 层的 dropout
    
    # ===== 目标模块选择（最重要！）=====
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    
    # ===== 其他设置 =====
    bias="none",                    # "none", "all", "lora_only"
    task_type=TaskType.CAUSAL_LM,   # 任务类型
    modules_to_save=None,           # 额外需要完整训练的模块（如 embedding/head）
    inference_mode=False,           # 推理模式（是否合并权重）
)
```

### 参数详解

**`r`（rank）**：LoRA 的秩，决定了低秩近似的表达能力。常见选择：
- r=8：极轻量级，适合快速实验或资源极度受限的场景
- r=16：默认推荐值，大多数场景下的良好平衡点
- r=32/64：更强的表达能力，适合复杂任务或追求最佳效果的场景
- r>128：接近全量微调，一般不建议超过 128

**`lora_alpha`**：缩放因子，控制 LoRA 更新量的幅度。前向传播时实际的缩放系数是 `alpha/r`：

$$h = W_0 x + \frac{\alpha}{r} BAx$$

所以 `lora_alpha=32, r=16` 意味着缩放系数为 2.0。通常 `lora_alpha = 2 * r` 是一个好的起点。增大 alpha 相当于增大 LoRA 部分的学习率（因为梯度会相应放大），减小 alpha 则降低 LoRA 的影响。

**`target_modules`**：**这是 LoRA 配置中最关键的参数**——它决定了哪些层会被注入 LoRA 适配器。选错了目标模块会导致微调完全无效。不同架构的目标模块名称不同：

| 模型 | 推荐 target_modules |
|-----|-------------------|
| LLaMA / Qwen / Mistral | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| LLaMA/Qwen（含 FFN） | `+ ["gate_proj", "up_proj", "down_proj"]` |
| GPT-2 | `["c_attn", "c_proj"]` |
| T5/BART | `["q", "v", "k", "o"]` |

如何查看你所用模型的具体层名？运行以下代码：

```python
def inspect_model_layers(model_name):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"\n{'='*60}")
    print(f"Layer names in {model_name}:")
    print(f"{'='*60}")
    
    for name, param in model.named_parameters():
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            print(f"  {name}: {param.shape}")
        if 'o_proj' in name:
            print(f"  {name}: {param.shape}")
        if 'gate_proj' in name or 'up_proj' in name:
            if 'gate_proj' in name:
                print(f"  ... (FFN layers exist)")
                break


inspect_model_layers("Qwen/Qwen2.5-7B-Instruct")
```

输出类似：
```
============================================================
Layer names in Qwen/Qwen2.5-7B-Instruct:
============================================================
  model.layers.0.self_attn.q_proj.weight: torch.Size([4096, 4096])
  model.layers.0.self_attn.k_proj.weight: torch.Size([4096, 4096])
  model.layers.0.self_attn.v_proj.weight: torch.Size([4096, 4096])
  model.layers.0.self_attn.o_proj.weight: torch.Size([4096, 4096])
  ... (FFN layers exist)
```

**`bias`**：是否为 LoRA 层添加可学习的偏置项。`"none"`（默认）不添加偏置，节省参数；`"lora_only"` 只给 LoRA 的 A/B 矩阵添加偏置；`"all"` 给所有层包括原始权重都添加偏置。除非你有特殊需求，否则 `"none"` 就够了。

**`modules_to_save`**：除了 LoRA 目标模块之外，还需要完整训练（不被冻结）的模块列表。最常见的用途是在微调分类任务时把最终的 classification head 加入这里：

```python
config = LoraConfig(
    ...,
    modules_to_save=["score"],  # 完整训练 LM Head 或分类头
)
```

## 完整的 LoRA 微调流水线

下面是一个生产级的 LoRA 微调模板，包含了数据准备、Tokenization、配置、训练和保存的完整流程：

```python
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
import torch


def full_lora_pipeline():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    data_path = "alpaca_data.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_conversation(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['output']},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(text, truncation=True, max_length=512)

    tokenized_dataset = dataset.map(
        format_conversation,
        remove_columns=dataset.column_names,
        num_proc=8,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="./output/qwen-lora-full",
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
        run_name="qwen-lora-alpaca",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    output_dir = "./output/qwen-lora-final"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ LoRA adapter saved to {output_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    merged_model = PeftModel.from_pretrained(base_model, output_dir)
    merged_model = merged_model.merge_and_unload()
    
    merged_output_dir = "./output/qwen-merged"
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)

    print(f"✓ Merged model saved to {merged_output_dir}")


full_lora_pipeline()
```

这个流程中有几个值得注意的设计决策：

1. **`apply_chat_template`**：使用 tokenizer 内置的聊天模板来格式化指令数据，确保输入格式与模型预训练时一致。不同的模型有不同的模板（ChatML、Alpaca、Mistral 等），`apply_chat_template` 自动处理了这些差异。

2. **FFN 层加入 LoRA**：除了 Attention 层之外，我们还把 `gate_proj`、`up_proj`（SwiGLU 的组成部分）加入了 LoRA。实验表明对 FFN 层做 LoRA 通常能带来额外的性能提升，代价是可训练参数增加（但仍远小于全量微调）。

3. **两阶段保存**：先保存 LoRA adapter（轻量级，只有几 MB），然后通过 `merge_and_unload()` 将 LoRA 权重合并到基础模型中并保存完整的合并模型。Adapter 格式适合后续继续微调或切换不同的基础模型；合并后的模型适合直接部署推理。

## 常见误区与调试指南

**误区一：target_modules 写错了**

症状：loss 不下降或下降极慢，模型输出没有变化。
排查：打印 `model.print_trainable_parameters()` 确认 trainable 参数数量是否合理。如果显示 0% 或极低的百分比（< 0.01%），说明 target_modules 没匹配到任何层。

```python
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 正确输出应该类似:
# trainable params: 33,554,432 || all params: 7,696,998,912 || trainable%: 0.44%
# 如果是:
# trainable params: 0 || ... || trainable%: 0.00%
# → target_modules 名称不对！
```

**误区二：lora_alpha 设置不当**

症状：训练初期 loss 震荡剧烈或不收敛。
原因：`lora_alpha/r` 的比值决定了 LoRA 更新的有效学习率。如果比值过大（比如 alpha=256, r=16 → 缩放系数 16），相当于 LoRA 部分的学习率被放大了 16 倍，容易不稳定。推荐保持 `alpha ≈ 2*r`。

**误区三：忘记在推理时加载 LoRA 权重**

症状：微调后模型的输出和微调前完全一样。
原因：你可能只加载了基础模型而忘记了 LoRA adapter。正确的做法是用 `PeftModel.from_pretrained()` 加载：

```python
# ❌ 错误：只加载了基础模型
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# ✅ 正确：加载基础模型 + LoRA adapter
model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, lora_adapter_path)
```

**误区四：padding 问题导致位置编码错乱**

症状：生成的文本质量差、重复、不连贯。
原因：如果数据预处理时 padding 策略不当（比如用了右填充而不是左填充），因果注意力的位置语义会被破坏。确保使用 `DataCollatorForLanguageModeling(mlm=False)` 或自定义的左填充 collator。

下一节我们将进入量化训练的世界——了解 QLoRA 如何让你在单张消费级显卡（如 RTX 4090 的 24GB 显存）上微调 7B 甚至更大的模型。
