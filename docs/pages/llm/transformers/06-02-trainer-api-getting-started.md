# Trainer API 入门：三步完成模型微调

## 这一节讲什么？

上一节我们理解了微调的原理和策略，现在终于要动手写代码了。你可能会想："微调应该很复杂吧？要手动写训练循环、处理 DataLoader、计算梯度、更新权重、保存 checkpoint……"

好消息是——Hugging Face 的 `Trainer` API 把这一切都封装好了。你只需要告诉它三个东西：**模型是什么、数据在哪里、超参数怎么设**，剩下的它全部帮你搞定。

这一节，我们将：
1. 理解 Trainer API 的设计哲学——为什么它能这么简洁
2. 掌握 TrainingArguments 的每一个关键参数
3. 用最少的代码完成一次完整的微调
4. 学会评估、保存和加载微调后的模型

---

## 一、从手动训练循环到 Trainer：代码量的飞跃

## 1.1 手动训练循环长什么样？

在 HF 出现之前（或者不用 Trainer 时），一个标准的 PyTorch 训练循环大约需要 80-120 行代码：

```python
# ===== 手动训练循环（简化版，约 80 行）=====
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

train_dataset = ...   # 预处理好
val_dataset = ...

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 3
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

best_val_loss = float('inf')
output_dir = "./checkpoints"
os.makedirs(output_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    print(f"  Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(os.path.join(output_dir, "best_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
        print(f"  ✅ 保存最佳模型 (val_loss={avg_val_loss:.4f})")

print("训练完成!")
```

## 1.2 用 Trainer 重写同样的逻辑

同样的事情，用 Trainer 只需要约 **25 行**：

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

train_dataset = ...
eval_dataset = ...
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics_func,
)

trainer.train()
trainer.save_model("./my_finetuned_model")
```

**代码量减少了约 70%**，而且功能更全——自动支持混合精度、梯度累积、分布式训练、日志记录、checkpoint 管理……

这就是为什么我们强烈推荐使用 Trainer API。

---

## 二、TrainingArguments —— 微调的"控制面板"

`TrainingArguments` 是 Trainer 的核心配置类，它包含了训练过程中几乎所有可调参数。让我们逐一理解最重要的那些。

## 2.1 必须掌握的核心参数

```python
def explain_training_arguments():
    """详解 TrainingArguments 的核心参数"""

    args = TrainingArguments(
        # ===== 输出与保存 =====
        output_dir="./my_model_output",      # 模型和日志的输出目录
        save_total_limit=3,                  # 最多保留几个 checkpoint（自动删除旧的）
        save_strategy="epoch",              # 保存频率: "no"/"steps"/"epoch"
        load_best_model_at_end=True,         # 训练结束后自动加载最佳模型
        metric_for_best_model="eval_f1",     # 用哪个指标判断"最佳"

        # ===== 训练基本参数 =====
        num_train_epochs=3,                  # 训练轮数
        per_device_train_batch_size=16,      # 每个 GPU 的训练 batch size
        per_device_eval_batch_size=32,       # 每个 GPU 的评估 batch size（可以更大）
        gradient_accumulation_steps=4,       # 梯度累积步数（等效 batch = 16×4=64）

        # ===== 优化器相关 =====
        learning_rate=2e-5,                  # 学习率（最重要！）
        weight_decay=0.01,                   # 权重衰减（L2 正则化）
        adam_beta1=0.9,                      # Adam 的 beta1
        adam_beta2=0.999,                    # Adam 的 beta2
        adam_epsilon=1e-8,                   # Adam 的 epsilon

        # ===== 学习率调度 =====
        lr_scheduler_type="linear",          # 调度器类型: linear/cosine/constant/polynomial
        warmup_ratio=0.1,                    # warmup 占总步数的比例（或用 warmup_steps）

        # ===== 正则化与稳定性 =====
        max_grad_norm=1.0,                    # 梯度裁剪阈值
        label_smoothing_factor=0.0,          # 标签平滑（分类任务常用 0.1）
        fp16=True,                            # 是否使用混合精度训练

        # ===== 评估与日志 =====
        eval_strategy="epoch",              # 评估频率: "no"/"steps"/"epoch"
        logging_strategy="steps",            # 日志频率
        logging_steps=50,                     # 每 N 步打印一次日志
        report_to="tensorboard",             # 日志后端: "tensorboard"/"wandb"/"mlflow"/"none"

        # ===== 数据加载 =====
        dataloader_num_workers=4,           # DataLoader 的 worker 数量
        dataloader_pin_memory=True,          # 是否将数据 pin 到内存（加速 GPU 传输）
        dataloader_drop_last=True,           # 丢弃最后一个不完整的 batch

        # ===== 分布式训练（多 GPU）=====
        # ddp_find_unused_parameters=False,  # DDP 相关
        # deepspeed="./ds_config.json",       # DeepSpeed 配置
    )

    print("=" * 60)
    print("TrainingArguments 参数分组速查")
    print("=" * 60)

    param_groups = {
        "输出管理": ["output_dir", "save_total_limit", "save_strategy",
                   "load_best_model_at_end"],
        "训练规模": ["num_train_epochs", "per_device_train_batch_size",
                   "gradient_accumulation_steps"],
        "优化器": ["learning_rate", "weight_decay", "adam_beta1/2"],
        "调度器": ["lr_scheduler_type", "warmup_ratio/steps"],
        "稳定性": ["max_grad_norm", "fp16", "label_smoothing_factor"],
        "监控": ["eval_strategy", "logging_steps", "report_to"],
    }

    for group_name, params in param_groups.items():
        print(f"\n【{group_name}】")
        for p in params:
            value = getattr(args, p, "N/A")
            if isinstance(value, bool):
                value = "✅" if value else "❌"
            print(f"  {p:<35s} = {value}")


explain_training_arguments()
```

## 2.2 关键参数的深度解读

### learning_rate —— 最重要也最容易搞错的参数

```python
def lr_sensitivity_analysis():
    """分析学习率对微调结果的影响"""
    import matplotlib.pyplot as plt
    import numpy as np

    lrs = [1e-3, 5e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6]
    final_losses = []
    final_accs = []

    for lr in lrs:
        np.random.seed(42)
        param = 0.0
        losses = []
        for step in range(200):
            grad = -2 * (param - 1.0) + np.random.randn() * 0.1
            param -= lr * grad
            loss = (param - 1.0) ** 2
            losses.append(loss)

        final_losses.append(losses[-1])
        final_accs.append(max(0, 1 - abs(param - 1.0)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(lrs)))

    for i, (lr, color) in enumerate(zip(lrs, colors)):
        axes[0].semilogy(lrs[:i+1], final_losses[:i+1], 'o-', color=color,
                       markersize=8, linewidth=2, label=f'lr={lr}')

    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Final Loss (log scale)')
    axes[0].set_title('学习率 vs 最终 Loss\n(越低越好)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(lrs)), final_accs, color=colors)
    axes[1].set_xticks(range(len(lrs)))
    axes[1].set_xticklabels([f'{lr}' for lr in lrs], rotation=45)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('学习率 vs 准确率')

    optimal_idx = np.argmax(final_accs)
    axes[1].axhline(y=max(final_accs), color='green', linestyle='--', alpha=0.7)
    axes[1].bar(optimal_idx, final_accs[optimal_idx],
               color='gold', edgecolor='black', linewidth=2)

    plt.tight_layout()
    plt.show()

    print(f"\n💡 经验法则:")
    print(f"  • 预训练模型微调的标准学习率范围: 1e-5 ~ 5e-5")
    print(f"  • BERT/GPT 类模型通常用 2e-5 作为起点")
    print(f"  • 更大的模型 → 用更小的 lr")
    print(f"  • 数据越多 → 可以适当增大 lr")
    print(f"  • 如果 loss 不下降或震荡 → 先尝试降低 lr")

lr_sensitivity_analysis()
```

### gradient_accumulation_steps —— 显存不够时的救星

```python
def demonstrate_gradient_accumulation():
    """演示梯度累积的工作原理"""
    effective_bs = 64
    actual_bs = 16

    n_accumulation = effective_bs // actual_bs

    print(f"目标有效 Batch Size: {effective_bs}")
    print(f"实际每步 Batch Size:  {actual_bs}")
    print(f"需要梯度累积次数:     {n_accumulation} 步")

    print(f"\n工作原理:")
    print(f"  Step 1: 前向传播 bs={actual_bs}, 计算梯度但不更新")
    print(f"  Step 2: 前向传播 bs={actual_bs}, 累加到上一步的梯度")
    print(f"  ...")
    print(f"  Step {n_accumulation}: 累加完毕，执行一步参数更新")
    print(f"\n  效果: 等效于一次性处理 {effective_bs} 条数据")
    print(f"  显存占用: 只相当于 bs={actual_bs} 的水平")

    print(f"\n⚠️ 注意事项:")
    print(f"  • 实际训练步数 = 原始步数 × n_accumulation")
    print(f"  • 所以 warmup_steps 和 total_steps 也要相应调整")
    print(f"  • 或者直接用 warmup_ratio（按比例自动计算）")

demonstrate_gradient_accumulation()
```

### eval_strategy 与 logging_strategy

```python
def demonstrate_eval_strategies():
    """对比不同评估策略的行为差异"""
    import math

    dataset_size = 1000
    batch_size = 32
    epochs = 3
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    total_steps = steps_per_epoch * epochs

    strategies = {
        "steps (every 50)": {"eval_strategy": "steps", "eval_steps": 50},
        "epoch (every epoch)": {"eval_strategy": "epoch"},
        "no (不评估)": {"eval_strategy": "no"},
    }

    for name, config in strategies.items():
        strategy = config["eval_strategy"]

        if strategy == "steps":
            eval_every = config.get("eval_steps", 50)
            n_evals = total_steps // eval_every
        elif strategy == "epoch":
            n_evals = epochs
        else:
            n_evals = 0

        print(f"{name:25s}: 共评估 {n_evals} 次 "
              f"(总步数={total_steps}, 每 epoch={steps_per_epoch}步)")

demonstrate_eval_strategies()
```

> **生产建议**：对于大多数任务，`eval_strategy="epoch"` 是最好的起点——每个 epoch 结束时做一次完整评估，信息足够且不会太频繁。只有在需要精细调试时才改用 `steps` 模式。

---

## 三、完整的微调实战

## 3.1 从零开始：情感分析微调

下面是一个**可以直接运行**的完整微调脚本：

```python
#!/usr/bin/env python3
"""
BERT 中文情感分类微调完整示例
运行方式: python finetune_sentiment.py
"""

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import numpy as np
from evaluate import load as evaluate_load


# ========== 1. 配置 ==========
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
NUM_LABELS = 3  # 负面/中性/正面
OUTPUT_DIR = "./sentiment_model_output"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5


# ========== 2. 加载数据集 ==========
print("=" * 60)
print("Step 1: 加载数据集")
print("=" * 60)

dataset = load_dataset("sew", "chnsenticorp", trust_remote_code=True)
print(f"原始数据集: {dataset}")

label_names = ["negative", "positive"]
if NUM_LABELS == 3:
    label_names = ["negative", "neutral", "positive"]

label2id = {name: i for i, name in enumerate(label_names)}
id2label = {i: name for i, name in enumerate(label_names)}

def label_to_id(example):
    example["label"] = 0 if example["label"] == 0 else (2 if NUM_LABELS > 2 and example["label"] > 0 else 1)
    return example

if NUM_LABELS == 3:
    dataset = dataset.map(label_to_id)

train_test = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_test["train"].select(range(min(2000, len(train_test["train"]))))
eval_dataset = train_test["test"].select(range(min(200, len(train_test["test"]))))

print(f"训练集: {len(train_dataset)} 条")
print(f"验证集: {len(eval_dataset)} 条")


# ========== 3. Tokenize ==========
print("\n" + "=" * 60)
print("Step 2: Tokenization")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=128,
    )

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=train_dataset.column_names,
    desc="Running tokenizer on train",
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=eval_dataset.column_names,
    desc="Running tokenizer on eval",
)

print(f"Tokenize 完成! Train: {len(tokenized_train)}, Eval: {len(tokenized_eval)}")


# ========== 4. 加载模型 ==========
print("\n" + "=" * 60)
print("Step 3: 加载模型")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)
print(f"模型参数量: {model.num_parameters():,}")


# ========== 5. 定义评估函数 ==========
metric = evaluate_load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# ========== 6. 设置 TrainingArguments ==========
print("\n" + "=" * 60)
print("Step 4: 配置训练参数")
print("=" * 60)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_strategy="steps",
    logging_steps=10,
    fp16=False,
    report_to="none",
    dataloader_num_workers=0,
    seed=42,
    data_seed=42,
)

for attr in ["output_dir", "learning_rate", "num_train_epochs",
             "per_device_train_batch_size", "gradient_accumulation_steps"]:
    val = getattr(args, attr, None)
    if val is not None:
        print(f"  {attr:<35s}= {val}")


# ========== 7. 初始化 Trainer 并训练 ==========
print("\n" + "=" * 60)
print("Step 5: 开始训练!")
print("=" * 60)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()


# ========== 8. 评估最终结果 ==========
print("\n" + "=" * 60)
print("Step 6: 最终评估")
print("=" * 60)

final_metrics = trainer.evaluate()
print(f"\n最终验证集结果:")
for key, value in final_metrics.items():
    print(f"  {key}: {value:.4f}")


# ========== 9. 保存模型 ==========
print("\n" + "=" * 60)
print("Step 7: 保存模型")
print("=" * 60)

save_path = "./my_sentiment_model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 模型已保存到: {save_path}")


# ========== 10. 测试推理 ==========
print("\n" + "=" * 60)
print("Step 8: 测试推理")
print("=" * 60)

from transformers import pipeline

classifier = pipeline("text-classification", model=save_path, tokenizer=save_path)

test_texts = [
    "这个产品非常好用，推荐给大家！",
    "质量太差了，浪费钱。",
    "还可以吧，没有特别惊喜。",
]

results = classifier(test_texts)
for text, res in zip(test_texts, results):
    print(f"  '{text[:20]}' → {res['label']} ({res['score']:.4f})")

print("\n🎉 微调完成!")
```

## 3.2 运行结果解读

当你运行上面的脚本后，Trainer 会输出类似下面的日志：

```
***** Running training *****
  Num examples = 1800
  Num Epochs = 3
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 1
  Total optimization steps = 338
  Number of trainable parameters = 102,261,122

{'loss': 1.0986, 'learning_rate': 0.0000, 'epoch': 0.0}
{'loss': 1.0523, 'learning_rate': 2.00e-06, 'epoch': 0.03}
{'loss': 0.8934, 'learning_rate': 4.00e-06, 'epoch': 0.06}
...
{'loss': 0.1234, 'learning_rate': 1.96e-05, 'epoch': 2.97}
{'train_runtime': 125.3, 'train_samples_per_second': 43.14,
 'train_steps_per_second': 2.70, 'train_loss': 0.0892, 'epoch': 3.0}

***** Evaluation results *****
  eval_loss: 0.1567
  eval_accuracy: 0.9350
```

关键指标解读：

| 指标 | 含义 | 健康值 |
|------|------|--------|
| `train_loss` | 训练损失 | 应该持续下降，最后趋于平稳 |
| `learning_rate` | 当前学习率 | warmup 阶段线性上升，之后按 scheduler 变化 |
| `eval_accuracy` | 验证准确率 | 应该逐 epoch 提升 |
| `samples_per_second` | 吞吐量 | 用于估算总训练时间 |

---

## 四、Trainer 的高级用法

## 4.1 回调函数（Callbacks）

Trainer 支持通过回调函数在训练的关键节点插入自定义逻辑：

```python
from transformers import TrainerCallback, EarlyStoppingCallback

class LoggingCallback(TrainerCallback):
    """自定义日志回调"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[Custom Log] Step {state.global_step}: loss={logs['loss']:.4f}")

class GradientCheckCallback(TrainerCallback):
    """梯度检查回调——检测异常梯度"""

    def on_step_end(self, args, state, control, **kwargs):
        if state.best_metric is not None:
            current = state.best_metric
            if current < 0.01:  # 假设这是 loss
                print(f"⚠️ Loss 异常低 ({current:.6f})，可能存在问题")
        return control


callbacks = [
    EarlyStoppingCallback(early_stopping_patience=2),  # 连续 2 个 epoch 无提升则停止
    LoggingCallback(),
]

trainer = Trainer(
    ...,
    callbacks=callbacks,
)
```

HF 内置的常用回调：

| Callback | 功能 | 使用场景 |
|----------|------|---------|
| `EarlyStoppingCallback` | 提前停止 | 防止过拟合 |
| `PrinterCallback` / `ProgressCallback` | 进度显示 | 默认已启用 |
| `TensorBoardCallback` | 写 TensorBoard 日志 | `report_to="tensorboard"` 时自动添加 |
| `WandBCallback` | 写 Weights & Biases | `report_to="wandb"` 时自动添加 |

## 4.2 断点续训（Resume Training）

```python
# 方式 1: 从最新的 checkpoint 继续
trainer.train(resume_from_checkpoint=True)

# 方式 2: 从指定 checkpoint 继续
trainer.train(resume_from_checkpoint="./results/checkpoint-500")

# 方式 3: 只恢复训练状态（不恢复 optimizer 等）
# 这适用于修改了某些参数后想保留模型权重的场景
```

## 4.3 使用预定义的 metrics

```python
from evaluate import load

f1_metric = load("f1")
precision_metric = load("precision")
recall_metric = load("recall")
accuracy_metric = load("accuracy")

def compute_metrics_full(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    results = {}
    results.update(accuracy_metric.compute(predictions=predictions, references=labels))
    results.update(f1_metric.compute(predictions=predictions, references=labels, average="weighted"))
    results.update(precision_metric.compute(predictions=predictions, references=labels, average="weighted"))
    results.update(recall_metric.compute(predictions=predictions, references=labels, average="weighted"))

    return results
```

---

## 小结

这一节我们从零掌握了 HF Trainer API 的使用：

1. **设计哲学**：Trainer 将 80-120 行的手动训练循环压缩为 ~25 行，封装了优化器、调度器、混合精度、分布式、日志、checkpoint 等全部细节
2. **TrainingArguments 核心参数**：
   - `learning_rate`: 最重要！微调用 2e-5，比预训练低 10-100x
   - `num_train_epochs`: 通常 3-5 轮就够
   - `per_device_train_batch_size`: 尽量大（受显存限制）
   - `gradient_accumulation_steps`: 显存不够时模拟大 batch
   - `warmup_ratio`: 前 5-10% 步线性 warmup
   - `fp16`: 必须开启（节省显存 + 加速）
   - `eval_strategy="epoch"`: 最实用的评估频率
3. **完整流程八步走**：配置 → 加载数据 → Tokenize → 加载模型 → 定义 metrics → 设 Arguments → 初始化 Trainer → train() → save_model() + 推理测试
4. **高级功能**：Callbacks（EarlyStopping/自定义日志）、断点续训、evaluate 库的多指标支持

下一节我们将深入训练过程的监控与调试——当 loss 不收敛、梯度爆炸或 NaN 出现时，如何定位和解决问题。
