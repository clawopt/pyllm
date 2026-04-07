# 完整微调脚本模板：覆盖分类、NER、QA 三大任务

## 这一节讲什么？

前面三节我们分别学习了微调原理、Trainer API 基础用法、以及训练监控调试。现在是时候把这些知识整合起来，提供一套**开箱即用的生产级模板**了。

这一节将给出三种最常见的 NLP 任务的完整微调脚本：
1. **文本分类**（单标签/多标签）
2. **命名实体识别（NER）**
3. **抽取式问答（QA）**

每个脚本都遵循相同的结构——配置区、数据加载、模型定义、训练、评估、推理——你可以直接复制修改使用。

---

## 一、通用前置代码

所有任务共享一些基础设置：

```python
#!/usr/bin/env python3
"""
Hugging Face 微调通用脚本模板
支持任务: text-classification / token-classification / question-answering

用法:
    python finetune_template.py --task classification --model_name hfl/chinese-roberta-wwm-ext
    python finetune_template.py --task ner --model_name hfl/chinese-roberta-wwm-ext
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding as DefaultDataCollator,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from evaluate import load as evaluate_load


# ========== 参数解析 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="HF Fine-tuning Template")

    # 任务类型
    parser.add_argument("--task", type=str, default="classification",
                       choices=["classification", "ner", "qa"],
                       help="任务类型")

    # 模型与数据
    parser.add_argument("--model_name", type=str,
                       default="hfl/chinese-roberta-wwm-ext",
                       help="预训练模型名称或路径")
    parser.add_argument("--train_data", type=str, default=None,
                       help="训练数据路径 (JSON/CSV/JSONL)")
    parser.add_argument("--eval_data", type=str, default=None,
                       help="验证数据路径")
    parser.add_argument("--output_dir", type=str, default="./finetuned_output",
                       help="模型输出目录")

    # 超参数
    parser.add_argument("--num_labels", type=int, default=3,
                       help="分类类别数 (仅 classification 需要)")
    parser.add_argument("--max_length", type=int, default=128,
                       help="最大序列长度")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # 高级选项
    parser.add_argument("--fp16", action="store_true",
                       help="是否使用混合精度训练")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                       help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ========== 日志工具 ==========
class TrainingLogger:
    """统一的训练日志记录器"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "training_log.json")
        os.makedirs(output_dir, exist_ok=True)
        self.history = {"args": {}, "metrics": []}

    def log_args(self, args):
        self.history["args"] = {k: v for k, v in vars(args).items()}

    def log_metrics(self, metrics, step=None, epoch=None):
        entry = {"step": step, "epoch": epoch}
        entry.update(metrics)
        self.history["metrics"].append(entry)

        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def summary(self):
        if not self.history["metrics"]:
            print("暂无指标记录")
            return

        final = self.history["metrics"][-1]
        best_loss = min(m.get("eval_loss", float("inf"))
                      for m in self.history["metrics"] if "eval_loss" in m)
        best_acc = max(m.get("eval_accuracy", 0)
                     for m in self.history["metrics"] if "eval_accuracy" in m)

        print("\n" + "=" * 50)
        print("📊 训练总结报告")
        print("=" * 50)
        print(f"  最终 eval_loss:     {final.get('eval_loss', 'N/A')}")
        print(f"  最终 eval_accuracy: {final.get('eval_accuracy', 'N/A')}")
        print(f"  最佳 eval_loss:     {best_loss}")
        print(f"  最佳 eval_accuracy: {best_acc}")
        print(f"  总评估次数:         {len(self.history['metrics'])}")


logger = None


# ========== 数据加载工具 ==========
def load_task_data(args):
    """根据任务类型加载数据"""
    task = args.task

    if task == "classification":
        ds = load_dataset("sew", "chnsenticorp", trust_remote_code=True)
        train_test = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
        train_ds = train_test["train"].select(
            range(min(2000, len(train_test["train"]))))
        eval_ds = train_test["test"].select(
            range(min(200, len(train_test["test"]))))

        label_names = ["negative", "positive"]
        if args.num_labels == 3:
            label_names = ["negative", "neutral", "positive"]

        def label_map(ex):
            ex["label"] = 0 if ex["label"] == 0 else (
                2 if args.num_labels > 2 and ex["label"] > 0 else 1)
            return ex

        if args.num_labels == 3:
            train_ds = train_ds.map(label_map)
            eval_ds = eval_ds.map(label_map)

        label2id = {name: i for i, name in enumerate(label_names)}
        id2label = {i: name for i, name in enumerate(label_names)}

        return {
            "train": train_ds,
            "eval": eval_ds,
            "label2id": label2id,
            "id2label": id2label,
            "text_column": "text",
            "label_column": "label",
        }

    elif task == "ner":
        ds = load_dataset("peoples_daily_ner", trust_remote_code=True)
        train_ds = ds["train"].select(range(min(500, len(ds["train"]))))
        eval_ds = ds["validation"].select(range(min(100, len(ds["validation"]))))

        ner_tags = ds["train"].features["ner_tags"].feature.names

        return {
            "train": train_ds,
            "eval": eval_ds,
            "label2id": {t: i for i, t in enumerate(ner_tags)},
            "id2label": {i: t for i, t in enumerate(ner_tags)},
            "text_column": "tokens",
            "label_column": "ner_tags",
            "ner_tags": ner_tags,
        }

    elif task == "qa":
        ds = load_dataset("cmrc_2018", split="train[:500]")
        train_ds = ds.select(range(min(400, len(ds))))
        eval_ds = ds.select(range(min(100, len(ds))))

        return {
            "train": train_ds,
            "eval": eval_ds,
            "label2id": {},
            "id2label": {},
            "text_columns": ["question", "context"],
            "label_columns": ["answers"],
        }

    else:
        raise ValueError(f"不支持的任务类型: {task}")


# ========== Tokenize 工具 ==========
def get_tokenize_fn(args, data_info):
    """根据任务类型获取 tokenize 函数"""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_len = args.max_length
    task = args.task

    if task == "classification":
        text_col = data_info["text_column"]

        def fn(examples):
            return tokenizer(
                examples[text_col],
                padding=False,
                truncation=True,
                max_length=max_len,
            )
        return fn, [data_info["text_column"], data_info["label_column"]]

    elif task == "ner":
        text_col = data_info["text_column"]
        label_col = data_info["label_column"]
        ner_tags = data_info["ner_tags"]

        def fn(examples):
            tokenized = tokenizer(
                examples[text_col],
                padding=False,
                truncation=True,
                max_length=max_len,
                is_split_into_words=True,
            )

            labels = []
            for i, ner_tags_list in examples[label_col]:
                word_ids = tokenized.word_ids(i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(ner_tags[word_idx])
                        previous_word_idx = word_idx
                    else:
                        label_ids.append(-100)
                labels.append(label_ids)

            tokenized["labels"] = labels
            return tokenized
        return fn, [text_col], [label_col]

    elif task == "qa":
        q_col, c_col = data_info["text_columns"]

        def fn(examples):
            tokenized = tokenizer(
                examples[q_col],
                examples[c_col],
                padding=False,
                truncation="only_second",
                max_length=max_len,
                return_overflowing_tokens=True,
            )
            return tokenized
        return fn, [q_col, c_col], []

    return fn


# ========== 模型加载工具 ==========
def load_model(args, data_info):
    """根据任务类型加载模型和 collator"""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    task = args.task

    if task == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=args.num_labels,
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )
        collator = DataCollatorWithPadding(tokenizer)

    elif task == "ner":
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(data_info["ner_tags"]),
            id2label={i: t for i, t in enumerate(data_info["ner_tags"])},
            label2id={t: i for i, t in enumerate(data_info["ner_tags"])},
        )
        collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100
        )

    elif task == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        collator = DefaultDataCollator()

    return model, tokenizer, collator


# ========== 评估函数 ==========
def get_compute_metrics(task):
    """根据任务类型获取评估函数"""

    if task == "classification":
        acc_metric = evaluate_load("accuracy")
        f1_metric = evaluate_load("f1")

        def fn(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            result = acc_metric.compute(predictions=preds, references=labels)
            result.update(f1_metric.compute(predictions=preds, references=labels, average="weighted"))
            return result
        return fn

    elif task == "ner":
        seq_eval = evaluate_load("seqeval")

        def fn(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=-1)
            true_labels = labels

            results = seq_val.compute(predictions=[predictions], references=[true_labels])
            return {
                "ner_f1": results["overall_f1"],
                "ner_precision": results["overall_precision"],
                "ner_recall": results["overall_recall"],
            }
        return fn

    elif task == "qa":
        squad_metric = evaluate_load("squad_v2")

        def fn(eval_pred):
            pred_vals, _ = eval_pred
            return {"qa_f1": 0.0}  # QA 需要特殊后处理，这里简化
        return fn

    return lambda x: {}


# ========== 主函数 ==========
def main():
    global logger

    args = parse_args()
    print("=" * 60)
    print(f"🚀 开始微调 | 任务: {args.task} | 模型: {args.model_name}")
    print("=" * 60)

    logger = TrainingLogger(args.output_dir)
    logger.log_args(args)

    # 加载数据
    print("\n[1/6] 加载数据集...")
    data_info = load_task_data(args)
    tokenize_fn, remove_text_cols, remove_label_cols = get_tokenize_fn(args, data_info)

    raw_train = data_info["train"]
    raw_eval = data_info["eval"]

    remove_cols = list(set(remove_text_cols + remove_label_cols))
    tokenized_train = raw_train.map(tokenize_fn, batched=True, num_proc=4,
                                       remove_columns=remove_cols)
    tokenized_eval = raw_eval.map(tokenize_fn, batched=True, num_proc=4,
                                     remove_columns=remove_cols)

    print(f"  训练集: {len(tokenized_train)} 条")
    print(f"  验证集: {len(tokenized_eval)} 条")

    # 加载模型
    print("\n[2/6] 加载模型...")
    model, tokenizer, collator = load_model(args, data_info)
    print(f"  参数量: {model.num_parameters():,}")

    # 配置训练参数
    print("\n[3/6] 配置训练...")
    compute_metrics_fn = get_compute_metrics(args.task)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=args.fp16 or torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
        if args.task != "qa" else "eval_f1",
        greater_is_better=False if args.task != "qa" else True,
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=3,
    )

    # 初始化 Trainer
    print("\n[4/6] 初始化 Trainer...")
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.001,
        ),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    # 训练!
    print("\n[5/6] 🏋️  开始训练!")
    print("=" * 60)
    trainer.train()

    # 最终评估
    print("\n[6/6] 📊 最终评估...")
    final_metrics = trainer.evaluate()
    logger.log_metrics(final_metrics)
    logger.summary()

    # 保存最终模型
    save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ 模型已保存到: {save_path}")

    # 快速测试
    print("\n🧪 推理测试:")
    from transformers import pipeline

    if args.task == "classification":
        pipe = pipeline("text-classification", model=save_path, tokenizer=save_path)
        test_texts = ["这个产品非常好！", "太差了。", "还可以吧。"]
        for text in test_texts:
            res = pipe(text)[0]
            print(f"  '{text}' → {res['label']} ({res['score']:.4f})")

    elif args.task == "ner":
        pipe = pipeline("token-classification", model=save_path, aggregation_strategy="simple")
        res = pipe("马云在杭州创立了阿里巴巴公司")
        print(f"  NER 结果: {res}")

    print("\n🎉 全部完成!")


if __name__ == "__main__":
    main()
```

---

## 二、各任务的特化说明

## 文本分类的注意事项

```python
# ===== 多标签分类 (Multi-label Classification) =====
# 如果一个样本可以属于多个类别:

# 方式 1: 使用 BCEWithLogitsLoss
from transformers import AutoModelForSequenceClassification
import torch.nn as nn

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=5,
    problem_type="multi_label_classification",  # 关键！自动切换为 BCE loss
)

# 方式 2: 自定义 loss（更灵活）
class MultiLabelClassifier(nn.Module):
    def __init__(self, pretrained_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])

        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

## NER 的注意事项

```python
# ===== NER 的特殊处理 =====

# 1. is_split_into_words=True
#    因为 NER 的输入是已经分好词的 token 列表（不是原始字符串）
#    这个参数让 tokenizer 知道不要再次分词

# 2. 标签对齐 (Label Alignment)
#    tokenize 后 token 数量可能变化（如 "苹果" → ["苹", "果"]）
#    需要通过 word_ids() 将原始标签映射到新的 token 序列上
#    特殊标记 ([CLS], [SEP], subword) 的标签设为 -100（在 loss 中被忽略）

# 3. pad_to_multiple_of=8
#    为了 GPU 效率，将序列长度填充到 8 的倍数
```

## QA 的注意事项

```python
# ===== QA 的特殊处理 =====

# 1. return_overflowing_tokens=True
#    长文档可能超出 max_length，允许溢出并返回多个样本
#    每个 overflow 样本是原文的一个滑动窗口片段

# 2. stride 参数
#    控制滑动窗口的重叠大小，避免答案被截断到两个窗口之间

# 3. 后处理
#    QA 模型输出 start_logits 和 end_logits
#    需要找到 (start, end) 对使得 score = start_logit[end_logit] 最大
#    且 end >= start（不能倒序）
```

---

## 小结

这一节提供了一个**生产级的通用微调脚本模板**：

1. **统一架构**： argparse 参数解析 → 数据加载 → Tokenize → 模型初始化 → Trainer 配置 → 训练 → 评估 → 保存 → 推理测试，八步标准流程
2. **三大任务全覆盖**：
   - **文本分类**：`AutoModelForSequenceClassification` + `DataCollatorWithPadding`
   - **NER**：`AutoModelForTokenClassification` + `DataCollatorForTokenClassification` + 标签对齐
   - **QA**：`AutoModelForQuestionAnswering` + 滑动窗口 + overflow 处理
3. **工程最佳实践**：`TrainingLogger` 记录完整训练日志、`EarlyStoppingCallback` 防过拟合、`parse_args` 支持命令行参数、多标签分类的 `problem_type="multi_label_classification"`
4. **扩展点**：每个函数都是独立模块（数据加载、tokenize、模型、评估），可以根据具体需求替换

下一节我们将讨论高级训练技巧——分布式训练、冻结层策略、对比学习等进阶主题。
