# 不同任务类型的 PEFT 配置策略

## 这一节讲什么？

前两节我们学习了 LoRA、Adapter、Prefix Tuning、(IA)³ 等 PEFT 方法的基本原理。但在实际应用中，**不同类型的 NLP 任务对 PEFT 的配置要求截然不同**——文本分类和代码生成的最优 LoRA 配置可能天差地别。

这一节我们将针对六大常见任务类型：
1. **序列分类**（情感分析、新闻分类）
2. **Token 分类 / 命名实体识别（NER）**
3. **问答（QA）**
4. **文本生成 / 指令微调**
5. **文本摘要**
6. **翻译**

逐一给出经过验证的 **PEFT 配置模板、数据格式要求和完整代码示例**，让你可以直接复制使用。

---

## 一、任务类型与 PEFT 配置的映射关系

## 1.1 为什么不同任务需要不同配置？

不同的 NLP 任务在以下几个维度上存在本质差异：

```python
def analyze_task_dimensions():
    """分析不同任务的维度差异"""

    print("=" * 75)
    print("NLP 任务维度分析与 PEFT 配置映射")
    print("=" * 75)

    dimensions = [
        ("任务类型", "输出空间", "上下文依赖", "推荐 rank", "推荐 target_modules"),
        ("─" * 15, "─" * 12, "─" * 14, "─" * 12, "─" * 30),
        ("序列分类", "小 (2~100 类)", "短-中", "8~16", "query, value"),
        ("NER/POS", "中等 (标签集)", "强 (依赖邻居)", "16~32", "query, value, key"),
        ("抽取式 QA", "大 (span 位置)", "强 (依赖 context)", "16~32", "query, value"),
        ("生成/指令", "极大 (词表大小)", "长 + 自回归", "32~128", "q,k,v,o + gate,up,down"),
        ("摘要", "大 (词表)", "长 + 压缩", "32~64", "q,k,v,o"),
        ("翻译", "大 (词表) × 2", "Encoder-Decoder", "32~64", "encoder+decoder 的 q,v"),
    ]

    for row in dimensions:
        print(f"{row[0]:<15} {row[1]:<12} {row[2]:<14} {row[3]:<12} {row[4]:<30}")

    print("\n💡 核心规律:")
    print("   • 输出空间越小 → rank 可以越小 (分类任务不需要太多表达力)")
    print("   • 上下文依赖越强 → 需要更多 Attention 层的微调")
    print("   • 生成任务 → 需要 FFN 层也参与微调 (影响生成多样性)")

analyze_task_dimensions()
```

---

## 二、序列分类（Sequence Classification）

## 2.1 任务特点

序列分类是最基础的 NLP 任务：输入一段文本，输出一个或多个类别标签。典型场景包括情感分析、新闻分类、意图识别等。

**PEFT 策略要点**：
- 输出空间有限 → 不需要太大的 rank
- 主要依赖 [CLS] token 的表示 → 重点调整 Attention 层
- 分类头本身参数很少 → 通常不冻结

## 2.2 完整代码模板

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np
from evaluate import load


class SequenceClassificationWithLoRA:
    """
    序列分类 + LoRA 微调完整流程
    以中文情感分析为例
    """

    def __init__(self, model_name="hfl/chinese-roberta-wwm-ext", num_labels=5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_and_prepare_data(self, dataset_path=None):
        """加载并准备数据"""
        if dataset_path:
            dataset = load_dataset("csv", data_files=dataset_path)
        else:
            # 使用示例数据集
            dataset = load_dataset("seara/spam", trust_remote_code=True)

        # 标签映射
        label_list = dataset["train"].features["label"].names
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=256,
                padding=False,
            )

        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=4,
            remove_columns=["text"],
        )

        return tokenized

    def setup_lora_model(self):
        """配置并创建 LoRA 模型"""

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        # 序列分类专用 LoRA 配置
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,       # 关键: 任务类型!
            r=16,                              # 分类任务不需要太大 rank
            lora_alpha=16,                     # α/r = 1.0 (适中缩放)
            target_modules=["query", "value"],  # BERT 类模型用 query/value
            lora_dropout=0.05,                 # 低 dropout (防止过拟合小数据)
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        metric_f1 = load("f1")
        metric_acc = load("accuracy")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        f1 = metric_f1.compute(predictions=predictions, references=labels)["f1"]
        acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]

        return {"f1": f1, "accuracy": acc}

    def train(self, output_dir="./cls_lora_output"):
        """执行训练"""

        dataset = self.load_and_prepare_data()
        model = self.setup_lora_model()

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-4,               # LoRA 推荐学习率
            per_device_train_batch_size=32,     # 分类可以较大 batch
            per_device_eval_batch_size=64,
            num_train_epochs=5,
            weight_decay=0.01,
            warmup_ratio=0.06,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("test") or dataset.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        # 保存
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"\n✅ 序列分类 LoRA 微调完成! 保存到 {output_dir}")
        return trainer, model


# 运行示例
if __name__ == "__main__":
    pipeline = SequenceClassificationWithLoRA(num_labels=2)
    trainer, model = pipeline.train()
```

## 2.3 分类任务的特殊注意事项

```python
def classification_gotchas():
    """分类任务 PEFT 的特殊坑点"""

    gotchas = [
        {
            "坑点": "忘记设置 task_type=SEQ_CLS",
            "后果": "peft 可能错误地处理分类头的参数",
            "修复": "LoraConfig(task_type=TaskType.SEQ_CLS)",
        },
        {
            "坑点": "类别不平衡时直接用 accuracy",
            "后果": "模型偏向多数类, 少数类完全被忽略",
            "修复": "使用 F1 / weighted-F1, 或添加 class_weight",
        },
        {
            "坑点": "多标签分类时 problem_type 设置错误",
            "后果": "损失函数不对, 梯度方向错误",
            "修复": "model.config.problem_type = 'multi_label_classification'",
        },
        {
            "坑点": "rank 设得过大 (如 r>64) 在小数据集上",
            "后果": "LoRA 过拟合, 验证集表现远差于训练集",
            "修复": "数据 < 2000 条时 r≤16, 并增加 lora_dropout",
        },
    ]

    for g in gotchas:
        print(f"\n⚠️  {g['坑点']}")
        print(f"   后果: {g['后果']}")
        print(f"   修复: {g['修复']}")

classification_gotchas()
```

---

## 三、命名实体识别（NER）与 Token Classification

## 3.1 任务特点

NER 要求模型对输入序列中的**每个 token** 进行标注（B-PER、I-LOC、O 等）。这与序列分类的关键区别在于：

- **标签对齐问题**：一个中文词可能被拆成多个 subword token，需要正确处理标签
- **强上下文依赖**：判断一个词是否是实体名称需要看前后文
- **评估单位是 entity 而非 token**：token-level 准确率高不代表 entity-level 效果好

## 3.2 完整代码模板

```python
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np
from evaluate import load


class NERWithLoRA:
    """
    NER 命名实体识别 + LoRA 微调
    使用 BIO 标注方案
    """

    def __init__(self, model_name="hfl/chinese-roberta-wwm-ext"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # BIO 标签体系 (可根据实际需求修改)
        self.label_list = [
            "O",           # 非实体
            "B-PER", "I-PER",      # 人名
            "B-ORG", "I-ORG",      # 组织机构
            "B-LOC", "I-LOC",      # 地名
            "B-MISC", "I-MISC",    # 其他实体
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

    def tokenize_and_align_labels(self, examples):
        """
        NER 特有的 tokenization 和标签对齐
        这是整个流程中最关键的一步!
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=256,
            is_split_into_words=True,  # 关键: 告诉 tokenizer 已经分好词了
            padding=False,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    # 特殊 token ([CLS], [SEP], [PAD]) 用 -100 忽略
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # 新词的第一个 subword token → 使用原始标签
                    label_ids.append(label[word_idx])
                else:
                    # 同一词的后续 subword token → 也用 -100 忽略
                    # (也可以选择继承标签, 但 -100 更安全)
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def setup_lora_model(self):
        """NER 专用的 LoRA 配置"""

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        # NER 推荐: 比 classification 更大的 rank, 因为需要更强的上下文建模能力
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,      # Token Classification!
            r=32,                               # NER 需要更大的 rank
            lora_alpha=32,                      # α/r = 1.0
            target_modules=["query", "value", "key"],  # NER 需要更强的 Attention 调整
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def compute_metrics(self, eval_pred):
        """使用 seqeval 计算 entity-level 指标"""

        metric = load("seqeval")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # 移除 -100 并转换为标签字符串
        true_labels = [[self.id2label[l] for l in label if l != -100]
                       for label in labels]
        true_predictions = [[self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                           for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(self, output_dir="./ner_lora_output"):
        """完整的 NER 训练流程"""

        # 加载示例数据 (People's Daily NER)
        try:
            dataset = load_dataset("peoples_daily_ner", trust_remote_code=True)
        except Exception:
            # 如果无法下载, 创建模拟数据
            dataset = self._create_dummy_data()

        tokenized = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            num_proc=4,
        )

        model = self.setup_lora_model()

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-4,               # NER 可以稍高的学习率
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=10,               # NER 通常需要更多 epoch
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation") or tokenized.get("test"),
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(
                self.tokenizer, padding=True, label_pad_token_id=-100
            ),
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"\n✅ NER LoRA 微调完成!")
        return trainer, model

    def _create_dummy_data(self):
        """创建模拟 NER 数据用于演示"""
        from datasets import Dataset

        dummy_data = {
            "tokens": [["北京", "是", "中国", "的首都"]],
            "ner_tags": [[[1, 0, 2, 0]]],  # B-PER, O, B-LOC, O
        }
        return Dataset.from_dict(dummy_data).train_test_split(test_size=0.2)


if __name__ == "__main__":
    ner_pipeline = NERWithLoRA()
    trainer, model = ner_pipeline.train()
```

## 3.3 NER 的关键细节：标签对齐详解

```python
def demonstrate_label_alignment():
    """详细展示 NER 中标签对齐的过程"""

    print("=" * 70)
    print("NER 标签对齐过程详解")
    print("=" * 70)

    example_tokens = ["北", "京", "是", "中", "国", "的", "首", "都"]
    original_labels = ["B-LOC", "I-LOC", "O", "B-LOC", "I-LOC", "O", "O", "O"]

    print(f"\n原始输入:")
    print(f"  Tokens:  {example_tokens}")
    print(f"  Labels:  {original_labels}")

    print(f"\n假设 Subword Tokenization 结果 (BERT 中文):")
    print(f"  '北京' → ['北', '京']  (每个汉字就是一个 token)")
    print(f"  '中国' → ['中', '国']")

    print(f"\nTokenizer 处理后的结果:")

    # 模拟 BERT tokenizer 的行为
    tokenized_result = ["[CLS]", "北", "京", "是", "中", "国", "的", "首", "都", "[SEP]"]
    word_ids = [None, 0, 0, 1, 2, 2, 3, 4, 4, None]

    aligned_labels = []
    prev_word_idx = None
    for idx, (token, widx) in enumerate(zip(tokenized_result, word_ids)):
        if widx is None:
            label = -100  # 特殊 token
        elif widx != prev_word_idx:
            label = original_labels[widx]
        else:
            label = -100  # 同一词的后续 subword
        aligned_labels.append(label if label == -100 else label)
        prev_word_idx = widx

        status = "忽略 (-100)" if (isinstance(aligned_labels[-1], int) and aligned_labels[-1] == -100) else f"{aligned_labels[-1]}"
        print(f"  [{idx:>2}] {token:<6} word_id={str(widx):>4} → {status}")

demonstrate_label_alignment()
```

---

## 四、抽取式问答（Question Answering）

## 4.1 任务特点

抽取式 QA 要求模型从给定的上下文（context）中**精确提取答案片段**。SQuAD 是这类任务的代表数据集。

**PEFT 策略要点**：
- 输出是答案在 context 中的 **start position** 和 **end position**
- 强依赖 context 与 question 的交互 → Cross-Attention 或联合编码很重要
- 需要处理**超长 context** 的截断策略

## 4.2 完整代码模板

```python
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np


class QAWithLoRA:
    """
    抽取式问答 + LoRA 微调
    SQuAD 格式数据处理
    """

    def __init__(self, model_name="hfl/chinese-roberta-wwm-ext"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 384
        self.doc_stride = 128  # 滑动窗口步长

    def prepare_qa_features(self, examples):
        """
        QA 数据预处理
        处理 context 可能超长的问题 (滑动窗口截断)
        """
        # 对 question + context 做 tokenization
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",         # 只截断 context
            max_length=self.max_length,
            stride=self.doc_stride,            # 滑动窗口步长
            return_overflowing_tokens=True,   # 返回所有截断窗口
            return_offsets_mapping=True,      # 返回字符偏移 (用于定位答案)
            padding=False,
        )

        # 映射回原始样本 (因为滑动窗口会产生多个子样本)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # 序列 ID (对应原始数据的哪一条)
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 找到 context 的起始和结束位置
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                # 无法回答的问题
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # 找到 context 的 token 起止索引
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # 将字符位置映射到 token 位置
                if not (offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

        tokenized_examples["start_positions"] = start_positions
        tokenized_examples["end_positions"] = end_positions
        return tokenized_examples

    def setup_lora_model(self):
        """QA 专用的 LoRA 配置"""

        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)

        lora_config = LoraConfig(
            task_type=TaskType.QA,              # Question Answering!
            r=32,
            lora_alpha=32,
            target_modules=["query", "value"],  # QA 重点是理解 question-context 关系
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def train(self, output_dir="./qa_lora_output"):
        """QA 训练流程"""

        dataset = load_dataset("cmrc_2018", trust_remote_code=True)  # 中文 QA 数据集

        tokenized = dataset.map(
            self.prepare_qa_features,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=4,
        )

        model = self.setup_lora_model()

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=12,     # QA 的 input 较长, batch 不能太大
            per_device_eval_batch_size=24,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            fp16=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=DefaultDataCollator(),
        )

        trainer.train()

        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"\n✅ QA LoRA 微调完成!")
        return trainer, model


if __name__ == "__main__":
    qa_pipeline = QAWithLoRA()
    trainer, model = qa_pipeline.train()
```

---

## 五、文本生成与指令微调（Causal LM）

## 5.1 任务特点

这是目前最热门的任务类型：让大语言模型学会遵循指令、进行对话、写代码等。

**PEFT 策略要点**：
- **最大的 rank**：生成任务的表达力需求最高
- **target_modules 最全**：通常包括 Attention 全部投影 + FFN 层
- **数据格式多样**：Alpaca / ShareGPT / OpenAI 格式等

## 5.2 多种数据格式的统一处理

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import torch


class InstructionTuningWithQLoRA:
    """
    指令微调 + QLoRA
    支持多种数据格式
    """

    SUPPORTED_FORMATS = ["alpaca", "sharegpt", "openai", "raw"]

    def __init__(self, base_model_path, format_type="alpaca"):
        self.base_model_path = base_model_path
        self.format_type = format_type.lower()
        assert self.format_type in self.SUPPORTED_FORMATS

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_alpaca(self, example):
        """Alpaca 格式: instruction + input + output"""
        if example.get("input") and len(example["input"].strip()) > 0:
            text = (
                "Below is an instruction that describes a task, "
                "paired with an input that provides further context.\n\n"
                "## Instruction:\n{instruction}\n\n"
                "## Input:\n{input}\n\n"
                "## Response:\n{output}"
            ).format(**example)
        else:
            text = (
                "Below is an instruction that describes a task.\n\n"
                "## Instruction:\n{instruction}\n\n"
                "## Response:\n{output}"
            ).format(**example)
        return {"text": text}

    def format_sharegpt(self, example):
        """ShareGPT 格式: 多轮对话"""
        conversations = example.get("conversations", [])
        messages = []
        for turn in conversations:
            role = turn.get("from", "human")
            content = turn.get("value", "")
            role_map = {"human": "User:", "gpt": "Assistant:", "system": "System:"}
            messages.append(f"{role_map.get(role, role)} {content}")

        text = "\n".join(messages)
        if not text.endswith(self.tokenizer.eos_token):
            text += self.tokenizer.eos_token
        return {"text": text}

    def format_openai(self, example):
        """OpenAI 格式: messages 列表"""
        messages = example.get("messages", [])
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = {"user": "User: ", "assistant": "Assistant: ", "system": "System: "}
            text_parts.append(prefix.get(role, "") + content)

        text = "\n".join(text_parts) + self.tokenizer.eos_token
        return {"text": text}

    def prepare_data(self, dataset_name_or_path, split="train[:5000]"):
        """数据准备流水线"""

        formatters = {
            "alpaca": self.format_alpaca,
            "sharegpt": self.format_sharegpt,
            "openai": self.format_openai,
            "raw": lambda x: x,  # raw 格式已经是纯文本
        }

        if dataset_name_or_path.endswith(".json") or dataset_name_or_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=dataset_name_or_path, split=split)
        else:
            dataset = load_dataset(dataset_name_or_path, split=split)

        formatter = formatters[self.format_type]
        formatted = dataset.map(formatter)

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding=False,
            )

        tokenized = formatted.map(
            tokenize_fn,
            batched=True,
            num_proc=4,
            remove_columns=formatted.column_names,
        )

        return tokenized

    def setup_qlora_model(self):
        """QLoRA 模型配置"""

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        # 生成任务的大规模 LoRA 配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,                                    # 大 rank 用于生成任务
            lora_alpha=16,                            # α/r = 0.25
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",     # 全部 Attention
                "gate_proj", "up_proj", "down_proj",         # FFN 也参与
            ],
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def train(self, output_dir="./qlora_output", dataset_name="tatsu-lab/alpaca"):
        """完整训练流程"""

        tokenized = self.prepare_data(dataset_name)
        model = self.setup_qlora_model()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            optim="paged_adamw_32bit",
            report_to="none",
            gradient_checkpointing=True,  # 节省显存
            dataset_text_field="text",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False, return_tensors="pt"
            ),
        )

        trainer.train()

        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"\n✅ QLoRA 指令微调完成! 保存到 {output_dir}")


if __name__ == "__main__":
    tuner = InstructionTuningWithQLoRA(
        base_model_path="huggingface/CodeLlama-7b-hf",
        format_type="alpaca",
    )
    tuner.train(dataset_name="tatsu-lab/alpaca")
```

---

## 六、快速参考：各任务 PEFT 配置速查表

## 6.1 配置速查

```python
def task_config_cheatsheet():
    """各任务类型 PEFT 配置速查表"""

    print("=" * 85)
    print("各任务类型 PEFT 配置速查表")
    print("=" * 85)

    configs = [
        {
            "task": "序列分类",
            "task_type": "TaskType.SEQ_CLS",
            "r": "8~16",
            "alpha/r": "1.0~2.0",
            "target_modules": "['query', 'value']",
            "dropout": "0.01~0.05",
            "lr": "2e-4 ~ 5e-4",
            "epochs": "3~5",
            "bs": "16~32",
            "special": "注意类别不平衡",
        },
        {
            "task": "NER/Token分类",
            "task_type": "TaskType.TOKEN_CLS",
            "r": "16~32",
            "alpha/r": "1.0",
            "target_modules": "['query', 'value', 'key']",
            "dropout": "0.05~0.1",
            "lr": "3e-4 ~ 5e-4",
            "epochs": "5~10",
            "bs": "8~16",
            "special": "标签对齐用 -100; 用 seqeval 评估",
        },
        {
            "task": "抽取式 QA",
            "task_type": "TaskType.QA",
            "r": "16~32",
            "alpha/r": "1.0",
            "target_modules": "['query', 'value']",
            "dropout": "0.05",
            "lr": "3e-4",
            "epochs": "2~4",
            "bs": "8~12",
            "special": "滑动窗口截断; DefaultDataCollator",
        },
        {
            "task": "生成/指令微调",
            "task_type": "TaskType.CAUSAL_LM",
            "r": "32~128",
            "alpha/r": "0.25~1.0",
            "target_modules": "['q','k','v','o', 'gate','up','down']",
            "dropout": "0.05~0.1",
            "lr": "1e-4 ~ 2e-4",
            "epochs": "1~3",
            "bs": "2~8 (+grad accum)",
            "special": "gradient_checkpointing; paged_adamw",
        },
        {
            "task": "摘要",
            "task_type": "TaskType.SEQ_2_SEQ_LM",
            "r": "32~64",
            "alpha/r": "0.5~1.0",
            "target_modules": "enc-dec q/v + dec q/v",
            "dropout": "0.05",
            "lr": "2e-4 ~ 3e-4",
            "epochs": "3~5",
            "bs": "4~8",
            "special": "Encoder-Decoder 结构; DataCollatorForSeq2Seq",
        },
    ]

    header = ["参数", "分类", "NER", "QA", "生成/指令", "摘要"]
    print(f"\n{header[0]:<20} {header[1]:<18} {header[2]:<22} {header[3]:<18} {header[4]:<25} {header[5]}")
    print("-" * 125)

    param_keys = ["task_type", "r", "alpha/r", "target_modules", "dropout", "lr", "epochs", "bs", "special"]
    for key in param_keys:
        row = [c[key] for c in configs]
        label = key.replace("_modules", "").replace("/", "")
        print(f"{label:<20} {row[0]:<18} {row[1]:<22} {row[2]:<18} {row[3]:<25} {row[4]}")

task_config_cheatsheet()
```

## 6.2 常见错误排查

```python
def task_specific_errors():
    """各任务类型的常见错误及解决方案"""

    errors = [
        {
            "任务": "分类",
            "错误": "RuntimeError: CUDA out of memory during training",
            "原因": "batch_size 太大 或 没开 gradient_checkpointing",
            "解决": "减小 bs 到 8~16; training_args.gradient_checkpointing=True",
        },
        {
            "任务": "NER",
            "错误": "F1 很高但实际抽取效果差",
            "原因": "用了 token-level accuracy 而非 entity-level F1",
            "解决": "必须用 seqeval 库; 检查是否正确处理了 -100",
        },
        {
            "任务": "QA",
            "错误": "Exact Match 为 0 但 F1 正常",
            "原因": "答案位置偏移了几个 token (offset mapping 错误)",
            "解决": "检查 offset_mapping 的逻辑; 增加 doc_stride 让覆盖更充分",
        },
        {
            "任务": "生成",
            "错误": "生成内容重复循环或质量很差",
            "原因": "rank 太小导致表达能力不足; 或学习率太高破坏预训练知识",
            "解决": "增大 r 到 64+; 降低 lr 到 1e-4; 增加 epochs",
        },
        {
            "任务": "通用",
            "错误": "print_trainable_parameters 显示 100% 可训练",
            "原因": "LoraConfig 没有正确应用; 或 target_modules 名字不匹配",
            "解决": "print(model) 检查模块名; 确认使用了 get_peft_model()",
        },
    ]

    print("\n" + "=" * 80)
    print("各任务类型常见错误排查")
    print("=" * 80)

    for e in errors:
        print(f"\n🔴 [{e['任务']}] {e['错误']}")
        print(f"   原因: {e['原因']}")
        print(f"   解决: {e['解决']}")

task_specific_errors()
```

---

## 七、本章小结

这一节我们系统性地学习了如何为不同类型的 NLP 任务配置 PEFT 方法。核心要点回顾：

| 任务 | 核心 PEFT 配置 | 关键注意事项 |
|------|---------------|-------------|
| **序列分类** | r=8~16, query+value | 类别不平衡处理 |
| **NER** | r=16~32, query+value+key | 标签对齐 (-100), seqeval 评估 |
| **QA** | r=16~32, query+value | 滑动窗口截断, offset_mapping |
| **生成/指令** | r=32~128, 全模块+FFN | QLoRA + grad checkpointing |
| **摘要** | r=32~64, encoder+decoder | Seq2Seq 特殊处理 |

**最重要的原则**：
1. **任务复杂度决定 rank 大小**——简单任务小 rank，复杂任务大 rank
2. **上下文依赖决定 target_modules 范围**——依赖强的任务需要更多 Attention 层参与
3. **数据量决定正则化强度**——数据少则增大 dropout、降低 rank
4. **永远先打印 trainable parameters**——确认配置生效后再开始训练

下一节我们将学习 **Adapter 合并与切换**——如何在生产环境中高效地管理多个 PEFT 适配器。
