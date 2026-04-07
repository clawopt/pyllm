# 数据预处理流水线：从原始文本到 DataLoader 的完整链路

## 这一节讲什么？

上一节我们学会了如何加载和浏览数据集，但原始数据还不能直接送入模型——你需要把文本转换成 token ID、处理变长序列的 padding、构造 attention mask、组织成 batch 等等。这一整套流程就是**数据预处理流水线（Data Processing Pipeline）**。

这一节是连接"数据"和"模型"的关键桥梁，内容非常实用——你几乎在每个微调项目中都会用到这些代码模式：

1. `map` 函数：Datasets 最强大的工具，批量处理每个样本
2. Tokenize 标准写法：`map` + tokenizer 的黄金组合
3. DataCollator：动态 padding、MLM mask、Seq2Seq padding
4. 从 CSV 到 DataLoader：完整的端到端示例

---

## 一、map 函数 —— Datasets 的瑞士军刀

## 1.1 基本用法

`map` 函数可以对数据集中的**每一个样本（或每批样本）**应用自定义函数，返回处理后的新数据集：

```python
from datasets import load_dataset

ds = load_dataset("imdb", split="train").select(range(100))

def add_length(example):
    """为每个样本添加文本长度字段"""
    example["text_length"] = len(example["text"])
    return example

ds_with_length = ds.map(add_length)
print(f"新增列 'text_length': {ds_with_length.column_names}")
print(f"前 3 条长度: {ds_with_length['text_length'][:3]}")
```

## 1.2 batched=True —— 批量处理的威力

这是 `map` 最重要也最容易被忽略的参数。当设置 `batched=True` 时，你的函数接收到的不再是单个样本，而是一个**样本批次**（dict of lists）：

```python
import time

def demonstrate_batched_vs_non_batched():
    """对比 batched 和 non-batched map 的性能差异"""
    from datasets import load_dataset

    ds = load_dataset("imdb", split="train").select(range(1000))

    def slow_process(example):
        import time
        time.sleep(0.001)  # 模拟耗时操作
        example["processed"] = len(example["text"])
        return example

    def fast_process(batch):
        import time
        batch["processed"] = [len(t) for t in batch["text"]]
        return batch

    start = time.time()
    ds_slow = ds.map(slow_process, num_proc=1)
    slow_time = time.time() - start

    start = time.time()
    ds_fast = ds.map(fast_process, batched=True, num_proc=1)
    fast_time = time.time() - start

    print(f"non-batched (逐条): {slow_time:.2f}s")
    print(f"batched (整批):   {fast_time:.2f}s")
    print(f"加速比: {slow_time / max(fast_time, 0.001):.1f}x")

demonstrate_batched_vs_non_batched()
```

**为什么 batched 更快？**
- 减少函数调用开销（调用 100 次而不是 10000 次）
- 更利于向量化操作（列表推导 vs 逐元素循环）
- 与多进程（`num_proc`）配合时效果更显著

## 1.3 remove_columns —— 清理不需要的字段

在预处理过程中，你经常需要删除一些中间字段以节省空间：

```python
def tokenize_and_clean(example, tokenizer):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
    )
    return encoding

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

tokenized_ds = ds.map(
    lambda x: tokenize_and_clean(x, tokenizer),
    batched=True,
    remove_columns=["text"],  # 删除原始 text 列，只保留 tokenized 结果
    num_proc=4,
)

print(f"Tokenize 后的列: {tokenized_ds.column_names}")
print(f"input_ids 示例: {tokenized_ds['input_ids'][0][:10]}")
```

## 1.4 num_proc —— 多进程并行

对于 CPU 密集型的预处理任务（如 tokenize），多进程可以带来显著的加速：

```python
import time

ds = load_dataset("imdb", split="train").select(range(5000))
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

proc_counts = [1, 2, 4]
results = []

for n_proc in proc_counts:
    start = time.time()
    _ = ds.map(tokenize_fn, batched=True, num_proc=n_proc, remove_columns=["text"])
    elapsed = time.time() - start
    results.append((n_proc, elapsed))
    print(f"num_proc={n_proc}: {elapsed:.2f}s")

import matplotlib.pyplot as plt
procs, times = zip(*results)
speedups = [times[0] / t for t in times]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar([str(p) for p in procs], times, color='steelblue')
ax1.set_xlabel('num_proc')
ax1.set_ylabel('Time (s)')
ax1.set_title('Tokenize 耗时')

ax2.bar([str(p) for p in procs], speedups, color='coral')
ax2.axhline(y=procs[-1], color='green', linestyle='--', label='理想加速比')
ax2.set_xlabel('num_proc')
ax2.set_ylabel('Speedup (vs proc=1)')
ax2.set_title('实际加速比')
ax2.legend()

plt.tight_layout()
plt.show()
```

> **注意**：`num_proc` 不是越大越好。每个进程都有启动开销和内存占用，且进程间通信也有成本。通常 2-4 个进程是最佳选择。

---

## 二、Tokenize 的标准写法

## 2.1 黄金模板

以下是你会在 90% 的 HF 微调项目中看到的 tokenize 写法：

```python
from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("sew", "chnsenticorp", split="train[:200]")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=False,          # 不在这里做 padding！
        truncation=True,
        max_length=128,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names,
)

print(f"Tokenize 完成!")
print(f"列: {tokenized_dataset.column_names}")
print(f"input_ids 长度分布 (前 10 条): {[len(x) for x in tokenized_dataset['input_ids'][:10]]}")
```

**关键细节解释**：

| 参数 | 值 | 原因 |
|------|-----|------|
| `padding=False` | 不在此处 pad | 每个 sample 长度不同，先保持原样；padding 由 DataCollator 在组成 batch 时统一做 |
| `truncation=True` | 截断超长序列 | 防止个别极长样本导致 OOM |
| `max_length=128` | 最大长度 | 根据模型和任务需求调整；BERT 通常用 128 或 512 |
| `remove_columns=...` | 删除原始列 | tokenized 后不再需要原始文本，节省内存 |
| `batched=True` | 批量处理 | 大幅提升 tokenize 速度 |
| `num_proc=4` | 多进程 | 进一步加速 |

> **为什么 padding=False？**
>
> 这是新手最容易犯的错误之一。如果在 `map` 阶段就设置 `padding="max_length"`，所有样本都会被 padding 到相同的固定长度（如 128）。这有两个问题：
> 1. **浪费计算**：短句子也会被填充到 128，大量 padding token 参与无意义的计算
> 2. **浪费显存**：一个 batch 中如果大部分是短句，整个 batch 都按最长的那条来 pad
>
> 正确的做法是在 **DataCollator** 阶段按 batch 内的最大长度动态 padding（见下文）。

## 2.2 处理多种输入格式

有些任务的输入不止一列：

```python
# ===== NLI 任务（前提 + 假设）=====
nli_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def tokenize_nli(examples):
    return nli_tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=128,
    )

nli_ds = load_dataset("clue", "afqmc", split="train[:100]")
tokenized_nli = nli_ds.map(tokenize_nli, batched=True, num_proc=2,
                              remove_columns=nli_ds.column_names)

print(f"NLI Tokenize 结果:")
print(f"  input_ids[0] 长度: {len(tokenized_nli['input_ids'][0])}")

# ===== QA 任务（问题 + 上下文）=====
qa_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def tokenize_qa(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    tokenized = qa_tokenizer(
        questions,
        contexts,
        truncation="only_second",     # 只截断 context（保留完整 question）
        max_length=384,
        stride=128,                   # 允许长文档滑动窗口重叠
        return_overflowing_tokens=True,
    )

    return tokenized

qa_ds = load_dataset("cmrc_2018", split="train[:50]")
tokenized_qa = qa_ds.map(tokenize_qa, batched=True, num_proc=2)
print(f"\nQA Tokenize 结果:")
print(f"  input_ids 形状: {len(tokenized_qa['input_ids'])} 条 "
      f"(可能因 overflow 比原始更多)")
```

---

## 三、DataCollator —— 动态组装 Batch

当 `map` + `tokenizer` 处理完数据后，你得到的是一个所有样本长度不一的 Dataset。PyTorch 的 DataLoader 要求同一个 batch 中的张量形状一致，这就需要 **DataCollator** 来完成最后的 padding 和组装工作。

## 3.1 DataCollatorWithPadding —— 最常用的 Collator

```python
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataloader = DataLoader(
    tokenized_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator,
)

batch = next(iter(dataloader))

print("Batch 内容:")
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape={list(value.shape)}, dtype={value.dtype}")
    else:
        print(f"  {key}: {type(value)}")

import matplotlib.pyplot as plt
attention_mask = batch["attention_mask"]
seq_lengths = attention_mask.sum(dim=1).tolist()

plt.figure(figsize=(10, 4))
plt.bar(range(len(seq_lengths)), seq_lengths, color='steelblue')
plt.xlabel('Sample Index in Batch')
plt.ylabel('Actual Sequence Length')
plt.title(f'Dynamic Padding 效果\n(Batch size=8, 各样本实际长度)')
plt.axhline(y=max(seq_lengths), color='red', linestyle='--',
           label=f'Max length = {max(seq_lengths)}')
plt.legend()
plt.show()
```

输出：
```
Batch 内容:
  input_ids: shape=[8, varies], dtype=torch.int64
  attention_mask: shape=[8, varies], dtype=torch.int64
  token_type_ids: shape=[8, varies], dtype=torch.int64
```

注意 `shape=[8, varies]`——每个 batch 的实际形状取决于该 batch 中**最长样本的长度**，这就是**动态 padding**的核心优势。

## 3.2 各种 DataCollator 对比

```python
from transformers import (
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
)

collators = {
    "DefaultDataCollator": DefaultDataCollator(),
    "DataCollatorWithPadding": DataCollatorWithPadding(tokenizer=tokenizer),
    "DataCollatorForLanguageModeling (MLM)": DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    ),
}

print("=" * 70)
print(f"{'Collator 类型':<40} {'主要用途':<30}")
print("=" * 70)
print(f"{'DefaultDataCollator':<40} {'简单堆叠 tensor（已 padded 的数据）':<30}")
print(f"{'DataCollatorWithPadding':<40} {'动态 padding 到 batch 内最大长度':<30}")
print(f"{'DataCollatorForLanguageModeling':<40} {'MLM: 自动  部分token':<30}")
print(f"{'DataCollatorForSeq2Seq':<40} {'Encoder-Decoder: 分别处理 src/tgt':<30}")
```

### DefaultDataCollator

最简单的 collator，假设数据已经被正确地 padding 过了：

```python
default_collator = DefaultDataCollator()

pre_padded_ds = tokenized_dataset.map(
    lambda x: tokenizer(
        x["input_ids"] if isinstance(x["input_ids"][0], int) else x["input_ids"],
        padding="max_length",
        max_length=128,
        return_tensors=None,
    ) if False else x,
    batched=True,
)

batch = default_collator([tokenized_dataset[i] for i in range(4)])
```

适用于你已经手动处理好 padding 的情况，或者非文本类数据（如分类标签）。

### DataCollatorForLanguageModeling —— MLM 的核心组件

这是 BERT 预训练的关键组件，它会**随机 mask 掉一部分 token**：

```python
mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,  # 15% 的概率被 mask
)

samples = [tokenized_dataset[i] for i in range(4)]
batch = mlm_collator(samples)

print("\nMLM Collator 输出:")
print(f"  input_ids shape: {batch['input_ids'].shape}")
print(f"  labels shape: {batch['labels'].shape}")

input_ids = batch['input_ids'][0].tolist()
labels = batch['labels'][0].tolist()

print(f"\n  第一个样本的 MLM 处理结果:")
print(f"  input_ids: {input_ids[:20]}...")
print(f"  labels:     {labels[:20]}...")

mask_positions = [(i, input_ids[i]) for i in range(len(input_ids))
                  if input_ids[i] == tokenizer.mask_token_id]
                 or (labels[i] != -100 and labels[i] != input_ids[i])]
if mask_positions:
    print(f"\n  被 Mask 的位置 (共 {len(mask_positions)} 个):")
    for pos, token_id in mask_positions[:5]:
        original_token = tokenizer.decode([labels[pos]])
        print(f"    pos={pos}:  (原始: '{original_token}')")
```

**MLM 的三种策略**（由 `mlm_probability` 控制）：
- 80% 概率 → 替换为 ``
- 10% 概率 → 替换为随机 token
- 10% 概率 → 保持不变（但仍然需要预测）

### DataCollatorForSeq2Seq —— Encoder-Decoder 专用

用于 T5/BART 等序列到序列模型：

```python
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

seq2seq_tokenizer = AutoTokenizer.from_pretrained("t5-base")

seq2seq_collator = DataCollatorForSeq2Seq(
    tokenizer=seq2seq_tokenizer,
    model=None,
    padding="longest",
    max_length=128,
    max_target_length=64,
    return_tensors="pt",
)

src_samples = ["translate English to German: I love programming"]
tgt_samples = ["Ich liebe Programmieren"]

batch = seq2seq_collator([
    {"input_ids": seq2seq_tokenizer.encode(s) for s in src_samples},
    {"labels": seq2seq_tokenizer.encode(t) for t in tgt_samples},
][0])

print(f"Seq2Seq batch keys: {list(batch.keys())}")
```

---

## 四、完整实战：从 CSV 到 DataLoader

让我们把以上所有知识整合起来，构建一个从原始文件到可训练 DataLoader 的完整流水线：

```python
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
)
from torch.utils.data import DataLoader

def build_classification_pipeline(csv_path, model_name, text_col="text",
                                  label_col="label", max_length=128,
                                  val_ratio=0.1, batch_size=32, num_proc=4):
    """
    构建完整的文本分类数据处理流水线

    Args:
        csv_path: CSV 文件路径
        model_name: 预训练模型名称
        text_col: 文本列名
        label_col: 标签列名
        max_length: 最大序列长度
        val_ratio: 验证集比例
        batch_size: 训练 batch size
        num_proc: map 函数的进程数

    Returns:
        (train_dataloader, val_dataloader, tokenizer, label_map)
    """

    print("=" * 60)
    print("Step 1: 加载原始数据")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"  原始数据: {len(df)} 行, 列: {list(df.columns)}")
    print(f"  标签分布:\n{df[label_col].value_counts()}")

    unique_labels = sorted(df[label_col].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    print(f"  标签映射: {label2id}")

    df["label_id"] = df[label_col].map(label2id)

    raw_dataset = Dataset.from_pandas(df[["text", "label_id"]])

    print(f"\n{'=' * 60}")
    print("Step 2: 数据划分")
    print("=" * 60)

    split_ds = raw_dataset.train_test_split(test_size=val_ratio, seed=42, stratify_by_column="label_id")
    print(f"  Train: {len(split_ds['train']):,} 条")
    print(f"  Val:   {len(split_ds['test']):,} 条")

    print(f"\n{'=' * 60}")
    print("Step 3: Tokenization")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_length,
        )

    tokenized_splits = {}
    for split_name in ["train", "test"]:
        tokenized_splits[split_name] = split_ds[split_name].map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc,
            remove_columns=[text_col],
            desc=f"Running tokenization on {split_name}",
        )

    lengths = [len(ids) for ids in tokenized_splits["train"]["input_ids"]]
    print(f"  Tokenize 完成!")
    print(f"  序列长度统计: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")

    print(f"\n{'=' * 60}")
    print("Step 4: 构建 DataLoader")
    print("=" * 60)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenized_splits["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
        drop_last=True,
    )

    val_loader = DataLoader(
        tokenized_splits["test"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    batch = next(iter(train_loader))
    print(f"  Train DataLoader: {len(train_loader)} batches")
    print(f"  Val DataLoader:   {len(val_loader)} batches")
    print(f"  Sample batch shapes:")
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: {list(v.shape)}")

    return train_loader, val_loader, tokenizer, id2label


# 使用示例（模拟数据）
import tempfile
import os

dummy_data = {
    "text": [
        "这个产品非常好用，强烈推荐给大家",
        "质量太差了，用了两天就坏了",
        "还可以吧，没有特别惊艳的地方",
        "客服态度很好，物流也很快",
        "价格实惠，性价比很高",
        "包装很精美，送礼很有面子",
    ] * 50,
    "label": ["正面", "负面", "中性", "正面", "正面", "正面"] * 50,
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
    pd.DataFrame(dummy_data).to_csv(f.name, index=False)
    tmp_csv = f.name

train_dl, val_dl, tok, label_map = build_classification_pipeline(
    csv_path=tmp_csv,
    model_name="uer/roberta-base-finetuned-chinanews-chinese",
    text_col="text",
    label_col="label",
    max_length=64,
    batch_size=8,
)

os.unlink(tmp_csv)

print(f"\n✅ 流水线构建成功! 标签映射: {label_map}")
```

---

## 五、Streaming 模式 —— TB 级数据的处理方案

如果你的数据集大到无法全部放入磁盘（比如几百 GB 的语料库），可以使用 Streaming 模式：

```python
from datasets import load_dataset

streaming_ds = load_dataset("json", data_files="huge_corpus/*.jsonl", streaming=True)

print(f"Streaming Dataset 类型: {type(streaming_ds)}")
print(f"迭代访问（不会一次性加载到内存）:")

count = 0
for example in streaming_ds["train"]:
    print(f"  [{count}] {str(example)[:100]}...")
    count += 1
    if count >= 5:
        break

print(f"\n总样本数未知（流式），可以无限迭代")
```

**Streaming 的注意事项**：
- **不支持随机访问**：不能 `ds[1000]`，只能顺序遍历或 `ds.skip(1000).take(10)`
- **不支持 shuffle（完全）**：只能设置 buffer_size 做近似洗牌
- **map 操作也是惰性的**：只在迭代到对应元素时才执行
- **适合场景**：预训练语料处理、在线学习、数据探索

---

## 小结

这一节我们掌握了从原始数据到模型输入的完整预处理流水线：

1. **map 函数**：Datasets 的核心工具，支持单条和批量（`batched=True`）两种模式；配合 `remove_columns` 清理中间字段；配合 `num_proc=N` 多进程加速
2. **Tokenize 标准写法**：`padding=False`（不在 map 阶段 pad）、`truncation=True`、`remove_columns` 删除原始列、`batched=True` + `num_proc` 并行化——这是 90% 项目中的标准模板
3. **DataCollator 四剑客**：
   - `DefaultDataCollator`：简单堆叠（已 padded 的数据）
   - `DataCollatorWithPadding`：**最常用**，动态 padding 到 batch 内最大长度
   - `DataCollatorForLanguageModeling`：MLM 预训练，自动 15% mask
   - `DataCollatorForSeq2Seq`：T5/BART 等 Encoder-Decoder 模型
4. **完整流水线**：CSV → DataFrame → Dataset → train_test_split → tokenize(map) → DataLoader(collator)，一条龙服务
5. **Streaming 模式**：TB 级数据的解决方案，惰性加载+顺序迭代

下一节我们将讨论数据增强与清洗技术——如何让有限的数据发挥更大价值。
