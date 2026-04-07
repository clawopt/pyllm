# Datasets 库核心功能：高效处理 TB 级 NLP 数据

## 这一节讲什么？

在机器学习项目中，有一句老话叫"数据决定了模型的上限"。无论你的模型架构多么精妙、训练技巧多么高超，如果数据质量差或者数据处理流程低效，最终效果都不会好。

Hugging Face 的 `datasets` 库就是为了解决**大规模数据集的高效加载与处理**而设计的。它和 `transformers` 是最佳搭档——一个负责模型，一个负责数据。这一节，我们将学习：

1. 为什么不用 Pandas/原生 Python 处理 ML 数据？
2. 如何从 Hub 加载、从本地文件构建数据集
3. Dataset / DatasetDict / Arrow Table 的底层架构
4. 切片、过滤、排序、洗牌等核心操作
5. 常用数据集一览

---

## 一、为什么需要 Datasets 库？

## 传统方式的痛点

很多人处理 NLP 数据的第一反应是用 Pandas：

```python
import pandas as pd

df = pd.read_csv("large_dataset.csv")
print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

# 问题 1: 全部加载到内存
# 一个 10GB 的 CSV 文件可能需要 30-50GB 内存（因为 Python 对象的开销）

# 问题 2: 文本列的内存爆炸
df["text"] = df["text"].astype(str)  # 每个 Python 字符串对象 ~49 bytes overhead

# 问题 3: 无法分布式处理
# Pandas 是单机的
```

## Datasets 的解决方案

`datasets` 库基于 **Apache Arrow** 列式存储格式，解决了上述所有问题：

```python
from datasets import load_dataset, Dataset

dataset = load_dataset("glue", "mrpc", split="train")

print(f"数据集类型: {type(dataset)}")
print(f"样本数量: {len(dataset):,}")
print(f"特征列: {dataset.column_names}")
print(f"第一个样本:")
print(dataset[0])
```

输出：
```
数据集类型: <class 'datasets.arrow_dataset.Dataset'>
样本数量: 3,668
特征列: ['sentence1', 'sentence2', 'label', 'idx']
第一个样本:
{
    'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
    'sentence2': 'Referring to him as only " the witness" , Amrozi accused his brother of deliberately distorting his evidence .',
    'label': 1,
    'idx': 0
}
```

## 核心优势对比

| 特性 | Pandas | Datasets (Arrow) |
|------|--------|-------------------|
| **内存模式** | 全量加载到 RAM | **内存映射 (mmap)**，按需读取 |
| **字符串存储** | Python 对象（高开销） | **Arrow 字符串数组**（紧凑） |
| **10GB CSV 内存占用** | ~30-50 GB | **~10 GB（1:1）** |
| **100GB+ 数据集** | OOM 崩溃 | **轻松处理（磁盘足够即可）** |
| **并行处理** | 有限支持 | **内置多进程 map** |
| **分布式** | 不支持 | **可与 Ray/Dask 集成** |
| **懒加载** | 不支持 | **Streaming 模式** |
| **与 HF 生态集成** | 需手动转换 | **原生支持** |

---

## 二、加载数据集的三种方式

## 方式 1：从 Hugging Face Hub 加载

Hub 上有数万个公开数据集，一行代码即可下载：

```python
from datasets import load_dataset

# ===== GLUE 基准套件 =====
glue = load_dataset("glue", "sst2")  # SST-2 情感分析
print(f"GLUE-SST2: {glue}")

# ===== SQuAD 问答数据集 =====
squad = load_dataset("squad")
print(f"\nSQuAD: {squad}")
print(f"  训练集: {len(squad['train']):,} 条")
print(f"  验证集: {len(squad['validation']):,} 条")
print(f"  示例:\n{squad['train'][0]}")

# ===== CNN/DailyMail 摘要数据集 =====
cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")
print(f"\nCNN/DailyMail: {cnn_dailymail}")
```

**DatasetDict 结构**：大多数数据集返回的是 `DatasetDict`，它是一个字典，键是数据划分（train/validation/test），值是对应的 `Dataset`：

```
DatasetDict({
    train: Dataset(num_rows=67349, features=[...])
    validation: Dataset(num_rows=1821, features=[...])
    test: Dataset(num_rows=1821, features=[...])
})
```

## 方式 2：从本地文件加载

```python
from datasets import load_dataset

# ===== 从 JSON 加载 =====
json_dataset = load_dataset("json", data_files="data/train.jsonl")
print(f"JSON 数据集: {json_dataset['train']}")

# ===== 从 CSV 加载 =====
csv_dataset = load_dataset("csv", data_files="data/reviews.csv", delimiter=",")
print(f"CSV 数据集: {csv_dataset['train']}")

# ===== 从多个文件加载 =====
multi_file = load_dataset(
    "csv",
    data_files={
        "train": ["data/part_001.csv", "data/part_002.csv", "data/part_003.csv"],
        "test": "data/test.csv",
    }
)
print(f"多文件数据集: {multi_file}")

# ===== 从 Parquet 加载（推荐！高效格式）=====
parquet_ds = load_dataset("parquet", data_files="data/large_data.parquet")
print(f"Parquet 数据集: {parquet_ds['train']}")
```

> **为什么推荐 Parquet？**
>
> - **列式压缩**：通常比 CSV 小 5-10 倍
> - **类型保留**：不会像 CSV 那样把所有东西都当字符串
> - **分区支持**：可以按某个字段分目录存储，查询时自动跳过无关分区
> - **读取速度快**：不需要解析文本，直接读二进制

## 方式 3：从 Python 对象创建

```python
from datasets import Dataset

data = {
    "text": [
        "今天天气真好",
        "这个产品太差了",
        "还可以吧，一般般",
        "非常满意，强烈推荐！",
    ],
    "label": [1, 0, 1, 1],
    "source": ["微博", "京东", "微信", "淘宝"],
}

dataset = Dataset.from_dict(data)
print(f"自定义数据集: {dataset}")
print(f"\n前两条:")
for i in range(2):
    print(f"  {i}: {dataset[i]}")
```

---

## 三、Dataset 的内部结构 —— Apache Arrow

理解 Dataset 的底层结构有助于你写出更高效的代码：

## Arrow 表的结构

```python
def inspect_arrow_structure():
    """检查 Dataset 的底层 Arrow 结构"""
    from datasets import load_dataset

    ds = load_dataset("glue", "mrpc", split="train")

    print("=" * 60)
    print("Dataset 内部结构分析")
    print("=" * 60)

    print(f"\nPython 类型: {type(ds)}")
    print(f"底层数据表: {ds.data.__class__.__name__}")
    print(f"行数: {ds.num_rows:,}")
    print(f"列数: {ds.num_columns}")
    print(f"特征定义: {ds.features}")

    for col in ds.column_names:
        feature = ds.features[col]
        arr = ds.data[col]
        nbytes = arr.nbytes
        print(f"\n  列 '{col}':")
        print(f"    类型: {feature}")
        print(f"    Arrow 类型: {arr.type}")
        print(f"    内存大小: {nbytes / 1024:.1f} KB")
        if hasattr(arr, 'n_buffers'):
            print(f"    Buffer 数量: {arr.n_buffers}")

inspect_arrow_structure()
```

输出示例：
```
============================================================
Dataset 内部结构分析
============================================================

Python 类型: <class 'datasets.arrow_dataset.Dataset'>
底层数据表: Table
行数: 3,668
列数: 4
特征定义: {'sentence1': Value(dtype='string'), 'sentence2': Value(dtype='string'),
           'label': ClassLabel(names=['not_equivalent', 'equivalent']),
           'idx': Value(dtype='int64')}

  列 'sentence1':
    类型: Value(dtype='string')
    Arrow 类型: large_string
    内存大小: 1,234.5 KB
    Buffer 数量: 3  (offset buffer + data buffer + validity buffer)

  列 'label':
    类型: ClassLabel(names=['not_equivalent', 'equivalent'])
    Arrow 类型: int64
    内存大小: 28.9 KB
```

**关键概念**：Dataset 使用 **memory-mapped file（内存映射文件）**，这意味着数据实际上存储在磁盘上，操作系统会根据需要将部分数据映射到内存。这就是为什么你可以打开一个远大于物理内存的数据集——操作系统只加载你实际访问的那部分。

---

## 四、核心操作：切片、过滤、排序、洗牌

## 4.1 索引与切片

Dataset 支持类似 Python list 和 NumPy array 的索引操作：

```python
from datasets import load_dataset

ds = load_dataset("glue", "sst2", split="train")

# 单条访问
sample_0 = ds[0]
print(f"第 0 条: label={sample_0['label']}, text={sample_0['sentence'][:50]}...")

# 切片访问（返回字典，每个 key 是一个列表）
batch = ds[10:13]
print(f"\n切片 [10:13]: 共 {len(batch['sentence'])} 条")
for i in range(3):
    print(f"  [{10+i}] {batch['sentence'][i][:40]}... → {batch['label'][i]}")

# 步长切片
every_100th = ds[::100]  # 每 100 条取一条
print(f"\n每 100 条采样: {len(every_100th['sentence'])} 条")

# 列选择（只取特定列）
text_only = ds["sentence"][:5]
print(f"\n只取 text 列的前 5 条:")
for t in text_only:
    print(f"  {t[:60]}...")
```

## 4.2 select —— 按索引选取子集

```python
indices = [0, 42, 100, 999, 5000]
subset = ds.select(indices)

print(f"select 后的子集: {len(subset)} 条")
print(f"原始索引被保留: {subset['idx'][:5]}")
```

## 4.3 filter —— 条件过滤

`filter` 是最常用的操作之一，它接受一个函数作为过滤条件：

```python
short_texts = ds.filter(lambda x: len(x["sentence"]) < 20)
print(f"短文本 (<20字符): {len(short_texts)} 条 ({len(short_texts)/len(ds)*100:.1f}%)")

positive_samples = ds.filter(lambda x: x["label"] == 1)
negative_samples = ds.filter(lambda x: x["label"] == 0)
print(f"正面情感: {len(positive_samples)} 条")
print(f"负面情感: {len(negative_samples)} 条")

# 复杂条件过滤
medium_positive = ds.filter(
    lambda x: x["label"] == 1 and 10 < len(x["sentence"]) < 50
)
print(f"中等长度正面样本: {len(medium_positive)} 条")
```

**性能提示**：`filter` 默认使用单进程。对于大数据集，加上 `num_proc` 参数启用多进程：

```python
fast_filter = ds.filter(lambda x: len(x["sentence"]) > 10, num_proc=4)
```

## 4.4 sort —— 排序

```python
sorted_by_length = ds.sort("sentence", reverse=True, keep_in_memory=True)
print("\n最长的 3 条文本:")
for i in range(3):
    length = len(sorted_by_length[i]["sentence"])
    print(f"  [{length:>4d} 字符] {sorted_by_length[i]['sentence'][:80]}...")

sorted_by_label = ds.sort("label")
print("\n按 label 排序后的前 10 个 label:")
print(sorted_by_label["label"][:10])
```

> **注意 `keep_in_memory` 参数**：默认情况下，sort 操作会在磁盘上创建临时文件进行外部排序（适合超大集合）。如果你确定结果能放进内存，设置 `keep_in_memory=True` 可以加速。

## 4.5 shuffle —— 洗牌

```python
shuffled = ds.shuffle(seed=42)
print(f"洗牌后前 5 个 label: {shuffled['label'][:5]}")
print(f"原始前 5 个 label: {ds['label'][:5]}")

# 可控制洗牌范围（用于快速预览）
small_sample = ds.shuffle(seed=42).select(range(100))
print(f"\n随机抽样 100 条: {len(small_sample)} 条")
```

**关于随机种子的最佳实践**：
- **固定 seed** 用于实验可复现性：`shuffle(seed=42)`
- **不设 seed** 用于生产环境的数据增强：每次 shuffle 结果不同
- **不同层级的 seed**：全局 seed 控制整体流程，局部 seed 控制单个操作

## 4.6 train_test_split —— 数据划分

```python
split_ds = ds.train_test_split(test_size=0.1, seed=42)
print(f"划分结果:")
print(f"  train: {len(split_ds['train']):,} 条 ({len(split_ds['train'])/len(ds)*100:.1f}%)")
print(f"  test:  {len(split_ds['test']):,} 条 ({len(split_ds['test'])/len(ds)*100:.1f}%)")

# 分层划分（保持类别比例）
stratified_split = ds.train_test_split(
    test_size=0.1,
    seed=42,
    stratify_by_column="label"
)
print(f"\n分层划分后各类别分布:")
from collections import Counter
train_dist = Counter(stratified_split["train"]["label"])
test_dist = Counter(stratified_split["test"]["label"])
print(f"  Train: {dict(train_dist)}")
print(f"  Test:  {dict(test_dist)}")
```

---

## 五、常用数据集一览

## NLP 经典基准

```python
def showcase_popular_datasets():
    """展示常用的 NLP 数据集"""

    datasets_info = [
        {
            "name": "GLUE",
            "loader": ('load_dataset', {"path": "glue"}),
            "tasks": "文本分类、相似度、NLI 等 9 种任务",
            "size": "~10K-400K/任务",
            "use_case": "通用 NLP 基准测试",
        },
        {
            "name": "SQuAD",
            "loader": ('load_dataset', {"path": "squad"}),
            "tasks": "抽取式问答",
            "size": "Train: 87K, Dev: 10K",
            "use_case": "阅读理解 / QA 系统",
        },
        {
            "name": "CNN/DailyMail",
            "loader": ('load_dataset', {"path": "cnn_dailymail", "version": "3.0.0"}),
            "tasks": "新闻摘要",
            "size": "Train: 287K, Val: 13K, Test: 11K",
            "use_case": "摘要生成模型评估",
        },
        {
            "name": "WMT (翻译)",
            "loader": ('load_dataset', {"path": "wmt14", "config": "de-en"}),
            "tasks": "机器翻译",
            "size": "4.5M 句对 (de→en)",
            "use_case": "翻译系统训练/评估",
        },
        {
            "name": "IMDb (影评)",
            "loader": ('load_dataset', {"path": "imdb"}),
            "tasks": "二分类情感分析",
            "size": "Train: 25K, Test: 25K",
            "use_case": "情感分析入门教学",
        },
        {
            "name": "AG News (新闻)",
            "loader": ('load_dataset', {"path": "ag_news"}),
            "tasks": "4 分类主题分类",
            "size": "Train: 120K, Test: 7.6K",
            "use_case": "文本分类实战",
        },
    ]

    print("=" * 90)
    print(f"{'数据集名称':<16} {'任务':<24} {'规模':<22} {'典型用途'}")
    print("=" * 90)

    for info in datasets_info:
        print(f"{info['name']:<16} {info['tasks']:<24} {info['size']:<22} {info['use_case']}")

showcase_popular_datasets()
```

## 中文数据集

```python
chinese_datasets = [
    ("LCQMC", "句子语义相似度", "load_dataset('clue', 'lcqmc')"),
    ("ChnSentiCorp", "中文情感分析", "load_dataset('sew', 'chnsenticorp')"),
    ("CLUE (综合)", "中文 NLP 基准", "load_dataset('clue')"),
    ("WMT 中文翻译", "中英翻译", "load_dataset('wmt19', 'zh-en')"),
]

print("\n常用中文数据集:")
for name, task, loader in chinese_datasets:
    print(f"  {name:<16} {task:<20} {loader}")
```

## 快速浏览任意数据集

```python
def quick_browse(dataset_name, config=None, num_examples=3):
    """快速浏览任意数据集"""
    from datasets import load_dataset

    try:
        kwargs = {"path": dataset_name}
        if config:
            kwargs["config"] = config

        ds = load_dataset(**kwargs, trust_remote_code=True)

        if isinstance(ds, dict):
            for split_name, split_ds in ds.items():
                print(f"\n=== Split: {split_name} ({len(split_ds):,} 条) ===")
                for i in range(min(num_examples, len(split_ds))):
                    sample = split_ds[i]
                    preview = {k: str(v)[:80] for k, v in sample.items()}
                    print(f"  [{i}]: {preview}")
        else:
            print(f"\n=== {dataset_name} ({len(ds):,} 条) ===")
            for i in range(min(num_examples, len(ds))):
                sample = ds[i]
                preview = {k: str(v)[:80] for k, v in sample.items()}
                print(f"  [{i}]: {preview}")

    except Exception as e:
        print(f"加载数据集 '{dataset_name}' 失败: {e}")


quick_browse("imdb", num_examples=2)
```

---

## 六、数据集保存与加载

## 保存到本地

```python
from datasets import load_dataset, Dataset

ds = load_dataset("imdb", split="train").shuffle(seed=42).select(range(100))

ds.save_to_disk("./my_imdb_subset")
print(f"已保存到 ./my_imdb_subset")

loaded_ds = Dataset.load_from_disk("./my_imdb_subset")
print(f"重新加载: {loaded_ds}")
assert len(loaded_ds) == 100
print("✅ 保存/加载一致性验证通过!")
```

## 保存为其他格式

```python
ds.to_csv("./my_data.csv")          # 导出为 CSV
ds.to_json("./my_data.json")         # 导出为 JSON
ds.to_parquet("./my_data.parquet")   # 导出为 Parquet（推荐）
ds.to_arrow("./my_data.arrow")       # 导出为 Arrow 格式
```

## 上传到 Hub

```python
from huggingface_hub import login

login()  # 输入 HF Token

ds.push_to_hub("username/my-dataset-name", private=True)
print("已上传到 Hub!")
```

---

## 常见陷阱与性能建议

> **陷阱 1：用循环逐条处理数据**
>
> ```python
> # ❌ 慢！Python 循环
> results = []
> for i in range(len(ds)):
>     results.append(process(ds[i]))
>
> # ✅ 快！用 map 函数（下一节详细讲）
> results = ds.map(process, batched=True, num_proc=4)
> ```

> **陷阱 2：先 shuffle 再 filter 导致全量扫描**
>
> ```python
> # 低效：shuffle 创建了完整副本，然后 filter 又遍历一遍
> result = ds.shuffle(seed=42).filter(lambda x: x["label"] == 1)
>
> # 高效：先缩小范围再 shuffle
> result = ds.filter(lambda x: x["label"] == 1).shuffle(seed=42)
> ```

> **陷阱 3：忘记 Arrow 是内存映射的，误以为所有数据都在内存中**
>
> Dataset 对象本身很轻量（只存元数据），真正的大数据在磁盘上。`len(ds)` 很快，但 `ds[:]` 会尝试把所有数据加载到内存！

> **性能建议清单**：
> 1. 大数据集优先用 **Parquet** 格式（比 CSV 小 5-10x，读取快 10-20x）
> 2. 过滤/转换用 **map + batched=True + num_proc=N** 而不是 Python 循环
> 3. 只取需要的列：`ds.remove_columns(["unused_col"])`
> 4. 流式加载超大数据集：`load_dataset(..., streaming=True)`
> 5. 本地缓存：HF 会缓存下载的数据集，避免重复下载

---

## 小结

这一节我们全面掌握了 Datasets 库的核心功能：

1. **设计动机**：基于 Apache Arrow 的内存映射机制，解决 Pandas 处理大文件的 OOM 问题；字符串存储效率提升 5-10 倍；原生支持 TB 级数据
2. **三种加载方式**：从 Hub（`load_dataset("glue")`）、从本地文件（JSON/CSV/Parquet）、从 Python 对象（`Dataset.from_dict()`）
3. **DatasetDict 结构**：标准化的 train/validation/test 划分组织方式
4. **五大核心操作**：索引切片（`ds[0:10]`）、条件过滤（`ds.filter(fn)`）、排序（`ds.sort()`）、洗牌（`ds.shuffle()`）、划分（`ds.train_test_split()`）
5. **Arrow 底层**：内存映射文件 + 列式存储 = 磁盘大小的数据也能高效操作
6. **保存与分享**：`save_to_disk` / `push_to_hub` / 多格式导出

下一节我们将深入数据处理的核心——`map` 函数、DataCollator、以及从原始数据到 DataLoader 的完整流水线。
