# 自定义数据集：从本地文件到 Hugging Face Hub

## 这一节讲什么？

前面的章节我们一直在使用 HF Hub 上已有的公开数据集（GLUE、SQuAD、IMDb 等）。但在实际项目中，你几乎总是需要处理**自己的数据**——可能是公司内部的标注数据、从数据库导出的文本、爬取的语料库等。

这一节，我们将掌握自定义数据集的完整生命周期：

1. **从常见格式构建**：CSV、JSON、Parquet、DataFrame → Dataset
2. **上传到 Hub**：分享你的数据集给社区或团队
3. **Dataset Card 编写**：好的文档是数据集被引用的前提
4. **Builder 模式**：为全新格式编写可复用的数据加载器
5. **流式加载**：处理 TB 级别超大数据

---

## 一、从各种格式创建 Dataset

## 1.1 从 Python 字典/列表

最简单直接的方式：

```python
from datasets import Dataset, DatasetDict

data = {
    "text": [
        "今天天气真好",
        "这个产品太差了",
        "物流很快，包装完好",
    ],
    "label": [1, 0, 1],
    "source": ["微博", "京东", "淘宝"],
}

dataset = Dataset.from_dict(data)
print(dataset)
print(f"\n第一个样本: {dataset[0]}")
print(f"列名: {dataset.column_names}")
print(f"特征类型: {dataset.features}")
```

## 1.2 从 Pandas DataFrame

这是最常见的场景——你的数据可能已经在 DataFrame 中了：

```python
import pandas as pd
from datasets import Dataset

df = pd.DataFrame({
    "text": ["好评", "差评", "中评"] * 100,
    "sentiment": ["positive", "negative", "neutral"] * 100,
    "rating": [5, 1, 3] * 100,
})

dataset = Dataset.from_pandas(df)
print(f"从 DataFrame 创建: {dataset}")

# 只选择需要的列
selected = Dataset.from_pandas(df[["text", "sentiment"]])
print(f"选择列后: {selected}")
```

> **注意**：`from_pandas` 会自动推断类型。如果 DataFrame 的 `object` 类型列包含混合内容（如 None 和字符串），可能导致 Arrow 类型推断失败。建议先用 `df[col].astype(str)` 或 `df.dropna()` 清洗数据。

## 1.3 从 JSON/JSONL 文件

```python
from datasets import load_dataset
import json
import tempfile
import os

json_data = [
    {"text": "样本一", "label": 0, "metadata": {"source": "web"}},
    {"text": "样本二", "label": 1, "metadata": {"source": "api"}},
    {"text": "样本三", "label": 0, "metadata": {"source": "db"}},
]

with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for item in json_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    jsonl_path = f.name

ds = load_dataset("json", data_files=jsonl_path, split="train")
print(f"从 JSONL 加载: {ds}")
print(f"嵌套字段 (metadata): {ds[0]['metadata']}")

os.unlink(jsonl_path)
```

## 1.4 从 Parquet 文件（推荐用于生产）

```python
def demonstrate_parquet_advantages():
    """展示 Parquet 格式的优势"""

    import pandas as pd
    import tempfile
    import os

    n_rows = 100000
    df = pd.DataFrame({
        "id": range(n_rows),
        "text": ["这是一段测试文本，用于比较不同存储格式的性能差异。"] * n_rows,
        "label": [i % 3 for i in range(n_rows)],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        parquet_path = os.path.join(tmpdir, "data.parquet")

        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)

        csv_size = os.path.getsize(csv_path) / 1024 / 1024
        pq_size = os.path.getsize(parquet_path) / 1024 / 1024

        print(f"文件大小对比 ({n_rows:,} 行):")
        print(f"  CSV:     {csv_size:.2f} MB")
        print(f"  Parquet: {pq_size:.2f} MB")
        print(f"  压缩比: {csv_size/pq_size:.1f}x")

        import time
        start = time.time()
        ds_csv = load_dataset("csv", data_files=csv_path, split="train")
        csv_load_time = time.time() - start

        start = time.time()
        ds_pq = load_dataset("parquet", data_files=parquet_path, split="train")
        pq_load_time = time.time() - start

        print(f"\n加载速度:")
        print(f"  CSV:     {csv_load_time:.3f}s")
        print(f"  Parquet: {pq_load_time:.3f}s")
        print(f"  加速比:  {csv_load_time/max(pq_load_time, 0.001):.1f}x")

demonstrate_parquet_advantages()
```

---

## 二、上传到 Hugging Face Hub

## 2.1 准备工作

```python
from huggingface_hub import login

login()  # 会提示输入 Token（从 https://huggingface.co/settings/tokens 获取）
```

## 2.2 上传数据集

```python
from datasets import Dataset, DatasetDict

dataset = Dataset.from_dict({
    "text": ["示例文本一", "示例文本二", "示例文本三"],
    "label": [0, 1, 0],
})

# 方式 1: 推送到 Hub（公开）
dataset.push_to_hub("your-username/my-dataset-name")

# 方式 2: 推送到 Hub（私有）
dataset.push_to_hub("your-username/private-dataset", private=True)

# 方式 3: 推送 DatasetDict（含 train/test 划分）
ddict = DatasetDict({
    "train": dataset,
    "test": dataset.select([0, 2]),
})
ddict.push_to_hub("your-username/my-dataset-with-splits")
```

## 2.3 Dataset Card —— 数据集的"身份证"

一个没有良好文档的数据集几乎不会被使用。Dataset Card 是你向用户说明数据集用途、结构、限制的关键文档：

```markdown
---
license: mit
task_categories:
  - text-classification
language:
  - zh
size_categories:
  - 1K<n<10K
---

# 数据集名称

## 数据集描述

简要描述这个数据集是什么、包含什么数据、收集方式等。

## 数据集结构

## 数据实例

提供一个具体的样例：

```
{'text': '...', 'label': 0}
```

## 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| text | string | 原始文本 |
| label | int64 | 类别标签 (0=负面, 1=正面) |

## 数据划分

| 划分 | 样本数 |
|------|--------|
| train | 800 |
| test | 200 |

## 使用示例

```python
from datasets import load_dataset

ds = load_dataset("your-username/dataset-name")
```

## 数据来源

描述数据的来源、标注过程、标注者信息等。

## 局限性与偏差

诚实说明数据集的已知问题和潜在偏差。

## 考虑事项

使用此数据集时需要注意的事项。
```

你可以通过以下方式添加 Dataset Card：

```python
from huggingface_hub import HfApi

api = HfApi()

card_content = """
---
language:
  - zh
license: mit
---

# 我的中文情感分析数据集

## 描述
这是一个用于中文情感分析的示例数据集...

## 使用方法
...
"""

api.upload_file(
    path_or_fileobj=card_content.encode("utf-8"),
    path_in_repo="README.md",
    repo_id="your-username/my-dataset-name",
    repo_type="dataset",
)
```

---

## 三、Builder 模式 —— 为新格式创建数据加载器

如果你的数据格式很特殊（比如公司内部的自定义二进制格式），或者你想让社区其他人也能方便地加载数据集，可以写一个 **Dataset Builder**：

## 3.1 Builder 模板

```python
from datasets import GeneratorBasedBuilder, DatasetInfo, Features, Value, ClassLabel, SplitGenerator
import json

class MyCustomDatasetBuilder(GeneratorBasedBuilder):
    """
    自定义数据集 Builder 模板

    继承 GeneratorBasedBuilder 并实现以下方法:
    - _info(): 定义数据集的元信息（特征、版本等）
    - _split_generators(): 定义如何生成各个数据划分
    - _generate_examples(): 定义如何逐条产生样本
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description="我的自定义数据集描述",
            features=Features({
                "text": Value("string"),
                "label": ClassLabel(names=["negative", "positive"]),
                "source": Value("string"),
            }),
            homepage="https://example.com",
            citation="""@misc{my_dataset_2024,
              title={My Custom Dataset},
              author={Your Name},
              year={2024}
            }""",
        )

    def _split_generators(self):
        return [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": "data/train.jsonl"},
            ),
            SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": "data/test.jsonl"},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())
                yield idx, {
                    "text": data["content"],
                    "label": data["label"],
                    "source": data.get("source", "unknown"),
                }
```

## 3.2 注册和使用 Builder

```python
import datasets

# 注册你的 Builder
datasets.register_module("my_custom_builder.py")

# 像使用内置数据集一样加载
ds = load_dataset("path/to/my_custom_builder.py")
print(ds)
```

---

## 四、流式加载（Streaming）—— 处理超大规模数据

当数据集大到无法全部放入磁盘时（比如整个 Common Crawl 语料），`streaming=True` 是唯一可行的方案：

```python
def streaming_demo():
    """演示流式加载的使用模式"""
    from datasets import load_dataset

    stream_ds = load_dataset(
        "json",
        data_files="huge_corpus/*.jsonl",
        streaming=True,
        split="train",
    )

    print(f"类型: {type(stream_ds)}")
    print(f"迭代访问（惰性加载）:")

    count = 0
    for example in stream_ds:
        if count < 5:
            text_preview = str(example)[:120]
            print(f"  [{count}] {text_preview}")
        count += 1
        if count >= 10:
            break

    print(f"\n已迭代 {count} 条（可以无限继续）")

    # 流式 map 也是惰性的
    def preprocess(example):
        example["length"] = len(example.get("text", ""))
        return example

    processed_stream = stream_ds.map(preprocess)

    print("\n流式 map 后（同样惰性）:")
    for i, ex in enumerate(processed_stream):
        if i >= 3:
            break
        print(f"  [{i}] length={ex['length']}")


streaming_demo()
```

**Streaming 的典型使用场景**：

| 场景 | 为什么用 Streaming |
|------|-------------------|
| 预训练语料（100GB+） | 无法全部下载到本地 |
| 在线学习 | 数据持续增长，不需要全量加载 |
| 数据探索 | 只想快速预览几条样本 |
| 管道式处理 | 边读边处理边输出 |

**Streaming 的限制**：

| 操作 | 支持？ | 说明 |
|------|--------|------|
| `ds[0]` | ❌ | 不支持随机索引 |
| `ds.shuffle()` | ⚠️ | 近似洗牌（buffer_size 控制）|
| `ds.select()` | ❌ | 不能按索引选子集 |
| `ds.filter()` | ✅ | 支持过滤 |
| `ds.map()` | ✅ | 惰性执行 |
| `ds.take(N)` | ✅ | 取前 N 条 |
| `ds.skip(N)` | ✅ | 跳过前 N 条 |

---

## 五、端到端实战：构建并发布一个完整的自定义数据集

下面是一个完整的从零到发布的流程：

```python
def full_lifecycle_demo():
    """完整的自定义数据集生命周期演示"""

    from datasets import Dataset, DatasetDict, load_dataset
    from huggingface_hub import HfApi, create_repo
    import tempfile
    import os

    print("=" * 60)
    print("Step 1: 准备原始数据")
    print("=" * 60)

    raw_data = {
        "text": [
            "这款手机拍照效果非常好，电池也很耐用",
            "质量太差了，用了两天就坏了，不推荐购买",
            "还可以吧，对得起这个价格",
            "超出预期的好产品！客服态度也很好",
            "物流太慢了，等了一周才到货",
            "包装精美，送礼很有面子，五星好评",
        ] * 50,
        "label": ["正面", "负面", "中性", "正面", "负面", "正面"] * 50,
        "category": ["数码", "数码", "数码", "数码", "物流", "礼品"] * 50,
    }

    ds = Dataset.from_dict(raw_data)
    print(f"原始数据集: {len(ds)} 条")
    print(f"标签分布: {dict(__import__('collections').Counter(ds['label']))}")

    print("\n" + "=" * 60)
    print("Step 2: 数据清洗与增强")
    print("=" * 60)

    ds = ds.filter(lambda x: len(x["text"].strip()) > 5)
    ds = ds.map(lambda x: {"text_length": len(x["text"])}, num_proc=2)
    print(f"清洗后: {len(ds)} 条")

    print("\n" + "=" * 60)
    print("Step 3: 划分训练/验证/测试集")
    print("=" * 60)

    splits = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    val_test = splits["test"].train_test_split(test_size=0.5, seed=42)

    final_ds = DatasetDict({
        "train": splits["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    for split_name, split_ds in final_ds.items():
        print(f"  {split_name}: {len(split_ds):,} 条")

    print("\n" + "=" * 60)
    print("Step 4: 保存到本地")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "my_sentiment_dataset")
        final_ds.save_to_disk(save_path)

        loaded = Dataset.load_from_disk(save_path)
        assert len(loaded["train"]) == len(final_ds["train"])
        print(f"✅ 保存/加载验证通过!")

        print(f"\n目录结构:")
        for root, dirs, files in os.walk(save_path):
            level = root.replace(save_path, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath) / 1024
                print(f"{subindent}{file} ({size:.1f}KB)")

    print("\n" + "=" * 60)
    print("Step 5: 导出为多种格式")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        final_ds["train"].to_csv(os.path.join(tmpdir, "train.csv"))
        final_ds["train"].to_json(os.path.join(tmpdir, "train.json"))
        final_ds["train"].to_parquet(os.path.join(tmpdir, "train.parquet"))

        formats = [("csv", ".csv"), ("json", ".json"), ("parquet", ".parquet")]
        for fmt_name, ext in formats:
            fpath = os.path.join(tmpdir, f"train{ext}")
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fmt_name:>8s}: {size_kb:>8.1f} KB")

    print("\n✅ 全部步骤完成!")


full_lifecycle_demo()
```

---

## 小结

这一节我们掌握了自定义数据集的完整生命周期管理：

1. **多种创建方式**：`from_dict()`（Python 对象）、`from_pandas()`（DataFrame）、`load_dataset("json/csv/parquet")`（文件）；**推荐 Parquet 用于生产环境**——压缩比 5-10x、加载速度 10-20x
2. **Hub 发布**：`push_to_hub()` 一行上传；支持公开/私有；**必须编写 Dataset Card**（README.md）——包含描述、字段说明、使用示例、局限性声明
3. **Builder 模式**：继承 `GeneratorBasedBuilder`，实现 `_info()`、`split_generators()`、`generate_examples()` 三个方法；适用于需要复用或开源的数据加载器
4. **Streaming 模式**：`streaming=True` 实现 TB 级数据的惰性加载；支持 filter/map/take/skip 等操作；不支持随机索引和完全 shuffle
5. **完整生命周期**：准备数据 → 清洗增强 → 划分数据集 → 本地保存 → 多格式导出 → Hub 发布

至此，**第5章 数据集与数据处理全部完成**！接下来进入核心章节——第6章模型微调 Fine-Tuning，这是让预训练模型适应你业务的关键技能。
