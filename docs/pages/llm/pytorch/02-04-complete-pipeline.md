# 2.4 完整数据管道实战：从原始文本到 Training Batch

想象一下这个场景：你手里有一个 50GB 的 JSONL 语料文件，里面装着几千万条文本数据，你的目标是训练一个 LLaMA-7B 级别的语言模型。你打开 PyTorch 的官方文档，发现 Dataset 和 DataLoader 的 API 看起来都不复杂，于是信心满满地开始写代码。结果呢？跑起来之后 GPU 利用率只有 20%，内存占用飙升到系统卡死，一个 epoch 要跑三天，而且中间还时不时因为某个样本格式不对而崩溃。这不是夸张——几乎每个第一次做 LLM 训练的人都会踩这些坑。前面三节我们分别学习了 Dataset 与 DataLoader 的基础机制、Collate Function 处理变长序列的技巧、以及高效处理 TB 级数据的各种方案，但知识是零散的，就像你学会了怎么砌砖、怎么和水泥、怎么搭脚手架，但还没见过一栋完整的房子是怎么盖起来的。这一节的目标就是把所有碎片拼在一起，搭建一条从"磁盘上的原始文本文件"到"GPU 上准备好的训练 Batch"的完整数据管道，并且在这个过程中你会看到真实生产环境中需要考虑的各种细节问题，比如数据统计如何帮助你诊断 padding 浪费、验证函数如何确保管道不会在训练中途出错、以及什么时候该用自己的实现、什么时候该直接用 HuggingFace 的现成工具。

## 从原始文本到 Batch：完整链路拆解

在动手写代码之前，我们先在脑子里走一遍整条数据流的全貌。假设你有一堆原始语料，可能是 .txt 格式的纯文本，也可能是 .jsonl 格式的结构化数据（每行一个 JSON 对象，里面有个 `text` 字段），这些数据首先要被读取和清洗，去掉空行和太短的无效内容。然后每一个文本片段都要送进 Tokenizer 变成一串整数 ID，注意这里我们不截断也不填充，而是保留每个文本的原始长度。接下来是关键的一步——滑窗切分。因为 LLM 训练的样本不是以"句子"为单位的，而是以"固定长度的 token 窗口"为单位的，所以我们需要把所有 token 拼成一个大序列，然后用一个滑动窗口去切，窗口大小就是 `max_seq_len`（比如 512 或 2048），每次移动的距离叫 `stride`（步长）。如果 stride 等于 max_seq_len，那相邻两个窗口之间没有重叠；如果 stride 小于 max_seq_len，就有重叠，重叠的好处是可以减少窗口边界处的信息丢失。切出来的每一个窗口就是一个训练样本，包含 input_ids 和 labels（labels 就是 input_ids 向右移一位）。最后这些样本被送入 DataLoader，DataLoader 通过多个 worker 进程并行读取样本，再经过自定义的 Collate Function 做动态左填充和 Attention Mask 构造，最终输出一个形状整齐的 Tensor Batch，可以直接 `.to("cuda")` 送进模型。

```
原始文件 (.txt / .jsonl)
    │
    ▼ 读取 & 清洗 → 过滤空行/过短文本
纯文本列表 List[str]
    │
    ▼ Tokenizer.encode() → 保留原始长度
Token IDs 列表 List[List[int]]
    │
    ▼ 合并 + 滑窗切分 (window=max_seq_len, step=stride)
训练样本列表 [{input_ids: [int], labels: [int]}]
    │
    │     ┌─────────── DataLoader ───────────┐
    │     │  Worker 0  Worker 1  Worker 2   │
    │     │   ↓          ↓         ↓        │
    │     │  sample    sample    sample      │
    │     │   ↓          ↓         ↓        │
    │     └────→ Collate Function ←────────┘│
    │              动态左 pad + Mask 构造    │
    ▼                                    │
Batch Dict {                              │
    input_ids:  (B, T)  tensor           │
    attention_mask: (B, T) tensor ───────┘
    labels:      (B, T) tensor
}
    │
    ▼ .to("cuda")
GPU Memory → Model Forward → Loss → Backward
```

理解了这条链路之后，我们来逐步实现它。下面的程序展示了一个完整的 `LMTextDataset` 类，它封装了从文件读取到滑窗切分的全部逻辑。

```python
import json
import torch
from torch.utils.data import Dataset
from typing import List, Optional


class LMTextDataset(Dataset):
    """语言模型训练数据集

    将原始文本文件转换为 (input_ids, labels) 样本对，
    支持滑窗切分以最大化 token 利用率。
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_seq_len: int = 512,
        stride: Optional[int] = None,
        min_length: int = 10,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len
        self.min_length = min_length

        # Step 1: 读取并清洗原始文本
        texts = self._load_texts(file_path)

        # Step 2: 全量 tokenize（不截断、不填充）
        all_token_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            if len(ids) >= self.min_length:
                all_token_ids.append(ids)

        # 把所有 token IDs 拼成一个扁平的大序列
        self.all_tokens = []
        for ids in all_token_ids:
            self.all_tokens.extend(ids)

        total_tokens = len(self.all_tokens)

        # Step 3: 滑窗切分为训练样本
        self.samples = []
        for i in range(0, total_tokens - self.max_seq_len, self.stride):
            chunk = self.all_tokens[i:i + self.max_seq_len]
            self.samples.append({
                'input_ids': chunk[:-1],   # input: 除最后一个 token
                'labels': chunk[1:],       # label: 除第一个 token（向右移一位）
            })

        self._report_stats()

    def _load_texts(self, file_path: str) -> List[str]:
        texts = []
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get('text', '') or obj.get('content', '')
                    if text and text.strip():
                        texts.append(text.strip())
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
        return texts

    def _report_stats(self):
        lengths = [len(s['input_ids']) for s in self.samples]
        print(f"Dataset: {len(self.samples)} samples, "
              f"seq_len range=[{min(lengths)}, {max(lengths)}], "
              f"mean={sum(lengths)/len(lengths):.1f}")

        utilization = sum(lengths) / (len(self.samples) * self.max_seq_len) * 100
        print(f"Token utilization: {utilization:.1f}%")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'input_ids': torch.tensor(s['input_ids'], dtype=torch.long),
            'labels': torch.tensor(s['labels'], dtype=torch.long),
        }
```

这里有几个值得深入讨论的设计决策。首先是 `_load_texts` 方法中的字段提取逻辑——实际生产中 JSONL 的 schema 千差万别，有的用 `text`，有的用 `content`，有的用 `instruction`+`output` 组合，甚至还有三层嵌套的结构。上面的代码做了一个最简单的 fallback 链：先找 `text`，找不到就找 `content`，都找不到就把整个 JSON 对象字符串化。这种做法在快速原型阶段够用，但在正式项目中你应该明确知道你的数据格式，硬编码对应的字段名更安全。其次是 `min_length` 参数的作用——过滤掉 tokenize 后少于指定 token 数的短文本。为什么需要这个？因为极短的文本（比如 "好的"、"谢谢"）即使被切进窗口里也几乎不包含有意义的上下文信息，还会稀释有效数据的密度。通常设为 10~20 就够了，具体取决于你的 `max_seq_len` 设置。

关于滑窗切分的 stride 参数，这是一个容易被误解但又极其重要的参数。当 stride 等于 max_seq_len 时，相邻两个窗口之间没有重叠，每个 token 只会出现在一个样本中，样本总数最少。当 stride 小于 max_seq_len 时，相邻窗口有重叠，token 利用率不变（因为每个窗口都是满的），但同一个 token 会出现在多个样本中，相当于做了数据增强。stride 越小，重叠越多，样本总数越多，训练时看到每个位置的上下文组合也更丰富。代价是训练时间变长（样本多了）以及模型可能对重叠区域过拟合。实践中最常见的做法是把 stride 设为 max_seq_len 的一半或四分之一，也就是 50% 或 75% 的重叠率。下面这段代码演示了不同 stride 对样本数量的影响：

```python
def count_samples(total_tokens, max_seq_len, stride):
    return (total_tokens - max_seq_len) // stride + 1

total = 1_000_000
max_len = 512

for s in [512, 256, 128]:
    n = count_samples(total, max_len, s)
    print(f"stride={s:>4d} → {n:>6d} samples (overlap={((max_len-s)/max_len*100):.0f}%)")

# 输出:
# stride= 512 →  1953 samples (overlap=0%)
# stride= 256 →  3905 samples (overlap=50%)
# stride= 128 →  7809 samples (overlap=75%)
```

可以看到，当 stride 从 512 减半到 256 时，样本数量翻倍了。这就是为什么在实际的大规模预训练任务中，stride 的选择会直接影响你需要多少个 training step 才能遍历完一轮数据。

## Collate Function：把散乱的样本组装成整齐的 Batch

有了 Dataset 之后，下一步就是定义 Collate Function。我们在 02-02 节已经详细讨论过左填充、Attention Mask 构造等原理，现在把这些知识封装成一个可复用的类。下面这个 `LLMDataCollator` 的设计思路是：接收 Dataset `__getitem__` 返回的字典列表，找出当前 batch 中最长的序列，把其他所有序列用 pad_token_id 左填充到同样长度，同时构造 attention_mask 和处理 labels。

```python
import torch.nn.functional as F


class LLMDataCollator:
    """LLM 数据整理器 — 动态左填充 + Attention Mask 构造"""

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_id: int = -100,
        max_length: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id
        self.max_length = max_length

    def __call__(self, features: list) -> dict:
        input_ids = [f['input_ids'] for f in features]

        max_len = min(
            max(len(ids) for ids in input_ids),
            self.max_length or float('inf'),
        )

        batch_input_ids = []
        batch_attention_mask = []

        for ids in input_ids:
            ids = ids[:max_len]
            pad_len = max_len - len(ids)
            batch_input_ids.append([self.pad_token_id] * pad_len + ids.tolist())
            batch_attention_mask.append([0] * pad_len + [1] * len(ids))

        batch = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
        }

        if 'labels' in features[0]:
            batch_labels = []
            for f in features:
                lbl = f['labels'][:max_len]
                pad_len = max_len - len(lbl)
                batch_labels.append([self.label_pad_id] * pad_len + lbl.tolist())
            batch['labels'] = torch.tensor(batch_labels, dtype=torch.long)

        return batch
```

这个 Collator 有几个细节值得注意。第一，`label_pad_id` 默认设为 -100 而不是 pad_token_id（通常是 0），这是因为在 CrossEntropyLoss 中设置 `ignore_index=-100` 时，所有值为 -100 的位置都不会参与 loss 计算，也不会贡献梯度。如果你错误地把 label 的 padding 位置也填成了 0（一个真实的 token ID），那么模型就会在这些位置被强制预测 token 0，引入噪声梯度。第二，`max_length` 参数提供了一个硬性上限——即使当前 batch 里最长的样本超过了这个值也会被截断。这在推理阶段或者当你使用动态 batching 但又不希望个别超长样本拖慢整个 batch 时非常有用。第三，注意输入的 `features` 是一个 Python list 而不是 Tensor，这意味着 Collate Function 是在 CPU 上执行的（worker 进程里），所有的 tensor 操作都不会占用 GPU 显存。

## 工厂函数：一行代码创建完整 DataLoader

现在 Dataset 和 Collator 都有了，最后一步是用工厂函数把它们组装成一个开箱即用的 DataLoader。工厂函数的意义在于把所有配置参数集中管理，避免每次创建 loader 时都要重复写一大堆参数。更重要的是，你可以在工厂函数里根据环境自动做一些合理的默认选择，比如检测是否有 CUDA 可用来决定是否开启 pin_memory，或者根据 CPU 核心数自动设置 num_workers。

```python
from torch.utils.data import DataLoader


def build_dataloader(
    file_path: str,
    tokenizer,
    batch_size: int = 8,
    max_seq_len: int = 512,
    stride: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    dataset = LMTextDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
    )

    collator = LLMDataCollator(
        pad_token_id=getattr(tokenizer, 'pad_token_id', 0),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
```

调用起来非常简洁：

```python
loader = build_dataloader(
    file_path="corpus.jsonl",
    tokenizer=tokenizer,
    batch_size=16,
    max_seq_len=1024,
    stride=512,
    num_workers=4,
)

for batch in loader:
    print(batch['input_ids'].shape)      # (16, T)  T 动态变化
    print(batch['attention_mask'].shape) # (16, T)
    print(batch['labels'].shape)         # (16, T)
    break
```

到这里，一条完整的数据管道就已经搭好了。但工程上还有一个关键环节不能少——**验证**。很多初学者犯的一个错误是：写完 pipeline 直接扔进训练循环，跑了几个小时后发现 loss 不收敛或者干脆报错退出，然后才开始排查是不是数据有问题。正确做法是在正式训练之前，先用一个小的验证函数检查管道输出的每一个 batch 是否符合预期。下面这个 `verify_pipeline` 函数做了几件事：检查 shape 是否正确、验证 labels 是否真的是 input_ids 右移一位的结果、统计有效 token 占比来评估 padding 浪费程度。

```python
def verify_pipeline(loader: DataLoader, num_batches: int = 3):
    print("=" * 60)
    print("Pipeline Verification")
    print("=" * 60)

    total_valid = 0
    total_tokens = 0
    ref_t = None

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        B, T = batch['input_ids'].shape

        assert batch['input_ids'].shape == (B, T), \
            f"Batch {i}: input_ids shape mismatch"
        assert batch['labels'].shape == (B, T), \
            f"Batch {i}: labels shape mismatch"
        assert batch['attention_mask'].shape == (B, T), \
            f"Batch {i}: mask shape mismatch"

        valid = batch['attention_mask'].sum().item()
        total_valid += valid
        total_tokens += B * T

        for b in range(B):
            mask = batch['attention_mask'][b].bool()
            positions = mask.nonzero(as_tuple=True)[0]
            if len(positions) >= 2:
                first_pos = positions[0].item()
                last_pos = positions[-1].item()
                expected_labels = batch['input_ids'][b, first_pos:last_pos - 1]
                actual_labels = batch['labels'][b, first_pos + 1:last_pos]
                assert torch.equal(expected_labels, actual_labels), \
                    f"Batch {i}, sample {b}: labels shift check FAILED"

        ref_t = T
        print(f"  Batch {i}: B={B}, T={T}, "
              f"valid_ratio={valid/(B*T)*100:.1f}% ✓")

    avg_util = total_valid / total_tokens * 100
    print(f"\n  Average token utilization: {avg_util:.1f}%")
    print(f"  All checks passed! Pipeline is ready for training.")
```

运行验证函数后如果看到"All checks passed"，你就可以放心地进入训练循环了。如果验证失败，错误信息会精确告诉你哪个 batch、哪个样本出了什么问题，这比在训练中途报 RuntimeError 然后去翻几千行的 traceback 要高效得多。这里有一个常见的陷阱需要注意：有些人在验证时只检查了 shape 和 dtype，但没有检查 labels 的 shift 正确性。这种情况下管道可能表面上看起来正常（shape都对，也没有 runtime error），但实际上 labels 和 input_ids 错位了一个位置，导致模型从一开始就在学习错误的映射关系，loss 可能会下降但生成质量永远上不去。所以上面验证函数里的那个 assert——逐样本对比 labels 是否等于 input_ids[1:]——虽然看起来繁琐，但它能捕获这类隐蔽性极强的 bug。

## 数据统计与诊断：让数字告诉你管道是否健康

一个成熟的数据管道不应该只是默默地吐出 batch，还应该能够告诉你关于数据集本身的统计信息，因为这些信息直接影响你对超参数的选择和对训练效果的判断。比如，如果你发现平均序列长度只有 max_seq_len 的 30%，那就说明大部分计算时间都浪费在了 padding token 上，这时候要么减小 max_seq_len，要么改用 packed sequence 方案消除 padding。又比如，如果你发现序列长度分布极度不均匀——大量短文本夹杂着少数超长文档——那可能需要在预处理阶段做一次长度分桶（bucketing），把相近长度的样本放在一起组成 batch 来提高效率。

下面这段代码展示了如何在 Dataset 初始化结束时收集并打印一组关键的统计指标：

```python
def analyze_dataset_statistics(samples, max_seq_len):
    lengths = [len(s['input_ids']) for s in samples]
    n = len(lengths)

    stats = {
        'count': n,
        'min': min(lengths),
        'max': max(lengths),
        'mean': sum(lengths) / n,
        'median': sorted(lengths)[n // 2],
        'p95': sorted(lengths)[int(n * 0.95)],
        'utilization': sum(lengths) / (n * max_seq_len) * 100,
    }

    print(f"\n{'─' * 50}")
    print(f"Dataset Statistics (max_seq_len={max_seq_len})")
    print(f"{'─' * 50}")
    print(f"  Total samples:     {stats['count']:>10,}")
    print(f"  Min length:        {stats['min']:>10d} tokens")
    print(f"  Max length:        {stats['max']:>10d} tokens")
    print(f"  Mean length:       {stats['mean']:>10.1f} tokens")
    print(f"  Median length:     {stats['median']:>10d} tokens")
    print(f"  P95 length:        {stats['p95']:>10d} tokens")
    print(f"  Token utilization: {stats['utilization']:>9.1f}%")
    print(f"{'─' * 50}")

    bucket_size = max(64, max_seq_len // 8)
    buckets = {}
    for l in lengths:
        key = (l // bucket_size) * bucket_size
        buckets[key] = buckets.get(key, 0) + 1

    print(f"\n  Length distribution:")
    max_count = max(buckets.values())
    for start in sorted(buckets.keys()):
        count = buckets[start]
        bar_len = int(count / max_count * 40)
        bar = '█' * bar_len
        pct = count / n * 100
        print(f"    [{start:>5d}-{start+bucket_size-1:>5d}] "
              f"{count:>5d} ({pct:>5.1f}%) {bar}")

    return stats
```

当你在一个真实的数据集上运行这个分析函数时，输出可能像这样：

```
──────────────────────────────────────────────────
Dataset Statistics (max_seq_len=512)
──────────────────────────────────────────────────
  Total samples:       125,430
  Min length:              64 tokens
  Max length:             512 tokens
  Mean length:          287.3 tokens
  Median length:          296 tokens
  P95 length:             488 tokens
  Token utilization:     56.1%
──────────────────────────────────────────────────

  Length distribution:
    [   64- 127]   8,234 ( 6.6%) ████████████████████
    [  128- 191]  15,102 (12.0%) ████████████████████████████████
    [  192- 255]  22,891 (18.3%) ██████████████████████████████████████████████████
    [  256- 319]  31,456 (25.1%) █████████████████████████████████████████████████████████████████████████████
    [  320- 383]  24,678 (19.7%) █████████████████████████████████████████████████████████████
    [  384- 447]  14,234 (11.3%) ████████████████████████████████████████
    [  448- 511]   8,835 ( 7.0%) █████████████████████████
```

从这个统计结果你能读出不少有用的信息。56.1% 的 token 利用率意味着接近一半的计算量花在了 padding token 上——这在 LLM 训练中是一个相当大的浪费。P95 长度是 488，非常接近 max_seq_len=512，说明只有极少数样本真正达到了最大长度限制。如果你把 max_seq_len 从 512 降到 384，利用率会大幅提升（因为去掉了 384-511 这个区间的 padding 开销），代价是截断了约 18% 的长样本。这就是一个典型的精度-效率权衡决策，而统计数据给了你做出明智选择的依据。

另一个重要的诊断维度是词表覆盖率——你的数据中有多少比例的 token ID 落在 tokenizer 的有效范围内，有多少是 unk_token（未知 token）。覆盖率过低通常意味着你的数据和 tokenizer 的训练语料差异太大，或者数据中包含了太多特殊字符、乱码、HTML 标签等噪声。下面是一个简单的覆盖率检查函数：

```python
def check_vocab_coverage(dataset, tokenizer, num_samples=1000):
    vocab_size = tokenizer.vocab_size
    unk_id = getattr(tokenizer, 'unk_token_id', None)
    total = 0
    unk_count = 0
    oov_count = 0

    indices = torch.randperm(len(dataset))[:num_samples]
    for idx in indices:
        sample = dataset[idx.item()]
        ids = sample['input_ids']
        total += ids.numel()
        if unk_id is not None:
            unk_count += (ids == unk_id).sum().item()
        oov_count += (ids >= vocab_size).sum().item()

    print(f"Vocab coverage check ({num_samples} samples):")
    print(f"  Total tokens:     {total:,}")
    if unk_id is not None:
        print(f"  UNK tokens:       {unk_count:,} ({unk_count/total*100:.2f}%)")
    print(f"  OOV tokens:       {oov_count:,} ({oov_count/total*100:.2f}%)")
    print(f"  Coverage:         {(1-oov_count/total)*100:.1f}%")
```

## 什么时候该用 HuggingFace Dataset？

到目前为止，我们一直在手写自己的 Dataset 类和数据处理逻辑。这种方式的最大好处是你完全掌控每一个细节，知道每一行代码在做什么，调试时也能精确定位问题。但随着项目规模增长，你会发现自己在重复实现一些通用功能：map 操作（对每个样本做变换）、filter 操作（按条件过滤）、shuffle（打乱）、分桶（按长度分组）、缓存（避免重复处理）等等。HuggingFace 的 `datasets` 库正是为了解决这些问题而生的。它的核心 API 设计非常直观：

```python
from datasets import load_dataset, Dataset

# 一行加载远程或本地数据
ds = load_dataset("json", data_files="corpus.jsonl", split="train")

# 链式操作：过滤 → map → shuffle → select
ds = (
    ds.filter(lambda x: len(x['text']) > 20)
    .map(lambda x: tokenizer(x['text'], truncation=False), num_proc=8)
    .shuffle(seed=42)
    .select(range(100_000))  # 取前 10 万条用于快速实验
)
```

HuggingFace Dataset 的优势在于它内置了内存映射（Arrow 格式）、多进程 map/filter、智能缓存和断点续处理等企业级特性，而且可以直接传给 Trainer 使用，不需要任何胶水代码。但它的劣势也很明显：当你需要高度定制化的逻辑时（比如复杂的滑窗切分策略、跨文件的增量更新、自定义的二进制格式支持），HF Dataset 的抽象层反而会成为障碍，你不得不深入研究它的内部机制甚至 fork 源码来修改。

一个实用的经验法则是这样的：如果你的数据格式是标准的（JSONL/JSON/CSV/Parquet/Arrow），处理流程也是常规的（tokenize → filter → shuffle → batch），那么直接用 HF Dataset 能节省大量的开发时间。但如果你的数据来源复杂多样（数据库查询结果、实时流、多种格式的混合），或者你有特殊的性能需求（比如必须控制在特定内存预算内处理 TB 级数据），那么自己写 Dataset 和 IterableDataset 往往更灵活。实际上在生产环境中常见的做法是两者结合：用 HF Dataset 做初始的加载和预处理（利用其高效的 Arrow 存储和多进程能力），然后把处理好的数据转成你自定义的格式供训练管道使用。

## 常见误区与排错指南

在实际搭建数据管道的过程中，有几个错误出现频率极高，值得单独拿出来讲一讲。

第一个常见错误是在 `__getitem__` 里做 padding。有些人觉得既然每个样本返回时都需要是相同 shape，不如直接在 Dataset 内部处理好。这样做的问题在于：Dataset 返回的是单个样本的信息，它在被 Collate Function 调用时并不知道同 batch 的其他样本有多长，所以只能 pad 到固定的 `max_seq_len`。这就导致了我们之前详细分析过的固定 padding 问题——大量短样本被无意义地填充到最大长度，token 利用率可能低至 25%。正确的做法是让 `__getitem__` 返回变长的原始数据，把 padding 推迟到 Collate Function 中执行，这样就能实现按 batch 内最长样本动态调整的"延迟 padding"策略。

第二个错误是忽略了 tokenizer 的 `padding` 和 `truncation` 参数。HuggingFace 的 Tokenizer 在调用 `encode()` 或 `__call__()` 时可以传入 `padding=True` 和 `truncation=True`，它会自动帮你做填充和截断。如果你在 Dataset 的 `__getitem__` 里用了带 `padding=True` 的 tokenizer，然后再在外层的 Collate Function 里再做一遍 padding，那就是双重 padding 了，不仅浪费计算，还可能导致 shape 不匹配的错误。记住一个原则：**在数据管道中，padding 只应该发生在一个地方——Collate Function**。

第三个错误跟 shuffle 有关。很多人习惯性地对所有 DataLoader 都开启 `shuffle=True`，包括验证集和测试集。这在大多数任务中是无害的，但在某些场景下会导致问题。比如你想做确定性评估（deterministic evaluation）来比较不同实验之间的效果，如果 val_loader 每次迭代都给出不同的 batch 顺序，你就无法保证两次评估看到完全相同的样本组合。另外，在分布式训练（DDP）中，shuffle 应该通过 DistributedSampler 来控制而不是 DataLoader 的 shuffle 参数，否则每个 rank 上的数据顺序可能不一致导致梯度同步异常。所以养成一个好习惯：训练集 shuffle=True，验证集和测试集 shuffle=False，分布式训练时用 DistributedSampler 替代 shuffle。

第四个错误是关于 `num_workers` 的误用。有人认为 num_workers 越多越好，直接设成 CPU 核心数甚至更多。但正如我们在 02-01 节分析的，num_workers 超过一定阈值后收益递减甚至转为负值，原因是进程间上下文切换的开销和数据拷贝的成本超过了并行带来的加速。另一个相关的问题是忘记设置 `persistent_workers=True`——如果不设置，每个 epoch 结束后 DataLoader 会销毁所有 worker 进程并在下一个 epoch 开始时重新创建，对于有数千个 worker 的场景来说这个开销不可忽视。简单的记忆规则是：只要 `num_workers > 0`，就加上 `persistent_workers=True`。

第五个错误比较隐蔽，跟数据泄露有关。假设你在构建 Dataset 时对数据做了某种全局操作（比如基于整个数据集计算均值和方差来做归一化，或者基于全局词频统计来做过滤），如果在划分 train/val/test 之前就做了这些操作，那么验证集和测试集的信息就已经"泄露"到了训练过程中。正确的顺序应该是：先划分数据集，然后只在训练集上计算统计量，再把同样的统计量应用到验证集和测试集上。虽然在 LLM 的纯语言建模任务中这个问题不那么突出（因为每个样本都是独立的文本片段），但在涉及分类、回归等有明确标签的任务中这一点至关重要。

## 第 2 章回顾与第 3 章预告

到这里，第 2 章"数据管道"的全部四个小节就结束了。让我们快速回顾一下这一章的知识地图。我们从最基础的 Dataset 两种模式（Map-style 和 Iterable-style）出发，理解了 DataLoader 如何通过多进程 Worker 池和 Queue 机制实现高效的数据加载，掌握了 num_workers、pin_memory、prefetch_factor 这些关键参数的含义和调优方法。接着我们深入了 Collate Function 的世界，搞清楚了为什么 LLM 必须使用左填充而不是右填充（因果约束），如何构造组合了 padding mask 和 causal mask 的 Attention Mask，以及动态 padding 相比固定 padding 在 token 利用率上的巨大优势（75% vs 25%）。然后在高效数据处理那一节，我们学习了三种应对 TB 级数据的方案：内存映射文件用于单大文件的零拷贝访问、IterableDataset 用于多文件流的增量读取、Parquet/Arrow 列式存储用于只读所需列的高效查询，还了解了近似去重算法如何将 O(N²) 的精确去重降低到 O(N)。最后在这一节，我们把所有组件组装成了端到端的完整管道，加上验证函数和统计诊断工具，让它具备了投入生产的素质。

下一章将是整个教程的第一个高潮点——我们将暂时离开数据处理的领域，进入模型架构的核心地带，用纯 PyTorch 从零开始一行一行地实现一个完整的 Transformer（GPT 架构）。这不仅是一次编码练习，更是理解"Attention Is All You Need"论文精髓的最佳途径。完成之后，当你再去阅读 LLaMA、Qwen、Mistral 这些前沿模型的源码时，你会发现它们本质上都是在同一个骨架上的变体——而这个骨架，就是你即将亲手搭建的那个。
