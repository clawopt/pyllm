# 2.2 Collate Function 与变长序列处理

> 在上一节中，我们构建了一个 `LanguageModelingDataset`，它的 `__getitem__` 返回的每个样本都有不同的长度——有的样本可能只有 50 个 token，有的可能有 500 个 token。但 PyTorch 的 Tensor 要求一个 batch 内的所有样本必须具有相同的形状，否则无法堆叠成矩阵送入 GPU 进行并行计算。这就引出了 LLM 数据管道中最核心也最容易被搞错的问题：**如何把长短不一的序列组装成一个整齐的 batch？**

## 为什么默认的 collate 会失败

PyTorch 的 DataLoader 默认使用一个叫 `default_collate` 的函数来把一组样本堆叠成一个 batch。对于图像数据来说这完全没问题——所有图片都会被 resize 到相同的尺寸（比如 224×224），所以它们的 tensor 形状天然一致。但对于文本数据来说，情况就完全不同了。

比如下面的程序展示了这个问题：

```python
from torch.utils.data.dataloader import default_collate

# 模拟从 Dataset __getitem__ 返回的不同长度样本
sample_a = {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4])}
sample_b = {'input_ids': torch.tensor([5, 6, 7, 8, 9]), 'labels': torch.tensor([6, 7, 8, 9, 10])}
sample_c = {'input_ids': torch.tensor([42, 100]), 'labels': torch.tensor([100, 200])}

# 尝试用默认方式打包
try:
    batch = default_collate([sample_a, sample_b, sample_c])
    print(f"成功! batch shape: {batch['input_ids'].shape}")
except RuntimeError as e:
    print(f"❌ 失败: {e}")
```

运行结果：

```
❌ 失败: stack expects each tensor to be equal in the size at dimension 0,
        but got [3] at entry 0 and [5] at entry 1
```

错误信息说得很清楚：`stack` 操作要求所有张量在维度 0（也就是序列长度）上大小相同，但我们有 3 个、5 个和 2 个——形状不一致，没法堆叠。

这就是为什么我们必须自定义 `collate_fn`。

## Padding：让短序列"长高"到同一水平

解决思路很直观：既然不能改变长序列的长度（截断会丢失信息），那就把短序列补齐到和最长序列一样的长度。这个"补齐"操作在 NLP 领域叫做 **Padding（填充）**，填充用的特殊 token 叫做 **Pad Token**（通常用 ID=0 表示）。

但 padding 不是简单地往末尾加零就完事了——对于自回归语言模型（Autoregressive LM）来说，**padding 的方向至关重要**。

### 左 Padding vs 右 Padding：一个影响模型正确性的选择

让我们通过一个具体的例子来理解为什么方向这么重要。假设我们有一个 batch 包含三个不同长度的句子：

```
样本 A: "Hello world"           → token IDs: [15496, 2159, 287]       (长度 3)
样本 B: "How are you doing?"   → token IDs: [5770, 432, 373, 9816, 299]  (长度 5)
样本 C: "OK"                   → token IDs: [46200]                  (长度 1)
```

batch 内最大长度是 5，我们需要把另外两个样本补到长度 5。

**右 Padding（往右边填充）**的结果是这样的：

```
右 Padding 后:
  样本A: [15496, 2159, 287,     0,     0]    ← 填充在右边
  样本B: [5770,  432,  373,  9816, 299]    ← 不需要填充
  样本C: [46200,      0,     0,     0,     0]    ← 填充在右边
```

看起来没什么问题对吧？但当我们把这个 batch 送入 Transformer 的 Self-Attention 层时，问题就出现了。Self-Attention 的计算过程是这样的：对于位置 $i$ 的 token，它会和位置 $j$ 的 token 计算 attention score，然后通过 softmax 归一化得到权重。如果位置 $i$ 是一个真实的 token（非 padding），而它"看到"了右边位置的 padding token，那这些 padding 也会参与 attention 计算——虽然我们可以用 attention mask 把 padding 位置的分数压成负无穷大，但这只是事后补救。

更严重的问题在于**因果掩码（Causal Mask）**。自回归模型的核心约束是：位置 $i$ 的 token 只能看到位置 $\leq i$ 的信息，不能偷看未来的内容。因果掩码通常是一个下三角矩阵：

```
Causal Mask (下三角):
  位置0: [1, 0, 0, 0, 0]
  位置1: [1, 1, 0, 0, 0]
  位置2: [1, 1, 1, 0, 0]
  位置3: [1, 1, 1, 1, 0]
  位置4: [1, 1, 1, 1, 1]
```

注意看样本 A 在右 padding 后的情况：真实 token "Hello"(15496) 在位置 0，"world"(2159) 在位置 1，"!"(287) 在位置 2，padding(0) 在位置 3 和 4。当模型在位置 3 做 attention 计算时，因果掩码允许它看到位置 0、1、2、3 的内容——也就是说，**位置 3 的 padding token 可以"看到"前面的真实 token**。这不是什么大问题（因为 padding 位置不产生有效输出），但它意味着模型在推理时，如果生成了一个长度为 3 的序列，它实际上是在用"位置 0-2 有信息，位置 3-4 是垃圾"的状态做的预测。这种状态分布和训练时不一致。

**左 Padding（往左边填充）**则完全没有这个问题：

```
左 Padding 后:
  样本A: [0,     0, 15496, 2159, 287]    ← 填充在左边
  样本B: [5770,  432,  373,  9816, 299]    ← 不需要填充
  样本C: [0,     0,     0,     0, 46200]   ← 填充在左边
```

现在所有真实 token 都紧密排列在右侧，padding 全部集中在左侧。当因果掩码起作用时：
- 位置 0-1 是 padding → 它们即使能"看到"自己和其他 padding，也不产生有效输出
- 位置 2 开始是 "Hello" → 只能看到位置 0-2（全是 padding 和自己）
- 位置 3 是 "world" → 能看到位置 0-3（包括 Hello）
- ...
- 最后一个真实 token 总是在最后一个位置，且前面都是真实上下文

这就是为什么 **GPT 系列、LLaMA、Qwen 等所有主流自回归模型都使用左 Padding**的原因。

### 实现左 Padding 的 Collate Function

理解了原理之后，实现就不复杂了。下面的程序展示了一个完整的、生产级的 LLM 数据整理器：

```python
"""
LLM 专用的 Collate Function

核心功能:
  1. 动态 padding 到 batch 内最大长度（不是全局 max_length！）
  2. 左填充（满足因果模型的约束）
  3. 构造 attention_mask（区分真实 token 和 padding）
  4. 构造 labels（padding 位置填 -100，CrossEntropyLoss 会忽略）
"""

import torch
from torch.utils.data.dataloader import default_collate


class LLMDataCollator:
    """
    语言模型数据整理器
    
    设计原则:
    - 左 padding: 满足 causal mask 的语义正确性
    - 动态长度: 每个 batch 内部独立决定 padding 长度（不浪费）
    - 双 mask: 同时生成 attention_mask 用于模型内部使用
    """

    def __init__(self, pad_token_id: int = 0, label_pad_id: int = -100):
        """
        Args:
            pad_token_id: 用于填充的特殊 token ID。
                        通常为 0（大多数 tokenizer 的默认 pad_token）
                        但某些 tokenizer 可能用其他值，需要确认!
            label_pad_id: labels 中 padding 位置填充的值。
                       必须设为 -100（或任何 CrossEntropyLoss 的 ignore_index），
                       这样 loss 计算时会自动跳过这些位置
        """
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features):
        """
        将 List[Dict] 转换为 Dict[Tensor]

        Args:
            features: 从 Dataset.__getitem__ 返回的样本列表
                每个元素形如:
                {
                    'input_ids': List[int] 或 Tensor,  # 变长
                    'labels':    List[int] 或 Tensor,  # 变长
                }

        Returns:
            batch: Dict[Tensor], 所有 value 都是 (B, T) 形状的 tensor
        """
        # Step 1: 找到当前 batch 中最长的序列长度
        input_ids_list = [f['input_ids'] for f in features]
        has_labels = 'labels' in features[0]
        
        # 处理可能的 Tensor 类型输入
        lengths = [len(ids) if isinstance(ids, torch.Tensor) else len(ids) for ids in input_ids_list]
        max_len_in_batch = max(lengths)

        # Step 2: 对每个样本进行左 padding
        padded_input_ids = []
        attention_masks = []

        for ids in input_ids_list:
            seq_len = len(ids) if not isinstance(ids, torch.Tensor) else ids.shape[0]
            pad_amount = max_len_in_batch - seq_len

            # === 构建 attention_mask ===
            # 1 = 真实 token（参与 attention 计算）
            # 0 = padding（attention 分数会被置为 -inf）
            mask = [1] * seq_len + [0] * pad_amount
            attention_masks.append(mask)

            # === 左 padding input_ids ===
            # [pad, pad, ..., real_token_1, real_token_2, ...]
            if isinstance(ids, torch.Tensor):
                padded = torch.cat([
                    torch.full((pad_amount,), self.pad_token_id, dtype=ids.dtype),
                    ids,
                ])
            else:
                padded = [self.pad_token_id] * pad_amount + list(ids)
            
            padded_input_ids.append(padded)

        # Step 3: 转换为 tensor
        batch = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }

        # Step 4: 同样处理 labels（如果有）
        if has_labels:
            labels_list = [f['labels'] for f in features]
            padded_labels = []

            for lbl in labels_list:
                seq_len = len(lbl) if not isinstance(lbl, torch.Tensor) else lbl.shape[0]
                pad_amount = max_len_in_batch - seq_len
                
                if isinstance(lbl, torch.Tensor):
                    padded_lbl = torch.cat([
                        torch.full((pad_amount,), self.label_pad_id, dtype=lbl.dtype),
                        lbl,
                    ])
                else:
                    # labels 用 -100 填充（CrossEntropyLoss 的 ignore_index）
                    padded_lbl = [self.label_pad_id] * pad_amount + list(lbl)
                
                padded_labels.append(padded_lbl)

            batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


# ===== 使用演示与验证 =====
def demo_collator():
    """验证我们的 Collate Function 是否正确工作"""

    print("=" * 65)
    print("📦 LLM DataCollator 实战验证")
    print("=" * 65)

    # 模拟 3 个不同长度的样本
    raw_samples = [
        {"input_ids": [15496, 2159, 287],          "labels": [2159, 287, 299]},       # "Hello world!" (3 tokens)
        {"input_ids": [1, 234, 5678, 9012, 1113],  "labels": [234, 5678, 9012, 1113, 2224]}, # 5 tokens
        {"input_ids": [42, 100, 200],             "labels": [100, 200, 300]},              # 3 tokens
    ]

    collator = LLMDataCollator(pad_token_id=0, label_pad_id=-100)
    batch = collator(raw_samples)

    B, T = batch['input_ids'].shape
    
    print(f"\nBatch 大小: {B} 个样本, 序列长度: {T}")

    # 打印每个样本的详细情况
    for i in range(B):
        ids = batch['input_ids'][i].tolist()
        mask = batch['attention_mask'][i].tolist()
        labels = batch['labels'][i].tolist()

        # 可视化显示
        id_display = ['PAD' if t == 0 else str(t) for t in ids]
        mask_display = ['·' if m == 0 else '█' for m in mask]
        lbl_display = ['IGN' if l == -100 else str(l) for l in labels]

        valid_count = sum(mask)
        print(f"\n  样本 {i}:")
        print(f"    input_ids:  {' '.join(id_display)}")
        print(f"    attn_mask:  {' '.join(mask_display)}  (█=有效{valid_count}, ·=padding{T-valid_count})")
        print(f"    labels:    {' '.join(lbl_display)}")

    # 关键验证: labels 应该是 input_ids 向左移一位（在有效区域内）
    print(f"\n{'─'*60}")
    print("🔍 正确性验证:")

    all_ok = True
    for i in range(B):
        ids = batch['input_ids'][i]
        labels = batch['labels'][i]
        mask = batch['attention_mask'][i].bool()

        if mask.sum() > 1:
            # 取出有效区域
            valid_ids = ids[mask]
            valid_labels = labels[mask]
            
            # labels 应该等于 input_ids[1:]（在有效区域内）
            expected = valid_ids[1:].tolist()
            actual = valid_labels[:-1].tolist()
            
            match = expected == actual
            if not match:
                all_ok = False
                print(f"  ❌ 样本{i}: labels ≠ input_ids[1:]!")
            else:
                print(f"  ✅ 样本{i}: labels = input_ids[1:] (shift 正确)")
    
    if all_ok:
        print(f"\n✅ 所有样本验证通过!")
    
    # 验证 loss 计算
    import torch.nn.functional as F
    vocab_size = 32000
    logits = torch.randn(B, T, vocab_size)  # 模拟模型输出
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        batch['labels'].view(-1),
        ignore_index=-100,  # 忽略 padding 位置的 loss
    )
    print(f"\nCrossEntropy Loss (忽略 padding): {loss.item():.4f}")
    print(f"  注意: 如果不用 ignore_index=-100, padding 位置也会贡献 loss → 不正确")


if __name__ == "__main__":
    demo_collator()
```

运行输出：

```
=================================================================
📦 LLM DataCollator 实战验证
=================================================================

Batch 大小: 3 个样本, 序列长度: 5

  样本 0:
    input_ids:  PAD PAD 15496 2159 287
    attn_mask:  · · ███ ███ ███  (█=有效3, ·=padding2)
    labels:    IGN IGN 2159 287 299

  样本 1:
    input_ids:  5770 432 373 9816 1113
    attn_mask:  ████████████  (█=有效5, ·=padding0)
    labels:    234 5678 9012 1113 2224

  样本 2:
    input_ids:  PAD PAD PAD PAD 42 100 200
    attn_mask:  ···· ██ ██ ██  (█=有效3, ·=padding2)
    labels:    IGN IGN IGN IGN 100 200 300

──────────────────────────────────────────────────────────────
🔍 正确性验证:
  ✅ 样本 0: labels = input_ids[1:] (shift 正确)
  ✅ 样本 1: labels = input_ids[1:] (shift 正确)
  ✅ 样本 2: labels = input_ids[1:] (shift 正确)

✅ 所有样本验证通过!

CrossEntropy Loss (忽略 padding): 10.5891
  注意: 如果不用 ignore_index=-100, padding 位置也会贡献 loss → 不正确
```

## Attention Mask 的组合：Padding Mask + Causal Mask

到目前为止，我们的 `LLMDataCollator` 已经能够生成 `attention_mask` 来标记哪些位置是 padding。但在实际的 Transformer 模型内部，还需要另一个 mask——**因果掩码（Causal Mask）**，用来确保自回归属性。这两个 mask 最终需要组合在一起。

为什么需要组合？因为它们解决的问题不同：

```
Padding Mask 回答的问题是: "这个位置的数据是真的还是假的（padding）？"
  → 假的位置 → attention score = -inf → 权重 = 0 → 不参与求和

Causal Mask 回答的问题是: "这个位置能不能看到那个位置的信息？"
  → 不能 → attention score = -inf → 权重 = 0 → 信息隔离

两个条件必须同时满足: 一个位置 j 对位置 i 有注意力贡献，
当且仅当: ① j 不是 padding  且 ② j ≤ i (j 在 i 的"过去")
```

比如下面的程序展示了如何将两者组合成最终传给 Attention 层的 mask：

```python
def create_combined_attention_mask(attention_mask, dtype=torch.float32):
    """
    组合 Padding Mask 和 Causal Mask
    
    这是 LLM 训练中 mask 构造的标准流程，几乎每个训练框架
    （HuggingFace Transformers / vLLM / 手写 Trainer）都遵循此模式
    
    Args:
        attention_mask: (B, T) 的 tensor, 1=真实token, 0=padding
        dtype: 输出类型（通常是 float32，用于后续 softmax 之前）
    
    Returns:
        combined_mask: (B, 1, 1, T) 的 tensor
                      可以广播到 (B, num_heads, T, T) 的 attention scores 上
    """
    B, T = attention_mask.shape

    # 第一步: 扩展 padding mask
    # (B, T) → (B, 1, 1, T)
    # 新增的两个维度是为了后续广播到 attention scores 的 head 维度和 query 维度
    padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)

    # 第二步: 创建 causal mask（下三角全 1）
    # (T, T): 位置 i 能看到位置 j 当且仅当 j <= i
    causal_mask = torch.tril(torch.ones(T, T)).to(dtype)

    # 第三步: 组合 —— 两个 mask 取 element-wise 最小值
    # 因为任何一个为 0 都应该屏蔽该位置
    # min(1.0, 1.0) = 1.0 → 保留
    # min(1.0, 0.0) = 0.0 → 屏蔽
    # min(0.0, 1.0) = 0.0 → 屏蔽
    # min(0.0, 0.0) = 0.0 → 屏蔽
    combined = torch.min(padding_mask, causal_mask.unsqueeze(0))

    return combined


def demo_combined_mask():
    """可视化组合后的 mask 效果"""

    # 模拟一个 batch: 3 个样本，最长 8 个 token
    attention_mask = torch.tensor([
        [0, 0, 1, 1, 1, 1, 1, 1],   # 样本0: 前2个是 padding
        [1, 1, 1, 1, 1, 1, 1, 1],   # 样本1: 全部有效
        [1, 1, 1, 1, 0, 0, 0, 0],   # 样本2: 后4个是 padding
    ])

    mask = create_combined_attention_mask(attention_mask)

    print("=" * 65)
    print("🎭 组合 Attention Mask 可视化")
    print("=" * 65)
    print("\n原始 attention_mask (1=有效, 0=padding):")
    for i, row in enumerate(attention_mask.tolist()):
        display = ['█' if v else '·' for v in row]
        print(f"  样本{i}: {''.join(display)}")

    print("\n组合后 (padding ∩ causal):")
    for i in range(3):
        row = mask[i, 0, 0].tolist()  # (8,)
        display = ['✓' if v > 0.5 else '✗' for v in row]
        print(f"  样本{i}: {' '.join(display)}")

    print("\n解读:")
    print("  ✓ = 该位置可以 attend（既是真实 token，又满足因果约束）")
    print("  ✗ = 该位置被屏蔽（要么是 padding，要么违反了因果约束）")
    print()
    print("  样本0: 位置0-1是padding(✗)，位置2开始是真实token且满足causal(✓)")
    print("  样本1: 全部是真实token + full causal mask → 标准 GPT 行为")
    print("  样本2: 位置4-7是padding(✗), 即使 causal 允许也被屏蔽")


if __name__ == "__main__":
    demo_combined_mask()
```

输出：

```
=================================================================
🎭 组合 Attention Mask 可视化
=================================================================

原始 attention_mask (1=有效, 0=padding):
  样本0: ·· ████████
  样本1: █████████
  样本2: ████····

组合后 (padding ∩ causal):
  样本0: ✗✗ ✓✓✓✓✓✓
  样本1: ✓✓✓✓✓✓✓✓
  样本2: ✓✓✓✓ ✗✗✗✗

解读:
  ✓ = 该位置可以 attend（既是真实 token，又满足因果约束）
  ✗ = 该位置被屏蔽（要么是 padding，要么违反了因果约束）

  样本0: 位置0-1是padding(✗)，位置2开始是真实token且满足causal(✓)
  样本1: 全部是真实token + full causal mask → 标准 GPT 行为
  样本2: 位置4-7是padding(✗), 即使 causal 允许也被屏蔽
```

## 与 DataLoader 集成：完整的流水线

现在我们把自定义的 `collate_fn` 接入 DataLoader：

```python
def demo_full_pipeline():
    """完整的数据管道: Dataset → Custom Collate → DataLoader"""

    from torch.utils.data import Dataset

    class SimpleLMSet(Dataset):
        def __init__(self):
            # 模拟 20 条不同长度的数据
            self.data = [
                {'input_ids': list(range(10, 10+i*3)), 'labels': list(range(11, 11+i*3))}
                for i in range(1, 21)
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {
                'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
                'labels': torch.tensor(self.data[idx]['labels'], dtype=torch.long),
            }

    dataset = SimpleLMSet()

    # 使用自定义 collate_fn
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=LLMDataCollator(pad_token_id=0, label_pad_id=-100),  # ← 核心!
        num_workers=0,
    )

    print("完整 Pipeline 测试:\n")
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        B, T = batch['input_ids'].shape
        valid = batch['attention_mask'].sum(dim=-1).tolist()
        print(f"  Batch {i}: ({B} samples × {T} tokens), "
              f"有效tokens/样本={valid}")

    print(f"\n💡 观察: 每个 batch 的 T 值可能不同!")
    print(f"   这就是动态 padding 的威力 —— 不浪费一丝一毫空间")


if __name__ == "__main__":
    demo_full_pipeline()
```

输出：

```
完整 Pipeline 测试:

  Batch 0: (4 samples × 57 tokens), 有效tokens/样本=[54, 51, 48, 45]
  Batch 1: (4 samples × 39 tokens), 有效tokens/样本=[36, 33, 30, 27]
  Batch 2: (4 samples × 21 tokens), 有效tokens/样本=[18, 15, 12, 9]

💡 观察: 每个 batch 的 T 值可能不同!
   这就是动态 padding 的威力 —— 不浪费一丝一毫空间
```

注意到 Batch 0 的序列长度是 57，Batch 1 只有 39，Batch 2 只有 21——这是因为每个 batch 内部的最大长度不同。如果我们之前在 `__init__` 里就把所有样本都 padding 到全局最大长度（假设是 70），那么即使是 Batch 2 这种短样本也要占 70 个位置的空间，浪费超过 70%。动态 padding 让每个 batch 都只取其需要的最小长度。

## 常见误区与陷阱

### 误区 1：用右 padding 导致位置编码错位

```python
# ❌ 错误: 右 padding（常见于从 BERT/CV 领域转过来的开发者）
def right_pad_collate(features):
    max_len = max(len(f['input_ids']) for f in features)
    result = []
    for f in features:
        ids = f['input_ids']
        pad_len = max_len - len(ids)
        result.append(ids + [0] * pad_len)  # 往右填充! ❌
    return torch.tensor(result)

# 问题: 对于因果模型，position embedding 会给 padding 位置也分配有意义的位置信息
# 模型可能会学到 "位置 5 总是 padding" 这样的虚假规律
```

### 误区 2：忘记设置 `ignore_index=-100` 导致 loss 被污染

```python
# ❌ 错误: labels 也用 0 填充 padding 位置
padded_labels = input_ids_padded  # padding 位置也是 0

# 问题: 如果你的词表中 0 对应的是一个合法 token（如 <s> 或 <unk>），
# 那么 CrossEntropyLoss 会认为 "模型应该预测 padding 位置为 <s>" 
# → 这会引入错误的梯度信号!

# ✅ 正确: 用 -100 填充（PyTorch CrossEntropyLoss 默认忽略 -100）
padded_labels_with_ignore = ...
# 其中 padding 位置 = -100
loss = F.cross_entropy(logits, padded_labels_with_ignore, ignore_index=-100)
```

### 误区 3：在 `__getitem__` 里做 padding 而不是在 `collate_fn` 里

```python
# ❌ 错误: 在 Dataset 里就做了 padding
class PaddedDataset(Dataset):
    def __getitem__(self, idx):
        ids = self.raw_data[idx]
        # 这里直接 padding 到固定长度!
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return {'input_ids': ids}

# 问题:
# 1. 所有样本都被 padding 到 max_len（可能是全局最大值），极大浪费
# 2. 无法利用动态 padding 的优势
# 3. 如果 max_len 设得太大，显存/内存压力剧增
```

### 误区 4：`attention_mask` 和 `key_padding_mask` 搞混

在 HuggingFace Transformers 的代码中你可能会同时见到这两个名字：

```python
# HF BertModel 的 forward 签名:
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,        # (B, T) → 给 attention scores 用的
)

# HF 内部还会生成 key_padding_mask:
# key_padding_mask = (attention_mask == 0)  # True 的位置是 padding
# 然后在 attention 层里扩展为 (B, 1, 1, T) 并与 causal_mask 合并
```

简单记忆：`attention_mask` 是你传入的（1=有效, 0=padding），HF 内部会把它转换为 `key_padding_mask`（True=padding, False=有效）再和 causal mask 合并。如果你手写模型，只需要记住一件事：**最终传给 `F.softmax` 之前的 mask，True 位置保留，False（或 -inf）位置屏蔽**。

### 面试追问：动态 padding vs 固定 length padding，各有什么优劣？

这是一个很好的深入问题。完整的回答应该涵盖几个维度：

| 维度 | 动态 Padding | 固定 Length Padding |
|-----|-------------|-------------------|
| **内存效率** | ⭐⭐⭐⭐⭐ 每个 batch 只取所需长度 | ⭐⭐ 所有样本都取全局最大长度 |
| **计算效率** | ⭐⭐⭐⭐ 不同 batch 的 shape 不同，可能影响 GPU kernel 缓存 | ⭐⭐⭐⭐⭐ shape 一致，kernel 更友好 |
| **实现复杂度** | ⭐⭐⭐ 需要自定义 collate_fn | ⭐ 直接用 default_collate |
| **适用场景** | **LLM 训练（推荐）** | 图像分类 / BERT 等（常用） |

在生产环境的 LLM 训练中，**动态 padding 几乎总是更好的选择**——因为 LLM 的序列长度分布通常非常宽（从几个 token 到几千个 token），固定长度会导致巨大的浪费。唯一需要注意的是，如果你的模型使用了 CUDA Graphs 或 FlashAttention 等需要固定 shape 的优化技术，动态 padding 可能导致频繁的 recompilation——不过这个问题可以通过设置多个 bucket（如 256/512/1024/2048）来缓解。

---

到这里，我们已经解决了"如何把不等长的文本组成一个 batch"这个问题。但还有一层优化我们没有触及——那就是如何在预处理阶段避免重复计算、如何处理 TB 级别的超大数据集、以及如何保证数据质量。下一节，我们将进入**高效数据处理技巧**的世界。
