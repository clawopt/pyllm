# 模型剪枝：从非结构化到结构化的瘦身艺术

## 这一节讲什么？

如果说量化是"让每个参数用更少的比特表示"，蒸馏是"让小模型学会大模型的知识"，那么**剪枝（Pruning）**就是更直接的思路——**直接把不重要的参数去掉**。

想象一下园丁修剪树木：不是所有树枝都对树的生长和美观有贡献。那些枯萎的、遮挡阳光的、过度生长的枝条，剪掉后树反而长得更好。模型也是如此——很多权重接近零、对输出几乎没有影响，直接移除它们既减小了模型体积，有时甚至能提升泛化能力（正则化效应）。

这一节我们将学习：
1. **非结构化剪枝**——将小权重置零（稀疏化）
2. **结构化剪枝**——移除整个神经元/头/层
3. **彩票假说（Lottery Ticket Hypothesis）**——为什么稀疏子网络可能更强
4. **HF 实战**——使用 `torch.nn.utils.prune` 进行实际操作

---

## 一、剪枝的核心直觉

## 1.1 为什么可以"剪掉"参数？

```python
def why_pruning_works():
    """解释为什么剪枝有效"""

    import numpy as np

    # 模拟一个训练好的线性层的权重分布
    np.random.seed(42)
    weights = np.random.randn(10000) * 0.3  # 大部分值集中在 0 附近

    # 统计不同阈值下的稀疏度
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]

    print("=" * 65)
    print("权重分布与剪枝潜力")
    print("=" * 65)

    for t in thresholds:
        pruned_count = (np.abs(weights) < t).sum()
        ratio = pruned_count / len(weights) * 100

        bar = "█" * int(ratio / 2)
        empty = "░" * (50 - int(ratio / 2))

        print(f"\n阈值 |t| < {t:.2f}: {ratio:>6.1f}% 可被剪掉 [{bar}{empty}]")

    print(f"\n💡 关键发现:")
    print(f"   • 训练后的权重分布通常近似正态分布 N(0, σ²)")
    print(f"   • 大量权重的绝对值很小 (接近 0)")
    print(f"   • 这些'小权重'对输出的贡献微乎其微 → 可以安全移除")
    print(f"   • 剪枝不仅压缩模型, 还起到正则化作用 (防止过拟合)")

why_pruning_works()
```

## 1.2 非结构化 vs 结构化剪枝

```python
def structured_vs_unstructured():
    """两种剪枝方式的对比"""

    comparison = [
        ("维度", "非结构化", "结构化"),
        ("─" * 15, "─" * 25, "─" * 25),
        ("粒度", "单个参数 (元素级)", "整个结构单元 (行/列/头/层)"),
        ("结果", "稀疏矩阵 (很多 0)", "密集但更小的矩阵"),
        ("加速", "❌ 需要特殊硬件/库支持", "✅ 直接加速 (标准矩阵运算)"),
        ("精度保持", "★★★★★ (同参数量下最优)", "★★★☆☆ (粗粒度约束)"),
        ("实现难度", "简单 (torch.prune)", "中等 (需要自定义逻辑)"),
        ("适用硬件", "需要稀疏计算支持 (NVIDIA A100+)", "所有 GPU/CPU"),
        ("代表方法", "Magnitude Pruning, Movement Pruning",
         "Head Pruning, Layer Dropping"),
        ("生产可用性", "有限 (vLLM 部分支持)", "广泛 (直接导出为 Dense 模型)"),
    ]

    print("=" * 80)
    print("非结构化 vs 结构化剪枝")
    print("=" * 80)
    for row in comparison:
        print(f"{row[0]:<16} {row[1]:<26} {row[2]:<26}")

structured_vs_unstructured()
```

---

## 二、非结构化剪枝：Magnitude Pruning

## 2.1 最简单也最有效的剪枝方法

幅度剪枝（Magnitude Pruning）的思想极其朴素：**如果某个权重的绝对值小于某个阈值，就把它设为零**。

$$W_{pruned}[i,j] = \begin{cases} W[i,j] & \text{if } |W[i,j]| > t \\ 0 & \text{if } |W[i,j]| \leq t \end{cases}$$

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def magnitude_pruning_demo():
    """幅度剪枝完整演示"""

    print("=" * 65)
    print("幅度剪枝 (Magnitude Pruning) 完整演示")
    print("=" * 65)

    # 创建一个简单的线性层
    linear = nn.Linear(784, 256)

    # 查看原始状态
    total_params = sum(p.numel() for p in linear.parameters())
    nonzero_before = torch.count_nonzero(linear.weight).item()

    print(f"\n[初始状态]")
    print(f"  参数总数: {total_params:,}")
    print(f"  非零参数: {nonzero_before:,} ({nonzero_before/total_params*100:.1f}%)")

    # 执行 L1 Unstructured 剪枝
    # 移除权重中绝对值最小的 30% 的连接
    prune.l1_unstructured(linear, name="weight", amount=0.3)

    nonzero_after = torch.count_nonzero(linear.weight).item()
    sparsity = (1 - nonzero_after / total_params) * 100

    print(f"\n[剪枝后 (amount=0.3)]")
    print(f"  非零参数: {nonzero_after:,}")
    print(f"  稀疏度: {sparsity:.1f}%")

    # 注意: 被剪掉的值变成了 0, 但 shape 不变!
    print(f"\n💡 重要:")
    print(f"   • 权重形状不变: {linear.weight.shape}")
    print(f"   • 被剪的位置存储为 0 (sparse tensor)")
    print(f"   • 有一个 mask 缓冲区记录哪些位置被剪掉了")

    # 查看剪枝 mask
    if hasattr(linear, "weight_mask"):
        mask = linear.weight_mask
        zero_in_mask = (mask == 0).sum().item()
        print(f"   • Mask 中有 {zero_in_mask:,} 个位置被标记为可剪")

magnitude_pruning_demo()
```

## 2.2 HF 内置剪枝方法一览

```python
def hf_pruning_methods():
    """HF torch.nn.utils.prune 支持的方法"""

    methods = [
        {
            "函数": "prune.l1_unstructured",
            "类型": "非结构化",
            "依据": "L1 范数 (绝对值)",
            "效果": "移除 |w| 最小的参数",
            "推荐": "✅ 最常用, 效果稳定",
        },
        {
            "函数": "prune.random_unstructured",
            "类型": "非结构化",
            "依据": "随机选择",
            "效果": "随机移除一定比例的参数",
            "推荐": "用于基线对照实验",
        },
        {
            "函数": "prune.l1_structured",
            "类型": "结构化",
            "依据": "L1 范数 (按维度聚合)",
            "效果": "按行或列整体移除",
            "推荐": "适合需要标准 dense 输出的场景",
        },
        {
            "函数": "prune.identity",
            "类型": "无操作",
            "依据": "不剪枝",
            "效果": "只注册 hook, 方便后续自定义",
            "推荐": "用于实现自定义剪枝策略",
        },
    ]

    print("=" * 75)
    print("HF 内置剪枝方法")
    print("=" * 75)
    for m in methods:
        print(f"\n🔧 {m['函数']}")
        for k, v in m.items():
            if k != "函数":
                print(f"   {k}: {v}")

hf_pruning_methods()
```

## 2.3 全局剪枝 vs 局部剪枝

```python
def global_vs_local_pruning():
    """全局剪枝和局部剪枝的区别"""

    explanation = """
┌──────────────────────────────────────────────────────────────┐
│                    局部剪枝 (Local Pruning)                    │
│                                                              │
│  对每个层/每个模块独立设置剪枝率                              │
│                                                              │
│  Layer 1: 剪掉 20% 的参数                                    │
│  Layer 2: 剪掉 20% 的参数                                    │
│  Layer 3: 剪掉 20% 的参数      ← 每层一样                     │
│  ...                                                         │
│                                                              │
│  问题: 有些层本身就比较"稀疏"(重要参数少),                    │
│        强制每层剪相同比例可能剪掉了重要信息!                   │
├──────────────────────────────────────────────────────────────┤
│                    全局剪枝 (Global Pruning)                   │
│                                                              │
│  把所有层的参数放在一起排序, 按全局重要性决定谁该被剪         │
│                                                              │
│  所有参数排序 (按 |w|):                                     │
│  ┌────────────────────────────────────────────┐              │
│  │████████████████░░░░░░░░░░░░░░░░░░░░░░░░│              │
│  │← 保留 70% ──→│← 剪掉 30% (最小权重的) ─→│              │
│  └────────────────────────────────────────────┘              │
│                                                              │
│  结果: Layer 1 可能只剪 10%, Layer 3 可能剪 40%               │
│       (因为 Layer 3 本身就有更多不重要参数)                    │
│                                                              │
│  ✅ 更合理! 但实现复杂度更高                                  │
└──────────────────────────────────────────────────────────────┘
"""
    print(explanation)

global_vs_local_pruning()
```

---

## 三、结构化剪枝：Attention Head 与 Layer Dropping

## 3.1 Attention Head Pruning

对于 Transformer 模型来说，**注意力头（Attention Head）是最自然的剪枝单位**。研究表明，并非所有的注意力头都在同等程度地工作——有些头几乎可以被安全地移除。

```python
def attention_head_pruning():
    """注意力头剪枝演示"""

    import torch
    import torch.nn as nn
    import math

    class HeadPrunableMultiHeadAttention(nn.Module):
        """
        可剪枝注意力的 Multi-Head Attention
        """

        def __init__(self, hidden_size, num_heads, dropout=0.0):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads

            assert hidden_size % num_heads == 0

            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.o_proj = nn.Linear(hidden_size, hidden_size)

            self.dropout = nn.Dropout(dropout)
            
            # 头重要性分数 (用于剪枝决策)
            self.head_importance = nn.Parameter(torch.ones(num_heads))
            
            # 活跃头的掩码 (True = 保留, False = 剪掉)
            self.register_buffer("head_mask", torch.ones(num_heads, dtype=torch.bool))

        def forward(self, x, attention_mask=None):
            batch_size, seq_len, _ = x.shape

            # QKV 投影
            Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # 应用 head mask (关键!)
            Q = Q * self.head_mask.view(1, -1, 1, 1).float()
            K = K * self.head_mask.view(1, -1, 1, 1).float()
            V = V * self.head_mask.view(1, -1, 1, 1).float()

            # Scaled Dot-Product Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                scores = scores + attention_mask

            attn_weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
            attn_weights = self.dropout(attn_weights)

            context = torch.matmul(attn_weights, V)

            # 合并头
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

            return self.o_proj(context)

        def compute_head_importance(self, attn_weights):
            """
            根据注意力权重计算每个头的重要性
            简单方法: 取注意力权重的熵 (熵越高 = 关注越分散 = 可能越不重要)
            """
            batch, heads, q_len, k_len = attn_weights.shape
            
            # 方法 1: 平均注意力权重 (越均匀可能越不重要)
            avg_attn = attn_weights.mean(dim=(-1, -2))  # (batch, heads)
            
            # 方法 2: 使用熵作为指标
            entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=-1).mean(dim=-1)
            
            return entropy.mean(dim=0)  # (heads,)

        def prune_heads(self, heads_to_prune: list):
            """
            剪枝指定的注意力头
            """
            with torch.no_grad():
                for head_idx in heads_to_prune:
                    if head_idx < self.num_heads:
                        self.head_mask[head_idx] = 0
                        print(f"  ✂️  剪掉 Head #{head_idx}")

            active_heads = self.head_mask.sum().item()
            print(f"  剩余活跃头: {active_heads}/{self.num_heads}")

    # 演示
    mha = HeadPrunableMultiHeadAttention(hidden_size=768, num_heads=12)

    x = torch.randn(2, 128, 768)

    print("=" * 60)
    print("Attention Head Pruning 演示")
    print("=" * 60)
    print(f"\n初始配置: 12 个注意力头, head_dim = 64")

    output_full = mha(x)
    print(f"\n全头输出形状: {output_full.shape}")

    # 剪掉 4 个头 (33%)
    print(f"\n执行剪枝: 移除 Head [2, 5, 7, 11]:")
    mha.prune_heads([2, 5, 7, 11])

    output_pruned = mha(x)
    print(f"剪枝后输出形状: {output_pruned.shape}")

    # 参数量对比
    params_full = sum(p.numel() for p in mha.parameters() if p.requires_grad)
    print(f"\n💡 参数变化:")
    print(f"   虽然 Q/K/V/O 投影层大小不变, 但计算量减少了 ~33%")
    print(f"   (被剪掉的头在 Attention 计算中被跳过)")


if __name__ == "__main__":
    attention_head_pruning()
```

## 3.2 Layer Dropping：跳过整层

Layer Dropping 是一种更激进的结构化剪枝——直接跳过 Transformer 中的某些层：

```python
class LayerDroppedTransformer(nn.Module):
    """
    支持 Layer Dropping 的 Transformer Encoder
    """

    def __init__(self, base_model, layers_to_drop=None):
        super().__init__()
        self.base_model = base_model
        self.num_layers = len(base_model.layer)
        
        # 层掩码
        if layers_to_drop is None:
            layers_to_drop = []
        self.register_buffer(
            "layer_mask",
            torch.tensor([i not in layers_to_drop for i in range(self.num_layers)], 
                          dtype=torch.bool)
        )

    def forward(self, x, attention_mask=None):
        hidden_states = x
        
        for i, layer in enumerate(self.base_model.layer):
            if not self.layer_mask[i]:
                # 跳过这一层!
                continue
            
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
            )
            hidden_states = layer_output[0]

        return hidden_states


def demo_layer_dropping():
    """Layer Dropping 演示"""

    print("=" * 60)
    print("Layer Dropping 演示")
    print("=" * 60)

    configs = [
        {"name": "12 层全部保留", "drop": []},
        {"name": "跳过最后 2 层", "drop": [10, 11]},
        {"name": "均匀间隔跳过 4 层", "drop": [2, 5, 8, 11]},
    ]

    for cfg in configs:
        n_active = 12 - len(cfg["drop"])
        ratio = n_active / 12 * 100
        print(f"\n{cfg['name']}: {n_active}/12 层活跃 ({ratio:.0f}% 参数参与计算)")

demo_layer_dropping()
```

---

## 四、彩票假说（Lottery Ticket Hypothesis）

## 4.1 这个惊人的发现

2018 年, Frankle & Carlin 提出了一个令人震惊的假设：

> **在一个随机初始化的大神经网络中, 存在一个"中奖彩票"子网络（winning ticket）。如果我们找到这个子网络并从头开始只训练它, 它的效果将优于或等于训练整个网络的最终结果。**

换句话说：**重要的不是你训练了什么, 而是你一开始选对了哪些连接。**

```python
def lottery_ticket_hypothesis():
    """彩票假说解释"""

    explanation = """
┌──────────────────────────────────────────────────────────────┐
│                  彩票假说 (Lottery Ticket Hypothesis)          │
│                                                               │
│  实验:                                                       │
│  1. 随机初始化一个神经网络                                   │
│  2. 正常训练到收敛                                          │
│  3. 剪枝掉 p% 的最小权重 (得到稀疏网络 M_p)                │
│  4. 将 M_p 重置回初始权重 (但保持同样的稀疏模式!)             │
│  5. 用原来的学习率重新训练 M_p                               │
│                                                               │
│  发现:                                                      │
│  • M_p 在重新训练后能达到和完整网络相当甚至更好的效果!       │
│  • 这意味着: 初始时某些连接就是"注定成功"的                 │
│                                                               │
│  直觉解释:                                                   │
│  • 想象一张彩票, 刮开后发现中了奖                           │
│  • 如果你在刮之前就知道哪张会中奖...                         │
│  • 你只需要买那一张就够了!                                  │
│                                                               │
│  实践意义:                                                   │
│  • Iterative Magnitude Pruning (IMP):                      │
│     反复进行"训练→剪枝→重置→再训练"                        │
│     每轮增加稀疏度, 最终获得高稀疏度的强子网络               │
│                                                               │
│  ⚠️ 争议:                                                    │
│  • 后续研究发现在极大规模模型上 LT 效果减弱                   │
│  • 但核心思想仍启发了大量稀疏训练研究                       │
└──────────────────────────────────────────────────────────────┘
"""
    print(explanation)

lottery_ticket_hypothesis()
```

---

## 五、Movement Pruning：动态剪枝

## 5.1 不依赖幅度的剪枝方法

Magnitude Pruning 只看权重的静态值，而 **Movement Pruning**（Sanh et al., 2020）关注的是**权重在训练过程中的移动方向和幅度**。

核心思想：如果在训练过程中某个权重几乎没有变化（即梯度接近零），说明它对学习目标贡献不大，可以在训练结束时剪掉它。

```python
def movement_pruning_concept():
    """Movement Pruning 概念解释"""

    concept = """
传统 Magnitude Pruning:
  训练完成 → 看 |w_final| 的大小 → 剪掉小的
  
  问题: 最终权重小 ≠ 不重要!
       可能这个权重一开始很大, 在训练过程中逐渐变小
       但它在中间阶段起了重要作用

Movement Pruning:
  训练过程中持续记录:
    score(w_i) = |w_i(T) - w_i(0)| × g_i_mean  (移动幅度 × 平均梯度)
  
  → 移动小且梯度小的权重 → 真的不重要 → 安全剪掉
  → 移动大但最终值小 → 曾经重要 → 应该保留
"""
    print(concept)

movement_pruning_concept()
```

---

## 六、剪枝实战：端到端流程

```python
def end_to_end_pruning_pipeline():
    """完整的剪枝流水线"""

    pipeline = """
┌──────────────────────────────────────────────────────────────┐
│                  模型剪枝完整流水线                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 预训练 (或加载预训练模型)                           │
│         ↓                                                     │
│  Step 2: 评估基线性能 (accuracy/F1/PPL 等)                  │
│         ↓                                                     │
│  Step 3: 选择剪枝策略                                       │
│         ├─ 非结构化 (追求极致压缩比)                           │
│         ├─ 结构化 Head (Transformer 推荐)                     │
│         └─ 结构化 Layer (激进压缩)                             │
│         ↓                                                     │
│  Step 4: 执行剪枝                                           │
│         ├─ 设置目标稀疏度 (如 50%, 70%, 90%)                  │
│         ├─ 全局 or 局部剪枝                                   │
│         └─ 生成 mask                                         │
│         ↓                                                     │
│  Step 5: Fine-tuning (可选但推荐)                            │
│         ├─ 对剪枝后的模型做少量 epoch 的微调                    │
│         ├─ 学习率通常比原训练低 10x                           │
│         └─ 目标: 恢复因剪枝导致的精度损失                       │
│         ↓                                                     │
│  Step 6: 评估剪枝后性能                                      │
│         ├─ 对比基线: 精度损失 < X%?                            │
│         ├─ 测量推理速度提升                                    │
│         └─ 测量模型大小减少                                    │
│         ↓                                                     │
│  Step 7: 导出部署                                             │
│         ├─ 结构化剪枝: 直接导出 dense 小模型                   │
│         └─ 非结构化剪枝: 需要稀疏推理引擎支持                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
"""
    print(pipeline)

end_to_end_pruning_pipeline()
```

---

## 七、常见误区与最佳实践

## 7.1 误区

```python
def pruning_gotchas():
    """剪枝中的常见误区"""

    gotchas = [
        {
            "误区": "剪枝比例越大越好",
            "真相": "存在一个甜点 (sweet spot), 通常在 50%~90% 稀疏度之间;"
                   "超过这个范围后精度急剧下降。具体取决于模型和任务。",
        },
        {
            "误区": "剪枝后不需要微调",
            "真相": "除非是迭代式剪枝 (IMP), 否则剪枝后通常需要少量"
                   "Fine-tuning 来恢复精度。一般 1~3 个 epoch 就够了。",
        },
        {
            "误区": "非结构化剪枝一定能加速",
            "真相": "非结构化剪枝产生的是稀疏矩阵, 标准 BLAS 库无法高效处理。"
                   "需要特殊的稀疏计算库 (如 NVIDIA Sparse Tensor Cores)。"
                   "A100 及以上才真正受益于稀疏加速。",
        },
        {
            "误区": "所有层都同等重要",
            "真相": "研究发现: 靠近输入和输出的层更重要, 中间层相对次要;"
                   "Attention 层比 FFN 层更容易剪枝而不损失精度。",
        },
        {
            "误区": "剪枝和量化不能同时使用",
            "真相": "完全可以! 先剪枝再量化 (或反过来) 是常见的优化组合。"
                   "DistilWhisper 就是先蒸馏再量化再剪枝的三重优化案例。",
        },
    ]

    print("=" * 76)
    print("模型剪枝常见误区")
    print("=" * 76)
    for g in gotchas:
        print(f"\n❌ {g['误区']}")
        print(f"   💡 {g['真相']}")

pruning_gotchas()
```

## 7.2 剪枝 vs 量化 vs 蒸馏的选择指南

```python
def compression_method_selection():
    """压缩方法选择决策树"""

    guide = """
╔══════════════════════════════════════════════════════════╗
║           模型压缩方法选择决策树                              ║
╠══════════════════════════════════════════════════════════╣
║                                                              ║
║  你的首要目标是?                                            ║
║  ├─ 减少显存占用          → 优先考虑 量化 (INT4/NF4)        ║
║  ├─ 减少推理延迟          → 优先考虑 剪枝 (结构化) 或 蒸馏   ║
║  ├─ 减少模型文件大小      → 量化 + 剪枝 组合使用             ║
║  └─ 在边缘设备上运行      → 蒸馏成小模型 + 量化              ║
║                                                              ║
║  你的硬件环境?                                              ║
║  ├─ 有 A100/A8000          → 所有方法都可以尝试             ║
║  ├─ 只有 RTX 3090/4090     → 量化为主, 轻度剪枝为辅          ║
║  ├─ 只有 CPU               → 蒸馏 + ONNX + INT8 量化         ║
║  └─ 手机/嵌入式           → 极致蒸馏 + TFLite/QNNP          ║
║                                                              ║
║  你能接受多少精度损失?                                        ║
║  ├─ < 1%                    → 量化 (BF16/INT8) 即可          ║
║  ├─ 1% ~ 5%                → 4-bit 量化 或 轻度剪枝          ║
║  ├─ 5% ~ 15%               → 重度剪枝 或强蒸馏              ║
║  └─ > 15%                   → 需要组合多种方法 + 大量调优      ║
║                                                              ║
║  推荐的组合方案:                                            ║
║  🥇 最佳性价比:  QLoRA (量化 + 微调)                       ║
║  🥈 生产部署:    蒸-distill-量化-剪枝 三步走                 ║
║  🥉 边缘设备:    Tiny model + INT4/INT8 + TFLite              ║
╚══════════════════════════════════════════════════════════╝
"""
    print(guide)

compression_method_selection()
```

---

## 八、本章小结

这一节我们系统学习了模型剪枝的完整知识体系：

| 方法 | 粒度 | 结果形式 | 加速 | 适用场景 |
|------|------|---------|------|---------|
| **Magnitude (非结构)** | 单个参数 | 稀疏矩阵 | 需特殊硬件 | 研究/实验 |
| **Random (非结构)** | 单个参数 | 稀疏矩阵 | 需特殊硬件 | 基线对照 |
| **Head Pruning (结构)** | 整个注意力头 | 密集小矩阵 | ✅ 直接加速 | Transformer |
| **Layer Dropping (结构)** | 整个 Transformer 层 | 密集小矩阵 | ✅✅ 最大加速 | 极限压缩 |
| **Movement Pruning** | 单个参数 (动态) | 稀疏矩阵 | 需特殊硬件 | 训练时剪枝 |

**核心要点**：
1. **剪枝的本质是"去除冗余"**——训练后的模型中大量参数接近零
2. **结构化剪枝比非结构化剪枝更适合生产部署**——因为它产生标准的密集矩阵
3. **对于 Transformer，Attention Head 是最自然的剪枝单位**
4. **剪枝后通常需要少量 Fine-tuning 来恢复精度**
5. **在实际应用中，剪枝往往和量化、蒸馏组合使用**

下一节我们将学习 **推理加速引擎**——ONNX Runtime、TensorRT、vLLM、TGI 等工具如何让模型跑得更快。
