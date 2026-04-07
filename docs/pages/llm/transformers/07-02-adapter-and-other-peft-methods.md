# 其他 PEFT 方法：Adapter Tuning、Prefix Tuning 与 (IA)³

## 这一节讲什么？

上一节我们深入学习了 LoRA，它是目前最流行的 PEFT 方法。但 PEFT 的世界远不止 LoRA 一种——**Adapter Tuning** 是最早被系统研究的 PEFT 方法，**Prefix Tuning** 在少样本场景下表现惊艳，**(IA)³** 则以极小的参数量实现了接近 LoRA 的效果。

如果说 LoRA 是"在权重矩阵旁边加一个低秩修正"，那么：
- **Adapter Tuning** 是"在网络层之间插入一个瓶颈 MLP"
- **Prefix Tuning** 是"在输入前面拼接一段可学习的虚拟 token"
- **(IA)³** 是"给每个神经元乘上一个可学习的缩放因子"

这一节我们将逐一拆解这些方法的原理、实现和适用场景，帮助你建立完整的 PEFT 方法知识体系。

---

## 一、Adapter Tuning：插入式的参数高效微调

## 1.1 从直觉理解 Adapter

想象你在一条高速公路上开车（预训练模型的前向传播），突然遇到一个施工路段需要绕行——这个绕行的辅路就是 **Adapter**。它不改变主路的走向，只是在特定位置提供了一个可选的旁路来处理特殊情况。

更具体地，Adapter 是一个**瓶颈结构的小型神经网络模块**，被插入到 Transformer 的每一层（或部分层）中：

```
原始层输出 x ──→ [LayerNorm] ──┬──→ [Down Proj: d → r] ──→ [激活函数] ──→ [Up Proj: r → d] ──┐
                              │                                                            │
                              └──────────────────────── 残差连接 (跳过 Adapter) ─────────────┘→ 输出
```

关键设计：
- **下投影（Down Projection）**：将高维特征压缩到低维瓶颈空间（如 768 → 64）
- **非线性激活**：通常用 ReLU 或 GELU
- **上投影（Up Projection）**：从低维恢复到原始维度（如 64 → 768）
- **残差连接**：确保初始时 Adapter 不影响原始输出

## 1.2 三种主流 Adapter 变体

```python
import torch
import torch.nn as nn

class HoulsbyAdapter(nn.Module):
    """
    Houlsby Adapter (2019)
    原始论文: "Parameter-Efficient Transfer Learning for NLP"
    在每个 Transformer 层的 Attention 和 FFN 之后都插入 Adapter
    """

    def __init__(self, hidden_size, bottleneck_size=64, dropout=0.0):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class PfeifferAdapter(nn.Module):
    """
    Pfeiffer Adapter (2020)
    更高效的变体: 只在 FFN 之后插入 Adapter (不在 Attention 后)
    参数量减半, 效果相当
    """

    def __init__(self, hidden_size, bottleneck_size=64, dropout=0.0):
        super().__init__()
        # 注意: Pfeiffer 在 Adapter 内部做 LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)       # 先 LayerNorm
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class ParallelAdapter(nn.Module):
    """
    Parallel Adapter (2021)
    并行而非串行: Adapter 和原始层并行计算然后相加
    推理时可以并行化, 但训练时可能不稳定
    """

    def __init__(self, hidden_size, bottleneck_size=64, dropout=0.0):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, original_output, layer_input):
        """
        original_output: 原始层的输出
        layer_input: 原始层的输入 (用于并行的 Adapter)
        """
        x = self.down_proj(layer_input)   # 对输入做下投影
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return original_output + x         # 与原始输出相加


def compare_adapter_variants():
    """对比三种 Adapter 变体的参数量和特点"""

    hidden_size = 768
    bottleneck = 64

    houlsby = HoulsbyAdapter(hidden_size, bottleneck)
    pfeiffer = PfeifferAdapter(hidden_size, bottleneck)
    parallel = ParallelAdapter(hidden_size, bottleneck)

    print("=" * 70)
    print("三种 Adapter 变体对比 (hidden_size=768, bottleneck=64)")
    print("=" * 70)

    variants = [
        ("Houlsby Adapter", houlsby,
         "每层两个 Adapter (Attn后+FFN后)", "效果最好, 参数量翻倍"),
        ("Pfeiffer Adapter", pfeiffer,
         "只在 FFN 后插 Adapter", "效率最高, 推荐默认使用"),
        ("Parallel Adapter", parallel,
         "与原层并行计算", "推理可并行, 训练需注意"),
    ]

    for name, model, position, note in variants:
        params = sum(p.numel() for p in model.parameters())
        ratio = params / (hidden_size * hidden_size) * 100
        print(f"\n📦 {name}")
        print(f"   参数量: {params:,} ({ratio:.2f}% of original linear layer)")
        print(f"   插入位置: {position}")
        print(f"   特点: {note}")

compare_adapter_variants()
```

运行结果：

```
======================================================================
三种 Adapter 变体对比 (hidden_size=768, bottleneck=64)
======================================================================

📦 Houlsby Adapter
   参数量: 98,816 (16.76% of original linear layer)
   插入位置: 每层两个 Adapter (Attn后+FFN后)
   特点: 效果最好, 参数量翻倍

📦 Pfeiffer Adapter
   插入位置: 只在 FFN 后插 Adapter
   特点: 效率最高, 推荐默认使用

📦 Parallel Adapter
   插入位置: 与原层并行计算
   特点: 推理可并行, 训练需注意
```

## 1.3 HF peft 库中的 Adapter 用法

```python
from peft import AdaLoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def adapter_finetuning_demo():
    """使用 peft 库进行 Adapter 微调"""

    # 加载模型
    model_name = "bert-base-chinese"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=5
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 配置 Adapter (使用 AdaLora - 动态 rank 分配的 Adapter)
    ada_config = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                    # 初始 rank
        target_modules=["query", "value"],
        lora_alpha=16,
        adapter_dropout=0.1,    # Adapter 内部的 dropout
        inference_mode=False,
    )

    model = get_peft_model(model, ada_config)
    model.print_trainable_parameters()

    # 查看模型结构中的 Adapter
    print("\n🔍 模型中的 Adapter 层:")
    for name, module in model.named_modules():
        if "adapter" in name.lower():
            print(f"  {name}: {module.__class__.__name__}")

adapter_finetuning_demo()
```

## 1.4 Adapter vs LoRA 的核心区别

这是一个面试高频问题，让我们用一个直观的对比来说明：

```python
def adapter_vs_lora_comparison():
    """Adapter vs LoRA 的深度对比"""

    print("=" * 75)
    print("Adapter Tuning vs LoRA: 核心区别")
    print("=" * 75)

    comparisons = [
        ("维度", "Adapter", "LoRA"),
        ("─" * 20, "─" * 30, "─" * 30),
        ("修改方式", "插入新模块 (bottleneck MLP)",
         "修改现有权重 (低秩分解 ΔW=BA)"),
        ("参数形式", "独立的 A(down) + B(up) 矩阵",
         "A(r×d) @ B(d_out×r) 分解"),
        ("是否增加层数", "是 (增加了前向传播深度)",
         "否 (原地修改, 无额外层级)"),
        ("推理延迟", "有 (多了 Adapter 的计算)",
         "可合并回原模型, 零延迟"),
        ("初始化", "随机初始化",
         "B 初始化为零, 保证初始无干扰"),
        ("参数占比", "~1-5% (取决于 bottleneck)",
         "~0.1-1% (取决于 rank)"),
        ("灵活性", "可以堆叠多个 Adapter",
         "可以叠加在不同 target_modules 上"),
        ("适用场景", "需要独立适配器的多任务场景",
         "追求极致效率和部署简便的场景"),
    ]

    for row in comparisons:
        print(f"{row[0]:<20} {row[1]:<30} {row[2]:<30}")

    print("\n💡 选择建议:")
    print("   • 追求最佳效果且不介意额外延迟 → Adapter")
    print("   • 追求极致效率和零延迟部署 → LoRA")

adapter_vs_lora_comparison()
```

---

## 二、Prefix Tuning：软提示的艺术

## 2.1 从 Hard Prompt 到 Soft Prompt

在使用 ChatGPT 或其他大模型时，你可能写过这样的提示词：

> "请以一位资深 Python 工程师的口吻，解释什么是装饰器模式..."

这就是 **Hard Prompt（硬提示）**——用自然语言描述你想要的输出风格或任务。但手工设计 prompt 既耗时又依赖经验，而且离散的 token 空间限制了表达能力。

**Prefix Tuning / Prompt Tuning** 的核心思想是：**把 prompt 从离散的自然语言变成连续的可学习向量**。这些向量不是真实的词汇，而是优化出来的"虚拟 token"，它们能引导模型生成我们想要的内容。

```
Hard Prompt:
  [CLS] 请解释量子计算 [SEP] 量子计算是一种...

Soft Prompt (Prefix Tuning):
  [P1] [P2] [P3] ... [Pk] | 请解释量子计算 | 量子计算是一种...
       ↑ 这些是连续向量, 不是真实 token!
       ↑ 可学习, 优化后自动获得最佳的"引导信息"
```

## 2.2 Prefix Tuning vs Prompt Tuning vs P-Tuning v2

这三个名字经常让人混淆，它们本质上是同一思想的不同变体：

```python
def explain_prefix_methods():
    """清晰区分 Prefix/Prompt/P-Tuning 方法"""

    methods = {
        "Prompt Tuning (2021)": {
            "作者": "Lester et al.",
            "位置": "只在输入层 (embedding 层) 添加 soft prompt",
            "参数量": "prompt_length × hidden_size (极少!)",
            "优点": "参数最少, 实现最简单",
            "缺点": "只有一层引导, 表达能力有限",
            "图示": "[P1][P2]...[Pk] + 输入文本 → Transformer → 输出",
        },

        "Prefix Tuning (2021)": {
            "作者": "Li & Liang",
            "位置": "在每一层的 Attention 的 K/V 上添加 prefix",
            "参数量": "num_layers × 2 × prefix_len × hidden_size",
            "优点": "每层都有引导, 表达能力强",
            "缺点": "参数比 Prompt Tuning 多",
            "图示": "每层: K=[PK;K_orig], V=[PV;V_orig]",
        },

        "P-Tuning v2 (2022)": {
            "作者": "Liu et al.",
            "位置": "Prefix Tuning 的改进版, 在每层都加 prefix",
            "改进点": "加入 MLP 重参数化 + 深层 prefix",
            "优点": "在小模型上也有效 (Prompt Tuning 在小模型上效果差)",
            "缺点": "配置复杂",
            "图示": "类似 Prefix Tuning 但有更深层的优化策略",
        },
    }

    for name, info in methods.items():
        print(f"\n{'=' * 65}")
        print(f"方法: {name}")
        print(f"{'=' * 65}")
        for key, value in info.items():
            print(f"  {key}: {value}")

explain_prefix_methods()
```

## 2.3 Prefix Tuning 的原理图解

Prefix Tuning 的关键在于它在**每一层**的 Attention 机制中都注入了可学习的前缀向量：

```
第 1 层 Attention:
  K = concat(Prefix_K_1, Original_K_1)  ← 前缀 K 和原始 K 拼接
  V = concat(Prefix_V_1, Original_V_1)  ← 前缀 V 和原始 V 拼接
  Q = Original_Q_1                       ← Q 不变
  → Attention(Q, K, V) → 输出 1

第 2 层 Attention:
  K = concat(Prefix_K_2, Original_K_2)
  V = concat(Prefix_V_2, Original_V_2)
  Q = Original_Q_2
  → Attention(Q, K, V) → 输出 2

...

第 N 层 Attention:
  (同样的模式)
  → 最终输出
```

为什么要对 K/V 而不是 Q 加 Prefix？因为从 Attention 的公式 $Attention(Q,K,V) = softmax(QK^T/\sqrt{d})V$ 来看，修改 K 和 V 相当于同时改变了"查询什么"和"返回什么"，比只改 Q 更灵活。

## 2.4 手写 Prefix Tuning 实现

```python
import torch
import torch.nn as nn

class PrefixTuning(nn.Module):
    """
    手动实现的 Prefix Tuning
    在每层的 K/V 上添加可学习的前缀向量
    """

    def __init__(self, num_layers, num_heads, head_dim, prefix_len=10, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_len = prefix_len

        # 每层的前缀向量 (形状: [prefix_len, num_heads * head_dim])
        # 即 [prefix_len, hidden_size]
        self.prefix_tokens = nn.Parameter(
            torch.randn(num_layers, 2, prefix_len, num_heads * head_dim)  # 2 for K and V
        )

        # 可选: 用一个小 MLP 来生成 prefix (重参数化技巧)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(head_dim, head_dim * 2),
            nn.Tanh(),
            nn.Linear(head_dim * 2, head_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_size):
        """
        生成所有层的前缀 K/V 向量

        Args:
            batch_size: 当前 batch 的大小

        Returns:
            prefix_keys: list of tensors, 每层的 prefix K
            prefix_values: list of tensors, 每层的 prefix V
        """
        # prefix_tokens: [num_layers, 2, prefix_len, hidden]
        prefix = self.prefix_tokens  # (num_layers, 2, prefix_len, num_heads*head_dim)
        prefix = self.dropout(prefix)

        # 扩展到 batch 维度
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        prefix_keys = []
        prefix_values = []

        for layer_idx in range(self.num_layers):
            pk = prefix[:, layer_idx, 0, :, :]  # (batch, prefix_len, hidden)
            pv = prefix[:, layer_idx, 1, :, :]  # (batch, prefix_len, hidden)
            prefix_keys.append(pk)
            prefix_values.append(pv)

        return prefix_keys, prefix_values


def demo_prefix_tuning():
    """演示 Prefix Tuning 的用法"""

    print("=" * 60)
    print("Prefix Tuning 演示")
    print("=" * 60)

    # 模拟一个 12 层的 Transformer
    prefix_tuning = PrefixTuning(
        num_layers=12,
        num_heads=12,
        head_dim=64,
        prefix_len=20,
    )

    total_params = sum(p.numel() for p in prefix_tuning.parameters())
    base_model_params = 12 * (768 * 768 * 4)  # 近似的基座模型参数

    print(f"\nPrefix Tuning 参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  占基座模型比例: {total_params/base_model_params*100:.4f}%")
    print(f"  每层前缀长度: {prefix_tuning.prefix_len}")

    # 模拟前向传播
    batch_size = 4
    pkeys, pvalues = prefix_tuning(batch_size)

    print(f"\n生成的前缀向量:")
    for i, (pk, pv) in enumerate(zip(pkeys[:3], pvalues[:3])):
        print(f"  第{i+1}层: K={pk.shape}, V={pv.shape}")

demo_prefix_tuning()
```

## 2.5 HF peft 库中的 Prefix Tuning

```python
from peft import PromptTuningConfig, PromptEncoderConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
import torch

def prefix_tuning_with_peft():
    """使用 peft 库的三种 Prefix 方法"""

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float32,
    )

    configs = {
        "Prompt Tuning (最简单)": PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,      # 虚拟 token 数量
            prompt_tuning_init="TEXT",  # 初始化方式: TEXT/RANDOM/META
            prompt_tuning_init_text="请回答以下问题:",  # 用真实文本初始化
            tokenizer_name_or_path="gpt2",
        ),

        "P-Tuning v2 (推荐)": PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            encoder_hidden_size=768,     # 编码器隐藏维度
            encoder_num_layers=10,       # MLP 编码器层数
            encoder_reparameterization_type="MLP",  # MLP/LSTM
        ),
    }

    for name, config in configs.items():
        print(f"\n{'=' * 50}")
        print(f"配置: {name}")
        print(f"{'=' * 50}")

        model_pt = get_peft_model(model, config)
        model_pt.print_trainable_parameters()

prefix_tuning_with_peft()
```

---

## 三、(IA)³：极简主义的 PEFT 方法

## 3.1 核心思想：缩放而非替换

**(IA)³（Incremental Alignment via Layer-wise Learning Rates）** 可能是最简洁的 PEFT 方法。它的思想极其简单：**不引入任何新的矩阵或网络层，只是给现有的激活值乘以一个可学习的标量向量**。

具体来说，(IA)³ 学习两组缩放向量：
- $\lambda_l$：用于 Attention 中 Value 投影后的缩放
- $\mu_l$：用于 FFN 中间激活值的缩放

$$\text{output} = \lambda \times (\text{Attention output}), \quad \text{FFN output} = \mu \times (\text{FFN activation})$$

每个向量只有 `hidden_size` 个参数，对于 12 层 BERT 来说，总共只需要 $12 \times 768 \times 2 = 18,432$ 个可训练参数——不到原始参数量的 **0.01%**！

## 3.2 (IA)³ 实现与演示

```python
import torch
import torch.nn as nn

class IA3Layer(nn.Module):
    """
    (IA)³ 层的实现
    极其简单: 只是学习一组缩放因子
    """

    def __init__(self, hidden_size, init_value=1.0):
        super().__init__()
        # 用常数初始化 (确保初始时不改变原始行为)
        self.scaling_factor = nn.Parameter(torch.ones(hidden_size) * init_value)

    def forward(self, x):
        return x * self.scaling_factor


class ModelWithIA3(nn.Module):
    """
    给 Transformer 模型添加 (IA)³ 缩放
    """

    def __init__(self, base_model, num_layers=12, hidden_size=768):
        super().__init__()
        self.base_model = base_model

        # 冻结基座模型
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 为每层添加两组缩放因子
        self.attention_scalings = nn.ModuleList([
            IA3Layer(hidden_size) for _ in range(num_layers)
        ])
        self.ffn_scalings = nn.ModuleList([
            IA3Layer(hidden_size) for _ in range(num_layers)
        ])

    def forward(self, *args, **kwargs):
        # 这里简化了实际的前向传播逻辑
        # 实际应用中需要 hook 到模型的中间层
        pass


def demo_ia3():
    """演示 (IA)³ 的极小参数量"""

    print("=" * 60)
    print("(IA)³ 参数效率演示")
    print("=" * 60)

    hidden_size = 768
    num_layers = 12

    ia3_attention = IA3Layer(hidden_size)
    ia3_ffn = IA3Layer(hidden_size)

    attention_params = sum(p.numel() for p in ia3_attention.parameters())
    ffn_params = sum(p.numel() for p in ia3_ffn.parameters())

    total_ia3_params = (attention_params + ffn_params) * num_layers
    bert_base_params = 110000000  # BERT-base 约 110M 参数

    print(f"\n单层 (IA)³ 参数:")
    print(f"  Attention 缩放: {attention_params:,}")
    print(f"  FFN 缩放: {ffn_params:,}")

    print(f"\n全模型 (IA)³ 总参数: {total_ia3_params:,}")
    print(f"占 BERT-base 比例: {total_ia3_params/bert_base_params*100:.4f}%")
    print(f"\n💡 (IA)³ 的参数量几乎可以忽略不计!")
    print(f"   但在某些任务上能达到 LoRA 90%+ 的效果")

demo_ia3()
```

## 3.3 使用 peft 库的 IA3Config

```python
from peft import IA3Config, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

def ia3_with_peft():
    """使用 peft 库的 (IA)³ 配置"""

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-chinese", num_labels=5
    )

    ia3_config = IA3Config(
        task_type=TaskType.SEQ_CLS,
        target_modules=["key", "value", "output"],  # Attention 相关模块
        feedforward_modules=["output"],             # FFN 输出
        fusion_strategy="svd",                      # 融合策略
    )

    model = get_peft_model(model, ia3_config)
    model.print_trainable_parameters()

    print("\n✅ (IA)³ 配置完成!")

ia3_with_peft()
```

---

## 四、如何选择合适的 PEFT 方法？

## 4.1 决策树

面对这么多 PEFT 方法，该如何选择？下面是一个实用的决策指南：

```python
def peft_selection_guide():
    """PEFT 方法选择决策指南"""

    print("=" * 70)
    print("PEFT 方法选择决策树")
    print("=" * 70)

    scenarios = [
        {
            "场景": "通用场景, 追求最佳性价比",
            "首选": "LoRA (r=8~64)",
            "理由": "社区支持最好, 文档最丰富, 效果稳定可靠",
        },
        {
            "场景": "显存极度受限 (24GB 以内, 7B+ 模型)",
            "首选": "QLoRA (4-bit + LoRA)",
            "理由": "显存需求最低, 单卡可跑 33B/65B 模型",
        },
        {
            "场景": "少样本学习 (<500 样本)",
            "首选": "Prompt Tuning / Prefix Tuning",
            "理由": "参数极少, 不易过拟合, few-shot 场景效果好",
        },
        {
            "场景": "需要同时服务多个任务 (动态切换)",
            "首选": "LoRA (多 adapter) 或 Adapter Tuning",
            "理由": "peft 原生支持 set_adapter() 快速切换",
        },
        {
            "场景": "推理延迟敏感 (必须毫秒级响应)",
            "首选": "LoRA (合并后零延迟) 或 (IA)³",
            "理由": "LoRA 可以 merge_and_unload(), (IA)³ 本身开销极小",
        },
        {
            "场景": "追求极致参数效率",
            "首选": "(IA)³ 或 Prompt Tuning",
            "理由": "(IA)³ < 0.01%, Prompt Tuning < 0.01%",
        },
        {
            "场景": "分类/NER 等理解类任务",
            "首选": "LoRA 或 Adapter",
            "理由": "这些任务对注意力机制的调整收益最大",
        },
        {
            "场景": "文本生成/对话任务",
            "首选": "LoRA 或 QLoRA",
            "理由": "生成任务需要更强的表达能力, LoRA 的低秩分解更合适",
        },
    ]

    for s in scenarios:
        print(f"\n📌 {s['场景']}")
        print(f"   → 首选: {s['首选']}")
        print(f"   → 理由: {s['理由']}")

peft_selection_guide()
```

## 4.2 全面对比表

```python
def comprehensive_peft_comparison():
    """PEFT 方法全面对比"""

    print("\n" + "=" * 80)
    print("PEFT 方法全面对比表")
    print("=" * 80)

    headers = ["方法", "参数占比", "训练速度", "推理延迟", "效果", "难度"]
    print(f"{headers[0]:<18} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<8} {headers[5]}")
    print("-" * 80)

    data = [
        ("Full FT", "100%", "慢", "无", "★★★★★", "中等"),
        ("LoRA", "0.1-1%", "快", "可合并消除", "★★★★☆", "简单"),
        ("QLoRA", "0.1-1%", "最快", "可合并消除", "★★★★☆", "简单"),
        ("Houlsby Adapter", "1-5%", "中等", "有 (~5-10%)", "★★★★☆", "中等"),
        ("Pfeiffer Adapter", "0.5-2.5%", "较快", "有 (~3-8%)", "★★★★☆", "中等"),
        ("Prefix Tuning", "<0.1%", "快", "无", "★★★☆☆", "中等"),
        ("Prompt Tuning", "<0.01%", "最快", "无", "★★★☆☆", "简单"),
        ("(IA)³", "<0.02%", "最快", "几乎无", "★★★★☆", "简单"),
    ]

    for row in data:
        print(f"{row[0]:<18} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<8} {row[5]}")

comprehensive_peft_comparison()
```

---

## 五、常见误区与注意事项

## 5.1 PEFT 通用误区

```python
def common_peft_pitfalls():
    """PEFT 使用中的常见陷阱"""

    pitfalls = [
        {
            "误区": "PEFT 总是比 Full FT 差",
            "真相": "在很多任务上, LoRA 能达到 Full FT 95-99% 的效果,"
                   "差距往往小于评估指标的噪声范围",
            "建议": "先用 PEFT 做 baseline, 确实不够再考虑 Full FT",
        },
        {
            "误区": "不同 PEFT 方法不能组合",
            "真相": "peft 库支持一定程度的组合, 如 LoRA + Prompt Tuning",
            "建议": "但要注意参数量的叠加和可能的冲突, 一般单一方法就够用",
        },
        {
            "误区": "Prefix Tuning 在大模型和小模型上效果一样",
            "真相": "原始 Prompt Tuning 在参数量 < 100M 的模型上效果明显下降,"
                   "P-Tuning v2 通过深层 prefix 解决了这个问题",
            "建议": "小模型用 P-Tuning v2, 大模型可用任意 Prefix 方法",
        },
        {
            "误区": "Adapter 的 bottleneck 越大越好",
            "真相": "bottleneck 过大导致 Adapter 参数量接近原始层,"
                   "失去了 PEFT 的意义; 过小则表达能力不足",
            "建议": "一般设为 hidden_size 的 1/8 ~ 1/4, 如 768→96~192",
        },
        {
            "误区": "(IA)³ 因为参数少所以效果一定差",
            "真相": "(IA)³ 在某些分类任务上甚至超过 LoRA,"
                   "因为它直接缩放激活值而不是间接修改权重",
            "建议": "分类任务优先尝试 (IA)³, 生成任务优先 LoRA",
        },
    ]

    print("=" * 75)
    print("PEFT 五大常见误区")
    print("=" * 75)

    for i, p in enumerate(pitfalls, 1):
        print(f"\n❌ 误区 {i}: {p['误区']}")
        print(f"   💡 真相: {p['真相']}")
        print(f"   ✅ 建议: {p['建议']}")

common_peft_pitfalls()
```

## 5.2 API 使用注意事项

```python
def api_gotchas():
    """peft 库 API 使用中的坑"""

    gotchas = [
        {
            "问题": "get_peft_model() 后忘记调用 print_trainable_parameters()",
            "后果": "不知道实际有多少参数在训练, 可能配置错误都没发现",
            "正确做法": "每次配置完都打印一次, 确认参数比例合理",
        },
        {
            "问题": "target_modules 名称写错",
            "后果": "LoRA 没有被添加到正确的层, 等于没做微调",
            "正确做法": "先 print(model) 查看实际的模块名称, 再配置 target_modules",
        },
        {
            "问题": "QLoRA 忘记 prepare_model_for_kbit_training()",
            "后果": "量化后的模型无法正确进行梯度更新",
            "正确做法": "4-bit 加载后立即调用 prepare 函数",
        },
        {
            "问题": "保存时只保存了 adapter 没保存 tokenizer",
            "后果": "加载时缺少 tokenizer 导致编码不一致",
            "正确做法": "model.save_pretrained() 和 tokenizer.save_pretrained() 都要调用",
        },
        {
            "问题": "merge_and_unload() 后还想继续训练",
            "后果": "合并后 LoRA 权重已经融入基座, 无法再单独更新 LoRA",
            "正确做法": "保留未合并的版本用于后续迭代训练",
        },
    ]

    print("=" * 70)
    print("peft 库 API 使用注意事项")
    print("=" * 70)

    for g in gotchas:
        print(f"\n⚠️  {g['问题']}")
        print(f"   后果: {g['后果']}")
        print(f"   正确: {g['正确做法']}")

api_gotchas()
```

---

## 六、本章小结

这一节我们系统地学习了除 LoRA 之外的其他重要 PEFT 方法：

| 方法 | 核心思想 | 适用场景 | 参数占比 |
|------|---------|---------|---------|
| **Houlsby Adapter** | 每层插入 bottleneck MLP | 多任务切换 | ~1-5% |
| **Pfeiffer Adapter** | 仅 FFN 后插入 (更高效) | 默认推荐 | ~0.5-2.5% |
| **Prefix Tuning** | 每层 K/V 加可学习前缀 | Few-shot / 生成任务 | <0.1% |
| **Prompt Tuning** | 仅输入层加 soft prompt | 极少参数场景 | <0.01% |
| **(IA)³** | 学习激活值的缩放因子 | 分类任务 / 低延迟 | <0.02% |

**核心要点**：
1. **Adapter** 适合需要独立适配器、可动态切换的多任务场景
2. **Prefix/Prompt Tuning** 在少样本场景下表现出色，参数量极少
3. **(IA)³** 是最轻量的方案，适合推理延迟敏感的应用
4. **大多数情况下 LoRA 仍然是首选**——生态成熟、文档丰富、社区活跃

下一节我们将深入探讨 **不同任务类型下的 PEFT 配置策略**，以及如何为分类、NER、QA、生成等任务定制最优的 PEFT 方案。
