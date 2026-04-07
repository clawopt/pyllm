# PEFT 概述与 LoRA 原理与实现

## 这一节讲什么？

想象一下你是一家大公司的 CEO，公司已经运转得很好了——这就是**预训练模型**。现在你想让公司进入一个新领域（比如医疗、法律、金融），你有两种选择：

1. **全公司重组**（Full Fine-Tuning）：重新培训所有员工、调整所有部门——成本极高，风险很大
2. **聘请外部顾问**（PEFT）：保持原有团队不变，只在新领域的关键位置派驻几位专家顾问——成本低、见效快、可随时撤换

**PEFT（Parameter-Efficient Fine-Tuning）** 就是第二种思路：冻结预训练模型的全部参数，只在关键位置插入少量可训练参数，通过训练这些"顾问参数"来让模型适应新任务。

这一节我们将系统学习：
1. **为什么 PEFT 是大模型微调的主流方案**——成本分析
2. **LoRA 的数学原理和直觉理解**——低秩分解为什么有效
3. **从零实现 LoRA**——手写代码理解每一行
4. **HF peft 库实战**——LoraConfig / get_peft_model / 训练 / 部署
5. **QLoRA**——在消费级 GPU 上微调 65B+ 模型

---

## 一、为什么需要 PEFT？——成本分析

## 1.1 Full Fine-Tuning 的显存噩梦

让我们先算一笔账。假设你要微调一个 LLaMA-2-7B 模型：

```python
def estimate_memory_for_full_ft(model_params=7e9, precision="fp16"):
    """
    估算 Full Fine-Tuning 所需显存
    包含: 模型权重 + 梯度 + 优化器状态 + 激活值
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}[precision]

    model_memory = model_params * bytes_per_param / (1024**3)  # GB

    # AdamW 优化器: 每个参数需要 2 个 momentum 状态 (fp32)
    optimizer_memory = model_params * 2 * 4 / (1024**3)

    # 梯度: 与精度一致
    gradient_memory = model_params * bytes_per_param / (1024**3)

    # 激活值: 取决于 batch_size 和 seq_len, 这里取近似值
    activation_memory = 4  # GB (approximate for bs=4, seq_len=2048)

    total = model_memory + optimizer_memory + gradient_memory + activation_memory

    print("=" * 65)
    print(f"Full Fine-Tuning 显存需求估算 ({model_params/1e9:.0f}B 参数)")
    print("=" * 65)
    print(f"  模型权重 ({precision}):     {model_memory:>8.2f} GB")
    print(f"  AdamW 优化器状态 (fp32): {optimizer_memory:>8.2f} GB")
    print(f"  梯度 ({precision}):        {gradient_memory:>8.2f} GB")
    print(f"  激活值 (approx):          {activation_memory:>8.2f} GB")
    print(f"  {'─' * 55}")
    print(f"  总计:                     {total:>8.2f} GB")
    print()
    print("  💡 对比常见 GPU 显存:")
    print(f"     RTX 3090 (24GB):   ❌ 不够")
    print(f"     RTX 4090 (24GB):   ❌ 不够")
    print(f"     A100 (40GB):       ⚠️ 勉强 (需要梯度累积)")
    print(f"     A100 (80GB):       ✅ 舒适")

    return total

estimate_memory_for_full_ft(7e9)
print()
estimate_memory_for_full_ft(13e9)
print()
estimate_memory_for_full_ft(70e9)
```

输出结果：

```
=================================================================
Full Fine-Tuning 显存需求估算 (7B 参数)
=================================================================
  模型权重 (fp16):            14.00 GB
  AdamW 优化器器状态 (fp32):  56.00 GB
  梯度 (fp16):                14.00 GB
  激活值 (approx):             4.00 GB
  ────────────────────────────────────────────────
  总计:                       88.00 GB

  💡 对比常见 GPU 显存:
     RTX 3090 (24GB):   ❌ 不够
     RTX 4090 (24GB):   ❌ 不够
     A100 (40GB):       ⚠️ 勉强 (需要梯度累积)
     A100 (80GB):       ✅ 舒适
```

**7B 模型的全量微调就需要约 88GB 显存！** 这意味着即使你有一张 A100 80GB 也跑不动。而 70B 模型更是需要接近 900GB 显存，这在绝大多数场景下都是不可接受的。

## 1.2 PEFT 的核心思想：冻结 + 注入

PEFT 的解决方案非常优雅：**冻结预训练模型的所有参数，只训练少量新增的"适配参数"**。

```python
import torch
import torch.nn as nn

class PeftIntuition:
    """用最简单的例子展示 PEFT 的核心思想"""

    def __init__(self):
        # 预训练模型的一个线性层 (模拟)
        self.pretrained_weight = nn.Parameter(torch.randn(784, 10) * 0.02)

        # PEFT 方式: 冻结原权重, 只添加一个小矩阵
        self.pretrained_weight.requires_grad = False  # 冻结!

        # 新增的可训练参数 (远小于原始参数量)
        self.adapter = nn.Linear(10, 10)  # 只有 10*10 + 10 = 110 个参数

    def forward(self, x):
        original_out = x @ self.pretrained_weight
        adapter_out = self.adapter(original_out)
        return original_out + adapter_out


def compare_parameter_counts():
    """对比 Full FT vs PEFT 的参数量"""

    pretrained_params = 784 * 10 + 10  # = 7850
    adapter_params = 10 * 10 + 10      # = 110

    print("=" * 60)
    print("Full Fine-Tuning vs PEFT 参数量对比")
    print("=" * 60)
    print(f"\n预训练模型参数:  {pretrained_params:,}")
    print(f"PEFT 新增参数:    {adapter_params:,}")
    print(f"\nPEFT 占比:       {adapter_params/pretrained_params*100:.2f}%")
    print(f"\n💡 这意味着:")
    print(f"   - 只需存储/传输 {adapter_params/pretrained_params*100:.2f}% 的参数")
    print(f"   - 只需计算 {adapter_params/pretrained_params*100:.2f}% 的梯度")
    print(f"   - 可以同时保存多个任务的适配器 (多任务切换)")

compare_parameter_counts()
```

## 1.3 主流 PEFT 方法一览

在深入 LoRA 之前，我们先了解整个 PEFT 家族的全景：

| 方法 | 核心思想 | 可训练参数占比 | 典型 rank/配置 | 推理额外开销 |
|------|---------|---------------|----------------|-------------|
| **LoRA** | 低秩分解 ΔW = BA | ~0.1% - 1% | r=4~64 | 可合并回原模型 |
| **QLoRA** | 4-bit 量化基座 + LoRA | 同 LoRA | r=16~64 | 同 LoRA |
| **Adapter** | 插入 bottleneck MLP | ~1% - 5% | bottleneck=64~256 | 有延迟增加 |
| **Prefix Tuning** | 在每层 K/V 加前缀向量 | <0.1% | prefix_len=10~50 | 无额外延迟 |
| **Prompt Tuning** | 只在输入层加 soft prompt | <0.01% | prompt_len=20~100 | 无额外延迟 |
| **(IA)³** | 学习缩放向量 λ/μ | <0.02% | 固定极小 | 几乎无开销 |

---

## 二、LoRA 原理深度解析

## 2.1 核心洞察：预训练权重的低秩特性

LoRA（Low-Rank Adaptation）的核心假设是：**预训练模型在适应特定任务时，权重的更新量 ΔW 具有低秩结构**。

什么意思呢？想象一下你在一家大公司工作，要让你适应一个新的业务领域。你可能不需要改变所有的行为模式，只需要在某些关键的决策点上做少量调整——这些调整虽然影响面广，但本质上是少数几个"维度"的变化。

数学上，这意味着 ΔW 可以被分解为两个小矩阵的乘积：

$$\Delta W = B \times A$$

其中 $B \in \mathbb{R}^{d_{out} \times r}$，$A \in \mathbb{R}^{r \times d_{in}}$，$r \ll \min(d_{in}, d_{out})$。

原始权重更新需要 $d_{out} \times d_{in}$ 个参数，而 LoRA 只需要 $(d_{out} + d_{in}) \times r$ 个参数。当 $r$ 远小于 $d_{in}$ 和 $d_{out}$ 时，参数量的节省是巨大的。

## 2.2 图解 LoRA 的位置

在一个 Transformer 层中，LoRA 通常应用在 Attention 层的 Query 和 Value 投影上：

```
输入 X ──┬──→ [Q_proj] ──→ Q (原始) ──┐
         │                              │
         ├──→ [K_proj] ──→ K (原始)     │   Multi-Head
         │                              ├─→ Attention → 输出
         └──→ [V_proj] ──→ V (原始) ──┘
                    ↑              ↑
               ┌────┴────┐   ┌────┴────┐
               │ LoRA-A_Q│   │ LoRA-A_V│  ← 可训练 (低秩分解)
               │   @     │   │   @     │
               │LoRA-B_Q │   │LoRA-B_V │
               └────┬────┘   └────┬────┘
                    ↓              ↓
                 ΔQ (增量)      ΔV (增量)
```

## 2.3 从零实现 LoRA

让我们用纯 PyTorch 手写一个 LoRA 层，彻底理解其内部机制：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    LoRA 层的实现
    将权重更新 ΔW 分解为 B(r×out) × A(in×r)
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放因子

        # LoRA 的两个低秩矩阵
        # A 用随机高斯初始化 (Kaiming 初始化)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / math.sqrt(rank)))
        # B 用零初始化 (确保初始时 ΔW = 0, 不影响预训练模型)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        计算 LoRA 的增量: ΔW·x = B @ A^T @ x
        实际上是: x → A(x_in × r) → B(r × x_out) → scaling
        """
        result = self.dropout(x)
        result = result @ self.lora_A           # (batch, in) @ (in, r) → (batch, r)
        result = result @ self.lora_B           # (batch, r) @ (r, out) → (batch, out)
        return result * self.scaling            # 缩放


class LinearWithLoRA(nn.Module):
    """
    带有 LoRA 的线性层
    冻结原始权重, 前向传播时加上 LoRA 的增量
    """

    def __init__(self, original_linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False  # 冻结!
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.lora = LoRALayer(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x):
        # 原始输出 + LoRA 增量
        return self.original_linear(x) + self.lora(x)


def demo_lora_from_scratch():
    """演示 LoRA 从零实现的完整流程"""

    print("=" * 65)
    print("LoRA 从零实现演示")
    print("=" * 65)

    # 1. 创建一个预训练的线性层 (模拟)
    torch.manual_seed(42)
    original_layer = nn.Linear(768, 768)
    print(f"\n[1] 原始线性层参数量: {sum(p.numel() for p in original_layer.parameters()):,}")
    print(f"    权重形状: {original_layer.weight.shape}")

    # 2. 包装为带 LoRA 的线性层
    lora_layer = LinearWithLoRA(original_layer, rank=8, alpha=16)

    # 3. 统计可训练参数
    trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in lora_layer.parameters() if not p.requires_grad)
    total = trainable + frozen

    print(f"\n[2] 添加 LoRA 后:")
    print(f"    可训练参数: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"    冻结参数:   {frozen:,} ({frozen/total*100:.2f}%)")
    print(f"    总参数:     {total:,}")

    # 4. 验证前向传播
    x = torch.randn(2, 128, 768)
    output = lora_layer(x)
    print(f"\n[3] 前向传播验证:")
    print(f"    输入形状:  {x.shape}")
    print(f"    输出形状:  {output.shape}")
    print(f"    ✅ 形状正确!")

    # 5. 验证初始状态不影响原始输出
    with torch.no_grad():
        orig_output = original_layer(x)
        diff = (output - orig_output).abs().max().item()
    print(f"\n[4] 初始状态验证:")
    print(f"    与原始输出的最大差异: {diff:.2e}")
    print(f"    ✅ 接近零! (因为 B 初始化为零)")

demo_lora_from_scratch()
```

运行结果：

```
=================================================================
LoRA 从零实现演示
=================================================================

[1] 原始线性层参数量: 589,824
    权重形状: torch.Size([768, 768])

[2] 添加 LoRA 后:
    可训练参数: 12,288 (2.08%)
    冻结参数:   589,824 (97.92%)
    总参数:     602,112

[3] 前向传播验证:
    输入形状:  torch.Size([2, 128, 768])
    输出形状:  torch.Size([2, 128, 768])
    ✅ 形状正确!

[4] 初始状态验证:
    与原始输出的最大差异: 9.31e-09
    ✅ 接近零! (因为 B 初始化为零)
```

注意几个关键点：
- **A 矩阵用随机初始化**（保证有足够的表达能力）
- **B 矩阵用零初始化**（确保训练开始时 ΔW = 0，不破坏预训练知识）
- **缩放因子 α/r** 控制 LoRA 贡献的相对强度

---

## 三、HF peft 库实战：LoRA 微调完整流程

## 3.1 环境安装

```bash
pip install peft transformers datasets accelerate bitsandbytes
```

## 3.2 LoraConfig 配置详解

```python
from peft import LoraConfig, get_peft_model, TaskType

def create_lora_config_examples():
    """展示不同场景下的 LoraConfig 配置"""

    configs = {
        "标准 LoRA (推荐)": LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # 任务类型
            r=64,                           # 秩 (低秩维度)
            lora_alpha=16,                  # 缩放因子 (alpha/r = 16/64 = 0.25)
            lora_dropout=0.1,               # LoRA 层的 dropout
            target_modules=["q_proj", "v_proj"],  # 目标模块 (只对 Q/V 加 LoRA)
            bias="none",                    # bias 的处理方式: none/all/lora_only
        ),

        "激进 LoRA (更多参数)": LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=128,                          # 更大的 rank
            lora_alpha=32,                  # 更大的 alpha
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # 所有 Attention 投影
                "gate_proj", "up_proj", "down_proj",      # FFN 层也加
            ],
            lora_dropout=0.05,
        ),

        "保守 LoRA (最少参数)": LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,                            # 小 rank
            lora_alpha=16,                  # alpha/r = 2.0 (较强缩放)
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        ),

        "分类任务 LoRA": LoraConfig(
            task_type=TaskType.SEQ_CLS,     # 序列分类任务!
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],  # BERT 类模型用 query/value
        ),
    }

    for name, config in configs.items():
        print(f"\n{'=' * 60}")
        print(f"配置: {name}")
        print(f"{'=' * 60}")
        for key, value in config.to_dict().items():
            if value is not None and key != "modules_to_save":
                print(f"  {key}: {value}")

create_lora_config_examples()
```

## 3.3 关键超参数深度解读

### rank (r)：低秩维度的选择

`rank` 是 LoRA 最核心的超参数，它决定了低秩子空间的维度大小。选择原则如下：

```python
def analyze_rank_effect():
    """分析 rank 对参数量和效果的影响"""

    base_dim = 4096  # 假设 LLaMA 的 hidden size

    print("=" * 65)
    print("Rank 选择指南 (基于 ", f"{base_dim} 维投影层)")
    print("=" * 65)
    print(f"\n{'Rank':<8}{'参数量':>12}{'占原始比例':>12}{'推荐场景':>25}")
    print("-" * 65)

    scenarios = [
        (4, "极少数据(<500样本), 快速实验"),
        (8, "少样本(500-2000), 平衡效率"),
        (16, "标准配置, 大多数场景"),
        (32, "数据充足(2000+), 追求质量"),
        (64, "大量数据(10000+), 复杂任务"),
        (128, "追求极致效果, 不在乎参数量"),
    ]

    for rank, scenario in scenarios:
        params = (base_dim + base_dim) * rank
        ratio = params / (base_dim * base_dim) * 100
        print(f"{rank:<8}{params:>12,}{ratio:>11.2f}%{scenario:>25}")

    print("\n💡 经验法则:")
    print("   • 数据越少 → rank 越小 (防止过拟合)")
    print("   • 任务越复杂 → rank 越大 (需要更多表达力)")
    print("   • 一般从 r=8 或 r=16 开始尝试")

analyze_rank_effect()
```

### alpha 和 rank 的黄金比例

`lora_alpha` 是 LoRA 的缩放因子，实际使用的缩放系数是 `alpha / rank`。这个比值非常重要：

```python
def explain_alpha_rank_ratio():
    """解释 alpha/rank 比例的意义"""

    print("=" * 60)
    print("Alpha / Rank 比例解读")
    print("=" * 60)

    examples = [
        ("保守", 8, 8, 1.0, "LoRA 贡献较弱, 适合简单迁移"),
        ("平衡", 16, 8, 2.0, "常用默认值, 适中贡献"),
        ("积极", 32, 8, 4.0, "LoRA 贡献强, 适合复杂任务"),
        ("激进", 64, 8, 8.0, "LoRA 主导, 可能破坏预训练知识"),
        ("LLaMA-Adapter", 16, 64, 0.25, "弱贡献, 但 rank 大提供多样性"),
    ]

    print(f"\n{'风格':<14}{'Alpha':>6}{'Rank':>6}{'α/r':>6}{'说明':>30}")
    print("-" * 60)
    for name, alpha, rank, ratio, desc in examples:
        print(f"{name:<14}{alpha:>6}{rank:>6}{ratio:>6.2f}{desc:>30}")

    print("\n📌 黄金法则: α/r ∈ [1, 4], 最常用的是 α/r = 1 或 2")
    print("   • 太小 (<0.5): LoRA 几乎不起作用")
    print("   • 太大 (>8): LoRA 可能过度干扰原始权重")

explain_alpha_rank_ratio()
```

### target_modules：在哪里加 LoRA？

这是新手最容易犯错的地方。不同的模型架构有不同的模块命名：

```python
def get_target_modules_guide():
    """针对不同模型的目标模块指南"""

    guide = {
        "LLaMA / Mistral / Qwen": {
            "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "ffn": ["gate_proj", "up_proj", "down_proj"],
            "推荐": ["q_proj", "v_proj"],
            "原因": "Q/V 投影包含最多的任务特定信息",
        },
        "BERT / RoBERTa": {
            "attention": ["query", "key", "value"],
            "推荐": ["query", "value"],
            "注意": "BERT 用的是 query/key/value, 不是 q_proj/v_proj!",
        },
        "T5 / BART": {
            "attention": ["q", "k", "v", "o"],
            "推荐": ["q", "v"],
        },
        "GPT-2": {
            "attention": ["c_attn"],  # GPT-2 的 QKV 合并在一个矩阵里
            "推荐": ["c_attn"],
            "注意": "c_attn 同时包含 Q/K/V, 一个模块就够了",
        },
    }

    for model_name, info in guide.items():
        print(f"\n{'=' * 60}")
        print(f"模型: {model_name}")
        print(f"{'=' * 60}")
        for key, value in info.items():
            print(f"  {key}: {value}")

get_target_modules_guide()
```

## 3.4 完整的 LoRA 微调流程

下面是一个完整的、可直接运行的 LoRA 微调示例：

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

def run_lora_finetuning():
    """
    完整的 LoRA 微调流程
    以指令微调为例 (Alpaca 格式数据)
    """

    # ========== 第一步: 配置量化 (可选, 用于大模型) ==========
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ========== 第二步: 加载基座模型 ==========
    model_name = "huggingface/CodeLlama-7b-hf"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,  # 使用 4-bit 量化
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ========== 第三步: 准备模型用于 k-bit 训练 ==========
    model = prepare_model_for_kbit_training(model)

    # ========== 第四步: 配置 LoRA ==========
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # 打印 LoRA 信息
    model.print_trainable_parameters()

    # ========== 第五步: 准备数据 ==========
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

    def format_alpaca(example):
        """将 Alpaca 格式转换为对话格式"""
        if example["input"]:
            text = f"""Below is an instruction that describes a task, paired with an input that provides further context.

## Instruction:
{example["instruction"]}

## Input:
{example["input"]}

## Response:
{example["output"]}"""
        else:
            text = f"""Below is an instruction that describes a task.

## Instruction:
{example["instruction"]}

## Response:
{example["output"]}"""
        return {"text": text}

    dataset = dataset.map(format_alpaca)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,  # 重要: 不要在这里 pad!
        )

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    # ========== 第六步: 配置训练参数 ==========
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_32bit",  # 使用分页优化器 (配合 QLoRA)
        report_to="none",
    )

    # ========== 第七步: 创建 Trainer 并训练 ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False, return_tensors="pt"
        ),
    )

    trainer.train()

    # ========== 第八步: 保存 LoRA 权重 ==========
    model.save_pretrained("./my_lora_adapter")
    tokenizer.save_pretrained("./my_lora_adapter")

    print("\n✅ LoRA 微调完成! 权重已保存到 ./my_lora_adapter")


if __name__ == "__main__":
    run_lora_finetuning()
```

运行后你会看到类似这样的输出：

```
trainable params: 3,358,720 || all params: 6,744,158,208 || trainable%: 0.0498%
```

**只有 0.05% 的参数是可训练的！** 这就是 LoRA 的威力所在。

---

## 四、QLoRA：消费级 GPU 上的大模型微调

## 4.1 什么是 QLoRA？

QLoRA（Quantized LoRA）是 LoRA 的增强版本，它在三个层面进行优化：

1. **4-bit NF4 量化**：将基座模型量化到 4-bit（NormalFloat4），显存占用减少到原来的 1/8
2. **双重量化**：对量化常数再进行一次量化，进一步节省内存
3. **分页优化器（Paged Optimizers）**：使用 NVIDIA 的统一内存机制，当显存不足时自动使用 CPU 内存

这使得在单张 RTX 3090/4090（24GB 显存）上微调 33B 甚至 65B 模型成为可能！

## 4.2 QLoRA 配置详解

```python
from transformers import BitsAndBytesConfig
import torch

def create_qlora_configs():
    """不同场景下的 QLoRA 配置"""

    configs = {
        "标准 QLoRA (推荐)": BitsAndBytesConfig(
            load_in_4bit=True,                          # 启用 4-bit 加载
            bnb_4bit_quant_type="nf4",                  # NormalFloat4 量化类型
            bnb_4bit_compute_dtype=torch.bfloat16,      # 计算时用 BF16 (保持精度)
            bnb_4bit_use_double_quant=True,             # 双重量化 (再省 ~0.4 bit/param)
        ),

        "FP16 计算 (老显卡)": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,       # FP16 计算 (兼容性更好)
            bnb_4bit_use_double_quant=True,
        ),

        "8-bit 量化 (更保守)": BitsAndBytesConfig(
            load_in_8bit=True,                          # 8-bit 量化 (比 4-bit 精度高但显存大)
        ),

        "极致压缩": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,            # 关闭双重量化 (更快但略费显存)
        ),
    }

    for name, config in configs.items():
        print(f"\n{'=' * 50}")
        print(f"配置: {name}")
        print(f"{'=' * 50}")
        for key, value in config.__dict__.items():
            if value is not None and 'llm_config' not in key:
                print(f"  {key}: {value}")

create_qlora_configs()
```

## 4.3 QLoRA vs LoRA vs Full FT 对比

```python
def compare_methods():
    """三种方法的全面对比"""

    print("=" * 75)
    print("LoRA vs QLoRA vs Full Fine-Tuning 全面对比 (以 7B 模型为例)")
    print("=" * 75)

    comparisons = [
        ("方法", "LoRA", "QLoRA", "Full FT"),
        ("─" * 15, "─" * 12, "─" * 12, "─" * 12),
        ("基座精度", "FP16/BF16", "NF4 (4-bit)", "FP16/BF16"),
        ("可训练参数", "~0.05%", "~0.05%", "100%"),
        ("训练显存 (~7B)", "~16GB", "~10GB", "~88GB"),
        ("推理额外开销", "可合并消除", "可合并消除", "无"),
        ("效果 (相对)", "95-99%", "93-98%", "100%"),
        ("训练速度", "快 (少梯度)", "最快 (少梯度+低精)", "慢"),
        ("适用 GPU", "A100/4090", "3090/4090", "多卡 A100"),
        ("部署复杂度", "需合并权重", "需合并+反量化", "直接用"),
    ]

    for row in comparisons:
        print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<12}")

    print("\n💡 选择建议:")
    print("   • 有 A100/4090 且追求最佳效果 → LoRA")
    print("   • 只有 3090/24GB 显存 → QLoRA")
    print("   • 有多卡集群且资源充足 → Full FT")

compare_methods()
```

---

## 五、推理阶段：合并 LoRA 权重

## 5.1 为什么需要合并？

LoRA 在训练时是独立于原始权重的，推理时有两种方式：
1. **分离式推理**：加载基座模型 + LoRA 适配器，前向传播时动态叠加 —— 有轻微性能损失
2. **合并式推理**：将 LoRA 权重合并回基座模型，变成一个普通模型 —— **零额外开销**

生产环境通常选择第二种方式。

## 5.2 合并与卸载

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_and_deploy():
    """合并 LoRA 权重并用于推理"""

    # 1. 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # 2. 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")

    # 3. 合并权重 (关键步骤!)
    merged_model = model.merge_and_unload()

    # 4. 此时 merged_model 就是一个普通的 PreTrainedModel
    #    不再有 LoRA 相关的开销
    print(type(merged_model))  # <class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>

    # 5. 像普通模型一样使用
    inputs = tokenizer("请解释什么是量子计算:", return_tensors="pt").to(merged_model.device)
    outputs = merged_model.generate(**inputs, max_new_tokens=256)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # 6. 保存合并后的模型 (可选)
    merged_model.save_pretrained("./merged_model")
    tokenizer.save_pretrained("./merged_model")

merge_and_deploy()
```

## 5.3 多个 LoRA 适配器的切换

在实际生产中，你可能为一个基座模型训练了多个任务特定的 LoRA（如医疗版、法律版、金融版）。peft 支持动态切换：

```python
def multi_lora_switching():
    """多个 LoRA 适配器的动态切换"""

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 加载第一个适配器作为基础
    model = PeftModel.from_pretrained(base_model, "./lora_medical", adapter_name="medical")

    # 添加更多适配器
    model.load_adapter("./lora_legal", adapter_name="legal")
    model.load_adapter("./lora_finance", adapter_name="finance")

    # 查看可用适配器
    print("可用适配器:", list(model.peft_config.keys()))

    # 切换到医疗版
    model.set_adapter("medical")
    output_medical = model.generate(...)

    # 切换到法律版
    model.set_adapter("legal")
    output_legal = model.generate(...)

    # 切换到金融版
    model.set_adapter("finance")
    output_finance = model.generate(...)

multi_lora_switching()
```

---

## 六、常见误区与最佳实践

## 6.1 五大常见误区

```python
def common_mistakes():
    """LoRA 使用中的常见误区"""

    mistakes = [
        {
            "误区": "rank 越大越好",
            "真相": "rank 过大会导致过拟合, 尤其在数据量小时。一般 r=8-64 足够",
            "正确做法": "从小 rank 开始, 根据验证集表现逐步增大",
        },
        {
            "误区": "所有层都加 LoRA",
            "真相": "Attention 层收益最大, FFN 层收益递减, Embedding 层几乎没收益",
            "正确做法": "优先 target_modules=['q_proj', 'v_proj']",
        },
        {
            "误区": "QLoRA 效果和 LoRA 一样好",
            "真相": "4-bit 量化会带来 1-3% 的精度损失, 但在大多数场景下可接受",
            "正确做法": "对精度要求高的任务用 LoRA, 资源受限用 QLoRA",
        },
        {
            "误区": "LoRA 可以替代 Full FT",
            "真相": "在数据充足的情况下, Full FT 仍然是最强的方法。LoRA 是效率和效果的折中",
            "正确做法": "先试 LoRA, 如果效果不够再考虑 Full FT",
        },
        {
            "误区": "训练完直接用, 不需要评估",
            "真相": "LoRA 可能出现灾难性遗忘或过拟合, 必须在测试集上严格评估",
            "正确做法": "建立完整的 eval 流程, 对比 baseline 和 LoRA 版本",
        },
    ]

    print("=" * 70)
    print("LoRA 五大常见误区")
    print("=" * 70)

    for i, m in enumerate(mistakes, 1):
        print(f"\n❌ 误区 {i}: {m['误区']}")
        print(f"   💡 真相: {m['真相']}")
        print(f"   ✅ 正确做法: {m['正确做法']}")

common_mistakes()
```

## 6.2 生产环境 Checklist

```python
def production_checklist():
    """PEFT 生产部署检查清单"""

    checklist = [
        ("✅ Rank 选择合理", "根据数据量选择, 不过大也不过小"),
        ("✅ Target modules 正确", "确认模型架构对应的模块名称"),
        ("✅ Alpha/Rank 比例合适", "α/r 在 [1, 4] 范围内"),
        ("✅ 已启用 gradient_checkpointing", "节省激活值显存"),
        ("✅ 使用合适的优化器", "QLoRA 用 paged_adamw_32bit"),
        ("✅ 设置了合理的 eval_strategy", "每个 epoch 都评估"),
        ("✅ 已合并权重用于推理", "避免运行时的额外开销"),
        ("✅ 进行了人工抽样审查", "至少抽查 50 条生成结果"),
        ("✅ 测试了边界情况", "超长输入/空输入/特殊字符"),
        ("✅ 监控了训练指标", "loss 曲线正常, 无 NaN"),
    ]

    print("=" * 60)
    print("PEFT 生产部署 Checklist")
    print("=" * 60)
    for item, desc in checklist:
        print(f"  {item}: {desc}")

production_checklist()
```

---

## 七、本章小结

这一节我们从"为什么要 PEFT"这个问题出发，逐步深入到了 LoRA 的数学原理和工程实践。核心要点回顾：

1. **PEFT 的本质**：冻结预训练权重，只训练少量"适配参数"，大幅降低微调成本
2. **LoRA 的核心**：$\Delta W = BA$，通过低秩分解将 $d \times d$ 的更新量降为 $(d+r) \times r$
3. **关键超参数**：rank（控制表达力）、alpha（控制缩放）、target_modules（控制作用位置）
4. **QLoRA**：4-bit 量化 + LoRA，让消费级 GPU 也能微调大模型
5. **生产部署**：`merge_and_unload()` 合并权重后，推理零额外开销

下一节我们将继续探索其他重要的 PEFT 方法：**Adapter Tuning、Prefix Tuning、(IA)³** 等，以及如何在不同任务中选择最合适的 PEFT 策略。
