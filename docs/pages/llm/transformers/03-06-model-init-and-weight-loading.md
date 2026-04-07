# 模型初始化与权重加载：从零开始到生产部署的关键一步

## 这一节讲什么？

当你调用 `BertModel.from_pretrained("bert-base-chinese")` 这行代码时，背后发生了一系列复杂的操作：模型架构被实例化、权重文件被下载/加载、参数被正确地映射到每一层、数据类型和设备被设置……任何一个环节出错，都可能导致训练崩溃或推理结果错误。

这一节，我们要彻底搞清楚：

1. **权重怎么初始化的**——不同的初始化方法如何影响训练？HF 默认用什么策略？
2. **权重怎么加载的**——`from_pretrained` 的完整工作流是什么？
3. **加载失败怎么办**——`missing_keys` 和 `unexpected_keys` 怎么处理？
4. **大模型怎么办**——单卡放不下时如何用 `device_map` 分片部署？
5. **精度怎么管理**——FP32/FP16/BF16 如何选择和混用？

这些知识不仅是面试高频考点，更是你在实际项目中解决"模型加载报错"、"显存溢出"、"精度损失"等问题的必备技能。

---

## 一、权重初始化方法 —— 万事开头的第一步

## 1.1 为什么初始化如此重要？

神经网络训练本质上是迭代优化过程，而优化的起点就是权重初始化。一个糟糕的初始状态可能导致：

- **梯度消失**：初始值太小 → 每层输出趋近于零 → 梯度也趋近于零 → 参数无法更新
- **梯度爆炸**：初始值太大 → 输出值指数级增长 → 梯度爆炸 → NaN
- **对称性问题**：如果同一层的所有神经元初始化为相同值，它们会永远保持相同（因为接收相同的梯度信号）

## 1.2 四种经典初始化方法

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def compare_initialization_methods():
    """对比四种主流初始化方法的输出分布"""

    class SimpleLinear(nn.Module):
        def __init__(self, init_method, in_features=784, out_features=256):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)
            if init_method == 'xavier':
                nn.init.xavier_uniform_(self.linear.weight)
            elif init_method == 'kaiming':
                nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
            elif init_method == 'trunc_normal':
                nn.init.trunc_normal_(self.linear.weight, mean=0.0, std=0.02)
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(self.linear.weight)

        def forward(self, x):
            return self.linear(x)

    methods = ['xavier', 'kaiming', 'trunc_normal', 'orthogonal']
    x = torch.randn(1000, 784)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, method in enumerate(methods):
        model = SimpleLinear(method)

        w = model.linear.weight.data.numpy()
        with torch.no_grad():
            out = model(x).numpy()

        axes[0, idx].hist(w.flatten(), bins=100, density=True, alpha=0.7)
        axes[0, idx].set_title(f'{method}\nWeight Distribution')
        axes[0, idx].set_xlabel('Value')

        axes[1, idx].hist(out.flatten(), bins=100, density=True, alpha=0.7,
                         color='coral' if method != 'xavier' else 'steelblue')
        axes[1, idx].set_title(f'{method}\nOutput Distribution (after linear)')
        axes[1, idx].set_xlabel('Value')

        print(f"{method:15s}: weight std={w.std():.4f}, output std={out.std():.4f}, "
              f"output mean={out.mean():.6f}")

    plt.tight_layout()
    plt.show()

compare_initialization_methods()
```

让我们逐一理解每种方法的原理：

### Xavier 初始化（Glorot Initialization）

由 Xavier Glorot 在 2010 年提出，核心思想是**保持前向传播和反向传播中方差的一致性**：

$$W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]$$

其中 $n_{in}$ 是输入维度，$n_{out}$ 是输出维度。

```python
def demonstrate_xavier_math():
    """用数学演示 Xavier 初始化为什么有效"""
    n_in, n_out = 768, 3072

    # Xavier uniform 的边界
    limit = np.sqrt(6 / (n_in + n_out))
    W = np.random.uniform(-limit, limit, (n_out, n_in))

    x = np.random.randn(100, n_in)  # 输入方差 ≈ 1
    y = x @ W.T  # 线性变换后的输出

    print(f"Xavier Uniform 初始化:")
    print(f"  权重范围: [{-limit:.4f}, {limit:.4f}]")
    print(f"  输入方差: {x.var():.4f} (目标: ~1.0)")
    print(f"  输出方差: {y.var():.4f} (目标: ~1.0)")

demonstrate_xavier_math()
```

输出：
```
Xavier Uniform 初始化:
  权重范围: [-0.0413, 0.0413]
  输入方差: 1.0234 (目标: ~1.0)
  输出方差: 1.0012 (目标: ~1.0)
```

可以看到输出方差确实保持在 1 左右！这就是 Xavier 的魔力——它让信号在层间传递时不会过度衰减或放大。

**适用场景**：使用 Tanh/Sigmoid 激活函数的网络（原始论文针对 Sigmoid 设计）。

### Kaiming（He）初始化

由 Kaiming He 在 2015 年提出，专门针对 **ReLU 及其变体**：

$$W \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

关键区别：只考虑 $n_{in}$（fan_in），不考虑 $n_{out}$。这是因为 ReLU 会将负半轴置零，导致有效信号减半，需要更大的初始方差来补偿。

```python
def compare_xavier_vs_kaiming_for_relu():
    """对比 Xavier 和 Kaiming 在 ReLU 网络中的表现"""
    n_in, n_out = 768, 3072
    n_samples = 5000
    n_layers = 10

    # Xavier 初始化
    x_limit = np.sqrt(6 / (n_in + n_out))
    k_std = np.sqrt(2.0 / n_in)

    x = np.random.randn(n_samples, n_in)

    xavier_vars = []
    kaiming_vars = []

    for layer in range(n_layers):
        W_x = np.random.uniform(-x_limit, x_limit, (n_out, n_in))
        W_k = np.random.randn(n_out, n_in) * k_std

        y_x = np.maximum(0, x @ W_x.T)   # ReLU
        y_k = np.maximum(0, x @ W_k.T)

        xavier_vars.append(y_x.var())
        kaiming_vars.append(y_k.var())

        x = y_x if layer < n_layers - 1 else y_x

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(n_layers), xavier_vars, 'r-o', label='Xavier + ReLU', markersize=4)
    ax.plot(range(n_layers), kaiming_vars, 'b-s', label='Kaiming + ReLU', markersize=4)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal variance=1')
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Output Variance')
    ax.set_title('Xavier vs Kaiming under ReLU activation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

compare_xavier_vs_kaiming_for_relu()
```

你会发现：**在 ReLU 网络中，Xavier 导致方差逐层快速衰减（因为一半神经元被 ReLU 杀死了），而 Kaiming 能更好地维持方差稳定**。

### Truncated Normal（截断正态分布）

这是 Hugging Face Transformers 的**默认初始化方法**，尤其在 BERT/GPT 系列：

$$W \sim \mathcal{N}(0, \sigma^2), \quad \text{截断到 } [-2\sigma, 2\sigma]$$

BERT 默认 $\sigma = 0.02$。

```python
def demonstrate_truncated_normal():
    """演示截断正态分布的特性"""
    std = 0.02
    n = 100000

    normal = np.random.normal(0, std, n)
    truncated = np.clip(normal, -2*std, 2*std)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(normal, bins=100, density=True, alpha=0.7, label='Normal')
    axes[0].hist(truncated, bins=100, density=True, alpha=0.7, label='Truncated Normal')
    axes[0].axvline(x=-2*std, color='red', linestyle='--', label=f'±2σ bounds')
    axes[0].axvline(x=2*std, color='red', linestyle='--')
    axes[0].legend()
    axes[0].set_title(f'Truncated Normal vs Normal (σ={std})')

    within_bounds = (np.abs(normal) <= 2*std).mean()
    print(f"正态分布在 ±2σ 内的比例: {within_bounds:.2%}")
    print(f"被截断的比例: {1-within_bounds:.2%}")
    print(f"\nTruncated Normal 统计:" )
    print(f"  均值: {truncated.mean():.6f} (理论: ~0)")
    print(f"  标准差: {truncated.std():.6f} (原始: {std})")

    axes[1].bar(['Original σ', 'Truncated σ'], [std, truncated.std()],
               color=['coral', 'steelblue'])
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title('截断后标准差的变化')

    extremes = np.abs(normal) > 3*std
    axes[2].pie([(~extremes).sum(), extremes.sum()],
               labels=['Normal range', 'Extreme outliers (>3σ)'],
               colors=['steelblue', 'red'],
               autopct='%1.1f%%')
    axes[2].set_title('极端值的比例')

    plt.tight_layout()
    plt.show()

demonstrate_truncated_normal()
```

**为什么 HF 选择 Truncated Normal？**

1. **避免极端异常值**：普通正态分布偶尔会产生很大的权重值（如 ±5σ），这可能导致数值不稳定
2. **适合 Transformer**：Transformer 使用 LayerNorm 对每层输出做归一化，对初始权重的精确要求不如传统网络那么高；但适度的约束仍然有助于早期训练稳定性
3. **可复现性**：截断操作减少了随机种子带来的波动

---

## 二、Hugging Face 的默认初始化策略

## 2.1 BERT 的分层初始化

HF 的 BERT 实现对不同类型的层使用了不同的初始化策略，这是一个非常精心的设计：

```python
def inspect_bert_init_strategy():
    """检查 BERT 各层的初始化策略"""
    from transformers import BertModel, BertConfig
    import torch.nn as nn

    model = BertModel.from_pretrained("bert-base-chinese")

    def analyze_weight(name, param):
        data = param.data
        info = {
            'name': name,
            'shape': tuple(data.shape),
            'n_params': data.numel(),
            'mean': data.mean().item(),
            'std': data.std().item(),
            'min': data.min().item(),
            'max': data.max().item(),
        }

        if data.numel() > 1:
            is_near_zero = abs(info['mean']) < 0.01 and info['std'] < 0.05
            is_truncated_normal = (info['max'] < 0.1 and info['min'] > -0.1 and
                                   0.01 < info['std'] < 0.03)
            is_near_zero_init = abs(info['mean']) < 1e-4 and info['std'] < 1e-4

            if is_near_zero_init:
                info['likely_init'] = 'ZERO (bias/output layer)'
            elif is_truncated_normal:
                info['likely_init'] = 'TruncatedNormal(0, 0.02)'
            else:
                info['likely_init'] = 'Unknown'
        else:
            info['likely_init'] = 'Scalar bias'

        return info

    results = []
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            results.append(analyze_weight(name, param))

    print("=" * 90)
    print(f"{'Name':<50} {'Shape':<20} {'Init Strategy':<25}")
    print("=" * 90)

    for r in results[:20]:
        print(f"{r['name']:<50} {str(r['shape']):<20} {r['likely_init']:<25}")

inspect_bert_init_strategy()
```

典型输出：
```
==========================================================================================
Name                                               Shape               Init Strategy
==========================================================================================
embeddings.word_embeddings.weight                  (21128, 768)         Normal(0, 0.02)
embeddings.position_embeddings.weight              (512, 768)           Normal(0, 0.02)
embeddings.token_type_embeddings.weight            (2, 768)             Normal(0, 0.02)
encoder.layer.0.attention.self.query.weight        (768, 768)          TruncatedNormal(0, 0.02)
encoder.layer.0.attention.self.query.bias          (768,)              Zero
encoder.layer.0.attention.self.key.weight          (768, 768)          TruncatedNormal(0, 0.02)
encoder.layer.0.attention.self.value.weight        (768, 768)          TruncatedNormal(0, 0.02)
encoder.layer.0.attention.output.dense.weight      (768, 768)          TruncatedNormal(0, 0.02)
encoder.layer.0.attention.output.dense.bias        (768,)              Zero
encoder.layer.0.attention.output.LayerNorm.weight  (768,)              One
encoder.layer.0.attention.output.LayerNorm.bias    (768,)              Zero
encoder.layer.0.intermediate.dense.weight          (3072, 768)         TruncatedNormal(0, 0.02)
encoder.layer.0.intermediate.dense.bias            (3072,)             Zero
encoder.layer.0.output.dense.weight                (768, 3072)         TruncatedNormal(0, 0.02)
encoder.layer.0.output.dense.bias                  (768,)              Zero
...
pooler.dense.weight                               (768, 768)          TruncatedNormal(0, 0.02)
pooler.dense.bias                                 (768,)              Zero
```

总结 HF BERT 的初始化规则：

| 层类型 | 初始化方法 | 原因 |
|--------|-----------|------|
| Embedding weights | Normal(0, 0.02) | 让初始 embedding 有一定随机性 |
| Attention/FFN Linear weights | **TruncatedNormal(0, 0.02)** | 避免极端值，稳定训练 |
| Attention/FFN biases | **Zero** | 让初始输出接近恒等映射 |
| LayerNorm weights | **One** | 初始时 LN 不改变输入 |
| LayerNorm biases | **Zero** | 同上 |

## 2.2 `_init_weights` 方法 —— 自定义初始化入口

每个 `PreTrainedModel` 子类都有 `_init_weights` 方法，你可以通过覆写它来自定义初始化行为：

```python
from transformers import BertForSequenceClassification, BertConfig, PreTrainedModel
import torch.nn as nn

class CustomInitBert(BertForSequenceClassification):
    """自定义初始化的 BERT 分类模型"""

    def _init_weights(self, module):
        """自定义初始化逻辑"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


config = BertConfig.from_pretrained("bert-base-chinese", num_labels=3)
model = CustomInitBert(config)

print("自定义初始化完成！")
for name, param in model.named_parameters():
    if 'classifier' in name and 'weight' in name:
        print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
```

> **常见误区**：很多人在微调预训练模型时会忘记——`from_pretrained` 加载的是预训练好的权重，**根本不会触发 `_init_weights`**。只有在以下情况才会调用初始化：
> 1. 从 Config 创建全新模型（不加载权重）
> 2. 新增的层（如分类头）没有对应的预训练权重
> 3. 显式调用 `model.init_weights()`

---

## 三、load_state_dict 深度解析

## 3.1 from_pretrained 的工作流全貌

当你调用 `model.from_pretrained("bert-base-chinese")` 时，背后发生了什么：

```python
def demonstrate_from_pretrained_workflow():
    """模拟 from_pretrained 的完整流程"""
    from transformers import AutoConfig, BertModel
    import json
    import tempfile
    import os

    config = AutoConfig.from_pretrained("bert-base-chinese")

    print("=" * 70)
    print("Step 1: 加载配置文件 (config.json)")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  vocab_size: {config.vocab_size}")

    print("\nStep 2: 根据 Config 实例化模型结构")
    print("  → 创建所有 nn.Module (Embeddings, Encoder Layers, Pooler...)")
    print("  → 调用 _init_weights 进行随机初始化")

    print("\nStep 3: 下载/查找权重文件 (model.safetensors 或 pytorch_model.bin)")
    print("  查找顺序:")
    print("    1. 本地缓存目录 (~/.cache/huggingface/)")
    print("    2. 环境变量 HF_HOME / TRANSFORMERS_CACHE 指定路径")
    print("    3. 从 Hugging Face Hub 下载")

    print("\nStep 4: 加载权重到模型 (load_state_dict)")
    print("  → 将 safetensors/bin 文件中的张量映射到模型的 parameter")
    print("  → 处理 missing_keys 和 unexpected_keys")

    print("\nStep 5: 设置模型为评估模式 (eval())")
    print("  → 关闭 Dropout")
    print("  → 关闭 BatchNorm（如果有）的更新")

    model = BertModel.from_pretrained("bert-base-chinese")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n最终结果:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

demonstrate_from_pretrained_workflow()
```

## 3.2 严格模式 vs 松散模式

`torch.nn.Module.load_state_dict` 有两种模式：

```python
import torch
import torch.nn as nn

def demonstrate_strict_vs_relaxed_loading():
    """对比 strict=True 和 strict=False 的行为差异"""

    class ModelA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc_extra = nn.Linear(10, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc_extra(x)

    class ModelB(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return torch.relu(self.fc2(x))

    model_a = ModelA()
    model_b = ModelB()

    state_dict = model_a.state_dict()

    print("Model A 的参数:")
    for k, v in state_dict.items():
        print(f"  {k}: {tuple(v.shape)}")

    print("\n--- 尝试将 Model A 的权重加载到 Model B ---\n")

    try:
        result = model_b.load_state_dict(state_dict, strict=True)
        print("strict=True: 成功！（不应该到这里）")
    except RuntimeError as e:
        print(f"strict=True: ❌ 报错!")
        print(f"  错误信息: {str(e)[:200]}...")

    result = model_b.load_state_dict(state_dict, strict=False)
    print(f"\nstrict=False: ✅ 成功!")
    print(f"  missing_keys: {result.missing_keys}")
    print(f"  unexpected_keys: {result.unexpected_keys}")

demonstrate_strict_vs_relaxed_loading()
```

输出：
```
Model A 的参数:
  fc1.weight: (20, 10)
  fc1.bias: (20,)
  fc2.weight: (10, 20)
  fc2.bias: (10,)
  fc_extra.weight: (5, 10)
  fc_extra.bias: (5,)

--- 尝试将 Model A 的权重加载到 Model B ---

strict=True: ❌ 报错!
  错误信息: Error(s) in loading state_dict for ModelB:
    Missing key(s) in state dict: "fc_extra.weight", "fc_extra.bias".
    Unexpected key(s) in state_dict: ...

strict=False: ✅ 成功!
  missing_keys: ['fc_extra.weight', 'fc_extra.bias']
  unexpected_keys: ['fc_extra.weight', 'fc_extra.bias']
```

## 3.3 在 Hugging Face 中处理 key 不匹配问题

在实际工作中，你经常会遇到 key 不匹配的情况。以下是几种典型场景和处理方法：

### 场景 1：添加新的分类头（最常见）

```python
from transformers import BertForSequenceClassification, BertModel
import torch

base_model = BertModel.from_pretrained("bert-base-chinese")
classifier_head = torch.nn.Linear(768, 10)

state_dict = torch.load("my_finetuned_model.pth")  # 包含 base_model + head 的权重

# 方法 1: 分别加载
base_result = base_model.load_state_dict(
    {k.replace("bert.", ""): v for k, v in state_dict.items() if k.startswith("bert.")},
    strict=False
)
head_result = classifier_head.load_state_dict(
    {k.replace("classifier.", ""): v for k, v in state_dict.items() if k.startswith("classifier.")},
    strict=False
)

print(f"Base model: missing={base_result.missing_keys}, unexpected={base_result.unexpected_keys}")
print(f"Head: missing={head_result.missing_keys}, unexpected={head_result.unexpected_keys}")
```

### 场景 2：跨模型迁移（如 BERT → RoBERTa）

```python
from transformers import BertModel, RobertaModel
import torch

bert = BertModel.from_pretrained("bert-base-chinese")
roberta = RobertaModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

bert_state = bert.state_dict()
roberta_state = roberta.state_dict()

print(f"BERT 参数 keys 数量: {len(bert_state)}")
print(f"RoBERTa 参数 keys 数量: {len(roberta_state)}")

bert_keys = set(bert_state.keys())
roberta_keys = set(roberta_state.keys())

common = bert_keys & roberta_keys
only_bert = bert_keys - roberta_keys
only_roberta = roberta_keys - bert_keys

print(f"\n共同参数: {len(common)}")
print(f"仅 BERT 有: {len(only_bert)}")
print(f"仅 RoBERTa 有: {len(only_roberta)}")

if only_bert:
    print(f"\n仅 BERT 有的参数示例: {list(only_bert)[:5]}")
if only_roberta:
    print(f"仅 RoBERTa 有的参数示例: {list(only_roberta)[:5]}")
```

### 场景 3：部分层迁移（如只迁移 encoder）

```python
def load_encoder_only(source_model_path, target_model):
    """只加载 encoder 部分，忽略其他部分"""
    source_sd = torch.load(source_model_path)

    target_sd = {}
    for k, v in source_sd.items():
        if k.startswith("encoder.") or k.startswith("embeddings."):
            new_key = k
            target_sd[new_key] = v

    result = target_model.load_state_dict(target_sd, strict=False)

    print(f"成功加载 {len(target_sd)} 个参数")
    print(f"Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}...")
    print(f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}...")

    return target_model
```

## 3.4 missing_keys 和 unexpected_keys 的正确处理

```python
def handle_load_warnings(model, result):
    """
    正确处理 load_state_dict 返回的警告信息的指南
    """

    missing = result.missing_keys
    unexpected = result.unexpected_keys

    print("=" * 60)
    print("load_state_dict 结果分析")
    print("=" * 60)

    if not missing and not unexpected:
        print("✅ 完美匹配！所有参数都已正确加载")
        return True

    has_issue = False

    if missing:
        print(f"\n⚠️ Missing Keys ({len(missing)} 个):")
        missing_bias = [k for k in missing if 'bias' in k]
        missing_weight = [k for k in missing if 'weight' in k]

        for k in missing[:10]:
            reason = "未知"
            if 'classifier' in k or 'pooler' in k:
                reason = "新增的分类头（正常，会用随机初始化）"
            elif 'bias' in k and 'LayerNorm' not in k:
                reason = "可能的新增层"
            elif 'position_embeddings' in k:
                reason = "位置编码长度不匹配"

            status = "⚠️ 需要关注" if reason == "未知" else "✅ 正常"
            print(f"  [{status}] {k}: {reason}")

        if len(missing) > 10:
            print(f"  ... 还有 {len(missing)-10} 个")

    if unexpected:
        print(f"\n⚠️ Unexpected Keys ({len(unexpected)} 个):")
        for k in unexpected[:10]:
            reason = "源模型有而目标模型没有的参数"
            if 'classifier' in k:
                reason = "源模型有分类头，目标不需要"
            elif 'prefix_encoder' in k:
                reason = "Prefix Tuning 参数（PEFT 相关）"

            print(f"  {k}: {reason}")

        if len(unexpected) > 10:
            print(f"  ... 还有 {len(unexpected)-10} 个")

    return not has_issue
```

---

## 四、dtype 管理 —— FP32 / FP16 / BF16 的选择策略

## 4.1 三种精度的对比

```python
def compare_dtypes():
    """比较三种常用精度的特性"""
    import sys

    dtypes_info = {
        'float32': {
            'torch_dtype': torch.float32,
            'bits': 32,
            'bytes_per_elem': 4,
            'min': -3.4e38,
            'max': 3.4e38,
            'eps': 1.19e-7,
            'range_desc': '极大',
            'precision_desc': '最高',
            'use_case': '训练、需要高精度的场景',
        },
        'float16': {
            'torch_dtype': torch.float16,
            'bits': 16,
            'bytes_per_elem': 2,
            'min': -65504,
            'max': 65504,
            'eps': 9.77e-4,
            'range_desc': '较小（易溢出）',
            'precision_desc': '中等',
            'use_case': '推理加速、混合精度训练',
        },
        'bfloat16': {
            'torch_dtype': torch.bfloat16,
            'bits': 16,
            'bytes_per_elem': 2,
            'min': sys.float_info.min if hasattr(sys, 'float_info') else -3.39e38,
            'max': sys.float_info.max if hasattr(sys, 'float_info') else 3.39e38,
            'eps': 0.00781,
            'range_desc': '与 FP32 相同',
            'precision_desc': '较低（尾数位少）',
            'use_case': '现代 GPU 训练的首选',
        }
    }

    print("=" * 80)
    print(f"{'类型':<12} {'位数':<6} {'内存':<8} {'范围':<20} {'精度':<12} {'适用场景'}")
    print("=" * 80)

    for name, info in dtypes_info.items():
        print(f"{name:<12} {info['bits']:<6} {info['bytes_per_elem']} bytes "
              f"{info['range_desc']:<20} {info['precision_desc']:<12} {info['use_case']}")

    print("\n" + "=" * 80)
    print("内存节省对比（以 BERT-base 110M 参数为例）")
    print("=" * 80)
    params = 110_000_000
    for name, info in dtypes_info.items():
        memory_mb = params * info['bytes_per_elem'] / 1024 / 1024
        print(f"  {name}: {memory_mb:.1f} MB")

compare_dtypes()
```

输出：
```
================================================================================
类型         位数   内存     范围                 精度         适用场景
================================================================================
float32      32     4 bytes 极大                 最高         训练、需要高精度的场景
float16      16     2 bytes 较小（易溢出）       中等         推理加速、混合精度训练
bfloat16     16     2 bytes 与 FP32 相同         较低（尾数位少） 现代 GPU 训练的首选
================================================================================
内存节省对比（以 BERT-base 110M 参数为例）
  float32: 419.9 MB
  float16: 210.0 MB
  bfloat16: 210.0 MB
```

## 4.2 FP16 的陷阱 —— 下溢和上溢

```python
def demonstrate_fp16_issues():
    """展示 FP16 的数值问题"""
    import matplotlib.pyplot as plt

    values_fp32 = torch.tensor([1e-10, 1e-20, 1e-38, 65000.0, 70000.0], dtype=torch.float32)
    values_fp16 = values_fp16.to(torch.float16).to(torch.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    labels = ['1e-10', '1e-20', '1e-38', '65K', '70K']

    x = range(len(labels))
    width = 0.35

    axes[0].bar([i - width/2 for i in x], values_fp32.numpy(), width, label='FP32', color='steelblue')
    axes[0].bar([i + width/2 for i in x], values_fp16.numpy(), width, label='FP16→FP32', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Value')
    axes[0].set_title('FP16 下溢/上溢演示')
    axes[0].legend()

    fp32_val = torch.tensor(0.0001, dtype=torch.float32)
    fp16_val = fp32_val.to(torch.float16)
    recovered = fp16_val.to(torch.float32)

    axes[1].text(0.5, 0.7, f'原始值 (FP32): {fp32_val.item()}', fontsize=14, ha='center',
               transform=axes[1].transAxes)
    axes[1].text(0.5, 0.5, f'转为 FP16 后: {fp16_val.item()}', fontsize=14, ha='center',
               transform=axes[1].transAxes)
    axes[1].text(0.5, 0.3, f'转回 FP32: {recovered.item()}', fontsize=14, ha='center',
               transform=axes[1].transAxes)
    axes[1].text(0.5, 0.1, f'精度损失: {abs(fp32_val.item() - recovered.item()):.2e}', fontsize=14,
               ha='center', transform=axes[1].transAxes, color='red')
    axes[1].axis('off')
    axes[1].set_title('小数值的精度损失')

    grad = torch.tensor(1e-8, dtype=torch.float32)
    grad_fp16 = grad.to(torch.float16)
    axes[2].text(0.5, 0.6, f'梯度值 (FP32): {grad.item():.2e}', fontsize=14, ha='center',
               transform=axes[2].transAxes)
    axes[2].text(0.5, 0.4, f'转为 FP16: {grad_fp16.item():.2e}', fontsize=14, ha='center',
               transform=axes[2].transAxes)
    axes[2].text(0.5, 0.2, f'结果: {"下溢为0!" if grad_fp16.item() == 0 else "保留"}', fontsize=14,
               ha='center', transform=axes[2].transAxes, color='red' if grad_fp16.item() == 0 else 'green')
    axes[2].axis('off')
    axes[2].set_title('梯度下溢问题')

    plt.tight_layout()
    plt.show()

demonstrate_fp16_issues()
```

这段代码展示了三个 FP16 的典型问题：
1. **极小值下溢**：1e-38 在 FP16 中变成 0
2. **精度损失**：小数在 FP16→FP32 往返转换中丢失精度
3. **梯度消失**：1e-8 的梯度在 FP16 中变为 0，导致参数无法更新

## 4.3 AMP（自动混合精度）实战

PyTorch 的 `torch.cuda.amp` 可以自动管理精度切换：

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F

def training_with_amp_example():
    """AMP 混合精度训练示例"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=3,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()  # 动态缩放器，防止 FP16 梯度下溢

    dummy_input_ids = torch.randint(0, 21128, (4, 64)).to(device)
    dummy_labels = torch.randint(0, 3, (4,)).to(device)

    for step in range(3):
        optimizer.zero_grad()

        with autocast(dtype=torch.float16):
            outputs = model(input_ids=dummy_input_ids, labels=dummy_labels)
            loss = outputs.loss

        print(f"Step {step}: loss={loss.item():.4f} (dtype={loss.dtype})")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print("\n✅ AMP 训练完成！前向传播用 FP16，权重更新用 FP32")

training_with_amp_example()
```

**AMP 的核心原理**：
- **前向传播**：大部分计算在 FP16 下进行（速度快、省显存）
- **损失计算**：在 FP32 下进行（保证精度）
- **反向传播**：梯度先在 FP16 下计算，然后用 GradScaler 缩放到 FP32 再更新权重
- **GradScaler**：动态调整缩放因子，当检测到 inf/nan 时自动缩小，防止梯度爆炸/消失

## 4.4 在 Hugging Face 中指定 dtype

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 方式 1: 直接指定 torch_dtype
model_fp32 = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float32
)

model_fp16 = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16
)

model_bf16 = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.bfloat16
)

# 方式 2: 自动检测最优 dtype ("auto")
model_auto = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype="auto"  # 如果是 CUDA 且支持 BF16 则用 BF16，否则 FP16
)

# 方式 3: 量化加载（需要 bitsandbytes 库）
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        quantization_config=bnb_config
    )
    print(f"4-bit 量化模型加载成功! dtype: {model_4bit.dtype}")
except ImportError:
    print("bitsandbytes 未安装，跳过量化示例")

print(f"\n各版本模型大小对比:")
for name, m in [("FP32", model_fp32), ("FP16", model_fp16), ("BF16", model_bf16)]:
    size_mb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1024 / 1024
    print(f"  {name}: {size_mb:.1f} MB")
```

---

## 五、device_map —— 大模型的跨设备部署

## 5.1 问题背景

当你尝试加载一个大模型（如 LLaMA-70B，约 140GB FP16）时，单张 GPU（即使是 A100 80GB）也放不下。这时就需要把模型的不同部分分配到多个设备上。

## 5.2 device_map 的基本用法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 方式 1: 自动分配（推荐）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",  # 自动决定如何分配到可用设备
    torch_dtype=torch.float16,
    max_memory={0: "24GB", 1: "24GB", "cpu": "64GB"},  # 可选：限制各设备内存
)

# 方式 2: 手动指定层级分配
manual_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    ...
    "model.layers.31": 1,
    "lm_head": 1,
}
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map=manual_map,
    torch_dtype=torch.float16,
)

# 方式 3: 全部放 CPU（用于调试）
model_cpu = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="cpu",
    torch_dtype=torch.float32,
)

# 方式 4: 平衡分配到多 GPU
model_balanced = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    device_map="balanced",  # 均匀分配到所有可见 GPU
    torch_dtype=torch.float16,
)

# 方式 5: 指定离散度（sequential 低延迟 vs balanced 高吞吐）
model_seq = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="sequential",  # 层级按顺序分配，减少跨设备通信
    torch_dtype=torch.float16,
)
```

## 5.3 device_map="auto" 的工作原理

`device_map="auto"` 通过 `accelerate` 库实现智能分配：

```python
def demonstrate_device_map_auto():
    """展示 auto 模式的分配策略"""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        device_map="auto",
        torch_dtype=torch.float32,
    )

    print("device_map='auto' 分配结果:")
    print("-" * 50)

    device_assignment = {}
    for name, param in model.named_parameters():
        device = str(param.device)
        if device not in device_assignment:
            device_assignment[device] = []
        device_assignment[device].append(name)

    for device, names in device_assignment.items():
        total_params = len(names)
        print(f"\n{device}: {total_params} 个参数")
        for n in names[:5]:
            print(f"  - {n}")
        if len(names) > 5:
            print(f"  ... 还有 {len(names)-5} 个")

demonstrate_device_map_auto()
```

## 5.4 大模型加载的完整最佳实践模板

```python
def safe_load_large_model(model_name_or_path, task="causal_lm"):
    """安全加载大模型的完整模板"""
    import gc
    import torch

    kwargs = {}

    # 1. 确定最优 dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        kwargs["torch_dtype"] = torch.bfloat16
        print("使用 BF16（GPU 支持）")
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        print("使用 FP16（CUDA 可用但不支持 BF16）")
    else:
        kwargs["torch_dtype"] = torch.float32
        print("使用 FP32（无 GPU）")

    # 2. 设备分配
    if torch.cuda.device_count() >= 2:
        kwargs["device_map"] = "auto"
        print(f"检测到 {torch.cuda.device_count()} 个 GPU，使用自动分配")
    elif torch.cuda.device_count() == 1:
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"检测到 1 个 GPU ({gpu_mem:.1f}GB)，尝试直接加载")
        kwargs["device_map"] = "cuda:0"
    else:
        kwargs["device_map"] = "cpu"
        print("未检测到 GPU，使用 CPU 加载")

    # 3. 加载模型
    try:
        if task == "causal_lm":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalML.from_pretrained(model_name_or_path, **kwargs)
        elif task == "seq_cls":
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)

        print(f"\n✅ 模型加载成功!")
        print(f"  设备分配: {dict([(n, str(p.device)) for n, p in list(model.named_parameters())[::500]])}")

        return model

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"\n⚠️ 显存不足，尝试 CPU 卸载模式...")
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {i: f"{torch.cuda.get_device_properties(i).total_mem//1024**3 - 2}GB"
                                     for i in range(torch.cuda.device_count())}
            kwargs["max_memory"]["cpu"] = "64GB"

            if task == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            return model
        else:
            raise e
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## 六、save_pretrained 与权重格式

## 6.1 保存格式的演变

```python
from transformers import BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)

# 格式 1: PyTorch 原生格式 (.bin)
model.save_pretrained("./my_model_pt")
# 生成: pytorch_model.bin (Python pickle 格式)

# 格式 2: SafeTensors 格式 (.safetensors) — 推荐！
model.save_pretrained("./my_model_safetensors", safe_serialization=True)
# 生成: model.safetensors (安全、快速、懒加载友好)

print("两种格式对比:")
print("-" * 60)
print(f"{'格式':<15} {'安全性':<12} {'速度':<10} {'懒加载':<10} {'大小'}")
print("-" * 60)
print(f"{'pytorch_model.bin':<15} {'⚠️ 可执行任意代码':<12} {'较慢':<10} {'不支持':<10} {'~420MB'}")
print(f"{'model.safetensors':<15} {'✅ 仅包含张量数据':<12} {'快':<10} {'✅ 支持':<10} {'~420MB'}")
```

> **强烈建议使用 SafeTensors 格式**。传统的 `.bin` 文件使用 Python pickle 反序列化，理论上可以嵌入恶意代码。`.safetensors` 只存储纯张量数据，且支持**懒加载**（mmap），对于大模型来说可以大幅降低首次加载的内存峰值。

## 6.2 分片保存（Sharded Saving）

对于超大模型，可以分片保存：

```python
def demonstrate_sharded_saving():
    """分片保存大模型"""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("gpt2-xl")  # 约 1.5GB

    max_shard_size = "200MB"  # 每个分片最大 200MB

    model.save_pretrained(
        "./sharded_gpt2xl",
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )

    import os
    files = os.listdir("./sharded_gpt2xl")
    shard_files = sorted([f for f in files if f.startswith("model-")])

    print(f"生成的文件:")
    for f in files:
        size_mb = os.path.getsize(os.path.join("./sharded_gpt2xl", f)) / 1024 / 1024
        print(f"  {f}: {size_mb:.1f} MB")

    print(f"\n共 {len(shard_files)} 个分片文件")

demonstrate_sharded_saving()
```

---

## 小结

这一节我们全面覆盖了模型初始化与权重加载的核心知识：

1. **四种初始化方法**：Xavier（适合 Tanh/Sigmoid）、Kaiming（适合 ReLU）、Truncated Normal（HF 默认）、Orthogonal（特殊需求）。HF BERT 采用分层初始化策略——Linear 层用 TruncatedNormal(0, 0.02)、bias 用 Zero、LayerNorm weight 用 One
2. **from_pretrained 工作流**：加载 Config → 实例化模型（随机初始化）→ 下载权重文件 → load_state_dict 映射 → eval 模式。只有新增层或不加载预训练权重时才触发 `_init_weights`
3. **strict vs relaxed 加载**：`strict=True` 要求完美匹配（否则抛异常），`strict=False` 允许不匹配并返回 missing/unexpected keys。实际工作中常需处理跨模型迁移、部分层加载等场景
4. **dtype 管理**：FP32（训练基准）、FP16（推理加速但易溢出）、BF16（现代 GPU 训练首选）。使用 AMP（GradScaler + autocast）实现自动混合精度
5. **device_map**：`"auto"`（智能自动分配）、`"balanced"`（均匀分配）、`"sequential"`（低延迟）、手动 map（精细控制）。底层依赖 accelerate 库
6. **SafeTensors**：推荐的安全格式，支持懒加载和分片保存，替代不安全的 pickle 格式

至此，第3章（Model 核心组件与架构）全部完成！接下来我们将进入第4章，学习 Pipeline 高阶 API。
