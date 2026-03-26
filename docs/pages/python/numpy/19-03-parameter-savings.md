# LoRA参数量节省

本篇文章详细计算 LoRA 带来的参数量节省，包括不同秩 r 下的压缩比、不同规模模型下的显存节省，以及 LoRA 与其他微调方法的对比。

## LoRA 参数量计算

### 基本公式

对于一个原始权重矩阵 W₀ ∈ ℝ^{d_out × d_in}：

- **原始参数量**：d_out × d_in
- **LoRA 参数量**：d_out × r + r × d_in = r × (d_out + d_in)
- **压缩比**：d_out × d_in / (r × (d_out + d_in))

```python
import numpy as np

def calculate_lora_params(d_out, d_in, r):
    """计算 LoRA 参数量

    参数:
        d_out: 输出维度
        d_in: 输入维度
        r: 秩
    返回:
        original_params, lora_params, compression_ratio
    """
    original_params = d_out * d_in
    lora_params = r * (d_out + d_in)
    compression_ratio = original_params / lora_params

    return original_params, lora_params, compression_ratio

print("=== LoRA 参数量计算 ===\n")

# 示例：典型 Transformer 注意力层
d_out, d_in = 768, 768  # Self-attention 中 QKV 投影

for r in [1, 2, 4, 8, 16, 32, 64]:
    orig, lora, ratio = calculate_lora_params(d_out, d_in, r)
    print(f"r={r:3d}: 原始={orig:>10,} | LoRA={lora:>8,} | 压缩比={ratio:>5.1f}x | 节省={100*(1-1/ratio):.2f}%")
```

## 不同规模模型的参数量分析

### GPT-2 模型

```python
def analyze_gpt2_lora():
    """分析 GPT-2 使用 LoRA 后的参数量变化"""
    print("\n=== GPT-2 LoRA 参数量分析 ===\n")

    # GPT-2 架构参数（近似）
    config = {
        'vocab_size': 50257,
        'hidden_size': 768,
        'num_heads': 12,
        'num_layers': 12,
        'intermediate_size': 3072,  # FFN 中间层
    }

    # 计算各层参数量
    # Self-Attention: Q, K, V, O 四个投影
    attn_params = 4 * config['hidden_size'] ** 2

    # FFN: W1, W2
    ffn_params = 2 * config['hidden_size'] * config['intermediate_size']

    # Layer Norm
    ln_params = 2 * config['hidden_size']

    # 每层总参数
    per_layer_params = attn_params + ffn_params + 2 * ln_params

    # 整个模型参数（不含 embeddings）
    total_params = per_layer_params * config['num_layers']

    print(f"每层参数分布:")
    print(f"  Self-Attention QKV+O: {attn_params:,}")
    print(f"  FFN W1+W2: {ffn_params:,}")
    print(f"  Layer Norm: {ln_params:,}")
    print(f"  每层总计: {per_layer_params:,}\n")

    print(f"12层模型总计（不含 embeddings）: {total_params:,}")
    print(f"约为 {total_params/1e6:.1f}M 参数\n")

    # LoRA 应用于 QKV（3 个投影）
    print("只在 QKV 上应用 LoRA (r=8):")
    lora_qkv = 3 * config['hidden_size'] * 8 * 2  # 3 * d * r * 2 (A+B)
    print(f"  LoRA 参数量: {lora_qkv:,}")
    print(f"  压缩比: {attn_params*3 / lora_qkv:.1f}x\n")

    # LoRA 应用于 QKV + O
    print("在 QKV + O 上应用 LoRA (r=8):")
    lora_all = 4 * config['hidden_size'] * 8 * 2
    print(f"  LoRA 参数量: {lora_all:,}")
    print(f"  压缩比: {attn_params*4 / lora_all:.1f}x")

analyze_gpt2_lora()
```

### LLaMA 模型

```python
def analyze_llama_lora():
    """分析 LLaMA 使用 LoRA 后的参数量变化"""
    print("\n=== LLaMA 7B LoRA 参数量分析 ===\n")

    # LLaMA 7B 配置（近似）
    config = {
        'hidden_size': 4096,
        'num_heads': 32,
        'num_layers': 32,
        'intermediate_size': 11008,  # FFN
    }

    # Self-Attention 参数量
    attn_params = 4 * config['hidden_size'] ** 2

    # 每层总参数
    per_layer_params = attn_params + 2 * config['hidden_size'] * config['intermediate_size']

    print(f"LLaMA 7B 每层参数: {per_layer_params:,} ≈ {per_layer_params/1e6:.1f}M")
    print(f"模型总计: {per_layer_params * config['num_layers'] / 1e9:.2f}B\n")

    # LoRA r=8
    r = 8
    lora_params = 4 * config['hidden_size'] * r * 2  # QKV + O
    compression = attn_params * 4 / lora_params

    print(f"只在 QKV+O 上应用 LoRA (r={r}):")
    print(f"  LoRA 参数量: {lora_params:,} ≈ {lora_params/1e6:.2f}M")
    print(f"  压缩比: {compression:.1f}x")

analyze_llama_lora()
```

## 显存节省分析

训练时的显存占用不仅包括参数，还包括优化器状态、梯度和激活值。

```python
def analyze_memory_savings():
    """分析 LoRA 的显存节省"""
    print("\n=== LoRA 显存节省分析 ===\n")

    # 假设使用 AdamW 优化器
    # 优化器状态：每个参数需要 2 个状态（m 和 v）

    d_model = 4096
    d_ff = 11008
    d_vocab = 50000
    seq_len = 512
    batch_size = 8

    # 全量微调
    # 参数
    full_params = 4 * d_model ** 2 + 2 * d_model * d_ff + 2 * d_model * d_vocab
    print(f"全量微调参数量: {full_params/1e6:.1f}M")

    # 梯度（与参数量相同）
    grad_size = full_params * 4  # float32 = 4 bytes
    print(f"梯度显存: {grad_size/1024/1024:.1f} MB")

    # 优化器状态
    opt_size = full_params * 2 * 4  # Adam 需要 2 个状态
    print(f"优化器状态: {opt_size/1024/1024:.1f} MB")

    # 激活值（粗略估计）
    activation_size = batch_size * seq_len * d_model * 32 * 4  # 假设 32 层
    print(f"激活值（估计）: {activation_size/1024/1024:.1f} MB\n")

    # LoRA 微调
    r = 8
    lora_params = 4 * d_model * r * 2  # QKV + O

    print(f"LoRA 参数量 (r={r}): {lora_params/1e6:.2f}M")

    # 只优化 LoRA 参数
    lora_grad_size = lora_params * 4
    print(f"LoRA 梯度显存: {lora_grad_size/1024/1024:.1f} MB")

    lora_opt_size = lora_params * 2 * 4
    print(f"LoRA 优化器状态: {lora_opt_size/1024/1024:.1f} MB\n")

    # 节省
    print("显存节省（仅优化器相关）:")
    saved_opt = opt_size - lora_opt_size
    saved_grad = grad_size - lora_grad_size
    print(f"  梯度: {saved_grad/1024/1024:.1f} MB")
    print(f"  优化器: {saved_opt/1024/1024:.1f} MB")
    print(f"  总计: {(saved_grad + saved_opt)/1024/1024:.1f} MB")

analyze_memory_savings()
```

## 不同微调方法对比

```python
def compare_finetune_methods():
    """对比不同微调方法的参数量"""
    print("\n=== 微调方法对比 ===\n")

    d_model = 768
    d_ff = 3072
    d_vocab = 50257
    num_layers = 12

    # 1. 全量微调
    full_params = num_layers * (4 * d_model**2 + 2 * d_model * d_ff) + d_model * d_vocab
    print(f"1. 全量微调: {full_params/1e6:.1f}M 参数")

    # 2. LoRA (r=8)
    lora_r8 = num_layers * (3 * d_model * 8 * 2)  # QKV
    print(f"2. LoRA (r=8, QKV): {lora_r8/1e6:.2f}M 参数 ({full_params/lora_r8:.0f}x 压缩)")

    # 3. LoRA (r=8, QKV+O)
    lora_r8_all = num_layers * (4 * d_model * 8 * 2)
    print(f"3. LoRA (r=8, QKV+O): {lora_r8_all/1e6:.2f}M 参数 ({full_params/lora_r8_all:.0f}x 压缩)")

    # 4. Adapter (bottleneck=64)
    adapter_bottle = 64
    adapter = num_layers * (2 * d_model * adapter_bottle)
    print(f"4. Adapter (bottleneck={adapter_bottle}): {adapter/1e6:.1f}M 参数 ({full_params/adapter:.0f}x 压缩)")

    # 5. Prompt Tuning (prefix length=20)
    prefix_len = 20
    prompt = num_layers * 2 * prefix_len * d_model
    print(f"5. Prompt Tuning (prefix={prefix_len}): {prompt/1e6:.2f}M 参数 ({full_params/prompt:.0f}x 压缩)")

    # 6. BitFit (只微调偏置)
    bitfit = num_layers * (4 * d_model + 2 * d_ff)  # 近似
    print(f"6. BitFit: {bitfit/1e6:.2f}M 参数 ({full_params/bitfit:.0f}x 压缩)")

compare_finetune_methods()
```

## 实际计算工具函数

```python
def calculate_trainable_params(d_model, num_layers, lora_targets=['q', 'k', 'v', 'o'], r=8):
    """计算 LoRA 可训练参数量

    参数:
        d_model: 模型维度
        num_layers: 层数
        lora_targets: 应用 LoRA 的目标层
        r: 秩
    返回:
        可训练参数量
    """
    targets = {
        'q': d_model * d_model,
        'k': d_model * d_model,
        'v': d_model * d_model,
        'o': d_model * d_model,
    }

    params_per_layer = sum(targets.get(t, 0) for t in lora_targets)
    total_params = num_layers * params_per_layer * 2  # A + B

    return total_params

def print_lora_summary():
    """打印 LoRA 参数量总结"""
    print("\n=== LoRA 参数量速查表 ===\n")

    configs = [
        ("GPT-2 Small", 768, 12),
        ("GPT-2 Medium", 1024, 24),
        ("GPT-2 Large", 1280, 36),
        ("LLaMA 7B", 4096, 32),
    ]

    print(f"{'模型':<15} {'层':<5} {'d_model':<10} {'r=4':<12} {'r=8':<12} {'r=16':<12}")
    print("-" * 70)

    for name, d_model, num_layers in configs:
        p4 = calculate_trainable_params(d_model, num_layers, r=4)
        p8 = calculate_trainable_params(d_model, num_layers, r=8)
        p16 = calculate_trainable_params(d_model, num_layers, r=16)

        print(f"{name:<15} {num_layers:<5} {d_model:<10} {p4/1e6:<12.2f} {p8/1e6:<12.2f} {p16/1e6:<12.2f}")

print_lora_summary()
```

## 常见问题

### Q: 为什么 r 不能太大？

A: r 越大，LoRA 参数量越大，接近全量微调，失去 LoRA 的优势。同时，r 太大可能导致过拟合，因为 LoRA 的表达能力与 r 成正比。

```python
def explain_rank_choice():
    """解释秩的选择"""
    print("\n=== 秩 r 的选择 ===\n")

    print("r 太小 (r=1,2):")
    print("  - 参数量极少，但表达能力有限")
    print("  - 可能欠拟合")
    print("  - 适合简单任务或小数据集\n")

    print("r 中等 (r=4,8,16):")
    print("  - 平衡参数量和表达能力")
    print("  - 适合大多数任务")
    print("  - LoRA 论文推荐的默认值是 r=8\n")

    print("r 较大 (r=32,64,128):")
    print("  - 表达能力更强")
    print("  - 接近全量微调")
    print("  - 适合复杂任务或大数据集\n")

    print("实践建议:")
    print("  - 从 r=8 或 r=16 开始")
    print("  - 根据任务复杂度调整")
    print("  - 监控验证集性能")

explain_rank_choice()
```

理解 LoRA 的参数量节省对于评估和选择微调策略非常重要。在实际应用中，需要在参数量、模型性能和训练效率之间做出权衡。
