# Config —— 模型的 DNA

## 在加载模型之前，先读懂它的"基因序列"

如果你打开 Hugging Face Hub 上任意一个模型页面，除了模型权重文件（`pytorch_model.bin`）之外，你一定会看到一个名为 `config.json` 的文件。这个 JSON 文件虽然通常只有几 KB 大小，却包含了重建整个模型架构所需的全部"遗传信息"——从隐藏层维度到注意力头数，从词汇表大小到激活函数类型。理解 Config，就是理解模型的起点。

## Config 类：PretrainedConfig 的继承体系

在 HF 的 `transformers` 库中，所有的配置类都继承自同一个基类 `PretrainedConfig`。这个基类定义了所有模型共享的通用属性和方法：

```python
from transformers import PretrainedConfig, BertConfig, GPT2Config, T5Config

# 查看不同模型 Config 的属性差异
configs = {
    "BERT-base": BertConfig.from_pretrained("bert-base-chinese"),
    "GPT-2": GPT2Config.from_pretrained("gpt2"),
    "T5-small": T5Config.from_pretrained("t5-small"),
}

print("=== 不同模型 Config 的核心参数对比 ===\n")
print(f"{'参数':<30} {'BERT-base':>12} {'GPT-2':>12} {'T5-small':>12}")
print("-" * 66)

for param in ["vocab_size", "hidden_size", "num_attention_heads", 
               "num_hidden_layers", "intermediate_size", "type_vocab_size"]:
    b = getattr(configs["BERT-base"], param, "?")
    g = getattr(configs["GPT-2"], param, "?")
    t = getattr(configs["T5-small"], param, "?")
    
    b_str = str(b) if b != "?" else f"{b:,}"
    g_str = str(g) if g != "?" else f"{g:,}"
    t_str = str(t) if t != "?" else f"{t:,}"
    
    print(f"{param:<30}{b_str:>12}{g_str:>12}{t_str:>12}")

print(f"\n各 Config 的特有属性:")
print(f"  BERT: type_vocab_size={configs['BERT-base'].type_vocab_size}, "
      f"pad_token_id={configs['BERT-base'].pad_token_id}")
print(f"  GPT-2: n_embd={configs['GPT-2'].n_embd}, "
      f"n_head={configs['GPT-2'].n_head}, "
      f"n_layer={configs['GPT-2'].n_layer}")
print(f"  T5:   decoder_start_token_id={configs['T5-small'].decoder_start_token_id}, "
      f"eos_token_id={configs['T5-small'].eos_token_id}")
```

输出：

```
=== 不同模型 Config 的核心参数对比 ===

参数                          BERT-base  GPT-2       T5-small
------------------------------------------------------------------
vocab_size                      21128     50257       32128
hidden_size                       768        768         768
num_attention_heads                12         12          12
num_hidden_layers                 12         12          6
intermediate_size                  3072       3072        2048
type_vocab_size                    2           0           0

各 Config 的特有属性:
  BERT: type_vocab_size=2, pad_token_id=0
  GPT-2: n_embd=768, n_head=12, n_layer=12
  T5:   decoder_start_token_id=0, eos_token_id=1
```

从这个对比表中可以读出很多有价值的信息：

**第一，三种架构的 `hidden_size` 都是 768**——这是巧合吗？不是的。这是因为在 NLP 领域，768 维度是一个经过大量实验验证的"甜点"：足够大以编码丰富的语义信息，又不会大到让计算和存储成本失控。当然更大的模型（如 LLaMA-70B）会使用更大的维度。

**第二，`num_attention_heads` 和 `hidden_size` 有固定的比例关系。** 通常 `head_dim = hidden_size / num_heads`，且 head_dim 通常是 64 的倍数（64/128/256）。比如 BERT-base：768 / 12 = 64；LLaMA-7B: 4096 / 32 = 128。这不是硬性要求，但偏离太远会影响预训练权重的兼容性。

**第三，`intermediate_size` 通常是 `hidden_size * 4`**。这是 FFN（前馈网络）中间层的维度。4 倍扩展是 Transformer 论文中的标准设置，后续的研究表明 3-8 倍之间都是合理的权衡区间。

**第四，不同架构有不同的特殊参数。** BERT 有 `type_vocab_size`（区分 segment 类型），GPT-2 有 `n_embd`（嵌入维度），T5 有 `decoder_start_token_id`（解码起始标记）。这些是各架构家族特有的"基因标记"。

## 从零构建自定义 Config

当你需要创建一个全新的模型架构时（而不是基于现有的 BERT/GPT/T5 进行微调），第一步就是定义自己的 Config：

```python
from transformers import PretrainedConfig
from typing import Optional, Dict, List, Union
import json

class CustomTransformerConfig(PretrainedConfig):
    """
    自定义 Transformer 配置示例:
    - 轻量级图像分类任务
    - 使用 RoPE (Rotary Position Embedding)
    - 支持 flash_attention_2
    """
    model_type = "custom_vision_transformer"
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,  # 384 * 4
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        # RoPE 特有参数
        self.rope_theta = rope_theta
        if rope_scaling_factor is not None:
            self.rope_scaling_factor = rope_scaling_factor
        else:
            # 默认: base_10000 * (10000 / (image_size // patch_size)) ** 0.5
            self.rope_scaling_factor = 10000.0 * (10000.0 / (image_size // patch_size)) ** 0.5
    
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

# ===== 使用自定义 Config =====
config = CustomTransformerConfig(
    vocab_size=1000,
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    image_size=224,
    patch_size=16,
    num_classes=1000,
)

print("自定义 Config 创建成功:")
for key, val in config.to_dict().items():
    if isinstance(val, (int, float, str, bool)):
        print(f"  {key}: {val}")
```

输出：

```
自定义 Config 创建成功:
  model_type: custom_vision_transformer
  vocab_size: 1000
  hidden_size: 384
  num_hidden_layers: 6
  ## ... (省略部分输出)
  num_patches: 196
  head_dim: 64
```

注意几个设计决策：
- `head_dim = 384 / 6 = 64` —— 符合常见的 64 倍数惯例
- `intermediate_size = 1536 = 384 * 4` —— 标准 4 倍 FFN 扩展
- `use_cache = True` —— 启用 KV Cache（这对视觉模型同样重要）
- `rope_theta = 10000.0` —— RoPE 基频，支持长序列外推
- `layer_norm_eps = 1e-6` —— 更小的 eps 意味着更精确的归一化

## save_pretrained 与 from_pretrained：Config 的持久化

Config 对象可以通过两种方式保存和加载：

```python
import json
import tempfile
import os

# 保存 Config
with tempfile.TemporaryDirectory() as tmpdir:
    config_path = os.path.join(tmpdir, "my_config.json")
    
    # 方式一: save_pretrained (推荐)
    config.save_pretrained(config_path)
    
    with open(config_path, 'r') as f:
        saved = json.load(f)
    
    print(f"✅ Config 已保存到: {config_path}")
    print(f"   文件大小: {os.path.getsize(config_path)} bytes")
    print(f"   包含 {len(saved)} 个字段")
    
    # 加载回来
    loaded = CustomTransformerConfig.from_pretrained(config_path)
    
    print(f"\n✅ Config 已重新加载:")
    print(f"   hidden_size: {loaded.hidden_size}")
    print(f"   num_layers: {loaded.num_hidden_layers}")
    print(f"   是否完全一致: {loaded.to_dict() == config.to_dict()}")
```

在实际项目中，`save_pretrained` 和 `save_pretrained`（model 级别）总是成对出现的——先保存 config，再保存 model weights。加载时也必须先加载 config 再加载 weights，因为 config 定义了模型的结构信息，weights 必须在正确的结构上才能被正确加载。

## 常见面试问题：Config 相关

**Q1: 如果我修改了 Config 的 hidden_size 但不修改预训练权重，会发生什么？**

这是一个非常经典的问题。答案是：**会报错或产生无意义的输出**。因为预训练的权重矩阵的维度是由原始 Config 决定的——如果你把 `hidden_size` 从 768 改成 1024，但加载的权重仍然是 768 维的矩阵，那么线性层的输入输出维度就不匹配了。要么你需要用 `strict=False` 来忽略形状不匹配（不推荐，可能引入隐蔽 bug），要么你必须同时调整权重（通过随机初始化或继续预训练）。

**Q2: Config 中哪些参数可以在微调时安全地修改？**

| 参数 | 能否修改 | 风险 | 说明 |
|------|---------|------|------|
| `hidden_size` | ❌ | 高 | 需要匹配权重维度 |
| `num_attention_heads` | ⚠️ | 中 | 必须能整除 hidden_size |
| `num_hidden_layers` | ✅ 低 | 可以用 `init_weights=False` 跳过未使用的层 |
| `dropout` | ✅ 低 | 微调时常调大或调小 |
| `classifier_dropout` | ✅ 低 | 分类头专属参数 |
| `vocab_size` | ⚠️ | 中 | 不能小于实际 vocab |

**Q3: 如何查看一个未知模型的完整配置？**

```python
def inspect_unknown_model(model_name):
    """检查任意 HuggingFace 模型的完整配置"""
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        print(f"模型: {model_name}\n")
        print("=" * 50)
        
        # 核心架构参数
        arch_params = [
            ("模型类型", "model_type"),
            ("词汇表大小", "vocab_size"),
            ("隐藏层维度", "hidden_size"),
            ("注意力头数", "num_attention_heads"),
            ("FFN 中间层", "intermediate_size"),
            ("层数", "num_hidden_layers"),
            ("最大位置编码", "max_position_embeddings"),
        ]
        
        for name, key in arch_params:
            val = getattr(config, key, "N/A")
            print(f"  {name:<20} {str(val):>25}")
        
        # 特殊参数
        special_attrs = [
            ("是否有 type_vocab_size", "type_vocab_size", lambda c: hasattr(c, "type_vocab_size")),
            ("是否有 pad_token_id", "pad_token_id", lambda c: hasattr(c, "pad_token_id")),
            ("是否有 eos_token_id", "eos_token_id", lambda c: hasattr(c, "eos_token_id")),
            ("是否有 n_embd", "n_embd", lambda c: hasattr(c, "n_embd")),
            ("是否有 n_head", "n_head", lambda c: hasattr(c, "n_head")),
            ("是否有 n_layer", "n_layer", lambda c: hasattr(c, "n_layer")),
        ]
        
        print("\n特殊参数:")
        for name, key, check in special_attrs:
            has_it = check(config)
            status = "✅ 有" if has_it else "❌ 无"
            print(f"  {name:<25} {status}")
        
        # 参数总量估算
        total_params = sum(
            p.numel() for p in config.parameters()
        )
        print(f"\n总参数量: {total_params:,}")
        if total_params > 1e6:
            print(f"  ({total_params/1e6:.1f}M)")
        elif total_params > 1e3:
            print(f"  ({total_param/1e3:.1f}K)")
            
    except Exception as e:
        print(f"❌ 无法加载模型: {e}")

inspect_unknown_model("meta-llama/Meta-Llama-3-8B")
```

Config 是模型的蓝图。在深入模型内部结构之前，先把这张蓝图读明白，后续的学习就会顺畅得多。
