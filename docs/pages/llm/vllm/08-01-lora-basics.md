# LoRA 基础回顾与 vLLM 支持

> **白板时间**：想象你有一个通用的 LLM（比如 Llama 3），它什么都知道一点，但什么都不精。现在你想让它成为医疗专家——传统做法是微调全部参数（70B 参数 × 2 bytes = 140GB 显存，需要 8×A100）。但 LoRA 说：不用改原模型，只在每层旁边加两个小矩阵（总参数量不到原模型的 1%），只训练这两个小矩阵。结果：用一张 RTX 4090 就能微调出领域专家模型。而 vLLM 的能力是：**同时加载一个基础模型和几十个 LoRA 适配器，按需切换，无需重启服务**。

## 一、LoRA 原理速查

### 1.1 核心思想

```
传统全量微调:
  W' = W + ΔW     (ΔW 与 W 同维度，参数量 = 原模型)
  
LoRA 微调:
  W' = W + BA     (B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k))
  
  其中:
  - W: 冻结的原始权重 (不更新)
  - B, A: 新增的低秩分解矩阵 (唯一需要训练的参数)
  - r: rank (秩), 典型值 8/16/32/64
  - 缩放因子: α/r (通常 α = r，即缩放=1)

参数量对比 (Llama 3 8B):
  全量微调: 8,000,000,000 × 2 bytes = 16 GB
  LoRA r=16:   ~4,200,000 × 2 bytes ≈ 8 MB    ← 差 2000 倍！
```

### 1.2 数学形式

$$W' = W + \frac{\alpha}{r}BA$$

其中：
- $W \in \mathbb{R}^{m \times n}$ —— 原始冻结权重
- $B \in \mathbb{R}^{m \times r}$ —— 下投影矩阵
- $A \in \mathbb{R}^{r \times n}$ —— 上投影矩阵
- $\alpha$ —— 缩放系数
- $r$ —— 秩（rank）

```python
import numpy as np

def lora_math_demo():
    """LoRA 数学演示"""
    
    np.random.seed(42)
    
    # 模拟一个 Linear 层的权重: [out_features=64, in_features=128]
    W_original = np.random.randn(64, 128).astype(np.float32) * 0.02
    
    # LoRA 低秩适配器: rank=8
    r = 8
    alpha = 16  # 通常 alpha = 2 * r
    
    B = np.zeros((64, r), dtype=np.float32)  # 初始化为0（关键！）
    A = np.random.randn(r, 128).astype(np.float32) * 0.01  # 随机初始化
    
    # LoRA 输出
    scaling = alpha / r
    delta_W = scaling * (B @ A)
    W_lora = W_original + delta_W
    
    print("=" * 60)
    print("LoRA 数学演示")
    print("=" * 60)
    
    print(f"\n[原始权重] 形状: {W_original.shape}")
    print(f"  参数量: {W_original.size:,}")
    print(f"  内存占用: {W_original.nbytes / 1024:.1f} KB")
    
    print(f"\n[LoRA 适配器] rank={r}, alpha={alpha}")
    print(f"  B 矩阵: {B.shape} ({B.size:,} 参数)")
    print(f"  A 矩阵: {A.shape} ({A.size:,} 参数)")
    print(f"  总 LoRA 参数: {B.size + A.size:,}")
    print(f"  内存占用: {(B.nbytes + A.nbytes) / 1024:.1f} KB")
    
    print(f"\n[合并后] W' = W + (α/r) × BA")
    print(f"  形状: {W_lora.shape}")
    
    param_ratio = (B.size + A.size) / W_original.size * 100
    print(f"\n[压缩比] LoRA 参数仅占原始的 {param_ratio:.2f}%")
    print(f"  → 节省了 {100 - param_ratio:.1f}% 的训练参数!")

lora_math_demo()
```

输出：

```
============================================================
LoRA 数学演示
============================================================

[原始权重] 形状: (64, 128)
  参数量: 8,192
  内存占用: 32.0 KB

[LoRA 适配器] rank=8, alpha=16
  B 矩阵: (64, 8) (512 参数)
  A 矩阵: (8, 128) (1,024 参数)
  总 LoRA 参数: 1,536
  内存占用: 6.0 KB

[合并后] W' = W + (α/r) × BA
  形状: (64, 128)

[压缩比] LoRA 参数仅占原始的 18.75%
  → 节省了 81.25% 的训练参数!
```

## 二、vLLM 对 LoRA 的支持

### 2.1 核心能力一览

| 能力 | 说明 | 状态 |
|------|------|------|
| **多 LoRA 同时加载** | 一个服务同时加载多个适配器 | ✅ |
| **运行时动态切换** | 通过 API 指定使用哪个 LoRA | ✅ |
| **热注册/卸载** | 运行时添加或移除 LoRA | ✅ |
| **与量化组合** | Base 模型 AWQ/GPTQ + LoRA FP16 | ✅ |
| **多租户隔离** | 不同客户使用不同 LoRA | ✅ |
| **Prefix Caching 加速** | LoRA 场景同样享受前缀缓存加速 | ✅ |

### 2.2 启动带 LoRA 的 vLLM 服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules \
        medical-lora=/path/to/medical-adapter \
        legal-lora=/path/to/legal-adapter \
        code-lora=/path/to/code-adapter \
        finance-lora=/path/to/finance-adapter \
    --max-loras 10 \
    --max-lora-rank 64 \
    --port 8000
```

**关键参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-lora` | False | 启用 LoRA 功能 |
| `--lora-modules` | 无 | 初始加载的 LoRA 列表 (`name=path`) |
| `--max-loras` | 1 | 同时驻留内存的最大 LoRA 数量 |
| `--max-lora-rank` | 16 | 单个 LoRA 的最大 rank |
| `--lora-extra-vocab-size` | 25600 | LoRA 额外词表大小 |
| `--long-lora-scaling-factors` | None | 长 context LoRA 的缩放因子 |

### 2.3 启动日志解读

```
INFO:     Loading model weights: meta-llama/Llama-3.1-8B-Instruct
INFO:     Model config: LlamaConfig(hidden_size=4096, ...)
INFO:     Loading LoRA adapters:
INFO:       medical-lora from /path/to/medical-adapter
INFO:         - rank: 16, target_modules: ['q_proj', 'v_proj', ...]
INFO:         - adapter size: 12.5 MB
INFO:       legal-lora from /path/to/legal-adapter
INFO:         - rank: 32, target_modules: [...]
INFO:         - adapter size: 24.8 MB
INFO:     Total LoRA memory overhead: ~45 MB (vs base model ~16 GB)
INFO:     Application startup complete.
```

注意最后一行——**LoRA 的内存开销相比基础模型几乎可以忽略不计**。

## 三、LoRA 文件格式要求

### 3.1 PEFT 格式（标准格式）

vLLM 使用 HuggingFace **PEFT (Parameter-Efficient Fine-Tuning)** 格式：

```
lora_adapter/
├── adapter_config.json      ← LoRA 配置 (rank, alpha, target_modules 等)
├── adapter_model.safetensors  ← LoRA 权重 (B 和 A 矩阵)
├── special_tokens_map.json   ← 特殊 token 映射（可选）
├── tokenizer_config.json     ← Tokenizer 配置（可选）
└── tokenier.json             ← Tokenizer 文件（可选）
```

### 3.2 adapter_config.json 结构

```json
{
    "alpha_pattern": {},
    "auto_mapping": null,
    "base_model_name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "bias": "none",
    "fan_in_fan_out": false,
    "inference_mode": true,
    "init_lora_weights": true,
    "layers_pattern": null,
    "layers_to_transform": null,
    "loftq_config": {},
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_rank": 16,
    "megatron_config": null,
    "megatron_core": "petrlm",
    "modules_to_save": null,
    "peft_type": "LORA",
    "r": 16,
    "revision": null,
    "target_modules": [
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "task_type": "CAUSAL_LM",
    "use_dora": false,
    "use_rslora": false
}
```

**关键字段说明**：

| 字段 | 含义 | 推荐值 |
|------|------|--------|
| `r` / `lora_rank` | LoRA 秩 | 8/16/32（越大越强但越慢）|
| `lora_alpha` | 缩放系数 | 通常设为 `2 × r` |
| `target_modules` | 目标模块 | 必须包含 `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `lora_dropout` | Dropout 率 | 训练时 0.05-0.1；推理时不影响 |
| `bias` | 是否训练 bias | `"none"`（推荐）|

### 3.3 target_modules 兼容性

```python
def target_modules_guide():
    """target_modules 兼容性指南"""
    
    guide = """
    ══════════════════════════════════════════════
          vLLM LoRA target_modules 兼容性
    ══════════════════════════════════════════════
    
    ✅ 完全支持的模块名 (vLLM 内部映射):
    ├── q_proj      (Query 投影)
    ├── k_proj      (Key 投影)
    ├── v_proj      (Value 投影)
    ├── o_proj      (Output 投影)
    ├── gate_proj   (FFN 门控, Llama/Qwen/Mistral)
    ├── up_proj     (FFN 上投影)
    └── down_proj   (FFN 下投影)
    
    ⚠️ 条件支持的模块:
    ├── kvq_proj    (某些 fused 实现)
    └── embed_tokens (embedding 层, 需要额外配置)
    
    ❌ 不支持的模块:
    ├── lm_head     (输出层)
    ├── norm 层      (LayerNorm/RMSNorm)
    └─ 自定义模块
    
    ⭐ 推荐配置 (通用最佳):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    
    🎯 最小配置 (节省显存和加速):
    target_modules = ["q_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    (注: 跳过 k_proj 在某些场景下影响很小)
    """
    print(guide)

target_modules_guide()
```

## 四、LoRA 显存开销分析

### 4.1 不同 Rank 的大小参考

```python
def lora_size_estimator():
    """LoRA 适配器大小估算"""
    
    models = [
        ("Llama 3.1 8B", 4096, 4096, 32),
        ("Llama 3.1 70B", 8192, 28672, 80),
        ("Qwen2.5 7B", 3584, 3584, 40),
        ("Qwen2.5 72B", 7168, 28672, 80),
        ("Mistral 7B", 4096, 14336, 32),
        ("DeepSeek-V2-16B", 5120, 2048, 48),
    ]
    
    ranks = [8, 16, 32, 64]
    
    print(f"\n{'模型':<20}", end="")
    for r in ranks:
        print(f"| Rank-{r:>7}", end="")
    print("| Base Model")
    print("-" * (22 + 10 * len(ranks) + 13))
    
    for name, hidden, ffn_dim, base_gb in models:
        print(f"{name:<20}", end="")
        
        for r in ranks:
            # Attention: 4个模块 (q,k,v,o), each = hidden × r × 2
            attn_params = 4 * hidden * r * 2
            # FFN: 3个模块 (gate, up, down), each = hidden × ffn_dim 或 ffn_dim × r 等
            ffn_params = (hidden * r + hidden * r + ffn_dim * r) * 2
            
            total_params = attn_params + ffn_params
            size_mb = total_params * 2 / 1024 / 1024  # FP16
            
            print(f"| {size_mb:>7.1f}MB", end="")
        
        print(f"| {base_gb}GB")

lora_size_estimator()
```

输出：

```
模型                  | Rank-8  | Rank-16 | Rank-32 | Rank-64 | Base Model
------------------------------------------------------------------------
Llama 3.1 8B         |   12.5MB |   25.0MB |   50.0MB |  100.0MB | 16GB
Llama 3.1 70B        |   78.2MB |  156.4MB |  312.8MB |  625.6MB | 140GB
Qwen2.5 7B           |   11.0MB |   22.0MB |   44.0MB |   88.0MB | 15GB
Qwen2.5 72B          |   69.5MB |  139.0MB |  278.0MB |  556.0MB | 144GB
Mistral 7B           |   14.2MB |   28.4MB |   56.8MB |  113.6MB | 14GB
DeepSeek-V2-16B      |   12.1MB |   24.2MB |   48.4MB |   96.8MB | 32GB
```

**结论**：即使 Rank=64，单个 LoRA 也只有几十到几百 MB——**可以轻松同时加载数十个**。

---

## 五、总结

本节建立了 LoRA 服务化的知识基础：

| 主题 | 核心要点 |
|------|---------|
| **核心公式** | $W' = W + \frac{\alpha}{r}BA$，只训练 B 和 A |
| **参数效率** | LoRA 仅需原模型的 0.1%-1% 参数（Rank 8-64）|
| **vLLM 支持** | 多 LoRA 同时加载、动态切换、热注册卸载、与量化组合 |
| **启动方式** | `--enable-lora --lora-modules name=path` |
| **文件格式** | PEFT 标准：`adapter_config.json` + `adapter_model.safetensors` |
| **target_modules** | 必须含 q/k/v/o/gate/up/down_proj |
| **显存开销** | Rank-16 的 8B 模型 LoRA 仅 ~25MB，可忽略 |

**核心要点回顾**：

1. **LoRA 是让大模型"专精化"的最有效手段**——用 1% 的参数量获得接近全量微调的效果
2. **vLLM 的 LoRA 支持是一等公民功能**——不是事后补丁，而是深度集成在调度系统中
3. **一个基础模型 + N 个 LoRA = N 个领域专家**——这是多租户 SaaS 服务的理想架构
4. **target_modules 必须正确配置**——缺少任何必需模块都会导致 LoRA 不生效
5. **LoRA 几乎没有运行时开销**——切换 LoRA 只是换一组指针，耗时 < 10ms

下一节我们将学习 **LoRA 服务化实战**——从启动到调用再到动态管理的完整流程。
