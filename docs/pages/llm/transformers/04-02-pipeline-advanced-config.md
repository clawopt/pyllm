# Pipeline 高级配置：精细控制每一个细节

## 这一节讲什么？

上一节我们学会了用三行代码调用各种 AI 能力，但实际生产中你很快会发现"默认配置"远远不够用——你可能需要控制生成的文本长度、调整输出的多样性、获取完整的概率分布而非仅返回 top-1 结果、处理超长文本的截断策略、在多 GPU 上高效运行等等。

这一节，我们将深入 Pipeline 的**每一个可配置参数**，不仅告诉你每个参数是什么，更要让你理解**为什么需要它、什么时候该调、调到什么值合适**。特别是对于文本生成任务的采样参数（temperature、top_k、top_p），这是所有大模型应用开发者必须掌握的核心技能。

---

## 一、模型与分词器的高级选择

## 1.1 指定非默认模型

虽然 HF 为每个任务选择了默认模型，但在实际项目中几乎总是需要替换：

```python
from transformers import pipeline

# 基本用法：指定模型名称
classifier = pipeline(
    "text-classification",
    model="uer/roberta-base-finetuned-chinanews-chinese",  # 中文新闻分类
)

# 模型和分词器可以分别指定（当它们不匹配时）
pipe = pipeline(
    "text-classification",
    model="my_local_model_path",
    tokenizer="bert-base-chinese",
)

# 从本地路径加载
local_pipe = pipeline(
    "text-classification",
    model="./my_finetuned_bert",  # 包含 config.json 和 model 文件的目录
)
```

## 1.2 framework 参数 —— PyTorch vs TensorFlow

```python
import pipeline

# 强制使用 PyTorch 后端
pt_pipe = pipeline("text-classification", framework="pt")

# 强制使用 TensorFlow 后端
tf_pipe = pipeline("text-classification", framework="tf")

print(f"PyTorch Pipeline 的模型类型: {type(pt_pipe.model).__name__}")
print(f"TensorFlow Pipeline 的模型类型: {type(tf_pipe.model).__name__}")
```

> **何时需要指定 framework？**
>
> - 当你的环境中同时安装了 PyTorch 和 TensorFlow，且默认选中的不是你想要的
> - 当你需要将训练好的模型从一种框架迁移到另一种框架做推理时
> - 大多数情况下不需要指定，HF 会自动检测并使用最优选项

## 1.3 模型修订版（Revision）与分支控制

Hugging Face Hub 上的模型可能有多个版本/分支：

```python
# 使用特定 commit hash 或 tag
pipe = pipeline(
    "text-generation",
    model="gpt2",
    revision="main",       # 主分支（默认）
    # revision="v1.0",     # 特定标签
    # revision="abc1234",  # 特定 commit
)

# 使用经过验证的安全版本
pipe = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    trust_remote_code=False,  # 不信任远程自定义代码（安全最佳实践）
)
```

---

## 二、设备管理与性能相关参数

## 2.1 device 参数详解

```python
import torch

# device 的多种指定方式
pipe_cpu = pipeline("text-classification", device=-1)        # CPU
pipe_gpu0 = pipeline("text-classification", device=0)         # 第一个 CUDA 设备
pipe_gpu1 = pipeline("text-classification", device=1)         # 第二个 CUDA 设备
pipe_mps = pipeline("text-classification", device="mps")      # Apple Silicon GPU
pipe_auto = pipeline("text-classification", device="auto")    # 自动选择最优设备

def check_device(pipe):
    """检查 Pipeline 实际使用的设备"""
    params = list(pipe.model.parameters())
    if params:
        return str(params[0].device)
    return "N/A (no parameters)"

for name, pipe in [("CPU", pipe_cpu), ("GPU 0", pipe_gpu0),
                    ("MPS", pipe_mps if torch.backends.mps.is_available() else pipe_cpu)]:
    print(f"{name:8s}: {check_device(pipe)}")
```

## 2.2 device_map —— 大模型的利器

对于单卡放不下的大模型，`device_map` 是必需的：

```python
from transformers import pipeline

# 自动分配到所有可用设备 + CPU 卸载
large_pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype=torch.float16,
    max_memory={0: "20GB", 1: "20GB", "cpu": "64GB"},
)

# 手动精细控制层分配
custom_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0, "model.layers.1": 0, ..., "model.layers.15": 0,
    "model.layers.16": 1, "model.layers.17": 1, ..., "model.layers.31": 1,
    "lm_head": 1,
}
```

## 2.3 batch_size —— 批量推理的速度-显存权衡

```python
from transformers import pipeline
import time

generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)

prompt = "The future of artificial intelligence"

sizes = [1, 4, 8, 16, 32]
results = []

for bs in sizes:
    prompts = [prompt] * bs

    start = time.time()
    outputs = generator(prompts, max_new_tokens=20, batch_size=bs)
    elapsed = time.time() - start

    throughput = bs / elapsed
    results.append((bs, elapsed, throughput))
    print(f"batch_size={bs:>3}: 总耗时={elapsed:.3f}s, 吞吐量={throughput:.1f} samples/s")

# 绘制速度对比图
import matplotlib.pyplot as plt
bss, times, tputs = zip(*results)

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'steelblue'
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Total Time (s)', color=color)
ax1.plot(bss, times, 'o-', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'coral'
ax2.set_ylabel('Throughput (samples/s)', color=color)
ax2.plot(bss, tputs, 's--', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Batch Size 对推理速度的影响')
plt.grid(True, alpha=0.3)
plt.show()
```

**关键发现**：
- **小 batch size（1-4）**：延迟低但吞吐量也低（GPU 利用率不足）
- **中等 batch size（8-16）**：通常是最优平衡点，GPU 计算单元被充分利用
- **大 batch size（32+）**：可能遇到显存瓶颈导致 OOM，或者由于序列长度不一致导致的 padding 浪费

---

## 三、生成任务的核心采样参数

这是 Pipeline 高级配置中**最重要**的部分。对于 `text-generation` 和 `text2text-generation` 任务，采样参数直接决定了输出质量。

## 3.1 解码策略全景图

首先理解几种基本解码策略的关系：

```
解码策略分类:

├── 确定性解码（每次输出相同）
│   ├── Greedy Search（贪婪搜索）：每步取概率最大的 token
│   └── Beam Search（束搜索）：保留 top-k 条候选路径
│
└── 随机性解码（每次输出不同）
    ├── Temperature Sampling（温度采样）：缩放 logits 分布
    ├── Top-K Sampling：只从概率最高的 K 个 token 中采样
    ├── Top-P (Nucleus) Sampling：从累积概率达到 P 的最小集合中采样
    └── 组合策略：Temperature + Top-K + Top-P 联合使用
```

让我们逐一深入。

## 3.2 Greedy vs Sampling —— 确定性与多样性的权衡

```python
from transformers import pipeline
import torch

generator = pipeline("text-generation", model="gpt2")

prompt = "人工智能的未来"

print("=" * 60)
print("=== Greedy Decoding (do_sample=False) ===")
greedy_results = generator(prompt, max_length=40, do_sample=False, num_return_sequences=3)
for i, r in enumerate(greedy_results):
    print(f"\n结果 {i+1}: {r['generated_text']}")

print("\n" + "=" * 60)
print("=== Random Sampling (do_sample=True) ===")
random_results = generator(prompt, max_length=40, do_sample=True, num_return_sequences=3,
                           temperature=0.8, seed=42)
for i, r in enumerate(random_results):
    print(f"\n结果 {i+1}: {r['generated_text']}")
```

观察输出你会发现：
- **Greedy 模式下三次结果完全相同**——因为每步都取最大概率 token
- **Sampling 模式下三次结果各不相同**——引入了随机性，产生更多样化的内容

> **Greedy 的问题**：容易陷入重复循环。比如 GPT-2 在 greedy 下可能会输出 "the the the the..." 这样的退化文本，因为模型对 "the" 这个词有很高的先验概率。

## 3.3 Temperature —— 控制输出的"创造性"

Temperature 是最直观的采样参数，它通过**缩放 logits 分布**来改变采样的随机程度：

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

其中 $z_i$ 是原始 logit，$T$ 是 temperature。

```python
def demonstrate_temperature_effect():
    """演示不同 temperature 对生成结果的影响"""
    import matplotlib.pyplot as plt
    import numpy as np

    generator = pipeline("text-generation", model="gpt2")

    prompt = "今天天气"
    temperatures = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, temp in enumerate(temperatures):
        row, col = idx // 3, idx % 3

        results = generator(prompt, max_length=50, do_sample=True,
                           temperature=temp, num_return_sequences=5,
                           pad_token_id=generator.tokenizer.eos_token_id)

        texts = "\n".join([f"{i+1}. {r['generated_text'][len(prompt):][:60]}"
                          for i, r in enumerate(results)])

        axes[row, col].text(0.05, 0.95, f"T = {temp}", fontsize=14, fontweight='bold',
                          transform=axes[row, col].transAxes, va='top')

        for i, r in enumerate(results):
            gen_part = r['generated_text'][len(prompt):]
            axes[row, col].text(0.05, 0.82 - i*0.16, f"{i+1}. {gen_part[:55]}",
                               fontsize=9, transform=axes[row, col].transAxes,
                               wrap=True)

        axes[row, col].axis('off')

        border_color = 'green' if temp <= 0.8 else ('orange' if temp <= 1.2 else 'red')
        for spine in axes[row, col].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

    plt.suptitle('Temperature 对生成结果多样性的影响\n'
                '(绿色=保守/确定, 橙色=适中, 红色=发散/混乱)', fontsize=14)
    plt.tight_layout()
    plt.show()

demonstrate_temperature_effect()
```

**Temperature 选择指南**：

| Temperature | 效果 | 适用场景 |
|-------------|------|---------|
| **0.1 - 0.3** | 非常保守，接近 greedy | 事实性问答、代码生成、数学推理 |
| **0.4 - 0.7** | 保守但有适度变化 | 正式写作、翻译、摘要 |
| **0.7 - 1.0** | 平衡创造性和连贯性 | 创意写作、对话系统（推荐默认值） |
| **1.0 - 1.5** | 较高的创造性 | 头脑风暴、诗歌创作 |
| **> 1.5** | 高度发散，可能不连贯 | 实验性用途 |

> **常见误区**："temperature 越高越好"或"temperature 越低越准"
>
> 都不对。过低的 temperature 导致输出单调重复；过高的 temperature 导致输出语无伦次。**0.7-0.9 是大多数场景的安全起点**。

## 3.4 Top-K Sampling —— 截断低概率尾部

Top-K 的思想很简单：只从概率最高的 K 个 token 中采样，其余全部排除：

```python
def demonstrate_top_k():
    """演示 Top-K 的工作原理"""
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    vocab_size = 100
    np.random.seed(42)
    logits = np.random.randn(vocab_size).astype(np.float32)
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].bar(range(vocab_size), probs, color='steelblue', alpha=0.7)
    axes[0].set_title('原始概率分布')
    axes[0].set_xlabel('Token ID')
    axes[0].set_ylabel('Probability')

    for idx, k in enumerate([10, 30, 50], 1):
        topk_probs = probs.copy()
        topk_indices = np.argsort(topk_probs)[::-1][k:]
        topk_probs[topk_indices] = 0
        topk_probs = topk_probs / topk_probs.sum()

        colors = ['coral' if p > 0 else 'lightgray' for p in topk_probs]
        axes[idx].bar(range(vocab_size), topk_probs, color=colors, alpha=0.7)
        axes[idx].set_title(f'Top-K (K={k})')
        axes[idx].set_xlabel('Token ID')
        remaining_mass = topk_probs.sum()
        axes[idx].text(0.95, 0.95, f'保留概率质量: {remaining_mass:.4f}',
                      transform=axes[idx].transAxes, ha='right', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.show()

demonstrate_top_k()
```

**Top-K 的问题**：K 值是固定的，但当分布形状变化时效果不稳定。如果概率非常集中（一个 token 占 90%），K=50 和 K=500 可能没有区别；如果概率很平缓，K=10 又太激进。

## 3.5 Top-P (Nucleus) Sampling —— 自适应的 Top-K

Top-P 解决了 Top-K 的固定性问题。它的思路是：从概率最高的 token 开始累加，直到累积概率超过 P，然后只在这个"核"集合中采样：

```python
def demonstrate_top_p():
    """演示 Top-P 的自适应特性"""
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    def nucleus_filter(probs, p):
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_sum = np.cumsum(sorted_probs)

        cutoff_idx = np.searchsorted(cumulative_sum, p) + 1
        cutoff_idx = min(cutoff_idx, len(probs))

        nuclei_indices = sorted_indices[:cutoff_idx]
        filtered_probs = np.zeros_like(probs)
        filtered_probs[nuclei_indices] = probs[nuclei_indices]
        filtered_probs /= filtered_probs.sum()

        return filtered_probs, cutoff_idx

    scenarios = [
        ("集中分布", lambda: F.softmax(torch.randn(100) * 0.5, dim=-1).numpy()),
        ("平缓分布", lambda: F.softmax(torch.randn(100) * 2.0, dim=-1).numpy()),
        ("双峰分布", lambda: {
            p = np.zeros(100); p[0]=0.4; p[1]=0.3; p[2:]=np.random.dirichlet([0.1]*98)*0.3; p/p.sum()
        } if False else None)[1],
    ]

    p_value = 0.9

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, dist_fn) in enumerate(scenarios):
        probs = dist_fn()
        filtered, k = nucleus_filter(probs, p_value)

        x = range(len(probs))
        width = 0.35

        axes[idx].bar([w - width/2 for w in x], probs, width, label='原始', alpha=0.6, color='steelblue')
        axes[idx].bar([w + width/2 for w in x], filtered, width, label=f'Top-P ({p_value})', alpha=0.6, color='coral')

        axes[idx].set_title(f'{name}\n(有效 K={k})')
        axes[idx].legend()
        axes[idx].set_xlabel('Token ID')
        axes[idx].set_ylabel('Probability')

    plt.suptitle(f'Top-P (Nucleus) Sampling: P={p_value} 自适应截断', fontsize=14)
    plt.tight_layout()
    plt.show()

demonstrate_top_p()
```

**Top-P 的优势**：
- **自适应**：集中分布时自动缩小候选集（如 K=5），平缓分布时自动扩大（如 K=40）
- **更稳定**：不需要针对不同模型手动调节 K 值
- **成为现代标准**：GPT-2/3、LLaMA、Claude 等主流模型默认使用 Top-P 或 Top-P + Temperature 组合

## 3.6 最佳组合策略

工业界最常用的配置是 **Temperature + Top-P** 的联合使用：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "请写一段关于深度学习的介绍"

configs = [
    {"name": "保守模式 (事实类)", "temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.1},
    {"name": "均衡模式 (通用)", "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.15},
    {"name": "创意模式 (写作)", "temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.2},
    {"name": "高度创意 (实验)", "temperature": 1.2, "top_p": 0.98, "repetition_penalty": 1.25},
]

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"配置: {cfg['name']}")
    print(f"  temperature={cfg['temperature']}, top_p={cfg['top_p']}, "
          f"repetition_penalty={cfg['repetition_penalty']}")

    results = generator(
        prompt,
        max_length=80,
        do_sample=True,
        **{k: v for k, v in cfg.items() if k != "name"},
        num_return_sequences=2,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    for i, r in enumerate(results):
        print(f"\n  输出 {i+1}: {r['generated_text'][len(prompt):]}")
```

## 3.7 Repetition Penalty —— 防止重复的利器

大模型的一个常见问题是**自重复**（repetition loop）——模型会不断重复相同的短语或句式。Repetition Penalty 通过降低已出现 token 的概率来缓解这个问题：

```python
def demonstrate_repetition_penalty():
    """演示 repetition_penalty 的效果"""
    from transformers import pipeline

    generator = pipeline("text-generation", model="gpt2")

    prompt = "The history of"

    penalties = [1.0, 1.1, 1.15, 1.2, 1.5, 2.0]

    for penalty in penalties:
        result = generator(
            prompt,
            max_length=60,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=penalty,
            pad_token_id=generator.tokenizer.eos_token_id,
        )[0]

        text = result["generated_text"]
        repeat_count = text.lower().split().count(text.lower().split()[0]) if text.split() else 0

        print(f"[penalty={penalty:.2f}] {text[:120]}...")
        print(f"  {'⚠️ 有重复' if repeat_count > 3 else '✓ 无明显重复'}\n")

demonstrate_repetition_penalty()
```

**Repetition Penalty 的工作原理**：
- 对于已经出现在生成结果中的 token，将其原始 logit 除以 penalty 值
- 如果 token 的原始 score > 0（即被模型"偏好"），除以 >1 的数会**降低**其概率
- 如果 token 的原始 score < 0（被模型"不喜欢"），除以 >1 的数会**提高**其概率（绝对值减小）

> **推荐值**：1.1 ~ 1.2 通常足够有效。超过 1.5 可能会导致输出变得不自然。

---

## 四、输入长度控制与截断策略

## 4.1 max_length 与 truncation

```python
from transformers import pipeline

classifier = pipeline("text-classification")

long_text = "这是一段非常非常长的文本。" * 100  # 构造超长文本

try:
    result = classifier(long_text)
    print("成功处理!")
except Exception as e:
    print(f"错误: {e}")

# 方案 1: 显式设置 max_length
result = classifier(long_text, truncation=True, max_length=512)
print(f"truncation=True, max_length=512: 成功!")

# 方案 2: 让模型自行决定（使用模型的最大位置编码长度）
result = classifier(long_text, truncation=True)
print(f"truncation=True (使用模型默认长度): 成功!")
```

## 4.2 截断位置的选择

```python
def demonstrate_truncation_strategies():
    """对比不同的截断策略"""

    classifier = pipeline("text-classification",
                         model="uer/roberta-base-finetuned-chinanews-chinese")

    text = "【重要声明】本文讨论了公司最新的财务报告，其中提到了营收增长和利润率的变化。" \
           "分析师认为这是一个积极的信号，预计未来股价会有良好表现。"

    configs = [
        {"max_length": 10, "truncation": True, "strategy": "只保留开头"},
        {"max_length": 10, "truncation": True, "stride": 0, "strategy": "默认从头截断"},
    ]

    for cfg in configs:
        result = classifier(text, **{k: v for k, v in cfg.items() if k != "strategy"})
        print(f"{cfg['strategy']}: {result}")
```

> **重要提示**：对于情感分析等任务，截断位置很重要！如果关键信息在文本末尾被截掉了，预测结果可能是错的。一些高级做法包括：
> - **滑动窗口**：对长文本分段处理再聚合结果
> - **首尾结合**：同时保留开头和结尾的信息
> - **Longformer/LED**：使用支持长文档的模型（最多支持 16K+ tokens）

---

## 五、return_all_scores —— 获取完整概率分布

默认情况下，分类任务只返回得分最高的类别。但很多时候我们需要**所有类别的概率**：

```python
from transformers import pipeline

classifier = pipeline("text-classification",
                     model="uer/roberta-base-finetuned-chinanews-chinese",
                     return_all_scores=True)

texts = ["国足今天赢了比赛", "苹果发布了新手机", "股市今日大涨"]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"\n文本: '{text}'")
    print("-" * 50)
    scores_sorted = sorted(result, key=lambda x: x["score"], reverse=True)
    for item in scores_sorted:
        bar_len = int(item["score"] * 50)
        bar = "█" * bar_len
        print(f"  {item['label']:12s} {item['score']:.4f} {bar}")
```

输出示例：
```
文本: '国足今天赢了比赛'
--------------------------------------------------
  体育          0.8923 ████████████████████████████████████████████████
  财经          0.0567 ███
  教育          0.0234 ██
  家居          0.0145 ▌
  游戏          0.0078 
  科技          0.0032 
  ...
```

**使用场景**：
- **置信度阈值过滤**：只接受最高概率 > 0.8 的预测
- **多标签分类**：选取所有概率 > 0.3 的标签
- **不确定性估计**：通过概率分布的熵来判断模型是否"不确定"
- **人机协同**：将低置信度的样本转给人工审核

---

## 六、Pipeline 与 Model 直接调用对比

为了更好地理解 Pipeline 的封装逻辑，我们做一个对比实验：

```python
def compare_pipeline_vs_manual():
    """对比 Pipeline 封装和手动调用的差异"""

    from transformers import (
        pipeline,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    import torch
    import torch.nn.functional as F
    import time

    text = "这家餐厅的服务态度非常好，菜品也很美味！"

    # ===== 方法 1: Pipeline ======
    pipe = pipeline("text-classification",
                   model="uer/roberta-base-finetuned-chinanews-chinese")

    start = time.time()
    pipe_result = pipe(text)
    pipe_time = time.time() - start

    # ===== 方法 2: 手动调用 ======
    model_name = "uer/roberta-base-finetuned-chinanews-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    start = time.time()
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        pred_label = model.config.id2label[pred_id]
        pred_score = probs[0, pred_id].item()

    manual_time = time.time() - start

    print("=" * 60)
    print("Pipeline vs 手动调用 对比")
    print("=" * 60)
    print(f"\n{'':15s} {'Pipeline':<20} {'Manual':<20}")
    print(f"{'预测标签':13s} {str(pipe_result[0]['label']):<20} {pred_label:<20}")
    print(f"{'置信度':13s} {pipe_result[0]['score']:<20.4f} {pred_score:<20.4f}")
    print(f"{'耗时':13s} {pipe_time:<20.4f}s {manual_time:<20.4f}s")

    match = pipe_result[0]["label"] == pred_label and \
            abs(pipe_result[0]["score"] - pred_score) < 1e-4
    print(f"\n结果一致性: {'✅ 完全一致' if match else '❌ 存在差异'}")

    print("\n--- Pipeline 内部做了什么而 Manual 没有 ---")
    print("1. 自动设备分配 (device)")
    print("2. 自动 batch 编码 (padding, truncation)")
    print("3. Softmax + argmax 后处理")
    print("4. 标签 ID → 标签名称映射")
    print("5. 错误处理和日志记录")

compare_pipeline_vs_manual()
```

---

## 小结

这一节我们全面掌握了 Pipeline 的高级配置能力：

1. **模型选择**：可以指定任意 Hub 模型、本地路径、特定 revision；支持 framework 选择（PT/TF）；`trust_remote_code` 安全开关
2. **设备管理**：device 参数支持 CPU/GPU/MPS/auto；device_map 支持大模型多卡分配；batch_size 控制批量推理的显存-速度权衡
3. **采样参数（核心重点）**：
   - **do_sample**：开启/关闭随机采样（False=贪婪确定性，True=随机性）
   - **temperature**：缩放概率分布（0.1~2.0，推荐 0.7-0.9）
   - **top_k**：固定数量截断（适合简单场景）
   - **top_p**：自适应核采样（推荐，现代标准）
   - **repetition_penalty**：防止重复（推荐 1.1-1.2）
   - 最佳实践：**temperature + top_p + repetition_penalty** 三件套组合使用
4. **长度控制**：max_length/truncation 处理超长文本，注意截断位置对语义的影响
5. **完整概率**：return_all_scores=True 获取所有类别分数，用于阈值过滤和多标签场景
6. **Pipeline vs Manual**：Pipeline 本质是 Model + Tokenizer + PostProcessor 的标准化封装，节省约 70% 的样板代码

下一节我们将学习如何创建自己的自定义 Pipeline。
