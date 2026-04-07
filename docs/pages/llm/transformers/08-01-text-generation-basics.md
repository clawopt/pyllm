# 文本生成基础：从 Logits 到文本的完整旅程

## 这一节讲什么？

当你向 ChatGPT 输入"请解释量子计算"并得到一大段流畅的回答时，你是否想过：**模型内部到底发生了什么？它如何从一个数学上的概率分布变成人类可读的文字？**

这一节是整个"文本生成与解码策略"章节的基石。我们将从头到尾追踪一个 token 的诞生过程：

1. **Logits 是什么？**——模型输出的原始分数，还不是概率
2. **Softmax 与温度参数**——将 logits 转换为概率分布
3. **Greedy Search**——最简单但最无聊的策略：每次都选最可能的词
4. **Beam Search**——维护多条候选路径，寻找全局最优
5. **采样方法（Sampling）**——引入随机性，让输出变得有趣

理解这些基础后，后面几节的 Top-K、Top-P、对比搜索等高级策略就都是在这个框架上的自然延伸。

---

## 一、回顾：Decoder-only 模型的前向传播

在讲生成之前，让我们先快速回顾一下 Decoder-only 模型（如 GPT）的前向传播过程。这是生成的起点。

```
输入: "今天天气"
        ↓
   [Tokenizer]
        ↓
input_ids = [871, 2747, 3893]    (3 个 token 的 ID)
        ↓
   [Embedding 层]
        ↓
token_embeddings: (batch=1, seq_len=3, hidden_size=768)
        ↓
+ position_embeddings + (可能的 token_type_embeddings)
        ↓
   [Transformer Encoder × N 层]
        ↓
hidden_states: (batch=1, seq_len=3, hidden_size=768)
        ↓
   [LM Head (线性投影)]
        ↓
logits: (batch=1, seq_len=3, vocab_size=50257)  ← 这就是我们的起点!
        ↓
   [解码策略: Greedy / Beam / Sampling ...]
        ↓
next_token_id = 2536   ← "真"
        ↓
拼接回输入 → 再次前向传播 → 下一个 token ...
```

关键点：**我们只关心最后一个位置的 logits**，因为自回归生成就是用已有序列预测下一个 token。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def trace_generation_journey():
    """追踪一次完整的前向传播和生成过程"""

    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt")

    print("=" * 65)
    print("GPT-2 前向传播追踪")
    print("=" * 65)
    print(f"\n输入文本: '{text}'")
    print(f"Token IDs: {inputs['input_ids'].tolist()[0]}")
    print(f"Token 解码: {[tokenizer.decode([t]) for t in inputs['input_ids'][0]]}")

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    print(f"\n[前向传播完成]")
    print(f"  logits 形状: {logits.shape}")
    print(f"  解释: (batch_size={logits.shape[0]}, "
          f"seq_len={logits.shape[1]}, vocab_size={logits.shape[2]})")

    # 只看最后一个位置
    next_token_logits = logits[0, -1, :]
    print(f"\n[最后一个位置的 logits]")
    print(f"  形状: {next_token_logits.shape} (vocab_size={len(next_token_logits)})")
    print(f"  最大值: {next_token_logits.max():.4f}")
    print(f"  最小值: {next_token_logits.min():.4f}")
    print(f"  均值: {next_token_logits.mean():.4f}")

    # Top-10 预测
    top_10 = torch.topk(next_token_logits, 10)
    print(f"\n[Top-10 预测 token]")
    for i in range(10):
        token_id = top_10.indices[i].item()
        score = top_10.values[i].item()
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1:>2}. [{token_id:>6}] '{token_text:<10}' logit={score:.4f}")

trace_generation_journey()
```

运行结果示例：

```
=================================================================
GPT-2 前向传播追踪
=================================================================

输入文本: 'The quick brown fox'
Token IDs: [464, 2068, 7586, 21831]
Token 解码: ['The', ' quick', ' brown', ' fox']

[前向传播完成]
  logits 形状: torch.Size([1, 4, 50257])
  解释: (batch_size=1, seq_len=4, vocab_size=50257)

[最后一个位置的 logits]
  形状: torch.Size([50257]) (vocab_size=50257)
  最大值: 28.3421
  最小值: -42.1567
  均值: -0.1234

[Top-10 预测 token]
   1. [  632] ' jumps'     logit=28.3421
   2. [  290] ' is'         logit=26.8912
   3. [  257] ' was'        logit=25.4456
   4. [  764] '.'           logit=24.1023
   ...
```

---

## 二、从 Logits 到概率：Softmax 与温度

## 2.1 为什么需要 Softmax？

模型输出的 logits 是**实数值**（可以是负数，范围不固定），而我们需要一个**概率分布**（所有值在 0 到 1 之间，总和为 1）。Softmax 就是做这个转换的：

$$P(token_i) = \frac{e^{logit_i / T}}{\sum_{j=1}^{|V|} e^{logit_j / T}}$$

其中 $T$ 是**温度参数（Temperature）**，它控制分布的"尖锐程度"。

## 2.2 温度参数的可视化理解

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def visualize_temperature_effect():
    """可视化温度参数对概率分布的影响"""

    # 模拟一组 logits (5 个候选 token)
    logits = torch.tensor([4.0, 2.0, 1.0, 0.5, -1.0])

    temperatures = [0.01, 0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(1, len(temperatures), figsize=(16, 4))

    tokens = ["jumps", "is", "was", ".", "over"]

    for idx, T in enumerate(temperatures):
        probs = F.softmax(logits / T, dim=-1)

        bars = axes[idx].bar(tokens, probs.numpy(), color='steelblue', alpha=0.7)
        axes[idx].set_ylim(0, 1)
        axes[idx].set_title(f'Temperature = {T}\nmax prob = {probs.max():.4f}', fontsize=11)

        for bar, prob in zip(bars, probs):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{prob:.3f}', ha='center', va='bottom', fontsize=9)

        if T <= 0.01:
            axes[idx].text(0.5, 0.5, '≈ Argmax\n(Greedy)',
                          transform=axes[idx].transAxes,
                          ha='center', va='center', fontsize=14,
                          color='red',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.suptitle('Temperature 参数对概率分布的影响', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('temperature_effect.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印数值对比
    print("\n" + "=" * 70)
    print("不同温度下的概率分布")
    print("=" * 70)
    print(f"\n{'Token':<10}", end="")
    for T in temperatures:
        print(f"T={T:<10}", end="")
    print()
    print("-" * 60)

    for i, token in enumerate(tokens):
        print(f"{token:<10}", end="")
        for T in temperatures:
            probs = F.softmax(logits / T, dim=-1)
            print(f"{probs[i]:<10.4f}", end="")
        print()

visualize_temperature_effect()
```

这个图会清晰地展示：
- **T → 0**：分布退化为 one-hot（只有一个 token 概率为 1），等同于 Greedy Search
- **T = 1**：标准的 Softmax 分布
- **T > 1**：分布变得更平坦（更随机），高概率 token 和低概率 token 的差距缩小
- **T → ∞**：分布趋近均匀（完全随机）

## 2.3 温度选择的实用指南

```python
def temperature_selection_guide():
    """温度参数选择指南"""

    guide = [
        {
            "场景": "事实性问答 / 数学计算",
            "推荐温度": "0.1 ~ 0.3",
            "原因": "需要确定性输出, 减少幻觉",
            "效果": "输出稳定、准确、可重复",
        },
        {
            "场景": "代码生成",
            "推荐温度": "0.2 ~ 0.5",
            "原因": "语法必须正确, 但可以有风格差异",
            "效果": "代码可运行, 实现方式有变化",
        },
        {
            "场景": "创意写作 / 故事创作",
            "推荐温度": "0.7 ~ 1.0",
            "原因": "需要多样性和创造性",
            "效果": "每生成都不同, 更有趣味性",
        },
        {
            "场景": "对话系统 / 聊天机器人",
            "推荐温度": "0.6 ~ 0.8",
            "原因": "平衡一致性和多样性",
            "效果": "回答自然但不至于天马行空",
        },
        {
            "场景": "翻译任务",
            "推荐温度": "0.3 ~ 0.5",
            "原因": "翻译追求准确性, 但允许地道表达",
            "效果": "翻译准确且符合目标语言习惯",
        },
    ]

    print("=" * 75)
    print("Temperature 选择指南")
    print("=" * 75)

    for g in guide:
        print(f"\n📌 {g['场景']}")
        print(f"   推荐温度: {g['推荐温度']}")
        print(f"   原因: {g['原因']}")
        print(f"   效果: {g['效果']}")

temperature_selection_guide()
```

---

## 三、Greedy Search：最快但最无趣

## 3.1 原理

Greedy Search（贪心搜索）是最简单的解码策略：**每一步都选择当前概率最大的那个 token**。

$$token_t = \arg\max_{v \in V} P(v | x_{<t})$$

优点：快！每步只需一次 argmax 操作。
缺点：容易陷入**重复循环**，缺乏多样性，且可能不是全局最优序列。

```python
def greedy_search_manual():
    """手动实现 Greedy Search"""

    import torch
    import torch.nn.functional as F

    # 模拟模型输出的 logits (假设 3 步生成)
    # 每一步的 shape: (vocab_size,)
    fake_logits_sequence = [
        torch.randn(50257),   # 第 1 步的 logits
        torch.randn(50257),   # 第 2 步的 logits
        torch.randn(50257),   # 第 3 步的 logits
    ]

    prompt_tokens = [464, 2068, 7586]  # "The quick brown"

    generated = list(prompt_tokens)
    print("=" * 60)
    print("Greedy Search 手动实现")
    print("=" * 60)
    print(f"\n初始输入: {' '.join(['[PAD]' if t == 0 else f'token_{t}' for t in generated])}")

    for step, logits in enumerate(fake_logits_sequence, 1):
        # 核心操作: argmax
        next_token = torch.argmax(logits).item()

        # 转换为概率看看
        probs = F.softmax(logits, dim=-1)
        top_prob = probs[next_token].item()

        generated.append(next_token)

        print(f"\n步骤 {step}:")
        print(f"  选中 token ID: {next_token}")
        print(f"  该 token 概率: {top_prob:.4f} ({top_prob*100:.2f}%)")
        print(f"  当前序列长度: {len(generated)}")

    print(f"\n✅ 最终生成的 token 序列 (长度 {len(generated)}):")
    print(f"   {generated}")

greedy_search_manual()
```

## 3.2 Greedy Search 的致命缺陷：重复循环

```python
def demonstrate_greedy_repetition():
    """展示 Greedy Search 的重复问题"""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "I love machine learning because"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("=" * 65)
    print("Greedy Search 的重复循环问题")
    print("=" * 65)
    print(f"\nPrompt: '{prompt}'")

    # Greedy Search
    greedy_output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,       # 关键: 不采样, 用 argmax
        repetition_penalty=1.0, # 不惩罚重复
    )
    greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

    print(f"\n[Greedy Search 结果]:")
    print(f"{greedy_text}")
    print(f"\n⚠️ 注意观察是否有词语或短语的重复出现!")

demonstrate_greedy_repetition()
```

你可能会看到类似这样的输出：

> I love machine learning because I love machine learning because I love machine learning because...

这就是 Greedy Search 的经典失败模式。一旦模型进入某个高概率的局部循环，它就无法逃脱了。

---

## 四、Beam Search：寻找更好的全局解

## 4.1 从 Greedy 到 Beam 的思维跳跃

Greedy Search 的问题在于它的**短视**——每步只看当前最优，不考虑未来。Beam Search 的核心思想是：**同时维护多条候选路径，每步扩展所有路径，然后保留最优的几条**。

```
Step 0:  "The"
           │
Step 1: ┌──┼────────┬────────┐
        quick   brown     A      (保留 top-3 beam)
           │       │        │
Step 2: ┌─┼─┐  ┌──┼──┐   ┌─┼─┐
      fox dog cat  fox jumped  bird flew  (每个 beam 扩展 top-3, 共 9 条路径)
              │       │         │
              ↓       ↓         ↓
           保留总得分最高的 3 条路径作为新的 beams
```

## 4.2 手写 Beam Search 实现

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple
import heapq


@dataclass(order=True)
class Beam:
    """Beam Search 中的一条候选路径"""
    score: float                  # 累积对数概率 (越小越好, 因为 heapq 是最小堆)
    token_ids: Tuple[int, ...]    # 已生成的 token 序列 (tuple 以便 hash)


class ManualBeamSearch:
    """
    手动实现的 Beam Search
    完整展示算法的工作原理
    """

    def __init__(self, model, tokenizer, beam_width: int = 3, length_penalty: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.length_penalty = length_penalty

    def generate(self, prompt: str, max_new_tokens: int = 20) -> List[str]:
        """
        执行 Beam Search 生成

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
        Returns:
            所有 beam 的最终结果列表
        """

        # Step 1: 编码 prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()

        # Step 2: 初始化 beam (只有一条路径, 得分为 0)
        beams = [Beam(score=0.0, token_ids=tuple(input_ids))]

        print(f"\n{'=' * 65}")
        print(f"Beam Search (width={self.beam_width}, max_tokens={max_new_tokens})")
        print(f"{'=' * 65}")
        print(f"\n初始 Prompt: '{prompt}' ({len(input_ids)} tokens)")

        # Step 3: 逐步扩展
        for step in range(max_new_tokens):

            # 所有 beam 都需要扩展
            all_candidates = []

            for beam in beams:
                # 准备输入
                input_tensor = torch.tensor([list(beam.token_ids)])

                # 前向传播获取 logits
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    next_logits = outputs.logits[0, -1, :]  # 最后一个位置

                # 转换为对数概率
                log_probs = F.log_softmax(next_logits, dim=-1)

                # 取 top-k 作为扩展候选
                topk_log_probs, topk_indices = torch.topk(log_probs, self.beam_width * 2)

                for i in range(len(topk_indices)):
                    new_token = topk_indices[i].item()
                    new_score = beam.score + topk_log_probs[i].item()
                    new_token_ids = beam.token_ids + (new_token,)

                    all_candidates.append(
                        Beam(score=new_score, token_ids=new_token_ids)
                    )

            # Step 4: 保留得分最高的 beam_width 条路径
            beams = heapq.nsmallest(self.beam_width, all_candidates)

            # 打印当前状态
            print(f"\n[Step {step+1}] 当前 Top-{self.beam_width} Beams:")
            for rank, beam in enumerate(beams, 1):
                # 应用 length penalty
                adjusted_score = beam.score / (len(beam.token_ids) ** self.length_penalty)
                decoded = self.tokenizer.decode(list(beam.token_ids))
                # 截断显示
                display_text = decoded[:60] + "..." if len(decoded) > 60 else decoded
                print(f"  #{rank} (score={beam.score:.2f}, adj={adjusted_score:.2f}): "
                      f"'{display_text}'")

            # 检查是否所有 beam 都生成了 EOS
            if all(b.token_ids[-1] == self.tokenizer.eos_token_id for b in beams):
                print(f"\n✅ 所有 beam 都到达 EOS, 提前终止!")
                break

        # 返回所有 beam 的结果
        results = []
        for rank, beam in enumerate(beams, 1):
            text = self.tokenizer.decode(list(beam.token_ids))
            results.append({
                "rank": rank,
                "score": beam.score,
                "text": text,
            })

        return results


def demo_beam_search():
    """演示 Beam Search"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    searcher = ManualBeamSearch(model, tokenizer, beam_width=3, length_penalty=1.0)
    results = searcher.generate(
        prompt="Artificial intelligence will",
        max_new_tokens=15,
    )

    print(f"\n{'=' * 65}")
    print("最终结果:")
    print(f"{'=' * 65}")
    for r in results:
        print(f"\n[Beam #{r['rank']}] Score: {r['score']:.2f}")
        print(f"  {r['text']}")


if __name__ == "__main__":
    demo_beam_search()
```

运行结果示例：

```
=================================================================
Beam Search (width=3, max_tokens=15)
=================================================================

初始 Prompt: 'Artificial intelligence will' (4 tokens)

[Step 1] 当前 Top-3 Beams:
  #1 (score=-1.23): 'Artificial intelligence will be'
  #2 (score=-2.45): 'Artificial intelligence will continue'
  #3 (score=-3.12): 'Artificial intelligence will change'

[Step 2] 当前 Top-3 Beams:
  #1 (score=-2.01): 'Artificial intelligence will be able'
  #2 (score=-3.56): 'Artificial intelligence will continue to'
  #3 (score=-4.23): 'Artificial intelligence will be the'

...
```

## 4.3 Beam Width 的选择

```python
def beam_width_analysis():
    """Beam Width 对生成质量和速度的影响"""

    print("=" * 70)
    print("Beam Width 选择分析")
    print("=" * 70)

    widths = [
        {"width": 1, "alias": "Greedy", "quality": "★☆☆☆☆", "diversity": "无",
         "speed": "最快", "note": "退化为 Greedy Search"},
        {"width": 2, "quality": "★★☆☆☆", "diversity": "极低",
         "speed": "很快", "note": "略优于 Greedy"},
        {"width": 3, "quality": "★★★☆☆", "diversity": "低",
         "speed": "快", "note": "常用默认值, 平衡质量和速度"},
        {"width": 5, "quality": "★★★★☆", "diversity": "中",
         "speed": "中等", "note": "质量更好, 速度下降明显"},
        {"width": 10, "quality": "★★★★★", "diversity": "中高",
         "speed": "慢", "note": "高质量, 但计算量 10x Greedy"},
        {"width": 20, "quality": "★★★★★", "diversity": "高",
         "速度": "很慢", "note": "边际收益递减, 一般不需要这么大"},
    ]

    print(f"\n{'Width':<8}{'质量':<10}{'多样性':<10}{'速度':<8}{'备注'}")
    print("-" * 65)
    for w in widths:
        print(f"{w['width']:<8}{w['quality']:<10}{w['diversity']:<10}{w['speed']:<8}{w['note']}")

    print("\n💡 经验法则:")
    print("   • 翻译/摘要等有'标准答案'的任务 → beam=3~5")
    print("   • 开放式生成 → 不建议用 Beam Search (用 Sampling)")
    print("   • 实时应用对延迟敏感 → beam=1~3")

beam_width_analysis()
```

## 4.4 Length Penalty：防止 Beam Search 偏好短序列

Beam Search 有一个天然倾向：**倾向于生成更短的序列**。因为每多生成一个 token，累积概率就要再乘以一个 < 1 的数，导致得分越来越小。长序列即使每个 token 的条件概率都很高，其乘积也可能小于一个较短但每个 token 概率极高的序列。

`length_penalty` 就是为了解决这个问题：

$$adjusted\_score = \frac{score}{|sequence|^{\alpha}}$$

其中 $\alpha > 1$ 时鼓励更长序列，$\alpha < 1$ 时偏好更短序列。

```python
def demonstrate_length_penalty():
    """展示 Length Penalty 的作用"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")

    penalties = [0.8, 1.0, 2.0]

    print("=" * 65)
    print("Length Penalty 效果对比")
    print("=" * 65)
    print(f"\nPrompt: '{prompt}'")

    for lp in penalties:
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            length_penalty=lp,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        word_count = len(text.split())
        print(f"\n[length_penalty={lp}] ({word_count} words)")
        print(f"  {text}")

demonstrate_length_penalty()
```

---

## 五、采样方法（Sampling）：引入随机性

## 5.1 为什么需要采样？

Greedy 和 Beam Search 都是**确定性的**——同样的输入永远产生同样的输出。但在很多场景下，我们需要**多样性**：

- 创意写作：每次生成应该有所不同
- 对话系统：不想每次都说一样的话
- 数据增强：需要生成多样化的训练样本

采样方法的核心思想：**按照概率分布随机抽取 token，而不是总是取最大值**。

## 5.2 基础采样（Pure Sampling）

最简单的采样就是直接按 Softmax 后的概率分布进行多项式抽样：

```python
import torch
import torch.nn.functional as F


def pure_sampling_demo():
    """纯概率采样演示"""

    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5])
    probs = F.softmax(logits, dim=-1)

    print("=" * 55)
    print("纯概率采样 (Multinomial Sampling)")
    print("=" * 55)
    print(f"\n原始 logits: {logits.tolist()}")
    print(f"概率分布:   {[f'{p:.4f}' for p in probs.tolist()]}")

    # 多次采样观察结果
    print(f"\n采样 10 次 (temperature=1.0):")
    samples = torch.multinomial(probs.unsqueeze(0), num_samples=10, replacement=True)[0]
    token_labels = ["A", "B", "C", "D", "E"]
    sampled_tokens = [token_labels[s.item()] for s in samples]
    counts = {}
    for s in sampled_tokens:
        counts[s] = counts.get(s, 0) + 1

    for token in token_labels:
        count = counts.get(token, 0)
        bar = "█" * count
        print(f"  {token}: {count}/10 {bar}")

    print(f"\n💡 观察: 高概率 token (A/B) 被抽到的次数更多,"
           f"但低概率 token (D/E) 也有机会被选中!")

pure_sampling_demo()
```

## 5.3 Temperature Sampling 的实际效果

```python
def temperature_sampling_comparison():
    """对比不同温度下的采样效果"""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt")

    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

    print("=" * 65)
    print("不同 Temperature 下的采样效果对比")
    print("=" * 65)
    print(f"\nPrompt: '{prompt}'\n")

    for T in temperatures:
        # 同一个 seed 下多次采样
        print(f"[Temperature = {T}]")
        for run in range(3):
            torch.manual_seed(42 + run)  # 不同 seed 产生不同随机结果
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=T,
                top_k=0,             # 不限制 top-k (纯 temperature sampling)
                top_p=1.0,           # 不限制 top-p
            )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"  Run {run+1}: {text[len(prompt):]}...")
        print()

temperature_sampling_comparison()
```

你会观察到明显的规律：
- **T=0.1**：几乎和 Greedy 一样，三次运行结果相同或极度相似
- **T=0.5**：有些变化，但大体方向一致
- **T=1.0**：每次都有明显不同的表达方式
- **T=1.5**：开始出现不太连贯的内容
- **T=2.0**：输出变得混乱，可能出现无意义的内容

---

## 六、手写完整的 Generate 循环

现在让我们把以上所有概念整合起来，手写一个完整的自回归生成循环：

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Callable


class CustomGenerator:
    """
    手写的文本生成器
    支持多种解码策略, 完整展示生成流程
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        strategy: str = "sampling",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_stop: bool = True,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        自定义生成方法

        Args:
            prompt: 输入提示
            max_new_tokens: 最大新生成 token 数
            strategy: "greedy" / "sampling"
            temperature: 温度参数 (仅 sampling 有效)
            top_k: Top-K 采样 (0 表示不使用)
            top_p: Top-P (Nucleus) 采样 (1.0 表示不使用)
            repetition_penalty: 重复惩罚系数
            eos_stop: 是否遇到 EOS 就停止
            callback: 每个 step 后调用的回调函数
        """

        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # 记录已生成的 token (用于重复惩罚)
        generated_tokens = input_ids[0].tolist()

        print(f"\n{'=' * 60}")
        print(f"自定义 Generate 循环 (strategy={strategy})")
        print(f"{'=' * 60}")
        print(f"Prompt: '{prompt}' ({len(generated_tokens)} tokens)")

        for step in range(max_new_tokens):
            # 前向传播
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[0, -1, :]

            # === 重复惩罚 ===
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty

            # === 温度缩放 ===
            if strategy == "sampling" and temperature > 0:
                next_token_logits = next_token_logits / temperature

            # === Top-K 截断 ===
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                ).values[..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # === Top-P (Nucleus) 截断 ===
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = (
                    sorted_indices_to_remove[..., :-1].clone()
                )
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # === 转换为概率 ===
            probs = F.softmax(next_token_logits, dim=-1)

            # === 选择下一个 token ===
            if strategy == "greedy":
                next_token = torch.argmax(probs).unsqueeze(0)
            else:  # sampling
                next_token = torch.multinomial(probs.unsqueeze(0), num_samples=1)

            # === 追加到序列 ===
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens.append(next_token.item())

            # 打印进度
            new_token_str = self.tokenizer.decode([next_token.item()])
            current_text = self.tokenizer.decode(input_ids[0])
            display = current_text[-60:] + "..." if len(current_text) > 60 else current_text

            top3 = torch.topk(probs, 3)
            top3_info = ", ".join([
                f"'{self.tokenizer.decode([t.item()])}'({p:.3f})"
                for t, p in zip(top3.indices, F.softmax(top3.values, dim=-1))
            ])

            print(f"  [Step {step+1:>2}] token='{new_token_str}' | "
                  f"Top-3: [{top3_info}] | seq_len={len(generated_tokens)}")

            # 回调
            if callback:
                callback(step, next_token.item(), probs)

            # EOS 检查
            if eos_stop and next_token.item() == self.tokenizer.eos_token_id:
                print(f"\n  ⏹️  遇到 EOS token, 提前终止!")
                break

        final_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"\n✅ 生成完成! 总计 {len(generated_tokens) - len(inputs['input_ids'][0])} 个新 token")
        return final_text


def demo_custom_generator():
    """演示自定义生成器"""

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    generator = CustomGenerator(model, tokenizer)

    # 测试不同策略
    result_greedy = generator.generate(
        prompt="Machine learning is",
        max_new_tokens=20,
        strategy="greedy",
    )

    print("\n" + "-" * 60 + "\n")

    result_sampling = generator.generate(
        prompt="Machine learning is",
        max_new_tokens=20,
        strategy="sampling",
        temperature=0.8,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.15,
    )


if __name__ == "__main__":
    demo_custom_generator()
```

---

## 七、HF `generate()` 方法核心参数速查

Hugging Face 的 `model.generate()` 方法封装了我们上面讨论的所有策略。以下是核心参数的快速参考：

```python
def generate_params_cheatsheet():
    """generate() 核心参数速查表"""

    params = [
        ("max_new_tokens", "int", "最大新生成的 token 数 (不含 prompt)", "50"),
        ("max_length", "int", "总序列最大长度 (含 prompt)", "❌ 建议用 max_new_tokens"),
        ("min_new_tokens", "int", "最少生成的 token 数", "1"),
        ("do_sample", "bool", "True=采样, False=Greedy", "False"),
        ("temperature", "float", "采样温度 (需 do_sample=True)", "1.0"),
        ("top_k", "int", "Top-K 采样 (0=禁用)", "50"),
        ("top_p", "float", "Top-P/Nucleus 采样 (1.0=禁用)", "1.0"),
        ("repetition_penalty", "float", "重复惩罚 (>1.0 惩罚重复)", "1.0"),
        ("length_penalty", "float", "长度惩罚 (Beam Search)", "1.0"),
        ("no_repeat_ngram_size", "int", "禁止重复的 n-gram 大小", "0"),
        ("num_beams", "int", "Beam Search 宽度 (1=Greedy)", "1"),
        ("early_stopping", "bool", "是否提前停止 (Beam Search)", "False"),
        ("num_return_sequences", "int", "返回几个不同结果", "1"),
        ("pad_token_id", "int", "填充 token ID", "必须设置!"),
        ("eos_token_id", "int", "结束 token ID", "自动检测"),
    ]

    print("=" * 80)
    print("HF generate() 核心参数速查表")
    print("=" * 80)
    print(f"\n{'参数名':<24}{'类型':<8}{'说明':<40}{'默认值'}")
    print("-" * 95)
    for name, type_, desc, default in params:
        print(f"{name:<24}{type_:<8}{desc:<40}{default}")

generate_params_cheatsheet()
```

---

## 八、常见误区与面试高频问题

## 8.1 常见误区

```python
def common_mistakes():
    """文本生成中的常见误区"""

    mistakes = [
        {
            "误区": "do_sample=False 比 do_sample=True 快很多",
            "真相": "差距不大。两者主要区别在最后一步 (argmax vs multinomial),"
                   "前向传播的开销完全一样",
        },
        {
            "误区": "Beam Search 总是比 Sampling 好",
            "真相": "对于开放式生成 (对话/创作), Beam Search 反而更差——"
                   "因为它缺乏多样性, 且倾向于生成通用但平庸的内容",
        },
        {
            "误区": "Temperature 越大越有创意",
            "真相": "Temperature > 1.5 后输出质量急剧下降, 变得语无伦次。"
                   "创意性和质量需要平衡, 推荐 0.7~1.0",
        },
        {
            "误区": "Top-K 和 Top-P 可以互相替代",
            "真相": "它们解决的是不同问题: Top-K 固定截断数量, Top-P 动态调整。"
                   "GPT 系列通常组合使用两者 (先 K 再 P)",
        },
        {
            "误区": "max_new_tokens 和 max_length 是一回事",
            "真相": "max_length 包含 prompt 的长度, max_new_tokens 只算新生的。"
                   "如果 prompt 已经很长, max_length=100 可能只能生成几个字!",
        },
    ]

    print("=" * 75)
    print("文本生成常见误区")
    print("=" * 75)
    for m in mistakes:
        print(f"\n❌ {m['误区']}")
        print(f"   💡 {m['真相']}")

common_mistakes()
```

## 8.2 面试高频问题

```python
def interview_questions():
    """文本生成面试高频问题"""

    questions = [
        {
            "Q": "Greedy Search 和 Beam Search 的核心区别是什么?",
            "A": "Greedy 每步只保留 1 个最优路径; Beam 同时保留 N 条候选路径。"
               "Beam 能找到更好的全局解, 但计算量是 Greedy 的 N 倍。",
        },
        {
            "Q": "为什么 Beam Search 在对话/创作任务上表现不好?",
            "A": "(1) Beam Search 倾向于生成高概率(即通用/安全)的内容, 缺乏惊喜;"
               "(2) 多个 beam 最终趋于趋同, 输出多样性差;"
               "(3) Length normalization 问题导致偏好短序列或长序列。",
        },
        {
            "Q": "Temperature 参数的本质作用是什么?",
            "A": "控制 Softmax 分布的熵(尖锐程度)。T→0 退化为确定性(argmax), "
               "T→∞ 趋近均匀分布(完全随机)。本质是在'确定性'和'随机性'之间的调节旋钮。",
        },
        {
            "Q": "Top-K 和 Top-P (Nucleus Sampling) 各自适合什么场景?",
            "A": "Top-K 固定截断数量, 简单高效, 适合大多数场景;"
               "Top-P 动态截断(累积概率达到阈值), 更智能地适应分布形状。"
               "实践中常组合使用: 先 Top-K 去除极端长尾, 再 Top-P 精细筛选。",
        },
        {
            "Q": "什么是重复惩罚(repetition_penalty)? 它是如何工作的?",
            "A": "对已经出现过的 token 的 logits 进行惩罚(除以 >1 的系数)。"
               "如果 logit > 0 则除以 penalty, 如果 < 0 则乘以 penalty。"
               "有效防止模型陷入重复循环, 推荐值 1.1~1.25。",
        },
    ]

    print("=" * 80)
    print("文本生成面试高频问题")
    print("=" * 80)
    for i, q in enumerate(questions, 1):
        print(f"\n📝 Q{i}: {q['Q']}")
        print(f"   💡 A: {q['A']}")

interview_questions()
```

---

## 九、本章小结

这一节我们从模型的原始输出 logits 出发，一步步构建了对文本生成的完整认知：

| 概念 | 核心思想 | 适用场景 | 局限性 |
|------|---------|---------|--------|
| **Greedy Search** | 每步取 argmax | 确定性任务 | 重复循环、局部最优 |
| **Beam Search** | 维护 N 条候选路径 | 翻译/摘要 | 多样性差、计算量大 |
| **Temperature Sampling** | 按 softmax 概率随机采样 | 创意写作/对话 | 可能产生不连贯内容 |
| **Top-K** | 只从 K 个最高概率 token 中采样 | 去除长尾噪声 | K 太大则无效 |
| **Top-P (Nucleus)** | 动态截断到累积概率 P | 自适应采样 | 与 Top-K 功能重叠 |

**核心要点**：
1. **Logits → Softmax → 采样/选择** 是生成的三步曲
2. **没有一种策略适用于所有任务**——需要根据任务特性选择
3. **Greedy/Beam 适合有"正确答案"的任务，Sampling 适合开放域任务**
4. **Temperature + Top-K + Top-P + Repetition Penalty** 是生产环境的标准配置

下一节我们将深入探索 **高级解码策略**：对比搜索、典型采样、约束解码等更精细的控制方法。
