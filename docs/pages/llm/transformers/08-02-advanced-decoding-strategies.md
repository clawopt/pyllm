# 高级解码策略：对比搜索、核采样与约束解码

## 这一节讲什么？

上一节我们掌握了文本生成的基础工具箱——Greedy、Beam Search、Temperature Sampling、Top-K、Top-P。但在实际应用中，这些基础策略往往不够用：

- **Top-P 采样**虽然比纯 Temperature Sampling 更智能，但仍然可能生成不连贯的内容
- **Beam Search** 虽然能找到"更优"路径，但在开放式任务上表现平庸
- 有时候我们需要**强制输出特定格式**（如 JSON、SQL），或者**避免某些不安全内容**

这一节将介绍几种更高级的解码策略：
1. **对比搜索（Contrastive Search）**——结合概率和表示相似度的新方法
2. **典型采样（Typical Sampling）**——基于信息量的自适应采样
3. **Min-P 采样**——设置绝对概率下界的新方法
4. **约束 decoding**——强制输出符合正则/语法/格式要求

---

## 一、对比搜索（Contrastive Search）

## 1.1 动机：为什么需要新方法？

传统的采样方法只关注**概率分布**本身，但忽略了一个重要信号：**相邻 token 之间的语义连贯性**。

想象一下这个场景：
- 模型预测下一个 token 是 "apple" (概率 30%) 或 "run" (概率 28%)
- 从纯概率角度看，两者差不多
- 但如果前文是 "I want to eat an..."，那么 "apple" 在语义上明显更连贯

对比搜索的核心思想就是：**同时考虑 token 的概率和它与上下文的表示相似度**。

## 1.2 数学原理

对比搜索选择下一个 token 的公式为：

$$token_t = \arg\max_v \left( \alpha \cdot P(v|x_{<t}) - (1-\alpha) \cdot \max_{j \in \mathcal{H}} \cos(h_v, h_j) \right)$$

其中：
- $P(v|x_{<t})$ 是 token $v$ 的预测概率（来自模型）
- $\cos(h_v, h_j)$ 是候选 token $v$ 的隐状态与历史隐状态的余弦相似度
- $\mathcal{H}$ 是历史隐状态集合（通常取最近几步）
- $\alpha$ 是平衡系数（通常 0.6~0.7）

直觉理解：**选一个既概率高又和前面说过的内容不太像的 token**——这天然地避免了重复！

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import numpy as np


class ContrastiveSearchDecoder:
    """
    对比搜索 (Contrastive Search) 解码器
    原论文: "A Contrastive Framework for Neural Text Generation"
    
    核心思想: 同时优化两个目标:
    1. 最大化当前 token 的预测概率 (保持流畅性)
    2. 最小化与历史隐状态的相似度 (保证多样性/去重)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        alpha: float = 0.6,          # 概率权重
        top_k: int = 5,              # 候选数量
        history_len: int = 5,        # 历史窗口大小
        use_hidden_states: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.top_k = top_k
        self.history_len = history_len
        self.use_hidden_states = use_hidden_states
        self.device = next(model.parameters()).device

    def _get_hidden_state(self, input_ids):
        """获取最后一个位置的隐藏状态"""
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            # 使用最后一层的隐藏状态
            last_hidden = outputs.hidden_states[-1]
            return last_hidden[:, -1, :]  # (batch, hidden_size)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """执行对比搜索生成"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # 初始化历史隐藏状态
        history_hidden = []

        print("=" * 65)
        print(f"对比搜索 (α={self.alpha}, top_k={self.top_k})")
        print("=" * 65)
        print(f"\nPrompt: '{prompt}'")

        for step in range(max_new_tokens):

            # 前向传播获取 logits 和隐藏状态
            outputs = self.model(input_ids, output_hidden_states=True)
            logits = outputs.logits[0, -1, :]           # 最后位置 logits
            current_hidden = outputs.hidden_states[-1][0, -1, :]  # 最后位置 hidden

            # Step 1: 概率分数 (目标函数第一项)
            probs = F.softmax(logits / 1.0, dim=-1)

            # Step 2: 取 Top-K 候选
            topk_probs, topk_indices = torch.topk(probs, self.top_k)
            score_prob = topk_probs.log()  # 使用 log 概率

            # Step 3: 相似度惩罚 (目标函数第二项)
            if len(history_hidden) > 0 and self.use_hidden_states:
                # 计算每个候选 token 的隐藏状态
                candidate_hiddens = []
                for idx in topk_indices:
                    temp_ids = torch.cat([input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    temp_output = self.model(temp_ids, output_hidden_states=True)
                    cand_hidden = temp_output.hidden_states[-1][0, -1, :]
                    candidate_hiddens.append(cand_hidden)

                candidate_hiddens = torch.stack(candidate_hiddens)  # (top_k, hidden)

                # 与所有历史隐藏状态计算最大相似度
                history_stack = torch.stack(history_hidden)  # (history_len, hidden)
                sim_matrix = F.cosine_similarity(
                    candidate_hiddens.unsqueeze(1),   # (top_k, 1, hidden)
                    history_stack.unsqueeze(0),       # (1, history_len, hidden)
                    dim=-1,
                )  # (top_k, history_len)

                max_sim, _ = sim_matrix.max(dim=1)  # (top_k,) 最大相似度
                score_sim = max_sim  # 相似度越高 → 惩罚越大
            else:
                score_sim = torch.zeros(self.top_k, device=self.device)

            # Step 4: 综合得分
            final_score = self.alpha * score_prob - (1 - self.alpha) * score_sim

            # Step 5: 选择最优 token
            best_idx = torch.argmax(final_score)
            next_token = topk_indices[best_idx].unsqueeze(0).unsqueeze(0)

            # 更新历史
            history_hidden.append(current_hidden)
            if len(history_hidden) > self.history_len:
                history_hidden.pop(0)

            # 追加到序列
            input_ids = torch.cat([input_ids, next_token], dim=1)

            new_token_str = self.tokenizer.decode([next_token.item()])
            prob_val = topk_probs[best_idx].item()
            sim_val = score_sim[best_idx].item() if score_sim.numel() > 1 else 0

            print(f"  [Step {step+1:>2}] '{new_token_str}' "
                  f"(prob={prob_val:.3f}, sim_penalty={sim_val:.3f})")

            if next_token.item() == self.tokenizer.eos_token_id:
                print(f"\n⏹️  遇到 EOS")
                break

        result = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"\n✅ 生成完成!")
        return result


def demo_contrastive_search():
    """演示对比搜索"""

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    decoder = ContrastiveSearchDecoder(model, tokenizer, alpha=0.6, top_k=5)

    result = decoder.generate(
        prompt="Deep learning has revolutionized",
        max_new_tokens=40,
    )
    print(f"\n结果:\n{result}")


if __name__ == "__main__":
    demo_contrastive_search()
```

## 1.3 对比搜索 vs 其他方法的效果对比

```python
def contrastive_vs_others():
    """对比搜索与其他方法的对比"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "The future of artificial intelligence"
    inputs = tokenizer(prompt, return_tensors="pt")

    strategies = {
        "Greedy": dict(do_sample=False),
        "Beam-5": dict(num_beams=5, do_sample=False, early_stopping=True),
        "Sampling(T=0.8)": dict(do_sample=True, temperature=0.8, top_k=50, top_p=0.92),
        "Sampling(T=1.2)": dict(do_sample=True, temperature=1.2, top_k=50, top_p=0.92),
    }

    print("=" * 70)
    print("不同解码策略效果对比")
    print("=" * 70)
    print(f"\nPrompt: '{prompt}'\n")

    for name, kwargs in strategies.items():
        output = model.generate(**inputs, max_new_tokens=40, **kwargs,
                                repetition_penalty=1.15, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
        word_count = len(text.split())
        
        # 简单检测重复
        words = text.lower().split()
        repeat_ratio = len(set(words)) / max(len(words), 1)
        
        status = "✅" if repeat_ratio > 0.7 else ("⚠️" if repeat_ratio > 0.5 else "❌")
        
        print(f"[{status} {name}] ({word_count} 词, 去重率={repeat_ratio:.0%})")
        print(f"  {text.strip()}\n")

contrastive_vs_others()
```

---

## 二、典型采样（Typical Sampling）

## 2.1 核心思想

典型采样（Typical Sampling）由 Meister et al. 在 2023 年提出，它的出发点很优雅：**一个"典型的"（typical）token 应该具有适中的信息量——不能太可预测（信息量太低），也不能太出乎意料（信息量太高）**。

具体做法是：
1. 对每个候选 token 计算其**信息量**（即负对数概率）：$-\log p(x)$
2. 计算所有候选 token 信息量的**熵**
3. 只保留信息量在 **均值 ± τ × 标准差** 范围内的 token（τ 是超参数）
4. 在保留的 token 中均匀采样

```python
def typical_sampling_demo():
    """典型采样原理演示"""

    import torch
    import torch.nn.functional as F
    import math

    # 模拟一个概率分布 (10 个候选 token)
    probs = torch.tensor([0.35, 0.25, 0.15, 0.10, 0.06, 0.04, 0.02, 0.015, 0.01, 0.005])
    tokens = ["the", "a", "is", ".", "in", "of", "to", "and", "very", "xyz"]

    print("=" * 65)
    print("典型采样 (Typical Sampling) 原理演示")
    print("=" * 65)

    # 计算信息量
    info_content = -torch.log(probs)
    mean_ic = info_content.mean().item()
    std_ic = info_content.std().item()

    print(f"\n{'Token':<8} {'P(x)':>8} {'信息量':>10} {'是否保留':>10}")
    print("-" * 45)

    tau = 1.0  # 典型性阈值
    retained = []
    filtered_out = []

    for i, (tok, p, ic) in enumerate(zip(tokens, probs, info_content)):
        is_retained = abs(ic.item() - mean_ic) <= tau * std_ic
        status = "✅ 保留" if is_retained else "❌ 过滤"
        print(f"{tok:<8} {p.item():>8.3f} {ic.item():>10.4f} {status:>10}")

        if is_retained:
            retained.append((tok, p))
        else:
            filtered_out.append((tok, p))

    print(f"\n📊 统计:")
    print(f"   信息量均值: {mean_ic:.4f}")
    print(f"   信息量标准差: {std_ic:.4f}")
    print(f"   阈值 τ = {tau}")
    print(f"   保留范围: [{mean_ic - tau*std_ic:.4f}, {mean_ic + tau*std_ic:.4f}]")
    print(f"\n   保留 {len(retained)} 个 token: {[r[0] for r in retained]}")
    print(f"   过滤 {len(filtered_out)} 个 token: {[f[0] for f in filtered_out]}")

    print(f"\n💡 直觉:")
    print(f"   • 'the'(p=0.35) 太可预测 → 信息量低 → 可能被过滤")
    print(f"   • 'xyz'(p=0.005) 太罕见 → 信息量高 → 可能被过滤")
    print(f"   • 中间概率的 token → 信息量适中 → 保留用于采样")

typical_sampling_demo()
```

## 2.2 HF 中的 Typical Sampling

```python
def typical_sampling_with_hf():
    """使用 HF 的 typical sampling 参数"""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "In the world of technology,"
    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        typical_p=1.0,             # 典型性阈值 (HF 参数名: typical_p)
        temperature=0.8,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n[Typical Sampling] 结果:\n{text}")

typical_sampling_with_hf()
```

---

## 三、Min-P 采样：新的绝对概率下界方法

## 3.1 为什么需要 Min-P？

Top-P（Nucleus Sampling）的问题在于它基于**相对概率**来截断——累积概率达到阈值就停止。但有时候我们希望用一个**绝对的**概率门槛：任何 token 如果其原始概率低于某个最小值就直接排除。

Min-P（2024 年提出）就是这样做的：

$$\text{保留条件}: P(token) \geq \min\_p \times P(token_{max})$$

其中 $P(token_{max})$ 是最高概率 token 的概率，$\min\_p$ 是超参数（推荐 0.05~0.1）。

```python
def min_p_sampling_demo():
    """Min-P 采样原理与实现"""

    import torch
    import torch.nn.functional as F

    # 模拟 logits
    logits = torch.tensor([5.0, 3.5, 2.0, 1.0, 0.5, -0.5, -1.5, -3.0])
    tokens = ["jumps", "runs", "walks", ".", ",", "the", "quickly", "!"]
    probs = F.softmax(logits, dim=-1)

    min_p_values = [0.0, 0.05, 0.1, 0.2]

    print("=" * 70)
    print("Min-P 采样: 不同阈值下的过滤效果")
    print("=" * 70)
    print(f"\n原始概率分布:")
    for t, p in zip(tokens, probs):
        bar = "█" * int(p * 50)
        print(f"  {t:<10} {p:.4f} {bar}")

    print()

    for min_p in min_p_values:
        threshold = min_p * probs.max().item()
        mask = probs >= threshold
        retained_probs = probs[mask]
        retained_tokens = [t for t, m in zip(tokens, mask) if m]

        print(f"[min_p={min_p:.2f}] 阈值={threshold:.4f} | "
              f"保留 {mask.sum().item()}/{len(tokens)} 个 | "
              f"{retained_tokens}")

min_p_sampling_demo()
```

---

## 四、约束 Decoding：强制输出符合规则

## 4.1 为什么需要约束？

在实际应用中，我们经常需要模型的输出满足特定的格式或语法要求：

```python
def constrained_decoding_scenarios():
    """约束解码的应用场景"""

    scenarios = [
        {
            "场景": "结构化数据提取",
            "需求": "输出必须是合法 JSON 格式",
            "示例": '{"name": "张三", "age": 25, "city": "北京"}',
        },
        {
            "场景": "SQL 生成",
            "需求": "输出必须是可执行的 SQL 语句",
            "示例": "SELECT * FROM users WHERE age > 18;",
        },
        {
            "场景": "代码补全",
            "需求": "必须符合编程语言语法",
            "示例": "def fibonacci(n):\n    if n <= 1:\n        return n",
        },
        {
            "场景": "安全过滤",
            "需求": "不能包含敏感词或有害内容",
            "示例": "自动跳过违禁词汇",
        },
        {
            "场景": "格式模板填充",
            "需求": "必须遵循预定义模板",
            "示例": "【产品名称】XXX，价格¥XXX，评分X.X星",
        },
    ]

    print("=" * 70)
    print("约束解码的应用场景")
    print("=" * 70)
    for s in scenarios:
        print(f"\n📌 {s['场景']}")
        print(f"   需求: {s['需求']}")
        print(f"   示例: {s['示例']}")

constrained_decoding_scenarios()
```

## 4.2 Logits Processor：自定义约束的核心接口

Hugging Face 的 `generate()` 方法支持通过 `logits_processor` 参数注入自定义逻辑。这是实现约束解码的标准方式：

```python
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseLogitsProcessor(ABC):
    """Logits Processor 基类"""

    @abstractmethod
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids: 已生成的 token 序列 (batch, seq_len)
            scores: 当前步的 logits (batch, vocab_size)
        Returns:
            处理后的 scores (修改后返回)
        """
        pass


class ForbiddenWordsProcessor(BaseLogitsProcessor):
    """
    禁止词处理器
    将指定 token 的 logit 设为 -inf, 使其永远不会被选中
    """

    def __init__(self, forbidden_words: List[str], tokenizer: PreTrainedTokenizerBase):
        self.forbidden_token_ids = set()
        for word in forbidden_words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.forbidden_token_ids.update(ids)

        print(f"🚫 禁止词处理器初始化: "
              f"禁止 {len(forbidden_words)} 个词 → {len(self.forbidden_token_ids)} 个 token ID")

    def __call__(self, input_ids, scores):
        for tid in self.forbidden_token_ids:
            if tid < scores.shape[-1]:
                scores[:, tid] = float('-inf')
        return scores


class JSONFormatProcessor(BaseLogitsProcessor):
    """
    JSON 格式约束处理器
    引导输出朝向合法 JSON 结构
    (简化版: 实际完整版需要维护解析栈状态)
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        # JSON 合法字符的 token ID 集合
        json_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                         '0123456789{}[]\":,._-+ \t\n')
        self.allowed_prefixes = set()

    def __call__(self, input_ids, scores):
        # 这是一个简化版本 —— 实际的 JSON 约束需要更复杂的状态机
        # 这里仅展示接口用法
        return scores


class RegexPatternProcessor(BaseLogitsProcessor):
    """
    正则表达式约束处理器
    确保生成的文本匹配给定的正则模式
    """

    def __init__(self, pattern: str, tokenizer: PreTrainedTokenizerBase):
        import re
        self.pattern = re.compile(pattern)
        self.tokenizer = tokenizer

    def _get_allowed_tokens(self, generated_text: str) -> set:
        """根据已生成文本和正则模式, 确定下一步允许的 token"""
        allowed = set(range(len(self.tokenizer)))  # 默认全部允许
        # 实际实现需要使用有限自动机 (DFA) 来高效计算
        return allowed

    def __call__(self, input_ids, scores):
        generated_text = self.tokenizer.decode(input_ids[0])
        allowed_tokens = self._get_allowed_tokens(generated_text)

        mask = torch.full_like(scores, float('-inf'))
        for tid in allowed_tokens:
            if tid < scores.shape[-1]:
                mask[:, tid] = scores[:, tid]

        return mask


def demo_logits_processors():
    """演示 Logits Processor 的使用"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Please generate a response without using these words:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # 创建处理器链
    processors = [
        ForbiddenWordsProcessor(["bad", "terrible", "awful", "horrible"], tokenizer),
    ]

    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        logits_processor=processors,  # 关键参数!
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n[带约束的结果]:\n{text}")

demo_logits_processors()
```

## 4.3 Outlines 库：生产级约束解码

对于复杂的约束（如完整的 JSON Schema、SQL 语法等），手动编写 Logits Processor 非常困难。推荐使用 **Outlines** 这个专门库：

```bash
pip install outlines
```

```python
import outlines

def demo_outlines_constraints():
    """Outlines 库的约束解码演示"""

    model = outlines.models.transformers("gpt2")

    # 正则约束: 必须是邮箱地址格式
    email_generator = outlines.generate.regex(
        model,
        r"[a-z]+@[a-z]+\.[a-z]+",
    )

    result = email_generator("Email address: ")
    print(f"[Regex 约束] {result}")

    # JSON 约束: 输出必须是合法 JSON
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    json_generator = outlines.generate.json(model, json_schema)
    result = json_generator("Generate a person: ")
    print(f"[JSON 约束] {result}")

# demo_outlines_constraints()  # 需要 outlines 库
```

---

## 五、Diverse Beam Search：让 Beam Search 不再单调

## 5.1 问题

标准 Beam Search 的多个 beam 最终往往产生非常相似的输出——因为它们都追求同一个"最优"目标。Diverse Beam Search 通过引入**分组和多样性惩罚**来解决这个问题。

## 5.2 实现思路

```
Standard Beam Search:
  Beam 1: "The cat sat on the mat"
  Beam 2: "The cat sat on the rug"     ← 和 Beam 1 很像!
  Beam 3: "The cat sat on the bed"     ← 也差不多!

Diverse Beam Search (groups=3, diversity=0.7):
  Group 1: 追求最高概率 → "The cat sat on the mat"
  Group 2: 惩罚与 G1 相似的 → "A dog played in the park"  ← 差异大!
  Group 3: 惩罚与 G1,G2 都相似的 → "She opened her book"  ← 完全不同!
```

```python
def diverse_beam_search_demo():
    """Diverse Beam Search 效果展示"""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("=" * 65)
    print("Standard Beam Search vs Diverse Beam Search")
    print("=" * 65)
    print(f"\nPrompt: '{prompt}'\n")

    # Standard Beam Search
    print("[Standard Beam Search, num_beams=5, num_return_sequences=3]")
    standard_outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=5,
        num_return_sequences=3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    for i, out in enumerate(standard_outputs, 1):
        text = tokenizer.decode(out, skip_special_tokens=True)[len(prompt):]
        print(f"  #{i}: {text}")

    print()

    # Diverse Beam Search
    print("[Diverse Beam Search, num_beams=5, num_beam_groups=3, diversity=0.7]")
    diverse_outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=9,                   # 总 beam 数 = groups × beams_per_group
        num_beam_groups=3,            # 分成 3 组
        diversity_penalty=0.7,        # 组间多样性惩罚
        num_return_sequences=3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    for i, out in enumerate(diverse_outputs, 1):
        text = tokenizer.decode(out, skip_special_tokens=True)[len(prompt):]
        print(f"  #{i}: {text}")

diverse_beam_search_demo()
```

---

## 六、各种解码策略的选择指南

## 6.1 决策树

```python
def decoding_strategy_selection():
    """解码策略选择决策树"""

    print("=" * 75)
    print("解码策略选择决策树")
    print("=" * 75)

    decisions = [
        {
            "问题": "任务是否有唯一正确的答案?",
            "是 →": "翻译 / 摘要 / 数据格式化 → 用 Beam Search (width=3~5)",
            "否 ↓": "",
        },
        {
            "问题": "是否需要严格的输出格式?",
            "是 →": "JSON / SQL / 正则 → 用 Constrained Decoding (outlines 库)",
            "否 ↓": "",
        },
        {
            "问题": "最看重什么?",
            "质量": "对比搜索 (Contrastive Search) — 平衡概率和多样性",
            "多样性": "Temperature Sampling (T=0.7~1.0) + Top-P",
            "速度": "Greedy Search (最快但最无趣)",
            "确定性": "Beam Search (width=1 即 Greedy)",
        },
    ]

    for d in decisions:
        print(f"\n❓ {d['问题']}")
        if d.get("是 →"):
            print(f"   ✅ {d['是 →']}")
        if d.get("否 ↓"):
            print(f"   ⬇️  继续判断...")
        if "质量" in d:
            print(f"   🎯 质量: {d['quality']}")
            print(f"   🎲 多样性: {d['多样性']}")
            print(f("   ⚡ 速度: {d['速度']}"))
            print(f("   🔒 确定性: {d['确定性']}"))

decoding_strategy_selection()
```

## 6.2 全面对比表

```python
def full_comparison_table():
    """解码策略全面对比"""

    strategies = [
        ("Greedy", "★☆☆☆☆", "★★★★★", "★★★★★", "极快", "重复循环"),
        ("Beam-3", "★★★☆☆", "★☆☆☆☆", "★★★★☆", "快", "缺乏多样性"),
        ("Beam-5+", "★★★★☆", "★☆☆☆☆", "★★★☆☆", "中等", "偏向通用内容"),
        ("Temp Sample", "★★★☆☆", "★★★★★", "★★★☆☆", "快", "可能不连贯"),
        ("Top-P", "★★★★☆", "★★★★☆", "★★★★☆", "快", "推荐默认方案"),
        ("Contrastive", "★★★★★", "★★★★☆", "★★★☆☆", "较慢", "新颖但需调参"),
        ("Typical", "★★★★☆", "★★★★☆", "★★★★☆", "快", "信息量适中"),
        ("Constrained", "N/A", "★☆☆☆☆", "★★★★★", "取决于约束", "严格合规"),
    ]

    print("\n" + "=" * 85)
    print("解码策略全面对比表")
    print("=" * 85)
    print(f"\n{'策略':<14} {'质量':<8} {'多样性':<8} {'速度':<8} {'适用'}")
    print("-" * 75)
    for s in strategies:
        print(f"{s[0]:<14} {s[1]:<8} {s[2]:<8} {s[3]:<8} {s[4]}")

full_comparison_table()
```

---

## 七、本章小结

这一节我们在上一节的基础上，深入了几种更高级的解码策略：

| 方法 | 核心机制 | 解决的问题 | 推荐场景 |
|------|---------|-----------|---------|
| **对比搜索** | 概率 + 表示相似度的权衡 | 重复循环 + 保持质量 | 开放域长文本生成 |
| **典型采样** | 基于信息量的自适应过滤 | 极端概率 token | 通用高质量生成 |
| **Min-P** | 绝对概率下界 | Top-P 的替代方案 | 新兴方法，值得关注 |
| **约束 Decoding** | Logits Processor / Outlines | 格式合规性 | 结构化输出 |
| **Diverse Beam** | 分组 + 组间多样性惩罚 | Beam Search 单调性 | 需要多样候选时 |

**核心要点**：
1. **没有银弹**——每种策略都有其适用范围和局限性
2. **组合使用**——实际生产中通常是 Temperature + Top-P + Repetition Penalty 的组合
3. **约束解码是结构化任务的必备技能**——学会 Logits Processor 接口
4. **对比搜索是一个很有潜力的方向**——在质量和多样性之间取得了很好的平衡

下一节我们将学习 **长度控制、重复控制与停止条件**——如何精确控制生成过程的边界。
