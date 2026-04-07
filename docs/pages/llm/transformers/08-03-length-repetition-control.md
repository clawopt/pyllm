# 长度控制、重复控制与停止条件

## 这一节讲什么？

在实际使用大模型生成文本时，你几乎一定会遇到这些问题：

1. **生成太长或太短**——模型不知道什么时候该停下来
2. **反复说同一句话**——陷入"车轱辘话"的死循环
3. **在错误的地方截断**——句子没说完就断了
4. **需要精确的字数/词数控制**——比如"写一段 200 字的摘要"

上一节我们学习了各种解码策略的"怎么选下一个 token"，这一节我们关注的是**何时停止、如何防止重复、如何精确控制长度**。这些是让生成结果从"能用"到"好用"的关键细节。

---

## 一、长度控制的完整工具箱

## 1.1 max_new_tokens vs max_length：一个常见的坑

这是新手最容易混淆的两个参数，让我们彻底搞清楚：

```python
def max_tokens_gotcha():
    """max_new_tokens vs max_length 的区别演示"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    long_prompt = "Artificial intelligence (AI) is intelligence demonstrated by machines, " \
                   "as opposed to natural intelligence displayed by animals including humans. " \
                   "AI research has been defined as the field of study of intelligent agents, " \
                   "which refers to any system that perceives its environment and takes actions " \
                   "that maximize its chance of achieving its goals."

    inputs = tokenizer(long_prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    print("=" * 70)
    print("max_new_tokens vs max_length 的致命区别")
    print("=" * 70)
    print(f"\nPrompt 长度: {prompt_len} tokens")

    # ❌ 错误做法: 用 max_length
    print(f"\n❌ [错误] max_length=60:")
    output_wrong = model.generate(
        **inputs,
        max_length=60,           # 包含 prompt 的总长度!
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_len = output_wrong.shape[1] - prompt_len
    text_wrong = tokenizer.decode(output_wrong[0], skip_special_tokens=True)
    print(f"   实际新生成: {generated_len} tokens (不是 60!)")
    print(f"   结果: {text_wrong[-80:]}...")

    # ✅ 正确做法: 用 max_new_tokens
    print(f"\n✅ [正确] max_new_tokens=50:")
    output_correct = model.generate(
        **inputs,
        max_new_tokens=50,       # 只限制新生的 token 数!
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_len = output_correct.shape[1] - prompt_len
    text_correct = tokenizer.decode(output_correct[0], skip_special_tokens=True)
    print(f"   实际新生成: {generated_len} tokens ✓")
    print(f"   结果: {text_correct[len(long_prompt):]}...")

    print("\n💡 规则: 永远优先使用 max_new_tokens!")
    print("   max_length = len(prompt) + max_new_tokens")
    print("   如果 prompt 很长, max_length 可能导致只生成 1~2 个字!")

max_tokens_gotcha()
```

## 1.2 min_new_tokens / min_length：强制最小长度

有时候模型会过早停止（比如生成了一个句号就结束了），`min_new_tokens` 可以强制它至少生成一定数量的 token：

```python
def demo_min_tokens():
    """min_new_tokens 的作用"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Q: What is 2+2?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("=" * 65)
    print("min_new_tokens 效果对比")
    print("=" * 65)
    print(f"\nPrompt: '{prompt}'")

    # 不设 min_new_tokens
    print(f"\n[无约束] 可能过早结束:")
    out1 = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
    print(f"  {tokenizer.decode(out1[0], skip_special_tokens=True)[len(prompt):]}")

    # 强制至少生成 10 个 token
    print(f"\n[min_new_tokens=10] 强制最短长度:")
    out2 = model.generate(**inputs, max_new_tokens=30, min_new_tokens=10,
                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(f"  {tokenizer.decode(out2[0], skip_special_tokens=True)[len(prompt):]}")

demo_min_tokens()
```

## 1.3 Length Penalty 深入解析

我们在 Beam Search 中已经接触过 `length_penalty`，但它的行为值得更深入地理解：

$$adjusted\_score = \frac{cumulative\_log\_probability}{|sequence|^{\alpha}}$$

```python
def length_penalty_deep_dive():
    """Length Penalty 的数学和行为分析"""

    import math

    # 假设有两个候选序列
    sequences = [
        {"text": "It is good.", "log_prob": -3.5, "len": 4},
        {"text": "It is a very good thing indeed.", "log_prob": -8.2, "len": 7},
    ]

    print("=" * 70)
    print("Length Penalty 数学原理")
    print("=" * 70)

    for seq in sequences:
        print(f"\n序列: \"{seq['text']}\"")
        print(f"  原始得分 (log prob): {seq['log_prob']}")
        print(f"  长度: {seq['len']} tokens")

        for alpha in [0.8, 1.0, 1.2, 1.5, 2.0]:
            adjusted = seq['log_prob'] / (seq['len'] ** alpha)
            print(f"  α={alpha:.1f}: 调整后得分 = {seq['log_prob']} / {seq['len']}^{α} = {adjusted:.4f}")

    print("\n💡 解读:")
    print("   α < 1 → 偏好短序列 (除数增长慢)")
    print("   α = 1 → 线性惩罚 (标准)")
    print("   α > 1 → 偏好长序列 (短序列被重度惩罚)")

length_penalty_deep_dive()
```

## 1.4 精确字数控制：自定义 StoppingCriteria

当内置参数不够用时（比如需要"恰好 200 字"），可以通过 `StoppingCriteria` 完全自定义停止逻辑：

```python
from transformers import StoppingCriteriaList, StoppingCriteria
import torch


class WordCountStoppingCriteria(StoppingCriteria):
    """
    自定义停止条件: 达到目标字数时停止
    """

    def __init__(self, target_word_count: int, tokenizer):
        self.target = target_word_count
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        word_count = len(generated_text.split())
        return word_count >= self.target


class ExactTokenCountStoppingCriteria(StoppingCriteria):
    """
    自定义停止条件: 精确控制生成的 token 数量
    配合 eos_token_id 使用可以实现精确截断
    """

    def __init__(self, initial_length: int, target_new_tokens: int):
        self.initial_length = initial_length
        self.target_new_tokens = target_new_tokens

    def __call__(self, input_ids, scores, **kwargs):
        current_new_tokens = input_ids.shape[1] - self.initial_length
        return current_new_tokens >= self.target_new_tokens


class MultiKeywordStoppingCriteria(StoppingCriteria):
    """
    多关键词停止条件: 遇到任一指定关键词就停止
    """

    def __init__(self, keywords: list, tokenizer):
        self.keywords = keywords
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(kw in generated_text for kw in self.keywords)


def demo_custom_stopping():
    """演示自定义停止条件"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Write a short story about a robot"
    inputs = tokenizer(prompt, return_tensors="pt")
    initial_len = inputs["input_ids"].shape[1]

    print("=" * 65)
    print("自定义停止条件演示")
    print("=" * 65)

    # 示例 1: 字数控制
    print("\n[示例 1] 目标 40 字:")
    stopping_criteria = StoppingCriteriaList([
        WordCountStoppingCriteria(target_word_count=40, tokenizer=tokenizer),
    ])
    output = model.generate(
        **inputs,
        max_new_tokens=100,              # 上限设高一些
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
    word_count = len(result.split())
    print(f"  生成结果 ({word_count} 字):\n  {result}")

    # 示例 2: 关键词停止
    print("\n[示例 2] 遇到 'The End' 就停止:")
    stopping_criteria2 = StoppingCriteriaList([
        MultiKeywordStoppingCriteria(["The End", "END", "***"], tokenizer),
    ])
    output2 = model.generate(
        **inputs,
        max_new_tokens=100,
        stopping_criteria=stopping_criteria2,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )
    result2 = tokenizer.decode(output2[0], skip_special_tokens=True)[len(prompt):]
    print(f"  生成结果:\n  {result2}")


if __name__ == "__main__":
    demo_custom_stopping()
```

---

## 二、重复控制：打破死循环的艺术

## 2.1 为什么模型会重复？

重复是大语言模型生成中最常见也最令人头疼的问题。根本原因有几个：

```python
def analyze_repetition_causes():
    """分析模型重复生成的根本原因"""

    causes = [
        {
            "原因": "Attention Sink",
            "解释": "某些 token (如句首) 会持续吸引注意力,"
                   "导致模型反复回到相同的表达模式",
            "类比": "就像人说话时也会不自觉地重复口头禅",
        },
        {
            "原因": "训练数据的重复模式",
            "解释": "如果训练数据中某种句式大量出现,"
                   "模型就会学会这种'安全但无聊'的表达",
            "例子": "新闻数据集中 'According to reports' 出现了上万次",
        },
        {
            "原因": "解码策略问题",
            "解释": "Greedy Search 天然容易陷入局部最优循环;"
                   "没有 repetition_penalty 时采样也可能选到已出现的词",
        },
        {
            "原因": "上下文窗口内的自增强",
            "解释": "一旦生成了某个短语, 它就成为新的上下文的一部分,"
                   "模型看到自己说过的话后可能再次生成类似内容",
        },
    ]

    print("=" * 75)
    print("模型重复生成的四大原因")
    print("=" * 75)
    for c in causes:
        print(f"\n🔍 {c['原因']}")
        print(f"   解释: {c['解释']}")
        if '类比' in c:
            print(f"   类比: {c['类比']}")
        if '例子' in c:
            print(f"   例子: {c['例子']}")

analyze_repetition_causes()
```

## 2.2 repetition_penalty：基础武器

`repetition_penalty` 是 HF `generate()` 方法中最常用的防重复机制：

$$\text{如果 } logit > 0: \quad new\_logit = \frac{logit}{penalty}$$
$$\text{如果 } logit < 0: \quad new\_logit = logit \times penalty$$

注意这个**非对称设计**：正 logits 被缩小（概率降低），负 logits 被放大（绝对值增大，概率进一步降低）。无论哪种情况，已经出现过的 token 被再次选中的概率都会降低。

```python
def repetition_penalty_demo():
    """repetition_penalty 的效果演示"""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "I think that artificial intelligence will"
    inputs = tokenizer(prompt, return_tensors="pt")

    penalties = [1.0, 1.05, 1.15, 1.25, 1.5, 2.0]

    print("=" * 70)
    print("Repetition Penalty 效果对比")
    print("=" * 70)
    print(f"\nPrompt: '{prompt}'\n")

    for penalty in penalties:
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.92,
            repetition_penalty=penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
        
        # 计算去重率
        words = text.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)

        status = "✅" if unique_ratio > 0.75 else ("⚠️" if unique_ratio > 0.5 else "❌")
        print(f"[{status} penalty={penalty:.2f}] 去重率={unique_ratio:.0%}")
        print(f"  {text.strip()}\n")

repetition_penalty_demo()
```

## 2.3 no_repeat_ngram_size：更强的 N-Gram 级别防重复

`repetition_penalty` 只作用于单个 token，而 `no_repeat_ngram_size` 可以防止**整个 n-gram 短语**的重复：

```python
def ngram_repeat_control():
    """N-Gram 级别的重复控制"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "The key to success is"
    inputs = tokenizer(prompt, return_tensors="pt")

    ngram_sizes = [0, 2, 3, 4]

    print("=" * 65)
    print("no_repeat_ngram_size 效果对比")
    print("=" * 65)
    print(f"\nPrompt: '{prompt}'\n")

    for size in ngram_sizes:
        label = "无限制" if size == 0 else f"N={size}"
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.9,
            no_repeat_ngram_size=size,
            repetition_penalty=1.0 if size > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
        print(f"[{label}] {text.strip()}\n")

ngram_repeat_control()
```

**注意事项**：
- `no_repeat_ngram_size=3` 意味着任何 3-token 组合不会出现两次
- 值太大会严重限制表达能力（比如不能两次使用 "in the"）
- 推荐值：2~4，通常 3 是最佳平衡点

## 2.4 frequency_penalty 和 presence_penalty：OpenAI 风格的精细控制

除了 `repetition_penalty`，还有两种更细粒度的惩罚方式：

```python
def explain_frequency_vs_presence_penalty():
    """frequency_penalty vs presence_penalty 的区别"""

    penalties = [
        {
            "名称": "repetition_penalty (HF)",
            "机制": "对每个已出现的 token 统一惩罚",
            "特点": "简单粗暴; 不管出现了多少次都一样惩罚",
            "公式": "logit > 0 时 /p, logit < 0 时 ×p",
            "推荐值": "1.1 ~ 1.25",
        },
        {
            "名称": "frequency_penalty (OpenAI)",
            "机制": "按出现次数线性累加惩罚",
            "特点": "出现越多惩罚越重; 第 2 次比第 1 次惩罚更大",
            "公式": "logit -= count × penalty_value",
            "推荐值": "0.0 ~ 2.0 (默认 0)",
        },
        {
            "名称": "presence_penalty (OpenAI)",
            "机制": "只要出现过就惩罚一次 (不管几次)",
            "特点": "介于 repetition 和 frequency 之间",
            "公式": "logit -= (count > 0 ? penalty_value : 0)",
            "推荐值": "0.0 ~ 2.0 (默认 0)",
        },
    ]

    print("=" * 80)
    print("三种重复惩罚机制对比")
    print("=" * 80)

    for p in penalties:
        print(f"\n📐 {p['名称']}")
        for k, v in p.items():
            if k != "名称":
                print(f"   {k}: {v}")

    print("\n💡 选择建议:")
    print("   • 一般场景 → repetition_penalty=1.15 (最简单)")
    print("   • 严重重复 → frequency_penalty + presence_penalty 组合")
    print("   • 需要精细控制 → 自定义 LogitsProcessor")

explain_frequency_vs_presence_penalty()
```

## 2.5 自定义 Logits Processor 实现高级防重复

对于更复杂的防重复需求，我们可以编写自定义的 Logits Processor：

```python
import torch
from transformers import PreTrainedTokenizerBase


class ExponentialDecayPenaltyProcessor:
    """
    指数衰减惩罚处理器
    
    已出现的 token 惩罚力度随出现次数指数增长:
      第 1 次: 不惩罚
      第 2 次: × 0.5
      第 3 次: × 0.25
      第 4 次: × 0.125
      ...
    """

    def __init__(self, decay_base: float = 0.5, tokenizer=None):
        self.decay_base = decay_base
        self.tokenizer = tokenizer
        self.token_counts = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        batch_size = input_ids.shape[0]

        for b in range(batch_size):
            seq = input_ids[b].tolist()

            # 统计当前序列中各 token 出现次数
            counts = {}
            for tid in seq:
                counts[tid] = counts.get(tid, 0) + 1

            # 对每个已出现的 token 应用指数衰减惩罚
            for tid, count in counts.items():
                if tid < scores.shape[-1]:
                    penalty = self.decay_base ** (count - 1)  # 第 1 次: decay^0 = 1
                    scores[b, tid] *= penalty

        return scores


class ContextWindowPenaltyProcessor:
    """
    滑动窗口内的高频惩罚
    
    只对最近 N 个 token 内的重复进行惩罚
    (避免整个历史都被惩罚导致无法使用常用词)
    """

    def __init__(self, window_size: int = 20, penalty: float = 1.5):
        self.window_size = window_size
        self.penalty = penalty

    def __call__(self, input_ids, scores):

        for b in range(input_ids.shape[0]):
            recent_tokens = input_ids[b, -self.window_size:].tolist() if input_ids.shape[1] >= self.window_size else input_ids[b].tolist()

            unique_recent = set(recent_tokens)
            for tid in unique_recent:
                count_in_window = recent_tokens.count(tid)
                if count_in_window > 1 and tid < scores.shape[-1]:
                    scores[b, tid] /= (self.penalty ** (count_in_window - 1))

        return scores


def demo_custom_penalty_processors():
    """演示自定义惩罚处理器"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "I believe that we should always strive to"
    inputs = tokenizer(prompt, return_tensors="pt")

    processors = [
        ExponentialDecayPenaltyProcessor(decay_base=0.6, tokenizer=tokenizer),
    ]

    output = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.85,
        top_k=50,
        top_p=0.93,
        logits_processor=processors,
        pad_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n[指数衰减惩罚] 结果:\n{result}")

demo_custom_penalty_processors()
```

---

## 三、EOS 控制与异常处理

## 3.1 EOS Token 处理

EOS（End of Sequence）token 是模型表示"我说完了"的特殊标记。正确处理 EOS 至关重要：

```python
def eos_handling_guide():
    """EOS Token 处理指南"""

    guide = [
        {
            "问题": "模型不输出 EOS (一直生成到 max_new_tokens)",
            "原因": "(1) 训练数据中 EOS 标注不一致;"
                   "(2) 生成的内容还没'说完';"
                   "(3) EOS token ID 设置错误",
            "解决": "设置合理的 max_new_tokens 作为保底;"
                   "检查 eos_token_id 是否正确;"
                   "考虑使用 StoppingCriteria",
        },
        {
            "问题": "模型过早输出 EOS (内容没说完就停了)",
            "原因": "训练数据中很多短样本导致模型学会了'早停';"
                   "或者 prompt 本身触发了 EOS 模式",
            "解决": "使用 min_new_tokens 强制最短长度;"
                   "设置 suppress_tokens=[eos_id] 在前 N 步禁止 EOS",
        },
        {
            "问题": "生成结果包含 EOS 后面的乱码",
            "原因": "decode 时没有正确跳过 EOS 及其后的 padding",
            "解决": "skip_special_tokens=True (推荐); "
                   "或在 decode 后手动截断到第一个 EOS",
        },
        {
            "问题": "多轮对话时 EOS 导致后续输入混乱",
            "原因": "对话历史中的 EOS 被当作真实结束信号",
            "解决": "在拼接对话历史时替换/移除中间的 EOS;"
                   "只在最后保留一个真正的 EOS",
        },
    ]

    print("=" * 78)
    print("EOS Token 处理指南")
    print("=" * 78)
    for g in guide:
        print(f"\n⚠️  {g['问题']}")
        print(f"   原因: {g['原因']}")
        print(f"   解决: {g['解决']}")

eos_handling_guide()
```

## 3.2 suppress_tokens：强制禁止特定 token

有时候我们需要完全禁止某些 token 出现在输出中：

```python
def demo_suppress_tokens():
    """suppress_tokens 的用法"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Tell me something interesting:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # 找出所有标点符号的 token IDs (简化版)
    punctuation_tokens = []
    for tok_string in [".", ",", "!", "?", ";", ":"]:
        ids = tokenizer.encode(tok_string, add_special_tokens=False)
        punctuation_tokens.extend(ids)

    print("=" * 65)
    print("suppress_tokens: 禁止标点符号输出")
    print("=" * 65)
    print(f"\n禁止的 token IDs: {punctuation_tokens}")

    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        suppress_tokens=punctuation_tokens,  # 关键!
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n[禁止标点后的输出]:\n{text}")

demo_suppress_tokens()
```

---

## 四、生产环境 Checklist

## 4.1 生成参数配置模板

```python
def production_generation_configs():
    """生产环境的生成参数模板"""

    configs = {
        "对话系统 (Chat)": {
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.92,
            "repetition_penalty": 1.15,
            "max_new_tokens": 2048,
            "min_new_tokens": 10,
        },

        "代码生成 (Code)": {
            "do_sample": True,
            "temperature": 0.3,
            "top_k": 30,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 4,
            "max_new_tokens": 1024,
        },

        "文章摘要 (Summary)": {
            "num_beams": 4,
            "do_sample": False,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "max_new_tokens": 256,
        },

        "翻译 (Translation)": {
            "num_beams": 5,
            "do_sample": False,
            "length_penalty": 0.6,
            "early_stopping": True,
            "max_new_tokens": 512,
        },

        "创意写作 (Creative)": {
            "do_sample": True,
            "temperature": 0.95,
            "top_k": 100,
            "top_p": 0.96,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 2048,
        },
    }

    print("=" * 80)
    print("生产环境生成参数配置模板")
    print("=" * 80)

    for name, params in configs.items():
        print(f"\n📋 {name}:")
        for k, v in params.items():
            print(f"   {k}: {v}")

production_generation_configs()
```

## 4.2 常见问题速查表

```python
def quick_troubleshooting():
    """生成问题快速排查"""

    issues = [
        ("输出全是重复", "加 repetition_penalty=1.15; 加 no_repeat_ngram_size=3"),
        ("输出太短", "检查 min_new_tokens; 检查是否提前遇到 EOS"),
        ("输出太长/不停止", "设置合理的 max_new_tokens; 检查 eos_token_id"),
        ("输出格式不对", "使用 Constrained Decoding / Outlines 库"),
        ("输出包含乱码", "确认 skip_special_tokens=True; 检查 tokenizer 匹配"),
        ("每次生成都一样", "设置 do_sample=True; 增加 temperature"),
        ("生成质量差", "调整 temperature (0.7~1.0); 尝试 Top-P 或对比搜索"),
        ("速度太慢", "减小 num_beams; 减小 max_new_tokens; 使用 vLLM/TGI"),
        ("显存 OOM", "减少 batch_size; 使用量化模型; 启用 gradient_checkpointing"),
    ]

    print("=" * 70)
    print("生成问题快速排查表")
    print("=" * 70)
    print(f"\n{'现象':<20} {'解决方案'}")
    print("-" * 70)
    for issue, solution in issues:
        print(f"{issue:<20} {solution}")

quick_troubleshooting()
```

---

## 五、本章小结

这一节我们深入学习了生成过程中的三大控制维度：

| 维度 | 核心参数 | 机制 | 典型值 |
|------|---------|------|--------|
| **长度控制** | max_new_tokens | 硬性上限 | 50~2048 |
| | min_new_tokens | 硬性下限 | 1~20 |
| | length_penalty | Beam Search 软性调节 | 0.6~2.0 |
| | StoppingCriteria | 完全自定义 | 按需 |
| **重复控制** | repetition_penalty | 单 token 惩罚 | 1.1~1.25 |
| | no_repeat_ngram_size | N-gram 短语防重复 | 2~4 |
| | frequency_penalty | 按次数累加惩罚 | 0~2 |
| | 自定义 Processor | 指数衰减/滑动窗口等 | 按需 |
| **停止条件** | eos_token_id | 正常终止信号 | 自动检测 |
| | suppress_tokens | 禁止特定 token | 按需 |
| | early_stopping | Beam Search 提前停 | True/False |

**核心要点**：
1. **永远用 max_new_tokens 而非 max_length**——这是最常见的低级错误
2. **repetition_penalty + no_repeat_ngram_size 是防重复的标准组合**
3. **StoppingCriteria 提供了无限的自定义空间**——字数控制、关键词检测等
4. **不同任务有不同的最优参数组合**——不要一套参数走天下

下一节我们将探索 **推测解码、Mixture of Experts 等高级生成技术**——这些是让生成速度提升数量级的前沿方法。
