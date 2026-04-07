# 自定义生成策略：Logits Processor 与 Guided Decoding

## 这一节讲什么？

前面四节我们学习了 Hugging Face `generate()` 方法提供的各种内置参数和策略。但在实际生产中，你经常会遇到内置功能无法满足的需求：

- 需要输出**严格的 JSON 格式**，但模型偶尔会漏掉一个逗号
- 需要**按特定模板填充**，比如"【产品名】XXX，价格¥XXX"
- 需要**实时过滤敏感词**，在生成过程中就拦截而非事后处理
- 需要**实现自定义的解码算法**，比如某种特殊的搜索策略

这一节我们将深入 `LogitsProcessor` 和 `StoppingCriteria` 这两个核心接口，学习如何完全掌控生成过程的每一个细节。

---

## 一、LogitsProcessor 接口深度解析

## 1.1 什么是 LogitsProcessor？

`LogitsProcessor` 是 Hugging Face 提供的**生成过程钩子（Hook）机制**。它在每个时间步、模型输出 logits 之后、采样之前被调用。你可以把它理解为"最后一道关卡"——在模型做出选择之前，你有机会修改它的"想法"。

```
完整生成循环 (每一步):

input_ids → [Model Forward] → raw_logits → [LogitsProcessor] → modified_logits
                                                    ↓
                                              [Softmax / Sampling]
                                                    ↓
                                              next_token
```

## 1.2 接口签名与约定

```python
from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Any, Optional


class BaseLogitsProcessor(ABC):
    """
    LogitsProcessor 的抽象基类
    
    关键约定:
    1. __call__ 接收 (input_ids, scores) 并返回修改后的 scores
    2. 不能改变 scores 的 shape 或 dtype
    3. 应该是可组合的 (多个 processor 可以链式调用)
    4. 必须支持 batch 操作 (scores 的第一维是 batch_size)
    """

    @abstractmethod
    def __call__(
        self,
        input_ids: torch.LongTensor,     # (batch, seq_len) 已生成的序列
        scores: torch.FloatTensor,         # (batch, vocab_size) 当前步的 logits
        **kwargs,
    ) -> torch.FloatTensor:
        """
        处理并返回修改后的 logits
        
        Args:
            input_ids: 已生成的 token ID 序列
            scores: 模型输出的原始 logits
            
        Returns:
            修改后的 scores (shape 不变!)
        """
        pass


class LogitsProcessorList(list):
    """LogitsProcessor 列表, 按顺序依次应用每个 processor"""

    def __call__(self, input_ids, scores, **kwargs):
        for processor in self:
            scores = processor(input_ids, scores, **kwargs)
        return scores
```

## 1.3 内置 Processors 源码解读

HF 内置了多个实用的 LogitsProcessor，理解它们的实现有助于你编写自己的：

```python
def show_builtin_processors():
    """展示 HF 内置的 LogitsProcessors 及其源码逻辑"""

    processors = [
        {
            "名称": "TemperatureLogitsWarper",
            "作用": "温度缩放",
            "核心代码": "scores = scores / temperature",
            "参数": "temperature (float)",
            "注意": "temperature=0 时退化为 argmax",
        },
        {
            "名称": "TopKLogitsWarper",
            "作用": "Top-K 截断",
            "核心代码": "indices_to_remove = scores < topk_values[..., -1, None]\n"
                       "scores[indices_to_remove] = -float('inf')",
            "参数": "top_k (int), filter_value (-inf)",
            "注意": "top_k=0 或 >= vocab_size 时不做任何事",
        },
        {
            "名称": "TopPLogitsWarper",
            "作用": "Top-P (Nucleus) 截断",
            "核心代码": "sorted_indices = argsort(scores, descending=True)\n"
                       "cumulative_probs = cumsum(softmax(sorted_scores))\n"
                       "remove_mask = cumulative_probs > top_p\n"
                       "scores[sorted_indices[remove_mask]] = -inf",
            "参数": "top_p (float), min_tokens_to_keep (int)",
            "注意": "保留至少 min_tokens_to_keep 个候选",
        },
        {
            "名称": "RepetitionPenaltyLogitsProcessor",
            "作用": "重复惩罚",
            "核心代码": "for token_id in seen_tokens:\n"
                       "    if scores[token_id] > 0:\n"
                       "        scores[token_id] /= penalty\n"
                       "    else:\n"
                       "        scores[token_id] *= penalty",
            "参数": "penalty (float)",
            "注意": "非对称处理正/负 logits",
        },
        {
            "名称": "NoRepeatNGramLogitsProcessor",
            "作用": "N-Gram 级别防重复",
            "核心代码": "检查最近 ngram_size-1 个 token + 候选 token 是否已出现\n"
                       "如果出现过 → 设置为 -inf",
            "参数": "ngram_size (int)",
            "注意": "需要维护历史 n-gram 集合, 有一定计算开销",
        },
        {
            "名称": "EncoderRepetitionPenaltyLogitsProcessor",
            "作用": "Encoder-Decoder 模型的编码器重复惩罚",
            "核心代码": "对 encoder_input_ids 中出现的 token 进行惩罚",
            "参数": "penalty (float)",
            "注意": "仅用于 Seq2Seq 模型",
        },
        {
            "名称": "ForcedBOSTokenLogitsProcessor",
            "作用": "强制第一个 token 为 BOS",
            "核心代码": "if step == 0:\n"
                       "    scores[:, bos_token_id] = float('inf')",
            "参数": "bos_token_id (int)",
            "注意": "自动添加, 用户通常不需要关心",
        },
        {
            "名称": "ForcedEOSTokenLogitsProcessor",
            "作用": "强制在 max_length 时输出 EOS",
            "核心代码": "if current_len >= max_length - 1:\n"
                       "    scores[:, eos_token_id] = float('inf')",
            "参数": "max_length, eos_token_id",
            "注意": "配合 early_stopping 使用",
        },
    ]

    print("=" * 80)
    print("HuggingFace 内置 LogitsProcessor 一览")
    print("=" * 80)

    for p in processors:
        print(f"\n🔧 {p['名称']}")
        print(f"   作用: {p['作用']}")
        print(f"   参数: {p['参数']}")
        print(f"   核心逻辑:")
        for line in p['核心代码'].split('\n'):
            print(f"      {line}")
        if '注意' in p:
            print(f"   ⚠️  {p['注意']}")

show_builtin_processors()
```

---

## 二、实战：构建完整的自定义 Processor 工具箱

## 2.1 敏感词过滤处理器

```python
import torch
from typing import Set, List
from transformers import PreTrainedTokenizerBase


class SensitiveWordFilterProcessor:
    """
    敏感词过滤器
    在生成过程中实时拦截敏感词汇
    """

    def __init__(self, sensitive_words: List[str], tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        
        # 将敏感词转换为 token ID 集合 (包含子词匹配)
        self.forbidden_token_ids: Set[int] = set()
        self.forbidden_prefixes: dict = {}  # prefix → set of forbidden next tokens

        for word in sensitive_words:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            
            # 单 token 敏感词
            if len(tokens) == 1:
                self.forbidden_token_ids.add(tokens[0])
            # 多 token 敏感词 (处理前缀匹配)
            elif len(tokens) > 1:
                for i in range(len(tokens) - 1):
                    prefix = tuple(tokens[:i+1])
                    if prefix not in self.forbidden_prefixes:
                        self.forbidden_prefixes[prefix] = set()
                    self.forbidden_prefixes[prefix].add(tokens[i+1])

        print(f"🛡️  敏感词过滤器初始化完成")
        print(f"   敏感词数量: {len(sensitive_words)}")
        print(f"   直接禁止的 token: {len(self.forbidden_token_ids)}")
        print(f"   前缀规则数: {len(self.forbidden_prefixes)}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()

            # 规则 1: 直接禁止的 token
            for tid in self.forbidden_token_ids:
                if tid < scores.shape[-1]:
                    scores[b, tid] = float('-inf')

            # 规则 2: 多 token 敏感词的前缀匹配
            for prefix_len in range(min(len(seq), 10), 0, -1):
                prefix = tuple(seq[-prefix_len:])
                if prefix in self.forbidden_prefixes:
                    for forbidden_next in self.forbidden_prefixes[prefix]:
                        if forbidden_next < scores.shape[-1]:
                            scores[b, forbidden_next] = float('-inf')
                    break  # 只检查最长匹配的前缀

        return scores

    def add_word(self, word: str):
        """动态添加敏感词"""
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            self.forbidden_token_ids.add(tokens[0])
        elif len(tokens) > 1:
            for i in range(len(tokens) - 1):
                prefix = tuple(tokens[:i+1])
                if prefix not in self.forbidden_prefixes:
                    self.forbidden_prefixes[prefix] = set()
                self.forbidden_prefixes[prefix].add(tokens[i+1])


def demo_sensitive_filter():
    """演示敏感词过滤"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    sensitive_words = ["bad", "terrible", "awful", "stupid"]
    
    processors = [SensitiveWordFilterProcessor(sensitive_words, tokenizer)]

    prompt = "Write a review of this product:"
    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        logits_processor=processors,
        pad_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n[带敏感词过滤的结果]:\n{result}")

demo_sensitive_filter()
```

## 2.2 格式模板引导处理器

```python
import re
import json
import torch
from transformers import PreTrainedTokenizerBase


class TemplateGuidedProcessor:
    """
    模板引导生成处理器
    强制输出符合预定义的文本模板
    """

    def __init__(self, template: str, tokenizer: PreTrainedTokenizerBase):
        """
        Args:
            template: 模板字符串, 使用 {field_name} 作为占位符
                     例如: "姓名:{name}, 年龄:{age}, 城市:{city}"
        """
        self.template = template
        self.tokenizer = tokenizer
        
        # 解析模板结构
        self.fields = re.findall(r'\{(\w+)\}', template)
        self.field_positions = []  # 每个 field 在模板中的字符位置
        self.current_field_idx = 0

        # 将模板转换为 token 序列
        self.template_tokens = tokenizer.encode(template, add_special_tokens=False)

        # 找出每个 field 占位符的位置
        current_pos = 0
        for field in self.fields:
            placeholder = "{" + field + "}"
            pos_in_template = template.find(placeholder, current_pos)
            self.field_positions.append({
                "name": field,
                "char_start": pos_in_template,
                "char_end": pos_in_template + len(placeholder),
            })
            current_pos = pos_in_template + len(placeholder)

        print(f"📋 模板引导处理器初始化")
        print(f"   模板: {template}")
        print(f"   字段: {self.fields}")

    def _get_current_allowed_tokens(self, generated_text: str) -> set:
        """根据当前生成位置, 计算允许的下一个 token 集合"""
        
        # 简化版: 检查当前应该填哪个字段
        # 实际实现需要更复杂的有限自动机 (DFA)
        
        allowed = set(range(len(self.tokenizer)))  # 默认全部允许

        # 检查是否已经完成了某个字段 (遇到分隔符或字段结束标记)
        for sep in [",", ";", "\n"]:
            sep_tokens = self.tokenizer.encode(sep, add_special_tokens=False)
            # 如果最后一个分隔符刚被生成, 可能需要切换到下一个字段
            pass

        return allowed

    def __call__(self, input_ids, scores):
        # 这是一个简化版本 —— 完整实现需要解析模板状态机
        return scores


class JSONSchemaProcessor:
    """
    JSON Schema 引导处理器
    确保生成的文本始终是合法 JSON
    (简化演示版; 生产环境推荐使用 outlines 库)
    """

    def __init__(self, schema: dict, tokenizer: PreTrainedTokenizerBase):
        self.schema = schema
        self.tokenizer = tokenizer
        
        # JSON 合法字符集合
        self.json_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                              '0123456789{}[]":,._-+ \t\n\r\'"/\\')
        
        # 预计算合法 token
        self.allowed_token_ids = set()
        for tid in range(tokenizer.vocab_size or 50257):
            try:
                decoded = tokenizer.decode([tid])
                if all(c in self.json_chars or c.isspace() for c in decoded):
                    self.allowed_token_ids.add(tid)
            except:
                pass

        print(f"📦 JSON Schema 处理器初始化")
        print(f"   允许的 token 数量: {len(self.allowed_token_ids)} / ~{tokenizer.vocab_size or 50257}")

    def __call__(self, input_ids, scores):
        """将非法 JSON 字符对应的 token 设为 -inf"""
        mask = torch.full_like(scores, float('-inf'))
        for tid in self.allowed_token_ids:
            if tid < scores.shape[-1]:
                mask[:, tid] = scores[:, tid]
        return mask


def demo_json_processor():
    """演示 JSON 格式约束"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }

    processors = [JSONSchemaProcessor(schema, tokenizer)]

    prompt = "Generate a person object in JSON format:\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        logits_processor=processors,
        pad_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
    print(f"\n[JSON 约束输出]:\n{result}")

demo_json_processor()
```

## 2.3 动态风格控制器

```python
class StyleControlProcessor:
    """
    动态风格控制处理器
    根据已生成内容的特征, 动态调整后续 token 的概率分布
    """

    def __init__(self, tokenizer, target_style: str = "formal"):
        self.tokenizer = tokenizer
        self.target_style = target_style
        
        # 不同风格的典型词汇 (简化示例)
        self.style_vocab = {
            "formal": ["therefore", "furthermore", "consequently", "nevertheless"],
            "casual": ["yeah", "kinda", "gonna", "wanna", "super", "awesome"],
            "technical": ["therefore", "algorithm", "optimization", "implementation"],
            "poetic": ["whisper", "echo", "dance", "bloom", "shimmer"],
        }

        # 获取目标风格的 token IDs
        self.preferred_tokens = set()
        for word in self.style_vocab.get(target_style, []):
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.preferred_tokens.update(ids)

        self.boost_factor = 1.5  # 对偏好 token 的增强系数

    def __call__(self, input_ids, scores):
        """增强目标风格相关 token 的概率"""

        for tid in self.preferred_tokens:
            if tid < scores.shape[-1]:
                scores[:, tid] *= self.boost_factor

        return scores


def demo_style_control():
    """演示风格控制"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    styles = ["formal", "casual"]

    prompt = "I think that the best approach would be to"

    for style in styles:
        processors = [StyleControlProcessor(tokenizer, style)]
        inputs = tokenizer(prompt, return_tensors="pt")

        output = model.generate(
            **inputs,
            max_new_tokens=35,
            do_sample=True,
            temperature=0.8,
            logits_processor=processors,
            pad_token_id=tokenizer.eos_token_id,
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
        print(f"\n[{style.upper()}]: {result}")

demo_style_control()
```

---

## 三、StoppingCriteria：精确控制何时停止

## 3.1 StoppingCriteria 接口

```python
from transformers import StoppingCriteria, StoppingCriteriaList
import torch


class BaseStoppingCriteria(StoppingCriteria):
    """停止条件基类"""

    def __init__(self):
        self.stop_reason = None  # 记录停止原因 (用于调试)

    def __call__(
        self,
        input_ids: torch.LongTensor,       # (batch, seq_len) 当前完整序列
        scores: torch.FloatTensor,           # (batch, vocab_size) 当前步 logits
        **kwargs,
    ) -> bool:
        """
        返回 True 表示应该停止生成
        """
        raise NotImplementedError
```

## 3.2 实用的 StoppingCriteria 实现

```python
class CombinedStoppingCriteria(StoppingCriteria):
    """
    组合停止条件: 满足任一条件即停止 (OR 逻辑)
    """

    def __init__(self, criteria_list: list):
        self.criteria_list = criteria_list

    def __call__(self, input_ids, scores, **kwargs):
        for criteria in self.criteria_list:
            if criteria(input_ids, scores, **kwargs):
                return True
        return False


class AllMustStopCriteria(StoppingCriteria):
    """
    组合停止条件: 所有条件都满足才停止 (AND 逻辑)
    """

    def __init__(self, criteria_list: list):
        self.criteria_list = criteria_list

    def __call__(self, input_ids, scores, **kwargs):
        for criteria in self.criteria_list:
            if not criteria(input_ids, scores, **kwargs):
                return False
        return True


class RegexMatchStoppingCriteria(StoppingCriteria):
    """
    正则表达式匹配停止条件
    当生成的文本匹配指定模式时停止
    """

    def __init__(self, pattern: str, tokenizer):
        import re
        self.pattern = re.compile(pattern)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return bool(self.pattern.search(text))


class ContentLengthStoppingCriteria(StoppingCriteria):
    """
    内容长度 (字符数) 停止条件
    """

    def __init__(self, max_chars: int, tokenizer):
        self.max_chars = max_chars
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return len(text) >= self.max_chars


class SentenceCompleteStoppingCriteria(StoppingCriteria):
    """
    句子完整性停止条件
    等到当前句子完成后才停止 (以句号/问号/感叹号为标志)
    """

    def __init__(self, tokenizer, min_sentences: int = 1):
        self.tokenizer = tokenizer
        self.min_sentences = min_sentences
        self.end_punctuation = {'.', '!', '?', '。', '！', '？'}

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        sentence_count = sum(1 for c in text if c in self.end_punctuation)
        return sentence_count >= self.min_sentences


def demo_stopping_criteria():
    """演示各种停止条件"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Tell me a short story about a cat"
    inputs = tokenizer(prompt, return_tensors="pt")

    # 组合使用多种停止条件
    stopping_criteria = StoppingCriteriaList([
        ContentLengthStoppingCriteria(max_chars=150, tokenizer),
        SentenceCompleteStoppingCriteria(tokenizer, min_sentences=2),
        RegexMatchStoppingCriteria(r"(The End|FIN)$", tokenizer),
    ])

    output = model.generate(
        **inputs,
        max_new_tokens=200,              # 上限设高
        stopping_criteria=stopping_criteria,  # 由 stopping criteria 决定实际停止时机
        pad_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
    char_count = len(result)
    print(f"\n[多条件停止控制] ({char_count} 字符):\n{result}")


if __name__ == "__main__":
    demo_stopping_criteria()
```

---

## 四、完整的生产级自定义生成 Pipeline

## 4.1 可配置的生成管道

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig,
)


@dataclass
class GenerationPolicy:
    """
    生成策略配置
    封装所有生成相关的参数和处理器
    """

    # 基础参数
    max_new_tokens: int = 256
    min_new_tokens: int = 1
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.92

    # 重复控制
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3

    # Beam Search (如果不用 sampling)
    num_beams: int = 1
    length_penalty: float = 1.0

    # 自定义处理器
    logits_processors: list = field(default_factory=list)
    stopping_criteria: list = field(default_factory=list)

    # 其他
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class CustomGenerationPipeline:
    """
    自定义生成管道
    统一管理模型、tokenizer、processors 和 generation config
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        default_policy: GenerationPolicy = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.default_policy = default_policy or GenerationPolicy()

        if self.default_policy.pad_token_id is None:
            self.default_policy.pad_token_id = tokenizer.pad_token_id
        if self.default_policy.eos_token_id is None:
            self.default_policy.eos_token_id = tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        policy: GenerationPolicy = None,
        **override_kwargs,
    ) -> str:
        """
        执行生成

        Args:
            prompt: 输入提示
            policy: 生成策略 (None 则用默认策略)
            **override_kwargs: 覆盖策略中的任何参数
        """
        p = policy or self.default_policy

        # 应用覆盖
        for k, v in override_kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)

        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 构建参数
        gen_kwargs = dict(
            max_new_tokens=p.max_new_tokens,
            min_new_tokens=p.min_new_tokens,
            do_sample=p.do_sample,
            temperature=p.temperature if p.do_sample else None,
            top_k=p.top_k if p.do_sample else None,
            top_p=p.top_p if p.do_sample else None,
            num_beams=p.num_beams if not p.do_sample else 1,
            length_penalty=p.length_penalty,
            repetition_penalty=p.repetition_penalty,
            no_repeat_ngram_size=p.no_repeat_ngram_size,
            pad_token_id=p.pad_token_id,
            eos_token_id=p.eos_token_id,
        )

        # 移除 None 值
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        # 添加自定义 processors
        if p.logits_processors:
            gen_kwargs["logits_processor"] = p.logitS_processors

        # 添加自定义 stopping criteria
        if p.stopping_criteria:
            from transformers import StoppingCriteriaList
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(p.stopping_criteria)

        # 生成
        output = self.model.generate(**inputs, **gen_kwargs)

        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result[len(prompt):]

    def add_processor(self, processor):
        """添加一个 logits processor"""
        self.default_policy.logits_processors.append(processor)

    def add_stopping_criterion(self, criterion):
        """添加一个停止条件"""
        self.default_policy.stopping_criteria.append(criterion)


def demo_pipeline():
    """演示完整的生成管道"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    pipeline = CustomGenerationPipeline(model, tokenizer)

    # 配置策略
    chat_policy = GenerationPolicy(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.75,
        top_k=40,
        top_p=0.90,
        repetition_penalty=1.18,
        no_repeat_ngram_size=3,
    )

    # 添加自定义处理器
    pipeline.add_processor(SensitiveWordFilterProcessor(
        ["bad", "hate", "violence"], tokenizer
    ))

    # 生成
    result = pipeline.generate(
        prompt="Hello! How can I help you today?",
        policy=chat_policy,
    )

    print(f"\n[Pipeline 输出]:\n{result}")


if __name__ == "__main__":
    demo_pipeline()
```

---

## 五、调试技巧与常见问题

## 5.1 如何调试 LogitsProcessor？

```python
class DebugLogitsProcessor:
    """调试用的 LogitsProcessor: 打印每一步的处理信息"""

    def __init__(self, tokenizer, top_n=5):
        self.tokenizer = tokenizer
        self.top_n = top_n
        self.step = 0

    def __call__(self, input_ids, scores):
        self.step += 1

        # 打印 Top-N 候选
        top_scores, top_indices = torch.topk(scores[0], self.top_n)
        print(f"\n[Debug Step {self.step}] Top-{self.top_n} candidates:")
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            token = self.tokenizer.decode([idx])
            prob = torch.softmax(scores[0], dim=-1)[idx].item()
            print(f"  '{token}' (id={idx}) logit={score:.4f} prob={prob:.4%}")

        # 检查是否有值被设为 -inf
        inf_count = (scores[0] == float('-inf')).sum().item()
        total = scores.shape[-1]
        if inf_count > 0:
            print(f"  ⚠️  {inf_count}/{total} tokens 被 mask 为 -inf")

        return scores
```

## 5.2 常见问题速查

```python
def custom_generation_gotchas():
    """自定义生成中的常见问题"""

    gotchas = [
        {
            "问题": "LogitsProcessor 没有生效",
            "原因": "(1) 参数名写错了 (logits_processor vs logits_processors); "
                   "(2) Processor 返回了新的 tensor 而非原地修改; "
                   "(3) do_sample=False 时某些 Warper 不生效",
            "解决": "确认参数名正确; 用 DebugLogitsProcessor 验证; "
                   "检查 do_sample 设置",
        },
        {
            "问题": "生成结果仍然不符合格式要求",
            "原因": "LogitsProcessor 只能降低概率不能完全禁止;"
                   "极端情况下模型仍可能选中概率极低的 token",
            "解决": "对于严格格式要求, 使用 Outlines 库的 constrained decoding;"
                   "或在后处理中做格式修正",
        },
        {
            "问题": "StoppingCriteria 导致提前停止",
            "原因": "条件判断过于激进; 或在 prompt 中误触发了停止条件",
            "解决": "只检查新生成的部分 (排除 prompt); "
                   "增加最小长度保护; 用 DebugStoppingCriteria 调试",
        },
        {
            "问题": "多个 Processor 之间的冲突",
            "原因": "两个 Processor 对同一个 token 做了相反的操作",
            "解决": "明确 Processor 的执行顺序 (列表前面的先执行); "
                   "避免功能重叠的 Processor 同时使用",
        },
        {
            "问题": "性能下降明显",
            "原因": "Processor 内部有复杂的计算 (如 N-Gram 匹配、正则扫描)",
            "解决": "预处理计算结果 (如预先建好敏感词索引); "
                   "使用高效的字符串匹配算法 (Aho-Corasick)",
        },
    ]

    print("=" * 78)
    print("自定义生成常见问题排查")
    print("=" * 78)
    for g in gotchas:
        print(f"\n❌ {g['问题']}")
        print(f"   原因: {g['原因']}")
        print(f"   解决: {g['解决']}")

custom_generation_gotchas()
```

---

## 六、本章小结

这一节我们深入学习了如何通过 `LogitsProcessor` 和 `StoppingCriteria` 完全掌控生成过程：

| 能力 | 实现方式 | 适用场景 |
|------|---------|---------|
| **内容过滤** | SensitiveWordFilterProcessor | 安全合规、敏感词拦截 |
| **格式约束** | JSONSchemaProcessor / TemplateGuidedProcessor | 结构化输出 |
| **风格控制** | StyleControlProcessor | 写作风格调整 |
| **精确停控** | StoppingCriteria (多种组合) | 长度/关键词/句子完整性控制 |
| **调试分析** | DebugLogitsProcessor | 开发阶段的问题排查 |

**核心要点**：
1. **LogitsProcessor 是生成过程的"中间件"**——你可以像 Web 中间件一样链式组合多个处理器
2. **StoppingCriteria 提供了灵活的终止控制**——可以组合 OR/AND 逻辑
3. **对于严格的格式约束，推荐 Outlines 库**——它基于有限自动机实现了更高效可靠的约束解码
4. **生产环境建议封装成 Pipeline 类**——统一管理策略配置、复用处理器

至此，**第 8 章 文本生成与解码策略**全部完成！下一章我们将进入 **第 9 章 模型压缩与推理优化**，学习如何让大模型跑得更快、占资源更少。
