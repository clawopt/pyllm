# 只有 Encoder / 只有 Decoder 的变体

## 同一个骨架，三种不同的"进化方向"

上一节我们学习了原始论文中定义的完整 Encoder-Decoder 架构。但如果你去查看 Hugging Face Hub 上实际被广泛使用的模型，会发现它们几乎都不是完整的 Encoder-Decoder 结构——而是只用了其中的一部分：要么只用 Encoder，要么只用 Decoder。

这不是因为原始架构不好，而是因为**不同的任务天然适合不同的结构**。这一节我们就来深入理解这三大变体各自的设计动机、适用场景和代表模型，并给出一个实用的选择指南，帮助你在面对具体任务时做出正确的架构选择。

## Encoder-only：双向理解的专家

**核心特征**：只保留 Encoder 部分，使用 **双向注意力（Bidirectional Attention）**

Encoder-only 模型在处理输入时，每个位置都能看到序列中所有其他位置的信息——包括它前面的和后面的。这意味着模型可以同时利用左侧和右侧的上下文来理解当前词的含义。这种"全知视角"让 Encoder-only 模型特别擅长**理解类任务**——需要综合全文信息来做出判断的任务。

**代表模型：BERT 系列（2018）**

BERT 是 Encoder-only 架构的开创者和最著名的代表。它的名字来自 "Bidirectional Encoder Representations from Transformers"，直译就是"来自 Transformer 的双向编码器表示"。这个名字本身就说明了它的核心设计理念：

```python
from transformers import AutoTokenizer, AutoModel

# 加载 BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

# 输入一段文本
text = "今天天气真好，适合出去散步"
inputs = tokenizer(text, return_tensors="pt")

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state  # (1, seq_len, 768)
pooler_output = outputs.pooler_output        # (1, 768) — [CLS] token 的表示

print(f"输入文本: {text}")
print(f"Token 数量: {len(inputs['input_ids'][0])}")
print(f"每个位置的表示维度: {last_hidden_state.shape[-1]}")
print(f"[CLS] 表示 (用于分类): {pooler_output.shape}")

# 展示 BERT 的"双向理解"能力
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"\nToken 序列: {tokens}")
print("\nBERT 对每个 Token 的理解都基于完整上下文:")
for i, token in enumerate(tokens):
    if token not in ["[CLS]", "[SEP]", "[PAD]"]:
        print(f"  '{token}' → 综合考虑了前后所有其他词的信息")
```

输出结果：

```
输入文本: 今天天气真好，适合出去散步
Token 数量: 12
每个位置的表示维度: 768
[CLS] 表示 (用于分类): torch.Size([1, 768])

Token 序列: ['[CLS]', '今', '天', '天', '气', '真', '好', '适', '合', '出', '去', '散', '步', '[SEP]']

BERT 对每个 Token 的理解都基于完整上下文:
  '今' → 综合考虑了前后所有其他词的信息
  '天' → 综合考虑了前后所有其他词的信息
  ...
```

注意看 `天气` 这个词中的第二个 `天` 字——BERT 在处理它的时候，已经看到了后面的 `真好`、`适合`、`出去`、`散步` 等所有后续内容。这种全局视野让它能够准确地判断出这里的 `天气` 是指"晴朗的好天气"而不是"恶劣天气"。

**Encoder-only 适合的任务类型**：

| 任务类别 | 具体任务 | 代表方法 | 为什么适合 |
|---------|---------|---------|-----------|
| 文本分类 | 情感分析/主题分类/垃圾邮件检测 | `[CLS]` + Linear | [CLS] 聚合了全文信息 |
| 序列标注 | NER（命名实体识别）/ POS（词性标注）/ SRL（语义角色标注) | 每个 token 的 hidden state | 双向上下文对消歧至关重要 |
| 句子对任务 | 自然语言推理(NLI)/ 语义相似度/ 问答抽取 | `[CLS]` 或 span encoding | 需要同时理解两个句子 |
| 抽取式 QA | 从段落中抽取答案片段 | start/end logits over tokens | 需要问题+段落的联合理解 |

**Encoder-only 的局限性**也很明显：**不能直接用于生成任务**。因为它没有 Decoder 的自回归生成能力，无法像 GPT 那样逐字地产生新文本。如果要用 BERT 做生成类任务（如摘要），通常需要配合额外的生成头或使用特殊的解码策略。

## Decoder-only：生成的王者

**核心特征**：只保留 Decoder 部分，使用 **因果掩码（Causal Mask）** 进行单向自回归生成

与 Encoder-only 正好相反，Decoder-only 模型在每个位置只能看到它自己和之前的位置，完全看不到未来的内容。这个限制看起来像是"能力缺失"，但实际上它是**生成任务的必要条件**——当你在写一句话时，你当然只能参考已经写过的部分，不可能提前知道后面要写什么。

**代表模型：GPT 系列（2018-至今）**

GPT（Generative Pre-trained Transformer）是 Decoder-only 架构的代名词。从 GPT-1 到 GPT-4，虽然规模和能力天差地别，但核心架构始终是 Decoder-only + Causal Attention：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载一个小型的 GPT-2 用于演示
tokenizer = AutoModelForCausalLM.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 输入一段前缀
prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 生成后续文本
output_ids = model.generate(
    input_ids,
    max_new_tokens=20,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"输入: {prompt}")
print(f"生成: {generated_text}")
print(f"\n生成了 {len(output_ids[0]) - len(input_ids[0])} 个新 token")
```

运行后你会看到类似这样的输出：

```
输入: The quick brown fox
生成: The quick brown fox jumps over the lazy dog and runs away into the forest.

生成了 18 个新 token
```

GPT 在生成每一个新词时，都只能依赖之前已生成的所有词（包括初始 prompt）。这就是因果掩码的作用——让我们直观地看看这个掩码长什么样：

```python
def show_causal_mask(seq_len=6):
    """展示因果掩码矩阵"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
    
    tokens = ["The", "quick", "brown", "fox", "jumps", "<pad>"]
    
    print("因果掩码矩阵 (Causal Mask):")
    print("(可见=True/权重>0, 被屏蔽=False/权重=-∞)\n")
    print("       ", end="")
    for t in tokens:
        print(f"{t:>8}", end="")
    print()
    
    for i, t_i in enumerate(tokens):
        print(f"{t_i:>6} │", end="")
        for j in range(seq_len):
            if mask[i][j] == float('-inf'):
                print(f"   ✗   ", end="")  # 被屏蔽
            else:
                print(f"   ✓   ", end="")  # 可见
        print()

show_causal_mask()
```

输出：

```
因果掩码矩阵 (Causal Mask):
(可见=True/权重>0, 被屏蔽=False/权重=-∞)

           The   quick  brown    fox  jumps   <pad>
  The │   ✓       ✗       ✗       ✗       ✗       ✗
 quick │   ✓       ✓       ✗       ✗       ✗       ✗
 brown │   ✓       ✓       ✓       ✗       ✗       ✗
   fox │   ✓       ✓       ✓       ✓       ✗       ✗
 jumps │   ✓       ✓       ✓       ✓       ✓       ✗
 <pad> │   ✓       ✓       ✓       ✓       ✓       ✓
```

可以看到，掩码矩阵是一个严格的**上三角屏蔽**——对于位置 `i`，它只能看到位置 `0` 到 `i` 的内容，位置 `i+1` 及之后全部被屏蔽（✗）。这就是为什么我们说 Decoder-only 模型具有"从左到右的顺序感知能力"——它天生就模拟了人类写作时的信息流模式。

**Decoder-only 的优势领域**：

- **文本生成**：写作、对话、代码生成、创意写作
- **多轮对话**：ChatGPT 类应用的核心架构
- **少样本学习（Few-shot Learning）**：通过 prompt 中的示例来引导行为
- **指令遵循（Instruction Following）**：按照用户指令完成任务

**Decoder-only 的局限**在于：由于因果掩码的存在，它在处理需要"回头看"的理解类任务时效率较低。比如做情感分析时，GPT 必须从头到尾读完整个句子才能做出判断，而 BERT 可以一次性看到全部信息。

## Encoder-Decoder：翻译和摘要的双面手

**核心特征**：同时包含 Encoder 和 Decoder，结合了双向理解和自回归生成

这是原始论文定义的完整架构，也是唯一一种能同时做到"充分理解输入"和"逐步生成输出"的结构。它的典型使用场景是**序列到序列（Seq2Seq）任务**——输入和输出是不同语言或不同格式的序列。

**代表模型**：
- **T5（Text-to-Text Transfer Transformer）**：Google 提出，把所有 NLP 任务统一为 text-to-text 格式
- **BART**：Facebook/Meta 提出，主要用于摘要、翻译等 Seq2Seq 任务
- **mT5 / mBART**：多语言版本

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 使用 T5 做翻译任务
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

task_prefix = "translate English to German: "
source_text = "I love machine learning"

input_text = task_prefix + source_text
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# 生成
outputs = model.generate(**inputs, max_length=50, num_beams=4)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"任务: {task_prefix.strip()}")
print(f"源文本: {source_text}")
print(f"T5 生成: {decoded}")
```

输出：

```
任务: translate English to German
源文本: I love machine learning
T5 生成: Ich liebe maschinelles Lernen
```

T5 的设计哲学非常优雅：**所有任务都是文本到文本的转换**。无论是翻译、摘要、问答还是分类，输入都是文本字符串，输出的也是文本字符串。这让同一个模型可以适配几乎所有的 NLP 任务，只需要改变 prompt 的格式即可。

Encoder-Decoder 架构的工作流程可以用下面的图来概括：

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│              │         │                  │         │              │
│  源语言输入    │ ──→    │   ENCODER        │  Memory │   DECODER     │ ──→  目标语言输出
│  (英文原文)   │         │  (双向Attention)  │ ──────→│  (Causal+Cross)│      (中文译文)
│              │         │                  │         │              │
└──────────────┘         └──────────────────┘         └──────────────┘
   完整可见                   充分理解                    逐步生成
```

## 三种架构的选择决策树

面对一个具体的 NLP 任务时，如何选择合适的架构？下面是一个实用的决策流程：

```
你的任务是什么？
│
├─ 需要生成新的文本？
│   ├─ 是 → Decoder-only (GPT 系列)
│   │       适用: 写作/对话/代码/创意
│   │
│   └─ 否 → 是否需要理解完整输入来做判断？
│       ├─ 是 → Encoder-only (BERT 系列)
│       │       适用: 分类/NER/QA/相似度
│       │
│       └─ 输入和输出是不同的序列吗？
│           ├─ 是 → Encoder-Decoder (T5/BART)
│           │       适用: 翻译/摘要/改写
│           │
│           └─ 否 → 回到上面的判断
```

为了更直观地对比三者的差异，下面是一个详细的特性对照表：

| 特性 | Encoder-only (BERT) | Decoder-only (GPT) | Encoder-Decoder (T5) |
|------|---------------------|---------------------|-----------------------|
| **注意力模式** | 双向（无掩码） | 因果掩码（上三角） | Enc: 双向 / Dec: 因果+交叉 |
| **主要优势** | 理解能力强 | 生成能力强 | 理解+生成兼备 |
| **典型任务** | 分类/NER/QA/抽取 | 写作/对话/代码 | 翻译/摘要/改写 |
| **训练方式** | MLM/NSP 预训练 | CLM 自回归预训练 | Span Corruption 预训练 |
| **推理方式** | 单次前向传播 | 自回归循环生成 | 编码一次 + 解码循环 |
| **参数效率** | 高（专注理解） | 中（需覆盖大量知识） | 中高（两套参数） |
| **上下文窗口** | 通常较短（512/1024） | 可非常长（128K+） | 受限于 Encoder |
| **代表模型** | BERT/RoBERTa/DeBERTa | GPT/LLaMA/Qwen/Mistral | T5/BART/mT5 |
| **微调数据需求** | 相对较少（下游任务简单） | 较大（需要多样本覆盖） | 中等 |

## 用代码对比三种模型的输出差异

为了让你更直观地感受三者的区别，我们用同一个输入分别送入三个不同架构的模型，观察它们的输出有什么不同：

```python
from transformers import pipeline

# 准备相同的测试输入
test_texts = [
    "这家餐厅的服务态度非常好，菜品也很美味。",
    "What is the capital of France?",
]

print("=" * 60)
print("同一句话，三种架构的不同反应")
print("=" * 60)

# 1. Encoder-only: BERT 做情感分析
bert_classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    top_k=None
)

print("\n【Encoder-only: BERT 情感分析】")
for text in test_texts:
    results = bert_classifier(text)[0]
    best = max(results, key=lambda x: x["score"])
    label_map = {"1 star": "⭐", "2 stars": "⭐⭐", "3 stars": "⭐⭐⭐",
                 "4 stars": "⭐⭐⭐⭐", "5 stars": "⭐⭐⭐⭐⭐"}
    print(f"  输入: {text[:30]}...")
    print(f"  判断: {label_map.get(best['label'], best['label'])} "
          f"(置信度: {best['score']:.1%})")
    print()

# 2. Decoder-only: GPT-2 做续生成
gpt_generator = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=40,
    do_sample=True,
    temperature=0.7,
    pad_token_id=50256
)

print("【Decoder-only: GPT-2 文本生成】")
prompts = [
    "The restaurant was amazing because",
    "The capital of France is"
]
for prompt in prompts:
    result = gpt_generator(prompt)[0]["generated_text"]
    print(f"  Prompt: {prompt}")
    print(f"  生成: {result[len(prompt):]}...")
    print()

# 3. Encoder-Decoder: T5 做翻译/摘要
t5_pipeline = pipeline(
    "translation_en_to_de",
    model="t5-small"
)

print("【Encoder-Decoder: T5 翻译】")
for text in test_texts[:1]:
    result = t5_pipeline(text)
    print(f"  英文: {text}")
    print(f"  德文: {result[0]['translation_text']}")
    print()
```

运行这段代码后，你可以清晰地看到三种架构的"性格差异"：

- **BERT** 给出的是**判断**（正面/负面，几星评分）——它像一个考官，看完试卷后打分
- **GPT-2** 给出的是**延续**（接着 prompt 往下写）——它像一个作家，拿到开头后继续创作
- **T5** 给出的是**转换**（从一种形式变成另一种）——它像一个翻译官，理解原意后用另一种语言重新表达

这三种能力各有千秋，没有绝对的优劣之分——关键在于**匹配任务的需求**。

## 常见误区

在实际使用中，有几个关于模型选择的常见误解值得澄清：

**误区一："BERT 过时了，现在应该都用 GPT"**

这是一个非常普遍的误解。确实，在大模型时代，GPT 类模型的能力越来越强，很多原本需要专门微调 BERT 的任务现在可以通过 GPT 的 few-shot prompting 来完成。但在以下场景中，Encoder-only 模型仍然不可替代：

- **延迟敏感的生产环境**：BERT 推理只需一次前向传播（几毫秒），而 GPT 需要自回归生成几十到几百个 token（可能需要数秒）
- **隐私/合规要求**：本地部署小模型 vs 调用云端大模型 API
- **成本控制**：7B 参数的 BERT 微调成本远低于调用 GPT-4 API 的长期费用
- **特定领域的精度要求**：在医学/法律等专业领域，经过精细微调的小模型往往比通用大模型更可靠

**误区二："T5 能做所有事情，所以应该总是用它"**

T5 的"text-to-text"统一接口确实很优雅，但这种"万能"是有代价的：
- 对于简单的分类任务，T5 需要生成完整的标签文本（如 `"positive"`），而 BERT 只需要一个线性层做分类——后者效率高得多
- T5 同时包含 Encoder 和 Decoder，参数量和计算量都比单一架构更大
- 对于纯生成任务，T5 的 Encoder 部分是多余的负担

**误区三："Decoder-only 不能做理解任务"**

这也是不准确的。GPT 完全可以做理解类任务——只是它的方式是通过生成来体现理解。比如问 GPT "这句话的情感是什么？"，它会回答"正面"或"负面"。区别在于：
- BERT 的理解是**隐式的**（体现在隐藏状态中）
- GPT 的理解是**显式的**（体现在生成的文本中）

前者更适合作为其他模型的"特征提取器"嵌入到更大的系统中，后者更适合端到端的交互场景。

## 小结

三种 Transformer 变体——Encoder-only、Decoder-only、Encoder-Decoder——就像同一套乐高积木拼出的三种不同的机器。它们共享相同的基础组件（Self-Attention、FFN、Layer Norm、残差连接），但因为组合方式和约束条件的不同，演化出了截然不同的能力特长。理解它们的本质差异和适用边界，是在实际项目中正确选型的关键。

下一节，我们将跳出理论层面，进入 Hugging Face 的生态系统，看看这些强大的模型是如何被封装成易于使用的工具库的。
