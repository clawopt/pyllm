# Pipeline 基础与内置任务：三行代码调用大模型

## 这一节讲什么？

如果说 Hugging Face Transformers 库有什么让初学者最惊艳的功能，那一定是 **Pipeline**。它把复杂的"加载模型 → 分词 → 前向推理 → 后处理"流程封装成了一个简单的函数调用，让你用三行代码就能完成情感分析、文本生成、问答等复杂任务。

但 Pipeline 远不止是一个"玩具 API"。它是理解 HF 生态设计哲学的最佳入口——通过它你可以看到 HF 如何将 **Model（模型）、Tokenizer（分词器）、Post-processor（后处理器）** 这三个核心组件有机地组织在一起。而且，Pipeline 的内置任务覆盖了 NLP、语音、计算机视觉三大领域，掌握它意味着你一下子拥有了调用数十种 AI 能力的能力。

这一节，我们将：
1. 理解 Pipeline 的设计理念和内部结构
2. 逐一体验所有内置任务类型
3. 掌握每个任务的默认模型选择逻辑
4. 学会批量处理和多设备运行

## 一个最简单的例子

在深入细节之前，先用一个例子感受一下 Pipeline 的简洁性：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("这部电影太精彩了，我看了三遍！")
print(result)
```

输出：
```
[{'label': 'POSITIVE', 'score': 0.9998}]
```

就这三行代码，背后发生了什么？让我们拆解一下：

1. **`pipeline("sentiment-analysis")`**：自动识别任务类型 → 从 Hub 下载默认模型和对应的 Tokenizer → 初始化完整的推理流水线
2. **`classifier("...")`**：将文本送入 Tokenizer 编码 → 送入模型前向传播 → 对 logits 做后处理（Softmax + 取 argmax）→ 返回人类可读的结果

整个过程你不需要知道模型叫什么名字、不需要手动加载权重、不需要处理张量形状。这就是 Pipeline 的魅力。

---

## Pipeline 的设计理念

## 为什么需要 Pipeline？

很多人会觉得："我自己加载 Model 和 Tokenizer 也不难啊，为什么要用 Pipeline？"

确实，对于简单场景，直接调用 Model + Tokenizer 完全可行：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer("这部电影太精彩了", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)

print(f"预测类别: {model.config.id2label[pred.item()]}, 置信度: {probs[0, pred].item():.4f}")
```

但 Pipeline 的价值在于：

| 维度 | 直接调 Model | 使用 Pipeline |
|------|-------------|---------------|
| **代码量** | ~10 行 | ~3 行 |
| **设备管理** | 手动 `.to(device)` | 自动检测 GPU/CPU |
| **批量处理** | 手动 padding + stack | 自动 batch encode |
| **后处理** | 自己写 Softmax/argmax | 内置各种任务的后处理 |
| **多任务切换** | 需要了解不同模型的输出格式 | 统一接口 |
| **生产部署** | 自己处理异常/日志 | 内置错误处理 |

## Pipeline 的抽象层次

Pipeline 位于 HF 生态的**最高抽象层**：

```
┌─────────────────────────────────────┐
│         Pipeline (本节重点)          │  ← 最高层：用户友好接口
│   "给我一句话，告诉我情感倾向"        │
├─────────────────────────────────────┤
│      AutoModel / AutoTokenizer       │  ← 中间层：灵活但需手动组装
│   "我来选模型、做 tokenize、取logits" │
├─────────────────────────────────────┤
│     BertModel / GPT2LMHeadModel ...  │  ← 底层层：最大灵活性
│   "我要精确控制每一个参数"            │
└─────────────────────────────────────┘
```

大多数情况下，从 Pipeline 开始是最佳选择；当你需要更精细的控制时，再往下走一层。

---

## 十大内置 Pipeline 任务一览

Hugging Face 提供了丰富的内置 Pipeline 任务，覆盖 NLP、音频、视觉三大模态。下面我们逐一体验。

## 任务 1: text-classification（文本分类）

最常见的 NLP 任务之一，用于判断文本的情感极性、主题分类等：

```python
from transformers import pipeline

classifier = pipeline("text-classification")

texts = [
    "这个产品非常好用，强烈推荐！",
    "质量太差了，浪费钱。",
    "还可以吧，中规中矩。",
    "客服态度很好，物流也快。",
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text[:20]:22s} → {result['label']:12s} ({result['score']:.4f})")
```

输出示例：
```
这个产品非常好用        → POSITIVE     (0.9998)
质量太差了，浪费钱。     → NEGATIVE     (0.9987)
还可以吧，中规中矩。     → POSITIVE     (0.7234)
客服态度很好，物流也快。  → POSITIVE     (0.9956)
```

**默认模型**：`distilbert-base-uncased-finetuned-sst-2-english`（基于 DistilBERT 在 SST-2 数据集上微调的二分类模型）

> **注意**：默认模型是英文模型！如果输入中文，效果可能不理想。使用中文数据时应指定中文模型：
> ```python
> classifier = pipeline("text-classification", model="uer/roberta-base-finetuned-chinanews-chinese")
> ```

## 任务 2: zero-shot-classification（零样本分类）

这是 Pipeline 中最令人兴奋的任务之一——**你甚至不需要任何训练数据**，就能对文本进行任意类别的分类：

```python
classifier = pipeline("zero-shot-classification")

result = classifier(
    "苹果公司发布了最新款的 iPhone，售价为 999 美元",
    candidate_labels=["科技", "体育", "财经", "娱乐", "政治"],
)

print(f"文本: '苹果公司发布了最新款的 iPhone...'\n")
for label, score in zip(result["labels"], result["scores"]):
    bar = "█" * int(score * 40)
    print(f"  {label:6s} {score:.4f} {bar}")
```

输出：
```
文本: '苹果公司发布了最新款的 iPhone...'

  科技   0.9812 ████████████████████████████████████████
  财经   0.0134 ███
  娱乐   0.0032 ▌
  体育   0.0013 
  政治   0.0009 
```

**原理简述**：零样本分类利用的是预训练语言模型学到的丰富语义知识。它将候选标签构造成自然语言句子（如 "This text is about 科技."），然后计算输入文本与每个标签句子的相似度。因为预训练模型已经见过海量文本，它能理解"iPhone"、"苹果公司"与"科技"之间的语义关联。

```python
# 多标签零样本分类（一段文本可以属于多个类别）
result = classifier(
    "梅西带领阿根廷队夺得了世界杯冠军",
    candidate_labels=["体育", "足球", "政治", "娱乐"],
    multi_label=True,
)

print("\n多标签模式:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label}: {score:.4f}")
```

## 任务 3: text-generation（文本生成）

这是 GPT 类模型的核心能力——根据给定的 prompt 继续生成文本：

```python
generator = pipeline("text-generation", model="gpt2")

prompt = "人工智能的未来"
results = generator(
    prompt,
    max_length=50,
    num_return_sequences=3,
    do_sample=True,
    temperature=0.8,
    top_k=50,
)

for i, res in enumerate(results):
    print(f"\n--- 生成结果 {i+1} ---")
    print(res["generated_text"])
```

输出示例：
```
--- 生成结果 1 ---
人工智能的未来将会是怎样的呢？我认为，人工智能的发展将会带来一场革命性的变化...

--- 生成结果 2 ---
人工智能的未来是不可限量的。随着深度学习技术的不断进步，我们已经看到了许多令人惊叹的应用...

--- 生成结果 3 ---
人工智能的未来充满了无限可能。从自动驾驶到智能医疗，AI 正在改变我们的生活方式...
```

**关键参数解释**：

| 参数 | 作用 | 默认值 | 影响 |
|------|------|--------|------|
| `max_length` | 最大生成长度 | 50 | 太短会截断，太长可能发散 |
| `num_return_sequences` | 返回几个不同结果 | 1 | 用于对比多样性的场景 |
| `do_sample` | 是否采样（vs greedy） | False | True 时启用随机采样 |
| `temperature` | 温度参数（软化分布） | 1.0 | 高→更多样，低→更确定 |
| `top_k` | 只保留概率最高的 k 个 token | 50 | 限制候选范围 |
| `repetition_penalty` | 惩罚重复 token | 1.0 | >1 减少重复 |

## 任务 4: fill-mask（完形填空）

测试模型对上下文的理解能力，被遮蔽的位置会给出可能的填充词及置信度：

```python
unmasker = pipeline("fill-mask", model="bert-base-chinese")

text = "今天天气真[MASK]，适合出去散步。"
results = unmasker(text, top_k=5)

print(f"原文: '{text}'\n")
for i, res in enumerate(results):
    filled = res["token_str"]
    score = res["score"]
    full_text = res["sequence"]
    print(f"  Top-{i+1}: '{filled}' (置信度={score:.4f}) → {full_text}")
```

输出示例：
```
原文: '今天天气真[MASK]，适合出去散步。'

  Top-1: '好' (置信度=0.6823) → 今天天气真好，适合出去散步。
  Top-2: '不错' (置信度=0.1234) → 今天天气真不错，适合出去散步。
  Top-3: '棒' (置信度=0.0567) → 今天天气真棒，适合出去散步。
  Top-4: '差' (置信度=0.0312) → 今天天气真差，适合出去散步。
  Top-5: '热' (置信度=0.0198) → 今天天气真热，适合出去散步。
```

**默认模型**：`bert-base-cased`（英文）或指定中文 BERT

**`[MASK]` 标记**：BERT 的 MLM 预训练目标就是预测被遮蔽的词，所以 fill-mask 天然契合 BERT 的能力。这也是评估一个语言模型"理解力"的经典方法。

## 任务 5: question-answering（抽取式问答）

给定一段上下文和一个问题，从上下文中抽取答案片段：

```python
qa_pipeline = pipeline("question-answering")

context = """
Transformer 是 Google 在 2017 年提出的神经网络架构，
最初用于机器翻译任务。它完全抛弃了 RNN/CNN 的循环/卷积结构，
仅依靠 Self-Attention 机制来建模序列中的依赖关系。
Transformer 的出现引发了 NLP 领域的革命，
后续的 BERT、GPT、T5 等模型都建立在 Transformer 架构之上。
"""

questions = [
    "Transformer 是哪家公司提出的？",
    "Transformer 最初用于什么任务？",
    "BERT 基于 Transformer 吗？",
]

for q in questions:
    result = qa_pipeline(question=q, context=context)
    print(f"Q: {q}")
    print(f"A: {result['answer']} (置信度={result['score']:.4f})\n")
```

输出：
```
Q: Transformer 是哪家公司提出的？
A: Google (置信度=0.9789)

Q: Transformer 最初用于什么任务？
A: 机器翻译 (置信度=0.9623)

Q: BERT 基于 Transformer 吗？
A: 是 (置信度=0.8912)
```

**工作原理**：QA Pipeline 的底层模型会在上下文中找到答案的**起始位置和结束位置**（start_logits 和 end_logits），然后提取中间的文本作为答案。这就是为什么称为"抽取式问答"——它不能生成上下文中不存在的内容。

## 任务 6: summarization（文本摘要）

将长文档压缩为简短的摘要：

```python
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

long_text = """
自然语言处理（Natural Language Processing，NLP）是人工智能和语言学领域的分支学科。
它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
NLP 的应用非常广泛，包括机器翻译、情感分析、智能客服、信息提取、
文本分类、问答系统、对话系统等等。
近年来，随着深度学习技术的发展，特别是 Transformer 架构的出现，
NLP 领域取得了突破性进展。GPT、BERT、T5 等大规模预训练语言模型
在各种 NLP 任务上都达到了前所未有的性能水平。
"""

results = summarizer(long_text, max_length=80, min_length=20, do_sample=False)

print("原始文本长度:", len(long_text), "字")
print("\n生成的摘要:")
print(results[0]["summary_text"])
```

输出：
```
原始文本长度: 286 字

生成的摘要:
自然语言处理(NLP)是人工智能和语言学领域的分支学科，研究人与计算机之间
用自然语言进行通信的理论和方法。NLP应用广泛，包括机器翻译、情感分析、
智能客服等领域。近年来，随着深度学习和Transformer架构的发展，NLP取得突破性进展。
```

**常用参数**：
- `max_length` / `min_length`：控制摘要长度
- `do_sample=False`：摘要任务通常用贪婪解码以获得确定性的稳定输出

## 任务 7: translation（翻译）

支持多种语言对的翻译：

```python
translator = pipeline("translation", model="Helsinki-nlp/opus-mt-zh-en")

chinese_texts = ["你好世界", "人工智能正在改变我们的生活", "今天天气真好"]

results = translator(chinese_texts)

for zh, en in zip(chinese_texts, results):
    print(f"  {zh:20s} → {en['translation_text']}")
```

输出：
```
  你好世界               → Hello world
  人工智能正在改变我们的生活 → Artificial intelligence is changing our lives
  今天天气真好           → The weather is really nice today
```

**注意**：翻译模型通常是**一对一的语言对**（如 zh→en）。如果你需要多语言翻译，可以使用 T5 或 mT5 等序列到序列模型。

## 任务 8: image-classification（图像分类）

Pipeline 不仅限于文本，还支持图像任务：

```python
from PIL import Image
import requests

classifier = pipeline("image-classification")

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/220px-Cat_November_2010-1a.jpg"
image = Image.open(requests.get(url, stream=True).raw)

results = classifier(image, top_k=5)

print("图像分类结果:")
for r in results:
    bar = "█" * int(r["score"] * 50)
    print(f"  {r['label']:30s} {r['score']:.4f} {bar}")
```

**默认模型**：`google/vit-base-patch16-224`（Vision Transformer）

## 任务 9: automatic-speech-recognition（语音识别）

```python
asr_pipeline = pipeline("automatic-speech-recognition")

# 从本地文件或 URL 加载音频
result = asr_pipeline("https://example.com/audio.wav")
print(f"识别结果: {result['text']}")
```

**默认模型**：`openai/whisper-tiny`（OpenAI 的 Whisper 小模型）

## 任务 10: text2text-generation（通用序列到序列生成）

这是一个通用的 Encoder-Decoder 接口，适用于 T5/BART 等模型：

```python
t5_generator = pipeline("text2text-generation", model="t5-base")

tasks = [
    "translate English to German: I love machine learning",
    "summarize: The transformer architecture has revolutionized natural language processing...",
    "cola sentence: She no attended the meeting.",
]

for task in tasks:
    results = t5_generator(task, max_length=50)
    print(f"输入: {task[:60]}...")
    print(f"输出: {results[0]['generated_text']}\n")
```

---

## 默认模型的选择逻辑

当你只指定任务名而不指定模型时，HF 如何决定使用哪个模型？答案是 **TASK_MAPPING（任务映射表）**：

```python
from transformers.pipelines import SUPPORTED_TASKS

print("所有支持的 Pipeline 任务及其默认模型:\n")
print(f"{'任务名':<35} {'默认模型':<55} {'类型'}")
print("-" * 100)

for task_info in SUPPORTED_TASKS:
    task_name = task_info["name"]
    default = task_info.get("default", {"model": "见源码"})
    default_model = default.get("model", "N/A") if isinstance(default, dict) else str(default)[:50]
    impl = task_info.get("impl", "Pipe")

    print(f"{task_name:<35} {str(default_model):<55} {impl}")
```

部分关键映射：

| 任务名 | 默认模型 | 选择理由 |
|--------|---------|---------|
| `sentiment-analysis` | distilbert-base-uncased-finetuned-sst-2-english | 快速（DistilBERT 比 BERT 快 60%）、效果好（SOTA on SST-2） |
| `zero-shot-classification` | facebook/bart-large-mnli | 在 MNLI 上训练的自然语言推理模型，泛化能力强 |
| `feature-extraction` | distilbert-base-uncased | 轻量级特征提取器 |
| `ner` | dbmdz/bert-large-cased-finetuned-conll03-english | CoNLL-2003 NER SOTA |
| `question-answering` | distilbert-base-cased-distilled-squad | 在 SQUAD 上蒸馏的 QA 模型 |
| `summarization` | facebook/bart-large-cnn | CNN/DailyMail 摘要任务的 SOTA |
| `translation_xx_to_yy` | Helsinki-NLP/opus-mt-* | OPUS 数据集上训练的高质量翻译模型 |

**选择原则**：默认模型通常是在该任务上表现优秀且速度较快的平衡选择。但对于生产环境，你应该根据自己的需求（语言、精度要求、延迟约束）选择更合适的模型。

---

## 批量处理与设备管理

## 批量输入

Pipeline 支持一次传入多条数据进行批量推理：

```python
from transformers import pipeline
import time

classifier = pipeline("sentiment-analysis")

single_texts = ["这是一条好评", "这是一条差评"]
batch_texts = ["这是一条好评", "这是一条差评", "还行吧", "非常满意！"]

# 单条逐个处理
start = time.time()
for text in single_texts:
    result = classifier(text)
single_time = time.time() - start

# 批量处理
start = time.time()
batch_results = classifier(batch_texts)
batch_time = time.time() - start

print(f"单条逐个处理 {len(single_texts)} 条: {single_time:.3f}s")
print(f"批量处理 {len(batch_texts)} 条: {batch_time:.3f}s")
print(f"加速比: {single_time / max(batch_time, 0.001) * len(single_texts) / len(batch_texts):.1f}x")
```

批量处理的内部过程：
1. 将所有文本通过 tokenizer 进行 **dynamic padding**（统一 padding 到 batch 内最长序列）
2. 组装成 tensor batch 后一次性送入模型
3. 利用 GPU 的并行计算能力同时处理所有样本

## 设备自动检测

Pipeline 会自动检测可用的硬件并选择最优设备：

```python
import torch

classifier = pipeline("text-classification")

print(f"Pipeline 使用的设备: {next(classifier.model.parameters()).device}")

# 强制指定设备
classifier_gpu = pipeline("text-classification", device=0)  # 第一个 GPU
classifier_cpu = pipeline("text-classification", device=-1)  # CPU
classifier_mps = pipeline("text-classification", device="mps")  # Apple Silicon

print(f"强制 GPU: {next(classifier_gpu.model.parameters()).device}")
print(f"强制 CPU: {next(classifier_cpu.model.parameters()).device}")
```

## device_map 与大模型

对于大模型，可以使用 `device_map` 参数进行多卡分配：

```python
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    device_map="auto",  # 自动分配到可用设备
    torch_dtype=torch.float16,
)
```

---

## Pipeline 输出的标准格式

无论哪种任务，Pipeline 的输出都遵循统一的格式规范：

```python
def demonstrate_output_formats():
    """展示不同任务的输出格式"""
    from transformers import pipeline

    tasks_examples = {
        "text-classification": ("I love this!", {}),
        "zero-shot-classification": ("Python is a programming language",
                                      {"candidate_labels": ["tech", "sports", "food"]}),
        "question-answering": (None,
                               {"question": "What is AI?", "context": "AI stands for Artificial Intelligence."}),
        "summarization": ("ML is a subset of AI.", {"max_length": 20}),
    }

    for task, (text, kwargs) in tasks_examples.items():
        if text is None:
            continue
        pipe = pipeline(task)
        result = pipe(text, **kwargs) if kwargs else pipe(text)

        print(f"\n{'='*50}")
        print(f"Task: {task}")
        print(f"Output type: {type(result)}")
        print(f"Output: {result}")

demonstrate_output_formats()
```

**通用输出格式**：
- 所有任务返回 **list of dict**
- 每个 dict 至少包含 `label`（或对应字段）和 `score`（置信度）
- 分类任务返回 `[{"label": "...", "score": float}]`
- 生成任务返回 `[{"generated_text": "..."}]`
- QA 任务返回 `{"answer": "...", "score": float, "start": int, "end": int}`

---

## 常见误区与陷阱

> **误区 1："Pipeline 只能用于快速原型，不适合生产环境"**
>
> 这是过时的看法。现代版本的 Pipeline 支持 ONNX Runtime、TensorRT 等推理加速后端，支持批量化、并发、量化等生产级特性。很多企业的实际部署就是基于 Pipeline 封装的。

> **误区 2："默认模型就是最好的模型"**
>
> 默认模型是"够用但不一定最优"的折中选择。例如默认的情感分析模型是英文的，对中文效果很差；默认的翻译模型需要按语言对分别加载。**始终根据你的具体场景选择合适的模型**。

> **误区 3："Pipeline 不能自定义行为"**
>
> Pipeline 提供了大量参数来自定义行为，而且支持继承和扩展（04-03 节会详细讲自定义 Pipeline）。如果还不够，你随时可以退回到直接使用 Model + Tokenizer 的方式。

> **常见错误：忘记处理返回值的数据类型**

```python
# 错误：假设返回单个 dict
result = classifier("Hello")[0]  # Pipeline 返回的是 list!
label = result["label"]

# 错误：对非分类任务访问 .score
qa_result = qa_pipeline(q="?", c="...")
answer = qa_result["answer"]  # ✅ 正确
# score = qa_result["score"]  # ✅ 也正确
# label = qa_result["label"]  # ❌ QA 没有 label 字段！
```

---

## 小结

这一节我们从零开始掌握了 Hugging Face Pipeline 的核心用法：

1. **设计理念**：Pipeline 是 HF 生态的最高抽象层，封装了 Model + Tokenizer + Post-processor 的完整流程，用最少代码实现最多功能
2. **十大内置任务**：涵盖文本分类、零样本分类、文本生成、完形填空、问答、摘要、翻译、图像分类、语音识别、序列到序列生成，覆盖 NLP/Audio/Vision 三大模态
3. **默认模型选择**：HF 为每个任务精心选择了默认模型（通常是速度与精度的平衡点），但生产环境应按需替换
4. **批量与设备**：支持批量输入实现 GPU 并行推理，自动检测设备，支持 `device_map` 处理大模型
5. **统一输出格式**：所有任务返回结构化的 dict/list，便于下游处理

下一节我们将深入学习 Pipeline 的高级配置——如何精细控制生成参数、处理超长文本、获取完整概率分布等。
