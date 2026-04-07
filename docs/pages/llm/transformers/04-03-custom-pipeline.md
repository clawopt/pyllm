# 自定义 Pipeline：打造专属的 AI 处理流水线

## 这一节讲什么？

HF 提供了十几种内置 Pipeline，覆盖了绝大多数常见任务。但在实际项目中，你迟早会遇到内置 Pipeline 无法满足需求的场景——比如你需要一个"情感分析 + 关键词高亮"的组合输出，或者需要对接公司内部的自定义预处理逻辑，或者想给标准流程加上特定的后处理步骤。

这时候，你就需要**自定义 Pipeline**。好消息是 HF 的 Pipeline 框架设计得非常开放，你可以通过继承 `Pipeline` 类并重写几个关键方法来创建完全符合自己需求的处理流水线。

这一节，我们将：
1. 理解 Pipeline 的内部架构和执行流程
2. 从零实现一个完整的自定义 Pipeline
3. 学习如何注册和使用自定义 Pipeline
4. 掌握 Pipeline 与直接调 Model 之间的选择策略

---

## Pipeline 的内部架构

## 核心执行流程

每个 Pipeline 的 `__call__` 方法被触发时，内部会按以下顺序执行：

```
用户调用: pipe("输入文本")
    │
    ▼
┌─ _sanitize_parameters() ─────────────────────┐
│  将用户传入的参数分类为三类:                    │
│  - preprocess_params (给 preprocess 用)         │
│  - forward_params (给 _forward 用)             │
│  - postprocess_params (给 postprocess 用)      │
│                                               │
│  例: pipe(text, top_k=5, batch_size=8)        │
│  → top_k 归入 forward_params                  │
│  → batch_size 归入 preprocess_params          │
└───────────────────────────────────────────────┘
    │
    ▼
┌─ preprocess(输入) ──────────────────────────┐
│  预处理: 文本清洗、Tokenize、Tensor 组装等     │
│  输出: model_inputs (模型可接受的格式)          │
└───────────────────────────────────────────────┘
    │
    ▼
┌─ _forward(model_inputs) ─────────────────────┐
│  前向推理: 将输入送入模型                      │
│  输出: model_outputs (模型的原始输出)           │
└───────────────────────────────────────────────┘
    │
    ▼
┌─ postprocess(model_outputs) ─────────────────┐
│  后处理: Softmax/argmax、格式转换、文本解码等   │
│  输出: 最终的用户友好结果                       │
└───────────────────────────────────────────────┘
    │
    ▼
返回结果给用户
```

这个四阶段设计（sanitize → preprocess → forward → postprocess）是所有 Pipeline 共同遵循的模式。

---

## 从零实现：关键词高亮情感分析 Pipeline

让我们通过一个完整的实战案例来学习自定义 Pipeline。我们的目标是创建一个情感分析 Pipeline，它不仅返回情感标签和置信度，还能**标记出对判断影响最大的关键词**。

## 第一步：继承 Pipeline 类

```python
from transformers import Pipeline
import torch
import torch.nn.functional as F

class KeywordHighlightSentimentPipeline(Pipeline):
    """
    自定义情感分析 Pipeline：在输出中高亮影响判断的关键词

    工作原理：
    1. 对完整文本做一次情感预测（获取基准分数）
    2. 逐个 mask 掉每个词，重新预测
    3. 如果某个词被 mask 后分数变化大 → 该词是关键影响因素
    """

    def _sanitize_parameters(self, **kwargs):
        """将用户参数分配到各阶段"""
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}

        if "top_k_keywords" in kwargs:
            postprocess_kwargs["top_k_keywords"] = kwargs["top_k_keywords"]

        if "highlight_threshold" in kwargs:
            postprocess_kwargs["highlight_threshold"] = kwargs["highlight_threshold"]

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs):
        """预处理：Tokenize 输入文本"""
        model_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        model_inputs["text"] = inputs  # 保存原始文本供后处理使用
        return model_inputs

    def _forward(self, model_inputs):
        """前向推理"""
        with torch.no_grad():
            outputs = self.model(**model_inputs)

        outputs["model_inputs"] = model_inputs  # 传递给后处理
        return outputs

    def postprocess(self, model_outputs, top_k_keywords=5, highlight_threshold=0.05):
        """后处理：计算关键词重要性并格式化输出"""

        logits = model_outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_label_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_label_id].item()

        label = self.model.config.id2label[pred_label_id]

        # 计算每个 token 的重要性分数
        text = model_outputs.model_inputs["text"]
        tokens = self.tokenizer.tokenize(text)
        input_ids = model_outputs.model_inputs["input_ids"]

        importance_scores = []
        base_confidence = confidence

        for i in range(1, len(input_ids[0]) - 1):  # 跳过 [CLS] 和 [SEP]
            masked_input_ids = input_ids.clone()
            masked_input_ids[0, i] = self.tokenizer.mask_token_id

            with torch.no_grad():
                masked_outputs = self.model(
                    input_ids=masked_input_ids,
                    attention_mask=model_outputs.model_inputs.get("attention_mask"),
                )
                masked_probs = F.softmax(masked_outputs.logits, dim=-1)
                new_confidence = masked_probs[0, pred_label_id].item()

            importance = abs(base_confidence - new_confidence)
            importance_scores.append({
                "token": tokens[i - 1] if i - 1 < len(tokens) else "[UNK]",
                "position": i,
                "importance": importance,
                "direction": "positive" if new_confidence < base_confidence else "negative",
            })

        importance_scores.sort(key=lambda x: x["importance"], reverse=True)
        top_keywords = [s for s in importance_scores if s["importance"] > highlight_threshold][:top_k_keywords]

        highlighted_text = self._apply_highlight(text, top_keywords)

        return {
            "label": label,
            "score": confidence,
            "highlighted_text": highlighted_text,
            "keywords": [
                {
                    "word": kw["token"],
                    "importance": round(kw["importance"], 4),
                    "direction": kw["direction"],
                }
                for kw in top_keywords
            ],
            "all_token_scores": importance_scores,
        }

    def _apply_highlight(self, text, keywords):
        """在原文中用标记包裹关键词"""
        result = text
        for kw in keywords:
            word = kw["token"].replace("##", "")  ## BERT WordPiece 续接标记
            if word and len(word) > 1 and word not in ["[CLS]", "[SEP]", "[MASK]"]:
                result = result.replace(word, f"【{word}】")
        return result
```

## 第二步：注册到 PIPELINE_REGISTRY

```python
from transformers import PIPELINE_REGISTRY, AutoModelForSequenceClassification, AutoTokenizer

PIPELINE_REGISTRY.register_pipeline(
    "keyword-sentiment",
    pipeline_class=KeywordHighlightSentimentPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"model": "uer/roberta-base-finetuned-chinanews-chinese"},
)
```

## 第三步：使用自定义 Pipeline

```python
pipe = pipeline("keyword-sentiment")

texts = [
    "这家餐厅的服务态度非常好，菜品也很美味，强烈推荐给大家！",
    "产品质量太差了，用了两天就坏了，客服还不理人。",
    "还可以吧，没有特别惊艳的地方，价格倒是挺便宜的。",
]

for text in texts:
    result = pipe(text, top_k_keywords=5, highlight_threshold=0.03)

    print(f"\n{'='*60}")
    print(f"原文: {text}")
    print(f"\n预测: {result['label']} (置信度: {result['score']:.4f})")
    print(f"\n高亮文本: {result['highlighted_text']}")

    print(f"\n关键词影响力:")
    for kw in result["keywords"]:
        direction_icon = "↑" if kw["direction"] == "positive" else "↓"
        print(f"  {kw['word']:10s} | 影响力: {kw['importance']:.4f} {direction_icon}")
```

输出示例：
```
============================================================

原文: 这家餐厅的服务态度非常好，菜品也很美味，强烈推荐给大家！

预测: POSITIVE (置信度: 0.9876)

高亮文本: 这家餐厅的【服务】【态度】非常【好】，【菜品】也很【美味】，【强烈推荐】给大家！

关键词影响力:
  好       | 影响力: 0.1234 ↑
  推荐     | 影响力: 0.0891 ↑
  服务     | 影响力: 0.0678 ↑
  美味     | 影响力: 0.0543 ↑
  态度     | 影响力: 0.0456 ↑
```

这个自定义 Pipeline 展示了如何扩展标准功能——它在常规的情感分析基础上增加了**可解释性维度**，让用户不仅知道"结论是什么"，还知道"为什么得出这个结论"。

---

## 更实用的案例：多轮对话 Pipeline

下面是一个更贴近生产环境的例子——封装一个支持上下文管理的对话 Pipeline：

```python
class ConversationPipeline(Pipeline):
    """
    支持多轮上下文管理的对话 Pipeline

    特性:
    - 自动维护对话历史
    - 支持系统提示词
    - 支持最大历史长度限制
    - 自动截断过长的对话
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
        self.system_prompt = ""

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}

        if "system_prompt" in kwargs:
            preprocess_kwargs["system_prompt"] = kwargs["system_prompt"]
        if "max_history" in kwargs:
            preprocess_kwargs["max_history"] = kwargs["max_history"]
        if "clear_history" in kwargs:
            preprocess_kwargs["clear_history"] = kwargs["clear_history"]
        if "temperature" in kwargs or "top_p" in kwargs:
            forward_kwargs.update({
                k: v for k, v in kwargs.items()
                if k in ("temperature", "top_p", "repetition_penalty", "max_new_tokens")
            })

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs, system_prompt=None, max_history=10, clear_history=False):

        if clear_history:
            self.conversation_history = []

        if system_prompt:
            self.system_prompt = system_prompt

        user_message = {"role": "user", "content": inputs}
        self.conversation_history.append(user_message)

        if len(self.conversation_history) > max_history * 2:
            self.conversation_history = self.conversation_history[-(max_history * 2):]

        prompt = self._build_chat_prompt(self.conversation_history, self.system_prompt)

        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings - 256,
        )

        return model_inputs

    def _build_chat_prompt(self, history, system_prompt=""):
        lines = []
        if system_prompt:
            lines.append(f"[System] {system_prompt}")

        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"[{role}] {msg['content']}")

        lines.append("[Assistant]")
        return "\n".join(lines)

    def _forward(self, model_inputs):
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_text = self.tokenizer.decode(outputs[0][model_inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True)

        self.conversation_history.append({"role": "assistant", "content": response_text})

        return {"response": response_text}

    def postprocess(self, model_outputs):
        return {
            "response": model_outputs["response"],
            "history_length": len(self.conversation_history),
        }


# 使用示例
conv_pipe = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    pipeline_class=ConversationPipeline,
)

response = conv_pipe(
    "你好，我想了解一下深度学习的基础知识",
    system_prompt="你是一个友好的AI助手，擅长用简单易懂的方式解释技术概念。",
    temperature=0.7,
)

print(f"Assistant: {response['response']}")
print(f"(当前对话轮数: {response['history_length'] // 2})")

# 第二轮（自动携带上下文）
response2 = conv_pipe("能详细讲一下反向传播吗？")
print(f"Assistant: {response2['response']}")
```

---

## Pipeline 支持的快捷方法

除了完整地继承 `Pipeline` 类，HF 还提供了两种更轻量的方式来自定义行为：

## 方式 1：使用 `pipeline()` 的回调函数

对于只需要修改预处理或后处理的场景：

```python
def custom_preprocess(text):
    """自定义预处理函数"""
    text = text.lower().strip()
    text = "".join(c for c in text if c.isalnum() or c.isspace())
    return text

# 在调用前手动预处理
raw_texts = ["  Hello!!!  ", "WORLD...  ", "  TeSt InPuT  "]
cleaned = [custom_preprocess(t) for t in raw_texts]
results = pipeline("sentiment-analysis")(cleaned)
```

## 方式 2：组合多个 Pipeline

```python
class CompositeAnalysisPipeline:
    """组合多个 Pipeline 的复合分析器"""

    def __init__(self):
        self.sentiment = pipeline("text-classification",
                                   model="uer/roberta-base-finetuned-chinanews-chinese")
        self.ner = pipeline("ner",
                            aggregation_strategy="simple")
        self.emotion = pipeline("text-classification",
                                model="li168card/Micro-emotion-Chinese")

    def __call__(self, text):
        sentiment_result = self.sentiment(text)[0]
        ner_result = self.ner(text)
        emotion_result = self.emotion(text)[0]

        entities = [(e["word"], e["entity_group"]) for e in ner_result]

        return {
            "text": text,
            "sentiment": sentiment_result["label"],
            "confidence": sentiment_result["score"],
            "entities": entities,
            "emotion": emotion_result["label"],
            "emotion_score": emotion_result["score"],
        }

analyzer = CompositeAnalysisPipeline()

result = analyzer("马云今天在杭州阿里巴巴总部宣布退休，员工们纷纷表示不舍。")

print(json.dumps(result, ensure_ascii=False, indent=2))
```

输出：
```json
{
  "text": "马云今天在杭州阿里巴巴总部宣布退休，员工们纷纷表示不舍。",
  "sentiment": "POSITIVE",
  "confidence": 0.9234,
  "entities": [["马云", "PER"], ["杭州", "LOC"], ["阿里巴巴", "ORG"]],
  "emotion": "期待",
  "emotion_score": 0.7856
}
```

---

## 何时使用 Pipeline vs 直接调用 Model

这是一个重要的工程决策问题：

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 快速原型验证 / Demo | **Pipeline** | 最快上手，代码最少 |
| 标准的 NLP 任务（分类/NER/QA） | **Pipeline** | 内置的后处理已经足够完善 |
| 需要精细控制中间层输出 | **Model 直接调用** | 可以访问 hidden_states, attentions 等 |
| 训练/微调 | **Model 直接调用** | Trainer 需要 Model 对象，不需要 Pipeline |
| 需要自定义复杂的前后处理 | **自定义 Pipeline** | 封装复用性好 |
| 超低延迟的在线服务 | **Model 直接调用 + ONNX/TensorRT** | 去掉 Pipeline 的额外开销 |
| 多模型集成 / 级联 | **Pipeline 组合或自定义 Pipeline** | 便于管理 |

> **经验法则**：如果你发现自己在一个项目里反复写相同的"load model → tokenize → forward → post-process"样板代码，那就是该抽成一个自定义 Pipeline 的时候了。

---

## 常见错误与调试技巧

## 错误 1：忘记重写必要方法

```python
class BrokenPipeline(Pipeline):
    pass  # 什么都没重写

pipe = pipeline("text-classification", pipeline_class=BrokenPipeline)
pipe("Hello")  # NotImplementedError!
```

**必须至少重写的方法**：`preprocess`、`_forward`、`postprocess`（或者 `_sanitize_parameters` 如果需要自定义参数）

## 错误 2：preprocess 返回格式不正确

```python
def bad_preprocess(self, inputs):
    return inputs  # 返回字符串而非 tensor dict！

# 正确做法
def good_preprocess(self, inputs):
    return self.tokenizer(inputs, return_tensors="pt")  # 返回包含 input_ids 等的 dict
```

## 错误 3：_forward 中没有正确处理设备

```python
def broken_forward(self, model_inputs):
    outputs = self.model(**model_inputs)  # 可能在错误的设备上运行!

# 正确做法
def correct_forward(self, model_inputs):
    device = next(self.model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    outputs = self.model(**model_inputs)
    return outputs
```

## 调试工具：查看 Pipeline 的每一步输出

```python
class DebuggablePipeline(Pipeline):
    """带调试输出的 Pipeline 基类"""

    def __init__(self, *args, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug

    def preprocess(self, inputs):
        result = super().preprocess(inputs)
        if self.debug:
            print(f"[DEBUG preprocess] Input: {inputs[:50]}...")
            print(f"[DEBUG preprocess] Output keys: {list(result.keys())}")
            if "input_ids" in result:
                print(f"[DEBUG preprocess] input_ids shape: {result['input_ids'].shape}")
        return result

    def _forward(self, model_inputs):
        result = super()._forward(model_inputs)
        if self.debug:
            print(f"[DEBUG forward] Output type: {type(result)}")
            if hasattr(result, 'logits'):
                print(f"[DEBUG forward] logits shape: {result.logits.shape}")
        return result

    def postprocess(self, model_outputs):
        if self.debug:
            print(f"[DEBUG postprocess] Input type: {type(model_outputs)}")
        result = super().postprocess(model_outputs)
        if self.debug:
            print(f"[DEBUG postprocess] Final output: {result}")
        return result
```

---

## 小结

这一节我们掌握了自定义 Pipeline 的完整技能：

1. **四阶段架构**：`_sanitize_parameters`（参数分拣）→ `preprocess`（预处理）→ `_forward`（前向推理）→ `postprocess`（后处理），这是所有 Pipeline 共同遵循的设计模式
2. **完整实现**：从零创建了 `KeywordHighlightSentimentPipeline`（关键词高亮情感分析）和 `ConversationPipeline`（多轮对话管理）两个实用案例
3. **注册机制**：通过 `PIPELINE_REGISTRY.register_pipeline()` 注册自定义 Pipeline，之后可以用 `pipeline("my-task-name")` 调用
4. **轻量替代方案**：回调函数预处理、Pipeline 组合类，适用于不需要完整继承的场景
5. **选择指南**：快速原型用 Pipeline、训练用 Model、复杂业务逻辑用自定义 Pipeline、极致性能用 ONNX/TensorRT
6. **调试技巧**：创建 DebuggablePipeline 基类来追踪每一步的输入输出

下一节我们将深入 Pipeline 底层原理与性能优化，包括 ONNX Runtime 加速、并发推理、量化部署等生产级话题。
