# 知识蒸馏：Teacher-Student 框架与实战

## 这一节讲什么？

如果说量化是"压缩模型的存储体积"，那么**知识蒸馏（Knowledge Distillation）**就是"压缩模型的能力"——让一个小模型学会大模型的知识，同时保持远小于原模型的参数量。

这个思想非常优雅，可以用一个日常类比来理解：
> 一位资深教授（Teacher 模型）在讲课，他的学生（Student 模型）不需要达到教授的全部知识水平，只需要学到教授对每个问题的"判断方式"就够了。学生比教授轻量得多，但在大多数问题上能给出和教授相近的答案。

这一节我们将学习：
1. **蒸馏的三种形式**——Response-based / Feature-based / Relation-based
2. **温度参数（Temperature）的作用**——软化概率分布的关键
3. **DistilBERT 案例**——从 BERT-base 到 DistilBERT 的完整流程
4. **LLM 蒸馏的新趋势**——从 BERT 到 LLaMA 的蒸馏实践

---

## 一、知识蒸馏的核心直觉

## 1.1 为什么需要蒸馏？

```python
def why_distillation():
    """解释为什么需要知识蒸馏"""

    reasons = [
        {
            "场景": "边缘设备部署",
            "问题": "手机/物联网设备无法运行 7B 参数的大模型",
            "方案": "将 7B Teacher 的知识蒸馏到 0.5B Student",
        },
        {
            "场景": "低延迟推理",
            "问题": "在线服务需要 < 50ms 的响应时间",
            "方案": "小模型推理快 5~10 倍, 且效果接近大模型",
        },
        {
            "场景": "成本控制",
            "问题": "每次 API 调用大模型成本高昂",
            "方案": "部署自托管的小型蒸馏模型",
        },
        {
            "场景": "隐私保护",
            "问题": "数据不能发送到云端 API",
            "方案": "本地运行小型蒸馏模型",
        },
    ]

    print("=" * 70)
    print("为什么需要知识蒸馏?")
    print("=" * 70)
    for r in reasons:
        print(f"\n📌 {r['场景']}")
        print(f"   问题: {r['问题']}")
        print(f"   方案: {r['方案']}")

why_distillation()
```

## 1.2 蒸馏的本质：软标签 vs 硬标签

这是理解蒸馏最关键的概念。让我们通过一个具体例子来说明：

```python
import torch
import torch.nn.functional as F


def soft_label_vs_hard_label():
    """软标签 vs 硬标签的对比"""

    # 假设一个三分类任务: [猫, 狗, 兔子]
    teacher_logits = torch.tensor([4.0, 2.5, -1.0])  # Teacher 的输出 logits

    # 硬标签 (Hard Label): one-hot 编码, 只保留最大值
    hard_label = torch.zeros(3)
    hard_label[torch.argmax(teacher_logits)] = 1.0

    # 软标签 (Soft Label): 用 Softmax 得到的概率分布
    soft_label_t1 = F.softmax(teacher_logits / 1.0, dim=-1)   # T=1 (标准)
    soft_label_t2 = F.softmax(teacher_logits / 3.0, dim=-1)   # T=3 (高温)
    soft_label_t5 = F.softmax(teacher_logits / 10.0, dim=-1)  # T=10 (极高温)

    labels = ["猫", "狗", "兔子"]

    print("=" * 65)
    print("硬标签 vs 软标签: 信息量的差异")
    print("=" * 65)
    print(f"\nTeacher Logits: {teacher_logits.tolist()}")
    print(f"\n{'标签类型':<12} {'猫':>8} {'狗':>8} {'兔子':>8}")
    print("-" * 45)
    print(f"{'Hard Label':<12} {hard_label[0]:>8.3f} {hard_label[1]:>8.3f} {hard_label[2]:>8.3f}")

    for t, sl in [(1.0, soft_label_t1), (3.0, soft_label_t2), (10.0, soft_label_t5)]:
        print(f"{'Soft (T='+str(t)+')':<12} {sl[0]:>8.4f} {sl[1]:>8.4f} {sl[2]:>8.4f}")

    print(f"\n💡 关键洞察:")
    print(f"   • Hard Label 只告诉我们'这是猫'(100%), 其他信息全丢了")
    print(f"   • Soft Label (T=3): '猫(72%) > 狗(25%) > 兔子(3%)'")
    print(f"   → 学生不仅知道答案是猫, 还知道'狗也是有可能的, 但兔子不太可能'")
    print(f"   → 这种'暗知识'(Dark Knowledge) 是蒸馏的核心价值!")
    print(f"   • 温度越高, 分布越平坦, 包含的信息越丰富")
    print(f"   • 但温度太高也会引入噪声, 通常 T ∈ [2, 5]")

soft_label_vs_hard_label()
```

---

## 二、蒸馏的三种形式

## 2.1 全景图

```python
def distillation_types():
    """三种蒸馏形式的详解"""

    types = [
        {
            "名称": "Response-based (基于响应)",
            "原理": "让学生模仿 Teacher 的最终输出分布 (概率/Logits)",
            "损失函数": "KL(Student_output || Teacher_output)",
            "优点": "实现简单; 不需要访问 Teacher 内部结构",
            "缺点": "传递的信息有限; 只学到了'答案', 没学到'思考过程'",
            "代表工作": "Hinton 原始 KD 论文 (2015), DistilBERT",
        },
        {
            "名称": "Feature-based (基于特征)",
            "原理": "让学生模仿 Teacher 中间层的特征表示",
            "损失函数": "MSE(Student_hidden || Teacher_hidden)",
            "优点": "传递更多内部信息; 学生学到更深层的表示",
            "缺点": "需要 Student 和 Teacher 有相似的结构;"
                   "需要处理维度不匹配的问题 (投影层)",
            "代表工作": "FitNets (2016), BERT-PKD (2020)",
        },
        {
            "名称": "Relation-based (基于关系)",
            "原理": "保持样本之间的关系结构与 Teacher 一致",
            "损失函数": "cosine(S_i, S_j) ≈ cosine(T_i, T_j)",
            "优点": "不要求维度匹配; 关注样本间关系而非绝对值",
            "缺点": "计算复杂度高 (O(n²)); 难以训练收敛",
            "代表工作": "RKD (Relation Knowledge Distillation, 2019)",
        },
    ]

    print("=" * 85)
    print("三种知识蒸馏形式对比")
    print("=" * 85)
    for t in types:
        print(f"\n📐 {t['名称']}")
        for k, v in t.items():
            if k != "名称":
                print(f"   {k}: {v}")

distillation_types()
```

## 2.2 Response-based 蒸馏的数学推导

$$\mathcal{L}_{KD} = \alpha \cdot T^2 \cdot KL(p^{teacher}_T \| p^{student}_T) + (1-\alpha) \cdot CE(y_{true}, p^{student}_1)$$

其中：
- $p^{teacher}_T = \text{softmax}(z^{teacher}/T)$ —— Teacher 在温度 $T$ 下的软标签输出
- $p^{student}_T = \text{softmax}(z^{student}/T)$ —— Student 在温度 $T$ 下的输出
- $KL(\|)$ —— KL 散度
- $CE$ —— 标准交叉熵损失（使用硬标签）
- $\alpha$ —— 蒸馏损失和任务损失的平衡系数
- $T^2$ 因子用于补偿温度缩放导致的梯度变化

```python
def kd_loss_implementation():
    """手写 KD 损失函数的实现"""

    import torch.nn as nn
    import torch.nn.functional as F

    class KDLoss(nn.Module):
        """
        Knowledge Distillation Loss
        
        结合:
        1. KL Divergence Loss (蒸馏损失, 使用软标签)
        2. Cross Entropy Loss (任务损失, 使用硬标签)
        """

        def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
            super().__init__()
            self.temperature = temperature
            self.alpha = alpha
            self.kl_div = nn.KLDivLoss(reduction="batchmean")

        def forward(self, student_logits, teacher_logits, true_labels=None):
            """
            Args:
                student_logits: (batch, num_classes) 学生的原始 logits
                teacher_logits: (batch, num_classes) 教师的原始 logits
                true_labels: (batch,) 真实标签 (可选, 如果没有则只算 KL loss)
            """
            # 软化概率分布
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

            # KL 散度损失 (乘以 T² 补偿梯度)
            kl_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

            if true_labels is not None:
                # 标准交叉熵损失 (使用 T=1, 即正常温度)
                ce_loss = F.cross_entropy(student_logits, true_labels)
                total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
                return total_loss, kl_loss.item(), ce_loss.item()

            return kl_loss

    # 测试
    kd_loss = KDLoss(temperature=4.0, alpha=0.7)

    batch_size = 32
    num_classes = 10

    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes) * 2  # Teacher 更"自信"
    true_labels = torch.randint(0, num_classes, (batch_size,))

    total, kl, ce = kd_loss(student_logits, teacher_logits, true_labels)

    print("=" * 60)
    print("KD Loss 实现验证")
    print("=" * 60)
    print(f"\nBatch Size: {batch_size}, Classes: {num_classes}")
    print(f"Temperature: 4.0, Alpha: 0.7")
    print(f"\nTotal Loss:   {total.item():.4f}")
    print(f"  ├─ KL Loss (×α=0.7): {kl:.4f} × 0.7 = {kl*0.7:.4f}")
    print(f"  └─ CE Loss (×0.3):  {ce:.4f} × 0.3 = {ce*0.3:.4f}")

kd_loss_implementation()
```

---

## 三、DistilBERT：从理论到实践的完整案例

## 3.1 DistilBERT 的设计决策

DistilBERT 是 Hugging Face 团队发布的经典蒸馏模型。它的设计决策值得深入学习：

```python
def distillbert_design():
    """DistilBERT 的设计决策分析"""

    decisions = [
        ("Student 结构", "移除 Embedding 层 Pooler 和 NSP 任务头,"
         "保留 6 层 Encoder (原 BERT-base 有 12 层)", 
         "参数减少 ~40%"),
        ("初始化策略", "直接复制 Teacher 前 6 层的权重作为初始值"
         "(而不是随机初始化)", "训练起点更好"),
        ("蒸馏目标", "Triple Loss:\n"
         "  1. MLM Loss (掩码语言建模)\n"
         "  2. Cosine Embedding Loss (中间层表示)\n"
         "  3. KL Distillation Loss (输出分布)"),
        ("数据集", "动态掩码 + 同一数据做三次不同掩码\n"
         "(增大有效数据量 3x)"),
        ("训练技巧", "Teacher 用 EMA 更新 (而非固定);\n"
         "梯度共享给 Student 和 Teacher;\n"
         "使用 AdamW optimizer"),
    ]

    print("=" * 70)
    print("DistilBERT 设计决策解析")
    print("=" * 70)
    for title, detail, benefit in decisions:
        print(f"\n📌 {title}")
        print(f"   做法: {detail}")
        print(f"   收益: {benefit}")

distillbert_design()
```

## 3.2 完整的蒸馏训练代码模板

```python
import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset


class DistillationTrainer(Trainer):
    """
    自定义 Trainer 用于知识蒸馏
    
    同时优化三个损失:
    1. MLM Loss (学生自身的预训练任务)
    2. Distillation Loss (模仿教师输出的分布)
    3. Embedding Loss (中间层表示对齐)
    """

    def __init__(self, teacher_model, distill_weight=0.7, embed_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # 冻结教师!
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.distill_weight = distill_weight
        self.embed_weight = embed_weight
        self.temperature = 4.0
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs):
        """计算组合损失"""

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        # === Student 前向传播 ===
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
        )
        student_logits = student_outputs.logits
        student_hidden = student_outputs.hidden_states[-1]  # 最后一层

        # === Teacher 前向传播 ===
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs.logits
            teacher_hidden = teacher_outputs.hidden_states[-1]

        # === Loss 1: MLM Loss ===
        mlm_loss = student_outputs.loss if student_outputs.loss is not None else \
                  nn.CrossEntropyLoss()(student_logits.view(-1, student_logits.size(-1)), 
                                              labels.view(-1))

        # === Loss 2: Distillation Loss (KL Divergence) ===
        # 只在非 masked 位置计算
        mask = (labels != -100).float()
        active_positions = mask.unsqueeze(-1).expand_as(student_logits) > 0
        
        student_soft = torch.log_softmax(
            student_logits[active_positions].view(-1, student_logits.size(-1)) / self.temperature, 
            dim=-1
        )
        teacher_soft = torch.softmax(
            teacher_logits[active_positions].view(-1, teacher_logits.size(-1)) / self.temperature, 
            dim=-1
        )
        distill_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # === Loss 3: Embedding Cosine Loss ===
        # 对齐最后一个隐藏层的表示
        cos_sim = nn.CosineSimilarity(dim=-1)
        embed_cosine = 1 - cos_sim(
            student_hidden[mask.bool()],
            teacher_hidden[mask.bool()]
        ).mean()

        # === 组合损失 ===
        total_loss = (
            (1 - self.distill_weight - self.embed_weight) * mlm_loss
            + self.distill_weight * distill_loss
            + self.embed_weight * embed_cosine
        )

        return total_loss


def run_distillation_training():
    """执行完整的蒸馏训练"""

    # Teacher: BERT-base (12 layers, 110M params)
    teacher_name = "bert-base-uncased"
    teacher_model = AutoModelForMaskedLM.from_pretrained(teacher_name)
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)

    # Student: 初始化为 Teacher 的前 6 层 (也可以用一个更小的架构)
    from transformers import BertConfig, BertForMaskedLM

    student_config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=6,      # 只有 6 层!
        num_attention_heads=12,
        intermediate_size=3072,
    )

    student_model = BertForMaskedLM(student_config)

    # 初始化 Student 权重 (复制 Teacher 的前 6 层)
    with torch.no_grad():
        for i in range(6):
            student_model.bert.encoder.layer[i].load_state_dict(
                teacher_model.bert.encoder.layer[i].state_dict()
            )
        student_model.bert.embeddings.load_state_dict(
            teacher_model.bert.embeddings.state_dict()
        )

    print("=" * 60)
    print("Distillation Training 配置")
    print("=" * 60)
    print(f"\nTeacher: {teacher_name} (~110M params)")
    print(f"Student: Custom 6-layer BERT (~66M params)")
    print(f"参数压缩比: ~40%")

    # 加载数据
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:50000]")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding=False,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir="./distill_output",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        distill_weight=0.7,
        embed_weight=0.1,
    )

    trainer.train()

    student_model.save_pretrained("./distilled_student")
    tokenizer.save_pretrained("./distilled_student")
    print("\n✅ 蒸馏完成! Student 模型已保存")


if __name__ == "__main__":
    run_distillation_training()
```

---

## 四、LLM 蒸馏的新趋势与实践

## 4.1 从 BERT 到 LLaMA：蒸馏范式变了什么？

```python
def llm_distillation_trends():
    """LLM 蒸馏的新趋势"""

    trends = [
        {
            "阶段": "传统 NLP 蒸馏 (2015~2020)",
            "Teacher/Student": "同架构, 不同规模 (如 BERT-base → DistilBERT)",
            "方法": "Feature-based + Response-based 组合",
            "关键论文": "DistilBERT (HuggingFace, 2019), TinyBERT (2020)",
            "特点": "Student 和 Teacher 结构相似; 中间层可直接对齐",
        },
        {
            "阶段": "早期 LLM 蒸馏 (2022~2023)",
            "Teacher/Student": "GPT-3/GPT-J → Alpaca/Vicuna (指令微调后)",
            "方法": "主要用合成数据微调 Student (API-generated data)",
            "关键论文": "Alpaca (Stanford, 2023), Vicuna (LMSYS, 2023)",
            "特点": "不再强调中间层对齐; 重点在于高质量指令数据的生成",
        },
        {
            "阶段": "当前前沿 (2024)",
            "Teacher/Student": "闭源大模型 (GPT-4/Claude) → 开源中小模型",
            "方法": "多轮迭代蒸馏 + Self-Play + Mixture of Experts",
            "关键论文": "DistillMistral, Phi-3 (Microsoft), Gemma (Google)",
            "特点": "合成数据质量成为核心竞争点; 多 Teacher 融合; 规则增强",
        },
    ]

    print("=" * 80)
    print("知识蒸馏技术演进")
    print("=" * 80)
    for t in trends:
        print(f"\n📅 {t['阶段']}")
        for k, v in t.items():
            if k != "阶段":
                print(f"   {k}: {v}")

llm_distillation_trends()
```

## 4.2 Phi-3 的蒸馏方法论

微软的 Phi-3 系列是目前最成功的 LLM 蒸馏案例之一。它的核心方法论：

```python
def phi3_methodology():
    """Phi-3 蒸馏方法论拆解"""

    methodology = """
┌──────────────────────────────────────────────────────────────┐
│                    Phi-3 蒸馏方法论                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 数据质量优先 (Data-Centric AI)                           │
│     • 不追求海量数据, 追求极高质量的教科书级别数据             │
│     • 来源: 教科书、代码仓库、经过验证的 Q&A 对              │
│     • 过滤: 去除低质量、重复、有毒内容                        │
│                                                              │
│  2. 多样化的 Teacher 模型                                    │
│     • GPT-4 (通用能力)                                        │
│     • CodeLlama (代码能力)                                    │
│     • 专门训练的数学 Teacher                                  │
│     → 每个 Teacher 负责自己擅长的领域                          │
│                                                              │
│  3. 课程式训练 (Curriculum Learning)                          │
│     • 先学简单的 (基础语言理解)                               │
│     → 再学中等的 (推理能力)                                   │
│     → 最后学难的 (复杂指令遵循)                               │
│                                                              │
│  4. 合成数据 + 真实数据混合                                  │
│     • 约 90% 高质量合成数据                                    │
│     • 约 10% 精选真实数据 (防止过度拟合合成数据的偏差)          │
│                                                              │
│  结果: 3.8B 参数的 Phi-3-mini 在多项基准上超越 GPT-3.5       │
│        且能在 iPhone 上流畅运行                                │
└──────────────────────────────────────────────────────────────┘
"""
    print(methodology)

phi3_methodology()
```

---

## 五、常见误区与面试高频问题

## 5.1 误区

```python
def distillation_gotchas():
    """知识蒸馏中的常见误区"""

    gotchas = [
        {
            "误区": "Teacher 必须比 Student 大很多",
            "真相": "Teacher 和 Student 可以规模接近甚至相同!"
                   "同规模的蒸馏可以提升 Student 的泛化能力和鲁棒性",
        },
        {
            "误区": "蒸馏只能压缩模型, 不能提升性能",
            "真相": "蒸馏后的 Student 往往比同等大小独立训练的模型效果更好"
                   "(因为继承了 Teacher 的暗知识和正则化效应)",
        },
        {
            "误区": "温度越高越好",
            "真相": "T 太高 (>10) 会把所有预测变成均匀分布, 失去区分性;"
                   "T 太低 (<2) 则退化为 Hard Label, 失去软标签的优势;"
                   "推荐范围: T ∈ [2, 5]",
        },
        {
            "误区": "蒸馏完成后就可以丢弃 Teacher",
            "真相": "生产中有时会保留 Teacher 做 Ensemble;"
                   "或者用 Teacher 继续生成新数据来进一步改进 Student",
        },
    ]

    print("=" * 75)
    print("知识蒸馏常见误区")
    print("=" * 75)
    for g in gotchas:
        print(f"\n❌ {g['误区']}")
        print(f"   💡 {g['真相']}")

distillation_gotchas()
```

## 5.2 面试高频问答

```python
def interview_qa():
    """知识蒸馏面试高频问题"""

    qa = [
        {
            "Q": "什么是'暗知识'(Dark Knowledge)?",
            "A": "Teacher 输出的软标签中包含的信息:"
               "除了'正确答案是什么'之外, 还有'错误答案之间的相对关系'。"
               "例如 Teacher 认为'猫 > 狗 >> 兔子', 这个排序关系就是暗知识。"
               "Hard Label 丢失了这些信息, 而 Soft Label 保留了它。",
        },
        {
            "Q": "蒸馏损失中的 T² 因子为什么要乘?",
            "A": "因为 Softmax 的梯度会被 1/T 缩小。当 T > 1 时,"
               "Softmax 输出的梯度变小了, 为了补偿这个缩放并保持"
               "蒸馏损失和任务损失在同一数量级, 需要乘以 T²。",
        },
        {
            "Q": "什么时候应该选择 Feature-based 而不是 Response-based?",
            "A": "当 Student 和 Teacher 结构相似时, Feature-based 能传递更多信息;"
               "当结构差异较大时, Feature-based 难以对齐, Response-based 更实用;"
               "实践中通常两者结合使用。",
        },
        {
            "Q": "DistilBERT 比 BERT-base 快多少? 精度下降多少?",
            "A": "DistilBERT 保留了 97% 的性能, 但速度快 60%, 参数少 40%。"
               "这是通过:(1) 移除 NSP 任务和 pooler;(2) 层数减半;(3)"
               "三重损失函数(MLM+Distill+Embedding)实现的。",
        },
    ]

    print("=" * 78)
    print("知识蒸馏面试高频问题")
    print("=" * 78)
    for q in qa:
        print(f"\n📝 {q['Q']}")
        print(f"   💡 {q['A']}")

interview_qa()
```

---

## 六、本章小结

这一节我们系统学习了知识蒸馏的完整知识体系：

| 形式 | 核心机制 | 信息传递 | 适用场景 | 代表工作 |
|------|---------|---------|---------|---------|
| **Response-based** | KL(softmax(z_s/T) ‖ softmax(z_T/T)) | 输出分布 | 通用场景 | Hinton KD, DistilBERT |
| **Feature-based** | MSE(h_s, h_T) 或 Cosine 相似度 | 中间层表示 | 结构相似时 | FitNets, BERT-PKD |
| **Relation-based** | 保持样本间关系一致 | 全局结构 | 特殊需求 | RKD |

**核心要点**：
1. **软标签（Soft Label）是蒸馏的灵魂**——它包含了 Hard Label 无法表达的"暗知识"
2. **温度参数 T 控制信息的丰富程度**——T 越大分布越平，但过大会引入噪声
3. **DistilBERT 是工业级蒸馏的最佳参考**——40% 参数减少，97% 性能保留
4. **当前 LLM 蒸馏的趋势是从"结构蒸馏"转向"数据蒸馏"**——高质量合成数据成为核心竞争力

下一节我们将学习 **模型剪枝（Pruning）**——另一种重要的模型压缩技术。
