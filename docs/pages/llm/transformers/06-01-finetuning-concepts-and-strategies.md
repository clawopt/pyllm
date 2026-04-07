# 微调基本概念与策略：让预训练模型适应你的任务

## 这一节讲什么？

想象一下，你雇了一位博学多才的通才顾问——他懂物理、历史、编程、经济学，几乎无所不知。现在你需要他帮你处理一件非常具体的事情：判断客户评论是好评还是差评。这位顾问当然有能力做这件事，但他之前的知识太泛了，可能不知道你们行业的特定术语（比如"品控""售后"在你们领域的特殊含义），也不熟悉你们的评论风格。

**微调（Fine-Tuning）**就是让这位已经知识渊博的预训练模型，在你的具体数据上再学习一段时间，把他的通用能力"聚焦"到你的特定任务上。这个过程不需要从头训练——那就像让这位博士重新上小学一样浪费——而是基于他已经掌握的知识进行针对性的调整。

这一节，我们将理解微调的完整图景：
- 为什么微调有效？背后的原理是什么？
- 微调前需要做哪些准备工作？
- 常见的微调策略有哪些？
- 有哪些陷阱会导致微调失败？

---

## 一、从预训练到微调：两阶段范式的由来

## 1.1 一个直观的类比

让我们用一个更贴近生活的例子来理解预训练和微调的关系：

```
┌─────────────────────────────────────────────┐
│           预训练 (Pre-training)              │
│                                             │
│  相当于：一个人读了整个图书馆的所有书         │
│  学到了：语法、常识、世界知识、推理能力       │
│  花费：数周~数月，数千 GPU 小时               │
│  结果：一个"什么都知道一点"的通用模型          │
│                                             │
└───────────────────┬─────────────────────────┘
                    │
                    ▼ 在你的数据上继续学习
┌─────────────────────────────────────────────┐
│            微调 (Fine-tuning)                │
│                                             │
│  相当于：这个人去参加了你们公司的入职培训      │
│  学习了：公司术语、业务流程、特定格式           │
│  花费：几小时~几天，1 张 GPU                  │
│  结果：一个精通你领域任务的专家模型             │
│                                             │
└─────────────────────────────────────────────┘
```

## 1.2 为什么微调有效？三个核心原因

### 原因一：特征复用（Feature Reuse）

预训练模型（如 BERT）的底层学到的特征具有极强的通用性：

```python
from transformers import AutoModel, AutoTokenizer
import torch
from datasets import load_dataset

model = AutoModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

texts_from_different_domains = [
    "苹果公司发布了新款 iPhone",           # 科技
    "医生建议每天吃一个苹果",                 # 健康
    "这个苹果又红又大，口感很脆",             # 农业/食品
    "小明画了一个红色的苹果",                 # 教育/绘画
]

inputs = tokenizer(texts_from_different_domains,
                     return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state

# 取 [CLS] 位置的向量（句子级表征）
cls_embeddings = hidden_states[:, 0, :]

# 计算不同领域文本之间的语义相似度
cos_sims = torch.nn.functional.cosine_similarity(
    cls_embeddings.unsqueeze(1), cls_embeddings.unsqueeze(0), dim=-1
)

print("跨领域文本在 BERT 空间中的相似度矩阵:")
domains = ["科技", "健康", "食品", "教育"]
print(f"{'':>8}", end="")
for d in domains:
    print(f"{d:>8}", end="")
print()
for i, d1 in enumerate(domains):
    print(f"{d1:>8}", end="")
    for j in range(4):
        print(f"{cos_sims[i,j].item():>8.3f}", end="")
    print()
```

输出示例：
```
跨领域文本在 BERT 空间中的相似度矩阵:
              科技     健康     食品     教育
      科技     1.000    0.723    0.654    0.598
      健康     0.723    1.000    0.689    0.612
      食品     0.654    0.689    1.000    0.567
      教育     0.598    0.612    0.567    1.000
```

注意观察：即使"苹果"这个词在不同语境下含义完全不同，BERT 的表征仍然能够捕捉到**句法结构相似性**和**部分共享语义**。底层的 Transformer 层学到的是字符组合模式、短语结构等**与任务无关的基础特征**，这些特征在任何 NLP 任务中都有价值。

### 原因二：层次化知识（Hierarchical Knowledge）

预训练模型的不同层捕获了不同抽象级别的信息：

```python
def analyze_layer_wise_representations():
    """分析 BERT 不同层对同一个词的编码差异"""
    import matplotlib.pyplot as plt
    import numpy as np

    model = AutoModel.from_pretrained("bert-base-chinese",
                                     output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    text = "银行批准了这笔贷款"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    all_hidden = outputs.hidden_states  # (13 x batch x seq_len x 768)
    target_token_idx = text.index("银") + 1  # +1 for [CLS]

    layer_vectors = []
    for layer_id, hs in enumerate(all_hidden):
        vec = hs[0, target_token_idx, :].numpy()  # (768,)
        layer_vectors.append(vec)

    layer_vectors = np.array(layer_vectors)  # (13, 768)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(layer_vectors, aspect='auto', cmap='RdBu')
    axes[0].set_xlabel('Hidden Dimension')
    axes[0].set_ylabel('Layer (0=Embedding, 12=Final)')
    axes[0].set_title('"银行"在各层的隐藏状态热力图')
    axes[0].set_yticks(range(13))
    axes[0].set_yticklabels(['Emb'] + [f'L{i}' for i in range(12)])

    norms = np.linalg.norm(layer_vectors, axis=1)
    axes[1].plot(range(13), norms, 'o-', color='steelblue', markersize=6)
    axes[1].axvline(x=6, color='red', linestyle='--', alpha=0.7, label='~中层分界')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('L2 Norm')
    axes[1].set_title('"银行"向量范数随层数变化')
    axes[1].legend()

   相邻层相似度 = []
    for i in range(len(layer_vectors) - 1):
        cos_sim = np.dot(layer_vectors[i], layer_vectors[i+1]) / (
            np.linalg.norm(layer_vectors[i]) * np.linalg.norm(layer_vectors[i+1]) + 1e-8
        )
        相似度.append(cos_sim)

    axes[2].plot(range(12), 相似度, 's-', color='coral', markersize=5)
    axes[2].set_xlabel('Layer Transition')
    axes[2].set_ylabel('Cosine Similarity')
    axes[2].set_title('相邻层之间的表示变化幅度')

    plt.tight_layout()
    plt.show()

    print("\n各层编码的信息类型:")
    print("  Layer 0-3 (浅层):   字符/子词级别的局部模式（字形、n-gram）")
    print("  Layer 4-7 (中层):   短语级别语义（词义、语法角色）")
    print("  Layer 8-11 (深层):  句子级语义（上下文含义、指代消解）")
    print("  Layer 12 (输出层): 任务相关的组合特征")

analyze_layer_wise_representations()
```

这段代码揭示了一个关键事实：**BERT 的底层（Layer 0-3）学到的是几乎所有 NLP 任务都需要的"基础能力"**——比如怎么识别一个词的边界、怎么理解基本的语法关系。当你微调时，这些底层权重通常只需要小幅调整就能适配新任务，而主要的变化发生在高层。

这就是为什么微调只需要很少的数据和很短的时间——因为大部分"苦力活"（学习基础语言理解）已经在预训练阶段完成了。

### 原因三：迁移学习的数学保证

从优化理论的角度看，预训练模型的参数位于损失函数的一个**良好的初始区域**。随机初始化的参数可能在损失函数的任意位置（包括平坦区、尖锐极值点、鞍点），而预训练后的参数已经被推到了一个**泛化性能好的宽谷底部**附近。从这个起点开始微调，更容易找到新任务的优质解。

---

## 二、Full Fine-Tuning vs PEFT：两种路线的选择

## 2.1 Full Fine-Tuning —— 更新全部参数

这是最直接的微调方式：加载预训练权重后，**更新模型的所有参数**：

```
预训练模型 (110M 参数)
    │
    ▼ 加载到内存
全部参数可训练 ──────────────── 110M 个参数参与梯度更新
    │
    ▼ 在你的数据上训练几个 Epoch
    │
    ▼ 保存为新的 checkpoint
微调后模型 (110M 参数，全部被修改过)
```

**优点**：
- 表达能力最强——所有参数都可以根据任务调整
- 通常能达到该架构下的最优效果
- 实现简单，HF Trainer 一行搞定

**缺点**：
- **显存需求大**：需要存储所有参数的梯度（参数量的 2-3 倍显存）
- **容易过拟合**：小数据集上更新全部参数容易"忘掉"预训练知识
- **存储成本高**：每个微调任务都需要保存一份完整的模型副本

## 2.2 PEFT —— 只更新少量参数

PEFT（Parameter-Efficient Fine-Tuning）只更新模型的一小部分参数（通常 <1%），其余参数冻结：

```
预训练模型 (110M 参数)
    │
    ▼ 冻结大部分参数
只有 ~1M 参数可训练 ←────────── LoRA / Adapter / Prefix 等
    │
    ▼ 在你的数据上训练
    │
    ▼ 只保存新增的小参数 (~几 MB)
原始权重不变 + 小的 adapter 权重
```

我们将在第 7 章详细讨论 PEFT。这里先建立一个直觉：**对于大多数实际场景，PEFT 是更实用的选择**，除非你有大量数据和充足的计算资源。

## 2.3 如何选择？

| 因素 | 选 Full Fine-Tuning | 选 PEFT |
|------|---------------------|---------|
| 数据量 | > 10K 标注样本 | < 10K 或 < 1K |
| 计算资源 | 多 GPU / A100 单卡 | 消费级 GPU / 甚至 CPU |
| 任务差异 | 与预训练任务差异很大 | 与预训练任务相近 |
| 部署需求 | 每个任务独立部署模型 | 可共享基础模型 |
| 实验/迭代速度 | 慢（每次都要全量训练） | 快（只需训练 adapter） |

---

## 三、微调前的准备工作清单

## 3.1 任务定义

在写任何代码之前，你必须清楚地回答以下问题：

```python
def task_definition_checklist():
    """微调前的任务定义检查清单"""
    checklist = {
        "任务类型": {
            "问题": "这是什么类型的任务？",
            "选项": [
                "文本分类（单标签/多标签）",
                "序列标注（NER/POS）",
                "抽取式问答",
                "文本生成/摘要/翻译",
                "句子对分类（NLI/Similarity）",
                "回归任务（打分）",
            ],
        },
        "输入格式": {
            "问题": "模型的输入是什么？",
            "示例": "单条文本 / 文本对 / 问题+上下文 / 结构化输入",
        },
        "输出格式": {
            "问题": "期望的输出是什么？",
            "示例": "类别标签 / 标签序列 / 答案片段 / 生成文本 / 连续值",
        },
        "类别数量": {
            "问题": "如果是分类任务，有多少个类别？",
            "提示": "类别不平衡吗？是否有多标签？",
        },
        "评估指标": {
            "问题": "用什么指标衡量好坏？",
            "选项": ["Accuracy", "F1-Score", "Precision/Recall", "BLEU/ROUGE", "RMSE/MAE"],
        },
    }

    for section, info in checklist.items():
        print(f"\n{'='*50}")
        print(f"【{section}】{info['问题']}")
        print(f"{'='*50}")
        if "选项" in info:
            for opt in info["选项"]:
                print(f"  ☐ {opt}")
        if "示例" in info:
            print(f"  示例: {info['示例']}")
        if "提示" in info:
            print(f"  ⚠️  {info['提示']}")

task_definition_checklist()
```

## 3.2 数据准备——质量 > 数量

很多新手认为"数据越多越好"，但**数据质量远比数量重要**。100 条高质量、干净、有代表性的标注数据，往往比 10000 条噪声数据效果好得多。

```python
def data_quality_check(dataset, text_col="text", label_col="label"):
    """快速数据质量检查"""

    issues = []

    total = len(dataset)
    empty_count = sum(1 for ex in dataset if not ex[text_col] or not ex[text_col].strip())
    if empty_count > 0:
        issues.append(f"空文本: {empty_count} 条 ({empty_count/total*100:.1f}%)")

    lengths = [len(ex[text_col]) for ex in dataset]
    very_short = sum(1 for l in lengths if l < 3)
    very_long = sum(1 for l in lengths if l > 2000)
    if very_short > 0:
        issues.append(f"极短文本(<3字): {very_short} 条")
    if very_long > 0:
        issues.append(f"超长文本(>2000字): {very_long} 条")

    from collections import Counter
    labels = dataset[label_col]
    counts = Counter(labels)
    max_count = max(counts.values())
    min_count = min(counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    if imbalance_ratio > 5:
        issues.append(f"类别不平衡: 最大/最小比={imbalance_ratio:.1f}x")

    dup_texts = set()
    dup_count = 0
    seen = set()
    for ex in dataset:
        t = ex[text_col].strip()
        if t in seen:
            dup_count += 1
        seen.add(t)
    if dup_count > 0:
        issues.append(f"完全重复: {dup_count} 条")

    print("\n📊 数据质量报告:")
    print(f"  总样本数: {total}")
    print(f"  平均长度: {sum(lengths)/len(lengths):.1f} 字符")
    print(f"  中位长度: {sorted(lengths)[len(lengths)//2]} 字符")
    print(f"  类别分布: {dict(counts)}")

    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个潜在问题:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print(f"\n✅ 数据质量检查通过!")

    return len(issues) == 0
```

## 3.3 基线模型选择

选择基线模型时需要考虑的因素：

```python
def model_selection_guide(task_type, language="zh", constraint="balanced"):
    """
    模型选择指南

    Args:
        task_type: classification / ner / qa / generation / similarity
        language: zh / en / multilingual
        constraint: balanced / fast / small / large
    """

    recommendations = {
        ("classification", "zh", "balanced"): [
            ("hfl/chinese-roberta-wwm-ext", "中文 RoBERTa，分类任务 SOTA 级基线"),
            ("uer/chinese-roberta-LARGE-ext", "更大版本，数据充足时使用"),
        ],
        ("classification", "zh", "small"): [
            ("hfl/chinese-macbert-base", "MacBERT，较小但效果好"),
            ("uer/chinese-bert-wwm-ext-base", "经典 BERT 中文版"),
        ],
        ("ner", "zh", "balanced"): [
            ("hfl/chinese-roberta-wwm-ext", "通用 NER 基线"),
        ],
        ("qa", "zh", "balanced"): [
            ("hfl/chinese-roberta-wwm-ext", "中文 QA 常用基线"),
            ("shibing624/macbert-large-chinese-squad", "专门在 SQUAD 风格 QA 上微调过的"),
        ],
        ("generation", "zh", "balanced"): [
            ("THUDM/chatglm2-6b", "中文生成能力强"),
            ("Qwen/Qwen-7B-Chat", "阿里通义千问"),
        ],
        ("similarity", "zh", "balanced"): [
            ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "多语言 Sentence-BERT"),
        ],
    }

    key = (task_type, language, constraint)
    models = recommendations.get(key, [])

    print(f"\n推荐模型 (任务={task_type}, 语言={language}, 约束={constraint}):")
    for model_name, reason in models:
        print(f"  📦 {model_name}")
        print(f"     → {reason}")

    return models


model_selection_guide("classification", "zh", "balanced")
```

---

## 四、常见微调陷阱与避坑指南

## 陷阱 1：灾难性遗忘（Catastrophic Forgetting）

**现象**：微调后模型在原任务上的表现急剧下降，甚至不如随机猜测。

**原因**：学习率太大或训练太久，模型"忘记"了预训练阶段学到的通用知识。

**演示**：

```python
def demonstrate_catastrophic_forgetting():
    """模拟灾难性遗忘现象"""
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = nn.Linear(10, 8)
            self.classifier = nn.Linear(8, 2)

        def forward(self, x):
            feat = torch.relu(self.feature(x))
            return self.classifier(feat)

    def train_on_task(model, data, labels, lr, epochs, task_name):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        losses = []
        accuracies = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
            losses.append(loss.item())
            accuracies.append(acc)

        return losses, accuracies

    np.random.seed(42)
    torch.manual_seed(42)

    model = SimpleModel()

    # 模拟"预训练"——在一个通用任务上训练
    pre_data = torch.randn(500, 10)
    pre_labels = torch.randint(0, 2, (500,))
    pre_losses, pre_accs = train_on_task(model, pre_data, pre_labels, lr=0.01, epochs=50,
                                        task_name="pretrain")
    pre_final_acc = pre_accs[-1]

    # 模拟"微调"——在一个新任务上用高学习率训练
    ft_data = torch.randn(50, 10) * 0.5 + 2  # 分布不同的新数据
    ft_labels = torch.randint(0, 2, (50,))

    lr_to_test = [0.001, 0.01, 0.1, 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(lr_to_test)))

    for idx, lr in enumerate(lr_to_test):
        model_copy = SimpleModel()
        model_copy.load_state_dict(model.state_dict())

        ft_losses, ft_accs = train_on_task(model_copy, ft_data, ft_labels,
                                           lr=lr, epochs=30, task_name="finetune")

        # 测试在"原任务"上的表现（遗忘程度）
        with torch.no_grad():
            old_task_out = model_copy(pre_data)
            old_preds = old_task_out.argmax(dim=1)
            old_acc_after_ft = (old_preds == pre_labels).float().mean().item()

        axes[0].plot(ft_accs, '-', color=colors[idx], linewidth=2,
                   label=f'lr={lr} (新任务)')
        axes[1].bar(idx, old_acc_after_ft * 100, color=colors[idx],
                   label=f'lr={lr}: 原={old_acc_after_ft*100:.1f}%')

    axes[0].axhline(y=pre_final_acc * 100, color='blue', linestyle='--',
                   label=f'预训练后原任务准确率 ({pre_final_acc*100:.1f}%)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('New Task Accuracy (%)')
    axes[0].set_title('新任务上的学习曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(y=pre_final_acc * 100, color='blue', linestyle='--',
                   label=f'微调前 ({pre_final_acc*100:.1f}%)')
    axes[1].set_xticks(range(len(lr_to_test)))
    axes[1].set_xticklabels([f'lr={l}' for l in lr_to_test])
    axes[1].set_ylabel('Original Task Accuracy (%)')
    axes[1].set_title('微调后在原任务上的表现\n(越低 = 遗忘越严重)')
    axes[1].legend()
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.show()

    print("\n💡 关键发现:")
    print("  • 学习率越大，在新任务上学得越快，但在原任务上遗忘也越严重")
    print("  • 这就是为什么微调时通常使用较小的学习率 (1e-5 ~ 5e-5)")
    print("  • 解决方案: 更小的 lr / 更少的 epochs / 冻结底层 / 使用 PEFT")

demonstrate_catastrophic_forgetting()
```

## 陷阱 2：学习率设置不当

| 错误做法 | 后果 | 正确做法 |
|---------|------|---------|
| 用预训练时的 lr (如 1e-3) | 摧毁预训练权重 | **降低 10-100 倍**（2e-5 ~ 5e-5） |
| 所有层用相同 lr | 底层特征被破坏 | **分层学习率**（底层小，顶层大） |
| 不用 warmup | 初期梯度爆炸 | **线性 warmup** 前 5-10% 步 |
| lr 太小 (如 1e-7) | 几乎不学习 | 监控 loss 变化确认在下降 |

## 陷阱 3：验证集泄漏

```python
# ❌ 错误：在划分之前做了全局操作
all_data = load_my_dataset()
all_data = augment(all_data)  # 数据增强！
all_tokens = tokenize(all_data)  # Tokenize！

# 此时再做划分已经晚了——增强和 tokenize 的统计信息来自全部数据
train, val = train_test_split(all_data)  # 泄漏！

# ✅ 正确：先划分，再分别处理
raw_train, raw_val = train_test_split(raw_data)
train = augment(train).map(tokenize_fn)
val = val.map(tokenize_fn)  # 注意：val 通常不做增强
```

## 陷阱 4：Batch Size 太小导致训练不稳定

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def demonstrate_batch_size_effect():
    """展示 batch size 对训练稳定性的影响"""
    np.random.seed(42)

    bs_list = [1, 4, 16, 64]
    n_epochs = 20

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for bs in bs_list:
        simulated_loss = []
        param = 0.0

        for epoch in range(n_epochs):
            noise_scale = 1.0 / np.sqrt(bs)  # batch 越大，梯度估计越准
            grad = np.random.randn() * noise_scale + 0.05  # 真实梯度方向 + 噪声
            param -= 0.01 * grad
            loss = (param - 1.0) ** 2 + np.random.randn() * noise_scale * 0.1
            simulated_loss.append(loss)

        variance = np.var(simulated_loss[n_epochs//2:])
        axes[0].plot(simulated_loss, '-o', markersize=3, label=f'bs={bs} (var={variance:.4f})')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('不同 Batch Size 的 Loss 曲线稳定性')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    variances = []
    for bs in bs_list:
        noise_scale = 1.0 / np.sqrt(bs)
        losses = [(np.random.randn() * noise_scale * 0.1) for _ in range(100)]
        variances.append(np.var(losses))

    axes[1].bar([str(bs) for bs in bs_list], variances, color='coral')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Loss Variance')
    axes[1].set_title('Batch Size 越大，Loss 波动越小')

    plt.tight_layout()
    plt.show()

demonstrate_batch_size_effect()
```

**核心结论**：batch size 越小，梯度的方差越大（因为每个 batch 的统计量不够），loss 曲线波动越剧烈。这不仅是美观问题——**剧烈的波动可能导致训练发散或收敛到次优解**。

---

## 五、微调策略速查表

最后，我们把微调的核心决策整理成一个快速参考表：

```python
def finetuning_strategy_quickref():
    """微调策略速查表"""

    strategies = {
        "标准微调（推荐起点）": {
            "learning_rate": "2e-5",
            "epochs": "3-5",
            "batch_size": "16-32 (视显存)",
            "warmup_steps": "总步数的 5-10%",
            "weight_decay": "0.01",
            "max_grad_norm": "1.0",
            "scheduler": "linear (with warmup)",
            "precision": "fp16 / bf16",
            "gradient_accumulation": "如果显存不够，设为 2-8",
        },
        "小数据集 (< 500)": {
            "learning_rate": "1e-5 ~ 2e-5 (更保守)",
            "epochs": "5-10 (但要注意 overfitting)",
            "data_augmentation": "必须！",
            "regularization": "dropout + weight_decay + early stopping",
            "consider_peft": "强烈建议 LoRA / Adapter",
        },
        "大数据集 (> 50K)": {
            "learning_rate": "可以稍大 (3e-5 ~ 5e-5)",
            "epochs": "2-3 (数据量大不需要太多轮)",
            "batch_size": "尽可能大 (32-256)",
            "multi_gpu": "DDP / FSDP",
            "mixed_precision": "必须开启 fp16/bf16",
        },
        "分类任务": {
            "model_head": "ForSequenceClassification (自动添加)",
            "class_weights": "如果不平衡必须加",
            "metric": "F1 (不只是 Accuracy)",
        },
        "NER / 序列标注": {
            "model_head": "ForTokenClassification",
            "alignment": "确保 labels 和 token_ids 对齐",
            "ignore_index": "-100 用于非实体 token",
        },
        "QA 任务": {
            "model_head": "ForQuestionAnswering",
            "postprocessing": "提取 start/end logits 的 argmax",
            "overlap_handling": "允许答案跨越多个 token",
        },
    }

    for name, config in strategies.items():
        print(f"\n{'='*60}")
        print(f"📋 {name}")
        print('='*60)
        for key, value in config.items():
            print(f"  {key:<25s}: {value}")


finetuning_strategy_quickref()
```

---

## 小结

这一节建立了微调的完整认知框架：

1. **为什么微调有效**：三大原理——特征复用（底层特征通用性强）、层次化知识（浅层→深层逐级抽象）、优化理论（预训练提供了良好初始点）。通过代码实验展示了同一词在不同层的表征差异和跨领域相似性
2. **Full FT vs PEFT**：Full Fine-Tuning 更新全部参数（效果最好但成本高），PEFT 只更新 <1% 参数（实用优先）。选择依据：数据量、算力、任务差异度
3. **准备工作清单**：任务定义（类型/输入/输出/指标）、数据质量检查（空值/重复/长度/不平衡）、基线模型选择（按任务/语言/约束条件匹配）
4. **四大陷阱**：灾难性遗忘（lr 太大导致）、学习率不当（需比预训练低 10-100x）、验证集泄漏（先划分再处理）、batch size 过小（梯度方差大）
5. **策略速查表**：标准配置、小数据集策略、大数据集策略、以及分类/NER/QA 各类任务的专属注意事项

下一节，我们将正式进入代码实战——使用 HF Trainer API 从零完成一次完整的微调。
