# 数据增强与清洗：让数据发挥最大价值

## 这一节讲什么？

在实际项目中，你拿到的原始数据往往存在各种问题：标注噪声、类别不平衡、样本量不足、包含无关信息或有害内容等。直接把这样的数据送入模型训练，效果可想而知。

这一节，我们将系统性地学习如何**提升数据质量**和**扩充数据规模**：

1. **文本增强技术**：同义词替换、回译、随机噪声——用少量数据生成更多样本
2. **数据去重**：MinHash/SimHash 处理大规模文本去重
3. **数据平衡**：过采样/欠采样/focal loss 应对长尾分布
4. **质量过滤**：语言检测、长度过滤、毒性检测

---

## 一、文本数据增强

## 1.1 为什么需要数据增强？

在 NLP 任务中，标注数据是昂贵的资源。一个高质量的领域情感分析数据集可能需要数周的人工标注工作，成本高达数十万元。数据增强的核心思想是：

> **通过对现有样本进行有意义的变换，人工生成新的训练样本，从而在不增加标注成本的前提下扩大训练集。**

但要注意：**不是所有增强方法都适用于所有任务**。比如同义词替换可能改变情感极性（"好"→"棒"没问题，但某些上下文中可能有微妙差异），回译可能引入翻译腔。

## 1.2 同义词替换（Synonym Replacement）

```python
import random
from datasets import Dataset

class SynonymReplacer:
    """基于词典的同义词替换增强器"""

    def __init__(self, synonym_dict=None):
        self.synonym_dict = synonym_dict or {
            "好": ["棒", "优秀", "出色", "赞", "不错"],
            "差": ["烂", "糟糕", "差劲", "不行"],
            "喜欢": ["爱", "钟爱", "喜爱"],
            "讨厌": ["厌恶", "反感", "不喜欢"],
            "大": ["巨大", "庞大", "硕大"],
            "小": ["微小", "细小", "渺小"],
        }

    def augment(self, text, replace_ratio=0.3, n_augments=2):
        """对文本进行同义词替换增强"""
        words = list(text)
        results = [text]

        for _ in range(n_augments):
            new_words = words.copy()
            n_replace = max(1, int(len(words) * replace_ratio))

            positions = random.sample(range(len(words)), min(n_replace, len(words)))
            for pos in positions:
                word = words[pos]
                if word in self.synonym_dict:
                    new_words[pos] = random.choice(self.synonym_dict[word])

            results.append("".join(new_words))

        return results


replacer = SynonymReplacer()

test_texts = [
    "这个产品非常好",
    "质量太差了",
    "我很喜欢这个设计",
]

print("=== 同义词替换增强 ===")
for text in test_texts:
    augmented = replacer.augment(text, replace_ratio=0.4, n_augments=3)
    print(f"\n原文: {text}")
    for i, aug_text in enumerate(augmented[1:], 1):
        print(f"  增强{i}: {aug_text}")
```

## 1.3 回译增强（Back-Translation）

回译的思路是：中文 → 英文 → 中文（或者经过更多中间语言）。由于每次翻译都会引入一些变化，最终得到的文本与原文语义相近但表达不同：

```python
def demonstrate_back_translation():
    """演示回译增强的效果（模拟）"""
    from transformers import pipeline

    try:
        zh_en = pipeline("translation", model="Helsinki-nlp/opus-mt-zh-en", device=-1)
        en_zh = pipeline("translation", model="Helsinki-nlp/opus-mt-en-zh", device=-1)

        original = "这款手机拍照效果非常出色，电池续航也很给力"
        print(f"原文: {original}")

        english = zh_en(original)[0]["translation_text"]
        print(f"中→英: {english}")

        back_zh = en_zh(english)[0]["translation_text"]
        print(f"英→中: {back_zh}")

        print(f"\n变化点:")
        for a, b in zip(original, back_zh):
            if a != b:
                print(f"  '{a}' → '{b}'")

    except Exception as e:
        print(f"回译模型未完全加载: {e}")
        print("\n模拟回译结果示例:")
        examples = [
            ("这款手机拍照效果非常出色，电池续航也很给力",
             "这部手机的摄影功能十分优异，电池使用时间也相当不错"),
            ("客服态度很好，物流也很快",
             "客户服务态度佳，配送速度也很快"),
        ]
        for orig, bt in examples:
            print(f"\n  原文: {orig}")
            print(f"  回译: {bt}")

demonstrate_back_translation()
```

> **回译的优势与风险**：
>
> - ✅ 保持语义一致性较高（比随机替换更可靠）
> - ✅ 引入自然的句式变化（因为翻译模型学到了多种表达）
> - ❌ 需要加载额外的翻译模型（资源开销大）
> - ❌ 可能引入"翻译腔"（不够地道的表达）
> - ❌ 多次回译可能导致语义漂移

## 1.4 EDA（Easy Data Augmentation）

EDA 是一组简单但有效的文本增强操作组合：

```python
import random
import re

class EDAAugmentor:
    """
    Easy Data Augmentation: 四种简单操作的组合
    论文: EDA: Easy Data Augmentation Techniques for Boosting
          Classifier Performance on Text Classification Tasks
    """

    def __init__(self, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1,
                 p_rd=0.1, num_aug=2):
        self.alpha_sr = alpha_sr  # 同义词替换比例
        self.alpha_ri = alpha_ri  # 随机插入比例
        self.alpha_rs = alpha_rs  # 随机交换比例
        self.p_rd = p_rd         # 随机删除概率
        self.num_aug = num_aug   # 增强倍数

    def _get_words(self, text):
        return list(text)

    def _synonym_replace(self, words, synonym_dict):
        n_sr = max(1, int(self.alpha_sr * len(words)))
        new_words = words.copy()
        random_set = random.sample(list(range(len(new_words))), min(n_sr, len(new_words)))

        for idx in random_set:
            word = new_words[idx]
            if word in synonym_dict:
                new_words[idx] = random.choice(synonym_dict[word])

        return new_words

    def _random_insert(self, words, synonym_dict):
        n_ri = max(1, int(self.alpha_ri * len(words)))
        new_words = words.copy()

        for _ in range(n_ri):
            add_word = random.choice(list(synonym_dict.keys()))
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, add_word)

        return new_words

    def _random_swap(self, words):
        n_rs = max(1, int(self.alpha_rs * len(words)))
        new_words = words.copy()

        for _ in range(n_rs):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return new_words

    def _random_delete(self, words):
        if len(words) == 1:
            return words

        new_words = [w for w in words if random.random() > self.p_rd]
        if len(new_words) == 0:
            return [random.choice(words)]

        return new_words

    def augment(self, text, synonym_dict=None):
        synonym_dict = synonym_dict or {
            "的": "之", "很": "非常", "好": "棒", "不": "没",
            "是": "为", "了": "过", "在": "于", "和": "与",
        }

        words = self._get_words(text)
        aug_texts = []

        for _ in range(self.num_aug):
            a_words = words.copy()
            aug_type = random.choice(["sr", "ri", "rs", "rd"])

            if aug_type == "sr":
                a_words = self._synonym_replace(a_words, synonym_dict)
            elif aug_type == "ri":
                a_words = self._random_insert(a_words, synonym_dict)
            elif aug_type == "rs":
                a_words = self._random_swap(a_words)
            else:
                a_words = self._random_delete(a_words)

            aug_texts.append("".join(a_words))

        return aug_texts


eda = EDAAugmentor(num_aug=3)

texts = ["这个产品真的很好用", "今天天气非常不错"]
for text in texts:
    augmented = eda.augment(text)
    print(f"\n原文: {text}")
    for i, aug in enumerate(augmented, 1):
        print(f"  EDA-{i}: {aug}")
```

## 1.5 使用 Datasets 的 map 实现批量增强

```python
from datasets import load_dataset
from transformers import AutoTokenizer

def build_augmented_dataset(base_dataset, augment_fn, copies=2):
    """对数据集进行增强并扩展"""

    original_data = {"text": [], "label": []}

    for example in base_dataset:
        text = example["text"]
        label = example["label"]

        original_data["text"].append(text)
        original_data["label"].append(label)

        augmented_texts = augment_fn(text, n_augments=copies)
        for aug_text in augmented_texts:
            original_data["text"].append(aug_text)
            original_data["label"].append(label)

    augmented_ds = Dataset.from_dict(original_data)
    return augmented_ds


ds = load_dataset("imdb", split="train").select(range(100))
augmentor = SynonymReplacer()

augmented_ds = build_augmented_dataset(ds, augmentor.augment, copies=2)

print(f"原始数据集大小: {len(ds)}")
print(f"增强后数据集大小: {len(augmented_ds)}")
print(f"扩增倍数: {len(augmented_ds) / len(ds):.1f}x")
```

---

## 二、数据去重

大规模爬取的语料库中往往存在大量重复或高度相似的文本。去重不仅能减少存储空间和计算开销，更重要的是**避免模型在训练时对重复样本过拟合**。

## 2.1 精确去重

```python
def exact_dedup(dataset, text_col="text"):
    """精确去重（完全相同的文本只保留一条）"""
    seen = set()
    unique_indices = []

    for i, example in enumerate(dataset):
        text_hash = hash(example[text_col])
        if text_hash not in seen:
            seen.add(text_hash)
            unique_indices.append(i)

    deduped = dataset.select(unique_indices)
    removed = len(dataset) - len(deduped)

    print(f"精确去重: {len(dataset)} → {len(deduped)} (移除 {removed} 条重复)")
    return deduped
```

## 2.2 SimHash —— 近似去重

对于"大致相同但不完全一致"的文本（如同一篇文章的不同版本、同一事件的多篇报道），需要使用**近似去重**算法。SimHash 是工业界最常用的方案之一：

```python
class SimHashDeduplicator:
    """
    SimHash 近似去重器

    原理：
    1. 将文本分词得到特征集合
    2. 对每个特征计算 hash（固定长度的二进制向量）
    3. 将所有 hash 向量加权求和（正特征+1，负特征-1）
    4. 对结果向量每个维度：正→1，负→0 → 得到 fingerprint
    5. 两个 fingerprint 的汉明距离 < threshold 则视为近似重复
    """

    def __init__(self, hash_bits=64, similarity_threshold=0.85):
        self.hash_bits = hash_bits
        self.threshold = int(hash_bits * (1 - similarity_threshold))

    def _tokenize(self, text):
        simple_tokens = []
        current_token = []
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                if current_token:
                    simple_tokens.append(''.join(current_token))
                    current_token = []
                simple_tokens.append(ch)
            else:
                current_token.append(ch)
        if current_token:
            simple_tokens.append(''.join(current_token))
        return [t for t in simple_tokens if len(t) > 0]

    def _hash_feature(self, feature):
        md5_hash = hashlib.md5(feature.encode('utf-8')).digest()
        integer_value = int.from_bytes(md5_hash[:8], byteorder='big')
        return integer_value >> (64 - self.hash_bits)

    def compute_simhash(self, text):
        tokens = self._tokenize(text)
        vector = [0] * self.hash_bits

        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        for token, count in token_counts.items():
            h = self._hash_feature(token)
            for i in range(self.hash_bits):
                bitmask = 1 << i
                if h & bitmask:
                    vector[i] += count
                else:
                    vector[i] -= count

        fingerprint = 0
        for i in range(self.hash_bits):
            if vector[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint

    def hamming_distance(self, hash1, hash2):
        x = hash1 ^ hash2
        distance = 0
        while x:
            distance += 1
            x &= x - 1
        return distance

    def deduplicate(self, texts, show_progress=True):
        fingerprints = []
        unique_indices = []
        seen = set()

        total = len(texts)

        for i, text in enumerate(texts):
            fp = self.compute_simhash(text)
            is_duplicate = False

            for seen_fp in seen:
                if self.hamming_distance(fp, seen_fp) <= self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen.add(fp)
                unique_indices.append(i)

            if show_progress and (i + 1) % 1000 == 0:
                print(f"  进度: {i+1}/{total}, 已保留 {len(unique_indices)}, "
                      f"去重率 {(i+1-len(unique_indices))/(i+1)*100:.1f}%")

        return unique_indices


import hashlib

deduplicator = SimHashDeduplicator(hash_bits=64, similarity_threshold=0.90)

test_corpus = [
    "人工智能正在改变世界",
    "人工智能正在改变世界。",
    "AI正在改变我们的世界",
    "深度学习是人工智能的一个分支",
    "深度学习是 AI 的一个重要分支",
    "今天的天气真好",
    "自然语言处理很有趣",
    "NLP 是一个非常有趣的研究方向",
] * 10

indices = deduplicator.deduplicate(test_corpus)
unique_texts = [test_corpus[i] for i in indices]

print(f"\nSimHash 去重结果:")
print(f"  原始: {len(test_corpus)} 条")
print(f"  去重后: {len(unique_texts)} 条")
print(f"\n保留的唯一文本:")
for t in unique_texts:
    print(f"  • {t}")
```

---

## 三、数据不平衡处理

## 3.1 诊断不平衡问题

```python
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

def diagnose_class_imbalance(dataset, label_col="label"):
    """诊断数据集中的类别不平衡问题"""
    labels = dataset[label_col]
    counts = Counter(labels)

    unique_labels = sorted(counts.keys())
    frequencies = [counts[l] for l in unique_labels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    axes[0].bar(range(len(unique_labels)), frequencies, color=colors)
    axes[0].set_xticks(range(len(unique_labels)))
    axes[0].set_xticklabels([str(l) for l in unique_labels])
    axes[0].set_ylabel('Sample Count')
    axes[0].set_title('各类别样本数量')

    total = sum(frequencies)
    percentages = [f / total * 100 for f in frequencies]
    axes[1].pie(frequencies, labels=[str(l) for l in unique_labels],
               autopct='%1.1f%%', colors=colors)
    axes[1].set_title('类别占比')

    sorted_pct = sorted(percentages, reverse=True)
    imbalance_ratio = sorted_pct[0] / max(sorted_pct[-1], 1)
    axes[2].bar(range(len(sorted_pct)), sorted_pct, color='coral')
    axes[2].axhline(y=total/len(unique_labels)/total*100, color='green',
                   linestyle='--', label='均衡线')
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_title(f'不平衡程度 (最大/最小比={imbalance_ratio:.1f}x)')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    print(f"\n统计摘要:")
    print(f"  总样本数: {total}")
    print(f"  类别数: {len(unique_labels)}")
    print(f"  最多类: {max(frequencies)} 样本 ({max(percentages):.1f}%)")
    print(f"  最少类: {min(frequencies)} 样本 ({min(percentages):.1f}%)")
    print(f"  不平衡比: {imbalance_ratio:.1f}x")

    if imbalance_ratio > 10:
        print(f"  ⚠️ 严重不平衡！建议采取处理措施")
    elif imbalance_ratio > 3:
        print(f"  ⚠️ 轻度不平衡，建议关注少数类的表现")
    else:
        print(f"  ✓ 分布相对均衡")


ds = load_dataset("ag_news", split="train").select(range(2000))
diagnose_class_imbalance(ds, "label")
```

## 3.2 过采样（Oversampling）

```python
def oversample_minority_classes(dataset, label_col="label", strategy="random",
                                 random_seed=42):
    """
    过采样：复制少数类样本以平衡数据集

    Args:
        strategy: "random" (随机复制) 或 "smote" (需额外库)
    """
    random.seed(random_seed)
    labels = dataset[label_col]
    counts = Counter(labels)

    max_count = max(counts.values())

    all_indices = list(range(len(dataset)))
    extra_indices = []

    for label, count in counts.items():
        if count < max_count:
            class_indices = [i for i in all_indices if labels[i] == label]
            n_needed = max_count - count

            if strategy == "random":
                sampled = random.choices(class_indices, k=n_needed)
                extra_indices.extend(sampled)

    balanced_indices = all_indices + extra_indices
    random.shuffle(balanced_indices)

    balanced_ds = dataset.select(balanced_indices)

    new_counts = Counter(balanced_ds[label_col])
    print(f"过采样前: {dict(counts)}")
    print(f"过采样后: {dict(new_counts)}")

    return balanced_ds
```

## 3.3 欠采样（Undersampling）与加权损失

```python
def create_balanced_sampler(dataset, label_col="label"):
    """创建 WeightedRandomSampler 用于欠采样效果"""

    from torch.utils.data import WeightedRandomSampler
    import torch

    labels = dataset[label_col]
    counts = Counter(labels)

    label_to_count = dict(counts)
    weights = 1. / torch.tensor(
        [label_to_count[label.item()] for label in labels],
        dtype=torch.float
    )

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    return sampler


def compute_class_weights(dataset, label_col="label"):
    """计算用于损失函数加权的类别权重"""
    labels = dataset[label_col]
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)

    weights = {}
    for label, count in counts.items():
        weights[label] = total / (n_classes * count)

    print(f"类别权重 (少数类权重更高):")
    for label in sorted(weights.keys()):
        print(f"  类别 {label}: weight={weights[label]:.3f} (样本数={counts[label]})")

    return weights
```

---

## 四、质量过滤规则引擎

```python
class DataQualityFilter:
    """
    数据质量过滤器
    组合多条规则对数据进行清洗
    """

    def __init__(self, rules=None):
        self.rules = rules or {}
        self.stats = {"total": 0, "passed": 0, "rejected": {}}

    def add_rule(self, name, filter_fn, description=""):
        self.rules[name] = {"fn": filter_fn, "desc": description}

    def check_length(self, min_len=5, max_len=5000):
        """添加长度过滤规则"""
        def rule(example):
            text = example.get("text", "")
            return min_len <= len(text.strip()) <= max_len
        self.add_rule("length_check", rule,
                     f"长度在 [{min_len}, {max_len}] 之间")
        return self

    def check_language(self, target_lang="zh", threshold=0.7):
        """添加语言检测规则"""
        try:
            from langdetect import detect

            def rule(example):
                text = example.get("text", "").strip()
                if not text:
                    return False
                try:
                    return detect(text) == target_lang
                except:
                    return False
            self.add_rule("language_check", rule,
                         f"语言为 {target_lang}")
        except ImportError:
            pass
        return self

    def check_repetition(self, max_repeat_ratio=0.3):
        """添加重复内容过滤"""
        def rule(example):
            text = example.get("text", "").strip()
            if not text:
                return False
            chars = list(text)
            unique_chars = set(chars)
            repeat_ratio = 1 - len(unique_chars) / len(chars)
            return repeat_ratio < max_repeat_ratio
        self.add_rule("repetition_check", rule,
                     f"重复字符比例 < {max_repeat_ratio}")
        return self

    def check_special_patterns(self, blocked_patterns=None):
        """添加模式匹配规则"""
        import re
        patterns = blocked_patterns or [
            r"http[s]?://\S+",           # URL
            r"<[^>]+>",                  # HTML 标签
            r"[\U0001F600-\U0001F64F]",  # Emoji
            r"^[\s\W\d]+$",              # 纯符号/数字
        ]

        compiled = [re.compile(p) for p in patterns]

        def rule(example):
            text = example.get("text", "").strip()
            for pattern in compiled:
                if pattern.search(text):
                    return False
            return True

        self.add_rule("pattern_check", rule, "无 URL/HTML/特殊模式")
        return self

    def filter_dataset(self, dataset, verbose=True):
        """对整个数据集执行过滤"""

        self.stats = {"total": len(dataset), "passed": 0, "rejected": {}}

        def apply_all_rules(example):
            for name, rule_info in self.rules.items():
                if not rule_info["fn"](example):
                    self.stats["rejected"][name] = \
                        self.stats["rejected"].get(name, 0) + 1
                    return False

            self.stats["passed"] += 1
            return True

        filtered = dataset.filter(apply_all_rules, num_proc=4)

        if verbose:
            print("\n" + "=" * 50)
            print("数据质量过滤报告")
            print("=" * 50)
            print(f"总样本数: {self.stats['total']:,}")
            print(f"通过筛选: {self.stats['passed']:,} ({self.stats['passed']/self.stats['total']*100:.1f}%)")
            print(f"被拒绝: {self.stats['total'] - self.stats['passed']:,}")
            if self.stats["rejected"]:
                print("\n各规则拒绝数量:")
                for name, count in sorted(self.stats["rejected"].items(),
                                          key=lambda x: -x[1]):
                    pct = count / self.stats['total'] * 100
                    desc = self.rules[name]["desc"]
                    print(f"  {name:<20s} {count:>6d} ({pct:>5.1f}%) — {desc}")

        return filtered


filter_engine = DataQualityFilter()
filter_engine.check_length(min_len=3, max_len=2000)\
           .check_repetition(max_repeat_ratio=0.5)\
           .check_special_patterns()

dummy_data = {
    "text": [
        "这是一个正常的中文文本",
        "短",                          # 太短
        "A" * 3000,                    # 太长
        "访问 https://example.com 了解更多",  # 含URL
        "哈哈哈哈哈哈哈哈哈",           # 高重复
        "正常的产品评论，质量不错",
        "<script>alert(1)</script>",  # HTML标签
        "又一个正常样本",
    ],
    "label": [1, 0, 1, 0, 1, 1, 0, 1],
}
dummy_ds = Dataset.from_dict(dummy_data)

cleaned_ds = filter_engine.filter_dataset(dummy_ds)
```

输出示例：
```
==================================================
数据质量过滤报告
==================================================
总样本数: 8
通过筛选: 3 (37.5%)
被拒绝: 5

各规则拒绝数量:
  length_check            2 (25.0%) — 长度在 [3, 2000] 之间
  repetition_check        1 (12.5%) — 重复字符比例 < 0.5
  pattern_check           2 (25.0%) — 无 URL/HTML/特殊模式
```

---

## 小结

这一节我们系统学习了数据增强与清洗的关键技术：

1. **文本增强四件套**：同义词替换（简单有效）、回译增强（语义保持好但资源消耗大）、EDA（四种操作组合）、基于 `map` 的批量增强流水线
2. **SimHash 近似去重**：通过将文本映射为定长指纹再计算汉明距离来识别近似重复文本；精确去重只能处理完全相同的文本
3. **不平衡数据处理**：诊断（可视化分布）→ 过采样（复制少数类）→ 欠采样（`WeightedRandomSampler`）→ 加权损失（`compute_class_weights`）；不平衡比 >10x 时必须处理
4. **质量过滤规则引擎**：长度检查、语言检测、重复检测、模式匹配（URL/HTML 等）；可组合使用；建议始终在生产流程中加入基本的质量门控

下一节是第5章的最后一节——自定义数据集，包括上传到 Hub 和 Builder 模式。
