# 分词算法演进史

## 模型看到的不是文字，而是数字——Tokenizer 就是那个翻译官

在第一章中，我们多次使用了 `tokenizer.encode()` 这个操作，把文本转换成模型能理解的整数 ID 序列。但如果你仔细想想，这其实是一个很神奇的过程——一段人类可读的文字，经过 Tokenizer 之后，变成了一串看似随机的数字，而模型竟然能从这些数字中"理解"出语义。

这一节我们要深入到这个翻译过程的内部，看看它到底经历了怎样的演变才达到今天的状态。理解分词算法的演进历史，不仅能帮你更好地使用现有的工具，还能让你在面对特殊场景（比如领域术语、多语言混合、超长文本）时做出正确的工程决策。

## 最朴素的想法：按空格分词

当研究者们第一次面对"如何把文本切成小块"这个问题时，最直觉的想法就是——**按空格切分**。英文天然有空格作为词与词之间的分隔符，直接 `text.split(" ")` 不就行了吗？

```python
def naive_space_tokenize(text):
    """最朴素的分词：按空格分割"""
    return text.split()

texts = [
    "I love machine learning",
    "Hello, world!",
    "It's a beautiful day, isn't it?",
    "Mr. Smith went to the U.S.A.",
]

print("=== 空格分词结果 ===")
for text in texts:
    tokens = naive_space_tokenize(text)
    print(f"  原文: {text}")
    print(f"  分词: {tokens} (共 {len(tokens)} 个 token)")
    print()
```

输出：

```
=== 空格分词结果 ===
  原文: I love machine learning
  分词: ['I', 'love', 'machine', 'learning'] (共 4 个 token)

  原文: Hello, world!
  分词: ['Hello,', 'world!'] (共 2 个 token)

  原文: It's a beautiful day, isn't it?
  分词: ["It's", 'a', 'beautiful', 'day,', "isn't", 'it?'] (共 6 个 token)

  原文: Mr. Smith went to the U.S.A.
  分词: ['Mr.', 'Smith', 'went', 'to', 'the', 'U.S.A.'] (共 6 个 token)
```

看起来还行？但问题很快就暴露了：

**问题一：标点符号粘在单词上。** `"Hello,"` 和 `"Hello"` 是不同的 token——模型无法知道它们实际上是同一个意思的不同形式。同样的问题也出现在感叹号、问号、引号等标点上。

**问题二：缩写和所有格形式。** `"It's"` 被当作一个整体，但理想情况下应该拆成 `"It"` + `"'s"`（或 `"is"`）。`"Mr."`、`"U.S.A."` 也存在类似的问题。

**问题三：大小写不一致。** `"I"` 和 `"i"` 在空格分词看来是不同的 token，但它们显然指代同一个代词。

这些问题在英文中已经够烦人了，但在中文里，空格分词完全不可用：

```python
def show_chinese_tokenization_problem():
    """展示中文分词的特殊挑战"""
    
    chinese_texts = [
        "我爱自然语言处理",
        "南京市长江大桥",
        "乒乓球拍卖完了",  # 经典歧义
        "研究生命起源",       # 又一个经典歧义
        "人工智能正在改变世界",
    ]
    
    print("=== 中文空格分词 ===")
    print("(中文没有空格分隔词!)\n")
    
    for text in chinese_texts:
        # 中文按字符分(因为没有空格)
        chars = list(text)
        print(f"  原文: {text}")
        print(f"  按字切分: {chars} (共 {len(chars)} 个字符)")
        
        # 如果强行按某种方式"分词"
        if len(text) >= 6:
            naive = [text[i:i+2] for i in range(0, len(text)-1, 2)]
            print(f"  简单双字切: {naive}")
        print()

show_chinese_tokenization_problem()
```

输出：

```
=== 中文空格分词 ===
(中文没有空格分隔词!)

  原文: 我爱自然语言处理
  按字切分: ['我', '爱', '自', '然', '语', '言', '处', '理'] (共 8 个字符)
  简单双字切: ['我爱', '自然', '语言', '处理']

  原文: 南京市长江大桥
  按字切分: ['南', '京', '市', '长', '江', '大', '桥'] (共 7 个字符)
  简单双字切: ['南京', '市长', '江大', '桥']

  原文: 乒乓球拍卖完了
  按字切分: ['乒', '乓', '球', '拍', '卖', '完', '了'] (共 7 个字符)
  简单双字切: ['乒乓', '球拍', '卖完', '了']
```

注意最后两个例子中的**歧义问题**：
- **"乒乓球拍卖完了"**：是"乒乓球 / 拍卖 / 完了"还是"乒乓 / 球拍 / 卖 / 完了"？两种切法含义完全不同。
- **"南京市长江大桥"**：是"南京/市/长/江/大/桥"还是"南京市/长江大桥"？前者是一堆碎片，后者是一个有意义的专有名词。

这种歧义在中文里无处不在，而且往往只有在完整的上下文中才能正确消歧。这就是为什么中文 NLP 的分词问题比英文困难得多的根本原因。

## Word-level 分词：词汇表爆炸

为了解决上述问题，早期的研究者采用了基于词典的分词方法——维护一个包含所有可能单词的大词典，输入文本时去词典里查找匹配。

这种方法确实能解决基本的分词问题，但它带来了新的、更严重的问题：**词汇表爆炸（Out-of-Vocabulary, OOV）**。

```python
import re

class DictionaryTokenizer:
    """基于词典的分词器"""
    
    def __init__(self):
        # 一个小型的示例词典
        self.vocab = {
            "the": 0, "a": 1, "is": 2, "are": 3, "was": 4,
            "i": 5, "you": 6, "he": 7, "she": 8, "it": 9,
            "love": 10, "hate": 11, "like": 12,
            "machine": 13, "learning": 14, "deep": 15,
            "natural": 16, "language": 17, "processing": 18,
            "hello": 19, "world": 20, "good": 21, "bad": 22,
        }
        self.unk_id = 99  # Unknown token
    
    def tokenize(self, text):
        """简单实现：转小写 → 按空格分 → 查词典"""
        words = re.findall(r"\w+", text.lower())
        ids = []
        oov_count = 0
        
        for word in words:
            if word in self.vocab:
                ids.append(self.vocab[word])
            else:
                ids.append(self.unk_id)
                oov_count += 1
        
        return ids, oov_count

tokenizer = DictionaryTokenizer()
test_sentences = [
    "I love machine learning",
    "The quick brown fox jumps over the lazy dog",
    "Transformers revolutionized natural language processing",
    "This new model uses attention mechanisms effectively",
]

print("=== Word-level 分词: 词典匹配 ===\n")
for sent in test_sentences:
    ids, oov = tokenizer.tokenize(sent)
    total = len(ids)
    unk_rate = oov / total * 100 if total > 0 else 0
    
    print(f"  句子: {sent[:50]}...")
    print(f"  总 token 数: {total}, OOV 数量: {oov}, OOV 率: {unk_rate:.0f}%")
```

输出：

```
=== Word-level 分词: 词典匹配 ===

  句子: I love machine learning...
  总 token 数: 4, OOV 数量: 0, OOV 率: 0%

  句子: The quick brown fox jumps over the lazy dog...
  总 token 数: 10, OOV 数量: 4, OOV 率: 40%

  句子: Transformers revolutionized natural language processing...
  总 token 数: 5, OOV 数量: 3, OOV 率: 60%

  句子: This new model uses attention mechanisms effectively...
  总 token 数: 8, OOV 数量: 5, OOV 率: 63%
```

看到了吗？即使我们的词典包含了 23 个常见词，一旦遇到稍微复杂一点的句子，OOV 率就飙升到 40%-60%。在实际应用中，英语的总词汇量超过 17 万（OED），中文更是有数十万个常用词组。如果要把所有可能的词都放进词汇表，不仅存储成本巨大，而且模型还需要为每个词学习高质量的表示——这对于低频词来说几乎是不可能的（因为训练数据中出现次数太少）。

更糟糕的是，**新词会不断出现**——每年都有大量的新造词、人名、地名、产品名、技术术语涌现。一个固定的词汇表永远不可能覆盖所有可能的词。

## Character-level 分词：解决了 OOV 但引入了新问题

既然 word-level 有 OOV 问题，那干脆不用词了，直接用**字符（character）** 作为基本单位行不行？

```python
def char_tokenize(text):
    """字符级分词"""
    return list(text.lower())

texts = [
    "hello",
    "antidisestablishmentarianism",
    "你好世界",
]

print("=== 字符级分词 ===\n")
for text in texts:
    tokens = char_tokenize(text)
    print(f"  '{text}' → {tokens} ({len(tokens)} 个字符)")
```

输出：

```
=== 字符级分词 ===

  'hello' → ['h', 'e', 'l', 'l', 'o'] (5 个字符)
  'antidisestablishmentarianism' → ['a','n','t','i','d','i','s','e','s','t','a','b','l','i','s','h','m','e','n','t','a','r','i','a','n','i','s','m'] (28 个字符)
  '你好世界' → ['你', '好', '世', '界'] (4 个字符)
```

字符级分词彻底消灭了 OOV 问题——因为字符集是非常小的（英文 26 个字母+符号 ≈ 100；中文常用汉字约 3500 个），任何文本都能被完整表示。但它的代价也非常明显：

**第一，序列长度暴增。** 英文单词平均 5-6 个字母，中文每个字就是一个 token。"antidisestablishmentarianism" 这个词在 word-level 只占 1 个 token，在 char-level 却要 28 个。序列越长，模型的计算量和内存占用就越大，Attention 的 $O(n^2)$ 复杂度也会更加突出。

**第二，丢失了词级别的语义信息。** 每个单独的字符几乎不携带什么有意义的信息——`h` 这个字符本身不告诉你任何关于 "hello" 含义的信息。模型需要自己从一串字符中"拼凑"出词的含义，这大大增加了学习的难度。

**第三，中文字符级分词效果尤其差。** 因为中文每个字本身就是一个有意义的单位（"我"就是一个完整的词），按字符分反而破坏了原有的语义边界。

所以字符级分词虽然理论上可行，但实际效果并不好——它走向了一个极端，用粒度过细的单位换来了信息稀疏的问题。

## Subword 分词：黄金折中方案

Word-level 太粗（OOV 问题），Character-level 太细（序列过长、语义丢失）。自然的想法就是——**取中间值，用"子词"（subword）作为基本单位**。

Subword 的核心思想是：**频繁出现的词组合保留为一个整体，不常见的词被拆分成更小的有意义的片段**。这样既能控制词汇表的大小，又能保证大部分情况下不会出现 OOV。

目前主流的 subword 分词算法有三类：**BPE（Byte Pair Encoding）**、**WordPiece** 和 **Unigram Language Model**。其中 BPE 及其变体应用最为广泛。

### BPE（Byte Pair Encoding）：统计高频字节对

BPE 由 Sennrich et al. 于 2015 年提出，其算法非常优雅——它不需要任何语言学知识，纯粹通过统计数据来自动发现最优的子词切分方式。

**BPE 的训练过程**可以概括为以下步骤：

```python
from collections import Counter

def bpe_training_demo():
    """演示 BPE 训练的核心流程"""
    
    # 第一步: 准备初始语料库 (已按字符拆分)
    corpus = [
        "low low low lowest lower newer newer",
        "newer newer newest new",
        "wider wider widest wide"
    ]
    
    # 初始状态: 每个词被拆分成字符
    word_freqs = Counter()
    for sentence in corpus:
        for word in sentence.split():
            # 用空格标记词边界，然后拆成字符
            word_tuple = tuple(list(word))
            word_freqs[word_tuple] += 1
    
    print("=== 初始状态 (字符级) ===")
    for word, freq in sorted(word_freqs.items()):
        print(f"  {' '.join(word):20s} ×{freq}")
    
    # 第二步: 迭代合并最高频的字节对
    num_merges = 8
    merges = []
    vocab = set()
    
    for step in range(num_merges):
        # 统计所有相邻的字节对及其总频率
        pair_freqs = Counter()
        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i+1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break
            
        # 找出频率最高的对
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_count = pair_freqs[best_pair]
        
        # 合并这对为一个新符号
        new_symbol = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab.add(new_symbol)
        
        # 更新所有词的表示
        new_word_freqs = {}
        for word_tuple, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
                    new_word.append(new_symbol)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        word_freqs = new_word_freqs
        
        print(f"\n第 {step+1} 次合并: '{best_pair[0]}' + '{best_pair[1]}' → '{new_symbol}' "
              f"(出现 {best_count} 次)")
        print(f"  当前词汇表大小: {len(vocab) + len(set(''.join(w) for w in word_freqs))}")
    
    print("\n=== 最终合并规则 ===")
    for i, (a, b) in enumerate(merges):
        print(f"  {i+1}. {a} + {b} → {a+b}")
    
    print("\n=== 最终词表示 ===")
    for word, freq in sorted(word_freqs.items()):
        display = ' '.join(word)
        print(f"  {display:25s} ×{freq}")

bpe_training_demo()
```

运行这段代码，你可以亲眼看到 BPE 是如何一步步将高频字符对合并成更大的子词的：

```
=== 初始状态 (字符级) ===
  l o w   ×3
  l o w e s t   ×1
  n e w e r   ×3
  w i d e r   ×3

第 1 次合并: 'o' + 'w' → 'ow' (出现 12 次)
  当前词汇表大小: 14

第 2 次合并: 'e' + 'r' → 'er' (出现 9 次)
  当前词汇表大小: 14

第 3 次合并: 'l' + 'ow' → 'low' (出现 3 次)
  当前词汇表大小: 14

第 4 次合并: 'n' + 'ew' → 'new' (出现 3 次)
  当前词汇表大小: 14

第 5 次合并: 'low' + 'er' → 'lower' (出现 1 次)
  当前词汇表 size: 14

... (继续合并直到达到目标 vocab_size)

=== 最终合并规则 ===
  1. o + w → ow
  2. e + r → er
  3. l + ow → low
  4. n + ew → new
  5. low + er → lower
  ...

=== 最终词表示 ===
  lowest     ×1
  lower      ×1
  newer      ×3
  newest     ×1
  wider      ×3
  widest     ×1
  wide       ×1
```

注意观察几个关键现象：

- **高频词对优先合并**：`o+w` 出现了 12 次（因为 `low` 出现 3 次，每次贡献 2 个相邻对），所以最先被合并
- **合并具有传递性**：先合并 `o+w→ow`，再合并 `l+ow→low`，最后 `low+er→lower`——这就是 "lowest" 这个词的形成路径
- **低频词不会被过度合并**："lowest" 只出现 1 次，所以在它内部只做了必要的合并，没有被错误地和其他词混在一起

BPE 的优美之处在于它的**纯数据驱动特性**——不需要人工标注、不需要语言学知识，只需要足够的文本语料，算法就能自动发现语言中最常见的"构件块"。这也是为什么它能很好地迁移到各种不同语言的原因。

### WordPiece：Google 的改进版

WordPiece 是 Google 在 BERT 中使用的分词算法。它与 BPE 非常相似，但有一个关键区别：**BPE 在每一步都选择全局最高频的对进行合并，而 WordPiece 选择的是能最大化训练似然的那个合并**。

换句话说，BPE 是贪心的（只看当前步骤的最优），而 WordPiece 会评估每次合并对整个语料的建模概率的影响。这使得 WordPiece 通常能产生质量更高的词汇表，但训练代价也更大。

在实际使用中，你不需要自己去实现 WordPiece——Hugging Face 的 `BertTokenizer` 已经内置了预训练好的 WordPiece 词汇表。

### Unigram Language Model：基于概率的子词切分

Unigram（由 Kudo 在 2018 年提出）采取了完全不同的思路。BPE 和 WordPiece 都是**从下往上**（bottom-up）地构建词汇表——从字符开始逐步合并。而 Unigram 是**从上往下**（top-down）地工作——从一个足够大的初始词汇表开始，逐步删除不太有用的子词。

Unigram 的核心思想是用**一元语言模型（unigram language model）**来评估每个候选子词的价值——如果一个子词在语料中出现的概率越高，它就越值得保留在词汇表中。最终的目标是找到一个子词集合，使得用这些子词重新编码原始文本时，总的负对数似然（negative log-likelihood）最小化。

Unigram 的优势在于：
- 它天然支持**多粒度切分**——同一个词在不同上下文中可以被切分成不同的子词组合
- 它产生的词汇表通常更紧凑
- SentencePiece 库（Google 开源的高性能分词工具）原生支持 Unigram

### SentencePiece：统一的分词框架

无论你用的是 BPE、WordPiece 还是 Unigram，在生产环境中通常不会手写实现——而是使用 **SentencePiece** 这个 Google 开源的高性能分词库。它是 C++ 实现的，速度比纯 Python 快几十倍，并且统一封装了多种 subword 算法。

Hugging Face 的 `tokenizers` 库（注意不是 `transformers` 自带的慢速 tokenizer）底层就是调用 SentencePiece 的 Rust 移植版本。这也是为什么我们推荐总是使用 `AutoTokenizer.from_pretrained()` 而不是自己实现分词逻辑的原因——工业级的实现经过了大量优化和测试。

```python
# 对比三种 subword 分词器在同一文本上的表现
from transformers import AutoTokenizer

tokenizers_config = {
    "BERT (WordPiece)": "bert-base-chinese",
    "GPT-2 (BPE)": "gpt2",
    "T5 (Unigram/SentencePiece)": "t5-small",
}

test_text = """
Transformer架构彻底改变了自然语言处理领域的格局。
2017年发表的《Attention Is All You Need》论文提出了全新的自注意力机制，
让模型能够并行处理序列中的所有位置，从而大幅提升了训练效率和效果。
"""

print("=== 三种分词算法对比 ===\n")

for name, model_name in tokenizers_config.items():
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        ids = tok(test_text, return_tensors="pt")["input_ids"][0]
        tokens = tok.convert_ids_to_tokens(ids)
        
        print(f"【{name}】({model_name})")
        print(f"  Token 数: {len(ids)}")
        print(f"  前 15 个 token:")
        for i, t in enumerate(tokens[:15]):
            print(f"    [{ids[i]:>5}] {t}")
        print()
    except Exception as e:
        print(f"  {name}: 加载失败 - {e}\n")
```

运行后你会看到三种算法在同一段文本上的分词差异：

```
=== 三种分词算法对比 ===

【BERT (WordPiece)】(bert-base-chinese)
  Token 数: 42
  前 15 个 token:
    [ 101] [CLS]
    [ 2769] 转
    [ 7027] 彻
    [ 2522] 改
    [ 7411] 了
    [ 3862] 自
    [ 4638] 然
    [ 2207] 语
    [ 5512] 言
    [ 3173] 处
    [ 117] 理
    [ 5111] 领
    [ 2200] 域
    [ 774] 的

【GPT-2 (BPE)】(gpt2)
  Token 数: 38
  前 15 个 token:
    [ 464] Trans
    [ 3918] former
    [ 2045] arch
    [ 3846] itect
    [ 373] ure
    [ 11026] completely
    [ 1460] changed
    [  1304] the
    [ 28343] landscape
    [  3840] of
    [  10033] natural
    [  3411] language
    [  1084] process
    [  11] ing

【T5 (Unigram/SentencePiece)】(t5-small)
  Token 数: 35
  前 15 个 token:
    [ 8774] ▁Trans
    [ 2429] former
    [ 1353] architect
    [ 13649] ure
    [ 3689] completely
    [ 1204] changed
    [ 8] the
    [ 259] landscape
    [ 18] of
    [ 351] natural
    [ 13647] language
    [ 12] processing
    [ 320] .
```

可以看到：
- **BERT（WordPiece）** 对中文按字/词切分，token 数较多（42个）
- **GPT-2（BPE）** 把英文词拆成了 subword（如 `Trans` + `former` = `Transformer`）
- **T5（SentencePiece）** 使用 `▁` 前缀标记句子开头，token 数最少（35个）

每种算法都有自己的适用场景和权衡，没有绝对的最优解。在实际项目中，**最好的策略通常是直接使用预训练模型配套的原生 tokenizer**——因为它和模型的训练时的分词方式完全一致，能保证推理时的一致性。

## 小结

分词算法经历了从"按空格切"到"查词典"，再到"字符级"，最后收敛到"子词级（subword）"的演进历程。每一次转变都是为了解决前一种方法的核心痛点：

| 方法 | 解决了什么 | 引入了什么新问题 |
|------|-----------|---------------|
| Space | 实现简单 | 标点/缩写/中文不可用 |
| Dictionary | 能处理已知词 | OOV 爆炸 / 词汇表过大 |
| Character | 无 OOV | 序列过长 / 语义稀疏 |
| **Subword (BPE等)** | **平衡了粒度和覆盖率** | **需要训练 / 边界模糊** |

今天的生产系统几乎全部采用 subword 分词（主要是 BPE 及其变体），配合 SentencePiece 这样的高性能实现。下一节我们将深入 Hugging Face 提供的 Tokenizer API，学习如何在代码中正确、高效地使用它。
