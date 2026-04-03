---
title: 特征工程实战：为 LLM 训练构造数据
description: 文本特征提取 / 数值分箱 / 交互特征 / SFT 数据集特征增强
---
# 特征工程实战


## 文本特征提取

### 基础统计特征

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'text': [
        'Transformer 是一种基于自注意力机制的深度神经网络架构...',
        'BERT 通过掩码语言模型进行预训练，能够理解上下文语义',
        'GPT 系列采用自回归方式生成文本，从左到右逐 token 预测',
        'LoRA 是一种参数高效微调方法，通过低秩分解减少训练参数量',
        'RAG 结合了检索和生成两个阶段，先检索相关文档再生成答案',
    ],
})

df['char_len'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_len'] = (df['char_len'] / df['word_count']).round(1)

df['punct_count'] = df['text'].str.count(r'[，。！？、；：""''（）【】]')
df['punct_density'] = (df['punct_count'] / df['char_len']).round(4)

df['chinese_ratio'] = (df['text'].str.count(r'[\u4e00-\u9fff]') / df['char_len']).round(3)
df['english_ratio'] = (df['text'].str.count(r'[a-zA-Z]') / df['char_len']).round(3)

print(df[['char_len', 'word_count', 'avg_word_len', 'punct_density', 'chinese_ratio']])
```

### 关键词与实体特征

```python
import pandas as pd

df = pd.DataFrame({
    'text': [
        'Transformer 使用多头注意力机制处理序列数据',
        'BERT 在 NLU 任务上表现优异，适合文本分类',
        'RLHF 通过人类反馈强化学习来对齐模型行为',
        'KV Cache 可以显著降低推理时的内存占用',
        'FlashAttention 将注意力计算的复杂度从 O(n²) 降低',
    ],
})

keywords = ['Transformer', 'BERT', 'RLHF', '注意力', '推理', '微调']

for kw in keywords:
    df[f'has_{kw}'] = df['text'].str.contains(kw, case=False).astype(int)
    df[f'{kw}_count'] = df['text'].str.lower().str.count(kw.lower())

keyword_cols = [c for c in df.columns if c.startswith('has_')]
df['keyword_diversity'] = df[keyword_cols].sum(axis=1)

print(df[['text']].join(df[keyword_cols + ['keyword_diversity']]))
```

## 数值分箱（Binning）

### cut()：等宽/自定义分箱

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'prompt': [f'问题 {i}' for i in range(100)],
    'token_count': np.random.randint(50, 2000, 100),
    'response_length': np.random.randint(20, 5000, 100),
})

df['length_tier'] = pd.cut(
    df['token_count'],
    bins=[0, 128, 512, 1024, 4096, float('inf')],
    labels=['短(<128)', '中(128-512)', '长(512-1024)', '超长(1024-4K)', '极长(>4K)']
)

df['response_quality'] = pd.cut(
    df['response_length'],
    bins=5,
    labels=['很差', '较差', '一般', '良好', '优秀']
)

print("=== Token 分布 ===")
print(df['length_tier'].value_counts().sort_index())

print("\n=== 响应长度分布 ===")
print(df['response_quality'].value_counts().sort_index())
```

### qcut()：等频分箱

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model_score': np.random.beta(2, 5, 1000) * 100,
})

df['score_quartile'] = pd.qcut(
    df['model_score'],
    q=4,
    labels=['Q1(差)', 'Q2(中下)', 'Q3(中上)', 'Q4(优)']
)

df['score_decile'] = pd.qcut(
    df['model_score'],
    q=10,
    labels=[f'D{i}' for i in range(1, 11)]
)

print("=== 四分位数分布（每份约250条）===")
print(df['score_quartile'].value_counts().sort_index())
```

### 自定义分箱函数

```python
import pandas as pd
import numpy as np

def smart_bin_token_count(tokens):
    """针对 SFT 数据的智能分箱"""
    if tokens < 32:
        return 'too_short'
    elif tokens < 256:
        return 'short'
    elif tokens < 1024:
        return 'medium'
    elif tokens < 4096:
        return 'long'
    elif tokens < 16000:
        return 'very_long'
    else:
        return 'ultra_long'

def compute_complexity_score(row):
    """综合复杂度评分"""
    score = 0
    if row['token_count'] > 1024:
        score += 2
    if row['has_code']:
        score += 1
    if row['has_math']:
        score += 1
    if row['has_multilingual']:
        score += 1
    return min(score, 5)


np.random.seed(42)
n = 500
df = pd.DataFrame({
    'prompt': [f'prompt_{i}' for i in range(n)],
    'token_count': np.random.exponential(400, n).astype(int) + 32,
    'has_code': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'has_math': np.random.choice([0, 1], n, p=[0.8, 0.2]),
    'has_multilingual': np.random.choice([0, 1], n, p=[0.9, 0.1]),
})

df['length_category'] = df['token_count'].apply(smart_bin_token_count)
df['complexity'] = df.apply(compute_complexity_score, axis=1)

print("=== 长度分类分布 ===")
print(df['length_category'].value_counts())

print("\n=== 复杂度分布 ===")
print(df['complexity'].value_counts().sort_index())
```

## 交互特征

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'prompt_tokens': np.random.randint(50, 2000, 200),
    'response_tokens': np.random.randint(100, 8000, 200),
    'turn_count': np.random.randint(1, 15, 200),
})

df['total_tokens'] = df['prompt_tokens'] + df['response_tokens']
df['compression_ratio'] = (df['response_tokens'] / df['prompt_tokens']).round(2)
df['tokens_per_turn'] = (df['total_tokens'] / df['turn_count']).round(1)
df['is_long_conversation'] = (df['turn_count'] >= 5).astype(int)
df['efficiency_score'] = (
    df['response_tokens'] / (df['prompt_tokens'] * df['turn_count'])
).round(3)

print(df[['prompt_tokens', 'response_tokens', 'turn_count',
          'total_tokens', 'compression_ratio', 'efficiency_score']].head(8))
```

## SFT 数据集特征增强完整流程

```python
import pandas as pd
import numpy as np
import re

class SFTFeatureEngineer:
    """SFT 数据集特征工程流水线"""

    def __init__(self, df):
        self.df = df.copy()

    def extract_text_features(self):
        text_col = 'prompt' if 'prompt' in self.df.columns else 'instruction'

        self.df['char_len'] = self.df[text_col].str.len()
        self.df['token_est'] = (self.df['char_len'] * 1.5).astype(int)
        self.df['word_count'] = self.df[text_col].str.split().str.len()

        self.df['question_marks'] = self.df[text_col].str.count(r'[？?]')
        self.df['exclamation_marks'] = self.df[text_col].str.count(r'[！!]')
        self.df['code_block_count'] = self.df[text_col].str.count(r'```')

        has_keywords = ['如何', '什么', '为什么', '怎么', '解释', '比较', '区别']
        for kw in has_keywords:
            self.df[f'has_{kw}'] = self.df[text_col].str.contains(kw).astype(int)

        question_cols = [c for c in self.df.columns if c.startswith('has_') and 'how' not in c]
        self.df['question_type_count'] = self.df[question_cols].sum(axis=1)

        return self

    def add_length_features(self):
        if 'response' in self.df.columns:
            self.df['response_char_len'] = self.df['response'].str.len()
            self.df['response_token_est'] = (self.df['response_char_len'] * 1.5).astype(int)
            self.df['output_input_ratio'] = (
                self.df['response_char_len'] / self.df['char_len'].replace(0, 1)
            ).round(2)

        self.df['length_tier'] = pd.cut(
            self.df['token_est'],
            bins=[0, 64, 256, 1024, 4096, float('inf')],
            labels=['tiny', 'short', 'medium', 'long', 'very_long']
        )
        return self

    def add_quality_flags(self):
        if 'response' in self.df.columns:
            self.df['response_too_short'] = (self.df.get('response_char_len', 0) < 20).astype(int)
            self.df['response_too_long'] = (self.df.get('response_char_len', 0) > 8000).astype(int)

        self.df['prompt_too_short'] = (self.df['char_len'] < 5).astype(int)
        self.df['has_repetition'] = self.df.apply(
            lambda r: len(set(str(r.get('prompt', '')).split())) < 3
            if len(str(r.get('prompt', '')).split()) > 5 else False,
            axis=1
        ).astype(int)

        return self

    def compute_risk_score(self):
        risk = 0
        risk += self.df.get('prompt_too_short', 0) * 3
        risk += self.df.get('response_too_short', 0) * 2
        risk += self.df.get('response_too_long', 0) * 1
        risk += self.df.get('has_repetition', 0) * 2
        self.df['risk_score'] = risk

        self.df['risk_level'] = pd.cut(
            self.df['risk_score'],
            bins=[-1, 0, 2, 5, 100],
            labels=['low', 'medium', 'high', 'critical']
        )
        return self

    def build(self):
        return (self
                .extract_text_features()
                .add_length_features()
                .add_quality_flags()
                .compute_risk_score()
                .df)


np.random.seed(42)
n = 300
raw_data = []
prompts = ['什么是注意力机制', '如何安装 PyTorch', '比较 BERT 和 GPT 的区别',
           '解释 RLHF 的原理', '什么是 KV Cache', 'LoRA 怎么工作']
responses = ['注意力机制是...', 'PyTorch 安装步骤如下...', 'BERT 是双向的而 GPT 是单向的...',
             'RLHF 通过奖励模型...', 'KV Cache 缓存键值对...', 'LoRA 注入低秩矩阵...']

for _ in range(n):
    raw_data.append({
        'prompt': np.random.choice(prompts) + ('？' if np.random.rand() > 0.3 else ''),
        'response': np.random.choice(responses) + ' 补充说明' * np.random.randint(0, 4),
    })

sft_df = pd.DataFrame(raw_data)

engineer = SFTFeatureEngineer(sft_df)
enriched = engineer.build()

print(f"原始: {len(sft_df.columns)} 列 → 增强: {len(enriched.columns)} 列")
print(f"\n新增特征列:")
new_cols = [c for c in enriched.columns if c not in ['prompt', 'response']]
for col in new_cols[:12]:
    print(f"  {col}: {enriched[col].dtype}")

print(f"\n风险等级分布:")
print(enriched['risk_level'].value_counts().sort_index())

print(f"\n高风险样本预览:")
high_risk = enriched[enriched['risk_level'].isin(['high', 'critical'])]
print(high_risk[['prompt', 'char_len', 'risk_score', 'risk_level']].head(6))
```
