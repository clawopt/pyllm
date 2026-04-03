---
title: 大模型驱动的高阶数据清洗
description: 传统正则+条件判断 vs LLM 自动清洗的对比、调用 LLM API 批量处理异常数据、工程实践与成本控制
---
# 当规则不够用：LLM 驱动的智能清洗

前面几节我们用 Pandas 原生的方法（`str.replace()`、`dropna()`、`fillna()` 等）处理了绝大多数常见的数据质量问题。但有些场景下纯规则的清洗力不从心——比如判断一段对话"是否有信息量"、检测"评分和内容是否矛盾"、识别"语义重复但文字不同"的近似重复。这些需要语义理解的任务，正是大模型擅长的领域。

## 什么时候该上 LLM

不是所有清洗任务都需要调用 LLM API。一个实用的决策原则是：**能用确定性规则解决的就不要用概率性模型**。具体来说：

- **用 Pandas/正则**：HTML 标签去除、空白过滤、长度截断、格式修正（速度极快、成本为零）
- **用 LLM**：语义质量判断、逻辑矛盾检测、风格一致性检查、复杂纠错（速度较慢、有 API 成本）

```python
import pandas as pd

df = pd.DataFrame({
    'prompt': ['什么是AI' * 100, '解释Python'] * 50,
    'response': [f'回答{i}' for i in range(150)],
})

rule_clean = df[
    df['prompt'].notna() &
    (df['prompt'].str.len() > 3) &
    (df['response'].notna()) &
    (df['response'].str.len() > 10)
].copy()

print(f"规则清洗: {len(df)} → {len(rule_clean)} 条 (零成本)")
```

这 150 条数据用规则就能干净地筛完。只有当规则搞不定的模糊场景才值得引入 LLM。

## 批量调用的基本模式

当你确实需要 LLM 来帮忙清洗时，核心思路是**批量发送而非逐条调用**——API 的网络延迟是固定的，批量处理能显著降低平均每条数据的延迟：

```python
import pandas as pd
import time
from openai import OpenAI

client = OpenAI()

dirty = pd.DataFrame({
    'text': [
        '这段话没有实际信息量',
        'GPT-4o is a large language model',
        '',
        'ERROR: timeout',
        '正常的有价值文本',
    ],
    'quality': [None, None, None, None, 4],
})

def llm_judge_quality(texts):
    prompt = """判断以下文本是否包含实质性的有用信息。
回复格式（仅输出 JSON）:
{"verdict": "keep" 或 "discard", "reason": "一句话原因"}

文本:
""" + "\n---\n".join(texts)
    
    start = time.time()
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,
    )
    
    latency = time.time() - start
    cost = response.usage.prompt_tokens * 0.15 + response.usage.completion_tokens * 0.6
    
    return response.choices[0].message.content, latency, cost / 1_000_000

result, lat, usd = llm_judge_quality(dirty['text'].tolist())
print(f"结果: {result}")
print(f"耗时: {lat:.1f}s | 成本: ${usd:.4f}")
```

几个工程实践要点：第一，用 `gpt-4o-mini` 而非 `gpt-4o` 做清洗任务——便宜 10 倍以上且对这种分类/判断任务完全够用；第二，把多条数据拼成一个 prompt 一次性发送而不是逐条循环调用——减少网络往返次数；第三，设置 `temperature=0.1` 让输出更确定、更容易解析。

## 成本控制策略

LLM 清洗的成本主要来自 token 消耗。控制成本的实用技巧：

1. **先规则后 LLM**：用 Pandas 预筛选掉明显脏的数据，只把"不确定的边界情况"发给 LLM
2. **批量大小适中**：每批 50-100 条，太大容易超出上下文限制或导致质量下降
3. **缓存结果**：相同内容的文本不需要重复调用，用哈希去重后再发
4. **设置预算上限**：在代码里加一个总 token 消耗计数器，超预算时自动回退到保守策略（全部保留）

到这里，第六章和第七章就全部结束了。我们建立了一套完整的数据质量管理流程：

```
拿到数据 → 快速检查(06-01) → 缺失值检测(06-02) → 重复值检测(06-03) 
→ 质量报告(06-04) → 缺失值处理(07-01) → 重复值清理(07-02) 
→ 类型优化(07-03) → 字符串清洗(07-04) → LLM 智能清洗(07-05)
```

这套流程覆盖了从"发现问题"到"解决问题"的全链路。接下来的章节将进入更具操作性的数据处理技术：筛选、变换、排序、聚合等核心操作。
