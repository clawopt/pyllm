---
title: 大模型驱动的高阶数据清洗
description: 传统正则+条件判断 vs LLM 自动清洗的对比、调用 LLM API 批量处理异常数据、工程实践与成本控制
---
# LLM 驱动的智能清洗


## 两种清洗范式的对比

```
传统方式:                    LLM 驱动方式:
┌──────────────┐            ┌──────────────────┐
│ 正则表达式    │            │                  │
│ 条件判断      │   ──→       │ LLM API         │
│ 规则引擎      │            │ (GPT-4o/Claude)  │
│ 确定性输出    │            │                  │
│              │            │ 概率性输出        │
└──────────────┘            └──────────────────┘

速度: 极快                    速度: 较慢（受 API 延迟限制）
精度: 高（规则明确时）        精度: 高（处理模糊/复杂情况）
成本: 几乎为零                成本: API 调用费用
适用: 格式修正、去噪、标准化   适用: 语义理解、分类、纠错
```

## 什么时候该用 LLM 清洗

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| HTML 标签去除 | **正则** | 规则确定，速度快 |
| 空白/None 过滤 | **Pandas** | 一行代码搞定 |
| 文本长度过滤 | **Pandas** | 数值比较，无需 AI |
| **格式错误识别** | **LLM** | "2025/13/32" vs "2025.01.13" 需要语义理解 |
| **逻辑矛盾检测** | **LLM** | "评分=5 但回复为空" 需要推理 |
| **内容质量判断** | **LLM** | "这段话是否有信息量" 是主观判断 |
| **语言/风格修正** | **LLM** | 需要生成能力 |
| **敏感信息脱敏** | **正则 + LLM** | 先规则后智能 |

## 实现方案一：批量调用 LLM API

### 基础框架

```python
import pandas as pd
import numpy as np
import time
from openai import OpenAI


class LLMCleaner:
    """基于 LLM 的数据清洗器"""
    
    def __init__(self, model='gpt-4o-mini', api_key=None, max_concurrent=5):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.max_concurrent = max_concurrent
        self.cost_log = []
    
    def _build_prompt(self, row, task_description):
        """构建清洗提示词"""
        prompt = f"""你是一个数据清洗助手。请根据以下规则处理给定的文本。

{task_description}

请直接返回清洗后的结果，不要添加任何解释。

原始数据:
{row.to_dict()}

清洗结果:"""
        return prompt
    
    def clean_single(self, prompt):
        """调用 LLM 清洗单条数据"""
        start = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        
        result = response.choices[0].message.content.strip()
        elapsed = time.time() - start
        
        usage = response.usage
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.6) / 1_000_000
        
        self.cost_log.append({
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'cost_usd': cost,
            'latency_s': elapsed,
        })
        
        return result
    
    def clean_batch(self, df, text_col, task_desc, 
                     batch_size=100, progress_interval=50):
        """批量清洗 DataFrame"""
        
        results = []
        total = len(df)
        
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i+batch_size]
            
            for idx, row in batch.iterrows():
                prompt = self._build_prompt(row, task_desc)
                cleaned = self.clean_single(prompt)
                
                results.append({
                    'original_idx': idx,
                    f'{text_col}_raw': row[text_col],
                    f'{text_col}_cleaned': cleaned,
                })
                
                if len(results) % progress_interval == 0:
                    print(f"已处理 {len(results):,}/{total:,} "
                          f"({len(results)/total*100:.1f}%)")
        
        result_df = pd.DataFrame(results)
        self._print_cost_summary()
        
        return result_df
    
    def _print_cost_summary(self):
        if not self.cost_log:
            return
        
        log_df = pd.DataFrame(self.cost_log)
        total_cost = log_df['cost_usd'].sum()
        total_tokens = log_df['prompt_tokens'].sum() + log_df['completion_tokens'].sum()
        avg_latency = log_df['latency_s'].mean()
        
        print(f"\n💰 LLM 调用成本汇总:")
        print(f"  总调用次数: {len(self.cost_log):,}")
        print(f"  总 Token 数: {total_tokens:,}")
        print(f"  总费用: ${total_cost:.4f}")
        print(f"  平均延迟: {avg_latency:.2f}s")
        print(f"  吞吐量: {3600/avg_latency:.1f} 条/小时")


np.random.seed(42)
dirty_data = pd.DataFrame({
    'id': range(100),
    'text': [
        '什么是注意力机制',
        'ERROR: connection timeout at 192.168.1.1',  # 错误标记
        '',                                              # 空
        '   ',                                            # 纯空白
        '<div>HTML标签</div>需要清理',                   # HTML 标签
        '2025/30/30 提交了反馈',                         # 无效日期
        '评分是10分但回复内容为空',                       # 逻辑矛盾
        '正常的问题文本' * 93,
    ]
})

cleaner = LLMCleaner(model='gpt-4o-mini')

TASK_DESC = """任务要求：
1. 如果文本包含错误标记（如 ERROR、HTTP 5xx），标记为 [NEEDS_REVIEW]
2. 如果文本为空或纯空白，标记为 [EMPTY]
3. 如果文本包含 HTML 标签，去除标签并保留文本内容
4. 如果日期格式明显错误，标记日期部分为 [INVALID_DATE]
5. 如果存在明显的逻辑矛盾，在末尾追加 [LOGIC_ERROR]
6. 正常文本保持原样"""

cleaned_result = cleaner.clean_batch(
    dirty_data, 
    text_col='text',
    task_desc=TASK_DESC,
    batch_size=20,
    progress_interval=25
)

print(cleaned_result.head(10))
```

## 方案二：混合策略（先规则后 LLM）

**推荐的生产级做法**：

```python
import pandas as pd
import numpy as np


class HybridCleaner:
    """混合清洗器：先用 Pandas 规则处理明确的脏数据，
    再用 LLM 处理模糊/复杂的边界情况"""
    
    def __init__(self, llm_cleaner=None):
        self.llm = llm_cleaner
        self.stats = {'rule': 0, 'llm': 0, 'total': 0}
    
    def clean(self, df, text_col):
        """两阶段清洗"""
        
        self.stats['total'] = len(df)
        
        stage1_start = time.time()
        
        mask_valid = (
            df[text_col].notna() &
            (df[text_col].str.len() > 3) &
            (~df[text_col].str.contains(r'^\s*$', regex=True)) &
            (~df[text_col].str.contains(r'ERROR|HTTP \d+', regex=True))
        )
        
        rule_cleaned = df[mask_valid].copy()
        rule_cleaned[f'{text_col}'] = (
            rule_cleaned[text_col]
            .str.replace(r'<[^>]+>', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        
        needs_llm = df[~mask_valid].copy()
        self.stats['rule'] = len(rule_cleaned)
        stage1_time = time.time() - stage1_start
        
        print(f"Stage 1 (规则): {len(df)} → {len(rule_cleaned)} 有效, "
              f"{len(needs_llm)} 待 LLM 处理 ({stage1_time:.2f}s)")
        
        if self.llm and len(needs_llm) > 0 and len(needs_llm) <= 1000:
            stage2_start = time.time()
            
            llm_result = self.llm.clean_batch(
                needs_llm,
                text_col=text_col,
                task_desc="修复以下问题：空值填[EMPTY]，错误标记保留但标注原因",
                batch_size=20,
                progress_interval=20
            )
            
            combined = pd.concat([
                rule_cleaned[[text_col]].rename(columns={text_col: f'{text_col}_cleaned'}),
                llm_result[[f'{text_col}_cleaned']].rename(columns={
                    f'{text_col}_cleaned': f'{text_col}_cleaned'
                })
            ], ignore_index=True)
            
            self.stats['llm'] = len(needs_llm)
            stage2_time = time.time() - stage2_start
            
            print(f"Stage 2 (LLM): {len(needs_llm)} 条 ({stage2_time:.2f}s)")
            
            final = combined
            
        else:
            if len(needs_llm) > 1000:
                print(f"⚠️  LLM 待处理数据过多 ({len(needs_llm)} 条)，跳过 LLM 阶段")
                print(f"   建议：先人工抽样检查或增加规则覆盖")
            
            rule_cleaned[f'{text_col}_cleaned'] = rule_cleaned[text_col]
            needs_llm[f'{text_col}_cleaned'] = '[SKIPPED] ' + needs_llm[text_col].fillna('')
            final = pd.concat([rule_cleaned, needs_llm], ignore_index=True)
            self.stats['llm'] = 0
        
        print(f"\n📊 清洗统计:")
        print(f"  规则处理: {self.stats['rule']:,} ({self.stats['rule']/self.stats['total']*100:.1f}%)")
        print(f"  LLM 处理: {self.stats['llm']:,} ({self.stats['llm']/self.stats['total']*100:.1f}%)")
        print(f"  合计:     {len(final):,}")
        
        return final.reset_index(drop=True)


hybrid = HybridCleaner(llm_cleaner=None)  # 暂不启用 LLM
result = hybrid.clean(dirty_data, 'text')
print(result[['id', 'text', 'text_cleaned']])
```

## 成本控制与优化策略

### 降低 LLM 清洗成本的技巧

```python
class CostOptimizedCleaner(LLMCleaner):
    """成本优化的 LLM 清洗器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}  # 结果缓存
    
    def clean_with_cache(self, df, text_col, cache_key_cols=['text']):
        """带缓存的清洗——相同文本不重复调用"""
        
        cache_hits = 0
        cache_misses = 0
        results = []
        
        for _, row in df.iterrows():
            key = tuple(row[c] for c in cache_key_cols if c in row.index)
            
            if key in self.cache:
                results.append(self.cache[key])
                cache_hits += 1
            else:
                prompt = self._build_prompt(row, "清洗此文本")
                cleaned = self.clean_single(prompt)
                self.cache[key] = cleaned
                results.append(cleaned)
                cache_misses += 1
        
        print(f"缓存命中率: {cache_hits}/{cache_hits+cache_misses} "
              f"({cache_hits/(cache_hits+cache_misses)*100:.1f}%)")
        
        df[f'{text_col}_cleaned'] = results
        return df
    
    def sample_then_clean(self, df, text_col, sample_rate=0.1,
                           seed=42):
        """先采样检查，再决定是否全量清洗"""
        
        sampled = df.sample(frac=sample_rate, random_state=seed)
        sample_result = self.clean_batch(
            sampled, text_col,
            task_desc="评估此数据的质量，标记 [GOOD] 或 [BAD]",
            batch_size=10
        )
        
        bad_rate = (sample_result[f'{text_col}_cleaned']
                     .str.contains('BAD').sum()) / len(sample_result)
        
        print(f"采样检查: 不良率 = {bad_rate*100:.1f}%")
        
        if bad_rate < 0.05:
            print("✅ 数据质量良好，跳过全量 LLM 清洗")
            return df
        else:
            print(f"⚠️ 不良率较高 ({bad_rate*100:.1f}%)，执行全量清洗")
            return self.clean_batch(df, text_col, task_desc="清洗所有数据")
```
