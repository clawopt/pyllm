---
title: LangChain + Pandas 完整实战
description: RAG 问答 Agent / 模型选型助手 / 数据质量审查 Agent / 多 Agent 协作
---
# Agent 综合应用场景


## 实战一：RAG 问答 Agent

```python
import pandas as pd
import numpy as np
from typing import Optional
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field


class KnowledgeBaseTool(StructuredTool):
    """知识库查询工具"""

    name: str = "knowledge_base"
    description: str = ("查询内部知识库获取相关信息。输入应为自然语言问题，"
                       "如 '什么是注意力机制'、'LoRA 怎么工作'")
    kb_df: pd.DataFrame

    class Args(BaseModel):
        question: str

    args_schema: Type[BaseModel] = Args

    def _run(self, question: str) -> str:
        q_lower = question.lower()

        results = []
        for _, row in self.kb_df.iterrows():
            content = str(row.get('content', '')).lower()
            title = str(row.get('title', '')).lower()
            keywords = str(row.get('keywords', '')).lower()

            score = 0
            if any(kw in content for kw in q_lower.split()[:5]):
                score += 3
            if any(kw in title for kw in q_lower.split()[:3]):
                score += 2
            if any(kw in keywords.split(',') for kw in q_lower.split()[:3]):
                score += 1

            if score >= 2:
                results.append((score, row.get('title', ''), row.get('content', '')[:300]))

        results.sort(key=lambda x: -x[0])

        if not results:
            return f"❌ 知识库中未找到关于 '{question}' 的信息"

        best_score, best_title, best_content = results[0]
        output = f"📚 **{best_title}**\n\n{best_content}"

        if len(results) > 1:
            output += f"\n\n相关条目 ({len(results)} 条):"
            for s, t, _ in results[1:4]:
                output += f"\n  • [{s}分] {t}"

        return output


kb_data = pd.DataFrame({
    'title': ['注意力机制 (Attention)', 'Transformer 架构', 'LoRA 微调',
             'KV Cache', 'RLHF 对齐训练', 'RAG 检索增强'],
    'content': [
        '注意力机制是 Transformer 的核心组件，通过计算 Query、Key、Value '
        '三者的相关性来实现序列信息的动态加权聚合...',
        'Transformer 是一种基于自注意力机制的深度神经网络架构，由 Encoder-Decoder '
        '组成，广泛应用于 NLP 和 LLM 领域...',
        'LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法，通过在预训练权重上'
        '注入低秩分解矩阵来适配下游任务...',
        'KV Cache 是推理优化技术，缓存 Key 和 Value 向量以避免重复计算，'
        '可将显存占用降低约 40%...',
        'RLHF (Reinforcement Learning from Human Feedback) 通过人类反馈的奖励模型来'
        '对齐 LLM 的行为，使其输出更符合人类偏好...',
        'RAG (Retrieval-Augmented Generation) 在生成前先从外部知识库检索相关文档，'
        '然后基于检索结果生成更准确的回答...',
    ],
    'keywords': ['attention,q,k,v,self,transformer', 'lora,low-rank,micro,fine-tune',
               'kv,key,value,cache,inference,speed', 'rlhf,reward,human,feedback',
               'rag,retrieval,augmented,generation,search'],
})

kb_tool = KnowledgeBaseTool(kb_df=kb_data)

print("=== RAG 知识库查询 ===")
print(kb_tool._run("什么是注意力机制？"))
print("\n" + kb_tool._run("LoRA 怎么工作的？"))
```

## 实战二：LLM 选型顾问 Agent

```python
class ModelAdvisorTool(StructuredTool):
    """模型选型建议工具"""

    name: str = "model_advisor"
    description: str = ("根据需求推荐最合适的 LLM 模型。参数: use_case(使用场景), "
                       "budget(预算/百万token), latency_requirement(延迟要求ms), "
                       "context_length(上下文窗口)")
    model_db: pd.DataFrame

    class Args(BaseModel):
        use_case: str = "general"
        budget: float = 10.0
        latency_requirement: int = 1000
        context_length: int = 8000

    args_schema: Type[BaseModel] = Args

    def _run(self, use_case: str = "general", budget: float = 10.0,
              latency_requirement: int = 1000,
              context_length: int = 8000) -> str:
        df = self.model_db.copy()

        def calc_match(row):
            score = 50.0

            case_map = {
                'code': {'claude-sonnet': 20, 'gpt-4o': 15, 'deepseek-chat': 12},
                'chat': {'gpt-4o-mini': 15, 'claude-haiku': 18, 'deepseek-chat': 14},
                'math': {'gpt-4o': 18, 'deepseek-r1': 16},
                'reasoning': {'gpt-4o': 17, 'claude-sonnet': 19, 'gemini-pro': 15},
            }
            if use_case in case_map:
                score += case_map[use_case].get(row['model'], 0)

            total_price = row.get('price_per_1M', 50)
            if total_price <= budget * 0.8:
                score += 20
            elif total_price <= budget * 1.5:
                score += 10
            elif total_price > budget * 3:
                score -= 10

            avg_lat = row.get('avg_latency_ms', 500)
            if avg_lat <= latency_requirement * 0.7:
                score += 15
            elif avg_lat <= latency_requirement:
                score += 8

            ctx = row.get('context_window', 4000)
            if ctx >= context_length:
                score += 10
            elif ctx >= context_length * 0.7:
                score += 5

            return round(score, 1)

        df['match_score'] = df.apply(calc_match, axis=1)
        top_3 = df.nlargest(3, 'match_score')

        lines = [f"🎯 根据您的需求推荐:\n"]
        lines.append(f"   场景: {use_case}")
        lines.append(f"   预算: ${budget}/M tokens")
        lines.append(f"   延迟: <{latency_requirement}ms")
        lines.append(f"   上下文: {context_length} tokens\n")

        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            medal = ['🥇', '🥈', '🥉'][i-1]
            lines.append(
                f"{medal} **{row['model']}** (匹配度: {row['match_score']})\n"
                f"     价格: ${row.get('price_per_1M', '?')}/M | "
                f"延迟: ~{row.get('avg_latency_ms', '?')}ms | "
                f"窗口: {row.get('context_window', '?')}"
            )

        return "\n".join(lines)


models_db = pd.DataFrame({
    'model': ['GPT-4o', 'GPT-4o-mini', 'Claude-Sonnet', 'Claude-Haiku',
              'DeepSeek-V3', 'DeepSeek-R1', 'Gemini-Pro',
              'Llama-3.1-70B', 'Qwen2.5-72B'],
    'price_per_1M': [12.50, 0.15, 18.00, 0.80, 0.42, 0.14, 6.25, 1.47],
    'avg_latency_ms': [800, 250, 650, 180, 280, 450, 350, 400],
    'context_window': [128000, 128000, 200000, 200000,
                     65536, 65536, 1000000, 131072],
})

advisor = ModelAdvisorTool(model_db=models_db)

print("=== LLM 选型顾问 ===")
print(advisor._run(use_case="code", budget=5.0, latency_requirement=500))
print("\n" + advisor._run(use_case="reasoning", budget=20.0, context_length=50000))
```
