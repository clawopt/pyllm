---
title: Agent 记忆与状态管理
description: DataFrame 作为对话记忆 / 多轮上下文管理 / 会话摘要 / 长期记忆持久化
---
# Agent 记忆与状态管理


## Agent 记忆的三层架构

```
┌─────────────────────────────────────┐
│           LLM Context Window         │ ← 短期记忆 (8K-128K tokens)
│  (当前对话 + 最近 Tool 结果)        │
├─────────────────────────────────────┤
│           Pandas DataFrame          │ ← 中期记忆 (结构化数据)
│  (完整历史、用户画像、知识库)       │
├─────────────────────────────────────┤
│           持久化存储                  │ ← 长期记忆 (Parquet/DB)
│  (历史会话、学习偏好、趋势分析)      │
└─────────────────────────────────────┘
```

## 短期记忆：DataFrame 作为当前会话状态

```python
import pandas as pd
import numpy as np
from datetime import datetime

class SessionMemory:
    """Agent 会话记忆（基于 Pandas）"""

    def __init__(self):
        self.conversation_log = []
        self.tool_calls = pd.DataFrame(columns=[
            'timestamp', 'tool_name', 'query', 'result_summary',
            'latency_ms', 'tokens_used', 'success'
        ])
        self.user_profile = {}

    def log_interaction(self, tool_name, query, result,
                           latency_ms=0, tokens=0, success=True):
        """记录一次交互"""
        entry = {
            'timestamp': datetime.now(),
            'tool_name': tool_name,
            'query': str(query)[:200],
            'result_summary': str(result)[:300],
            'latency_ms': int(latency_ms),
            'tokens_used': int(tokens),
            'success': success,
        }
        self.conversation_log.append(entry)

        new_row = pd.DataFrame([entry])
        self.tool_calls = pd.concat(
            [self.tool_calls, new_row], ignore_index=True
        )
        return self

    def get_recent_context(self, n=5):
        """获取最近 N 次交互作为上下文"""
        recent = self.tool_calls.tail(n)
        if len(recent) == 0:
            return "无历史交互"

        context_parts = [f"最近操作 {i+1}: "
                        f"{row['tool_name']}({row['query']}) → "
                        f"{'✅' if row['success'] else '❌'}"
                        for i, (_, row) in enumerate(recent.iterrows())]
        return "\n".join(context_parts)

    def get_session_stats(self):
        """获取本次会话统计"""
        if len(self.tool_calls) == 0:
            return "暂无数据"

        total = len(self.tool_calls)
        success_rate = self.tool_calls['success'].mean() * 100
        avg_latency = self.tool_calls['latency_ms'].mean()
        total_tokens = self.tool_calls['tokens_used'].sum()

        tool_dist = self.tool_calls['tool_name'].value_counts()

        return {
            'total_interactions': total,
            'success_rate': round(success_rate, 1),
            'avg_latency_ms': round(avg_latency_ms, 1),
            'total_tokens': int(total_tokens),
            'tools_used': dict(tool_dist.to_dict()),
        }


memory = SessionMemory()
memory.log_interaction('filter', 'score > 85', '找到 23 条', 45, 120, True)
memory.log_interaction('aggregate', '按 model 分组', '5 个模型', 120, 80, True)
memory.log_interaction('top_n', 'Top 3 延迟', 'GPT/Claude/Llama', 35, 50, True)
memory.log_interaction('filter', 'price < 1', '找到 15 条', 30, 90, False)

print("=== 会话记忆 ===")
print(memory.get_recent_context(3))
print(f"\n统计:")
for k, v in memory.get_session_stats().items():
    print(f"  {k}: {v}")
```

## 中期记忆：用户画像 DataFrame

```python
class UserProfileManager:
    """基于 DataFrame 的用户画像管理"""

    def __init__(self):
        self.profile = pd.DataFrame({
            'user_id': [],
            'preferred_models': [],
            'favorite_tasks': [],
            'avg_budget_per_query': [],
            'query_frequency': [],
            'last_active': [],
            'skill_level': [],  # beginner / intermediate / advanced
            'language_preference': [],
        })

    def update_from_interaction(self, user_id, model_used, task_type,
                                   cost, skill_hint='unknown'):
        """根据交互更新用户画像"""

        if user_id not in self.profile['user_id'].values:
            new_row = {
                'user_id': user_id,
                'preferred_models': [model_used],
                'favorite_tasks': [task_type],
                'avg_budget_per_query': cost,
                'query_frequency': 1,
                'last_active': datetime.now(),
                'skill_level': skill_hint,
                'language_preference': 'zh',
            }
            new_df = pd.DataFrame([new_row])
            self.profile = pd.concat([self.profile, new_df], ignore_index=True)
        else:
            idx = self.profile[self.profile['user_id'] == user_id].index[0]

            models = self.profile.at[idx, 'preferred_models']
            if model_used not in models:
                models.append(model_used)
            self.profile.at[idx, 'preferred_models'] = models

            tasks = self.profile.at[idx, 'favorite_tasks']
            if task_type not in tasks:
                tasks.append(task_type)
            self.profile.at[idx, 'favorite_tasks'] = tasks

            old_budget = self.profile.at[idx, 'avg_budget_per_query']
            n = self.profile.at[idx, 'query_frequency']
            new_avg = (old_budget * n + cost) / (n + 1)
            self.profile.at[idx, 'avg_budget_per_query'] = round(new_avg, 4)
            self.profile.at[idx, 'query_frequency'] += 1
            self.profile.at[idx, 'last_active'] = datetime.now()

        return self.get_recommendation(user_id)

    def get_recommendation(self, user_id):
        """基于画像给出推荐"""
        if user_id not in self.profile['user_id'].values:
            return {"model": "gpt-4o-mini", "reason": "新用户，使用默认推荐"}

        row = self.profile[self.profile['user_id'] == user_id].iloc[0]

        budget = row['avg_budget_per_query']
        level = row['skill_level']

        if budget < 0.5:
            rec_model = "deepseek-chat"
            reason = f"预算敏感(${budget:.3f}/次)"
        elif level == 'beginner':
            rec_model = "gpt-4o-mini"
            reason = "新手友好，容错性好"
        elif 'code' in str(row['favorite_tasks']):
            rec_model = "claude-sonnet"
            reason = "代码能力强"
        else:
            rec_model = "gpt-4o"
            reason = "综合能力最强"

        return {"model": rec_model, "reason": reason}


mgr = UserProfileManager()
mgr.update_from_interaction('user_001', 'gpt-4o', 'code', 2.50, 'intermediate')
mgr.update_from_interaction('user_001', 'claude', 'analysis', 3.00, 'intermediate')
mgr.update_from_interaction('user_001', 'deepseek-chat', 'chat', 0.14, 'advanced')

rec = mgr.get_recommendation('user_001')
print(f"推荐模型: {rec['model']} ({rec['reason']})")

print("\n用户画像:")
print(mgr.profile[['user_id', 'preferred_models', 'avg_budget_per_query',
                       'query_frequency']].to_string(index=False))
```

## 长期记忆：持久化到 Parquet

```python
class PersistentMemory:
    """长期记忆：持久化到文件"""

    def __init__(self, storage_path='/tmp/agent_memory'):
        import os
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.history_file = f'{storage_path}/interaction_history.parquet'
        self.profiles_file = f'{storage_path}/user_profiles.parquet'

    def save_history(self, session_memory: SessionMemory):
        """保存会话历史到 Parquet"""
        if len(session_memory.tool_calls) > 0:
            session_memory.tool_calls.to_parquet(self.history_file, index=False)
            print(f"✓ 历史已保存 ({len(session_memory.tool_calls)} 条)")

    def load_history(self):
        """加载历史记录"""
        import os
        if os.path.exists(self.history_file):
            return pd.read_parquet(self.history_file)
        return pd.DataFrame()

    def save_profiles(self, profile_mgr: UserProfileManager):
        """保存用户画像"""
        if len(profile_mgr.profile) > 0:
            profile_mgr.profile.to_parquet(self.profiles_file, index=False)
            print(f"✓ 用户画像已保存 ({len(profile_mgr.profile)} 个用户)")

    def get_trend_analysis(self):
        """分析历史趋势"""
        history = self.load_history()
        if len(history) == 0:
            return "暂无历史数据"

        history['date'] = pd.to_datetime(history['timestamp']).dt.date

        daily = history.groupby('date').agg(
            interactions=('tool_name', 'count'),
            success_rate=('success', 'mean'),
            avg_cost=('tokens_used', 'mean'),
            unique_tools=('tool_name', 'nunique'),
        ).round(2)

        print("=== 历史趋势 ===")
        print(daily.tail(7))
        return daily


persist = PersistentMemory()
persist.save_history(memory)
trend = persist.get_trend_analysis()
```
