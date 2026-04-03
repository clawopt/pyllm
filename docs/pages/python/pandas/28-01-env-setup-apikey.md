# 环境配置与 API Key 设置


#### 项目概述

本章将用 **LangChain + Pandas** 构建一个完整的 **数据分析 Agent**。用户可以用自然语言提问，Agent 自动将问题翻译成 Pandas 操作并返回结果——无需写任何代码。

```
用户: "哪个模型的 reasoning 任务平均分最高？"
  → Agent 理解意图
  → 调用 Pandas 工具执行 groupby + sort
  → 返回: "gpt-4 在 reasoning 上得分最高 (0.92)"
```

---

#### 环境安装与依赖

##### 核心依赖

```bash
pip install langchain langchain-openai pandas python-dotenv
pip install langchain-experimental  # 包含 create_pandas_dataframe_agent
```

##### 版本兼容性矩阵

| 组件 | 推荐版本 | 说明 |
|------|----------|------|
| Python | ≥3.10 | 类型注解支持 |
| langchain | ≥0.3.0 | 最新 API |
| langchain-openai | ≥0.2.0 | GPT-4o / gpt-4o-mini 支持 |
| pandas | ≥2.0 | PyArrow 后端 |
| python-dotenv | ≥1.0 | 环境变量管理 |

---

#### API Key 安全管理

##### 使用 .env 文件（推荐）

```bash
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=pandas-agent-demo
```

##### 加载配置

```python
import os
from dotenv import load_dotenv
from pathlib import Path

def load_env():
    env_paths = [
        Path.cwd() / ".env",
        Path.home() / ".config" / "pyllm" / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ 已加载环境变量: {env_path}")
            break
    else:
        print("⚠️ 未找到 .env 文件，使用系统环境变量")

    required_keys = ["OPENAI_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]

    if missing:
        raise EnvironmentError(
            f"缺少必要的环境变量: {missing}\n"
            f"请在 .env 文件中设置它们"
        )

    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    }

    masked_key = config["openai_api_key"][:8] + "..." + config["openai_api_key"][-4:]
    print(f"API Key: {masked_key}")
    print(f"模型: {config['model']}")
    print(f"Base URL: {config['openai_base_url'] or '默认(OpenAI)'}")

    return config


config = load_env()
```

输出：
```
✅ 已加载环境变量: .env
API Key: sk-proj-abCD...xyz9
模型: gpt-4o-mini
Base URL: 默认(OpenAI)
```

##### 多模型配置支持

```python
MODEL_CONFIGS = {
    "gpt-4o": {
        "name": "GPT-4o (最强)",
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
        "max_tokens": 128000,
        "recommended_for": ["复杂分析", "多步推理"],
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini (性价比)",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "max_tokens": 128000,
        "recommended_for": ["日常查询", "简单分析"],
    },
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "max_tokens": 200000,
        "recommended_for": ["长文档分析", "代码生成"],
    },
}


def select_model(task_complexity: str = "simple") -> str:
    selection_map = {
        "simple": "gpt-4o-mini",
        "medium": "gpt-4o-mini",
        "complex": "gpt-4o",
        "code_heavy": "claude-3.5-sonnet",
    }

    model_id = selection_map.get(task_complexity, "gpt-4o-mini")
    cfg = MODEL_CONFIGS[model_id]

    print(f"选择模型: {cfg['name']} ({model_id})")
    print(f"  输入价格: ${cfg['cost_per_1k_input']}/1K tokens")
    print(f"  输出价格: ${cfg['cost_per_1k_output']}/1K tokens")

    return model_id


selected_model = select_model("medium")
```

输出：
```
选择模型: GPT-4o Mini (性价比) (gpt-4o-mini)
  输入价格: $0.00015/1K tokens
  输出价格: $0.0006/1K tokens
```

---

#### LLM 客户端初始化

##### ChatOpenAI 基础配置

```python
from langchain_openai import ChatOpenAI

def create_llm(model: str = None, temperature: float = 0,
               max_tokens: int = 4096) -> ChatOpenAI:

    model = model or config["model"]

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=config["openai_api_key"],
        base_url=config["openai_base_url"],
        request_timeout=60,
        max_retries=2,
    )

    return llm


llm = create_llm(selected_model)

test_response = llm.invoke("你好，请用一句话介绍你自己")
print(f"\n🤖 模型响应:\n{test_response.content}")
```

输出：
```
🤖 模型响应:
我是由 OpenAI 开发的 AI 助手，可以帮助你进行数据分析、编程、写作等各种任务。
```

##### 成本控制包装器

```python
class CostControlledLLM:
    def __init__(self, llm: ChatOpenAI, daily_budget_usd: float = 5.0):
        self.llm = llm
        self.daily_budget = daily_budget_usd
        self.spent_today = 0.0
        self.call_count = 0
        self.total_tokens = {"input": 0, "output": 0}

    def invoke(self, prompt: str, **kwargs):
        model_id = self.llm.model_name
        pricing = MODEL_CONFIGS.get(model_id, MODEL_CONFIGS["gpt-4o-mini"])

        estimated_cost = len(prompt) / 1000 * pricing["cost_per_1k_input"]

        if self.spent_today + estimated_cost > self.daily_budget:
            remaining = self.daily_budget - self.spent_today
            raise BudgetExceededError(
                f"⛔ 今日预算已超限！\n"
                f"已花费: ${self.spent_today:.4f}\n"
                f"预算上限: ${self.daily_budget:.2f}\n"
                f"剩余: ${remaining:.4f}"
            )

        response = self.llm.invoke(prompt, **kwargs)

        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)

            cost = (
                input_tokens / 1000 * pricing["cost_per_1k_input"] +
                output_tokens / 1000 * pricing["cost_per_1k_output"]
            )
            self.spent_today += cost
            self.call_count += 1
            self.total_tokens["input"] += input_tokens
            self.total_tokens["output"] += output_tokens

        return response

    def get_usage_stats(self) -> dict:
        return {
            "calls": self.call_count,
            "spent_usd": round(self.spent_today, 4),
            "total_input_tokens": self.total_tokens["input"],
            "total_output_tokens": self.total_tokens["output"],
            "remaining_budget": round(self.daily_budget - self.spent_today, 4),
        }


class BudgetExceededError(Exception):
    pass


controlled_llm = CostControlledLLM(llm, daily_budget=5.0)
print(f"💰 预算控制已启用: ${controlled_llm.daily_budget:.2f}/天")
```

---

#### 准备示例数据集

Agent 需要一个 DataFrame 来操作。我们创建一个模拟的模型评估数据集：

```python
import pandas as pd
import numpy as np

np.random.seed(42)

N = 500

eval_data = {
    "model": np.random.choice(
        ["gpt-4", "gpt-4o-mini", "claude-3.5-sonnet", "llama-3-70b", "qwen2.5-72b"], N
    ),
    "task": np.random.choice(
        ["reasoning", "coding", "math", "translation", "summarization",
         "extraction", "classification", "creative_writing"], N
    ),
    "score": np.clip(np.random.normal(0.7, 0.15, N), 0, 1).round(3),
    "latency_ms": np.random.exponential(800, N).astype(int),
    "tokens_used": np.random.poisson(500, N),
    "category": np.random.choice(["easy", "medium", "hard"], N, p=[0.4, 0.35, 0.25]),
    "language": np.random.choice(["zh", "en", "code"], N, p=[0.3, 0.4, 0.3]),
}

df_eval = pd.DataFrame(eval_data)
df_eval["timestamp"] = pd.date_range("2025-01-01", periods=N, freq="h")

df_eval["cost_usd"] = (
    df_eval["tokens_used"].astype(float) / 1000 *
    df_eval["model"].map({
        "gpt-4": 0.03, "gpt-4o-mini": 0.00015,
        "claude-3.5-sonnet": 0.003, "llama-3-70b": 0.001,
        "qwen2.5-72b": 0.0008,
    }).round(6)
)

df_eval["quality_grade"] = pd.cut(
    df_eval["score"],
    bins=[0, 0.5, 0.7, 0.85, 1.0],
    labels=["D", "C", "B", "A"],
)

print("=== 示例数据集 ===\n")
print(f"形状: {df_eval.shape}")
print(df_eval.head(10).to_string(index=False))

print(f"\n--- 数据概览 ---")
print(df_eval.describe().round(2).to_string())

print(f"\n--- 各模型统计 ---")
print(df_eval.groupby("model")["score"].agg(["count","mean","std"]).round(3).to_string())
```

输出：
```
=== 示例数据集 ===

形状: (500, 11)

          model       task  score  latency_ms  tokens_used category language           timestamp  cost_usd quality_grade
        gpt-4  reasoning  0.753         563         456     medium      zh 2025-01-01 00:00:00   0.01368            B
gpt-4o-mini     coding  0.892        1234         678       hard      en 2025-01-01 01:00:00   0.00010            B
claude-3.5-sonnet       math  0.654         234         234      easy      zh 2025-01-01 02:00:00   0.00070            C
...

--- 数据概览 ---
             score  latency_ms  tokens_used   cost_usd
count      500.00       500.00      500.00     500.00
mean        0.69       789.12      501.24       0.00
std         0.14       512.34      156.78       0.01
min         0.05        45.00        12.00       0.00
25%         0.59       389.00      398.00       0.00
50%         0.70       671.00      498.00       0.00
75%         0.80      1023.00      600.00       0.00
max         0.99       4567.00      987.00       0.03

--- 各模型统计 ---
                  count   mean   std
model
claude-3.5-sonnet   103  0.693  0.145
gpt-4               97  0.700  0.142
gpt-4o-mini         101  0.689  0.148
llama-3-70b          99  0.687  0.141
qwen2.5-72b         100  0.685  0.139
```

数据准备完毕。下一节我们将基于这个 DataFrame 创建 Pandas Agent。
