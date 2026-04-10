# 04-3 PARAMETER 调优艺术

## 为什么需要调参

你可能会想：模型训练好了不就能直接用吗，为什么还需要调参数？答案是——**同一个模型在不同参数设置下的表现差异可能比换一个模型还要大**。想象一下：一个 7B 参数量的模型在 temperature=0 时像一个刻板的搜索引擎，只会给出最"安全"的回答；而同样的模型在 temperature=1.5 时变成了一个天马行空的创意作家。参数调优本质上是在控制模型的**"性格"**。

Ollama 的 PARAMETER 指令让你可以把这些参数永久绑定到自定义模型上，这样每次运行时都不需要手动指定。这一节我们将系统地学习每个参数的含义、推荐值和调优方法。

## Temperature（温度）：最核心的参数

Temperature 是所有参数中影响最大、使用频率最高的一个。它控制的是模型输出概率分布的**"锐利程度"**：

```
┌─────────────────────────────────────────────────────────────┐
│              Temperature 对输出的影响                         │
│                                                             │
│  t = 0.0    ████████████████████                            │
│             ↑                                              │
│        概率全部集中在最高概率的 token 上                      │
│        → 输出完全确定、可复现                                │
│        → 适合：事实查询、代码生成、格式化输出                 │
│                                                             │
│  t = 0.7    ████░░████░░███░░████                           │
│             ↑         ↑         ↑                          │
│        概率适度分散到多个候选 token                           │
│        → 输出有变化但保持质量                                 │
│        → 适合：通用对话、翻译、写作（默认推荐值）              │
│                                                             │
│  t = 1.5    ██░░░█░░░██░░░█░░░██                             │
│           ↑     ↑     ↑     ↑                              │
│        概率高度分散、低概率 token 也有机会被选中               │
│        → 输出富有创造性但可能出现不连贯                       │
│        → 适合：头脑风暴、诗歌创作、角色扮演                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 不同任务的 Temperature 推荐值

| 任务类型 | 推荐 Temperature | 理由 |
|---------|-----------------|------|
| SQL 生成 | 0.0 - 0.1 | SQL 语法必须精确，不允许任何偏差 |
| 代码补全 | 0.1 - 0.2 | 代码逻辑正确性优先于多样性 |
| 数据提取/JSON | 0.0 - 0.1 | 格式必须严格匹配 schema |
| 技术文档翻译 | 0.2 - 0.4 | 术语一致性比文采更重要 |
| 通用问答 | 0.5 - 0.7 | 平衡准确性和自然度 |
| 创意写作 | 0.8 - 1.2 | 需要多样性和意外性 |
| 角色扮演 | 0.9 - 1.3 | 角色个性需要通过多样化表达体现 |
| 头脑风暴 | 1.2 - 1.5 | 数量优先于质量，鼓励非常规想法 |

### 实际对比实验

```python
#!/usr/bin/env python3
"""Temperature 对比实验"""

import requests
import json

def test_temperature(model, prompt, temperatures):
    """同一 prompt 在不同 temperature 下的输出"""
    
    print(f"\n{'='*70}")
    print(f"  模型: {model}")
    print(f"  Prompt: {prompt[:50]}...")
    print(f"{'='*70}\n")
    
    for temp in temperatures:
        resp = requests.post("http://localhost:11434/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temp, "seed": 42}
        }, timeout=60)
        
        data = resp.json()
        content = data["message"]["content"]
        
        print(f"🌡️  Temperature = {temp}")
        print(f"{'─'*66}")
        print(content[:400])
        if len(content) > 400:
            print("...")
        print()

if __name__ == "__main__":
    test_temperature(
        "qwen2.5:7b",
        "请用一句话描述量子计算对密码学的影响。",
        [0.0, 0.3, 0.7, 1.0, 1.5]
    )
```

你会清楚地看到随着 temperature 升高，输出从千篇一律变得越来越有个性，但同时也越来越容易偏离主题或出现语法错误。

## Top-K 和 Top-P：采样策略的精细控制

Temperature 控制整体分布形状，而 Top-K 和 Top-P 则是**更精细的采样过滤器**。它们经常与 temperature 配合使用：

```dockerfile
# 常见组合模式
PARAMETER temperature 0.7          # 整体随机性
PARAMETER top_k 40                 # 只从概率最高的 40 个 token 中选择
PARAMETER top_p 0.9                # 或累积概率达到 90% 时截断
```

### 工作原理图解

```
假设模型对下一个 token 的预测概率分布如下：

Token:   "的"(35%)  "是"(20%)  "一"(15%)  "了"(10%)  "很"(8%)
         "不"(5%)    "会"(3%)   "在"(2%)   ... (共 50K 个候选)
         
Top-K = 4:
→ 只保留前 4 个: ["的","是","一","了"] (累计 80%)
→ 其余 49,996 个 token 概率设为 0

Top-P = 0.9:
→ 从高到低累加概率直到 ≥ 90%
→ 可能保留: ["的","是","一","了","很","不"] (累计 93%)
→ 其余设为 0

实际效果：
top_k 小 + top_p 小 → 更确定、更保守
top_k 大(或0) + top_p 大(或1.0) → 更自由、更多样
```

### 推荐配置矩阵

| 场景 | temperature | top_k | top_p | 效果 |
|------|------------|--------|-------|------|
| 代码/SQL | 0.1 | 20 | 0.9 | 高度确定性 |
| 翻译 | 0.3 | 40 | 0.95 | 准确且流畅 |
| 通用对话 | 0.7 | 40 | 0.9 | **默认推荐** |
| 写作 | 0.9 | 50 | 0.95 | 自然流畅 |
| 创意任务 | 1.2 | 60 | 0.98 | 极具多样性 |

## Num_ctx（上下文长度）：双刃剑参数

`num_ctx` 决定了模型一次能"记住"多少个 token。更大的上下文意味着你可以输入更长的文档、进行更多轮的对话，但代价也是显著的：

```python
#!/usr/bin/env python3
"""上下文长度 vs 内存占用 vs 性能的关系演示"""

def context_size_impact():
    """展示不同上下文长度的资源消耗"""
    
    # KV Cache 大约每个 token 需要:
    # - 7B 模型: ~0.5 MB/token (fp16)
    # - 13B 模型: ~1.0 MB/token
    # - 70B 模型: ~5.0 MB/token
    
    model_sizes = {
        "7B": {"bytes_per_token": 0.5 * 1024 * 1024},
        "14B": {"bytes_per_token": 1.0 * 1024 * 1024},
        "32B": {"bytes_per_token": 2.0 * 1024 * 1024},
        "70B": {"bytes_per_token": 5.0 * 1024 * 1024},
    }
    
    ctx_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000]
    
    print(f"\n{'模型':>6s} │ {'上下文':>8s} │ {'KV Cache':>12s} │ "
          f"{'额外内存':>10s} │ {'延迟影响':>8s}")
    print(f"{'─'*6}─┼─{'─'*8}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*8}")
    
    for model_name, info in model_sizes.items():
        bpt = info["bytes_per_token"]
        for ctx in ctx_sizes:
            kv_cache_mb = (ctx * bpt) / (1024 * 1024)
            
            # 延迟随上下文增长近似 O(n) 到 O(n^2) 取决于注意力实现
            if ctx <= 8192:
                latency_factor = f"x{1 + ctx/8192:.1f}"
            elif ctx <= 32768:
                latency_factor = f"x{2 + (ctx-8192)/16384:.1f}"
            else:
                latency_factor = f"x{4 + (ctx-32768)/32768:.1f}"
            
            print(f"{model_name:>6s} │ {ctx:>8d} │ {kv_cache_mb:>10.1f}MB │ "
                  f"+{kv_cache_mb*0.2:>7.0f}% │  {latency_factor}")

context_size_impact()
```

输出结果会让你对上下文长度的代价有一个直观认识：

```
  模型 │   上下文  │   KV Cache  │   额外内存  │  延迟影响
  ──────┼──────────┼─────────────┼────────────┼──────────
    7B  │     2048 │       1.0MB │      +0.2% │  x1.2
    7B  │     8192 │       4.0MB │      +0.8% │  x2.0
    7B  │    32768 │      16.0MB │      +3.2% │  x3.0
    7B  │   128000 │      64.0MB │     +12.8% │  x6.0
   70B  │     8192 │      40.0MB │      +8.0% │  x2.0
   70B  │    32768 │     160.0MB │     +32.0% │  x3.0
   70B  │   128000 │     640.0MB │    +128.0% │  x6.0
```

**关键结论**：对于 70B 模型，将 num_ctx 设为 128K 需要额外的 **640MB 内存**仅用于 KV Cache，而且推理速度会下降约 6 倍。所以除非你真的需要处理超长文档，否则不要盲目追求大上下文。

### 推荐策略

```bash
# 短对话 / 快速问答
PARAMETER num_ctx 2048

# 标准对话 / 文档分析
PARAMETER num_ctx 8192

# 长文档处理 / RAG 场景
PARAMETER num_ctx 16384

# 超长文本（仅在确实需要时）
PARAMETER num_ctx 32768
```

## Stop 序列：让模型知道何时闭嘴

Stop 参数可能是最容易被忽视但最有实用价值的参数之一。它告诉模型：**当你的输出中出现了以下字符串时，立即停止生成**。

```dockerfile
# 常见的 Stop 设置

# 防止模型继续"自言自语"
PARAMETER stop "</s>"

# 防止思维链内容泄漏到最终输出
PARAMETER stop "</think>"

# 多轮对话中防止模拟用户消息
PARAMETER stop "User:"
PARAMETER stop "Human:"
PARAMETER stop "Question:"

# JSON 输出时防止额外内容
PARAMETER stop "```"
```

### 一个 Stop 参数救了我的项目的真实案例

我在做一个日志分析工具时遇到一个问题：模型总是会在正式回答之前输出一段"思考过程"，然后才给出答案。用户看到的输出是这样的：

```
让我来分析这个问题...

首先，我注意到错误发生在 14:32:05 左右...
其次，堆栈跟踪显示 NullPointerException...
最后，我认为根本原因是...

## 分析报告
[这里才是我想要的正式输出]
```

那段"让我来分析..."的开头完全是多余的。解决方案很简单：

```dockerfile
SYSTEM """你是日志分析专家。直接给出分析结果，
不要输出任何思考过程或开场白。"""
PARAMETER stop "让我"
PARAMETER stop "我来"
PARAMETER stop "好的"
PARAMETER stop "当然"
```

虽然这个方案有点粗暴（直接阻止常见开场白），但在实际项目中效果非常好——输出变得干净利落，可以直接被下游程序解析。

## Repeat_penalty（重复惩罚）：对抗模型的复读机倾向

大语言模型有一种天然的倾向：**在生成长文本时会重复相同的短语或段落**。这是因为自回归生成的特性——模型倾向于重复已经生成过的模式以维持"局部一致性"。Repeat_penalty 就是用来对抗这种倾向的：

```dockerfile
# 默认值通常就够用了
PARAMETER repeat_penalty 1.15

# 如果模型还是频繁重复，可以加大
PARAMETER repeat_penalty 1.3

# 对于需要大量重复内容的场景（如列表生成），适当降低
PARAMETER repeat_penalty 1.05
```

### 工作原理

```
正常情况下 token "苹果" 的概率 = 0.08

repeat_penalty = 1.15 时：
如果 "苹果" 最近已出现过：
  新概率 = 0.08 / 1.15 = 0.07 (降低)

repeat_penalty = 1.3 时：
如果 "苹果" 最近已出现过：
  新概率 = 0.08 / 1.3 = 0.062 (更大程度降低)

repeat_penalty = 0.8 时 (< 1.0):
如果 "苹果" 最近已出现过：
  新概率 = 0.08 / 0.8 = 0.10 (反而增加！鼓励重复)
```

**注意**：repeat_penalty < 1.0 会**鼓励**重复，这在极少数特定场景下有用（比如生成包含规律性重复元素的内容）。

## Seed（随机种子）：可复现性的钥匙

在开发和测试阶段，你需要模型的输出是**可复现的**——同样的输入应该产生同样的输出，这样才能做回归测试和效果对比。Seed 参数就是为此设计的：

```dockerfile
# 固定种子 → 同样的输入永远产生同样的输出
PARAMETER seed 42

# 不设种子 → 每次运行都不同（默认行为）
# （不需要写这行，省略即可）
```

```python
#!/usr/bin/env python3
"""验证 seed 参数的可复现性"""

import requests

def test_reproducibility(model, prompt, seed=42):
    """测试相同 seed 是否产生相同输出"""
    
    results = []
    for i in range(3):
        resp = requests.post("http://localhost:11434/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.7, "seed": seed},
            "stream": False
        }, timeout=30)
        content = resp.json()["message"]["content"]
        results.append(content)
        print(f"Run {i+1}: {content[:80]}...")
    
    if results[0] == results[1] == results[2]:
        print("\n✅ 三次运行结果完全一致 — seed 生效！")
    else:
        print("\n⚠️ 结果不一致 — 可能的原因:")
        print("   - 模型不完全支持确定性生成")
        print("   - GPU 浮点运算的非确定性")

test_reproducibility("qwen2.5:7b", "什么是机器学习？用一句话回答。")
```

## 高级采样参数

除了上面介绍的常用参数外，Ollama 还支持几个高级采样参数。它们在日常使用中出现频率较低，但在特定场景下非常有用：

### Mirostat：自动调节温度

Mirostat 是一种自适应算法，它能**动态调整有效温度**以维持目标熵值（perplexity）。这意味着你不需要手动调 temperature——Mirostat 会自动平衡创造性和连贯性：

```dockerfile
# Mirostat 版本 2（推荐）
PARAMETER mirostat 2
PARAMETER mirostat_tau 5.0       # 目标 perplexity（越低越保守）
PARAMETER mirostat_eta 0.1       # 学习速率（通常不改）

# Mirostat 版本 1（旧版）
PARAMETER mirostat 1
PARAMETER mirostat_tau 5.0
PARAMETER mirostat_eta 0.1
```

**适用场景**：当你不确定该用什么 temperature 时，Mirostat 是一个很好的"自动挡"替代品。特别是对于长文本生成任务，Mirostat 能避免"开头很好但后面越来越跑偏"的问题。

### TFS（Tail Free Sampling）

TFS 通过**截断概率分布的尾部**来过滤掉那些"不太合理"的 token：

```dockerfile
# TFS = 1.0 表示不启用（默认）
PARAMETER tfs 1.0

# TFS < 1.0 启用尾部截断，值越小越激进
# 推荐范围: 0.9 - 1.0
PARAMETER tfs 0.95
```

### Typical P

Typical P 基于**token 的"典型性"**（即它在上下文中出现的意外程度）进行采样：

```dockerfile
# Typical P = 1.0 不启用（默认）
PARAMETER typical_p 1.0

# 启用时推荐值: 0.9 - 1.0
PARAMETER typical_p 0.95
```

## 参数调优工作流

下面是一套系统的参数调优方法论：

```python
#!/usr/bin/env python3
"""Ollama 模型参数优化工作流"""

import requests
import json
import itertools
from datetime import datetime

class ParameterOptimizer:
    """网格搜索式参数优化器"""
    
    def __init__(self, model, base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.results = []
    
    def evaluate(self, prompt, expected_keywords, options):
        """评估一组参数的效果"""
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": options
                },
                timeout=120
            )
            
            data = resp.json()
            content = data["message"]["content"].lower()
            
            # 计算关键词命中率
            hits = sum(1 for kw in expected_keywords if kw.lower() in content)
            score = hits / len(expected_keywords) if expected_keywords else 0
            
            return {
                "options": options,
                "score": score,
                "output_length": len(data["message"]["content"]),
                "tokens_per_second": data.get("eval_count", 0) / 
                    (data.get("total_duration", 1) / 1e9),
                "content_preview": data["message"]["content"][:200]
            }
        except Exception as e:
            return {"options": options, "score": -1, "error": str(e)}
    
    def grid_search(self, test_cases, param_grid):
        """网格搜索最优参数"""
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        total = len(combinations) * len(test_cases)
        done = 0
        
        print(f"🔬 开始参数优化: {len(combinations)} 种参数组合 x "
              f"{len(test_cases)} 个测试用例 = {total} 次评估\n")
        
        for combo in combinations:
            options = dict(zip(keys, combo))
            total_score = 0
            
            for case in test_cases:
                result = self.evaluate(
                    case["prompt"],
                    case["expected_keywords"],
                    options
                )
                total_score += result.get("score", 0)
                done += 1
            
            avg_score = total_score / len(test_cases)
            self.results.append({
                "options": options,
                "avg_score": round(avg_score, 3),
                "details": result
            })
            
            if done % 10 == 0:
                print(f"  进度: {done}/{total} ({done//total*100}%)")
        
        self.results.sort(key=lambda x: x["avg_score"], reverse=True)
        
        return self.results[:10]

# 使用示例
if __name__ == "__main__":
    optimizer = ParameterOptimizer("qwen2.5:7b")
    
    test_cases = [
        {
            "prompt": "请用 Python 写一个快速排序函数",
            "expected_keywords": ["def", "quicksort", "partition", "return", "list"]
        },
        {
            "prompt": "解释什么是微服务架构",
            "expected_keywords": ["服务", "独立", "通信", "部署", "扩展"]
        }
    ]
    
    param_grid = {
        "temperature": [0.2, 0.4, 0.6, 0.8],
        "top_k": [20, 40, 60],
        "top_p": [0.85, 0.9, 0.95],
        "repeat_penalty": [1.1, 1.15, 1.2]
    }
    
    best = optimizer.grid_search(test_cases, param_grid)
    
    print(f"\n{'='*70}")
    print(f"  🏆 TOP 10 最优参数组合")
    print(f"{'='*70}\n")
    
    for i, r in enumerate(best, 1):
        opts = r["options"]
        print(f"#{i:2d} 得分: {r['avg_score']:.3f}")
        print(f"    temp={opts['temperature']}  top_k={opts['top_k']}  "
              f"top_p={opts['top_p']}  rep_pen={opts['repeat_penalty']}")
        print(f"    输出预览: {r['details'].get('content_preview', '')[:100]}")
        print()
```

## 本章小结

这一节我们全面学习了 Ollama 的参数调优体系：

1. **Temperature 是最重要的参数**，控制输出的随机性和创造性，不同任务有截然不同的推荐值
2. **Top-K / Top-P** 作为 Temperature 的补充，提供更精细的采样控制
3. **Num_ctx 是双刃剑**——更大的上下文带来更强的能力但也显著增加内存和延迟
4. **Stop 序列**能有效防止模型输出多余内容和格式泄漏
5. **Repeat_penalty** 解决了模型的"复读机"问题
6. **Seed 参数**确保开发和测试阶段的可复现性
7. **Mirostat / TFS / Typical_P** 是高级采样选项，适合特殊需求
8. **系统化的调优工作流**可以通过网格搜索找到最优参数组合

下一节我们将学习如何通过 ADAPTER 指令加载 LoRA 权重，为模型注入领域能力。
