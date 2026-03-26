# 温度采样

温度采样（Temperature Sampling）是语言模型生成文本时的核心解码策略之一。它通过调整 softmax 的"温度"参数来控制输出分布的熵，从而影响生成文本的多样性和确定性。本篇文章详细介绍温度采样的原理，以及如何使用 `np.random.choice` 基于 logits 实现各种采样策略。

## 温度采样的原理

标准的 softmax 函数定义为：

```
P(token_i) = exp(logit_i) / Σ exp(logit_j)
```

温度采样在 softmax 之前对 logits 进行缩放：

```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)
```

其中 T 是温度参数：
- **T = 1.0**：标准 softmax
- **T → 0**：趋近于贪心解码（最高概率几乎为 1）
- **T → ∞**：趋近于均匀分布（完全随机）

```python
import numpy as np

def softmax(logits, axis=-1):
    """标准 softmax"""
    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

def temperature_softmax(logits, temperature, axis=-1):
    """温度 softmax

    参数:
        logits: 模型输出的原始得分
        temperature: 温度参数 T
    """
    if temperature == 0:
        raise ValueError("Temperature cannot be 0")

    scaled_logits = logits / temperature
    return softmax(scaled_logits, axis=axis)

# 示例 logits
logits = np.array([2.0, 1.0, 0.5, 0.1, 0.05])

print("=== 不同温度下的概率分布 ===\n")
for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
    probs = temperature_softmax(logits, temp)
    print(f"温度={temp}: {probs}")
    print(f"  最高概率: {probs.max():.4f}, 熵: {-np.sum(probs * np.log(probs + 1e-10):.4f})\n")
```

## np.random.choice 实现采样

NumPy 的 `np.random.choice` 是实现采样的核心函数：

```python
def sample_from_probs(probs):
    """根据概率分布采样

    参数:
        probs: 概率分布（和为1）
    返回:
        采样的索引
    """
    return np.random.choice(len(probs), p=probs)

# 示例
probs = np.array([0.4, 0.3, 0.2, 0.1])
samples = [sample_from_probs(probs) for _ in range(1000)]
print("采样分布（理论 vs 实际）:")
print(f"理论分布: {probs}")
print(f"实际分布: {np.bincount(samples, minlength=len(probs)) / 1000}")
```

## 完整采样器实现

```python
class TemperatureSampler:
    """温度采样器"""

    def __init__(self, temperature=1.0, top_k=None, top_p=None):
        self.temperature = temperature
        self.top_k = top_k  # Top-K 采样
        self.top_p = top_p  # Nucleus/Top-P 采样

    def sample(self, logits):
        """从 logits 采样下一个 token

        参数:
            logits: (vocab_size,) 模型输出的原始得分
        返回:
            采样的 token ID
        """
        # 应用温度
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Top-K 采样
        if self.top_k is not None:
            top_k_indices = np.argpartition(logits, -self.top_k)[-self.top_k:]
            masked_logits = np.full_like(logits, -1e9)
            masked_logits[top_k_indices] = logits[top_k_indices]
            logits = masked_logits

        # Top-P 采样
        if self.top_p is not None:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_probs = softmax(logits[sorted_indices])
            cumsum = np.cumsum(sorted_probs)

            # 找到累积概率超过 top_p 的位置
            cutoff_idx = np.searchsorted(cumsum, self.top_p) + 1
            top_p_indices = sorted_indices[:cutoff_idx]

            masked_logits = np.full_like(logits, -1e9)
            masked_logits[top_p_indices] = logits[top_p_indices]
            logits = masked_logits

        # 转为概率分布
        probs = softmax(logits)

        # 采样
        token_id = np.random.choice(len(probs), p=probs)

        return token_id

# 示例
np.random.seed(42)
sampler = TemperatureSampler(temperature=1.0, top_k=50)

logits = np.random.randn(10000)  # 模拟模型输出
next_token = sampler.sample(logits)
print(f"采样的 token ID: {next_token}")
```

## 不同采样策略对比

```python
def compare_sampling_strategies():
    """对比不同采样策略"""
    print("\n=== 采样策略对比 ===\n")

    # 模拟 logits（某些 token 得分很高）
    logits = np.array([5.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01])

    strategies = [
        ("贪心 (T=0)", lambda x: np.argmax(x)),
        ("温度 0.5", TemperatureSampler(temperature=0.5).sample),
        ("温度 1.0", TemperatureSampler(temperature=1.0).sample),
        ("温度 2.0", TemperatureSampler(temperature=2.0).sample),
        ("Top-K (K=3)", TemperatureSampler(top_k=3).sample),
    ]

    for name, strategy in strategies:
        probs = temperature_softmax(logits, temperature=1.0 if "T=" not in name else float(name.split("=")[1].rstrip(')')))
        if "贪心" in name:
            chosen = np.argmax(logits)
        else:
            chosen = strategy(logits)
        print(f"{name}: 选择 token {chosen}")

compare_sampling_strategies()
```

## 温度与熵的关系

温度直接影响输出分布的熵（Entropy），熵衡量分布的不确定性：

```python
def compute_entropy(probs):
    """计算分布的熵

    H = -Σ P(x) * log(P(x))
    """
    return -np.sum(probs * np.log(probs + 1e-10))

def temperature_vs_entropy():
    """展示温度与熵的关系"""
    print("\n=== 温度与熵的关系 ===\n")

    # 模拟 logits
    logits = np.array([3.0, 2.0, 1.0, 0.5, 0.2, 0.1])

    for temp in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        probs = temperature_softmax(logits, temp)
        entropy = compute_entropy(probs)
        max_prob = probs.max()

        print(f"温度={temp:4.1f}: 熵={entropy:5.2f}, 最大概率={max_prob:.4f}")

    print("\n结论:")
    print("  - 温度越低，熵越低，分布越集中（更确定性）")
    print("  - 温度越高，熵越高，分布越均匀（更多样性）")

temperature_vs_entropy()
```

## 实践建议

### 何时使用不同温度

```python
def get_recommended_temperature(task):
    """根据任务推荐温度

    参数:
        task: 任务类型
    返回:
        推荐温度
    """
    recommendations = {
        "代码生成": (0.2, 0.5),  # 低温度，更确定性
        "机器翻译": (0.3, 0.7),
        "文本摘要": (0.5, 0.8),
        "对话生成": (0.7, 1.0),  # 中等温度
        "创意写作": (0.8, 1.2),  # 稍高温度
        "头脑风暴": (1.0, 1.5),  # 高温度，更多样
    }

    return recommendations.get(task, (0.7, 1.0))

print("任务温度推荐:")
for task, (low, high) in get_recommended_temperature("代码生成").items():
    print(f"  {task}: {low} - {high}")
```

### 采样失败处理

```python
def safe_sample(logits, temperature=1.0, top_k=None, top_p=None):
    """安全的采样函数，处理各种边界情况

    参数:
        logits: 模型输出
        temperature: 温度
        top_k: Top-K
        top_p: Top-P
    返回:
        采样的 token ID
    """
    # 处理 NaN 或 Inf
    if not np.all(np.isfinite(logits)):
        print("警告: logits 包含 NaN 或 Inf，使用均匀分布")
        return np.random.randint(0, len(logits))

    # 数值稳定性：减去最大值
    logits = logits - np.max(logits)

    # 应用温度
    if temperature > 0:
        logits = logits / temperature

    # Top-K
    if top_k is not None and top_k > 0:
        top_k = min(top_k, len(logits))
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        masked_logits = np.full_like(logits, -1e9)
        masked_logits[top_k_indices] = logits[top_k_indices]
        logits = masked_logits

    # Top-P
    if top_p is not None and 0 < top_p < 1:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_probs = softmax(logits[sorted_indices])
        cumsum = np.cumsum(sorted_probs)

        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        top_p_indices = sorted_indices[:cutoff_idx]

        masked_logits = np.full_like(logits, -1e9)
        masked_logits[top_p_indices] = logits[top_p_indices]
        logits = masked_logits

    # 转为概率并采样
    probs = softmax(logits)

    # 处理概率和不为 1 的情况
    if not np.isclose(probs.sum(), 1.0):
        probs = probs / probs.sum()

    return np.random.choice(len(probs), p=probs)

# 测试
logits = np.random.randn(10000)
for temp in [0.1, 0.5, 1.0, 2.0]:
    token = safe_sample(logits, temperature=temp)
    print(f"温度={temp}: 采样 token {token}")
```

## 常见误区

**误区一：温度设为 0**

温度为 0 会导致除零错误。应该使用贪心解码（argmax）代替：

```python
# 错误
# probs = temperature_softmax(logits, temperature=0)

# 正确：贪心解码
greedy_token = np.argmax(logits)
```

**误区二：忽略数值稳定性**

当 logits 值很大时，exp 可能溢出：

```python
# 正确做法：先减去最大值
logits_shifted = logits - np.max(logits)  # 数值稳定
exp_logits = np.exp(logits_shifted / temperature)
```

**误区三：Top-K 和 Top-P 同时使用不当**

Top-K 和 Top-P 都是为了限制采样空间，但同时使用可能导致冲突：

```python
# 可能导致问题
sampler = TemperatureSampler(temperature=0.8, top_k=10, top_p=0.9)

# 应该选择一个策略，或者按顺序应用
```

温度采样是控制语言模型输出的关键技巧。掌握这些技术，可以让你根据不同任务需求灵活调整生成结果的多样性和质量。
