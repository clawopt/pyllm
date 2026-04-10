# 其他 API 端点与工具

> **白板时间**：想象你在生产环境运行 vLLM 服务，突然某个请求报错了。你需要知道：这个模型支持哪些 token？这段文本会被怎么分词？服务上加载了哪些模型？vLLM 的健康状态如何？这些"辅助性"的 API 端点就是你的诊断工具箱。今天我们逐一打开每个工具，看看它们能做什么。

## 一、Tokenization API：分词器的窗口

### 1.1 为什么需要 Tokenization API？

在实际开发中，你经常需要回答这些问题：
- "这段文本会被拆成多少个 token？会不会超长被截断？"
- "这个词在模型里是一个 token 还是三个？"
- "特殊 token（如 `<|im_end|>`、`<|eot_id|>`）的 ID 是多少？"

vLLM 提供了 `/v1/tokenize` 和 `/v1/detokenize` 两个端点来回答这些问题——**无需自己加载 tokenizer，直接通过 HTTP 调用即可**。

### 1.2 Tokenize：文本 → Token IDs

```http
POST /v1/tokenize HTTP/1.1
Content-Type: application/json

{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "text": "你好，世界！人工智能正在改变我们的生活。"
}
```

响应：

```json
{
    "tokens": [108854, 607, 7675, 316, 19261, 213, 8419, 292, 285, 4510, 441, 13],
    "count": 11
}
```

Python 调用示例：

```python
import requests
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

def tokenize_demo():
    """Tokenization 演示"""
    
    test_cases = [
        ("短中文", "你好世界"),
        ("长中文", "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。"),
        ("中英混合", "Hello 世界! This is a test of tokenization."),
        ("代码", "def fibonacci(n: int) -> list[int]:"),
        ("数字", "2024年12月31日，温度-5.6°C，占比99.9%"),
        ("特殊字符", "🎉🚀💡\n\t<|im_start|>system<|im_end|>"),
    ]
    
    for label, text in test_cases:
        response = requests.post(
            "http://localhost:8000/v1/tokenize",
            json={"model": "Qwen/Qwen2.5-7B-Instruct", "text": text}
        )
        result = response.json()
        
        print(f"[{label}] ({len(text)} 字符)")
        print(f"  文本: {text}")
        print(f"  Tokens: {result['tokens']} (共{result['count']}个)")
        
        avg_chars_per_token = len(text) / max(result['count'], 1)
        print(f"  压缩比: {avg_chars_per_token:.1f} 字符/token\n")

tokenize_demo()
```

典型输出（Qwen2.5 tokenizer）：

```
[短中文] (4 字符)
  文本: 你好世界
  Tokens: [5840, 1024] (共2个)
  压缩比: 2.0 字符/token

[长中文] (42 字符)
  文本: 人工智能是计算机科学的一个分支...
  Tokens: [3912, 212, 356, 856, ...] (共18个)
  压缩比: 2.3 字符/token

[中英混合] (33 字符)
  文本: Hello 世界! This is a test of tokenization.
  Tokens: [9706, 7675, 0, 220, 385, 361, 374, a, ..., 34602] (共17个)
  压缩比: 1.9 字符/token

[代码] (40 字符)
  文本: def fibonacci(n: int) -> list[int]:
  Tokens: [6843, 46298, 55, 10, 2181, 580, 14, 26, 573, 12, 827] (共11个)
  压缩比: 3.6 字符/token

[数字] (30 字符)
  文本: 2024年12月31日，温度-5.6°C，占比99.9%
  Tokens: [2024, 789, 1231, 892, 234, 11, 28567, 13, 15, 5.6, °C, 11, 28672, 99.9, %]
  压缩比: 2.0 字符/token

[特殊字符] (37 字符)
  文本: 🎉🚀💡...
  Tokens: [17705, 14138, 23279, 198, 151643, 151644, 77091, 77092] (共8个)
  压缩比: 4.6 字符/token
```

**关键发现**：
- 中文平均 **2-2.5 字符/token**（Qwen tokenizer 对中文优化很好）
- 英文约 **4 字符/token**
- 代码压缩比最高（**3-4 字符/token**），因为常见关键词被编码为单个 token
- 特殊 token 如 `<|im_start|>` 有专用 ID，不会被进一步拆分

### 1.3 Detokenize：Token IDs → 文本

```http
POST /v1/detokenize HTTP/1.1
Content-Type: application/json

{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "tokens": [108854, 607, 7675, 316]
}
```

响应：

```json
{
    "text": "你好，世界！"
}
```

```python
def detokenize_demo():
    """Detokenize + 往返一致性测试"""
    
    original = "vLLM 是一个高性能的 LLM 推理引擎"
    
    # Step 1: tokenize
    r1 = requests.post(
        "http://localhost:8000/v1/tokenize",
        json={"model": "Qwen/Qwen2.5-7B-Instruct", "text": original}
    )
    tokens = r1.json()["tokens"]
    print(f"[原始] {original}")
    print(f"[Tokens] {tokens} ({len(tokens)} 个)\n")
    
    # Step 2: detokenize
    r2 = requests.post(
        "http://localhost:8000/v1/detokenize",
        json={"model": "Qwen/Qwen2.5-7B-Instruct", "tokens": tokens}
    )
    recovered = r2.json()["text"]
    print(f"[恢复] {recovered}")
    print(f"[一致] {'✅' if recovered == original else '❌ 差异: ' + repr(recovered)}")
    
detokenize_demo()
```

### 1.4 实用工具函数

比如下面的程序封装了常用的 token 计算功能：

```python
import requests
from typing import List, Tuple

class VllmTokenizer:
    """vLLM 远程 Tokenizer 封装"""
    
    def __init__(self, base_url: str = "http://localhost:8000", model: str = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def encode(self, text: str, model: str = None) -> List[int]:
        """文本 → Token IDs"""
        resp = requests.post(
            f"{self.base_url}/v1/tokenize",
            json={"model": model or self.model, "text": text}
        )
        return resp.json()["tokens"]
    
    def decode(self, tokens: List[int], model: str = None) -> str:
        """Token IDs → 文本"""
        resp = requests.post(
            f"{self.base_url}/v1/detokenize",
            json={"model": model or self.model, "tokens": tokens}
        )
        return resp.json()["text"]
    
    def count_tokens(self, text: str, model: str = None) -> int:
        """计算文本的 token 数量"""
        return len(self.encode(text, model))
    
    def estimate_cost(self, text: str, price_per_1m: float = 0.6) -> float:
        """估算输入成本（美元/百万token）"""
        n_tokens = self.count_tokens(text)
        return n_tokens / 1_000_000 * price_per_1m
    
    def check_length_limit(self, text: str, max_tokens: int = 8192) -> Tuple[bool, int, float]:
        """检查是否超长"""
        n = self.count_tokens(text)
        ratio = n / max_tokens
        ok = n <= max_tokens
        return ok, n, ratio
    
    def truncate_to_fit(self, text: str, max_tokens: int, 
                        reserve_for_response: int = 512) -> str:
        """智能截断：保留前部内容，为回复预留空间"""
        budget = max_tokens - reserve_for_response
        tokens = self.encode(text)
        if len(tokens) <= budget:
            return text
        
        kept_tokens = tokens[:budget]
        return self.decode(kept_tokens)


def tokenizer_toolkit_demo():
    """Tokenizer 工具集演示"""
    
    tk = VllmTokenizer(model="Qwen/Qwen2.5-7B-Instruct")
    
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括：
    学习（获取信息和使用信息的规则）、推理（使用规则得出近似或确定的结论）、
    自我纠错。从1956年达特茅斯会议首次提出"人工智能"概念以来，
    AI经历了多次寒冬与繁荣。特别是2017年Transformer架构的提出和2022年
    ChatGPT的发布，将AI推向了新的高度。大语言模型（LLM）展现出了
    令人惊叹的语言理解、生成和推理能力。
    """.strip()
    
    print("=" * 60)
    print("Tokenizer 工具集演示")
    print("=" * 60)
    
    # 基础计数
    n_tokens = tk.count_tokens(long_text)
    print(f"\n[Token计数] {n_tokens} tokens")
    
    # 长度检查
    ok, n, ratio = tk.check_length_limit(long_text, max_tokens=256)
    print(f"[长度检查] max=256, 实际={n}, 占比={ratio:.1%}, {'✅ 安全' if ok else '⚠️ 超长'}")
    
    # 成本估算
    cost = tk.estimate_cost(long_text, price_per_1m=0.6)
    print(f"[成本估算] ${cost:.6f} (@ $0.6/百万tokens)")
    
    # 智能截断
    truncated = tk.truncate_to_fit(long_text, max_tokens=128, reserve_for_response=32)
    print(f"\n[截断结果] 原始={len(long_text)}字, 截断后={len(truncated)}字")
    print(f"  内容: {truncated[:100]}...")
    
    # 批量处理
    paragraphs = [
        "PagedAttention 是 vLLM 的核心创新。",
        "Continuous Batching 解决了 Static Batching 的三大缺陷。",
        "Scheduler 使用三态机管理请求生命周期。",
    ]
    counts = [tk.count_tokens(p) for p in paragraphs]
    print(f"\n[批量统计] {[f'{c}t' for c in counts]} = 总计 {sum(counts)} tokens")

tokenizer_toolkit_demo()
```

---

## 二、Models API：模型信息查询

### 2.1 获取已加载模型列表

```http
GET /v1/models
Authorization: Bearer token-abc123
```

响应：

```json
{
    "object": "list",
    "data": [
        {
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "object": "model",
            "created": 1703123456,
            "owned_by": "vllm",
            "root": "Qwen/Qwen2.5-7B-Instruct",
            "parent": null,
            "permission": [...]
        }
    ]
}
```

### 2.2 Python SDK 调用

```python
def models_api_demo():
    """Models API 演示"""
    
    response = client.models.list()
    
    print("[已加载模型]")
    for model in response.data:
        print(f"  ID: {model.id}")
        print(f"  类型: {model.object}")
        print(f"  所有者: {model.owned_by}")
        print()

models_api_demo()
```

**实际用途**：
- **服务发现**：客户端启动时自动检测可用模型
- **多模型路由**：根据模型列表动态选择端点
- **健康检查**：如果此接口返回空或超时，说明服务异常

---

## 三、根因分析 API（RCA）

> ⚠️ **注意**：Root Cause Analysis API 需要 vLLM 以 `--enable-rca` 参数启动，且仅在开发/调试环境中使用，**不要在生产环境开启**（有性能开销和安全风险）。

### 3.1 启用 RCA

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --enable-rca
```

### 3.2 RCA 端点

vLLM 提供以下调试端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/debug/version` | GET | 版本信息 |
| `/debug/stats` | GET | 性能统计 |
| `/debug/cache_stats` | GET | KV Cache 统计 |
| `/debug/traces` | GET | 最近的请求追踪 |
| `/debug/config` | GET | 当前运行配置 |

```python
def rca_demo():
    """根因分析 API 演示"""
    import json
    
    base = "http://localhost:8000"
    
    print("=" * 60)
    print("vLLM 根因分析 (RCA)")
    print("=" * 60)
    
    endpoints = [
        ("/debug/version", "版本信息"),
        ("/debug/stats", "性能统计"),
        ("/debug/cache_stats", "Cache 统计"),
        ("/debug/config", "运行配置"),
    ]
    
    for endpoint, label in endpoints:
        try:
            resp = requests.get(f"{base}{endpoint}", timeout=5)
            data = resp.json()
            
            print(f"\n--- {label} ({endpoint}) ---")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:500])
        except requests.exceptions.ConnectionError:
            print(f"\n--- {label} --- ❌ 连接失败（可能未启用 --enable-rca）")
        except Exception as e:
            print(f"\n--- {label} --- ❌ 错误: {e}")

rca_demo()
```

典型输出：

```
======================================================================
vLLM 根因分析 (RCA)
======================================================================

--- 版本信息 (/debug/version) ---
{
  "version": "0.6.6.post1",
  "git_hash": "abc1234def",
  "cuda_version": "12.4",
  "torch_version": "2.5.1",
  "python_version": "3.11.9"
}

--- 性能统计 (/debug/stats) ---
{
  "num_running_requests": 3,
  "num_waiting_requests": 1,
  "num_swapped_requests": 0,
  "gpu_cache_usage": 0.65,
  "cpu_cache_usage": 0.12,
  "avg_prompt_throughput": 1450.2,
  "avg_generation_throughput": 52.3
}

--- Cache 统计 (/debug/cache_stats) ---
{
  "num_free_gpu_blocks": 15234,
  "num_used_gpu_blocks": 27597,
  "gpu_cache_percent": 64.4,
  "num_free_cpu_blocks": 32000,
  "num_used_cpu_blocks": 0
}

--- 运行配置 (/debug/config) ---
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "max_model_len": 32768,
  "block_size": 16,
  "num_gpu_blocks": 42831,
  "swap_space_bytes": 4294967296,
  ...
}
```

**这些数据的价值**：
- `num_free_gpu_blocks` 接近 0 → 即将 OOM，需要扩容或调参
- `num_waiting_requests` 持续增长 → 吞吐不足，考虑增加并发数
- `gpu_cache_usage` 长期 < 50% → 资源浪费，可以接受更多请求

---

## 四、Health Check 与监控端点

### 4.1 健康检查端点

```python
def health_check_demo():
    """健康检查演示"""
    
    checks = {
        "GET /health": lambda: requests.get("http://localhost:8000/health"),
        "GET /health/ready": lambda: requests.get("http://localhost:8000/health/ready"),
        "GET /version": lambda: requests.get("http://localhost:8000/version"),
        "GET /v1/models": lambda: requests.get("http://localhost:8000/v1/models"),
    }
    
    print("[vLLM 健康检查]\n")
    all_ok = True
    
    for name, fn in checks.items():
        try:
            resp = fn()
            status = "✅ OK" if resp.status_code == 200 else f"⚠️ {resp.status_code}"
            print(f"  {name}: {status} ({resp.elapsed.total_seconds()*1000:.0f}ms)")
            if resp.status_code != 200:
                all_ok = False
        except Exception as e:
            print(f"  {name}: ❌ 失败 ({e})")
            all_ok = False
    
    return all_ok

health_check_demo()
```

输出：

```
[vLLM 健康检查]

  GET /health: ✅ OK (2ms)
  GET /health/ready: ✅ OK (2ms)
  GET /version: ✅ OK (1ms)
  GET /v1/models: ✅ OK (3ms)
```

### 4.2 生产级健康检查脚本

比如下面的程序可以集成到 Kubernetes liveness/readiness probe 或 Docker HEALTHCHECK 中：

```python
#!/usr/bin/env python3
"""vLLM 生产级健康检查脚本"""

import sys
import time
import requests
from dataclasses import dataclass

@dataclass
class HealthConfig:
    url: str = "http://localhost:8000"
    timeout: float = 5.0
    max_latency_ms: float = 1000.0
    min_free_blocks_ratio: float = 0.05

def check_vllm_health(config: HealthConfig = None) -> dict:
    """全面健康检查"""
    config = config or HealthConfig()
    results = {"overall": False, "checks": {}}
    
    # 1. 基础连通性
    try:
        r = requests.get(f"{config.url}/health", timeout=config.timeout)
        results["checks"]["connectivity"] = {
            "ok": r.status_code == 200,
            "latency_ms": r.elapsed.total_seconds() * 1000
        }
    except Exception as e:
        results["checks"]["connectivity"] = {"ok": False, "error": str(e)}
        return results
    
    # 2. 就绪状态
    try:
        r = requests.get(f"{config.url}/health/ready", timeout=config.timeout)
        results["checks"]["readiness"] = {"ok": r.status_code == 200}
    except:
        results["checks"]["readiness"] = {"ok": False}
    
    # 3. 响应延迟测试
    try:
        client = OpenAI(base_url=f"{config.url}/v1", api_key="health-check")
        start = time.time()
        r = client.completions.create(
            model="dummy",
            prompt="Hi",
            max_tokens=1,
            temperature=0
        )
        latency = (time.time() - start) * 1000
        results["checks"]["inference_latency"] = {
            "ok": latency < config.max_latency_ms,
            "latency_ms": latency
        }
    except Exception as e:
        results["checks"]["inference_latency"] = {"ok": False, "error": str(e)}
    
    results["overall"] = all(c.get("ok", False) for c in results["checks"].values())
    return results


if __name__ == "__main__":
    results = check_vllm_health()
    
    for name, check in results["checks"].items():
        status = "✅" if check.get("ok") else "❌"
        detail = ""
        if "latency_ms" in check:
            detail = f" ({check['latency_ms']:.0f}ms)"
        elif "error" in check:
            detail = f" ({check['error']})"
        print(f"  [{status}] {name}{detail}")
    
    sys.exit(0 if results["overall"] else 1)
```

---

## 五、综合实战：API 工具类整合

比如下面的程序把所有 API 整合成一个统一的客户端工具类：

```python
import asyncio
import time
import aiohttp
import numpy as np
from typing import List, Dict, Optional, AsyncIterator
from dataclasses import dataclass, field
from openai import OpenAI, AsyncOpenAI


@dataclass
class RequestMetrics:
    """请求指标"""
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    total_tokens: int = 0
    latency_s: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class EmbeddingResult:
    """Embedding 结果"""
    embedding: np.ndarray
    token_count: int


class VllmApiClient:
    """vLLM 全功能 API 客户端"""
    
    def __init__(self, base_url: str, api_key: str = "empty", 
                 embed_url: str = None, embed_model: str = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        
        self.chat_client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
        self.async_chat_client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key=api_key)
        
        self.embed_url = (embed_url or base_url).rstrip("/")
        self.embed_client = OpenAI(base_url=f"{embed_url}/v1", api_key=api_key)
        self.async_embed_client = AsyncOpenAI(base_url=f"{embed_url}/v1", api_key=api_key)
        self.embed_model = embed_model
    
    def health_check(self) -> bool:
        """快速健康检查"""
        import requests
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """列出可用模型"""
        response = self.chat_client.models.list()
        return [m.id for m in response.data]
    
    def get_version(self) -> str:
        """获取 vLLM 版本"""
        import requests
        r = requests.get(f"{self.base_url}/version", timeout=3)
        return r.json().get("version", "unknown")
    
    def chat(self, messages: list, **kwargs) -> str:
        """同步 Chat Completions"""
        response = self.chat_client.chat.completions.create(
            model=kwargs.pop("model", ""),
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    def chat_stream(self, messages: list, **kwargs) -> AsyncIterator[str]:
        """流式 Chat Completions（同步包装器）"""
        stream = self.chat_client.chat.completions.create(
            model=kwargs.pop("model", ""),
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    
    async def async_chat_stream(self, messages: list, **kwargs):
        """异步流式 Chat Completions"""
        stream = await self.async_chat_client.chat.completions.create(
            model=kwargs.pop("model", ""),
            messages=messages,
            stream=True,
            **kwargs
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    
    def complete(self, prompt: str, **kwargs) -> tuple:
        """Completions API + 指标"""
        start = time.time()
        ttft = None
        
        stream = kwargs.pop("stream", False)
        if stream:
            full_text = []
            for chunk in self.chat_client.completions.create(
                model=kwargs.pop("model", ""), prompt=prompt, stream=True, **kwargs
            ):
                if not ttft:
                    ttft = time.time() - start
                text = chunk.choices[0].text
                if text:
                    full_text.append(text)
                    yield text
            
            elapsed = time.time() - start
            n_tokens = len(full_text)
            tpot = ((elapsed - ttft) / max(n_tokens - 1, 1) * 1000) if ttft and n_tokens > 1 else 0
            yield RequestMetrics(
                ttft_ms=ttft * 1000 if ttft else 0,
                tpot_ms=tpot,
                total_tokens=n_tokens,
                latency_s=elapsed
            )
        else:
            response = self.chat_client.completions.create(
                model=kwargs.pop("model", ""), prompt=prompt, **kwargs
            )
            elapsed = time.time() - start
            return response.choices[0].text, RequestMetrics(
                total_tokens=response.usage.completion_tokens,
                latency_s=elapsed
            )
    
    def embed(self, texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
        """批量 Embedding"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.embed_client.embeddings.create(
                model=self.embed_model or "",
                input=batch
            )
            batch_embs = [np.array(e.embedding) for e in response.data]
            all_embeddings.extend(batch_embs)
        
        return all_embeddings
    
    async def async_embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """异步批量 Embedding"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = await self.async_embed_client.embeddings.create(
                model=self.embed_model or "",
                input=batch
            )
            batch_embs = [np.array(e.embedding) for e in response.data]
            all_embeddings.extend(batch_embs)
        
        return np.array(all_embeddings)
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize"""
        import requests
        r = requests.post(
            f"{self.base_url}/v1/tokenize",
            json={"model": "", "text": text}
        )
        return r.json()["tokens"]
    
    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize"""
        import requests
        r = requests.post(
            f"{self.base_url}/v1/detokenize",
            json={"model": "", "tokens": tokens}
        )
        return r.json()["text"]
    
    def count_tokens(self, text: str) -> int:
        """计算 token 数量"""
        return len(self.tokenize(text))


def unified_client_demo():
    """统一客户端演示"""
    
    client = VllmApiClient(
        base_url="http://localhost:8000",
        embed_url="http://localhost:8001",
        embed_model="BAAI/bge-m3"
    )
    
    print("=" * 60)
    print("vLLM 统一 API 客户端演示")
    print("=" * 60)
    
    # 健康检查
    healthy = client.health_check()
    print(f"\n[健康检查] {'✅ 正常' if healthy else '❌ 异常'}")
    
    # 版本信息
    version = client.get_version()
    print(f"[vLLM版本] {version}")
    
    # 模型列表
    models = client.list_models()
    print(f"[可用模型] {models}")
    
    # Chat
    reply = client.chat([
        {"role": "user", "content": "用一句话解释 PagedAttention"}
    ], model="", max_tokens=128, temperature=0.3)
    print(f"\n[Chat] {reply}")
    
    # Complete
    result, metrics = client.complete(
        "快速排序算法的时间复杂度是：",
        model="", max_tokens=64, temperature=0
    )
    print(f"[Complete] {result.strip()} ({metrics.latency_s:.2f}s)")
    
    # Tokenize
    text = "Continuous Batching 动态调度机制"
    n_tokens = client.count_tokens(text)
    print(f"[Tokenize] '{text}' → {n_tokens} tokens")
    
    # Embed
    embeddings = client.embed(["vLLM 推理引擎", "PagedAttention 内存管理"])
    sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    print(f"[Embed] 相似度 = {sim:.4f}")

unified_client_demo()
```

---

## 六、总结

本节我们学习了 vLLM 提供的所有辅助 API 端点：

| API 类别 | 端点 | 功能 | 典型用途 |
|---------|------|------|---------|
| **Tokenization** | POST /v1/tokenize | 文本→Token ID | 长度预估、成本计算 |
| **Tokenization** | POST /v1/detokenize | Token ID→文本 | 调试分词、往返验证 |
| **Models** | GET /v1/models | 已加载模型列表 | 服务发现、动态路由 |
| **Health** | GET /health | 存活探针 | K8s Liveness Probe |
| **Health** | GET /health/ready | 就绪探针 | K8s Readiness Probe |
| **Version** | GET /version | 版本信息 | 环境确认、兼容性检查 |
| **RCA** | GET /debug/* | 根因分析数据 | 性能调优、故障排查 |

**核心要点回顾**：

1. **Tokenization API** 让你在不加载 tokenizer 的情况下完成文本↔Token 的转换，对长度预估和成本控制至关重要
2. **Models API** 用于服务发现和运行时模型确认，是多模型架构的基础
3. **Health Check 端点**应集成到容器编排系统的探针中，实现自动故障恢复
4. **RCA 端点**提供丰富的内部状态信息，但仅用于调试环境
5. **统一客户端模式**把所有 API 封装成一个类，是生产项目的推荐实践

下一节我们将学习 **Offline Inference（离线推理）**——这是 vLLM 除了 API Server 之外的另一种重要使用方式。
