# 推理性能基准测试

> **白板时间**：你刚优化了 vLLM 的配置——调了 TP 大小、改了 batch size、启用了量化。老板问："优化后到底快了多少？"你不能说"感觉快了一点"。你需要**数据说话**：TTFT 从 2.3s 降到了 0.8s，TPOT 从 85ms 降到了 32ms，吞吐量从 45 req/s 提升到了 128 req/s。这就是基准测试的价值——它把主观感受变成可比较、可追踪的客观数据。

## 一、关键性能指标

### 1.1 指标体系

```
vLLM 性能指标金字塔：

         ┌──────────────────┐
         │   业务指标        │ ← 用户真正关心的
         │  QPS / 并发用户数 │
         ├──────────────────┤
         │   系统指标        │ ← 运维关注的
         │  吞吐 / 延迟分布  │
         ├──────────────────┤
         │   引擎指标        │ ← vLLM 内部的
         │ TTFT / TPOT / GPU利用率│
         └──────────────────┘
```

### 1.2 核心指标详解

| 指标 | 全称 | 定义 | 重要性 |
|------|------|------|--------|
| **TTFT** | Time to First Token | 用户发送请求 → 收到第一个 token 的延迟 | **用户体验核心指标**（感知速度） |
| **TPOT** | Time Per Output Token | 每个生成 token 的平均间隔 | "打字速度"感 |
| **Latency (E2E)** | End-to-End Latency | 发送请求 → 收到最后一个 token 的总延迟 | 整体响应时间 |
| **Throughput** | Tokens/sec | 每秒生成的 token 总数 | 系统效率指标 |
| **Requests/sec** | RPS / QPS | 每秒处理的请求数 | 服务容量指标 |
| **GPU Utilization** | GPU 利用率 | GPU 实际计算时间占比 | 资源效率指标 |

### 1.3 TTFT vs TPOT 的意义

```python
def ttft_tpot_explain():
    """TTFT 和 TPOT 的直观理解"""
    
    explanation = """
    ┌────────────────────────────────────────────────────┐
    │                 请求生命周期                         │
    ├────────────────────────────────────────────────────┤
    │                                                    │
    │  用户发出请求                                       │
    │      │                                             │
    │      │ ←←←← TTFT (Time to First Token) ←←←←←←←←  │
    │      │     Prompt 处理阶段                          │
    │      │     （读取 prompt + 计算 KV Cache）          │
    │      ▼                                             │
    │  ╭────────╮                                        │
    │  │ 第1个  │ ← 用户看到第一个字！                   │
    │  │ Token  │                                        │
    │  ╰────────╯                                        │
    │      │                                             │
    │      │ ← TPOT → ← TPOT → ← TPOT → ← TPOT →       │
    │      │     Token 生成阶段（逐个输出）               │
    │      ▼                                             │
    │  ╭────────╮                                        │
    │  │ 最后1个│                                        │
    │  │ Token  │                                        │
    │  ╰────────╯                                        │
    │                                                    │
    │  TTFT 影响: "这个服务快不快?"（第一印象）            │
    │  TPOT 影响: "输出流畅吗?"（阅读体验）               │
    │                                                    │
    │  目标值参考:                                       │
    │    在线聊天: TTFT < 500ms, TPOT < 50ms             │
    │    批量处理: TTFT < 2s,   TPOT < 100ms             │
    └────────────────────────────────────────────────────┘
    """
    print(explanation)

ttft_tpot_explain()
```

## 二、vLLM 内置 Benchmark 工具

### 2.1 Throughput Benchmark

测量纯吞吐能力（tokens/sec）：

```bash
# 基本用法
python -m vllm.benchmark.benchmark_throughput \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 1000 \
    --output-json throughput_results.json

# 完整参数
python -m vllm.benchmark.benchmark_throughput \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --dtype auto \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --num-prompts 1000 \                    # 测试请求数
    --prompt-len 512 \                      # 平均 prompt 长度
    --generation-len 128 \                  # 平均生成长度 \
    --qps 10 \                              # 目标 QPS (0=不限流)
    --output-json results.json              # 输出 JSON 结果
```

**输出示例**：
```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "num_prompts": 1000,
  "throughput": 2456.3,
  "avg_prompt_throughput": 8923.4,
  "avg_generation_throughput": 2456.3,
  "prompt_tokens_per_req": 128,
  "generation_tokens_per_req": 32,
  "total_input_tokens": 128000,
  "total_output_tokens": 32000,
  "total_time": 13.03
}
```

### 2.2 Serving Benchmark

模拟真实 API 服务场景（含 TTFT/TPOT）：

```bash
# 基本用法（需要先启动 vLLM 服务）
python -m vllm.benchmark.benchmark_serving \
    --model Qwen/Qwen2.5-7B-Instruct \
    --endpoint http://localhost:8000 \
    --num-prompts 500 \
    --request-rate 10

# 完整参数
python -m vllm.benchmark.benchmark_serving \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --endpoint http://localhost:8000/v1 \
    --api-key token-abc123 \
    --num-prompts 1000 \
    --request-rate 50 \                     # 每秒请求速率
    --max-tokens 256 \                       # 最大生成 tokens
    --input-len 1024 \                       # 输入长度
    --output-len 128 \                       # 输出长度 \
    --percentage-varied-requests 20 \        # % 的请求使用不同参数
    --additional-sampled-tokens 0.2 \        # 生成长度的变化幅度
    --share-gpu-memory-fraction 0.9 \
    --output-format json \
    --save-result serving_results.json
```

**输出字段说明**：

| 字段 | 含义 |
|------|------|
| `mean_ttft_ms` | 平均首 token 延迟 |
| `median_ttft_ms` | 中位数 TTFT |
| `p99_ttft_ms` | P99 TTFT（99% 的请求在此时间内收到首个 token） |
| `mean_tpot_ms` | 平均每 token 间隔 |
| `median_tpot_ms` | 中位数 TPOT |
| `p99_tpot_ms` | P99 TPOT |
| `mean_e2e_latency_ms` | 平均端到端延迟 |
| `requests_per_second` | 实际 QPS |
| `successful_requests` | 成功完成的请求数 |
| `failed_requests` | 失败的请求数 |
| `total_output_tokens` | 总输出 token 数 |

## 三、自定义 Benchmark 框架

### 3.1 完整的 A/B 测试框架

比如下面的程序可以对比两种配置的性能差异：

```python
import asyncio
import time
import statistics
import aiohttp
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    name: str
    endpoint: str
    model: str
    num_requests: int = 200
    concurrency: int = 10
    max_tokens: int = 128
    temperature: float = 0.7
    
@dataclass
class RequestResult:
    """单次请求结果"""
    request_id: int
    ttft_ms: float
    tpot_ms: float
    e2e_latency_ms: float
    output_tokens: int
    success: bool
    error: str = ""

@dataclass 
class BenchmarkReport:
    """基准测试报告"""
    config: BenchmarkConfig
    results: List[RequestResult] = field(default_factory=list)
    
    @property
    def successful(self):
        return [r for r in self.results if r.success]
    
    @property
    def ttft_p50(self):
        return statistics.median(r.ttft_ms for r in self.successful)
    
    @property
    def ttft_p99(self):
        sorted_ttfts = sorted(r.ttft_ms for r in self.successful)
        return sorted_ttfts[int(len(sorted_ttfts) * 0.99)]
    
    @property
    def tpot_mean(self):
        return statistics.mean(r.tpot_ms for r in self.successful)
    
    @property
    def e2e_p50(self):
        return statistics.median(r.e2e_latency_ms for r in self.successful)
    
    @property
    def qps(self):
        total_time = max(r.e2e_latency_ms for r in self.results) / 1000
        return len(self.successful) / max(total_time, 0.001)
    
    @property
    def throughput_tok_s(self):
        total_tokens = sum(r.output_tokens for r in self.successful)
        total_time = max(r.e2e_latency_ms for r in self.results) / 1000
        return total_tokens / max(total_time, 0.001)


class VllmBenchmark:
    """vLLM 性能基准测试工具"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._prompts = None
    
    def _generate_prompts(self) -> list[str]:
        """生成测试 prompt 列表"""
        templates = [
            "请解释{}的概念，用简单的语言。",
            "写一段关于{}的技术文档。",
            "总结一下{}的主要特点。",
            "{}的优点和缺点是什么？",
            "如何学习{}？给出5条建议。",
        ]
        
        topics = [
            "机器学习", "深度学习", "自然语言处理", "计算机视觉", "强化学习",
            "Transformer", "注意力机制", "BERT", "GPT", "LLM",
            "PagedAttention", "KV Cache", "量化", "张量并行", "LoRA",
            "RAG", "向量数据库", "Prompt Engineering", "Fine-tuning", "推理优化",
        ]
        
        import random
        random.seed(42)
        prompts = []
        for i in range(self.config.num_requests):
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            prompts.append(template.format(topic))
        
        return prompts
    
    async def _single_request(
        self, session: aiohttp.ClientSession, 
        prompt: str, req_id: int
    ) -> RequestResult:
        """执行单次请求并收集指标"""
        
        start = time.time()
        ttft = None
        token_count = 0
        
        try:
            async with session.post(
                f"{self.config.endpoint}/v1/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "stream": True,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                
                if response.status != 200:
                    return RequestResult(req_id, 0, 0, 0, 0, False, 
                                       f"HTTP {response.status}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        import json as json_mod
                        data = json_mod.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        
                        if delta.get("content") and ttft is None:
                            ttft = (time.time() - start) * 1000
                        
                        if delta.get("content"):
                            token_count += 1
                    except:
                        pass
                
                e2e = (time.time() - start) * 1000
                tpot = ((e2e - ttft) / max(token_count - 1, 1)) if ttft and token_count > 1 else 0
                
                return RequestResult(
                    request_id=req_id,
                    ttft_ms=ttft or 0,
                    tpot_ms=tpot,
                    e2e_latency_ms=e2e,
                    output_tokens=token_count,
                    success=True,
                )
                
        except Exception as e:
            return RequestResult(req_id, 0, 0, 0, 0, False, str(e))
    
    async def run(self) -> BenchmarkReport:
        """执行完整基准测试"""
        
        import random
        prompts = self._generate_prompts()
        report = BenchmarkReport(config=self.config)
        
        semaphore = asyncio.Semaphore(self.config.concurrency)
        
        print(f"\n{'='*60}")
        print(f"基准测试: {self.config.name}")
        print(f"  端点: {self.config.endpoint}")
        print(f"  请求数: {self.config.num_requests}")
        print(f"  并发度: {self.config.concurrency}")
        print(f"{'='*60}\n")
        
        start_all = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i, prompt in enumerate(prompts):
                async def bounded_request(p=prompt, rid=i):
                    async with semaphore:
                        result = await self._single_request(session, p, rid)
                        report.results.append(result)
                        
                        if i % 50 == 0 and i > 0:
                            elapsed = time.time() - start_all
                            done = sum(1 for r in report.results)
                            pct = done / self.config.num_requests * 100
                            print(f"  进度: {done}/{self.config.num_requests} "
                                  f"({pct:.0f}%) | {elapsed:.1f}s")
                
                tasks.append(bounded_request())
            
            await asyncio.gather(*tasks)
        
        return report


def compare_configs():
    """A/B 对比两种配置"""
    
    configs = [
        BenchmarkConfig(
            name="配置A: TP=1, 无量化",
            endpoint="http://localhost:8000",
            model="Qwen/Qwen2.5-7B-Instruct",
            num_requests=100,
            concurrency=5,
            max_tokens=64,
        ),
        BenchmarkConfig(
            name="配置B: TP=1, AWQ INT4",
            endpoint="http://localhost:8001",
            model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            num_requests=100,
            concurrency=5,
            max_tokens=64,
        ),
    ]
    
    reports = []
    for cfg in configs:
        benchmark = VllmBenchmark(cfg)
        report = asyncio.run(benchmark.run())
        reports.append(report)
    
    print(f"\n{'='*70}")
    print(f"A/B 性能对比")
    print(f"{'='*70}")
    
    header = f"{'指标':<22}"
    for r in reports:
        header += f" | {r.config.name:<25}"
    print(header)
    print("-" * 75)
    
    metrics = [
        ("成功率", lambda r: f"{len(r.successful)}/{len(r.results)} ({len(r.successful)/len(r.results):.0%})"),
        ("TTFT P50 (ms)", lambda r: f"{r.ttft_p50:.1f}"),
        ("TTFT P99 (ms)", lambda r: f"{r.ttft_p99:.1f}"),
        ("TPOT Mean (ms)", lambda r: f"{r.tpot_mean:.1f}"),
        ("E2E P50 (ms)", lambda r: f"{r.e2e_p50:.1f}"),
        ("QPS", lambda r: f"{r.qps:.1f}"),
        ("Throughput (tok/s)", lambda r: f"{r.throughput_tok_s:.0f}"),
    ]
    
    for metric_name, fn in metrics:
        row = f"{metric_name:<22}"
        for r in reports:
            row += f" | {fn(r):<25}"
        print(row)
    
    if len(reports) >= 2:
        r_a, r_b = reports[0], reports[1]
        
        print("\n[提升分析]")
        ttft_speedup = r_a.ttft_p50 / max(r_b.ttft_p50, 0.001)
        qps_improvement = (r_b.qps - r_a.qps) / max(r_a.qps, 0.001) * 100
        tok_speedup = r_b.throughput_tok_s / max(r_a.throughput_tok_s, 0.001)
        
        print(f"  TTFT: {'↓' if ttft_speedup > 1 else '↑'} {abs(ttft_speedup-1)*100:.1f}%")
        print(f"  QPS:  {'↑' if qps_improvement > 0 else '↓'} {abs(qps_improvement):.1f}%")
        print(f"  吞吐: {tok_speedup:.2f}x")


compare_configs()
```

### 3.2 典型输出

```
======================================================================
A/B 性能对比
======================================================================
指标                   | 配置A: TP=1, 无量化     | 配置B: TP=1, AWQ INT4  
-----------------------------------------------------------------------
成功率                  | 100/100 (100%)         | 100/100 (100%)
TTFT P50 (ms)          | 423.5                  | 287.2
TTFT P99 (ms)          | 892.1                  | 567.3
TPOT Mean (ms)         | 38.9                   | 26.4
E2E P50 (ms)           | 1654.2                 | 1123.8
QPS                    | 12.3                   | 18.7
Throughput (tok/s)     | 1892                   | 2756

[提升分析]
  TTFT: ↓ 32.2%
  QPS:  ↑ 52.0%
  吞吐: 1.46x
```

## 四、不同配置下的性能矩阵

### 4.1 参考数据（Llama 3 8B on A100 80GB）

| 配置 | TP | 量化 | TTFT P50 | TPOT | Throughput | VRAM |
|------|-----|------|---------|------|-----------|------|
| Baseline | 1 | FP16 | 450ms | 42ms | 1200 t/s | 16GB |
| +AWQ INT4 | 1 | INT4 | 280ms | 28ms | 1800 t/s | 5GB |
| +TP=2 | 2 | FP16 | 380ms | 48ms | 2100 t/s | 8GB/GPU |
| +TP=4 | 4 | FP16 | 420ms | 58ms | 2400 t/s | 4GB/GPU |
| TP=2+AWQ | 2 | INT4 | 220ms | 22ms | 2900 t/s | 2.5GB/GPU |

### 4.2 选型决策依据

```python
def benchmark_based_decision():
    """基于 benchmark 数据的选型决策"""
    
    decision_rules = """
    ┌────────────────────────────────────────────────────────┐
    │              基于 Benchmark 的决策矩阵                  │
    ├────────────────┬──────────┬───────────┬────────────────┤
    │ 你的首要目标    │ 推荐配置  │ 预期收益   │ 注意事项       │
    ├────────────────┼──────────┼───────────┼────────────────┤
    │ 最低 TTFT      │ AWQ+小模型│ ↓40-50%   │ 牺牲少量质量   │
    │ (< 300ms)      │ PrefixCache│           │                │
    ├────────────────┼──────────┼───────────┼────────────────┤
    │ 最高吞吐量     │ TP=2+AWQ │ 2-3x      │ 需要 2 卡       │
    │ (> 2500 t/s)   │          │           │                │
    ├────────────────┼──────────┼───────────┼────────────────┤
    │ 最大并发用户   │ TP+大batch│ 3-5x QPS  │ TTFT 可能增加   │
    │ (> 100 QPS)    │ delay=0.5│           │                │
    ├────────────────┼──────────┼───────────┼────────────────┤
    │ 最省显存       │ AWQ INT4 │ ↓70% VRAM │ 需要预转换模型  │
    │ (< 8GB)        │ 或 GPTQ  │           │                │
    ├────────────────┼──────────┼───────────┼────────────────┤
    │ 平衡方案       │ TP=1+AWQ │ 各项均衡  │ 单卡部署最简单  │
    │ (推荐默认)     │ +PrefixC │           │                │
    └────────────────┴──────────┴───────────┴────────────────┘
    """
    print(decision_rules)

benchmark_based_decision()
```

---

## 五、总结

本节全面覆盖了 vLLM 性能基准测试的方法论和工具：

| 工具/方法 | 适用场景 | 核心输出 |
|----------|---------|---------|
| **benchmark_throughput** | 离线吞吐测试 | tokens/sec（纯引擎性能） |
| **benchmark_serving** | 在线服务测试 | TTFT/TPOT/E2E/QPS（真实 API 场景） |
| **自定义 A/B 框架** | 配置对比 | 完整延迟分布 + 统计显著性 |
| **指标金字塔** | 监控设计 | 业务→系统→引擎 三层指标体系 |

**核心要点回顾**：

1. **TTFT 是用户体验的第一印象**——在线聊天场景应优先优化（< 500ms）
2. **TPOT 决定"打字速度感"**——< 50ms 感觉流畅，> 100ms 感觉卡顿
3. **永远用 P99 而非平均值做 SLA**——平均值掩盖长尾问题
4. **benchmark_throughput 测的是引擎极限**，benchmark_serving 测的是实际服务能力
5. **A/B 测试是唯一科学的优化验证方法**——不要凭感觉判断配置好坏

下一节我们将学习 **硬件选型与成本参考**——如何用最合理的预算获得最佳性能。
