# 08-2 推理参数调优

## 参数与性能的量化关系

在 04-3 节中我们学习了每个参数的含义和推荐值。这一节从**性能影响**的角度重新审视这些参数——每一个参数的选择都会直接或间接地影响推理速度、内存占用和输出质量。

## Num_ctx：上下文长度的代价模型

`num_ctx`（上下文窗口大小）是影响性能最显著的参数之一，但它的代价往往被低估：

```
┌─────────────────────────────────────────────────────────────┐
│           num_ctx 对资源的影响 (以 7B 模型为例)               │
│                                                             │
│  KV Cache 大小公式:                                        │
│    KV_Size = 2 × num_layers × num_ctx × hidden_dim × 2B   │
│                                                             │
│  对于 Qwen2.5-7B (32层, hidden=4096):                       │
│                                                             │
│  num_ctx     KV Cache 大小    额外内存    占总内存比例        │
│  ────────    ─────────────    ────────    ────────────       │
│  512         ~50 MB           ~0 MB      可忽略              │
│  2048        ~200 MB          ~200 MB    +5%                │
│  4096        ~400 MB          ~400 MB    +10%               │
│  8192        ~800 MB          ~800 MB    +20%               │
│  16384       ~1.6 GB          ~1.6 GB     +40% ⚠️            │
│  32768       ~3.2 GB          ~3.2 GB     +80% 🔴            │
│  128000      ~12.4 GB         >12 GB     不可能(!)           │
│                                                             │
│  同时: 更大的 ctx → 每个 token 的 attention 计算量增大       │
│        → 每秒生成的 token 数下降                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 实测数据：num_ctx vs 延迟

```python
#!/usr/bin/env python3
"""num_ctx 性能基准测试"""

import requests
import time


def benchmark_num_ctx():
    
    model = "qwen2.5:7b"
    base_url = "http://localhost:11434/api/chat"
    
    ctx_sizes = [512, 2048, 4096, 8192]
    
    # 使用不同长度的问题
    questions = {
        512: "你好",
        2048: "请用三句话解释什么是机器学习。" * 10,
        4096: "请详细说明 Docker 和虚拟机的区别，包括架构、性能、隔离性、使用场景等五个维度。"
                 "请每个维度至少写三段话。" * 8,
        8192: ("请写一篇关于人工智能发展史的长文，" 
                 "包括图灵测试、专家系统、深度学习爆发、" 
                 "大语言模型时代等阶段。" * 20),
    }
    
    print(f"\n{'='*65}")
    print(f"  num_ctx 性能基准 (模型: {model})")
    print(f"{'='*65}")
    print(f"{'ctx':>7s} {'首token':>10s} {'每token':>12s} "
          f"{'生成100tok':>12s} {'KV估算':>10s}")
    print(f"{'-'*7} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    
    for ctx in ctx_sizes:
        # 设置 num_ctx
        prompt = questions.get(ctx, "Hello")
        
        start = time.time()
        resp = requests.post(base_url, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_ctx": ctx,
                "num_predict": 100,
                "temperature": 0.1
            }
        }, timeout=120)
        
        total_time = time.time() - start
        data = resp.json()
        
        ttft = data.get("prompt_eval_duration", 0) / 1e9  # Time to First Token
        eval_count = data.get("eval_count", 0)
        tokens_per_sec = eval_count / (total_time - ttft) if total_time > ttft else 0
        
        kv_estimate = 2 * 32 * ctx * 4096 * 2 / (1024**3)
        
        print(f"{ctx:>7d} {ttft*1000:>8.0f}ms "
              f"{1000/tokens_per_sec if tokens_per_sec > 0 else 0:>8.1f}ms/tok "
              f"{total_time:.2f}s {kv_estimate:>9.2f}GB")


if __name__ == "__main__":
    benchmark_num_ctx()
```

## Num_batch：批处理大小的双刃剑

`num_batch` 控制 Ollama 同时处理多少个请求（或在内部处理多少个序列位置）：

```bash
# 交互式场景（单个用户对话）
OLLAMA_NUM_BATCH=1    # 推荐：最低延迟优先

# 批处理场景（离线任务）
OLLAMA_NUM_BATCH=8    # 推荐：吞吐优先

# API 服务场景（多用户并发）
OLLAMA_NUM_BATCH=4    # 平衡延迟和吞吐
```

**关键权衡**：

| num_batch | 延迟 | 吞吐 | GPU 显存占用 | 适用场景 |
|----------|------|------|-------------|---------|
| 1 | **最低** | 低 | 最小 | 实时对话 |
| 4 | 中等 | 高 | 中等 | 小型 API 服务 |
| 8 | 较高 | **最高** | 较大 | 离线批处理 |
| 16+ | 高 | 可能反而下降 | 很大 | 极少使用 |

**注意**：当 `num_batch > 1` 时，Ollama 会将多个请求打包成一个 batch 一起做矩阵运算。这提高了 GPU 利用率，但也增加了单请求的等待时间（因为要等其他请求一起 batch）。对于交互式应用，`num_batch=1` 几乎总是最佳选择。

## Num_predict：控制输出长度来节省计算

```python
#!/usr/bin/env python3
"""num_predict 对响应时间和质量的影响"""

import requests
import time

def test_num_predict():
    
    model = "qwen2.5:7b"
    question = "请解释什么是微服务架构"
    
    predicts = [64, 256, 1024, 2048, 4096]
    
    print(f"\n问题: {question}\n")
    print(f"{'predict':>10s} {'耗时':>8s} {'tokens':>8s} "
          f"{'速度':>10s} {'回答完整性'}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*25}")
    
    for np in predicts:
        start = time.time()
        resp = requests.post("http://localhost:11434/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
            "options": {"num_predict": np, "temperature": 0.5}
        }, timeout=180)
        
        elapsed = time.time() - start
        data = resp.json()
        content = data["message"]["content"]
        tokens = data.get("eval_count", 0)
        tps = tokens / elapsed if elapsed > 0 else 0
        
        completeness = "完整" if len(content) > 100 else \
                        "中等" if len(content) > 30 else "被截断"
        
        print(f"{np:>10d} {elapsed:>7.1f}s {tokens:>8d} "
              f"{tps:>9.1f} t/s  {completeness}")


if __name__ == "__main__":
    test_num_predict()
```

输出示例：

```
predict      耗时    tokens    速度       回答完整性
---------- -------- -------- ---------- -------------------------
        64     0.8s       58      72.5 t/s  被截断
       256     1.5s      198     132.0 t/s  中等
      1024     4.2s      512     121.9 t/s  完整
      2044     8.1s     1024     126.4 t/s  完整
      4096    15.3s     2056     134.4 t/s  完整（冗余）
```

**发现**：`num_predict` 从 256 增加到 4096，速度几乎不变（因为自回归是逐 token 生成的），但总耗时线性增长。所以：
- **需要快速回答** → 设小值（256-512），配合 stop 序列让模型自然结束
- **需要完整回答** → 设较大值（1024-2048）
- **避免无限生成** → **始终设置合理的 `num_predict`**，不要依赖默认值

## Keep_alive：内存管理的核心旋钮

`OLLAMA_KEEP_ALIVE` 决定模型在内存中保持加载的时间：

```bash
# 场景 A：频繁切换模型的开发环境
OLLAMA_KEEP_ALIVE=0    # 用完立即卸载，释放内存给下一个模型

# 场景 B：固定使用一个模型的生产服务
OLLAMA_KEEP_ALIVE=-1   # 永久驻留（-1 或 "infinity" 表示不卸载）

# 场景 C：多用户共享，有主力和备用模型
OLLAMA_KEEP_ALIVE=5m    # 主力模型常驻，备用模型 5 分钟不用就卸载
OLLAMA_MAX_LOADED_MODELS=2  # 最多同时保留 2 个模型
```

### 内存竞争的实际表现

```python
def memory_competition_demo():
    """模拟多模型内存竞争"""
    
    # 假设系统有 16GB 可用内存给 Ollama
    total_ram = 16  # GB
    
    models = [
        ("qwen2.5:7b", 4.3),   # Q4_K_M 大小
        ("llama3.1:8b", 4.7),
        ("nomic-embed-text", 0.27),
        ("llava", 4.5),
        ("deepseek-coder:6.7b", 3.8),
    ]
    
    loaded = []
    used_ram = 0
    
    print("\n模拟多模型加载/卸载过程:\n")
    
    for model_name, size_gb in models:
        available = total_ram - used_ram
        
        if size_gb <= available:
            loaded.append(model_name)
            used_ram += size_gb
            status = "✅ 已加载"
        else:
            # 需要卸载最久未使用的模型
            if loaded:
                evicted = loaded.pop(0)
                used_ram -= next(s for m, s in models if m == evicted)
                status = f"⚠️ 卸载 {evicted} 后加载"
                loaded.append(model_name)
                used_ram += size_gb
            else:
                status = f"❌ 内存不足 (需{size_gb}GB，可用{available:.1f}GB)"
        
        bar_loaded = "█" * int(used_ram / total_ram * 20)
        bar_free = "░" * (20 - len(bar_loaded))
        
        print(f"  [{bar_loaded}{bar_free}] {used_ram:>5.1f}/{total_ram}GB | "
              f"+{model_name:<25s} ({size_gb}GB) {status}")


if __name__ == "__main__":
    memory_competition_demo()
```

## 性能基准测试方法论

```python
#!/usr/bin/env python3
"""完整的 Ollama 性能基准测试框架"""

import requests
import time
import statistics
import json
from datetime import datetime


class OllamaBenchmark:
    """Ollama 性能基准测试套件"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.results = []
    
    def check_health(self):
        """健康检查"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except:
            return False
    
    def run_latency_test(self, model, iterations=10):
        """延迟测试: 单次请求的端到端时间"""
        
        latencies = []
        
        for i in range(iterations):
            start = time.perf_counter()
            
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"测试{i}: 你好"}],
                    "stream": False,
                    "options": {"num_predict": 50}
                },
                timeout=60
            )
            
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        
        return {
            "test_type": "latency",
            "model": model,
            "iterations": iterations,
            "avg_ms": round(statistics.mean(latencies), 2),
            "p50_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(sorted(latencies)[int(len(latencies)*0.95)], 2),
            "p99_ms": round(sorted(latencies)[int(len(latencies)*0.99)], 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
        }
    
    def run_throughput_test(self, model, duration_sec=30):
        """吞吐量测试: 单位时间内的 token 生成数"""
        
        import threading
        queue = []
        total_tokens = 0
        lock = threading.Lock()
        start_time = time.time()
        
        def worker():
            nonlocal total_tokens
            while time.time() - start_time < duration_sec:
                try:
                    resp = requests.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "写一首短诗"}],
                            "stream": False,
                            "options": {"num_predict": 100}
                        },
                        timeout=30
                    )
                    data = resp.json()
                    tokens = data.get("eval_count", 0)
                    
                    with lock:
                        total_tokens += tokens
                        
                except Exception:
                    continue
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        actual_duration = time.time() - start_time
        throughput = total_tokens / actual_duration if actual_duration > 0 else 0
        
        return {
            "test_type": "throughput",
            "model": model,
            "duration_sec": round(actual_duration, 1),
            "total_tokens": total_tokens,
            "tokens_per_sec": round(throughput, 1),
        }
    
    def run_full_benchmark(self, models=None):
        """运行完整基准测试"""
        
        if not self.check_health():
            print("❌ Ollama 服务不可达!")
            return
        
        if models is None:
            models = ["qwen2.5:1.5b", "qwen2.5:7b"]
        
        print(f"\n{'='*70}")
        print(f"  Ollama 性能基准测试")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        all_results = []
        
        for model in models:
            print(f"\n🔬 测试模型: {model}")
            
            latency_result = self.run_latency_test(model, iterations=5)
            all_results.append(latency_result)
            
            print(f"  延迟: P50={latency_result['p50_ms']}ms  "
                  f"P95={latency_result['p95_ms']}ms  "
                  f"P99={latency_result['p99_ms']}ms")
            
            throughput_result = self.run_throughput_test(model, duration_sec=15)
            all_results.append(throughputput_result)
            
            print(f"  吞吐: {throughput_result['tokens_per_sec']} tok/s")
        
        # 保存结果
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": all_results
        }
        
        with open("ollama_benchmark.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 结果已保存到 ollama_benchmark.json")
        return all_results


if __name__ == "__main__":
    bench = OllamaBenchmark()
    bench.run_full_benchmark(["qwen2.5:1.5b", "qwen2.5:7b"])
```

## 本章小结

这一节从性能角度深入分析了每个推理参数的影响：

1. **`num_ctx` 是最大的隐性性能杀手**——每增加一倍上下文，KV Cache 和 attention 计算量翻倍；按需设置，不要一味求大
2. **`num_batch=1` 是交互式场景的最优选择**——批处理虽然提高吞吐但牺牲延迟
3. **`num_predict` 应该始终显式设置**——防止模型无限生成消耗资源
4. **`keep_alive` 是内存管理的核心旋钮**——决定模型驻留策略和多模型共存行为
5. **完整的基准测试框架**涵盖 P50/P95/P99 延迟和 tokens/sec 吞吐两个核心指标
6. **优化工作流**：基线测试 → 单变量扫描 → 组合优化 → 验证

下一节我们将讨论如何通过并发控制和队列管理进一步提升系统吞吐。
