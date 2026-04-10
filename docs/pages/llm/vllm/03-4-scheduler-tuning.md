# 调度性能调优实战

## 白板导读

前三节我们建立了完整的理论框架：Static Batching 的缺陷 → Continuous Batching 的思想 → Scheduler 三状态机 → Preemption 抢占机制。现在要进入最实用的环节——**如何根据你的具体场景调优 vLLM 的调度参数**。这一节将提供不同场景（在线聊天 / API 服务 / 批量处理 / 混合负载）的推荐配置、参数调优的决策树、以及一套完整的 Benchmark 方法论来验证调优效果。你将学会如何用数据驱动的方式找到最优配置，而不是凭感觉瞎猜。

---

## 4.1 场景驱动的参数配置

### 配置速查表

```python
"""
vLLM 调度参数场景化配置速查
"""

SCENARIOS = {
    "online_chat": {
        "name": "在线聊天 / 实时对话",
        "description": "用户与 AI 一对一实时交互，体感延迟是第一优先级",
        "characteristics": [
            "请求到达间隔不规律",
            "平均序列长度短 (100-500 tokens)",
            "输出长度中等 (100-300 tokens)",
            "并发数低-中 (10-50)",
            "TTFT 要求 < 200ms",
        ],
        "recommended_params": {
            "--max-num-seqs": 64,
            "--max-num-batched-tokens": "auto",  # 或设为 2048
            "--scheduler-delay-factor": 0.05,       # 极低延迟
            "--max-model-len": 4096,              # 不需要太长
            "--gpu-memory-utilization": 0.90,
            "--enable-prefix-caching": "",             # 自动启用
        },
        "rationale": "低 delay-factor 让新请求几乎零等待加入 RUNNING；"
                     "较小的 max-model-len 减少 KV Cache 预留；"
                     "prefix-caching 让相同 System Prompt 只算一次。",
    },
    
    "api_service": {
        "name": "通用 LLM API 服务",
        "description": "多个应用通过 HTTP API 调用，需要平衡延迟和吞吐",
        "characteristics": [
            "请求频率高且稳定",
            "序列长度多样化 (50-8000 tokens)",
            "输出长度变化大 (50-2000 tokens)",
            "并发数中-高 (50-200)",
            "P99 延迟 < 5s 可接受",
        ],
        "recommended_params": {
            "--max-num-seqs": 128,
            "--max-num-batched-tokens": "auto",
            "--scheduler-delay-factor": 0.1,       # 平衡点
            "--max-model-len": 8192 或 16384,
            "--gpu-memory-utilization": 0.92,
            "--enable-prefix-caching": "",
            "--swap-space": 4,
        },
        "rationale": "delay-factor=0.1 是默认推荐值，在 TTFT 和吞吐间取得平衡；"
                     "较大的 max-model-len 支持 RAG 等长文档场景；"
                     "开启 swap-space 允许偶尔 preempt 长请求。",
    },
    
    "batch_processing": {
        "name": "离线批量处理 / 评估",
        "description": "大规模数据集推理，总完成时间优先于单请求延迟",
        "characteristics": [
            "请求批量提交 (成百上千)",
            "序列长度相对均匀",
            "通常不需要流式输出",
            "追求最大 GPU 利用率",
            "可以容忍较高的 TTFT",
        ],
        "recommended_params": {
            "--max-num-seqs": 256,
            "--max-num-batched-tokens": "auto",  # 或设为 8192+
            "--scheduler-delay-factor": 0.5,       # 高吞吐模式
            "--max-model-len": 16384 or 32768,
            "--gpu-memory-utilization": 0.95,      # 激进利用
            "--disable-log-requests": "",             # 减少日志 I/O
        },
        "rationale": "高 delay-factor 让每 iteration 能凑更多请求一起处理 → "
                     "更大的有效 batch size → 更高的 GPU利用率；"
                     "disable-log-requests 在高吞吐下减少 I/O 开销。",
    },
    
    "mixed_workload": {
        "name": "混合负载 (长短请求共存)",
        "description": "同时有简单分类任务和复杂生成任务",
        "characteristics": [
            "请求类型多样 (分类/翻译/长文生成/代码)",
            "序列长度差异巨大 (10 - 32000+ tokens)",
            "部分请求有严格的 SLA 要求",
            "可能出现突发流量",
        ],
        "recommended_params": {
            "--max-num-seqs": 128,
            "--max-num-batched-tokens": "auto",
            "--scheduler-delay-factor": 0.15,      # 中等延迟
            "--max-model-len": 16384,
            "--gpu-memory-utilization": 0.90,
            "--swap-space": 8,                   # 较大 swap 空间
            "--enable-prefix-caching": "",
        },
        "rationale": "适中的 delay-factor 在长短请求间取得折中；"
                     "较大 swap-space 支持频繁的 preempt；"
                     "prefix-caching 对共享 System Prompt 的请求效果显著。",
    },
}


def print_scenario_guide():
    """打印完整的场景配置指南"""
    
    print("=" * 80)
    print("vLLM 调度参数场景化配置指南")
    print("=" * 80)
    
    for key, config in SCENARIOS.items():
        print(f"\n{'█'*80}")
        print(f"📋 {config['name']}")
        print(f"   {config['description']}")
        print(f"\n   特征:")
        for ch in config["characteristics"]:
            print(f"   • {ch}")
        
        print(f"\n   推荐启动命令:")
        params = config["recommended_params"]
        cmd_parts = ["python -m vllm.entrypoints.openai.api_server",
                   "--model Qwen/Qwen2.5-7B-Instruct"]
        for flag, value in params.items():
            if value:
                cmd_parts.append(f"{flag} {value}")
            else:
                cmd_parts.append(flag)
        
        # 格式化为多行命令
        cmd = " \\\n  ".join(cmd_parts)
        print(f"   {cmd}")
        
        print(f"\n   📝 调优理由:")
        print(f"   {config['rationale']}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    print_scenario_guide()
```

输出：

```
================================================================================
vLLM 调度参数场景化配置指南
================================================================================

████████████████████████████████████████████████████████████████████████████████████████████████████████████████

📋 在线聊天 / 实时对话
   用户与 AI 一对一实时交互，体感延迟是第一优先级

   特征:
   • 请求到达间隔不规律
   • 平均序列长度短 (100-500 tokens)
   • 输出长度中等 (100-300 tokens)
   • 并发数低-中 (10-50)
   • TTFT 要求 < 200ms

   推荐启动命令:
   python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-7B-Instruct \
     --max-num-seqs 64 \
     --max-model-len 4096 \
     --scheduler-delay-factor 0.05 \
     --gpu-memory-utilization 0.90 \
     --enable-prefix-caching

   📝 调优理由:
   低 delay-factor 让新请求几乎零等待加入 RUNNING；较小的 max-model-len 减少 KV Cache 
   预留；prefix-caching 让相同 System Prompt 只算一次。

████████████████████████████████████████████████████████████████████████████████████████████████████████████████

📋 通用 LLM API 服务
   多个应用通过 HTTP API 调用，需要平衡延迟和吞吐

   特征:
   • 请求频率高且稳定
   • 序列长度多样化 (50-8000 tokens)
   • 输出长度变化大 (50-2000 tokens)
   • 并发数中-高 (50-200)
   • P99 延迟 < 5s 可接受

   推荐启动命令:
   python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-7B-Instruct \
     --max-num-seqs 128 \
     --max-model-len 8192 \
     --scheduler-delay-factor 0.1 \
     --gpu-memory-utilization 0.92 \
     --enable-prefix-caching \
     --swap-space 4

   📝 调优理由:
   delay-factor=0.1 是默认推荐值，在 TTFT 和吞吐间取得平衡；较大的 max-model-len 支持 
   RAG 等长文档场景；开启 swap-space 允许偶尔 preempt 长请求。

... (其余场景省略，结构相同)
```

---

## 4.2 参数调优决策树

当你在实际环境中遇到性能问题时，可以用以下决策树来定位应该调整哪个参数：

```
问题: 吞吐量不达标?
│
├─ TTFT 很高 (>1s)?
│   └─→ 降低 scheduler-delay-factor (1.0 → 0.5 → 0.1 → 0.05)
│   └─→ 如果已经是 0.05 还很高 → 检查是否模型加载慢 / 网络 I/O 问题
│
├─ GPU 利用率低 (<70%)?
│   ├─ running 序列数很少?
│   │   └─→ 提高 max-num-seqs (64 → 128 → 256)
│   ├─ 但显存不够加更多 seq?
│   │   └─→ 降低 max-model-len 或启用量化
│   └─→ 检查是否有大量时间浪费在等待上
│
├─ P99 延迟过高?
│   ├─ 单个请求本身就慢?
│   │   └─→ 这是模型/量化问题，不是调度问题
│   └─ 排队时间长?
│       ├─ waiting 队列积压?
│       │   └─→ 增加 max-num-seqs + 考虑扩容
│       └─ 抢占频繁?
│           └─→ 增大 swap-space 或优化 victim 选择策略
│
└─ 错误率突增 (OOM)?
    ├─ 真的显存不够?
    │   └─→ 减小模型 / 加大量化 / 多卡 TP
    └─ Block 碎片化严重?
        └─→ 可能是 max-model-len 过大导致内部碎片
```

### 常见反模式（Anti-Patterns）

| 反模式 | 表现 | 正确做法 |
|:---|:---|:---|
| **max-model-len = 32768** 用于聊天 | 大量 KV Cache 预留给每个短对话，利用率极低 | 聊类设置：聊天用 4096，RAG 用 16384 |
| **delay-factor = 0** | 每个 iteration 都立即调度，batch size 小 | 设为 0.05-0.15 |
| **max-num-seqs = 1024** | 太多并发导致每个请求分到的计算时间不足 | 根据实际 QPS 设置，通常 64-256 |
| **swap-space = 0** | 无法抢占，Block 满时拒绝率飙升 | 至少设 4GB |
| **所有场景用同一套参数** | 在线服务用了 batch 处理的高 delay-factor | 按场景选择 |

---

## 4.3 Benchmark 测试方法论

### 使用 vLLM 内置 Benchmark 工具

vLLM 提供了两个内置 benchmark 工具：

#### Throughput Benchmark（测量 tokens/sec）

```bash
# 基础吞吐量测试
python -m vllm.benchmark.benchmark_throughput \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 1000 \
    --output-json throughput_result.json

# 参数说明:
# --num-prompts: 发送的请求数量
# --output-json: 结果保存为 JSON 方便后续分析
# 默认使用 ShareGPT 数据集格式的 prompt
```

#### Serving Latency Benchmark（测量 TTFT + TPOT）

```bash
# 服务端延迟测试（模拟真实客户端行为）
python -m vllm.benchmark.benchmark_serving \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 1000 \
    --request-rate 10 \
    --output-json latency_result.json

# 参数说明:
# --request-rate: 每秒发送的请求数量 (QPS)
# 会测量:
#   - TTFT_P50/P95/P99 (首个 Token 时间)
#   - TPOT_P50/P95/P99 (每个 Token 间隔)
#   - 总吞吐 (tokens/s)
#   - 成功率和错误分布
```

### 自定义 Benchmark 脚本

对于更精细的控制，你可以写自己的 benchmark 脚本：

```python
"""
vLLM 自定义 Benchmark 框架
支持 A/B 测试不同配置的性能对比
"""

import time
import json
import statistics
import asyncio
from openai import AsyncOpenAI
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class BenchResult:
    name: str
    config: dict
    samples: List[float] = field(default_factory=list)
    
    @property
    def p50(self) -> float:
        return statistics.median(self.samples) if self.samples else 0
    
    @property
    def p95(self) -> float:
        s = sorted(self.samples)
        return s[int(len(s) * 0.95)] if s else 0
    
    @property
    def p99(self) -> float:
        s = sorted(self.samples)
        return s[int(len(s) * 0.99)] if s else 0
    
    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0
    
    @property
    def throughput(self) -> float:
        return len(self.samples) / (sum(self.samples) / self.mean) if self.mean > 0 else 0
    
    def to_dict(self):
        d = asdict(self)
        d.update({
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "mean_ms": round(self.mean, 2),
            "throughput": round(self.throughput, 2),
            "num_samples": len(self.samples),
        })
        return d


async def run_benchmark(
    base_url: str,
    prompts: List[str],
    config_name: str,
    config: dict,
    num_samples: int = None,
) -> BenchResult:
    """运行一组 benchmark 测试"""
    
    client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="dummy")
    results = []
    
    actual_samples = num_samples or len(prompts)
    
    for i in range(actual_samples):
        prompt = prompts[i % len(prompts)]
        
        start = time.perf_counter()
        
        try:
            stream = await client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                **config,
            )
            
            # 计算 TTFT（首个 token 到达时间）
            _ = stream.choices[0].message.content
            
            ttft = (time.perf_counter() - start) * 1000
            results.append(ttft)
            
        except Exception as e:
            results.append(-1)  # 错误记为 -1
        
        if i and i % 50 == 0:
            print(f"  [{config_name}] 进度: {i}/{actual_samples} "
                  f"(当前 P50: {statistics.median(results):.1f}ms)")
    
    return BenchResult(
        name=config_name,
        config=config,
        samples=[r for r in results if r >= 0],
    )


async def compare_configs():
    """对比两种不同配置的性能"""
    
    base_url = "http://localhost:8000"
    
    # 测试 Prompt 列表
    prompts = [
        "解释什么是 RESTful API",
        "用 Python 写一个快速排序",
        "量子计算的核心原理",
        "比较 TCP 和 UDP 的区别",
        "什么是微服务架构",
        "Docker 和虚拟机的区别",
        "Kubernetes 的核心概念",
        "SQL 注入攻击的防御方法",
        "Git rebase vs merge",
    ] * 20  # 共 200 个 prompt
    
    configs = {
        "保守配置 (low-delay)": {
            "temperature": 0.7,
            "max_tokens": 150,
        },
        "平衡配置 (balanced)": {
            "temperature": 0.7,
            "max_tokens": 300,
        },
        "激进配置 (high-throughput)": {
            "temperature": 0.7,
            "max_tokens": 512,
        },
    }
    
    all_results = []
    
    for name, cfg in configs.items():
        result = await run_benchmark(base_url, prompts, name, cfg)
        all_results.append(result)
        print(f"\n✅ [{name}] 完成:")
        print(f"   P50: {result.p50:.1f}ms | "
              f"P95: {result.p95:.1f}ms | "
              f"P99: {result.p99:.1f}ms | "
              f"Mean: {result.mean:.1f}ms")
    
    # 打印对比表
    print("\n" + "=" * 75)
    print(f"{'配置':<25} {'P50(ms)':>10} {'P95(ms)':>11} {'P99(ms)':>11} "
          f"{'吞吐':>8}")
    print("-" * 75)
    
    for r in all_results:
        marker = "★" if r == min(all_results, key=lambda x: x.p50) else " "
        print(f"{marker}{r.name:<24} {r.p50:>9.1f}s "
              f"{r.p95:>10.1f}s {r.p99:>10.1f}s "
              f"{r.throughput:>7.1f}")
    
    # 最佳 vs 最差
    best = min(all_results, key=lambda x: x.p50)
    worst = max(all_results, key=lambda x: x.p50)
    improvement = worst.p50 / max(best.p50, 0.001)
    
    print(f"\n🏆 最佳配置: {best.name} (P50={best.p50:.1f}ms)")
    print(f"⚠️ 最差配置: {worst.name} (P50={worst.p50:.1f}ms)")
    print(f"📈 性能差距: {improvement:.1f}x")


if __name__ == "__main__":
    asyncio.run(compare_configs())
```

---

## 4.4 完整调优 Checklist

最后，这是一个在生产环境上线前应该执行的完整检查清单：

```bash
#!/bin/bash
# vllm_tuning_checklist.sh — 上线前调优检查清单

echo "=========================================="
echo "   vLLM 调参上线检查清单"
echo "=========================================="

echo ""
echo "[1/8] 场景确认"
echo "  你的部署场景是？"
echo "    ① 在线聊天/实时交互"
echo "    ② 通用 API 服务"
echo "    ③ 离线批处理"
echo "    ④ 混合负载"
echo ""

echo "[2/8] 关键参数检查"
echo "  --max-num-seqs:     [建议: 聊天 64, API 128, 批量 256]"
echo "  --max-model-len:    [聊天 4096, API 8192/16384, 批量 16384]"
echo "  --scheduler-delay:  [聊天 0.05, 通用 0.1, 批量 0.5]"
echo "  --gpu-mem-util:    [通用 0.90-0.92, 极限 0.95]"
echo "  --swap-space:       [无抢占 0, 标准 4, 重度 8-16]"
echo ""

echo "[3/8] Prefix Caching"
echo "  是否有大量请求共享相同的 System Prompt?"
echo "  → 是: 确保 enable-prefix-caching 已生效（默认自动）"
echo "  → 否: 此项影响不大"
echo ""

echo "[4/8] 监控就绪"
echo "  Prometheus + Grafana 已部署? [yes/no]"
echo "  关键 Dashboard 已创建? [yes/no]"
echo "  告警规则已配置? [yes/no]"
echo "  /metrics 端点可访问? [yes/no]"
echo ""

echo "[5/8] 压力测试结果"
echo "  已运行 benchmark_throughput? [yes/no]"
echo "  已运行 benchmark_serving? [yes/no]"
echo "  TTFT P50 < 目标值? [yes/no]"
echo "  P99 延迟 < 目标值? [yes/no]"
echo "  吞吐量 > 目标值? [yes/no]"
echo ""

echo "[6/8] 异常预案"
echo "  OOM 时的降级方案? [yes/no]"
echo "  高延迟时的告警规则? [yes/no]"
echo "  故障自动重启? [yes/no]"
echo ""

echo "[7/8] 文档记录"
echo "  最终配置已写入文档? [yes/no]"
echo "  调优依据已记录? [yes/no]"
echo ""

echo "[8/8] 签字确认"
echo "  以上全部 OK? 开始部署! 🚀"
echo "=========================================="
```

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **四大场景配置** | **在线聊天**（delay=0.05, seqs=64, len=4096）→ **API 服务**（delay=0.1, seqs=128, len=8192, swap=4）→ **批处理**（delay=0.5, seqs=256, len=16K, util=0.95）→ **混合**（delay=0.15, seqs=128, swap=8） |
| **决策树** | TTFT 高 → 降 delay；利用率低 → 升 num-seqs 或查 model-len；错误率高 → 查 OOM/swap |
| **反模式** | 全场景同一套参数 / max-len 过大 / delay=0 / swap=0 / seqs 过大 |
| **Benchmark** | `benchmark_throughput`（tokens/s）+ `benchmark_serving`（TTFT/TPOT/吞吐）；自定义脚本支持 A/B 对比 |
| **Swap 权衡** | 抢占代价与序列长度线性增长（512tok≈10ms, 8192tok≈112ms）；短请求抢占几乎免费 |
| **Prefix Caching** | 混合负载场景下收益最大（共享 System Prompt 的请求可能占 30%+） |

> **一句话总结**：调度参数没有"万能最优值"，只有"最适合你的场景的值"。核心原则是：**先明确场景特征（请求长度分布、并发级别、延迟要求），再对照场景速查表选择起始配置，最后用 Benchmark 数据做最终验证**。记住 vLLM 调优是一个迭代过程——先用默认值跑一轮 benchmark，根据数据定位瓶颈，针对性调整参数，再跑一轮验证，直到指标满足预期。
