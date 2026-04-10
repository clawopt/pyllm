# 08-1 硬件层面的优化

## 性能瓶颈在哪里

在开始优化之前，你需要先搞清楚一个问题：**你的 Ollama 到底慢在哪里？** 是模型加载慢？还是 token 生成速度慢？还是网络延迟？不同的瓶颈需要完全不同的优化策略。

```
┌─────────────────────────────────────────────────────────────┐
│              Ollama 请求生命周期与耗时分解                     │
│                                                             │
│  用户发起请求                                               │
│      │                                                      │
│      ▼ (通常 <1ms)                                         │
│  ┌─────────────┐                                          │
│  │ HTTP 接收    │  解析 JSON / 验证参数                      │
│  └──────┬──────┘                                          │
│         │                                                    │
│         ▼ (首次加载: 5-30s, 已加载: <100ms)                  │
│  ┌─────────────┐                                          │
│  │ 模型加载     │  从磁盘读取 GGUF → mmap 到内存              │
│  │             │  (如果模型不在内存中)                        │
│  └──────┬──────┘                                          │
│         │                                                    │
│         ▼ (取决于 prompt 长度和硬件)                         │
│  ┌─────────────┐                                          │
│  │ Prompt 处理  │  Tokenize → Embedding lookup →            │
│  │             │  KV Cache 写入                              │
│  └──────┬──────┘                                          │
│         │                                                    │
│         ▼ (主要耗时区域，占总时间 80-95%)                    │
│  ┌─────────────┐                                          │
│  │ Token 生成   │  每个 token 的自回归推理循环               │
│  │             │  → 这是性能优化的主战场 ←←←←                │
│  └──────┬──────┘                                          │
│         │                                                    │
│         ▼ (<1ms)                                            │
│  ┌─────────────┐                                          │
│  │ 响应返回     │  序列化为 SSE 流 / JSON                   │
│  └─────────────┘                                          │
│                                                             │
│  ⚡ 关键洞察: Token 生成阶段占据了绝大部分时间              │
│     而这一阶段的性能 = f(硬件算力 + 模型大小 + 参数配置)     │
└─────────────────────────────────────────────────────────────┘
```

## CPU 模式优化

### AVX 指令集利用

Ollama 底层的 llama.cpp 在编译时会自动检测并使用 CPU 的 SIMD 指令集。确保你的编译版本启用了所有可用的指令：

```bash
# 检查当前 CPU 支持的指令集
sysctl -a | grep -i "cpu.*features"   # macOS
cat /proc/cpuinfo | grep flags        # Linux

# macOS Apple Silicon:
# → 自动支持 NEON/AMX 加速，无需额外配置

# Linux x86_64:
# → 确保 llama.cpp 编译时启用 AVX2 / AVX-512
# → Ollama 官方发布版已包含这些优化
```

### 多核并行

Ollama 通过 `OLLAMA_NUM_PARALLEL` 控制并行度——但这不是简单地说"用更多核心就更快"。理解并行的工作方式很重要：

```python
#!/usr/bin/env python3
"""并行度对性能的影响分析"""

import requests
import time
import concurrent.futures


def benchmark_parallelism(parallel_count: int, 
                           model: str = "qwen2.5:1.5b",
                           num_requests: int = 10):
    """测试不同并行度下的吞吐量"""
    
    url = "http://localhost:11434/api/chat"
    
    def single_request(i):
        start = time.time()
        resp = requests.post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": f"请数到{i}"}],
            "stream": False,
            "options": {"num_predict": 50}
        }, timeout=30)
        
        elapsed = time.time() - start
        tokens = resp.json().get("eval_count", 0)
        return {"id": i, "time": elapsed, "tokens": tokens}
    
    # 设置环境变量（通过 API 无法直接设置，
    # 这里模拟并发请求来观察行为）
    print(f"\n{'='*60}")
    print(f"  并发测试: {num_requests} 个请求")
    print(f"  模型: {model}")
    print(f"{'='*60}")
    
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=parallel_count
    ) as executor:
        results = list(executor.map(single_request, range(num_requests)))
    
    total_time = time.time() - start_total
    
    total_tokens = sum(r["tokens"] for r in results)
    avg_latency = sum(r["time"] for r in results) / len(results)
    max_latency = max(r["time"] for r in results)
    throughput = total_tokens / total_time if total_time > 0 else 0
    
    print(f"  并行度: {parallel_count}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均延迟: {avg_latency:.2f}s")
    print(f"  最大延迟: {max_latency:.2f}s")
    print(f"  吞吐量: {throughput:.1f} tokens/s")
    print(f"  总 tokens: {total_tokens}")


if __name__ == "__main__":
    for p in [1, 2, 4]:
        benchmark_parallelism(p)
```

### 内存带宽：被忽视的真正瓶颈

很多人以为 CPU 频率决定推理速度，但实际上对于大模型推理，**内存带宽往往是更大的瓶颈**：

```
为什么内存带宽这么重要？

一次前向传播中，模型需要对每个 token：
1. 读取权重矩阵 W (d × d) → 内存读取: d² × 2 bytes
2. 读取 KV Cache (2 × seq_len × d) → 内存读取: seq_len × d × 4 bytes  
3. 计算 attention 和 FFN
4. 写回新的 KV Cache

对于一个 7B (d=4096) 模型的单个层：
- 权重读取: 4096 × 4096 × 2B = 32 MB (!!)
- 如果有 28 层，每步 token 要读 ~900 MB

结论: 内存带宽直接决定了每秒能处理多少 token

DDR4-2400: ~19 GB/s → 约 20 tok/s (7B 模型)
DDR5-4800: ~38 GB/s → 约 40 tok/s (7B 模型)
Apple M2 Unified Memory: ~200 GB/s → 约 200+ tok/s (7B 模型)

💡 这就是为什么 Apple Silicon 跑大模型比同频 x86 快得多！
```

## GPU 模式优化

### NVIDIA GPU

```bash
# 1. 确认 CUDA 可用
nvidia-smi

# 2. 检查 Ollama 是否检测到 GPU
ollama ps
# 应该能看到 GPU 信息和显存占用

# 3. 关键环境变量
export CUDA_VISIBLE_DEVICES=0          # 指定使用哪块 GPU
export OLLAMA_GPU_LAYERS=99              # 将尽可能多的层放到 GPU
                                        # (默认自动选择最优值)
```

### GPU 层分配策略

`OLLAMA_GPU_LAYERS` 是最强大的 GPU 优化参数——它让你精确控制哪些模型层运行在 GPU 上：

```python
#!/usr/bin/env python3
"""GPU 层分配策略分析"""

def analyze_gpu_layer_strategy(model_params="7B", gpu_vram_gb=8):
    """
    分析不同 GPU 层数量的效果
    
    Llama/Qwen 架构的近似内存需求:
    - 每个 Transformer 层的权重 ≈ params * 2 / layers bytes (fp16)
    - 7B 模型约 32 层
    - 每层权重 ≈ 7e9 * 2 / 32 ≈ 437 MB (fp16)
    - KV Cache 每层 ≈ 2 * d * seq_len * 2 bytes
    """
    
    import math
    
    # 近似层数（不同模型架构略有差异）
    layer_map = {
        "1B": 24, "3B": 28, "7B": 32, "14B": 40,
        "32B": 64, "70B": 80
    }
    
    n_layers = layer_map.get(model_params, 32)
    bytes_per_param = 2  # fp16
    
    total_params = float(model_params.replace("B", "")) * 1e9
    params_per_layer = total_params / n_layers
    weight_per_layer_mb = (params_per_layer * bytes_per_param) / (1024**2)
    
    print(f"\n{'='*65}")
    print(f"  模型: {model_params} ({n_layers} 层)")
    print(f"  GPU 显存: {gpu_vram_gb} GB")
    print(f"  每层权重: ~{weight_per_layer_mb:.0f} MB (fp16)")
    print(f"{'='*65}\n")
    
    print(f"{'GPU 层数':>10s} {'GPU 显存':>10s} {'CPU 卸载':>12s} "
          f"{'速度预估':>12s} {'适用场景'}")
    print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*25}")
    
    for gpu_layers in range(0, min(n_layers + 1, n_layers)):
        gpu_mem = gpu_layers * weight_per_layer_mb / 1024  # GB
        
        if gpu_mem > gpu_vram_gb * 0.9:  # 留 10% 给 KV Cache
            break
        
        cpu_layers = n_layers - gpu_layers
        gpu_ratio = gpu_layers / n_layers
        
        # 速度估算（简化公式）
        speedup_estimate = 1 + gpu_ratio * 4  # GPU 比 CPU 快约 5x
        speed_str = f"{speedup_estimate:.1f}x (相对纯CPU)"
        
        if gpu_layers == 0:
            scenario = "纯 CPU 模式"
        elif gpu_layers <= n_layers * 0.3:
            scenario = "部分加速"
        elif gpu_layers <= n_layers * 0.7:
            scenario = "⭐ 推荐"
        elif gpu_layers < n_layers:
            scenario = "接近全GPU"
        else:
            scenario = "全 GPU (需更大显存)"
        
        print(f"{gpu_layers:>10d} {gpu_mem:>9.1f}GB "
              f"{cpu_layers:>12d} {speed_str:>12s} {scenario}")


if __name__ == "__main__":
    # 不同显卡配置
    configs = [
        ("7B", 6),    # RTX 3060 12G / RTX 4060 8G
        ("7B", 8),    # RTX 3090 / RTX 4070 Ti
        ("7B", 16),   # RTX 4090 / A5000
        ("7B", 24),   # A6000 / RTX 6000 Ada
        ("14B", 16),  # RTX 4090
        ("14B", 24),  # A6000
        ("32B", 24),  # 双卡或高端卡
        ("70B", 48),  # A100 80G 或多卡
    ]
    
    for params, vram in configs:
        analyze_gpu_layer_strategy(params, vram)
        print()
```

### Apple Silicon 专项优化

Apple Silicon（M1/M2/M3/M4）的统一内存架构是跑大模型的理想平台：

```bash
# Apple Silicon 特有的优化选项

# 1. Metal/MPS 加速（默认已启用）
# Ollama 在 Mac 上自动使用 Metal Performance Shaders

# 2. 统一内存优势
# 不需要像 NVIDIA 那样做 GPU↔CPU 数据拷贝
# 所有内存都可以被 GPU 直接访问

# 3. 查看实际使用的后端
ollama ps
# 应该显示 "Metal" 作为计算后端

# 4. M系列芯片的内存带宽参考:
# M1: ~68 GB/s
# M1 Pro/Max: ~200-400 GB/s
# M2: ~100 GB/s (base), ~200 GB/s (Pro/Max)
# M3: ~100-400 GB/s (取决于型号)
```

```python
#!/usr/bin/env python3
"""Apple Silicon 各型号跑大模型的性能参考"""

APPLE_SILICON_DATA = {
    "M1": {
        "unified_memory_max": 16,  # GB
        "memory_bandwidth": 68,       # GB/s
        "neural_engine": True,
        "approx_toks_per_sec_7b": 35,  # q4_K_M, 估算值
    },
    "M1 Pro": {
        "unified_memory_max": 32,
        "memory_bandwidth": 200,
        "neural_engine": True,
        "approx_toks_per_sec_7b": 90,
    },
    "M1 Max/Ultra": {
        "unified_memory_max": 64/128,
        "memory_bandwidth": 400,
        "neural_engine": True,
        "approx_toks_per_sec_7b": 150,
    },
    "M2": {
        "unified_memory_max": 24,
        "memory_bandwidth": 100,
        "neural_engine": True,
        "approx_toks_per_sec_7b": 55,
    },
    "M2 Pro/Max": {
        "unified_memory_max": 36/96/128,
        "memory_bandwidth": 200/400,
        "neural_engine": True,
        "approx_toks_per_sec_7b": 120/180,
    },
    "M3": {
        "unified_memory_max": 24/36/48,
        "memory_bandwidth": 150/300,
        "neural_engine": True,
        "approx_toks_per_sec_7b": 80/140,
    },
}


def apple_silicon_guide():
    """Apple Silicon 选型指南"""
    
    print("\n" + "=" * 70)
    print("  🍎 Apple Silicon 大模型运行指南")
    print("=" * 70)
    
    models = [
        ("phi3:mini", 2.3, "极小模型"),
        ("qwen2.5:3b", 1.8, "轻量中文"),
        ("qwen2.5:7b", 4.3, "通用甜点"),
        ("llama3.1:8b", 4.7, "英文全能"),
        ("qwen2.5:14b", 8.6, "高质量"),
        ("llava/minicpm-v", 4.5, "视觉模型"),
        ("qwen2.5:32b", 19.0, "企业级"),
        ("llama3.1:70b", 40.0, "旗舰级"),
    ]
    
    print(f"\n{'芯片':<15s} {'内存':>6s} | ", end="")
    for name, size, _ in models:
        print(f"{name:<18s}", end="")
    print()
    print("-" * (15 + 8 + len(models) * 19))
    
    for chip, info in APPLE_SILICON_DATA.items():
        max_ram = info["unified_memory_max"]
        bw = info["bandwidth"]
        
        ram_str = f"{max_ram}GB" if isinstance(max_ram, int) else \
                 f"{max_ram[0]}-{max_ram[-1]}GB"
        
        print(f"{chip:<15s} {ram_str:>6s} | ", end="")
        
        for model_name, size_gb, _ in models:
            needed = size_gb + 2  # +2GB 给系统和 KV Cache
            max_r = max_ram if isinstance(max_ram, int) else max_ram[-1]
            
            if needed <= max_r:
                toks = info["approx_toks_per_sec_7b"]
                if model_name.endswith("7b") or "8b" in model_name:
                    status = f"✅ ~{toks} t/s"
                elif model_name.endswith("32b"):
                    status = f"✅ ~{toks//2} t/s"
                elif model_name.endswith("70b"):
                    status = "⚠️ 慢但可用" if max_r >= 48 else "❌ 不够"
                else:
                    status = "✅ 可运行"
            else:
                status = "❌ 内存不足"
            
            print(f"{status:<18s}", end="")
        print()


if __name__ == "__main__":
    apple_silicon_guide()
```

## 各模型内存需求精确表

| 模型 | 参数量 | FP16 需要 | Q4_K_M 需要 | Q4_K_M + KV(4K) | Q4_K_M + KV(32K) |
|------|--------|----------|------------|-------------------|--------------------|
| phi3:mini | 3.8B | 7.6 GB | **1.9 GB** | ~2.5 GB | ~4 GB |
| tinyllama | 1.1B | 2.2 GB | **0.55 GB** | ~1 GB | ~2 GB |
| gemma2:2b | 2B | 4 GB | **1.0 GB** | ~1.5 GB | ~3 GB |
| qwen2.5:3b | 3B | 6 GB | **1.5 GB** | ~2 GB | ~4 GB |
| qwen2.5:7b | 7B | 14 GB | **3.5 GB** | ~4.5 GB | ~8 GB |
| llama3.1:8b | 8B | 16 GB | **4.0 GB** | ~5 GB | ~9 GB |
| deepseek-coder:6.7b | 6.7B | 13.4 GB | **3.4 GB** | ~4 GB | ~8 GB |
| mistral:7b | 7B | 14 GB | **3.5 GB** | ~4.5 GB | ~9 GB |
| qwen2.5:14b | 14B | 28 GB | **7.0 GB** | ~8.5 GB | ~16 GB |
| qwen2.5:32b | 32B | 64 GB | **16 GB** | ~19 GB | ~35 GB |
| llama3.1:70b | 70B | 140 GB | **35 GB** | ~40 GB | ~75 GB |

**经验法则**: `最小内存 = 模型 Q4 大小 × 1.2`（给 KV Cache 和系统留余量）

## 本章小结

这一节从硬件层面全面分析了 Ollama 的性能优化空间：

1. **Token 生成阶段占据 80-95% 的总耗时**——这是优化的主战场
2. **CPU 模式的关键瓶颈是内存带宽**而非 CPU 频率——DDR5 > DDR4，Apple 统一内存 >> 传统 NUMA 架构
3. **`OLLAMA_NUM_PARALLEL`** 控制并行度，但受限于物理核心数和内存带宽
4. **NVIDIA GPU 通过 `OLLAMA_GPU_LAYERS`** 精确控制层分配，不是越多越好——需要在 GPU 显存和 CPU 卸载之间找到平衡点
5. **Apple Silicon 是本地部署的理想平台**——统一内存消除了数据拷贝开销，M2 Pro/Max 以上可以流畅运行 7B 甚至 14B 模型
6. **内存需求的 1.2x 法则**可以快速判断你的硬件能否跑某个模型

下一节我们将学习推理参数层面的调优。
