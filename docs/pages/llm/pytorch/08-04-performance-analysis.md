# 8.4 推理性能分析与优化 Checklist

前三节我们学习了三种不同层次的推理优化技术：图级优化（torch.compile 的算子融合与 Python 开销消除）、模型级量化（INT4/FP8 精度压缩）、以及格式转换（ONNX/GGUF/SafeTensors 导出）。这些工具都很有力，但面对一个具体的性能问题时——比如"我的模型推理延迟是 800ms，需要降到 200ms 以内"——你需要一个系统化的方法来定位瓶颈、选择正确的优化策略并验证效果。这一节就是你的"推理性能诊断手册"：从测量方法到瓶颈识别，再到完整的优化 Checklist。

## 性能测量的正确姿势

在开始任何优化之前，你必须先建立准确的基线测量。很多初学者犯的第一个错误就是用 `time.time()` 来测量 GPU 操作的耗时——这在 CPU 上勉强可用，但在 GPU 上是完全错误的，因为 GPU 操作是异步的：调用 CUDA kernel 后控制权立即返回给 Python，实际的计算还在 GPU 上异步执行。如果你用 `time.time()` 测量，你测量的只是"发起操作的时间"，而不是"操作完成的时间"。

### 正确的 GPU 计时方法

```python
import torch
import time


def correct_gpu_timing():
    model = GPT(GPTConfig(vocab_size=1000, n_embed=256, num_heads=4, num_layers=4)).cuda().eval()
    x = torch.randint(0, 1000, (1, 256)).cuda()
    
    print("=== GPU Timing Methods Comparison ===\n")
    
    # ❌ 错误方式: time.time() — 只测量发起时间
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
    wrong_time = (time.time() - start) / 50 * 1000
    
    # ✅ 正确方式: torch.cuda.synchronize() + time.perf_counter()
    with torch.no_grad():
        for _ in range(10):  # warmup
            _ = model(x)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(200):
            _ = model(x)
    torch.cuda.synchronize()
    correct_time = (time.perf_counter() - start) / 200 * 1000
    
    # ✅ 最佳方式: torch.profiler (最详细)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                     torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
    
    print(f"[WRONG] time.time():     {wrong_time:>8.2f} ms/call")
    print(f"[OK]    synchronize:     {correct_time:>8.2f} ms/call")
    
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    print(f"\n[PROFILER] Top 10 CUDA kernels by time:")
    print(table)


correct_gpu_timing()

# 典型输出:
# === GPU Timing Methods Comparison ===
#
# [WRONG] time.time():         2.34 ms/call  ← 偏小！因为没等GPU完成
# [OK]    synchronize:       12.56 ms/call  ← 正确值
#
# [PROFILER] Top 10 CUDA kernels by time:
# -------------------------------------------------------
# Name                                          Self CPU   Total CPU  ...
# aten::matmul                                  0.120ms    0.120ms  ...
# aten::linear                                   0.080ms    0.350ms  ...
# aten::scaled_dot_product_attention             0.010ms    3.200ms  ...
# ...（详细的 kernel 级别耗时分解）
```

注意看错误方式和正确方式的差距：`time.time()` 报告的是 2.34ms，而实际 GPU 执行时间是 12.56ms —— 差了 5 倍多！如果基于错误的测量数据来做优化决策，你可能会把精力花在完全不需要优化的地方上。

### 完整的推理基准测试框架

下面是一个生产级的基准测试函数，它不仅测量总延迟，还分解了各个阶段的耗时：

```python
@torch.no_grad()
def benchmark_inference(model, tokenizer, prompts, max_new_tokens=128,
                         warmup=5, repeats=20):
    """全面的推理性能基准测试"""
    import json
    
    model.eval()
    results = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_len = input_ids.shape[1]
        
        # Warmup
        for _ in range(warmup):
            _ = model.generate(
                input_ids,
                max_new_tokens=min(16, max_new_tokens),
                do_sample=False,
            )
        
        torch.cuda.synchronize()
        
        # Prefill phase timing
        times_prefill = []
        times_decode = []
        total_tokens = []
        
        for _ in range(repeats):
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            torch.cuda.synchronize()
            t_total = time.perf_counter() - t_start
            
            generated_len = output_ids.shape[1] - input_len
            
            times_prefill.append(None)  # 难以单独分离 prefill 时间
            times_decode.append(t_total)
            total_tokens.append(generated_len)
        
        avg_time_ms = sum(times_decode) / len(times_decode) * 1000
        avg_tokens = sum(total_tokens) / len(total_tokens)
        tokens_per_sec = avg_tokens / (avg_time_ms / 1000)
        
        result = {
            'prompt': prompt[:50] + ('...' if len(prompt) > 50 else ''),
            'input_length': input_len,
            'output_length': avg_tokens,
            'total_latency_ms': avg_time_ms,
            'tokens_per_second': tokens_per_sec,
            'latency_per_token_ms': avg_time_ms / avg_tokens if avg_tokens > 0 else float('inf'),
        }
        results.append(result)
        
        print(f"Prompt: {result['prompt']}")
        print(f"  Input: {result['input_length']} tokens → "
              f"Output: {result['output_length']:.0f} tokens")
        print(f"  Latency: {result['total_latency_ms']:.0f}ms | "
              f"Throughput: {result['tokens_per_second']:.0f} tok/s | "
              f"Per-token: {result['latency_per_token_ms']:.1f} ms/tok")
    
    return results


prompts = [
    "请用三句话解释什么是深度学习。",
    "写一首关于春天的诗。",
    "Python和JavaScript有什么区别？",
]

benchmark_inference(your_model, your_tokenizer, prompts, max_new_tokens=64)
```

## 瓶颈分析：Time Breakdown

LLM 推理的时间消耗主要分布在两个阶段：

**Prefill（预填充）阶段**：处理输入 prompt。对于长度为 T_in 的输入，模型需要做一次完整的前向传播，计算所有 T_in 个位置的 logits。这个阶段的特点是：
- **Compute-bound**（计算密集）：需要大量矩阵乘法
- 并行度高（所有位置可以同时计算）
- 总耗时约 O(T_in × d²)（d 是隐藏维度）

**Decode（解码）阶段**：逐 token 生成输出。每一步只生成一个新 token，但为了计算这个新 token 的概率分布，仍然需要对整个当前序列（包括之前生成的所有 token）做一次前向传播（因为有因果 Attention 的存在）。这个阶段的特点是：
- **Memory-bandwidth bound**（内存带宽受限）：每步的主要开销是从 HBM 读取权重
- 无法并行（必须串行生成每个 token）
- 总耗时约 O(T_out × T_avg × d)，其中 T_avg 是平均序列长度

对于一个典型的 7B 模型推理请求：

```
Time breakdown (7B BF16, A100, seq_len=2048):

Prefill (input=512 tokens):
├── Embedding lookup:          ~0.5 ms
├── Transformer blocks (×32): ~45 ms   ← 主要瓶颈
│   ├── Attention (~60%):      ~27 ms
│   └── FFN (~40%):           ~18 ms
├── LM Head projection:        ~0.3 ms
└── Total prefill:             ~46 ms

Decode (generate 256 tokens, one at a time):
├── Per-step overhead:         ~0.05 ms
├── Per-step forward pass:     ~6.2 ms   ← 每步都要做！
├── Sampling (top-k/top-p):    ~0.01 ms
└── Total decode:              ~1589 ms  (256 × 6.2)

─────────────────────────────────────
Total latency:                  ~1635 ms
Prefill: 2.8% | Decode: 97.2%
```

从这个分解中可以清楚地看到：**Decode 阶段占据了 97% 以上的总时间**。这就是为什么 KV Cache（下一节会详细讲）和各种 decoding 优化（Speculative Decoding、Continuous Batching）如此重要——它们针对的正是这个最大的瓶颈。

## 系统化优化 Checklist

基于以上分析，以下是按优先级排序的 LLM 推理优化 Checklist：

### 第一优先级：基础优化（投入产出比最高）

- [ ] **启用 `torch.compile(mode="reduce-overhead")`**
  - 预期收益：10~30% 加速
  - 成本：一行代码
  - 注意：首次调用有 warmup 开销

- [ ] **使用 BF16 而非 FP32**
  - 预期收益：~2× 加速（Tensor Core 利用率提升）
  - 成本：改一行 dtype 参数
  - 条件：Ampere+ GPU

- [ ] **使用 SDPA 替代手动 Attention**
  - 预期收益：20~40% Attention 层加速（FlashAttention）
  - 成本：替换几行代码
  - 条件：Ampere+ GPU + PyTorch 2.2+

- [ ] **启用 KV Cache**
  - 预期收益：Decode 阶段减少重复计算
  - 成本：修改 generate 函数或使用 HuggingFace 的 `use_cache=True`
  - 这是 Decode 优化的基础

### 第二优先级：中级优化

- [ ] **INT4 权重量化（GPTQ/AWQ/torchao）**
  - 预期收益：体积减小 75%+，推理加速 20~50%（memory-bound 场景）
  - 成本：一次性离线量化过程
  - 注意：质量损失通常 < 2%

- [ ] **Batching / Continuous Batching**
  - 预期收益：吞吐量提升 3~10×（通过提高 GPU 利用率）
  - 成本：需要实现请求队列和批处理逻辑
  - 适用场景：高并发服务

- [ ] **FlashAttention-2 / FlashDecoding**
  - 预期收益：Attention 计算 2~4× 加速
  - 成本：安装 flash-attn 库
  - 条件：Ampere+ GPU

- [ ] **模型导出为最优格式**
  - Web 服务 → TorchScript / ONNX
  - 本地部署 → GGUF (Q4_K_M) + Ollama
  - 移动端 → CoreML / TFLite

### 第三优先级：高级优化

- [ ] **Speculative Decoding（推测性解码）**
  - 预期收益：Decode 阶段 2~3× 加速
  - 成本：需要一个小的 draft model
  - 原理：用小模型快速预测多个 token，大模型并行验证

- [ ] **PagedAttention (vLLM)**
  - 预期收益：解决 KV Cache 内存碎片问题，提升并发能力
  - 成本：使用 vLLM 或 SGLang 框架
  - 适用场景：高并发、变长序列的服务

- [ ] **FP8 推理（Hopper GPU）**
  - 预期收益：比 BF16 再快 2~3×
  - 成本：需要 H100/H800 硬件
  - 条件：Blackwell/Hopper 架构

- [ ] **TensorRT / TensorRT-LLM**
  - 预期收益：终极推理性能（接近硬件极限）
  - 成本：较高的工程集成成本
  - 适用场景：对延迟有极致要求的商业服务

## 性能回归测试

每次优化后都应该运行一次完整的回归测试，确保：
1. 输出质量没有显著下降（perplexity 对比、人工评估抽样）
2. 不同输入长度的行为一致（短文本/长文本/边界情况）
3. 在目标硬件上的表现符合预期

```python
def regression_test(model_orig, model_optimized, test_cases):
    """优化前后的回归对比"""
    model_orig.eval()
    model_optimized.eval()
    
    print("\n=== Optimization Regression Test ===")
    print(f"{'Case':>15s} | {'Orig PPL':>10s} | {'Opt PPL':>10s} | "
          f"{'Orig Time':>11s} | {'Opt Time':>10s} | {'Speedup':>8s}")
    print("-" * 80)
    
    for name, text in test_cases:
        tokens = tokenizer.encode(text, truncation=True, max_length=512)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_orig = model_orig(input_ids)
            torch.cuda.synchronize()
            t_orig = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            out_opt = model_optimized(input_ids)
            torch.cuda.synchronize()
            t_opt = time.perf_counter() - t0
        
        ppl_orig = compute_ppl(out_orig['logits'], input_ids)
        ppl_opt = compute_ppl(out_opt['logits'], input_ids)
        
        speedup = t_orig / t_opt
        
        print(f"{name:>15s} | {ppl_orig:>10.4f} | {ppl_opt:>10.4f} | "
              f"{t_orig*1000:>10.1f}ms | {t_opt*1000:>9.1f}ms | {speedup:>7.2f}×")


test_cases = [
    ("Short", "你好"),
    ("Medium", "请解释一下量子计算的基本原理"),
    ("Long", "写一篇关于人工智能发展历史的文章，不少于500字"),
]

regression_test(original_model, optimized_model, test_cases)
```

## 第 8 章小结

到这里，第 8 章"推理优化"的全部四个小节就完成了。我们从 `torch.compile()` 的图编译原理出发，理解了 Dynamo 字节码追踪和 Inductor kernel 生成的两阶段流水线；然后深入了模型量化的世界，比较了动态量化、静态量化、INT4 权重量化、FP8 量化等多种方案的质量和速度权衡；接着学习了 ONNX（跨平台标准）、GGUF（本地推理之王）、SafeTensors（安全存储格式）三种主流导出格式及其适用场景；最后在本节建立了完整的性能测量方法论、瓶颈分析框架和系统化的优化 Checklist。

回顾整个教程的知识地图：第 1 章建立了 PyTorch 基础认知，第 2 章构建了生产级的数据管道，第 3 章从零手写了 Transformer/GPT 模型，第 4 章实现了训练循环与优化策略，第 5 章用 Lightning 提升了工程效率，第 6 章掌握了 HF Trainer + PEFT 微调链，第 7 章突破了分布式训练的规模限制，第 8 章完成了推理优化的全栈知识。最后一章将把这些知识整合到生产环境中——如何构建可靠的 MLOps 流水线、监控模型健康状态、以及管理模型的完整生命周期。
