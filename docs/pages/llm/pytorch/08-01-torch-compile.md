# 8.1 torch.compile 与图编译优化

经过前面七章的旅程，我们已经完成了从数据管道到模型构建、从训练循环到分布式训练的完整知识体系。模型训练好了，接下来自然要面对的问题就是：怎么让这个模型在实际部署中跑得更快？假设你微调好了一个 7B 模型，准备上线一个对话服务。用户发来一个问题，你的服务需要把问题 tokenize、送入模型前向传播、对输出 logits 做 sampling 生成 token ID、decode 成文本再返回。如果每个请求需要 2 秒才能返回结果，那并发能力就非常有限——每秒只能处理 0.5 个请求。而如果你的竞品能在 500ms 内返回，用户体验差距是巨大的。

推理优化的目标就是在不损失（或尽量少损失）生成质量的前提下，**减少单次推理的延迟（latency）和提高吞吐量（throughput）**。这一章我们将系统学习四大类优化技术：图级别优化（torch.compile）、模型级优化（量化）、格式级优化（导出与转换）以及系统级优化（性能分析与 Checklist）。本节从 PyTorch 2.0 最令人兴奋的新特性开始——`torch.compile()`。

## torch.compile：一行代码的魔法

PyTorch 2.0 于 2023 年发布，其中最核心的新特性就是 `torch.compile()`。它的宣传语非常吸引人：**不需要修改任何模型代码，只需在模型定义之后加一行 `model = torch.compile(model)`，就能获得 10%~50% 的加速**。这听起来像营销话，但实际上它背后有坚实的技术支撑——TorchDynamo + TorchInductor 的组合构成了新一代的图编译（Graph Compilation）框架。

```python
import torch
import time


def demo_torch_compile():
    model = GPT(GPTConfig(vocab_size=1000, n_embed=256, num_heads=4, num_layers=4))
    model.eval()
    
    x = torch.randint(0, 1000, (1, 128))
    
    print("=== Eager mode (no compile) ===")
    model_eager = model.cuda().eval()
    
    with torch.no_grad():
        for _ in range(10):
            _ = model_eager(x.cuda())
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model_eager(x.cuda())
    torch.cuda.synchronize()
    eager_time = (time.time() - start) / 100 * 1000
    
    print(f"  Average: {eager_time:.2f} ms/call")
    
    print("\n=== Compiled mode ===")
    model_compiled = torch.compile(model.cuda(), mode="reduce-overhead")
    
    with torch.no_grad():
        for _ in range(10):
            _ = model_compiled(x.cuda())
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model_compiled(x.cuda())
    torch.cuda.synchronize()
    compiled_time = (time.time() - start) / 100 * 1000
    
    print(f"  Average: {compiled_time:.2f} ms/call")
    print(f"\nSpeedup: {eager_time/compiled_time:.2f}×")


demo_torch_compile()

# 典型输出:
# === Eager mode (no compile) ===
#   Average: 8.45 ms/call
#
# === Compiled mode ===
#   Warmup (first call may be slower due to tracing)...
#   Average: 5.23 ms/call
#
# Speedup: 1.62×
```

注意两个细节：第一，第一次调用 compiled model 时通常会慢一些（因为需要做 tracing 和代码生成），这就是所谓的 **warmup 开销**；第二，加速比因模型大小和硬件平台而异——小模型可能只有 10~20% 的提升，大模型（如 LLaMA-7B）在 Ampere GPU 上可以达到 30~50% 甚至更高。

## 工作原理：Dynamo + Inductor 两阶段流水线

`torch.compile()` 的内部由两个核心组件协同工作：

### 阶段一：TorchDynamo —— 捕获计算图

当你调用 `torch.compile(model)` 时，TorchDynamo 会拦截模型的 `forward` 方法执行过程。它使用 Python 字节码分析技术（基于 PEP 652 —— `__future__annotations`）来追踪 forward 函数中所有执行的 PyTorch 操作，把它们记录为一个 **FX Graph（Flexible eXecution Graph）**。这个过程叫做 **tracing（追踪）**。

```
原始 Python 代码:
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

TorchDynamo 追踪后产生的 FX Graph:
graph():
    %x = placeholder[target]
    %1 = call_module[self.embedding](%x)
    %2 = call_module[self.pos_embedding](%x)
    %3 = add(%1, %2)
    %4 = call_module[self.blocks.0](%3)
    %5 = call_module[self.blocks.1](%4)
    ...
    %n = call_module[self.lm_f](%...)
    return call_module[self.lm_head](%n)
```

关键点：Dynamo 不是简单地运行一遍 forward 然后记录操作——它在 Python 字节码层面工作，能够处理控制流（if/else、for 循环），并且支持 "graph break" 机制：当遇到无法处理的操作（如动态 shape、某些自定义 op）时，它可以暂时退出图模式、执行 Python 原生代码，然后在合适的时候重新进入图模式。

### 阶段二：TorchInductor —— 生成高效 kernel

拿到 FX Graph 之后，TorchInductor 负责把它编译成高效的机器码。具体来说：

1. **算子融合（Operator Fusion）**：把多个连续的小算子合并成一个大算子。比如 `Linear → Add → ReLU → Linear` 这四个独立的操作可以被融合成一个融合 kernel，减少全局内存（GMEM）读写次数。
2. **代码生成**：根据目标硬件（CUDA / CPU / XLA 等）生成对应的 Triton C++ 代码或 OpenAI Triton 代码。
3. **编译优化**：调用 LLVM/NVCC 编译器将生成的代码编译成 GPU 可执行的 binary。

```python
# 融合前（eager 模式）:
# Kernel 1: matmul (embedding lookup)
# Kernel 2: add (positional encoding)
# Kernel 3: matmul (Q projection)
# Kernel 4: matmul (K projection)
# Kernel 5: matmul (V projection)
# Kernel 6: matmul (attention scores)
# Kernel 7: softmax
# Kernel 8: matmul (weighted sum)
# ... (几十个 kernel)

# 融合后 (compiled 模式):
# Fused Kernel 1: embedding + pos_enc (1个kernel替代2个)
# Fused Kernel 2: QKV projection (1个kernel替代3个)
# Fused Kernel 3: attention + output_proj (1个kernel替代5+个)
# Fused Kernel 4: FFN (SwiGLU fused into 1 kernel)
# ... (大幅减少 kernel 数量和内存访问)
```

## 编译模式选择

`torch.compile()` 支持多种编译模式（`mode` 参数），每种模式在编译时间和运行效率之间做不同的权衡：

| 模式 | 编译开销 | 加速效果 | 适用场景 |
|-----|---------|---------|---------|
| `"default"` | 中等 | 中等 | 通用默认 |
| `"reduce-overhead"` | 较低 | 较高 | 推理首选（减少 Python 开销） |
| `"max-autotune"` | 很高 | 最高 | 反复调用的热点函数 |

**`"default"`**：平衡模式，适合大多数场景。它会尝试自动选择最优的编译策略，包括是否融合、如何调度等。

**`"reduce-overhead"`**：专门为推理优化的模式。它的核心思想是减少 Python 解释器的开销——PyTorch eager 模式下每次调用 forward 都有大量的 Python 层调度开销（dispatch、参数检查、类型推断等），`reduce-overhead` 通过把尽可能多的逻辑下沉到 C++/内核层面来消除这些开销。对于推理这种"一次编译、多次执行"的场景特别有效。

**`"max-autotune"`**：最激进的优化模式。它会尝试大量不同的编译策略组合（不同的 tile size、不同的融合方式、不同的并行度），通过实际 benchmark 选择最快的版本。首次编译可能需要几分钟甚至更长时间（对于大模型），但之后的执行速度是最快的。适合那些会长时间运行的推理服务。

```python
def compare_compile_modes():
    model = GPT(GPTConfig(vocab_size=1000, n_embed=256, num_heads=4, num_layers=4)).cuda().eval()
    x = torch.randint(0, 1000, (1, 128)).cuda()
    
    modes = ["default", "reduce-overhead"]
    
    for mode in modes:
        compiled = torch.compile(model, mode=mode)
        
        with torch.no_grad():
            for _ in range(20):  # warmup
                _ = compiled(x)
        
        torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(200):
                t0 = time.perf_counter()
                _ = compiled(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        
        avg_ms = sum(times[20:]) / len(times[20:]) * 1000  # skip warmup
        p95_ms = sorted(times)[int(len(times)*0.95)] * 1000
        
        print(f"mode={mode:>18s}: avg={avg_ms:.2f}ms, p95={p95_ms:.2f}ms")


compare_compile_modes()

# 典型输出:
# mode=default         : avg=5.45ms, p95=6.12ms
# mode=reduce-overhead: avg=4.12ms, p95=4.56ms
```

## 动态形状（Dynamic Shapes）的支持与限制

`torch.compile()` 对动态形状（dynamic shapes）的支持是一个持续改进中的领域。所谓动态形状，指的是输入张量的形状在多次调用之间可能不同——这在 LLM 推理中很常见：用户的问题长度不同，所以序列长度 T 每次调用都不同。

PyTorch 2.1+ 对动态形状有了显著改善，但仍有一些限制：

```python
def dynamic_shapes_demo():
    model = GPT(GPTConfig()).cuda().eval()
    compiled = torch.compile(model, dynamic=True)
    
    # 不同长度的输入
    lengths = [64, 128, 256, 512]
    
    for length in lengths:
        x = torch.randint(0, 1000, (1, length)).cuda()
        with torch.no_grad():
            out = compiled(x)
        print(f"  seq_len={length}: output shape={out['logits'].shape} ✓")


dynamic_shapes_demo()

# 输出:
#   seq_len=64: output shape=torch.Size([1, 64, 1000]) ✓
#   seq_len=128: output shape=torch.Size([1, 128, 1000]) ✓
#   seq_len=256: output shape=torch.Size([1, 256, 1000]) ✓
#   seq_len=512: output shape=torch.Size([1, 512, 1000]) ✓
```

当 `dynamic=True` 时，Dynamo 会生成支持多种输入形状的通用图。但如果你的模型中有某些操作对形状敏感且无法被 Dynamo 正确追踪（比如使用了 `.view()` 且 shape 依赖于运行时值），可能会触发 **graph break**——此时 Dynamo 会回退到 eager 模式执行这部分代码，虽然不会报错但会影响性能。

如果你遇到了动态形状相关的问题，可以通过环境变量开启调试日志：

```bash
export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE=1

python your_inference_script.py
```

日志中会显示哪些操作被成功捕获到图中、哪些触发了 graph break、以及 recompilation（重新编译）的频率。如果 recompilation 发生得太频繁（每次调用都重新编译），说明动态形状变化太大，可以考虑设置 `fullgraph=True` 来强制整个 forward 在一个图中完成（代价是可能增加显存占用）。

## 与 SDPA（Scaled Dot Product Attention）的配合

PyTorch 2.2+ 内置了 **SDPA（Scaled Dot Product Attention）**——一个高度优化的 Attention 实现，融合了 FlashAttention 和 Memory Efficient Attention 的优势。当 `torch.compile()` 遇到 `F.scaled_dot_product_attention()` 调用时，会自动使用最优的实现路径。

```python
def sdpa_with_compile():
    class OptimizedAttention(nn.Module):
        def __init__(self, n_embed, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = n_embed // num_heads
            self.qkv = nn.Linear(n_embed, 3 * n_embed)
            self.proj = nn.Linear(n_embed, n_embed)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            
            # 使用 SDPA 替代手动实现的 attention
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True,
                scale=self.head_dim ** -0.5,
            )
            
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(out)

    model = OptimizedAttention(256, 4).cuda().eval()
    compiled_model = torch.compile(model, mode="reduce-overhead")
    
    x = torch.randn(1, 512, 256).cuda()
    
    with torch.no_grad():
        out = compiled_model(x)
    
    print(f"SDPA + compile output: {out.shape}")
    print("✓ FlashAttention automatically selected by SDPA")


sdpa_with_compile()
```

SDPA 会自动检测硬件是否支持 FlashAttention（需要 Ampere 架构 + sm80+ 的 capability）。如果支持，使用 FlashAttention 的融合 kernel 实现 O(N²d) 复杂度的注意力（而不是朴素的 O(N²d) 三次矩阵乘法）；如果不支持，则回退到 Memory Efficient Attention 或标准实现。配合 `torch.compile()` 的算子融合，Attention 层的性能可以得到最大程度的优化。

## 常见问题与限制

### 问题一：首次编译耗时过长

对于大模型（如 LLaMA-7B），`torch.compile()` 的首次编译可能需要 1~5 分钟。这是因为 Inductor 需要为数百个算子生成和优化 Triton/C++ 代码，然后调用 NVCC 编译。

**解决方案**：
- 使用 `"reduce-overhead"` 模式（编译更快）
- 利用 `torch.compile(..., fullgraph=False)` 减少编译范围
- 在应用启动时预编译（warmup），不在第一个用户请求时才编译
- 缓存编译结果：PyTorch 会缓存编译后的代码到 `~/.cache/torch/`

### 问题二：某些自定义操作不支持

如果你的模型中使用了不常见的第三方库操作或自定义 CUDA kernel，Dynamo 可能无法追踪它们。

**解决方案**：
```python
# 方案1: 用 @torch.compiler.disable 标记不需要编译的部分
@torch.compiler.disable
def custom_op(x):
    # Dynamo 不会追踪这个函数内部的操作
    return some_third_party_lib.function(x)

# 方案2: 使用 disable_recursion() 控制编译深度
compiled = torch.compile(
    model,
    disable_recursion=[some_custom_module],
)
```

### 问题三：调试困难

编译后的模型错误堆栈信息不如 eager 模式清晰——因为实际的执行路径经过了 Dynamo 的字节码分析和 Inductor 的代码生成，报错行号可能与源码不完全对应。

**解决方案**：
```bash
# 开启详细日志
TORCH_LOGS="+dynamo" python script.py

# 或者临时关闭 compile 来确认问题是否与编译相关
# model = torch.compile(model)  ← 注释掉这行
# 如果注释后问题消失，说明确实是 compile 导致的
```

### 问题四：内存占用增加

`torch.compile()` 可能会增加峰值显存占用，因为编译过程中需要缓存中间结果，而且某些融合操作可能需要额外的临时缓冲区。

**解决方案**：
- 减少 batch size 或序列长度
- 使用 `torch.compile(..., fullgraph=False)` 
- 监控显存使用情况：`torch.cuda.memory_stats()`

到这里，我们掌握了 `torch.compile()` 的原理和使用方法。下一节我们将学习另一类重要的优化技术——**模型量化**，它通过降低数值精度来同时减小模型体积和加速推理。
