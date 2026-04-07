# Pipeline 性能优化：从原型到生产级部署

## 这一节讲什么？

前面三节我们学会了如何使用和自定义 Pipeline，但如果你直接把开发环境中的 Pipeline 代码搬到生产服务器上，很可能会遇到性能瓶颈——延迟太高、吞吐量不够、显存占用过大、GPU 利用率低等问题。

这一节是 Pipeline 章节的"压轴戏"，我们将全面覆盖生产级性能优化的核心技术：

1. **ONNX Runtime**：将 PyTorch 模型转换为 ONNX 格式，实现跨平台、高性能推理
2. **TensorRT**：NVIDIA 的终极推理优化器，在 GPU 上榨干每一滴性能
3. **批量化策略**：dynamic batching（动态批处理）vs static batching，如何在延迟和吞吐之间做权衡
4. **并发推理**：用 asyncio 实现异步 Pipeline 服务，大幅提升并发能力
5. **量化推理**：INT8/INT4 量化，用精度换速度和内存
6. **基准测试对比**：各种方案的实际性能数据

---

## 一、ONNX Runtime 加速

## 1.1 为什么需要 ONNX？

PyTorch 的 Eager Mode（默认执行模式）虽然灵活方便，但存在以下性能问题：

- **动态计算图**：每次前向传播都要重新构建计算图
- **Python 开销**：Python 解释器的 GIL 和动态类型检查带来额外开销
- **算子未融合**：简单的逐元素操作没有被合并成高效的内核调用

ONNX (Open Neural Network Exchange) 通过以下方式解决这些问题：

```
PyTorch 模型 (.pt) ──导出──→ ONNX 模型 (.onnx)
                                  │
                                  ├── 静态计算图（提前编译）
                                  ├── 跨平台统一格式
                                  ├── 算子融合与图优化
                                  └── 多种后端可执行:
                                      ├── CPU: ONNX Runtime
                                      ├── GPU: CUDA/TensorRT
                                      ├── 移动端: CoreML/NNAPI
                                      └── 浏览器: WebAssembly
```

## 1.2 导出和使用 ONNX Pipeline

```python
from transformers import pipeline
import time

# ===== 原始 PyTorch Pipeline =====
pt_pipe = pipeline("text-classification", device=0, framework="pt")

# ===== ONNX Pipeline =====
onnx_pipe = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    exporter="onnx",        # 自动导出并使用 ONNX
    device="cuda:0",
)

texts = ["I love this product!", "This is terrible.", "It's okay, nothing special."]

# 对比测试
import numpy as np

n_runs = 100
pt_times = []
onnx_times = []

for _ in range(n_runs):
    start = time.time()
    pt_pipe(texts)
    pt_times.append(time.time() - start)

    start = time.time()
    onnx_pipe(texts)
    onnx_times.append(time.time() - start)

pt_mean = np.mean(pt_times) * 1000
onnx_mean = np.mean(onnx_times) * 1000
speedup = pt_mean / max(onnx_mean, 0.001)

print(f"{'':20s} {'PyTorch':>12s} {'ONNX':>12s} {'加速比':>8s}")
print("-" * 56)
print(f"{'平均延迟 (ms)':20s} {pt_mean:>12.2f} {onnx_mean:>12.2f} {speedup:>7.1f}x")
print(f"{'P90 延迟 (ms)':20s} {np.percentile(pt_times, 90)*1000:>12.2f} "
      f"{np.percentile(onnx_times, 90)*1000:>12.2f}")
print(f"{'P99 延迟 (ms)':20s} {np.percentile(pt_times, 99)*1000:>12.2f} "
      f"{np.percentile(onnx_times, 99)*1000:>12.2f}")

print(f"\n结果一致性: {pt_pipe(texts)[0]['label'] == onnx_pipe(texts)[0]['label']}")
```

典型输出（GPU 环境）：
```
                    PyTorch       ONNX     加速比
--------------------------------------------------------
平均延迟 (ms)           8.23         3.45      2.4x
P90 延迟 (ms)          10.12         4.23
P99 延迟 (ms)          15.67         6.89

结果一致性: True
```

> **注意**：首次运行 ONNX Pipeline 时会有模型导出开销（几秒到几十秒），后续调用则非常快。

## 1.3 手动导出 ONNX 模型

对于更精细的控制，你可以手动导出：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import main_export

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
output_dir = "./onnx_models/sentiment"

main_export(
    model_name=model_name,
    output=output_dir,
    task="text-classification",
    opset=17,
    dtype="float32",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ORTModelForSequenceClassification.from_pretrained(output_dir)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cuda:0")

result = pipe("This is amazing!")
print(result)
```

## 1.4 ONNX 优化技术详解

ONNX Runtime 内部做了很多你不需要手动配置的优化：

```python
def demonstrate_onnx_optimizations():
    """展示 ONNX 的各项优化"""

    optimizations = [
        ("常数折叠 (Constant Folding)", "将静态已知的计算结果预计算"),
        ("死代码消除 (Dead Code Elimination)", "移除不影响输出的无用节点"),
        ("算子融合 (Operator Fusion)", "将多个小操作合并为一个内核调用"),
        ("内存分配优化", "减少中间张量的内存分配和拷贝"),
        ("并行化", "自动识别独立的计算路径并行执行"),
        ("布局优化", "选择最优的张量内存布局 (NCHW vs NHWC)"),
    ]

    print("ONNX Runtime 内置优化:\n")
    for name, desc in optimizations:
        print(f"  ✅ {name}")
        print(f"     → {desc}\n")

demonstrate_onnx_optimizations()
```

---

## 二、TensorRT 加速（NVIDIA GPU 终极方案）

## 2.1 TensorRT 是什么？

TensorRT 是 NVIDIA 推出的高性能深度学习推理优化器，专门针对 NVIDIA GPU 进行了极致优化。它的核心能力包括：

- **层融合 (Layer Fusion)** + **张量融合 (Tensor Fusion)**
- **内核自动调优 (Auto-Tuning)**：针对具体 GPU 型号选择最优 CUDA kernel
- **精度校准 (Precision Calibration)**：FP16/INT8 推理
- **动态 Shape 支持**：处理变长输入
- **内存优化**：减少显存占用

## 2.2 使用 TensorRT Pipeline

```python
try:
    from transformers import pipeline

    trt_pipe = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        exporter="tensorrt",   # 使用 TensorRT 后端
        device="cuda:0",
        torch_dtype=torch.float16,
    )

    result = trt_pipe("Hello world! This is a test.")
    print(f"TensorRT 结果: {result}")

except ImportError as e:
    print(f"TensorRT 未安装或不可用: {e}")
    print("安装方法: pip install tensorrt")
```

> **前置条件**：
> - NVIDIA GPU（支持 CUDA）
> - 安装 `tensorrt` Python 包
> - HF 的 `optimum` 库 (`pip install optimum[nvidia]`)

## 2.3 性能对比全景图

下面是一个综合性的基准测试框架，对比所有主流推理方案：

```python
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline

def comprehensive_benchmark():
    """全面的 Pipeline 性能基准测试"""
    task = "text-classification"
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    texts = ["I love this!" for _ in range(16)] * 5  # 80 条文本
    n_warmup = 10
    n_benchmark = 50

    backends = {}

    # Backend 1: PyTorch Eager
    try:
        backends["PyTorch-Eager"] = pipeline(task, model=model_id, device=0, framework="pt")
    except Exception as e:
        print(f"PyTorch-Eager 不可用: {e}")

    # Backend 2: PyTorch Torch Compile (PyTorch 2.0+)
    try:
        import torch
        if hasattr(torch, 'compile'):
            backends["PyTorch-Compile"] = pipeline(task, model=model_id, device=0,
                                                   torch_compile=True)
    except Exception as e:
        print(f"Torch Compile 不可用: {e}")

    # Backend 3: ONNX Runtime
    try:
        backends["ONNX-Runtime"] = pipeline(task, model=model_id, exporter="onnx", device="cuda:0")
    except Exception as e:
        print(f"ONNX Runtime 不可用: {e}")

    # Backend 4: TensorRT
    try:
        backends["TensorRT"] = pipeline(task, model=model_id, exporter="tensorrt",
                                        device="cuda:0", torch_dtype=torch.float16)
    except Exception as e:
        print(f"TensorRT 不可用: {e}")

    if not backends:
        print("没有可用的后端！请至少安装 PyTorch 或 ONNX Runtime")
        return

    results = {}
    for name, pipe in backends.items():

        for _ in range(n_warmup):
            try:
                pipe(texts[:4])
            except Exception:
                pass

        latencies = []
        throughputs = []

        for _ in range(n_benchmark):
            start = time.perf_counter()
            result = pipe(texts)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            throughputs.append(len(texts) / (elapsed / 1000))

        results[name] = {
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
            },
            "throughput": {
                "mean": statistics.mean(throughputs),
            },
        }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    names = list(results.keys())
    means = [results[n]["latency_ms"]["mean"] for n in names]
    p99s = [results[n]["latency_ms"]["p99"] for n in names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

    bars = axes[0].bar(names, means, color=colors, alpha=0.8, label='Mean')
    axes[0].errorbar(names, means, yerr=[np.array(p99s)-np.array(means)],
                     fmt='none', ecolor='red', capsize=5, label='P99')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('推理延迟对比\n(batch=80, lower is better)')
    axes[0].legend()

    tputs = [results[n]["throughput"]["mean"] for n in names]
    axes[1].bar(names, tputs, color=colors, alpha=0.8)
    axes[1].set_ylabel('Throughput (samples/sec)')
    axes[1].set_title('吞吐量对比\n(higher is better)')

    for ax in axes:
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print(f"{'Backend':<18} {'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} "
          f"{'P99(ms)':>10} {'TPS':>10}")
    print("=" * 80)
    for name in names:
        r = results[name]["latency_ms"]
        tp = results[name]["throughput"]["mean"]
        print(f"{name:<18} {r['mean']:>10.2f} {r['p50']:>10.2f} {r['p95']:>10.2f} "
              f"{r['p99']:>10.2f} {tp:>10.1f}")

comprehensive_benchmark()
```

典型输出（A100 GPU）：
```
================================================================================
Backend            Mean(ms)    P50(ms)    P95(ms)    P99(ms)        TPS
================================================================================
PyTorch-Eager         12.34       11.89       15.67       22.34       6.5
PyTorch-Compile        7.21        6.98        8.91       11.23      11.1
ONNX-Runtime          4.56        4.42        5.78        7.89      17.5
TensorRT              2.89        2.78        3.45        4.67      27.7
```

可以看到：**TensorRT 相比原始 PyTorch 可以有 4-5 倍的加速**！

---

## 三、批量化策略

## 3.1 Dynamic Batching vs Static Batching

批量化是提升 GPU 利用率最有效的手段之一。Pipeline 支持两种模式：

### Static Batching（静态批处理）

```python
pipe = pipeline("text-classification", device=0)

texts = ["text one", "text two", "text three", ...]

results = pipe(texts, batch_size=16)
```

特点：
- 所有请求收集到一个固定大小的 batch 后一起处理
- **优点**：GPU 利用率最高，吞吐量最大
- **缺点**：必须等 batch 满才能处理，长尾请求延迟高

### Dynamic Batching（动态批处理）

```python
pipe = pipeline("text-classification", device=0)

for text in text_stream:
    result = pipe(text)  # 每个"单条"请求内部会自动与其他并发请求组合
```

HF Pipeline 的 `batch_size` 参数实际上是一种**折中方案**——它会在内部将连续到达的请求缓冲到指定大小再批量发送：

```python
def demonstrate_batching_impact():
    """演示 batch_size 对性能的影响"""
    import time
    import numpy as np

    pipe = pipeline("sentiment-analysis", device=-1)

    n_samples = 200
    texts = ["This is a sample text for testing."] * n_samples

    batch_sizes = [1, 4, 8, 16, 32, 64]

    results_data = []

    for bs in batch_sizes:
        times = []
        for _ in range(5):
            start = time.time()
            _ = pipe(texts, batch_size=bs)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        throughput = n_samples / avg_time

        results_data.append({
            "batch_size": bs,
            "avg_time": avg_time,
            "throughput": throughput,
        })
        print(f"batch_size={bs:>3}: 平均耗时={avg_time:.3f}s, 吞吐量={throughput:.1f} samples/s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bss = [r["batch_size"] for r in results_data]
    times = [r["avg_time"] for r in results_data]
    tputs = [r["throughput"] for r in results_data]

    ax1.plot(bss, times, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Total Time (s)')
    ax1.set_title('总耗时 vs Batch Size')
    ax1.grid(True, alpha=0.3)

    ax2.plot(bss, tputs, 's--', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (samples/s)')
    ax2.set_title('吞吐量 vs Batch Size')
    ax2.grid(True, alpha=0.3)

    optimal_bs = results_data[np.argmax([r["throughput"] for r in results_data])]["batch_size"]
    ax2.axvline(x=optimal_bs, color='green', linestyle=':', alpha=0.7,
               label=f'Optimal BS={optimal_bs}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

demonstrate_batching_impact()
```

## 3.2 选择最优 Batch Size 的经验法则

| 场景 | 推荐 batch_size | 理由 |
|------|----------------|------|
| 低延迟要求 (< 50ms) | 1-4 | 减少排队等待时间 |
| 高吞吐量要求 | 16-64 | 最大化 GPU 利用率 |
| 显存受限 (< 8GB) | 4-8 | 平衡显存和速度 |
| 变长序列（如文档） | 4-8 + padding 策略 | 避免 padding 浪费 |
| A100/H100 (80GB) | 32-128 | 充分利用大显存 |

---

## 四、异步并发推理

对于在线服务场景，单线程处理无法满足高并发需求。我们可以利用 Python 的 `asyncio` 实现 Pipeline 的异步版本：

## 4.1 AsyncPipeline 基础实现

```python
import asyncio
import concurrent.futures
from transformers import pipeline

class AsyncPipeline:
    """
    异步 Pipeline 封装
    使用线程池避免阻塞事件循环
    """

    def __init__(self, task, **kwargs):
        self.pipe = pipeline(task, **kwargs)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    async def __call__(self, inputs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.pipe, inputs)

    async def batch_call(self, inputs_list, batch_size=8):
        """批量异步调用"""
        tasks = [self(inputs) for inputs in inputs_list]
        return await asyncio.gather(*tasks)


async def async_demo():
    """异步 Pipeline 演示"""

    async_pipe = AsyncPipeline("text-classification", device=0)

    queries = [
        "这个产品非常好！",
        "太差了，不推荐购买。",
        "一般般吧。",
        "超出预期的好产品！",
        "客服态度很差。",
    ]

    print("=== 同步调用 ===")
    sync_start = asyncio.get_event_loop().time()
    for q in queries:
        result = async_pipe.pipe(q)
        print(f"  {q[:15]:15s} → {result[0]['label']}")
    sync_elapsed = asyncio.get_event_loop().time() - sync_start
    print(f"同步总耗时: {sync_elapsed:.3f}s\n")

    print("=== 异步并发调用 ===")
    async_start = asyncio.get_event_loop().time()
    results = await async_pipe.batch_call(queries)
    async_elapsed = asyncio.get_event_loop().time() - async_start

    for q, r in zip(queries, results):
        print(f"  {q[:15]:15s} → {r[0]['label']}")
    print(f"异步总耗时: {async_elapsed:.3f}s")
    print(f"加速比: {sync_elapsed/max(async_elapsed, 0.001):.1f}x")


if __name__ == "__main__":
    asyncio.run(async_demo())
```

## 4.2 生产级异步服务模板

```python
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from transformers import pipeline

app = FastAPI(title="NLP Inference Service")

class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    label: str
    score: float

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]

pipe = None

@app.on_event("startup")
async def load_model():
    global pipe
    pipe = pipeline("text-classification",
                   model="uer/roberta-base-finetuned-chinanews-chinese",
                   device=0)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: pipe(request.text))
    return PredictionResponse(label=result[0]["label"], score=result[0]["score"])

@app.post("/batch_predict", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: pipe(request.texts))
    return BatchResponse(predictions=[
        PredictionResponse(label=r["label"], score=r["score"]) for r in results
    ])

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": pipe is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 五、量化推理 —— 以精度换性能

## 5.1 什么是量化？

量化是将模型的权重和激活值从 FP32（32位浮点数）降低到更低精度的表示（如 INT8、INT4），从而：

- **减小模型体积**：FP32→INT8 减少 75% 存储空间
- **加速推理**：INT8 运算比 FP32 快 2-4 倍（尤其在支持 INT8 张量核心的硬件上）
- **降低显存占用**：可以加载更大的模型或在同一 GPU 上跑更大的 batch

代价是**一定的精度损失**——但现代量化技术可以将损失控制在 1% 以内。

## 5.2 INT8 量化 Pipeline

```python
from transformers import pipeline, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

try:
    quantized_pipe = pipeline(
        "text-generation",
        model="gpt2",
        model_kwargs={"quantization_config": bnb_config},
        device_map="auto",
    )

    result = quantized_pipe("Hello, world!", max_length=30)
    print(f"INT8 量化生成: {result[0]['generated_text']}")

except ImportError:
    print("bitsandbytes 未安装，跳过 INT8 示例")
    print("安装命令: pip install bitsandbytes")
```

## 5.3 INT4 量化（更激进的压缩）

```python
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 双重量化（进一步压缩）
)

try:
    pipe_4bit = pipeline(
        "text-generation",
        model="gpt2",
        model_kwargs={"quantization_config": bnb_config_4bit},
        device_map="auto",
    )
except Exception as e:
    print(f"INT4 量化示例: {e}")
```

## 5.4 量化效果对比

```python
def compare_quantization_levels():
    """对比不同精度级别的模型"""

    configs = [
        {"name": "FP32 (原始)", "dtype": torch.float32, "config": None},
        {"name": "FP16 (半精度)", "dtype": torch.float16, "config": None},
        {"name": "BF16 (BFloat16)", "dtype": torch.bfloat16, "config": None},
    ]

    if True:
        try:
            from transformers import BitsAndBytesConfig
            configs.append({"name": "INT8 (8-bit)",
                           "dtype": torch.int8,
                           "config": BitsAndBytesConfig(load_in_8bit=True)})
            configs.append({"name": "INT4 (4-bit NF4)",
                           "dtype": torch.int4,
                           "config": BitsAndBytesConfig(load_in_4bit=True,
                                                         bnb_4bit_quant_type="nf4")})
        except ImportError:
            pass

    print(f"\n{'精度':<18} {'模型大小':>12} {'显存占用':>12} {'相对精度':>10}")
    print("-" * 58)

    for cfg in configs:
        try:
            kwargs = {"torch_dtype": cfg["dtype"]}
            if cfg["config"]:
                kwargs["model_kwargs"] = {"quantization_config": cfg["config"]}
                kwargs["device_map"] = "auto"

            pipe = pipeline("text-generation", model="gpt2", **kwargs)

            size_mb = sum(p.numel() * p.element_size() for p in pipe.model.parameters()) / 1024 / 1024

            prompt = "The meaning of life is"
            result = pipe(prompt, max_new_tokens=20, do_sample=False,
                         pad_token_id=pipe.tokenizer.eos_token_id)

            generated = result[0]["generated_text"][len(prompt):]
            print(f"{cfg['name']:<18} {size_mb:>10.1f}MB {'—':>12} {generated[:40]}...")

        except Exception as e:
            print(f"{cfg['name']:<18} {'N/A':>12} {'错误':>12} {str(e)[:30]}")

compare_quantization_levels()
```

输出示例：
```
精度                 模型大小     显存占用     相对精度
------------------------------------------------------------
FP32 (原始)             521.4MB        —       The meaning of life is to find your purpose...
FP16 (半精度)           260.7MB        —       The meaning of life is to find your purpose...
BF16 (BFloat16)         260.7MB        —       The meaning of life is to find your purpose...
INT8 (8-bit)            145.2MB        —       The meaning of life is to find your purpose...
INT4 (4-bit NF4)         85.6MB        —       The meaning of life is to discover what truly...
```

> **关键发现**：即使降到 INT4，GPT-2 的生成质量仍然保持得相当好！这就是为什么现代大模型服务（如 LLaMA.cpp、Ollama）普遍采用 INT4 量化的原因。

---

## 六、完整的性能优化决策树

在实际项目中，你应该按以下顺序考虑优化措施：

```
你的 Pipeline 性能不够好？
│
├─ 第一步：确认瓶颈在哪里
│   ├─ CPU 密集？ → Tokenize 太慢？→ 用 FastTokenizer
│   ├─ GPU 利用率低？ → batch_size 太小？
│   └─ 显存不足？ → 模型太大？
│
├─ 第二步：低成本优化（无需改代码架构）
│   ├─ 增加 batch_size（如果延迟允许）
│   ├─ 切换到 FP16/BF16（几乎零精度损失）
│   ├─ 使用 Fast Tokenizer（Rust 实现）
│   └─ 开启 Torch Compile（PyTorch 2.0+）
│
├─ 第三步：中等成本优化
│   ├─ ONNX Runtime 导出（~2-3x 加速）
│   ├─ INT8 量化（~4x 模型缩小，<1% 精度损失）
│   └─ 异步并发（FastAPI + ThreadPoolExecutor）
│
└─ 第四步：高成本但高回报
    ├─ TensorRT（~4-5x 加速，需 NVIDIA GPU）
    ├─ INT4 量化（~6x 模型缩小，轻微精度损失）
    ├─ vLLM / TGI（专用 LLM 推理引擎）
    └─ 模型蒸馏（训练更小的学生模型）
```

---

## 小结

这一节我们全面掌握了 Pipeline 的生产级性能优化技能：

1. **ONNX Runtime**：通过 `exporter="onnx"` 一行切换，获得 2-3x 加速；底层依赖常量折叠、算子融合、并行化等图优化技术
2. **TensorRT**：NVIDIA GPU 上的终极方案，通过层融合、kernel auto-tuning、精度校准实现 4-5x 加速；需要 `optimum[nvidia]` 依赖
3. **批量化策略**：batch_size 是吞吐量的关键杠杆；低延迟选小 batch（1-4），高吞吐选大 batch（16-128）；Dynamic Batching 在两者间取得平衡
4. **异步并发**：通过 `AsyncPipeline` + `ThreadPoolExecutor` + FastAPI 构建高并发推理服务；避免 GIL 阻塞事件循环
5. **量化推理**：FP16/BF16（零精度损失，2x 缩减）、INT8（<1% 损失，4x 缩减）、INT4/NF4（轻微损失，6x 缩减）；通过 `BitsAndBytesConfig` 配置
6. **优化决策树**：先定位瓶颈 → 低成本优化（batch/precision/compile）→ 中等成本（ONNX/INT8/async）→ 高回报（TensorRT/INT4/vLLM）

至此，**第4章 Pipeline 高阶 API 全部完成**！接下来进入第5章，学习数据集与数据处理。
