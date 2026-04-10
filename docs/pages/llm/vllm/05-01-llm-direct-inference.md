# LLM 类直接推理

> **白板时间**：想象你在实验室里做实验——不需要启动 HTTP 服务、不需要处理网络协议、不需要考虑并发控制。你只想快速地对一批文本做推理，看看效果如何。这就是 `LLM` 类存在的意义：**把 vLLM 的推理引擎当作一个普通的 Python 对象来用**。上一章我们已经在 04-05 中初步接触了离线推理，这一节我们将更系统地从零构建对 `LLM` 类的理解，包括它的生命周期管理、输出对象的完整解析、以及与 API Server 模式的深度对比。

## 一、LLM 类的生命周期

### 1.1 初始化：模型加载的那一刻

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
)
```

这行代码背后发生了什么？

```
┌─────────────────────────────────────────────────────┐
│              LLM() 初始化流程                        │
├─────────────────────────────────────────────────────┤
│  1. 解析 EngineArgs（合并默认值 + 用户参数）          │
│     ↓                                               │
│  2. 加载模型配置（config.json → ModelConfig）        │
│     ↓                                               │
│  3. 加载 Tokenizer（tokenizer_config.json + 文件）   │
│     ↓                                               │
│  4. 计算 KV Cache 内存需求                           │
│     └→ max_num_seqs × max_model_len × per_token_kv  │
│     ↓                                               │
│  5. 初始化 CacheEngine（分配 GPU Block Pool）        │
│     └→ 显存中划分出 KV Cache 区域                    │
│     ↓                                               │
│  6. 加载模型权重到 GPU                              │
│     └→ safetensors / pt / dtensor 格式              │
│     ↓                                               │
│  7. 编译 CUDA Kernels（PagedAttention 等）           │
│     ↓                                               │
│  8. 创建 Scheduler + Executor + Worker               │
│     ↓                                               │
│  9. 返回 LLM 实例（ready to generate）              │
└─────────────────────────────────────────────────────┘
```

**初始化耗时参考**（RTX 4090, PCIe 4.0）：

| 模型 | 权重大小 | 加载时间 | 说明 |
|------|---------|---------|------|
| Qwen2.5-0.5B | ~1 GB | ~3s | 几乎瞬时 |
| Qwen2.5-1.5B | ~3 GB | ~5s | 很快 |
| Qwen2.5-7B | ~15 GB | ~15-20s | 首次需下载 |
| Qwen2.5-14B | ~30 GB | ~25-35s | 需要双卡或量化 |
| Qwen2.5-72B | ~145 GB | ~60-90s | 需要 4×A100 |

### 1.2 EngineArgs：完整的参数体系

`LLM` 构造函数的所有参数最终都汇聚到 `EngineArgs` 数据类中：

```python
from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class EngineArgs:
    """vLLM 引擎完整参数"""
    
    # === 模型相关 ===
    model: str                              # 模型 ID 或本地路径
    tokenizer: Optional[str] = None         # Tokenizer 路径（None=同 model）
    tokenizer_mode: str = "auto"            # auto / slow
    trust_remote_code: bool = False         # 是否信任远程代码
    dtype: str = "auto"                     # auto/half/bfloat16/float16
    quantization: Optional[str] = None      # awq/gptq/fp8/bitsandbytes
    revision: Optional[str] = None          # 模型 revision/branch
    code_revision: Optional[str] = None     # 代码 revision
    download_dir: Optional[str] = None      # 模型下载目录
    load_format: str = "auto"               # auto/safetensors/pt/dtensor/dummy
    
    # === 并行相关 ===
    tensor_parallel_size: int = 1           # TP 大小
    pipeline_parallel_size: int = 1         # PP 大小
    worker_use_ray: bool = False            # 使用 Ray 管理 worker
    ray_address: Optional[str] = None       # Ray 地址
    
    # === 性能相关 ===
    block_size: int = 16                    # PagedAttention Block 大小
    max_model_len: Optional[int] = None     # 最大序列长度
    num_lookahead_slots: int = 0            # 预分配 slot 数
    seed: int = 0                           # 随机种子
    gpu_memory_utilization: float = 0.9     # GPU 显存使用率上限
    swap_space: int = 4                     # CPU swap 空间 (GiB)
    cpu_offload_gb: float = 0               # CPU 卸载显存 (GiB)
    
    # === 功能开关 ===
    enforce_eager: bool = False             # 强制 eager mode
    max_seq_len_to_capture: int = 8192      # CUDA Graph 最大捕获长度
    disable_log_stats: bool = False         # 禁用统计日志
    enable_prefix_caching: bool = False     # 启用前缀缓存
    distributed_executor_backend: Optional[str] = None
    
    # === LoRA ===
    enable_lora: bool = False               # 启用 LoRA
    max_loras: int = 1                      # 最大 LoRA 数量
    max_lora_rank: int = 16                 # 最大 LoRA rank
    lora_extra_vocab_size: int = 25600      # LoRA 额外词表大小
    long_lora_scaling_factors: Optional[tuple] = None
    
    # === Speculative Decoding ===
    speculative_model: Optional[str] = None # 推测解码草案模型
    speculative_draft_tensor_parallel_size: Optional[int] = None
    num_speculative_tokens: int = 5         # 每步推测 token 数
    speculative_max_model_len: Optional[int] = None
    speculative_disable_by_batch_size: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None
    spec_decoding_acceptance_method: str = "rejection_sampling"
    typical_acceptance_fraction: float = 0.5
    
    # === 多模态 ===
    limit_mm_per_prompt: Optional[dict] = None  # 多模态限制
```

### 1.3 常见初始化配置模板

比如下面的程序提供了几种典型场景的配置：

```python
def get_llm_configs():
    """不同场景的 LLM 配置模板"""
    
    configs = {
        "开发测试": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "dtype": "auto",
            "gpu_memory_utilization": 0.7,
            "max_model_len": 2048,
        },
        
        "单卡生产": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.90,
            "max_model_len": 16384,
            "enable_prefix_caching": True,
        },
        
        "双卡大模型": {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "tensor_parallel_size": 2,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.90,
            "max_model_len": 8192,
        },
        
        "量化低显存": {
            "model": "TheBloke/Llama-2-13B-chat-AWQ",
            "quantization": "awq",
            "dtype": "auto",
            "gpu_memory_utilization": 0.92,
            "max_model_len": 8192,
        },
        
        "LoRA 服务": {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "enable_lora": True,
            "max_loras": 10,
            "max_lora_rank": 64,
            "dtype": "auto",
        },
        
        "推测解码加速": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "speculative_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "num_speculative_tokens": 5,
            "dtype": "auto",
        },
    }
    
    return configs


def demo_configs():
    """演示各配置的效果"""
    configs = get_llm_configs()
    
    for name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"[{name}]")
        print(f"{'='*50}")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        if name == "开发测试":
            llm = LLM(**config)
            outputs = llm.generate(["说你好"], 
                                   SamplingParams(max_tokens=16))
            print(f"  测试输出: {outputs[0].outputs[0].text.strip()}")

demo_configs()
```

## 二、Output 对象深度解析

### 2.1 完整数据结构

每次调用 `llm.generate()` 返回的是 `RequestOutput` 对象列表。每个 `RequestOutput` 包含了该请求的全部信息：

```python
@dataclass
class RequestOutput:
    """一次 generate 调用的完整输出"""
    
    prompt: str                          # 输入 prompt
    prompt_token_ids: List[int]          # prompt 的 token ID 列表
    prompt_logprobs: Optional[List]      # prompt 各 token 的 logprobs
    
    outputs: List[CompletionOutput]      # 所有输出候选（n > 1 时有多个）
    finished: bool                       # 是否已完成
    
    metrics: Optional[RequestMetrics]    # 性能指标（如果启用）


@dataclass
class CompletionOutput:
    """单个输出候选"""
    
    index: int                           # 候选索引（0-based）
    text: str                            # 生成的文本
    token_ids: List[int]                 # 生成的 token ID 列表
    cumulative_logprob: float            # 累积 log probability
    logprobs: Optional[List[ProbsInfo]]  # 每个 token 的 logprobs
    finish_reason: Union[str, None]      # 完成原因
    stop_reason: int = 0                 # 停止原因编码
```

### 2.2 完整输出解析示例

```python
from vllm import LLM, SamplingParams

def full_output_analysis():
    """完整解析 RequestOutput 的每个字段"""
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    
    params = SamplingParams(
        temperature=0.7,
        max_tokens=32,
        n=2,                    # 生成 2 个候选
        best_of=2,
        logprobs=3,             # 返回 top-3 logprobs
        prompt_logprobs=3,      # 返回 prompt 的 logprobs
    )
    
    prompts = ["人工智能的三要素是什么？"]
    outputs = llm.generate(prompts, params)
    
    output = outputs[0]
    
    print("=" * 70)
    print("RequestOutput 完整解析")
    print("=" * 70)
    
    # Prompt 信息
    print(f"\n[Prompt]")
    print(f"  文本: {output.prompt}")
    print(f"  Token IDs: {output.prompt_token_ids[:20]}...")
    print(f"  Token 数: {len(output.prompt_token_ids)}")
    print(f"  已完成: {output.finished}")
    
    # Prompt LogProbs
    if output.prompt_logprobs:
        print(f"\n[Prompt LogProbs (前5个token)]")
        for i, plog in enumerate(output.prompt_logprobs[:5]):
            if plog:
                top = list(plog.keys())[0]
                lp = plog[top]
                print(f"  [{i}] '{top}' logprob={lp.logprob:.4f}")
    
    # 输出候选
    print(f"\n[输出候选] 共 {len(output.outputs)} 个\n")
    
    for cand in output.outputs:
        print(f"--- 候选 #{cand.index} ---")
        print(f"  文本: {cand.text.strip()}")
        print(f"  Token IDs: {cand.token_ids}")
        print(f"  Token 数: {len(cand.token_ids)}")
        print(f"  Cumulative LogProb: {cand.cumulative_logprob:.4f}")
        print(f"  Finish Reason: {cand.finish_reason}")
        
        if cand.logprobs:
            print(f"  [Token-by-Token LogProbs (前5个)]")
            for i, lp in enumerate(cand.logprobs[:5]):
                decoded = lp.decoded_token
                logprob_val = lp.logprob
                prob = 2 ** logprob_val
                
                top_tokens = sorted(
                    lp.logprob.items(),
                    key=lambda x: x[1].logprob,
                    reverse=True
                )[:3]
                
                top_str = ", ".join(
                    f"'{t}'({2**tp.logprob:.1%})" 
                    for t, tp in top_tokens
                )
                
                print(f"    [{i}] '{decoded}' "
                      f"(logprob={logprob_val:.4f}, prob={prob:.4%}) "
                      f"| Top-3: [{top_str}]")
        print()

full_output_analysis()
```

典型输出：

```
======================================================================
RequestOutput 完整解析
======================================================================

[Prompt]
  文本: 人工智能的三要素是什么？
  Token IDs: [198, 5840, 3912, 213, 285, 4510, 441, 33, 29471]
  Token 数: 9
  已完成: True

[输出候选] 共 2 个

--- 候选 #0 ---
  文本: 数据、算法和算力是人工智能发展的三大核心要素。
  Token IDs: [37456, 1234, 5678, ...]
  Token 数: 18
  Cumulative LogProb: -12.3456
  Finish Reason: stop

--- 候选 #1 ---
  文本: 通常认为，AI三要素包括：算法（Algorithm）、数据（Data）、计算力（Computing Power）。
  Token IDs: [38901, 2345, 6789, ...]
  Token 数: 22
  Cumulative LogProb: -14.5678
  Finish Reason: length
```

### 2.3 finish_reason 含义详解

| finish_reason | 含义 | 处理建议 |
|--------------|------|---------|
| `"stop"` | 正常结束（遇到停止符或 EOS） | 结果可用 |
| `"length"` | 达到 `max_tokens` 上限 | 可能被截断，考虑增大 max_tokens |
| `"end_id"` | 遇到 end-of-sequence token | 正常结束 |
| `"abort"` | 请求被取消 | 需要重试 |
| `"tool_calls"` | 生成了工具调用 | 解析 tool_calls 字段 |

### 2.4 Metrics 性能指标

当启用统计日志时（`disable_log_stats=False`），可以获取详细的性能指标：

```python
def metrics_demo():
    """性能指标分析"""
    
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="auto",
        disable_log_stats=False,
    )
    
    params = SamplingParams(temperature=0.7, max_tokens=64)
    
    prompts = ["解释机器学习"] * 10
    outputs = llm.generate(prompts, params)
    
    output = outputs[0]
    
    if output.metrics:
        m = output.metrics
        print("[性能指标]")
        print(f"  Prompt 处理耗时: {m.prompt_latency:.3f}s")
        print(f"  生成阶段耗时: {m.generation_latency:.3f}s")
        print(f"  总耗时: {m.latency:.3f}s")
        print(f"  Prompt Tokens: {m.num_prompt_tokens}")
        print(f"  生成 Tokens: {m.num_generated_tokens}")
        print(f"  总 Tokens: {m.num_prompt_tokens + m.num_generated_tokens}")
        print(f"  吞吐量: {(m.num_prompt_tokens + m.num_generated_tokens) / m.latency:.1f} tok/s")
        
        if hasattr(m, 'first_token_time'):
            ttft = m.first_token_time
            print(f"  TTFT: {ttft*1000:.0f}ms")
        
        if hasattr(m, 'time_in_queue'):
            print(f"  排队时间: {m.time_in_queue*1000:.0f}ms")

metrics_demo()
```

## 三、generate() 方法深入

### 3.1 方法签名

```python
def generate(
    self,
    prompts: Union[str, List[str],           # 输入：字符串或列表
                List[int], List[List[int]]],  # 或直接传 token IDs
    sampling_params: SamplingParams,          # 采样参数
    use_tqdm: bool = True,                    # 是否显示进度条
) -> List[RequestOutput]:
    ...
```

**关键特性**：
- **支持混合输入**：可以同时传字符串和 token ID 列表
- **自动批处理**：无论传入多少 prompt，内部统一调度
- **进度条集成**：默认启用 tqdm 显示进度

### 3.2 不同输入方式对比

```python
def input_methods_comparison():
    """三种输入方式的对比"""
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    sp = SamplingParams(max_tokens=16, temperature=0)
    
    # 方式1：字符串输入
    out1 = llm.generate(["你好"], sp)
    print(f"[字符串] {out1[0].outputs[0].text.strip()}")
    
    # 方式2：Token IDs 输入（需要手动 tokenize）
    import requests
    r = requests.post("http://localhost:8000/v1/tokenize",
                      json={"model": "", "text": "你好"})
    tokens = r.json()["tokens"]
    out2 = llm.generate([tokens], sp)
    print(f"[TokenIDs] {out2[0].outputs[0].text.strip()}")
    
    # 方式3：Prompt Token IDs（指定 prompt 部分）
    out3 = llm.generate(
        prompts=["你好"],
        sampling_params=sp,
        prompt_token_ids=[tokens],  # 显式指定 prompt 的 token ids
    )
    print(f"[带PromptIDs] {out3[0].outputs[0].text.strip()}")

input_methods_comparison()
```

### 3.3 单次 vs 批量的性能差异

```python
import time
import numpy as np

def batching_performance():
    """批量 vs 逐条调用的性能对比"""
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    sp = SamplingParams(max_tokens=32, temperature=0.3)
    
    n_samples = 64
    prompts = [f"第{i}条：请用一句话总结" for i in range(n_samples)]
    
    # 逐条调用
    start = time.time()
    individual_results = []
    for p in prompts:
        r = llm.generate([p], sp)
        individual_results.append(r[0])
    individual_time = time.time() - start
    
    # 批量调用
    start = time.time()
    batch_results = llm.generate(prompts, sp)
    batch_time = time.time() - start
    
    speedup = individual_time / batch_time
    
    print("=" * 50)
    print(f"[性能对比] {n_samples} 条样本")
    print("=" * 50)
    print(f"逐条调用: {individual_time:.2f}s ({individual_time/n_samples:.3f}s/条)")
    print(f"批量调用: {batch_time:.2f}s ({batch_time/n_samples:.3f}s/条)")
    print(f"加速比: {speedup:.1f}x")
    print(f"\n结果一致性: {len(individual_results) == len(batch_results)}")

batching_performance()
```

典型结果：

```
==================================================
[性能对比] 64 条样本
==================================================
逐条调用: 45.23s (0.707s/条)
批量调用: 12.34s (0.193s/条)
加速比: 3.7x
```

**为什么批量更快？**
1. **GPU 利用率更高**：多个请求共享同一个 kernel launch
2. **Continuous Batching 效应**：短请求完成后立即释放资源给新请求
3. **减少 Python 开销**：单次函数调用 vs N 次

## 四、资源管理与清理

### 4.1 手动释放 GPU 资源

```python
def resource_management():
    """GPU 资源管理"""
    
    import torch
    
    # 创建 LLM 实例
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    
    outputs = llm.generate(["测试"], SamplingParams(max_tokens=8))
    print(f"[使用中] VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # 显式删除
    del llm
    
    # 清理 CUDA 缓存
    torch.cuda.empty_cache()
    
    print(f"[释放后] VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

resource_management()
```

### 4.2 上下文管理器模式

```python
from contextlib import contextmanager

@contextmanager
def llm_context(**kwargs):
    """LLM 上下文管理器，确保资源正确释放"""
    llm = LLM(**kwargs)
    try:
        yield llm
    finally:
        del llm
        import torch
        torch.cuda.empty_cache()


# 使用方式
with llm_context(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto") as llm:
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=16))
    print(outputs[0].outputs[0].text)

# 退出 with 块后自动释放
```

### 4.3 多模型切换

```python
def multi_model_switching():
    """在有限显存下切换不同模型"""
    
    models = [
        ("Qwen2.5-0.5B", {"model": "Qwen/Qwen2.5-0.5B-Instruct"}),
        ("TinyLlama", {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}),
    ]
    
    results = {}
    
    for name, config in models:
        print(f"\n加载模型: {name}...")
        with llm_context(dtype="auto", **config) as llm:
            out = llm.generate(
                ["自我介绍"], 
                SamplingParams(max_tokens=32, temperature=0.7)
            )
            results[name] = out[0].outputs[0].text.strip()
            print(f"  → {results[name][:60]}...")
        print(f"  已释放 {name}")

multi_model_switching()
```

## 五、与 API Server 模式的决策指南

### 5.1 决策矩阵

```
选择离线推理还是 API Server？
│
├─ 需要服务外部客户端（Web/App/其他服务）
│   └─→ API Server ✅（唯一选择）
│
├─ 只在自己的 Python 进程中使用
│   ├─ 需要最高效率（无网络开销）
│   │   └─→ 离线推理 ✅
│   ├─ 需要 Jupyter Notebook 交互
│   │   └─→ 离线推理 ✅
│   └─ 需要与其他进程共享模型
│       └─→ API Server ✅（多进程共享一个服务）
│
├─ 需要运行一次性批量任务
│   └─→ 离线推理 ✅（脚本跑完即退出）
│
├─ 需要长期运行的服务
│   └─→ API Server ✅（守护进程 + 自动重启）
│
└─ 需要调试和分析模型行为
    ├─ 快速实验 → 离线推理 ✅
    └─ 生产级调试 → API Server + RCA 端点 ✅
```

### 5.2 混合使用模式

在实际项目中，两种模式常常结合使用：

```python
"""混合使用模式的架构示例"""

# 场景：数据处理管道
#
# 1. 用离线推理做批量预处理（Embedding 提取、分类标注）
# 2. 用 API Server 做在线实时推理（用户查询响应）

# ===== 离线部分：批量预处理 =====
from vllm import LLM, SamplingParams

embed_llm = LLM(
    model="BAAI/bge-m3",
    dtype="auto",
)

documents = [...]  # 大量文档
embeddings = embed_llm.generate(documents, SamplingParams(...))

# 保存到向量库
save_to_vector_db(embeddings)

# ===== 在线部分：API Server 已经在另一个进程中运行 =====
from openai import OpenAI

chat_client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="token"
)

# 用户查询时通过 API 调用
response = chat_client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[...],
)
```

---

## 六、总结

本节系统学习了 `LLM` 类的使用方法：

| 主题 | 核心要点 |
|------|---------|
| **生命周期** | 初始化（加载模型+分配KV Cache+编译kernel）→ 使用 → 删除释放 |
| **EngineArgs** | 40+ 参数覆盖模型/并行/性能/功能/LoRA/推测解码/多模态等全部维度 |
| **RequestOutput** | 包含 prompt、prompt_token_ids、outputs(多个候选)、metrics 完整信息 |
| **CompletionOutput** | text/token_ids/cumulative_logprob/logprobs/finish_reason 六大属性 |
| **generate()** | 支持字符串/token IDs/prompt_token_ids 三种输入，自动批处理 |
| **批量优势** | 比逐条调用快 3-4x（GPU利用率 + Continuous Batching） |
| **资源管理** | `del` + `cuda.empty_cache()` 或上下文管理器确保显存释放 |

**核心要点回顾**：

1. **`LLM()` 初始化是一次性成本**——模型加载后反复调用 `generate()` 无需重新加载
2. **`RequestOutput` 是信息宝库**——不仅有生成的文本，还有完整的 token 级别 logprobs 和性能指标
3. **始终优先使用批量调用**——把所有 prompt 打成一个 list 传进去，让 vLLM 内部的调度器发挥威力
4. **资源管理不可忽视**——长时间运行的脚本要注意及时释放不用的 LLM 实例
5. **离线推理和 API Server 不是互斥的**——可以在同一架构中组合使用

下一节我们将学习 **高效批量处理技巧**，涵盖大数据集分批策略、断点续传、多进程并行等生产级话题。
