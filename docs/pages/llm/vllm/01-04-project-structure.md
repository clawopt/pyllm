# vLLM 项目结构与代码入口

## 白板导读

前面三节我们一直在"使用"vLLM——安装它、启动它、调用它的 API。但从这一节开始，我们要稍微"打开引擎盖"，看看 vLLM 内部到底是怎么组织的。理解项目结构不仅能帮你更好地排查问题（知道报错发生在哪个模块），还能为后续深入 PagedAttention 原理和性能调优打下基础。你不需要读完 vLLM 的全部源码（那有几十万行），但你需要知道核心模块的职责划分、两种运行模式的区别、以及配置系统的工作方式。这些知识就像一张地图——当你后续在调优参数遇到问题时，能快速定位到"哦，这个参数是在 Engine 初始化阶段生效的"或"这个行为是 Scheduler 负责的"。

---

## 4.1 项目目录结构概览

vLLM 是一个纯 Python 项目（加上自定义 CUDA C++/CUDA kernels），其代码组织清晰、模块化程度高。以下是核心目录结构：

```
vllm/
├── entrypoints/              # 入口点：不同运行模式的启动入口
│   ├── openai/
│   │   └── api_server.py     # ★ OpenAI 兼容 HTTP API Server 入口
│   └── llm/
│       └── cli.py            # 离线推理命令行工具入口
│
├── core/                     # ★ 核心引擎：调度器 + 引擎 + 注意力
│   ├── scheduler.py          # Scheduler（调度器）：Continuous Batching 的实现
│   ├── scheduler_utils.py    # Scheduler 辅助数据结构
│   ├── engine.py             # LLMEngine：总指挥官，协调所有组件
│   ├── engine_args.py        # EngineArgs：配置参数定义与解析
│   ├── block_manager.py      # BlockSpaceManager：PagedAttention 的块管理
│   ├── block_table.py        # Block Table 数据结构
│   ├── attention/            # 注意力机制实现
│   │   ├── backends/         # 不同后端（FlashAttention / PagedAttention / xFormers）
│   │   └── ops/              # 自定义 CUDA Kernel 实现
│   └── sampler.py            # 采样器（temperature / top_k / top_p 等）
│
├── worker/                   # Worker 管理：分布式场景下的工作节点
│   ├── worker.py             # 单个 Worker 的主循环
│   └── model_runner.py       # 模型执行器（加载模型、执行 forward pass）
│
├── model_executor/           # 模型执行器抽象层
│   ├── gpu_model_executor.py # GPU 执行器（最常用）
│   └── cpu_model_executor.py # CPU 执行器（offload 场景）
│
├── models/                   # 模型注册与适配
│   ├── registry.py           # 模型注册表（哪些模型被支持）
│   ├── __init__.py           # 自动注册所有支持模型
│   └── <model_name>/         # 各模型的特殊适配逻辑
│       ├── weights.py        # 权重加载适配
│       └── configs.py        # 配置适配
│
├── transformers_utils/       # HuggingFace Transformers 兼容层
│   ├── config.py             # 配置转换（HF config → vLLM config）
│   ├── tokenizer.py          # Tokenizer 包装
│   └── tokenizer_group.py    # 多 tokenizer 管理（离线批处理用）
│
├── distributed/              # 分布式通信
│   ├── communication_op.py   # All-Reduce / All-Gather / Broadcast
│   └── tensor_parallel.py    # 张量并行具体实现
│
├── spec_decode/              # Speculative Decoding（推测解码）
│   ├── speculative_model.py  # Draft Model 管理
│   └── metrics.py            # Accept Rate 等指标统计
│
├── lora/                     # LoRA 适配器支持
│   ├── manager.py            # LoRA Manager（加载/切换/卸载）
│   └── layers.py             # LoRA 层实现
│
├── logging/                  # 日志系统
│   └── init_logger.py        # 日志格式化与初始化
│
├── metrics/                  # Prometheus 指标导出
│   └── prometheus.py         # /metrics 端点实现
│
└── assets/                   # 静态资源（OpenAI 兼容的 OpenAPI schema）
    └── openapi_spec.json
```

### 核心组件关系图

```
                    用户请求 (HTTP)
                         │
                    ┌────▼─────────────────────────┐
                    │      API Server               │
                    │  (entrypoints/openai/)        │
                    │  - OpenAI 协议解析             │
                    │  - 认证 / 限流 / 日志           │
                    └────┬─────────────────────────┘
                         │
                    ┌────▼─────────────────────────┐
                    │      LLM Engine               │
                    │  (core/engine.py)              │
                    │  - 总指挥，协调所有子组件        │
                    └──┬──────┬──────────┬─────────┘
                       │      │          │
            ┌──────────▼┐  ┌─▼──────┐  ┌▼──────────┐
            │Scheduler  │  │Model   │  │Cache      │
            │(core/)    │  │Runner  │  │Engine     │
            │-请求排队   │  │(worker/)│  │(core/)    │
            │-状态管理   │  │-模型加载│  │-Block分配  │
            │-Preemption│  │-Forward│  │-Block回收  │
            └───────────┘  └────────┘  └──────────┘
                                    │
                            ┌───────▼────────┐
                            │  GPU (CUDA)     │
                            │  PagedAttention │
                            │  Custom Kernels │
                            └────────────────┘
```

---

## 4.2 两种运行模式详解

vLLM 有两种截然不同的使用模式，对应不同的代码入口和使用场景。

### 模式一：API Server 模式（生产首选）

这是我们在前面几节中一直使用的模式——启动一个 HTTP 服务，对外提供 OpenAI 兼容的 RESTful API。

**入口文件**：`entrypoints/openai/api_server.py`

**启动方式**：
```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
```

**内部流程**：

```python
# 简化后的 api_server.py 核心逻辑
import uvicorn
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.openai.serving_engine import OpenAIServing

def main():
    # 1. 解析命令行参数 → EngineArgs
    engine_args = EngineArgs.from_cli_args()
    
    # 2. 创建 LLMEngine（一次性加载模型到 GPU）
    engine = LLMEngine.from_engine_args(engine_args)
    
    # 3. 创建 OpenAI 兼容服务层
    openai_serving = OpenAIServing(engine, ...)
    
    # 4. 启动 Uvicorn ASGI Server（基于 FastAPI）
    app = create_app(openai_serving)  # 注册路由: /v1/chat/completions 等
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
```

**关键特点**：
- 基于 **FastAPI** + **Uvicorn** 构建（异步高性能 Web 框架）
- 提供 `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 等标准端点
- 支持 SSE 流式输出
- 内置 **Prometheus metrics** 端点 (`/metrics`)
- 支持多 worker 并发处理请求

**适用场景**：
- ✅ 生产环境部署
- ✅ 需要服务多个客户端应用
- ✅ 需要 HTTP 负载均衡和水平扩展
- ✅ 需要与 LangChain / LlamaIndex / Next.js 等框架集成

### 模式二：离线推理模式（批量处理首选）

这种模式下不启动 HTTP 服务，而是直接在 Python 中调用 `LLM` 类进行批量生成。

**入口类**：`vllm.LLM`（在 `__init__.py` 中暴露）

**使用方式**：
```python
from vllm import LLM, SamplingParams

# 直接创建 LLM 实例
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

# 批量推理（直接传入 prompt 列表）
outputs = llm.generate(
    ["解释什么是量子计算", "写一首关于春天的诗", "Python GIL 是什么"],
    sampling_params,
)

for output in outputs:
    print(f"输入: {output.prompt}")
    print(f"输出: {output.outputs[0].text}")
    print("-" * 40)
```

**内部流程**：

```python
# 简化后的 LLM 类核心逻辑
class LLM:
    def __init__(self, model, **kwargs):
        # 1. 解析参数 → EngineArgs
        self.engine_args = EngineArgs(model=model, **kwargs)
        
        # 2. 创建 LLMEngine（与 API Server 模式共享同一个 Engine！）
        self.llm_engine = LLMEngine.from_engine_args(self.engine_args)
    
    def generate(self, prompts, sampling_params):
        # 3. 将 prompts 转换为 RequestOutput 格式
        requests = self._build_requests(prompts, sampling_params)
        
        # 4. 送入 Engine 执行推理循环
        outputs = self.llm_engine.generate(requests)
        
        # 5. 返回结果
        return outputs
```

**关键特点**：
- 无需 HTTP 开销，进程内直接调用
- `LLM.generate()` 返回完整结果（非流式）
- 适合**批量评估、数据标注、离线处理**
- 同样使用 PagedAttention 和 Continuous Batching（底层引擎完全相同）

**适用场景**：
- ✅ 数据集批量推理（如对 100 万条数据做分类）
- ✅ 模型评估 benchmark（MMLU / HumanEval / C-Eval）
- ✅ A/B 测试（同一组 prompt 对比不同模型）
- ✅ Embedding 提取（向量化大量文档）
- ❌ 不适合需要实时响应的在线服务

### 两种模式的对比总结

| 维度 | API Server 模式 | 离线推理模式 |
|:---|:---|:---|
| **入口** | `api_server.py` | `LLM` 类 |
| **协议** | HTTP (REST + SSE) | Python 函数调用 |
| **延迟** | 有网络开销 | 最小开销 |
| **并发** | 多 client 并发连接 | 单进程内 batch |
| **调试** | 需要看日志 | 可直接 breakpoint |
| **典型用途** | 在线服务 | 批量处理 |
| **底层引擎** | 完全相同（同一个 `LLMEngine`） | 完全相同 |

> **重要认知**：无论你用哪种模式，底层的 **LLMEngine → Scheduler → ModelRunner → PagedAttention** 这条调用链是完全一样的。API Server 只是在 Engine 外面包了一层 FastAPI；离线模式只是直接调用了 Engine。这意味着你在一种模式下调优的经验（比如调整 scheduler 参数）可以直接迁移到另一种模式。

---

## 4.3 EngineArgs 配置系统

vLLM 的所有启动参数最终都会汇聚到一个叫 `EngineArgs` 的数据类中。理解它能帮你搞清楚每个参数在什么时候生效、怎么传递到底层组件。

### EngineArgs 核心字段

```python
@dataclass
class EngineArgs:
    """vLLM 引擎的全部配置参数"""
    
    # ===== 模型相关 =====
    model: str                          # 模型名称或路径
    tokenizer: Optional[str] = None     # Tokenizer 路径（默认与 model 相同）
    revision: Optional[str] = None      # 模型版本
    tokenizer_mode: str = "auto"        # tokenizer 模式
    trust_remote_code: bool = False     # 是否信任远程代码
    dtype: str = "auto"                 # 权重数据类型
    download_dir: Optional[str] = None  # 下载目录
    
    # ===== 并行相关 =====
    tensor_parallel_size: int = 1       # TP 大小
    pipeline_parallel_size: int = 1     # PP 大小
    distributed_executor_backend: Optional[str] = None  # 分布式后端 (ray/mp)
    worker_use_ray: bool = False        # 是否使用 Ray
    
    # ===== 性能相关 =====
    max_model_len: int = 2048           # 最大上下文长度
    gpu_memory_utilization: float = 0.90 # GPU 显存利用率
    swap_space: float = 4               # CPU swap 空间 (GB)
    cpu_offload_gb: float = 0           # CPU offload 大小 (GB)
    num_gpu_blocks_override: Optional[int] = None  # 手动指定 Block 数
    max_num_seqs: int = 256             # 最大并发序列数
    max_num_batched_tokens: Optional[int] = None  # 最大批 token 数
    scheduler_delay_factor: float = ... # 调度延迟因子
    
    # ===== 功能开关 =====
    enable_lora: bool = False           # LoRA 支持
    max_loras: int = 1                  # 最大 LoRA 数量
    max_lora_rank: int = 16             # LoRA 最大 rank
    enable_prefix_caching: bool = False # 前缀缓存
    
    # ===== 推测解码 =====
    speculative_model: Optional[str] = None  # Draft 模型路径
    num_speculative_tokens: int = 5     # 每个 step 推测 token 数
    
    # ===== 服务相关 =====
    enforce_eager: bool = False         # 强制 eager 模式（禁用 CUDA Graph）
    ...
```

### 参数传递链路

当你执行 `--max-model-len 16384` 时，这个值经历了以下传递过程：

```
命令行参数: --max-model-len 16384
    │
    ▼
EngineArgs.from_cli_args() 解析
    │
    ▼
EngineArgs.max_model_len = 16384
    │
    ▼
LLMEngine.from_engine_args(engine_args)
    │
    ├─→ CacheEngine: 根据 max_model_len 计算 Block 数量
    │       num_blocks = gpu_memory / (block_size * layer * ...)
    │
    ├─→ SchedulerConfig.max_num_seqs = 根据显存自动计算
    │
    └─→ ModelRunner: 设置模型的 position_embedding 长度
```

这就是为什么修改一个参数可能会影响多个组件的行为——它们都从 `EngineArgs` 读取配置。

---

## 4.4 关键模块深度解读

### Scheduler： Continuous Batching 的大脑

位置：`core/scheduler.py`

Scheduler 是 vLLM 调度系统的核心，实现了第三章将详细讲解的 Continuous Batching 逻辑。这里先了解它的接口：

```python
class Scheduler:
    """请求调度器"""
    
    def __init__(self, scheduler_config, cache_engine, ...):
        self.waiting: List[SequenceGroup] = []   # 等待队列
        self.running: List[SequenceGroup] = []   # 运行中队列
        self.finished: List[SequenceGroup] = []  # 完成队列
        self.block_manager: BlockSpaceManager     # PagedAttention 块管理器
    
    def schedule(self) -> SchedulerOutput:
        """
        核心调度方法 —— 每个 iteration 调用一次
        
        返回 SchedulerOutput，包含:
        - decided_seq_groups: 本轮要执行的序列
        - num_lookahead_slots: 预分配 slot 数
        - preempted: 被抢占的序列
        """
        # 1. 从 waiting 中挑选可以加入 running 的请求
        # 2. 检查是否需要 preempt 正在运行的请求
        # 3. 构建并返回 SchedulerOutput
        ...
    
    def free_finished(self, seq_group):
        """释放已完成序列的所有 Block"""
        self.block_manager.free(seq_group)
```

### BlockSpaceManager：PagedAttention 的内存管家

位置：`core/block_manager.py`

这是 PagedAttention 的具体实现——管理所有 Block 的分配、释放和映射：

```python
class BlockSpaceManager:
    """PagedAttention 块空间管理器"""
    
    def __init__(self, block_size, num_gpu_blocks, num_cpu_blocks):
        self.block_size = block_size        # 默认 16 tokens/block
        self.gpu_allocator: BlockAllocator  # GPU Block 分配器
        self.cpu_allocator: BlockAllocator  # CPU Block 分配器（swap 用）
    
    def can_allocate(self, seq_group) -> bool:
        """检查是否有足够的空闲 Block 来容纳新序列"""
        required_blocks = calc_required_blocks(seq_group, self.block_size)
        return self.gpu_allocator.get_free_blocks() >= required_blocks
    
    def allocate(self, seq_group):
        """为新序列分配 Block"""
        blocks = self.gpu_allocator.allocate(num_blocks)
        seq_group.block_table = [b.block_id for b in blocks]
    
    def free(self, seq_group):
        """释放序列占用的所有 Block（归还到池中）"""
        for block_id in seq_group.block_table:
            self.gpu_allocator.free(block_id)
    
    def swap(self, seq_group, dest: Literal["cpu", "gpu"]):
        """将序列的 KV Cache 在 GPU/CPU 之间交换（Preemption 时使用）"""
        ...
```

### ModelRunner：模型执行者

位置：`worker/model_runner.py`

ModelRunner 负责：
1. 将 HuggingFace 模型权重加载到 GPU
2. 执行每一轮的 forward pass（模型前向传播）
3. 采样输出下一个 token

```python
class ModelRunner:
    """模型执行器"""
    
    def __init__(self, model_config, ...):
        self.model = self._load_model()        # 加载模型权重
        self.block_tables: Tensor = None        # 当前 batch 的 Block Table
    
    def execute_model(self, scheduler_output) -> SamplerOutput:
        """
        执行一轮模型推理
        
        Args:
            scheduler_output: Scheduler 决定的本批次信息
            
        Returns:
            SamplerOutput: 包含生成的 next token probabilities
        """
        # 1. 准备输入（从 Block Table 收集 KV Cache）
        # 2. 执行 model.forward()（含 PagedAttention 计算）
        # 3. 采样得到 next token
        # 4. 返回结果
        ...
```

---

## 4.5 日志系统与调试技巧

### 日志级别控制

vLLM 使用 Python 标准 logging 模块。通过环境变量控制日志详细程度：

```bash
# 只显示 INFO 及以上（默认）
VLLM_LOGGING_LEVEL=INFO python -m vllm.entrypoints.openai.api_server ...

# 显示 DEBUG 信息（排查问题时非常有用）
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server ...

# 关闭大部分日志（高吞吐生产环境推荐）
VLLM_LOGGING_LEVEL=WARNING python -m vllm.entrypoints.openai.api_server ...
```

### 关键日志关键字段

当你在排查问题时，以下日志字段是最有用的：

| 日志关键词 | 出现位置 | 含义 |
|:---|:---|:---|
| `Loading model weights` | 启动阶段 | 模型加载进度 |
| `GPU blocks` / `CPU blocks` | 启动完成时 | PagedAttention Block 分配情况 |
| `KV cache memory` | 启动完成时 | KV Cache 占用的显存大小 |
| `Maximum concurrency` | 启动完成时 | 理论最大并发数 |
| `Exception during inference` | 运行时 | 推理异常（通常伴随 traceback） |
| `Prefill batch` | 每次请求 | Prompt 处理阶段 |
| `Decode batch` | 每次请求 | Token 生成阶段 |
| `Finished request` | 请求结束时 | 请求完成统计 |

### 常用调试技巧

```python
# 技巧一：查看当前 Engine 状态
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
engine = llm.llm_engine

print(f"已用 GPU Blocks: {engine.cache_engine.gpu_cache.num_used}")
print(f"空闲 GPU Blocks: {engine.cache_engine.gpu_cache.num_free}")
print(f"Running 序列数: {len(engine.scheduler.running)}")
print(f"Waiting 序列数: {len(engine.scheduler.waiting)}")

# 技巧二：打印完整的 EngineArgs 配置
import json
args_dict = engine.engine_args.to_dict()
print(json.dumps(args_dict, indent=2, default=str))

# 技巧三：单步执行观察 Scheduler 行为
output = engine.scheduler.schedule()
print(f"本轮决定执行 {len(output.decided_seq_groups)} 个序列")
print(f"本轮抢占 {len(output.preempted)} 个序列")
```

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **目录结构** | `entrypoints/`(入口) → `core/`(引擎+调度+注意力) → `worker/`(模型执行) → `models/`(模型适配) → `distributed/`(分布式) |
| **两种模式** | **API Server**（HTTP 服务，FastAPI+Uvicorn，生产在线）vs **离线推理**（`LLM.generate()`，Python 内调用，批量处理） |
| **共享引擎** | 两种模式底层都是同一个 `LLMEngine` → Scheduler → ModelRunner → PagedAttention |
| **EngineArgs** | 所有参数的数据类容器，是配置系统的枢纽；`--max-model-len` 经由 EngineArgs 影响多个组件 |
| **Scheduler** | 三队列状态机（WAITING → RUNNING → FINISHED）+ Preemption 抢占机制 |
| **BlockSpaceManager** | PagedAttention 的内存管家：allocate/free/swap 三大操作 |
| **ModelRunner** | 加载权重 + 执行 forward pass + 采样的执行者 |
| **日志调试** | `VLLM_LOGGING_LEVEL=DEBUG` 查看详细信息；关注 GPU Blocks / KV Cache / Prefill/Decode 等关键字 |

> **一句话总结**：vLLM 的代码架构遵循清晰的分层设计——**API 层（FastAPI）→ 引擎层（LLMEngine 总协调）→ 调度层（Scheduler 决定谁该跑）→ 执行层（ModelRunner 真正跑模型）→ 内存层（BlockSpaceManager 管 PagedAttention Block）**。理解了这个层次图，后续阅读源码或排查问题就能快速定位："这个行为属于哪一层？应该去哪个模块找答案？"
