# vLLM 教程大纲

---

## 总体设计思路

**vLLM 是目前最主流的高性能 LLM 推理与服务引擎**——它以 PagedAttention（分页注意力）和 Continuous Batching（连续批处理）两项核心创新，将 LLM 推理吞吐量提升了 2-4 倍，同时将 GPU 显存利用率从 50-60% 提升到 90%+。如果说 Ollama 是"让普通人跑大模型"的瑞士军刀，那 vLLM 就是"让企业级服务跑好大模型"的工业级引擎。

本教程的设计遵循以下原则：

1. **从"为什么选 vLLM"到"如何调到极致性能"**：先讲清 PagedAttention 和连续批处理的原理优势，再深入每个调优细节
2. **从单卡启动到分布式集群**：从 `python -m vllm.entrypoints.openai.api_server` 一行命令开始，逐步进阶到多 GPU 张量并行、多节点部署、Kubernetes 编排
3. **覆盖全场景**：个人研究 / 团队内部服务 / 企业生产环境 / 多模态 / LoRA 微调模型服务化
4. **实战导向 + 面试深度**：每节都有可直接运行的代码和命令示例，同时讲透底层原理（面试官最爱问 PagedAttention 的虚拟内存类比）

### vLLM vs Ollama vs TGI 定位对比

| 维度 | Ollama | vLLM | TGI (HuggingFace) |
|------|--------|------|-------------------|
| **目标用户** | 个人开发者 / 小团队 | 算法工程师 / MLOps / 平台团队 | HF 生态用户 |
| **上手难度** | ⭐ 极低 | ⭐⭐⭐ 中等 | ⭐⭐⭐ 中等 |
| **推理性能** | 基准（llama.cpp 后端） | **最高**（PagedAttention） | 高 |
| **吞吐量** | 低（串行/有限并行） | **极高**（连续批处理） | 高 |
| **显存利用率** | ~60% | **~95%+**（PagedAttention） | ~80% |
| **API 兼容性** | OpenAI 兼容 | **原生 OpenAI 兼容** | OpenAI 兼容 |
| **量化支持** | GGUF 系列 | AWQ/GPTQ/FP8/SqueezeLLM | AWQ/GPTQ/Bitsandbytes |
| **LoRA 服务** | 基础支持 | **原生支持（--enable-lora）** | 支持 |
| **分布式推理** | 不支持 | **Tensor Parallelism + Pipeline Parallelism** | 支持 |
| **多模态** | LLaVA 等（通过 llama.cpp） | **VLM 原生支持（Llava-OneVision/Qwen2-VL）** | 有限 |
| **适用场景** | 本地体验 / 开发测试 | **生产级高并发服务** | HF 模型快速部署 |

---

## 第01章：vLLM 入门与核心理念（4节）

### 定位
面向第一次接触 vLLM 的开发者。解决"我已经会用 Ollama 跑模型了，为什么还需要 vLLM？它能带来什么不同？"这个根本问题。

### 01-01 为什么需要 vLLM？（与 Ollama/TGI 对比）
- **传统 LLM 推理的两大痛点**：
  - 显存浪费：KV Cache 按最大序列长度预分配 → 大量碎片（像操作系统早期固定分区内存管理）
  - 吞吐瓶颈：静态批处理（Static Batching）→ 必须等齐所有请求才开始 → 首个 Token 延迟（TTFT）飙升
- **vLLM 的两个核心创新**：
  - **PagedAttention**：借鉴 OS 虚拟内存的分页思想，将 KV Cache 切分为固定大小的 Block，按需分配/释放 → 显存利用率 90%+
  - **Continuous Batching**：边处理边接收新请求 → 类似地铁列车"行驶中上下客"→ TTFT 降低 2-5x
- **性能数据说话**：
  - 与 HuggingFace Transformers 对比：吞吐量提升 **14-24x**（官方 benchmark）
  - 与 TGI 对比：吞吐量提升 **1.5-3x**
  - 显存节省：相同 batch size 下少用 **30-60%** 显存
- **什么时候该用 vLLM？**
  - ✅ 需要 >10 并发请求的生产服务
  - ✅ 需要最大化 GPU 利用率
  - ✅ 需要服务 LoRA 微调后的模型
  - ❌ 只需要偶尔跑一跑对话（用 Ollama 更方便）
- **第一个体验**：`pip install vllm` → `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct`

### 01-02 安装与环境准备
- **系统要求**：
  - Linux（推荐 Ubuntu 20.04+ / CentOS 7+）
  - NVIDIA GPU（CUDA 11.8+ / 12.1+ 推荐）
  - Python 3.9-3.12
  - 最小显存要求：取决于模型（7B 至少 16GB，70B 至少 4×A100 80GB）
- **安装方式**：
  - 标准安装：`pip install vllm`（自动匹配 CUDA 版本）
  - 指定 CUDA 版本：`VLLM_USE_CUDA=1 pip install vllm`
  - 从源码编译（高级）：`git clone` → `python setup.py install`
  - Docker 安装：`vllm/vllm-openai:latest` 官方镜像
- **GPU 驱动与 CUDA 环境**：
  - `nvidia-smi` 验证驱动版本（≥ 525.60.13）
  - `nvcc --version` 验证 CUDA 工具链
  - PyTorch + CUDA 版本匹配表
- **常见安装问题排查**：
  - CUDA 版本不匹配 → `pip install --force-reinstall`
  - 缺少依赖库 → `apt-get install build-essential`
  - Docker 中 GPU 不可见 → NVIDIA Container Toolkit 配置
  - Apple Silicon 支持（Metal MPS 后端，实验性）
- **验证安装成功**：`python -c "import vllm; print(vllm.__version__)"`

### 01-03 第一个模型服务：启动与调用
- **最简启动命令**：
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
  ```
- **关键参数速览**：
  - `--model`：HuggingFace 模型 ID 或本地路径
  - `--tensor-parallel-size`：张量并行 GPU 数量
  - `--max-model-len`：最大上下文长度
  - `--gpu-memory-utilization`：GPU 显存使用率上限（默认 0.9）
  - `--dtype`：权重数据类型（auto/half/bfloat16/float16）
  - `--quantization`：量化方式（awq/gptq/fp8/squeezellm）
  - `--enable-lora`：启用 LoRA 适配器加载
- **OpenAI 兼容 API 调用**：
  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [...]}'
  ```
- **Python openai SDK 调用**：
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")
  response = client.chat.completions.create(model="...", messages=[...])
  ```
- **流式输出（SSE）**：`stream=True` 参数的使用
- **支持的模型列表**（截至最新版）：
  - Llama 3.1 / 3.2 / 3.3 系列
  - Qwen2.5 系列（含 VL 多模态）
  - Mistral / Mixtral 系列
  - DeepSeek 系列
  - Yi / InternLM / Baichuan 等国产模型
  - Gemma 2 / Phi-3 等小模型

### 01-04 vLLM 项目结构与代码入口
- **项目目录结构概览**：`vllm/` 下核心模块
  - `entrypoints/openai/api_server.py` — OpenAI 兼容服务器入口
  - `entrypoints/llm/` — 直接 LLM 推理入口
  - `core/` — 核心引擎（scheduler/engine/attention）
  - `worker/` — 分布式 worker 管理
  - `model_executor/` — 模型执行器
- **两种运行模式**：
  - **API Server 模式**：作为 HTTP 服务对外提供 OpenAI 兼容接口（生产首选）
  - **离线推理模式**：`LLM.generate()` 直接在 Python 中批量生成（批量处理/评估场景）
- **配置系统**：EngineArgs 解析流程
- **日志与调试**：`VLLM_LOGGING_LEVEL` 环境变量

---

## 第02章：PagedAttention 核心原理深度解析（4节）

### 定位
这是 vLLM 的灵魂所在。理解 PagedAttention 不仅有助于用好 vLLM，更是 LLM 推理优化领域的**必考面试题**。本章将从 OS 虚拟内存的经典类比出发，逐层拆解 PagedAttention 的设计思想、实现机制和性能收益。

### 02-1 传统 Attention 的 KV Cache 问题
- **Transformer Attention 回顾**：
  - Self-Attention 计算公式：$Attention(Q,K,V) = softmax(QK^T/\sqrt{d_k})V$
  - KV Cache 的作用：缓存之前 token 的 K 和 V 向量，避免重复计算
  - KV Cache 形状：`(num_layers, 2, num_kv_heads, seq_len, head_dim)`
- **KV Cache 内存增长规律**：
  - 每个 token 新增的 KV Cache 大小 ≈ `2 × num_layers × hidden_size × dtype_bytes`
  - 以 Llama 3 8B 为例：每 token 新增 ~ 2 MB（fp16），4096 context = ~8 GB
  - 公式推导与各模型对照表
- **传统方案的内存管理缺陷**：
  - **预分配策略**：按 `max_seq_len` 预分配 KV Cache → 浪费严重
    - 例：max_seq_len=8192，但平均只用 1024 → 87.5% 显存浪费
  - **外部碎片化**：不同请求序列长度不一 → 无法紧凑排列 → 内存空洞
  - **内部碎片化**：即使预分配了固定大小，实际未填满的部分也是浪费
- **类比**：就像操作系统的**固定分区内存管理**（1950s 方案）vs **分页虚拟内存**（1960s 方案）

### 02-2 PagedAttention 设计思想
- **核心灵感来源**：OS 的 Virtual Memory / Paging 系统
  - 物理内存被划分为固定大小的 Page（如 4KB）
  - 进程的逻辑地址空间映射到离散的物理 Page
  - Page Table 维护映射关系
  - 按需分配 / 交换淘汰
- **PagedAttention 映射到 LLM 场景**：
  | OS 概念 | PagedAttention 对应 |
  |--------|---------------------|
  | Physical Memory | GPU VRAM (显存) |
  | Page (4KB) | **Block** (固定大小，如 16 tokens) |
  | Process Address Space | 一个 Sequence (请求) 的 KV Cache |
  | Page Table | **Block Table** (记录逻辑块→物理块的映射) |
  | Page Fault | **Block 分配请求** (新 token 到来时) |
  | Page Eviction | **Block 回收** (序列结束时) |
  | Memory Fragmentation | **几乎为零** (固定大小 Block) |
- **Block 的设计参数**：
  - 默认 Block Size = 16 tokens（可配置 `--block-size`）
  - 每个 Block 存储连续的 16 个 token 的 KV Cache
  - 一个序列占用 N 个 Block = ceil(seq_len / block_size)
- **Block Manager（块管理器）**：
  - Block Pool：预分配的所有可用 Block 组成的池
  - 分配算法：新序列到来时从池中取空闲 Block
  - 释放算法：序列结束（或被 preempted）时归还 Block
  - 内存拷贝优化：PagedAttention 的 **Memory Copy-on-Write**（共享前缀时避免重复存储）

### 02-3 PagedAttention 实现细节
- **Block Table 数据结构**：
  - 每个序列维护一个 Block Table：`list[int]`（物理 block 编号列表）
  - 示例：序列长度 50，block_size=16 → 占用 4 个 block [3, 7, 12, 21]
- **Attention 计算中的改动**：
  - 传统方式：直接对连续的 KV Cache 做 matmul
  - PagedAttention 方式：根据 Block Table 收集分散的 Block → 拼接（或软拼接）→ 计算
  - **关键优化**：不需要真的物理拼接！用 indexing 操作即可
- **Prefix Caching（前缀缓存共享）**：
  - 多个请求有相同的 System Prompt 时 → 共享前缀对应的 Block
  - 通过引用计数（refcount）管理共享 Block
  - 这是 vLLM 在 **Prompt 处理阶段加速的关键技术**
- **PagedAttention 的 CUDA Kernel 实现**：
  - 自定义 `paged_attention_v1/v2` kernel
  - 相比 PyTorch 原生 attention 的性能提升分析
  - FlashAttention vs PagedAttention 的关系（互补而非替代）
- **代码走读**：核心类 `BlockSpaceManager` 和 `CacheEngine` 的职责划分

### 02-4 性能收益量化分析
- **显存利用率提升**：
  - 传统方案：~50-65%（大量预留空间 + 外部碎片）
  - PagedAttention：~90-98%（仅剩最后一个 Block 的内部碎片）
  - 数学证明：最大内部碎片 = (block_size - 1) / avg_seq_len × 100%
- **吞吐量提升实测数据**：
  - 同硬件、同模型下 vLLM vs HuggingFace 的 requests/sec 对比
  - 不同 batch size 下的延迟分布变化
  - 长短混合场景下的优势放大
- **支持的最大并发数**：
  - 传统方案受限于 max_seq_len × batch_size ≤ 显存总量
  - PagedAttention 受限于 Block Pool 总数（灵活得多）
- **代价是什么？**
  - 额外的 Block Table 管理开销（< 1%，可忽略）
  - 非连续内存访问导致的 cache miss 增加（约 2-5% kernel 效率损失）
  - 总体净收益：显著正向（吞吐量提升远超开销）

---

## 第03章：Continuous Batching 与调度系统（4节）

### 定位
PagedAttention 解决了"存不下"的问题，Continuous Batching 解决了"不够快"的问题。两者协同工作构成了 vLLM 的性能护城河。

### 03-1 从 Static Batching 到 Continuous Batching
- **Static Batching（静态批处理）的问题**：
  - 工作流：收集满一批请求 → padding 到统一长度 → 送入模型 → 全部完成 → 下一批
  - 问题 1：**首 Token 延迟（TTFT）灾难**——早到的请求必须等待晚到的请求凑齐
  - 问题 2：**Padding 浪费**——短序列填充到最长序列长度 → 无效计算
  - 问题 3：**Tail Latency**——一个慢请求拖慢整批
  - 生活类比：**公交车必须坐满才发车** → 早到的乘客干等
- **Iterative Batching（迭代批处理）的改进**：
  - 每次只生成一步（一个 token），然后重新组队
  - 但仍然需要在每步之间做复杂的 padding/unpadding
- **Continuous Batching（连续批处理）**：
  - 工作流：当前批次正在生成中 → 有新请求到达 → 立即加入批次 → 正在运行的请求不受影响
  - 生活类比：**地铁列车行驶中乘客可以随时上下车**
  - 关键特性：
    - 迭代级别调度（iteration-level scheduling）
    - 动态批次组成（每轮 iteration 的 batch 成员可能不同）
    - 无需 padding（每个序列独立维护自己的位置）
- **vLLM 的 Scheduler 实现**：

### 03-2 vLLM Scheduler 工作机制
- **Scheduler 核心循环**（每个 iteration）：
  ```
  1. WAITING 队列中的请求 → 检查资源是否足够（Block 数量）
     → 若足够则移至 RUNNING 队列
  2. RUNNING 队列中的请求 → 执行一步 forward pass（生成一个 token）
  3. 检查 RUNNING 请求是否完成（达到 stop 条件或 max_tokens）
     → 完成的移至 FINISHED 队列，释放 Block
  4. 检查是否有 RUNNING 请求需要 Preemption（抢占）
     → 长序列优先级降低，为新请求腾出空间
  5. 重复以上步骤
  ```
- **三种状态的生命周期**：
  - `WAITING`：已接收但尚未开始推理（排队中）
  - `RUNNING`：正在进行推理（每 step 生成一个 token）
  - `FINISHED`：已完成或被中断
- **调度策略**：
  - FCFS（First-Come First-Served）：基础策略
  - Priority Scheduling：基于请求优先级的调度
  - Preemption（抢占）：当资源不足时暂停长序列，腾出空间给新请求
- **关键参数**：
  - `--max-num-batched-tokens`：每 iteration 最大处理的 token 数
  - `--max-num-seqs`：同时运行的最大序列数
  - `--scheduler-delay-factor`：调度延迟因子（等待更多请求以增加 batch size）

### 03-3 Preemption 与资源管理
- **为什么需要抢占（Preemption）？**
  - 场景：Block Pool 已满，但有新的高优先级请求到达
  - 选择：拒绝新请求 OR 暂停某个正在运行的低优先级请求
  - vLLM 选择后者（Preemption）
- **Preemption 的实现**：
  - **SWAP OUT**：将被抢占序列的 KV Cache 从 GPU 移到 CPU 内存
  - **SWAP IN**：当资源充足时再从 CPU 恢复回 GPU
  - 这就是 vLLM 的 **CPU Offload** 机制
- **Preemption 策略选择**：
  - 抢占谁？通常选择最长的序列（占 Block 最多）
  - 抢多少？只释放足够新请求使用的 Block 数量
  - 何时恢复？Block Pool 有足够空闲时自动 swap-in
- **资源限制参数详解**：
  - `--gpu-memory-utilization`：KV Cache 可用的 GPU 显存比例
  - `--swap-space`：CPU 上用于 swap 的空间大小（GB，默认 4）
  - `--max-num-seqs`：硬性限制最大并发序列数
- **OOM 防护**：vLLM 如何优雅地处理显存耗尽情况

### 03-4 调度性能调优实战
- **调度延迟 vs 吞吐量的权衡**：
  - `--scheduler-delay-factor` 设为 0：立即调度 → 高 TTFT，低 throughput
  - 设为较大值（如 0.5）：等待更多请求 → 较低 TTFT，高 throughput
  - 推荐设置：交互场景 0.1-0.2，批处理场景 0.3-0.5
- **监控调度状态**：
  - Metrics 端点查看当前 RUNNING/WAITING/FINISHED 序列数
  - Block Pool 使用率实时监控
  - Preemption 发生频率统计
- **典型场景调优案例**：
  - 场景 A：在线聊天（低延迟优先）→ `--max-num-seqs 64`, `--scheduler-delay-factor 0.1`
  - 场景 B：离线批量生成（高吞吐优先）→ `--max-num-seqs 256`, `--scheduler-delay-factor 0.5`
  - 场景 C：混合负载（长短请求混合）→ 启用 swap + 合理 preemption 策略
- **Benchmark 方法**：使用 vLLM 内置 benchmark 工具测量不同配置下的 P99 延迟和吞吐

---

## 第04章：OpenAI 兼容 API 服务（5节）

### 定位
vLLM 最大的工程价值之一是提供了**开箱即用的 OpenAI 兼容 HTTP API**。这意味着任何已经对接过 OpenAI API 的应用（LangChain、LlamaIndex、Next.js AI ChatGPT 等），只需修改 `base_url` 就能无缝切换到 vLLM。

### 04-1 API Server 启动与配置
- **完整启动命令模板**：
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --disable-log-requests
  ```
- **全部启动参数分类整理**：
  - 模型相关：model / dtype / trust-remote-code / download-dir / load-format
  - 并行相关：tensor-parallel-size / pipeline-parallel-size / worker-use-ray
  - 性能相关：max-model-len / gpu-memory-utilization / max-num-seqs / scheduler-delay-factor
  - 功能相关：enable-lora / enable-auto-tool-choice / chat-template-path
  - 服务相关：host / port / ssl-keyfile / ssl-certfile / disable-log-requests
- **配置文件方式**：YAML/TOML 配置替代超长命令行
- **健康检查端点**：GET `/health` 返回服务状态
- **模型信息端点**：GET `/v1/models` 返回已加载模型列表

### 04-2 Chat Completions API 深度指南
- **接口定义**：POST `/v1/chat/completions`
- **请求体完整字段说明**：
  - `model`：模型名称（必须）
  - `messages`：消息数组（role: system/user/assistant/tool）
  - `stream`：是否流式输出（bool）
  - `temperature / top_p`：采样参数
  - `max_tokens`：最大生成 token 数
  - `stop`：停止序列
  - `presence_penalty / frequency_penalty`：惩罚系数
  - `logprobs / top_logprobs`：返回概率信息
  - `tools` / `tool_choice`：Function Calling 支持
  - `response_format`：JSON Mode / JSON Schema 约束
- **响应体结构详解**：
  - 非流式：`choices[0].message.content` / `finish_reason` / `usage`
  - 流式（SSE）：`data: {"choices":[{"delta":{"content":"..."}}]}` / `[DONE]`
  - `usage.prompt_tokens / completion_tokens / total_tokens`
- **Function Calling（工具调用）**：
  - vLLM 对 tools 参数的原生支持
  - `tool_choice: "auto"` / `"none"` / `{"type": "function", "function": {"name": "..."}}`
  - 响应中的 `tool_calls` 字段解析
  - 支持工具调用的模型列表（Llama 3.1/3.2/3.3, Qwen 2.5, Mistral 等）
- **完整 curl 示例集**（普通对话 / 流式 / Function Calling / JSON Mode）

### 04-3 Completions & Embeddings API
- **Completions API**（补全式）：POST `/v1/completions`
  - 适用场景：纯文本续写 / 代码补全 / 非对话式任务
  - `prompt` 字段（字符串或字符串数组）
  - Echo 参数（返回 prompt 作为 response 前缀）
  - Logprob 返回（每个 token 的概率分布）
- **Embeddings API**：POST `/v1/embeddings`
  - vLLM 原生支持 Embedding 模型服务化
  - 支持的 Embedding 模型（BGE / E5 / GTE / Mxbai 等）
  - 输出维度与归一化选项
  - 批量嵌入性能（比单独调用快 10-50x）
- **Tokenization API**：
  - POST `/v1/tokenize`：文本 → token IDs
  - POST `/v1/detokenize`：token IDs → 文本
- **错误码体系**：
  - 400 Bad Request（参数错误 / 模型不存在）
  - 422 Unprocessable Entity（请求格式问题）
  - 500 Internal Error（推理引擎异常）
  - 503 Service Unavailable（模型加载中 / 过载）

### 04-4 Python SDK 集成实战
- **OpenAI SDK 无缝切换**（推荐方式）：
  ```python
  from openai import OpenAI
  client = OpenAI(
      base_url="http://localhost:8000/v1",
      api_key="not-needed-for-vllm"
  )
  # 所有 OpenAI SDK 的方法完全兼容
  resp = client.chat.completions.create(
      model="Qwen/Qwen2.5-7B-Instruct",
      messages=[{"role": "user", "content": "你好"}],
      stream=True
  )
  for chunk in resp:
      print(chunk.choices[0].delta.content or "", end="")
  ```
- **异步客户端（AsyncOpenAI）**：
  - `asyncio` + `AsyncOpenAI` 高并发调用模式
  - 连接池配置与超时控制
- **vLLM 原生 Python 推理接口**（非 Server 模式）：
  ```python
  from vllm import LLM, SamplingParams
  llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
  outputs = llm.generate(["提示词1", "提示词2"], SamplingParams(temperature=0.7))
  for output in outputs:
      print(output.outputs[0].text)
  ```
  - 离线模式的适用场景：批量评估 / 数据集处理 / A/B 测试
- **流式输出处理**：
  - SSE 协议解析
  - 逐 token 渲染到前端
  - Time-to-First-Token (TTFT) 与 Throughput 的分别测量
- **错误处理与重试**：
  - 指数退避重试策略
  - Circuit Breaker（熔断器）模式

### 04-5 多语言与框架接入速查
- **JavaScript / TypeScript**：openai npm 包直接兼容
- **Go**：github.com/openai/openai-go
- **Java / Spring AI**：Spring AI vLLM 集成配置
- **C# / .NET**：Azure.AI.OpenAI 包指向 vLLM
- **curl / Shell**：纯 bash 脚本调用 + jq 解析
- **LangChain 集成**（详见 Ch07）
- **LlamaIndex 集成**（详见 Ch07）
- **Next.js Vercel AI SDK**：`@ai-sdk/openai-provider` + vLLM base_url

---

## 第05章：离线推理与批量处理（3节）

### 定位
不是所有场景都需要 HTTP API 服务。当你需要对百万条数据进行批量推理（如数据标注、评估打分、embedding 提取）时，vLLM 的离线推理模式能提供最高的效率。

### 05-1 LLM 类直接推理
- **基础用法**：
  ```python
  from vllm import LLM, SamplingParams
  
  # 初始化（一次性加载模型到 GPU）
  llm = LLM(
      model="Qwen/Qwen2.5-7B-Instruct",
      tensor_parallel_size=1,
      gpu_memory_utilization=0.90,
      max_model_len=8192,
  )
  
  # 采样参数
  sampling_params = SamplingParams(
      temperature=0.7,
      top_p=0.9,
      max_tokens=256,
      stop=["\n\n"],
  )
  
  # 批量推理
  prompts = ["解释量子计算", "写一首关于春天的诗", "Python GIL 是什么"]
  outputs = llm.generate(prompts, sampling_params)
  
  for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      print(f"输入: {prompt}\n输出: {generated_text}\n---")
  ```
- **SamplingParams 完整参数表**：
  - temperature / top_p / top_k / min_p
  - max_tokens / min_tokens
  - presence_penalty / frequency_penalty
  - repetition_penalty / stop / stop_token_ids
  - skip_special_tokens / spaces_between_special_tokens
  - logprobs / prompt_logprobs / detokenize
- **Output 对象详解**：
  - `output.outputs[0].text`：生成的文本
  - `output.outputs[0].token_ids`：生成的 token ID 列表
  - `output.outputs[0].logprobs`：每个 token 的 log probability
  - `output.outputs[0].finish_reason`：完成原因（length/stop/end_id）
  - `output.metrics`：推理耗时统计（prompt_tokens / generation latency 等）

### 05-2 高效批量处理技巧
- **大数据集分批策略**：
  - 一次送入太多 prompts → OOM；太少 → GPU 利用率不足
  - 最优 batch size 探索：二分法找到临界点
  - 动态 batching：根据剩余显存动态调整每批大小
- **进度条与日志**：tqdm 集成 / 自定义 callback
- **结果持久化**：JSON Lines 格式 / Parquet / SQLite
- **错误恢复机制**：
  - 断点续传（记录已完成的 index）
  - 异常捕获与重试
  - 失败样本隔离与后续重跑
- **多进程并行**：
  - 多 GPU 时每个 GPU 跑一个 LLM 实例
  - multiprocessing + Queue 分发任务
  - 或使用 Ray 进行分布式编排
- **性能基准测试脚本**：自动化测量不同 batch size 下的 tokens/sec

### 05-3 特殊推理场景
- **Log Probability 提取**：
  - 用于 Reward Model 训练数据准备
  - 用于自一致性解码（Self-Consistency Decoding）
  - `SamplingParams(logprobs=5)` 返回 top-5 概率
- **Beam Search（束搜索）**：
  - `SamplingParams(n=4, use_beam_search=True)` 
  - 适用场景：翻译 / 摘要 / 需要确定性高质量输出的任务
  - 与采样的区别与选择指南
- **Structured Output（结构化输出）**：
  - JSON Schema 约束输出格式
  - 正则表达式约束（用于特定格式如日期/邮箱）
  - vLLM 的 Guided Decoding 实现
- **Prefix Caching 加速批量推理**：
  - 当多个 prompts 有共同前缀（如同一 System Prompt）
  - 自动共享 KV Cache → Prompt Eval 阶段几乎零成本
  - 如何构造 prompts 以最大化前缀命中率

---

## 第06章：多 GPU 与分布式推理（5节）

### 定位
单个 GPU 跑不动 70B 模型？需要更高的吞吐量？vLLM 的分布式推理能力是其企业级应用的核心竞争力。本章覆盖从单机多卡到多机集群的全部方案。

### 06-1 张量并行（Tensor Parallelism, TP）
- **为什么需要 TP？**
  - 70B 模型的参数量（140B fp16 参数 = 280GB 显存需求）
  - 单卡 A100 80GB 也放不下 → 必须切分
- **TP 原理**：
  - 将模型的每一层按行/列切分到多个 GPU
  - Megatron-LM 式的 Column-Parallel + Row-Parallel Linear
  - Attention 层的多头分割（每个 GPU 负责部分 attention heads）
  - Forward Pass 时的 All-Reduce 通信
- **vLLM 中的 TP 配置**：
  ```bash
  --tensor-parallel-size 4   # 4 卡 TP
  # 或
  --tp 4                     # 简写
  ```
- **通信开销分析**：
  - All-Reduce 带宽需求 ∝ hidden_size × 2 bytes
  - NVLink vs PCIe Gen4/Gen5 的性能差异
  - TP=2 vs TP=4 vs TP=8 的扩展效率（Scaling Efficiency）
- **实践建议**：
  - 7B-14B 模型：TP=1（单卡足够）
  - 32B-70B 模型：TP=2-4（双卡/四卡 A100）
  - 100B+ 模型：TP=4-8（需要 H100/H800 + NVLink）
  - Apple UltraFusion：M-series 芯片的统一内存天然适合大模型

### 06-2 流水线并行（Pipeline Parallelism, PP）
- **PP 原理**：
  - 将模型的层（layers）分组到不同的 GPU（Stage）
  - Micro-batch 在 Stage 之间流水线传递
  - GPipe / PipeDream-Flush 等调度策略
- **vLLM 的 PP 支持**：
  ```bash
  --pipeline-parallel-size 2
  ```
- **TP + PP 组合（2D/3D 并行）**：
  - TP × PP = 所需总 GPU 数
  - 例：TP=2, PP=2 → 4 GPU（比 TP=4 更省通信带宽）
  - 适用场景：跨机器部署（PP 的 stage 可以在不同节点上）
- **Pipeline Bubble 问题**：最后一个 micro-batch 的空闲时间

### 06-3 多节点分布式部署
- **Ray 集成模式**（推荐）：
  ```bash
  # Head node
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --worker-use-ray \
    --ray-address auto
    
  # Worker nodes
  ray start --address="<head-node-ip>:6379"
  ```
  - Ray 自动管理 Worker 注册 / 心跳检测 / 故障恢复
- **手动分布式模式**：
  - 每个节点独立启动 worker process
  - 通过 RPC 通信
  - 适用于 Ray 不可用的环境
- **网络要求**：
  - 节点间带宽 ≥ 25 GbE（InfiniBand 更佳）
  - 防火墙端口开放（RPC 端口范围）
  - SSH 免密登录配置
- **多节点部署 Checklist**：
  - 所有节点安装相同版本的 vllm / torch / cuda
  - 所有节点能访问模型文件（NFS / 共享存储 / 各自下载一份）
  - 时间同步（NTP）
  - GPU 驱动版本一致

### 06-4 推理性能基准测试
- **vLLM 内置 Benchmark 工具**：
  ```bash
  # Throughput benchmark
  python -m vllm.benchmark.benchmark_throughput \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 1000 \
    --output-json result.json
  
  # Latency benchmark (TTFT + TPOT)
  python -m vllm.benchmark.benchmark_serving \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 1000 \
    --request-rate 10
  ```
- **关键指标解读**：
  - **TTFT（Time to First Token）**：首个 Token 延迟（用户体验的核心指标）
  - **TPOT（Time Per Output Token）**：每个输出 Token 的平均间隔（打字速度感）
  - **Throughput（tokens/sec）**：总吞吐量（系统效率指标）
  - **Requests/sec**：请求数/秒（服务容量指标）
- **不同配置的性能矩阵**：
  - TP=1/2/4/8 下的吞吐量和延迟对比
  - 不同 input length + output length 组合
  - 不同 concurrency level 下的表现
- **与竞品对比方法论**：
  - 统一的评测协议（相同硬件 / 相同模型 / 相同 dataset）
  - 公平对比注意事项（warmup / quantization / optimization level）

### 06-5 硬件选型与成本参考
- **单卡方案参考**：
  | GPU | 显存 | 适合模型（FP16） | 参考价格 | 吞吐量（7B QPS） |
  |-----|------|-----------------|---------|-----------------|
  | RTX 4090 | 24GB | ≤ 7B | ¥13,000 | ~30-50 |
  | RTX 6000 Ada | 48GB | ≤ 14B | ¥45,000 | ~40-70 |
  | A5000 | 48GB | ≤ 14B | ¥35,000 | ~35-60 |
  | A100 80GB | 80GB | ≤ 34B | ¥140,000 | ~60-100 |
  | H100 80GB | 80GB | ≤ 34B | ¥280,000 | ~120-200 |
  | H200 141GB | 141GB | ≤ 72B | ¥400,000+ | ~150-250 |
- **多卡方案参考**：
  - 双 A100 80GB：70B 模型 TP=2，吞吐 ~40-80 req/s
  - 四 H100 80GB：70B 模型 TP=4，吞吐 ~80-150 req/s
  - 八 H100 80GB：70B+ 模型 TP=8，吞吐 ~150-300 req/s
- **云 GPU 方案**：
  - AWS p4d.24xlarge（8×A100 80GB）：~$32/hour
  - GCP A2-HIGHGPU-8G（8×A100 80GB）：~$28/hour
  - AutoDL / 阿云 PAI / 等国内平台性价比对比
- **Apple Silicon 方案**：
  - Mac Studio M2 Ultra 192GB：可跑 70B Q4，速度约 15-25 tok/s
  - 适合开发和小规模服务，不适合高并发生产

---

## 第07章：量化技术（4节）

### 定位
量化是将 FP16/BF16 权重转换为更低精度（INT8/INT4/FP8）的过程，能在几乎不损失质量的前提下减少 2-4 倍的显存占用并加速推理。vLLM 支持业界主流的量化方案。

### 07-1 量化基础知识回顾
- **为什么要量化？**
  - 显存占用：FP16 每 parameter 2 bytes → INT4 仅 0.5 bytes（4x 减少）
  - 计算加速：INT8/INT4 矩阵乘法比 FP16 快 2-4x（利用 Tensor Core）
  - 内存带宽瓶颈缓解：传输的数据量减少 → 受限场景下提速
- **量化类型分类**：
  - **训练后量化（PTQ）**：在预训练模型上直接量化（AWQ/GPTQ/FP8）
  - **量化感知训练（QAT）**：训练过程中模拟量化误差
  - **混合精度**：不同层使用不同精度（Attention 用 FP16，FFN 用 INT4）
- **精度 vs 性能权衡曲线**：
  - FP16 → BF16 → FP8 → INT8 → INT4（质量递减，速度递增）
  - "够用就好"原则：大多数任务 INT4/AWQ 足够
- **vLLM 支持的量化格式一览**：
  | 格式 | 精度 | 显存压缩比 | 速度提升 | vLLM 参数 |
  |------|------|-----------|---------|----------|
  | AWQ (Activation-aware) | INT4 | ~4x | 2-3x | `--quantization awq` |
  | GPTQ | INT4 | ~4x | 2-3x | `--quantization gptq` |
  | FP8 | 8-bit Float | ~2x | 1.5-2x | `--quantization fp8` |
  | SqueezeLLM | INT4 | ~4x | 2-3x | `--quantization squeezellm` |
  | BitsAndBytes | NF4/INT8 | ~4x/2x | 1.5-2x | `--quantization bitsandbytes` |

### 07-2 AWQ（Activation-Aware Weight Quantization）
- **AWQ 原理**：
  - 观察 activations 的幅度分布 → 保护重要通道（channels with large activations）不被过度量化
  - 只量化每层 1% 的 salient weights 保持 FP16 → 其余量化为 INT4
  - 相比 GPTQ：同等精度下 AWQ 通常效果更好
- **使用方法**：
  ```bash
  # 使用 HuggingFace 上的 AWQ 量化模型
  python -m vllm.entrypoints.openai.api_server \
    --model casperhao/Meta-Llama-3.1-8B-Instruct-AWQ \
    --quantization awq \
    --port 8000
  ```
- **自己转换 AWQ 模型**：
  - 使用 `autoawq` 库进行量化
  - 校准数据集的选择（领域相关文本效果更好）
  - 转换后的模型上传到 HuggingFace / 本地路径
- **AWQ 模型质量评估**：
  - 与原模型在标准 benchmark 上的对比（MMLU / HumanEval / C-Eval）
  - 主观评估（A/B test）

### 07-3 GPTQ 与其他量化方案
- **GPTQ（Group-wise Post Training Quantization）**：
  - 基于 Optimal Brain Quantization 的近似求解
  - 按 group（通常 128 列为一组）独立量化
  - 生态成熟：HuggingFace 上 GPTQ 模型数量最多
  - 使用方法：`--quantization gptq`
- **FP8 量化**：
  - Hopper 架构（H100/H800）原生支持 FP8 Tensor Core
  - 推理速度接近 FP16 精度，显存减半
  - 需要较新的 CUDA 版本和驱动
  - 使用方法：`--quantization fp8`
- **SqueezeLLM**：
  - 结合稀疏性 + 量化双重优化
  - 对已有稀疏性的模型特别有效
- **BitsAndBytes (NF4)**：
  - NormalFloat 4-bit 数据类型
  - 信息论最优的均匀量化
  - 与 HuggingFace PEFT/QLoRA 生态完美配合
- **如何选择量化方案？**
  - 追求最佳 INT4 质量 → AWQ
  - 模型生态丰富度优先 → GPTQ
  - H100 硬件 + 最高速度 → FP8
  - 与微调生态联动 → BitsAndBytes

### 07-4 量化模型性能对比与选型
- **统一 Benchmark 方法论**：
  - 固定硬件（如 1×A100 80GB）
  - 固定模型（如 Llama 3.1 8B Instruct）
  - 统一测试集（MT-Bench / AlpacaEval 2.0）
  - 测量维度：质量分数 / TTFT / TPOT / Throughput / 显存占用 / 首次加载时间
- **典型结果参考**（Llama 3.1 8B 在 A100 上）：
  | 格式 | 显存占用 | TTFT (P50) | TPOT | Throughput | MT-Bench |
  |------|---------|-----------|------|-----------|----------|
  | FP16 | ~16GB | 450ms | 45ms | 120 tok/s | 8.2 |
  | AWQ INT4 | ~5GB | 280ms | 28ms | 180 tok/s | 7.9 |
  | GPTQ INT4 | ~5GB | 290ms | 29ms | 175 tok/s | 7.8 |
  | FP8 | ~8GB | 380ms | 38ms | 145 tok/s | 8.1 |
  | NF4 (BnB) | ~5GB | 300ms | 30ms | 170 tok/s | 7.7 |
- **量化对不同任务类型的影响**：
  - 数学推理：对精度最敏感 → 建议 FP8 或 AWQ
  - 通用对话：INT4 几乎无损
  - 代码生成：中等敏感 → AWQ 优于 GPTQ
  - 长文档理解：影响较小
- **成本效益分析**：
  - 量化后可以用更小的 GPU 跑同样大小的模型
  - 或者同样的 GPU 跑更大的 batch size → 吞吐翻倍
  - 量化本身的计算成本（一次性）vs 长期推理收益

---

## 第08章：LoRA 适配器服务化（3节）

### 定位
LoRA（Low-Rank Adaptation）是目前最高效的大模型微调方式——只需要训练不到 1% 的参数就能让基础模型获得领域能力。vLLM 原生支持加载和服务化 LoRA 适配器，无需合并权重即可高效推理。

### 08-1 LoRA 基础回顾与 vLLM 支持
- **LoRA 原理速查**：
  - 在预训练权重的每个 Linear 层旁添加低秩分解矩阵 A 和 B
  - 冻结原始权重，只训练 A 和 B（参数量仅为原始的 0.1%-1%）
  - 推理时：$W' = W + BA$（原始权重 + 低秩增量）
  - 训练成本极低（单卡数小时），效果接近全量微调
- **vLLM 的 LoRA 支持**：
  - `--enable-lora` 启用 LoRA 功能
  - `--lora-modules <name>=<path>` 指定适配器路径
  - 运行时动态切换 LoRA（无需重启服务）
  - API 层面通过 `model` 参数指定 lora adapter（格式：`base-model@lora-name`）
- **支持的 LoRA 格式**：
  - HuggingFace PEFT 格式（adapter_config.json + adapter_model.safetensors）
  - Ptuning-v2 格式（部分支持）
  - 多个 LoRA 同时加载（`--max-loras` / `--max-lora-size`）
- **LoRA 的显存开销**：
  - 每个 LoRA adapter 约 10-100MB（取决于 rank 和 target modules）
  - Base model 加载一次，LoRA 按需切换 → 极低的额外显存
  - 支持同时加载数十个 LoRA adapter

### 08-2 LoRA 服务化实战
- **启动带 LoRA 的 vLLM 服务**：
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules medical-lora=/path/to/medical-adapter \
                   legal-lora=/path/to/legal-adapter \
                   code-lora=/path/to/code-adapter \
    --max-loras 10 \
    --max-lora-size 128
  ```
- **API 调用指定 LoRA**：
  ```bash
  # 调用 medical-lora 适配器
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3.1-8B-Instruct@medical-lora",
      "messages": [{"role": "user", "content": "诊断以下症状"}]
    }'
  
  # 切换到 legal-lora（同一个请求换 model 名即可）
  ```
- **列出可用 LoRA**：GET `/v1/lora` 返回已注册的 LoRA 列表
- **动态添加/删除 LoRA**：
  - POST `/v1/lora/register` 运行时注册新 LoRA
  - DELETE `/v1/lora/{lora_name}` 卸载 LoRA 释放显存
- **Python 完整示例**：
  - LoRA 热切换客户端封装
  - 多租户场景（不同客户用不同 LoRA）
  - A/B Testing 框架（同一请求同时发给 base + lora，对比结果）

### 08-3 LoRA 最佳实践与性能优化
- **LoRA 训练与服务衔接**：
  - 训练时注意：target_modules 应与 vLLM 兼容（q_proj/k_proj/v_proj/o_proj）
  - 导出为 PEFT 格式的正确方法
  - r（rank）值的选择：8/16/32/64 对质量和大小的影响
  - alpha（缩放系数）= r 通常是好的起点
- **多 LoRA 并发性能**：
  - `--max-loras` 控制同时驻留内存的 LoRA 数量
  - `--max-lora-size` 控制 LoRA adapter 的最大大小（MB）
  - LoRA 切换的开销（通常 < 10ms，因为只是指针切换）
  - Prefix Caching 对 LoRA 场景的加速效果
- **常见问题排查**：
  - LoRA 不生效 → 检查 target_modules 是否匹配
  - LoRA 加载失败 → 检查 safetensors 格式和 adapter_config.json
  - 多 LoRA 显存溢出 → 减小 max-loras 或增大 gpu-memory-utilization
  - 输出质量差 → 检查 alpha/r 设置和训练数据质量
- **LoRA + 量化组合**：
  - Base model 用 AWQ/GPTQ 量化 + LoRA adapter 保持 FP16
  - 进一步降低显存需求
  - 注意：部分量化方案与 LoRA 的兼容性问题

---

## 第09章：LangChain / LlamaIndex 框架集成（4节）

### 定位
vLLM 作为 OpenAI 兼容的服务端，可以无缝接入所有主流 LLM 编排框架。这一章讲解如何在 LangChain、LlamaIndex 等框架中使用 vLLM 替代云端 API。

### 09-1 LangChain + vLLM
- **方式一：ChatOpenAI 兼容模式（推荐）**：
  ```python
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(
      base_url="http://localhost:8000/v1",
      api_key="not-needed",
      model="Qwen/Qwen2.5-7B-Instruct",
      temperature=0.7,
  )
  response = llm.invoke("用 Python 写一个快速排序")
  ```
  - 优点：零学习成本，所有 LangChain 功能直接可用
  - 包括：Chain / Agent / Tool / RAG Retriever / Memory / Streaming
- **方式二：vLLM 原生 LangChain 集成**：
  ```python
  from langchain_community.llms import VLLM
  llm = VLLM(
      model="Qwen/Qwen2.5-7B-Instruct",
      trust_remote_code=True,
      max_new_tokens=128,
      top_p=0.9,
      temperature=0.7,
      vllm_kwargs={"tensor_parallel_size": 1},
  )
  ```
  - 优点：更细粒度的 vLLM 参数控制
  - 适合离线批量推理场景
- **完整 RAG Pipeline 示例**：
  - Document Loader → Text Splitter → vLLM Embeddings → Vector Store → Retriever → Chain
  - 使用 vLLM 同时做 LLM + Embedding（如果模型支持）
- **Agent + Tool Calling**：
  - ReAct Agent 模式
  - Function Calling（需要支持 tools 的模型）
  - 自定义 Tool 定义
- **Streaming 输出**：`astream()` / `astream_events()` 逐 token 流式返回

### 09-2 LlamaIndex + vLLM
- **初始化**：
  ```python
  from llama_index.llms.vllm import Vllm
  llm = Vllm(
      model="Qwen/Qwen2.5-7B-Instruct",
      tokenizer="Qwen/Qwen2.5-7B-Instruct",
      tokenizer_mode="auto",
      context_window=32768,
      generate_kwargs={"temperature": 0.7},
      max_tokens=512,
      verbose=False,
  )
  ```
- **Embedding 模型集成**：
  ```python
  from llama_index.embeddings.vllm import VllmEmbedding
  embed_model = VllmEmbedding(model="BAAI/bge-m3")
  ```
- **索引构建与查询**：
  - VectorStoreIndex.from_documents() 使用 vLLM
  - Query Engine / Chat Engine
  - Streaming Response
- **本地 RAG 完整示例**（~100 行代码）
- **多模态支持**：LlamaIndex MultiModal + vLLM VLM

### 09-3 其他框架与生态集成
- **OpenAI SDK 通用适配**（万能钥匙）：
  - 任何支持 `base_url` 参数的 OpenAI 客户端都能连 vLLM
  - Vercel AI SDK (`@ai-sdk/openai`)
  - LiteLLM（统一网关：vLLM + OpenAI + Anthropic 统一入口）
  - One API（开源 API 管理/分发平台）
- **Haystack (deepset)**：
  - `VLLMGenerator` 组件
  - Pipeline 集成
- **Dify / FastGPT**：
  - 开源 LLM 应用平台的 vLLM 接入配置
  - 可视化工作流编排
- **Flowise**：
  - 拖拽式 LangChain UI + vLLM 节点

### 09-4 生产级架构设计
- **vLLM 服务层架构**：
  ```
  Client App
      │
      ▼
  ┌─────────────┐
  │  Nginx LB    │ ← 负载均衡 + TLS + Rate Limiting
  └──────┬──────┘
         │
  ┌──────▼──────────────────────────────┐
  │        API Gateway / Sidecar          │
  │  (Auth / Rate Limit / Request Log)    │
  └──────┬──────────────┬───────────────┘
         │              │
  ┌──────▼──┐    ┌──────▼──┐
  │ vLLM-A   │    │ vLLM-B   │  ← 多实例
  │ :8000    │    │ :8001    │     (TP=2 each)
  └──────┬───┘    └──────┬───┘
         │              │
  ┌──────▼──────────────▼──┐
  │   Redis (Semantic Cache)│  ← 语义缓存层
  └────────────────────────┘
  ```
- **缓存层设计**：
  - 语义缓存（相似问题命中缓存直接返回）
  - Redis + 向量相似度检索
  - TTL 和缓存失效策略
- **降级与熔断**：
  - vLLM 不可用时 fallback 到 OpenAI API
  - 熔断器模式（Circuit Breaker）
  - 超时与重试策略
- ** observability**：
  - Prometheus metrics（vLLM 内置 `/metrics` 端点）
  - 结构化日志
  - OpenTelemetry tracing

---

## 第10章：生产部署与运维（5节）

### 定位
把 vLLM 从开发环境搬到生产环境，涉及容器化、安全加固、监控告警、高可用、成本优化等一系列工程化话题。

### 10-1 Docker 容器化部署
- **官方镜像**：`vllm/vllm-openai:latest`
  - 镜像层次结构（CUDA runtime → Python deps → vllm）
  - 多种变体：`vllm-openai` / `vllm-serve` / `vllm-dev`
- **docker-compose 模板**：
  ```yaml
  services:
    vllm:
      image: vllm/vllm-openai:latest
      ports:
        - "8000:8000"
      volumes:
        - ./models:/root/.cache/huggingface
        - ./data:/data
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                capabilities: [gpu]
                device_ids: ['0', '1']
      command: >
        --model Qwen/Qwen2.5-7B-Instruct
        --tensor-parallel-size 2
        --max-model-len 16384
        --gpu-memory-utilization 0.92
  ```
- **GPU 直通配置**：
  - NVIDIA Container Toolkit 安装与配置
  - `--gpus all` vs 指定 GPU ID
  - `--runtime=nvidia` 运行时参数
- **模型存储策略**：
  - 容器内下载（首次启动慢，重启保留）
  - Volume 挂载（推荐：模型文件持久化到宿主机）
  - NFS 共享存储（多节点共用一套模型文件）
- **镜像优化**：
  - 多阶段构建减小镜像体积
  - 模型预埋到镜像 vs 运行时下载
  - 安全基线（非 root 用户 / 最小权限）

### 10-2 Kubernetes 编排
- **Deployment YAML 模板**：
  - GPU 资源申请（`nvidia.com/gpu: 2`）
  - Init Container 预下载模型
  - Probe 配置（liveness / readiness / startup）
  - HPA（水平自动伸缩）策略
- **Service 与 Ingress**：
  - ClusterIP Service 暴露集群内部访问
  - Ingress Controller + TLS 终结
  - DNS 配置与域名绑定
- **模型存储方案**：
  - PVC 持久化卷（每个 Pod 独享）
  - HostPath 挂载（节点本地存储）
  - CSI Driver（云厂商对象存储）
  - PV + 模型预热 Job
- **多副本部署**：
  - ReplicaSet 管理
  - Pod 反亲和性（分散到不同节点）
  - Session Affinity（StatefulSet for sticky sessions）
- ** Helm Chart 部署**（可选）：
  - 社区 Helm Chart 使用
  - 自定义 values.yaml

### 10-3 监控、日志与告警
- **vLLM 内置 Metrics 端点**：
  - GET `/metrics`（Prometheus 格式）
  - 关键指标：
    - `vllm:num_requests_running`：当前运行中的请求数
    - `vllm:num_requests_waiting`：排队中的请求数
    - `vllm:gpu_cache_usage_perc`：KV Cache 显存使用百分比
    - `vllm:num_requests_success/fail`：成功/失败计数
    - `vllm:e2e_request_latency_seconds`：端到端延迟直方图
    - `vllm:time_to_first_token_seconds`：TTFT 直方图
    - `vllm:time_per_output_token_seconds`：TPOT 直方图
    - `vllm:generation_tokens_total`：生成 token 总数
- **Prometheus + Grafana Stack**：
  - scrape 配置
  - Dashboard 模板（延迟分布 / 吞吐趋势 / GPU 利用率 / 错误率）
- **结构化日志**：
  - 日志格式（JSON / 含 request_id）
  - 日志聚合（Loki / ELK）
  - LogQL 查询示例
- **告警规则**：
  - 服务不可达（P0）
  - P99 延迟超过阈值（P1）
  - GPU Cache 使用率 > 95%（P1）
  - 错误率突增（P1）
  - 排队深度持续增长（P2）
- **链路追踪**：OpenTelemetry 集成

### 10-4 高可用与弹性伸缩
- **故障类型与应对**：
  - 单 Pod 崩溃 → K8s RestartPolicy 自动重建
  - 单节点故障 → Pod 调度到其他节点
  - GPU 驱动异常 → Node Taint + Pod 驱逐
  - 机房级故障 → 多区域部署 + DNS 切换
- **弹性伸缩策略**：
  - KHPA（基于自定义指标的 HPA）
  - 基于 GPU 利用率的扩缩容
  - 基于队列深度的扩缩容（预测式）
  - 冷启动优化（Model Warm-up / Snapshot）
- **蓝绿部署 / 金丝雀发布**：
  - 新模型版本的平滑升级
  - 流量逐步切换
  - 自动回滚机制
- **容量规划**：
  - 请求量预测 → GPU 需求计算
  - 成本模型（自有 vs 租赁 vs 云 serverless）

### 10-5 性能调优终极指南
- **模型加载优化**：
  - `--load-format` 选择（auto / dtensor / shammo / dummy）
  - `--cpu-offload-gb` CPU 卸载部分层
  - `--enforce-eager` 禁用 CUDA Graph（debug 用）
- **推理加速技术**：
  - **Speculative Decoding（推测解码）**：
    - 用小模型（Draft Model）快速生成候选 token
    - 大模型（Target Model）并行验证
    - 接受的 token 数 / 总候选数 = Acceptance Rate
    - 典型加速 1.5x-2.5x（acceptance rate 60-75%）
    - `--speculative-model <draft_model_path>` 启用
  - **CUDA Graphs**：减少 kernel launch overhead（默认启用）
  - **FlashAttention**：vLLM 默认集成 FlashAttention-2/3
- **系统级优化**：
  - **NVLink/NVSwitch**：多卡间高带宽互联（900 GB/s vs PCIe 64 GB/s）
  - **Tensor Core 利用**：确保 dtype 匹配硬件（H100 用 FP8/BF16）
  - **内存锁定**：`--locked-memory` 防止页面换出
  - **NUMA 感知**：多 socket 服务器上的 NUMA 绑定
  - **Over-subscription**：`--num-lookahead-slots` 预分配 slot
- **端到端优化 Checklist**：
  - [ ] 使用正确的量化方案
  - [ ] TP size 与 GPU 数量匹配
  - [ ] `--max-model-len` 按需设置（不要过大）
  - [ ] `--gpu-memory-utilization` 设为 0.90-0.95
  - [ ] 启用 Speculative Decoding（如果有 draft model）
  - [ ] 启用 Prefix Caching（多请求共享前缀场景）
  - [ ] 使用 CUDA Graphs（默认开启，确认未被禁用）
  - [ ] Nginx 反代关闭缓冲（`proxy_buffering off`）
  - [ ] 连接池复用（HTTP Keep-Alive）
  - [ ] 语义缓存层（减少重复推理）

---

## 附录

### A. 常用命令速查表
- 服务启动：api_server 完整命令模板
- 离线推理：LLM.generate() 模板
- Benchmark：throughput / serving benchmark 命令
- 模型管理：下载 / 查看 / 转换

### B. 启动参数完整字典
- 模型参数：model / dtype / trust-remote-code / revision / etc.
- 并行参数：tensor-parallel-size / pipeline-parallel-size / etc.
- 性能参数：max-model-len / gpu-memory-utilization / etc.
- 功能参数：enable-lora / enable-auto-tool-choice / etc.
- 调度参数：max-num-seqs / scheduler-delay-factor / etc.
- 服务参数：host / port / ssl / etc.
- 官方文档参数链接

### C. 支持的模型完整列表
- 按架构分类：Llama / Qwen / Mistral / DeepSeek / Phi / Gemma / Yi / ...
- 每个模型的支持状态（✅完全支持 / ⚠️部分支持 / ❌不支持）
- 推荐的量化方案和配置

### D. 常见错误与解决方案
- `CUDA out of memory` → 减小 tensor-parallel-size / 降低 max-model-len / 启用量化
- `OSError: Shared memory limit` → `--shm-size` Docker 参数 / `dev/shm` 挂载
- `ValueError: Unknown model type` → 检查模型注册 / 更新 vLLM 版本
- `RuntimeError: Expected all tensors to be on the same device` → TP 配置错误
- LoRA 相关错误 → target_modules / format 检查
- 性能不如预期 → 检查 CUDA Graphs / FlashAttention / 量化配置

### E. vLLM vs Ollama vs TGI 选型决策树
```
你的场景是什么？
├── 个人学习 / 本地体验
│   └── 🎯 Ollama（最简单）
├── 团队内部工具（< 10 并发）
│   ├── 需要简单部署 → Ollama
│   └── 需要高性能 → vLLM 单卡
├── 生产服务（> 10 并发）
│   ├── 需要最大吞吐 → 🎯 vLLM（PagedAttention + Continuous Batching）
│   ├── 需要 HF 生态深度整合 → TGI
│   └── 需要多模型 / LoRA → vLLM（原生支持）
├── 需要分布式多卡
│   └── 🎯 vLLM（TP/PP/Ray 集群）
└── 需要多模态（视觉语言模型）
    ├── Ollama（通过 llama.cpp，简单）
    └── 🎯 vLLM（原生 VLM 支持，性能更强）
```

### F. 学习路线图
- **入门阶段（1-2 周）**：安装 → 跑通第一个 API → OpenAI SDK 对接
- **进阶阶段（2-4 周）**：PagedAttention 原理 → 量化部署 → LoRA 服务化
- **精通阶段（1-2 月）**：分布式部署 → K8s 编排 → 性能调优 → 生产运维
- **专家阶段（持续）**：源码贡献 / 自定义 Kernel / 新架构支持
