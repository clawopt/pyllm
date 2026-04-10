# 为什么需要 vLLM？

## 白板导读

如果你已经用 Ollama 在本地跑通过大模型对话，你可能会问：Ollama 已经很好用了，一行命令就能启动模型，为什么还需要 vLLM？答案是——这取决于你的场景。打个比方，**Ollama 像是一辆家用轿车，好开、省心、停哪儿都方便；而 vLLM 则是一辆 F1 赛车，它不是为了"能跑起来"设计的，而是为了在赛道上榨干每一滴性能而生的**。当你需要同时服务 100 个用户、每秒处理 500 个请求、把 4 张 A100 显卡的利用率拉到 95% 以上的时候，Ollama 的串行处理模式就会成为瓶颈，而这正是 vLLM 的主场。

vLLM 来自 UC Berkeley 的 SkyLab 团队（就是那篇著名的 LLaMA 论文的同一批人），它的核心创新有两项：**PagedAttention**（分页注意力机制）和 **Continuous Batching**（连续批处理）。这两项技术分别解决了传统 LLM 推理中的两个老大难问题——显存浪费和吞吐瓶颈。本章先不急着讲原理（那是第二章和第三章的事），而是站在一个更高的视角回答三个问题：vLLM 到底是什么？它和 Ollama、TGI 这些工具有什么区别？什么情况下你应该选择 vLLM 而不是其他方案？最后，我们用三行命令跑通第一个 vLLM 服务，让你亲手感受一下它的威力。

---

## 传统 LLM 推理的两大痛点

在理解 vLLM 的价值之前，我们需要先搞清楚它在解决什么问题。传统的大语言模型推理——无论是直接用 HuggingFace Transformers 跑、还是用 Ollama 跑、甚至是最早的 `model.generate()` —— 都面临两个根本性的效率瓶颈。

### 痛点一：KV Cache 的显存浪费

当一个大模型处理输入文本时，它并不是一次性读完所有内容就完事了。Transformer 的自注意力机制要求模型在生成每个新 token 时，都要"看到"之前所有的 token。为了不重复计算，推理引擎会把每个 token 对应的 Key 向量和 Value 向量缓存起来——这就是 **KV Cache**。

问题在于，传统的 KV Cache 管理方式非常粗暴。引擎会为**每一个请求预先分配最大序列长度（max_seq_len）的 KV Cache 空间**。如果你的 max_seq_len 设成 8192 个 token，那么不管用户的实际输入只有 10 个 token 还是 8000 个 token，系统都会一口气分配掉 8192 个位置的空间。更糟糕的是，不同请求的实际序列长度参差不齐，导致显存中充满了无法被其他请求利用的空洞。这就好比一家餐厅预订了 100 个座位，但每次只来 20 位客人——80 个座位白白空着，后面的客人还进不来。

用一个具体的数字来说明这个问题。以 Llama 3 8B 模型为例：

```
单条 KV Cache 大小计算：
  层数 = 32
  KV heads = 8 (GQA 分组后)
  Head dim = 128
  Dtype = FP16 = 2 bytes
  每个 token 的 KV Cache ≈ 32 × 2 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB

如果 max_seq_len = 8192：
  单个请求预分配 = 128 KB × 8192 = 1 GB

Batch size = 16：
  总 KV Cache 预分配 = 1 GB × 16 = 16 GB

但实际平均只用 1024 tokens/请求：
  实际使用 = 128 KB × 1024 × 16 = 2 GB
  浪费率 = (16 - 2) / 16 = 87.5%
```

**87.5% 的显存被浪费了！** 这意味着你花大价钱买的 A100 80GB 显卡，有将近 14GB 的空间什么都没干就吃掉了。而且这只是 KV Cache 的浪费，还没算上模型权重本身占用的显存。

### 痛点二：静态批处理的延迟灾难

第二个问题出在**批处理策略**上。传统的推理引擎使用的是**静态批处理（Static Batching）**，它的工作流程是这样的：

```
静态批处理的工作流：

时间轴 →

[收集请求 A] [收集请求 B] [收集请求 C] ... [凑齐一批！]
                                                    │
                                    ┌───────────────▼───────────────┐
                                    │   Padding 到统一长度            │
                                    │   送入 GPU 并行计算              │
                                    │   等待所有请求全部完成           │
                                    └───────────────┬───────────────┘
                                                     │
                                          [返回结果] [开始下一批]
```

这种模式有三个致命缺陷：

**第一，首 Token 延迟（TTFT, Time to First Token）极高。** 第一个到达的请求必须等待后续所有请求凑齐一整批才能开始处理。如果 batch size 是 16，而你恰好是第 1 个到的，你需要等后面 15 个人都到齐了才能上车——就像公交车必须坐满才发车一样。在高并发场景下，这意味着用户的等待时间可能长达数秒甚至十几秒，体验极差。

**第二，Padding 浪费计算资源。** 同一批次中，有的请求只有 50 个 token，有的却有 2000 个 token。为了让它们能组成一个矩阵送入 GPU 计算，短请求必须用无意义的 padding token 填充到最长请求的长度。这些填充位置的注意力计算完全是无效劳动，白白消耗 GPU 算力。

**第三，长尾效应拖慢整体吞吐。** 一批请求中只要有一个"慢户"（比如生成了超长的回复），整批请求都必须等它完成才能进入下一轮。这个慢请求成了整个系统的瓶颈。

---

## vLLM 的两项核心创新

针对上述两大痛点，vLLM 提出了两套革命性的解决方案。

### PagedAttention：让显存利用率从 60% 飙升到 95%+

PagedAttention 的设计灵感来自操作系统的**虚拟内存分页机制（Virtual Memory Paging）**。这是计算机科学史上最经典的设计之一——早在 1960 年代，操作系统工程师们就发现，给每个进程预分配连续的物理内存会导致严重的碎片化问题。他们的解决方案是：将内存切分为固定大小的"页（Page）"，进程的逻辑地址空间通过"页表（Page Table）"映射到离散的物理页上，按需分配，用完归还。

vLLM 把这套思想原封不动地搬到了 GPU 显存管理上：

| 操作系统概念 | PagedAttention 对应 |
|:---|:---|
| 物理内存 | GPU VRAM（显存） |
| Page（通常 4KB） | **Block**（默认 16 个 token 为一块） |
| 进程地址空间 | 一个 Sequence（一次请求）的 KV Cache |
| 页表（Page Table） | **Block Table**（记录逻辑块→物理块的映射关系） |
| 缺页中断（Page Fault） | 新 token 到来时申请 Block |
| 页面回收 | 序列结束时释放 Block |
| 内存碎片 | **接近于零**（因为 Block 是固定大小的） |

这意味着什么？意味着 vLLM 不再为每个请求预分配一大块连续的 KV Cache 空间，而是像操作系统管理内存一样，按需分配一个个小的 Block。当一个请求只需要 100 个 token 的 KV Cache 时，它只占用 ceil(100/16) = 7 个 Block；当另一个请求需要 8000 个 token 时，它动态地占用 500 个 Block。请求结束后，Block 立即回收供下一个请求使用。

效果是立竿见影的：**GPU 显存利用率从传统方案的 50-65% 提升到了 90-98%**。同样的硬件，vLLM 能塞下更多的并发请求，或者运行更大的模型。

> **面试高频题**："请解释 vLLM 的 PagedAttention 原理。" 回答要点：①类比 OS 虚拟内存分页 → ②KV Cache 切分为固定大小 Block → ③Block Table 维护映射 → ④按需分配/释放消除碎片 → ⑤显存利用率从 ~60% 提升到 ~95%。如果能进一步解释 Prefix Caching（前缀共享）和 Copy-on-Write（共享前缀时避免重复存储），那就是加分项了。

### Continuous Batching：边跑边接人

如果说 PagedAttention 解决了"存不下"的问题，Continuous Batching 解决的就是"不够快"的问题。它彻底抛弃了"凑齐一批发车"的静态模式，改为**迭代级别的动态调度**：

```
Continuous Batching 工作流：

Iteration 1:  [A生成token1] [B生成token1]
               ↓ 新请求 C 到达！
Iteration 2:  [A生成token2] [B生成token2] [C生成token1]  ← C 立即加入！
               ↓ 新请求 D 到达！
Iteration 3:  [A生成token3] [B生成token3] [C生成token2] [D生成token1]
               ↓ A 完成了！释放资源
Iteration 4:  [B生成token4] [C生成token3] [D生成token2] [E生成token1]  ← E 也加入了
               ...
```

注意看发生了什么变化：新请求 C 和 D 不再需要等 A 和 B 处理完，而是在下一轮迭代就立刻加入正在运行的批次。已经完成的请求 A 立即退出，腾出的空间立即接纳新请求 E。整个过程没有任何"等待凑齐"的时间窗口。

用生活化的比喻来说：**静态批处理像是公交车（必须坐满才发车），而 Continuous Batching 像是地铁列车（行驶中乘客可以随时上下车）**。

这项改进带来的效果是惊人的：

- **TTFT（首个 Token 延迟）降低 2-5x**：不再需要等齐一批请求
- **Padding 开销归零**：每个请求独立维护自己的序列长度
- **GPU 利用率大幅提升**：几乎每一刻 GPU 都在满载运算
- **总体吞吐量提升 14-24x**（相比 HuggingFace Transformers，官方 benchmark 数据）

---

## vLLM vs Ollama vs TGI：如何选择？

现在我们有了足够的背景知识来做一场客观的工具对比。这三个工具各有各的定位，没有绝对的"最好"，只有"最适合你的场景"。

### 核心维度对比

| 维度 | Ollama | vLLM | TGI (HuggingFace) |
|:---|:---|:---|
| **目标用户** | 个人开发者 / 小团队 | 算法工程师 / MLOps / 平台团队 | HF 生态深度用户 |
| **上手难度** | ⭐ 极简 | ⭐⭐⭐ 中等 | ⭐⭐⭐ 中等 |
| **推理性能** | 基准水平（llama.cpp 后端） | **业界顶尖** | 高 |
| **吞吐量** | 低（有限并行） | **极高**（连续批处理） | 高 |
| **显存利用率** | ~60% | **~95%+**（PagedAttention） | ~80% |
| **API 兼容性** | OpenAI 兼容 | **原生 OpenAI 兼容** | OpenAI 兼容 |
| **量化支持** | GGUF 系列 | AWQ/GPTQ/FP8/SqueezeLLM/NF4 | AWQ/GPTQ/Bitsandbytes |
| **LoRA 服务化** | 基础支持 | **原生支持（--enable-lora）** | 支持 |
| **分布式推理** | ❌ 不支持 | **TP + PP + Ray 多节点** | 支持 |
| **多模态 VLM** | LLaVA 等（通过 llama.cpp） | **原生支持（Llava-OneVision/Qwen2-VL）** | 有限 |
| **离线批量推理** | 不适用 | **LLM 类直接调用** | 支持 |
| **Speculative Decoding** | ❌ | ✅ 支持 | 部分 |

### 选型决策指南

```
你的场景是什么？
│
├── 🏠 个人学习 / 本地体验 / 偶尔跑个对话
│   └── 🎯 选 Ollama
│      理由：最简单，brew install 一行搞定，不用管 CUDA
│
├── 👥 团队内部工具（< 10 人同时用）
│   ├── 只需要基本功能 → Ollama 就够了
│   └── 开始关注性能和并发 → 可以试试 vLLM 单卡部署
│
├── 🏢 生产级 API 服务（> 10 并发 / SLA 要求）
│   ├── 需要最大化 GPU 吞吐 🎯 → vLLM（PagedAttention + Continuous Batching）
│   ├── 需要 HF 生态深度整合 → TGI
│   ├── 需要服务 LoRA 微调模型 🎯 → vLLM（--enable-lora 原生支持）
│   └── 需要多机多卡分布式 🎯 → vLLM（Tensor Parallelism + Ray 集群）
│
└── 🔬 研究 / 批量评估 / 数据标注
    └── 🎯 选 vLLM（离线推理模式 LLM.generate()，批量处理效率最高）
```

简单总结一句话：**Ollama 是"让普通人用上大模型"的瑞士军刀，vLLM 是"让企业用好大模型"的工业引擎，TGI 是"HuggingFace 生态用户"的自然选择。**

---

## 性能数据说话

光说原理不够，让我们看看真实的 benchmark 数据。以下数据来自 vLLM 官方测试（Llama 3 8B Instruct，单张 A100 80GB）：

### 与 HuggingFace Transformers 对比

| 场景 | HF Transformers | vLLM | 提升 |
|:---|:---|:---|:---|
| **吞吐量（tokens/s）** | 23 tok/s | **487 tok/s** | **21.2x** |
| **请求延迟（P99）** | 12.3s | **0.58s** | **21.2x 更快** |
| **并发能力** | 1-2 请求 | **64+ 请求** | **32x+** |
| **显存利用率** | ~55% | **~94%** | **+39pp** |

### 与 TGI 对比

| 配置 | TGI throughput | vLLM throughput | vLLM 领先 |
|:---|:---|:---|:---|
| Llama 3 8B, sharegpt 输入 | 18.2 req/s | **28.6 req/s** | **+57%** |
| Llama 3 8B, 长输入混合 | 12.5 req/s | **22.1 req/s** | **+77%** |
| Mixtral 8x7B, 共享前缀场景 | 15.8 req/s | **31.2 req/s** | **+98%** |

可以看到，在**共享前缀（Prefix Sharing）**场景下 vLLM 的优势尤其明显——这正是 PagedAttention 的 Copy-on-Write 机制发挥作用的地方：多个请求如果有相同的 System Prompt，它们的 KV Cache 前缀可以共享同一份物理 Block，完全不需要重复存储。

---

## 第一个 vLLM 服务：三步跑通

理论讲了这么多，是时候动手了。下面我们在一台有 NVIDIA GPU 的机器上（或者云 GPU 实例上），用三条命令完成从安装到调用 vLLM 的全过程。

### 第一步：安装 vLLM

```bash
# 确保 Python 3.9+
python --version

# 安装 vLLM（自动匹配你当前的 CUDA 版本）
pip install vllm

# 验证安装成功
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

第一次安装可能需要几分钟，因为它会下载与你的 CUDA 版本匹配的 PyTorch 和相关依赖。如果遇到 CUDA 版本不匹配的问题，可以用以下方式指定：

```bash
# 如果你有 CUDA 12.1
VLLM_USE_CUDA=1 pip install vllm

# 或者从源码编译（高级用法，不推荐新手尝试）
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e . --no-build-isolation
```

### 第二步：启动 API Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --trust-remote-code
```

启动时你会看到类似这样的输出：

```
INFO 01-15 14:30:00 api_server.py:168] vLLM API server started on http://0.0.0.0:8000
INFO 01-15 14:30:00 engine.py:224] Initializing an LLM engine with config:
  model='Qwen/Qwen2.5-7B-Instruct', dtype=auto
INFO 01-15 14:30:00 model_runner.py:317] Loading model weights from Qwen/Qwen2.5-7B-Instruct
  Loading safetensors checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
  Loading safetensors checkpoint shards:  25%|██▊       | 1/4 [00:03<00:11,  3.85s/it]
  Loading safetensors checkpoint shards:  50%|█████     | 2/4 [00:07<00:08,  3.89s/it]
  Loading safetensors checkpoint shards:  75%|███████▌  | 3/4 [00:11<00:04,  3.88s/it]
  Loading safetensors checkpoint shards: 100%|██████████| 4/4 [00:14<00:00,  3.71s/it]
INFO 01-15 14:30:15 model_runner.py:340] Model loading took 14.52 seconds
INFO 01-15 14:30:15 engine.py:296] GPU memory usage: 6.2 / 72.0 GB (8.6%)
INFO 01-15 14:30:15 api_server.py:172] Ready to serve requests!
```

注意几个关键信息：
- **模型加载耗时约 14 秒**（首次从 HuggingFace Hub 下载模型文件会更久，之后会有缓存）
- **GPU 显存使用了 6.2GB**（7B 模型 FP16 约 14GB 权重 + KV Cache 预留空间）
- **`Ready to serve requests!` 出现后说明服务已就绪**

### 第三步：发送请求

打开一个新的终端，用 curl 发送一个简单的聊天请求：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手"},
            {"role": "user", "content": "用一句话解释什么是量子计算"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }'
```

你会收到类似这样的响应：

```json
{
    "id": "chatcmpl-a3f8c2d1",
    "object": "chat.completion",
    "created": 1736941800,
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "量子计算是一种利用量子力学原理（如叠加态和纠缠）进行信息处理的计算方式。与传统计算机用比特（0或1）不同，量子计算机用量子比特（qubit）来表示同时处于多种状态的 信息，从而在某些特定问题上实现指数级的计算加速。"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 22,
        "completion_tokens": 86,
        "total_tokens": 108
    }
}
```

注意到什么了吗？**这个响应格式和 OpenAI API 完全一致**。`choices[0].message.content`、`usage.prompt_tokens`、`finish_reason`——所有字段都是 OpenAI 规范的标准格式。这就是 vLLM 的杀手锏之一：**OpenAI 兼容性是原生的、开箱即用的**。

### 用 Python SDK 调用（更推荐的方式）

curl 只是验证工具。在实际开发中，你会用 Python 的 openai 库来调用：

```python
"""
vLLM 第一个示例：用 OpenAI 兼容 SDK 调用本地 vLLM 服务
"""

from openai import OpenAI


def main():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed-for-vllm",
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "你是一个Python技术专家"},
            {"role": "user", "content": "写一个快速排序算法，并解释它的时间复杂度"}
        ],
        temperature=0.7,
        max_tokens=512,
    )

    reply = response.choices[0].message.content
    usage = response.usage

    print("=" * 50)
    print("🤖 vLLM 响应:")
    print(reply)
    print("=" * 50)
    print(f"📊 Token 使用: {usage.prompt_tokens} prompt + "
          f"{usage.completion_tokens} completion = "
          f"{usage.total_tokens} total")


if __name__ == "__main__":
    main()
```

运行结果：

```
==================================================
🤖 vLLM 响应:

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

时间复杂度分析：
- 最佳情况：O(n log n) — 每次 pivot 都正好选中位数
- 平均情况：O(n log n) — 数学期望值
- 最坏情况：O(n²) — 数组已排序且总是选到最小/最大元素
- 空间复杂度：O(log n) — 递归栈深度

优化建议：随机选择 pivot 或使用三数取中法可避免最坏情况。
==================================================
📊 Token 使用: 24 prompt + 187 completion = 211 total
```

### 流式输出体验

对于交互式应用（如聊天界面），流式输出是必不可少的——用户不想等到模型想完了所有字才一次性看到结果。vLLM 原生支持 SSE（Server-Sent Events）流式协议：

```python
"""
vLLM 流式输出示例：逐 Token 实时显示
"""

from openai import OpenAI


def stream_chat():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed-for-vllm",
    )

    stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "user", "content": "用诗意的语言描述春天的到来"}
        ],
        stream=True,
        temperature=0.8,
        max_tokens=300,
    )

    print("🤖 (流式输出): ", end="", flush=True)

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            content = delta.content
            print(content, end="", flush=True)
            full_response += content

    print(f"\n\n{'=' * 40}")
    print(f"✅ 总计 {len(full_response)} 字符")


if __name__ == "__main__":
    stream_chat()
```

运行效果：

```
🤖 (流式输出): 春天，是一位轻盈的画师，提着蘸满嫩绿的笔，
悄然来到沉睡了一冬的大地。她先是在枯枝上点染几抹鹅黄，
又在河面撒下一层薄薄的雾霭。风不再是凛冽的刀刃，
而变成了温柔的手指，拂过山岗时，野花便一朵朵睁开了眼。

燕子衔着湿润的泥归来，在屋檐下修补去年的旧巢。
泥土松软的气息从地面升腾起来，混着青草萌发的清香。
这不是轰轰烈烈的登场，而是一种缓慢的、确信无疑的苏醒——
仿佛整个世界都在屏息等待着，那一声春雷落下。

========================================
✅ 总计 247 字符
```

你可以看到文字是**逐段涌现**出来的，而不是等了很久之后突然全部显示。这种体验上的差异对于终端用户来说是决定性的——研究表明，用户感知的响应速度主要取决于 TTFT（首个可见内容的出现时间），而不是总生成时间。

---

## 支持的模型生态

vLLM 不是只支持某一个特定模型的专用引擎，它是一个**通用的 LLM 推理框架**，支持 HuggingFace 上绝大多数主流开源大模型。截至最新版本，以下是主要支持的模型系列：

### 通用对话模型

| 模型系列 | 代表版本 | 特点 | 推荐量化 |
|:---|:---|:---|:---|
| **Llama** | 3.1 / 3.2 / 3.3 (8B/70B/405B) | Meta 旗舰，生态最大 | AWQ / GPTQ |
| **Qwen2.5** | 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B | 中文能力最强 | AWQ / GPTQ |
| **Mistral / Mixtral** | 7B / 8x7B / 8x22B | MoE 架构，高效 | GPTQ |
| **DeepSeek** | V2 / V3 / R1 | 推理/代码能力强 | AWQ |
| **Yi** | 1.5 / 34B / Lightning | 零一万物出品 | AWQ |
| **Gemma 2** | 2B / 9B / 27B | Google 开源 | GPTQ |
| **Phi-3** | mini / medium / 3.8B | 微软小模型 | AWQ |
| **InternLM2** | 1.8B / 7B / 20B | 书生·浦语 | AWQ |

### 多模态视觉语言模型（VLM）

| 模型 | 能力 | 说明 |
|:---|:---|:---|
| **Llava-OneVision** | 图像+视频理解 | 统一多模态架构 |
| **Qwen2-VL** | 图像理解 | 中文多模态 SOTA |
| **MiniCPM-V** | 图像+视频 | 端侧多模态 |
| **Pixtral** | 图像生成+理解 | Mistral 出品 |

### Embedding 模型

| 模型 | 维度 | 适用场景 |
|:---|:---|:---|
| **BGE-M3** | 1024 | 多语言、多粒度检索 |
| **GTE-Qwen2** | 1536 | 通用嵌入 |
| **Mxbai-Embed-Large** | 1024 | 多语言检索 |
| **E5-Mistral** | 4096 | 高质量检索 |

> **使用提示**：大部分模型只需在 `--model` 参数中传入 HuggingFace 的模型 ID（如 `meta-llama/Llama-3.1-8B-Instruct`），vLLM 会自动下载并加载。也可以指定本地路径 `--model /path/to/local/model`。

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **vLLM 定位** | 企业级高性能 LLM 推理引擎，不是玩具而是工业级基础设施 |
| **核心创新一** | **PagedAttention**：借鉴 OS 虚拟内存分页思想，KV Cache 按 Block 按需分配，显存利用率从 ~60% → ~95% |
| **核心创新二** | **Continuous Batching**：迭代级别动态调度，新请求即时加入批次，TTFT 降低 2-5x |
| **性能提升** | 相比 HuggingFace Transformers 吞吐量提升 **14-24x**，相比 TGI 提升 **1.5-3x** |
| **vs Ollama** | Ollama = 家用轿车（简单易用）；vLLM = F1 赛车（极致性能） |
| **API 兼容性** | 原生 OpenAI 兼容，改一行 `base_url` 即可从 OpenAI SDK 切换到 vLLM |
| **快速上手** | `pip install vllm` → 启动 server → `openai(base_url="...")` 三步完成 |
| **选型原则** | <10 并发用 Ollama，>10 并发 / 分布式 / 生产级 SLA → vLLM |

> **一句话总结**：vLLM 通过 PagedAttention 和 Continuous Batching 两项核心创新，解决了传统 LLM 推理中"显存浪费"和"吞吐瓶颈"两大顽疾，使 GPU 利用率和推理吞吐量达到了业界领先水平。如果你需要构建一个能扛住高并发、榨干 GPU 每一滴性能的生产级 LLM 服务，vLLM 是目前最优的选择——没有之一。
