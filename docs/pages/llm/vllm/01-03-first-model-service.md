# 第一个模型服务：启动与调用

## 白板导读

环境准备好了，模型也下载好了，现在是激动人心的时刻——按下启动键，看着 vLLM 把一个 70 亿参数的大语言模型加载到 GPU 里，然后通过 HTTP API 让它为你工作。这一节我们将从最简单的单行命令开始，逐步深入每一个启动参数的含义，然后分别用 curl、Python openai SDK、以及 vLLM 原生接口三种方式来调用服务。你会在实践中理解为什么 vLLM 的 API 设计如此受欢迎——因为它**完全兼容 OpenAI 的接口规范**，意味着你之前所有对接 OpenAI GPT-4 的代码，只需要改一行 `base_url` 就能切换到本地 vLLM。

---

## 3.1 最简启动命令

让我们从最简单的情况开始：在一台有 NVIDIA GPU 的机器上，启动一个 Qwen2.5-7B-Instruct 模型的推理服务。

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

就这两行（实际是一行被 `\` 折行了）。执行后你会看到终端开始输出大量日志信息：

```
INFO 01-15 14:00:00 api_server.py:168] Initializing vLLM with config:
  model='Qwen/Qwen2.5-7B-Instruct'
  dtype=auto
INFO 01-15 14:00:00 config.py:975] Setting max_model_len to 32768 for model Qwen/Qwen2.5-7B-Instruct
INFO 01-15 14:00:00 config.py:1002] Setting RoPE scaling to 'linear' for model Qwen/Qwen2.5-7B-Instruct
INFO 01-15 14:00:00 model_runner.py:317] Loading model weights from Qwen/Qwen2.5-7B-Instruct
  Loading safetensors checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
  Loading safetensors checkpoint shards:  25%|██▊       | 1/4 [00:03<00:11,  3.82s/it]
  Loading safetensors checkpoint shards:  50%|█████     | 2/4 [00:07<00:08,  3.87s/it]
  Loading safetensors checkpoint shards:  75%|███████▌  | 3/4 [00:11<00:04,  3.86s/it]
  Loading safetensors checkpoint shards: 100%|██████████| 4/4 [00:14<00:00,  3.71s/it]
INFO 01-15 14:00:14 model_runner.py:340] Model loading took 14.32 seconds
INFO 01-15 14:00:14 engine.py:296] # GPU blocks: 42831, # CPU blocks: 2048
INFO 01-15 14:00:14 engine.py:302] Maximum concurrency for 32K context: 69 sequences
INFO 01-15 14:00:14 engine.py:306] KV cache memory: 12.42 GiB
INFO 01-15 14:00:14 gpu_mapping.py:123] Found 1 GPU(s).
INFO 01-15 14:00:14 gpu_executor.py:89] Initializing VLLM on GPU 0...
INFO 01-15 14:00:16 llm_engine.py:161] init engine (profile=False) took 16.21 seconds
INFO 01-15 14:00:16 api_server.py:172] Application startup complete.
INFO 01-15 14:00:16 api_server.py:174] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

让我们逐段解读这些日志中隐藏的关键信息：

**第一阶段：配置解析**
```
Setting max_model_len to 32768
```
vLLM 自动从模型的 `config.json` 中读取了最大上下文长度为 **32768 tokens**（约 25000 个汉字或 40000 个英文单词）。Qwen2.5 系列支持超长上下文是它的一大卖点。

**第二阶段：模型加载**
```
Loading safetensors checkpoint shards: 4 files, ~14 seconds
```
Qwen2.5-7B 的权重文件被切分为 4 个 shard（分片），每个约 3.5GB。vLLM 使用 `safetensors` 格式加载（比 PyTorch 的 `.pt` 格式更安全、更快）。加载时间取决于磁盘 I/O 速度和 GPU PCIe 带宽。

**第三阶段：PagedAttention 初始化（核心！）**
```
# GPU blocks: 42831, # CPU blocks: 2048
KV cache memory: 12.42 GiB
Maximum concurrency for 32K context: 69 sequences
```
这几行日志揭示了 PagedAttention 的核心工作：
- vLLM 在 GPU 上预分配了 **42831 个 Block**（每个 Block 默认存 16 个 token 的 KV Cache）
- 同时在 CPU 上预留了 **2048 个 Block** 作为 swap 空间
- 这些 Block 总共占用 **12.42 GB 显存**作为 KV Cache
- 以最大上下文 32K 计算，理论上同时支持 **69 个序列**并发运行
- 模型权重本身约占 14GB（FP16），加上 12.42 GB KV Cache = 总计约 **26.42 GB**

> **这就是 PagedAttention 的威力所在**：26.42 GB / 24 GB（RTX 4090）看起来超了？不，因为 `--gpu-memory-utilization` 默认值 0.90 意味着 vLLM 只使用 24 × 0.9 = 21.6 GB 给模型+KV Cache，它会自动调整 Block 数量来适配这个限制。实际运行时如果显存不够，Block 数量会自动减少。

---

## 3.2 启动参数完全指南

上面的最简命令只用了两个参数。但在生产环境中，你需要了解全部关键参数才能把 vLLM 调到最优状态。下面按功能分类逐一介绍。

### 模型相关参数

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--model` | 字符串 | 必填 | HuggingFace 模型 ID 或本地路径 |
| `--dtype` | 字符串 | `auto` | 权重数据类型：`auto`(自动选择) / `half`(FP16) / `bfloat16` / `float16` / `float32` |
| `--trust-remote-code` | bool | `false` | 是否信任模型自带的自定义代码（Qwen/Llama/DeepSeek 等都需要设为 true） |
| `--revision` | 字符串 | `main` | 模型的特定版本/分支 |
| `--download-dir` | 路径 | HF 缓存目录 | 模型文件的下载存储位置 |
| `--load-format` | 字符串 | `auto` | 加载格式：`auto` / `pt` / `safetensors` / `dummy` / `dtensor` / `sharded_state` |

```bash
# 示例：加载特定版本的模型，指定下载目录
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --revision "v0.1.0" \
    --download-dir /data/models \
    --trust-remote-code
```

### 并行与分布式参数

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--tensor-parallel-size` | 整数 | `1` | 张量并行 GPU 数量（TP） |
| `--pipeline-parallel-size` | 整数 | `1` | 流水线并行 Stage 数（PP） |
| `--worker-use-ray` | bool | `false` | 是否使用 Ray 进行多 worker 管理 |

```bash
# 双卡张量并行示例（适合 13B-34B 模型）
--tensor-parallel-size 2

# 四卡张量并行示例（适合 70B 模型）
--tensor-parallel-size 4

# TP + PP 组合（8 卡：TP=2, PP=4）
--tensor-parallel-size 2 \
--pipeline-parallel-size 4
```

### 性能与资源参数（最重要！）

| 参数 | 类型 | 默认值 | 说明 | 调优建议 |
|:---|:---|:---|:---|:---|
| `--max-model-len` | 整数 | 模型默认值 | 最大上下文长度（token 数） | 按需设置，不要盲目求大 |
| `--gpu-memory-utilization` | 浮点数 | `0.90` | GPU 显存用于模型+KV Cache 的比例 | 0.85-0.95 之间调整 |
| `--max-num-seqs` | 整数 | `256` | 同时运行的最大序列数 | 高并发场景可增大 |
| `--max-num-batched-tokens` | 整数 | 自动计算 | 每 iteration 最大处理 token 数 | 控制单次计算量 |
| `--scheduler-delay-factor` | 浮点数 | 无 | 调度延迟因子 | 低延迟用小值，高吞吐用大值 |

这三个参数对性能的影响最为直接，值得详细展开：

#### max-model-len：上下文长度的双刃剑

`--max-model-len` 决定了单个请求最多能处理多少个 token 的输入 + 输出。它的值越大，KV Cache 预分配的空间就越多，能并发的请求数就越少。

```
max-model-len = 4096   → 每个 seq 最多占 ceil(4096/16)=256 Blocks → 能并发更多请求
max-model-len = 32768  → 每个 seq 最多占 2048 Blocks → 并发能力大幅下降
```

**调优建议**：
- 纯对话场景（短文本交互）：`4096` 或 `8192` 就够了
- 文档问答 / RAG 场景（需要塞入大段文档）：`16384` 或 `32768`
- 长论文分析 / 代码仓库理解：`32768` 或更大
- **不要设得比实际需求大太多**，否则浪费显存降低并发能力

#### gpu-memory-utilization：显存的精妙平衡

这个参数控制 vLLM 可以使用多少比例的 GPU 显存：

```
gpu-memory-utilization = 0.90 → 用 90% 显存（留 10% 给 CUDA 运行时开销）
gpu-memory-utilization = 0.95 → 更激进的利用（风险：偶尔 OOM）
gpu-memory-utilization = 0.80 → 保守策略（更稳定但浪费空间）
```

**什么时候该调低？**
- 你同时在 GPU 上跑其他程序（如训练任务、监控 agent）
- 显存接近极限时频繁 OOM
- 使用了需要额外显存的特殊算子

**什么时候可以调高？**
- GPU 专用于 vLLM 推理（没有其他进程争抢）
- 使用了量化模型（权重变小，可以给 KV Cache 更多空间）
- 需要最大化并发数

### 功能开关参数

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--enable-lora` | bool | `false` | 启用 LoRA 适配器支持 |
| `--enable-auto-tool-choice` | bool | `false` | 启用自动工具选择 |
| `--tool-call-parser` | 字符串 | - | 工具调用解析器：`hermes` / `glaive` 等 |
| `--chat-template-path` | 路径 | 自动检测 | 自定义聊天模板路径 |
| `--disable-log-requests` | bool | `false` | 关闭请求日志（高吞吐时推荐开启以减少 I/O） |

### 服务端参数

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--host` | 字符串 | `0.0.0.0` | 监听地址 |
| `--port` | 整数 | `8000` | 监听端口 |
| `--ssl-keyfile` | 路径 | - | SSL 私钥文件路径 |
| `--ssl-certfile` | 路径 | - | SSL 证书文件路径 |

### 一份生产级启动命令模板

综合以上所有参数，这是一份可以直接用于生产环境的完整启动命令：

```bash
#!/bin/bash
# start_vllm_prod.sh — vLLM 生产级启动脚本

MODEL="Qwen/Qwen2.5-7B-Instruct"
HOST="0.0.0.0"
PORT="8000"
TP_SIZE=1          # 张量并行数（根据你的 GPU 数量修改）
MAX_LEN=16384       # 最大上下文长度
GPU_UTIL=0.92       # GPU 显存利用率
MAX_SEQS=128        # 最大并发序列数
SCHED_DELAY=0.2     # 调度延迟因子
DISABLE_LOG=true     # 关闭请求日志

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --max-num-seqs $MAX_SEQS \
    --scheduler-delay-factor $SCHED_DELAY \
    --trust-remote-code \
    $(if $DISABLE_LOG; then echo "--disable-log-requests"; fi) \
    2>&1 | tee vllm_$(date +%Y%m%d_%H%M%S).log
```

---

## 3.3 服务健康检查

服务启动后，第一件事不是急着发请求，而是确认服务真的准备好了。vLLM 提供了一个专门的健康检查端点：

```bash
# 健康检查
curl http://localhost:8000/health

# 正常响应：
# {"status":"ok"}
```

如果返回 `{"status":"ok"}`，说明模型已加载完成、GPU 正常、API Server 已就绪。

其他有用的检查端点：

```bash
# 查看已加载的模型列表
curl http://localhost:8000/v1/models
# 返回：
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "Qwen/Qwen2.5-7B-Instruct",
#       "object": "model",
#       "owned_by": "vllm",
#       "permission": null,
#       "root": "Qwen/Qwen2.5-7B-Instruct"
#     }
#   ]
# }

# 查看 vLLM 版本信息
curl http://localhost:8000/version
# {"version":"0.6.1"}
```

---

## 3.4 用 curl 发送第一个请求

现在让我们发送真正的请求。先从最原始的 curl 开始，这有助于你理解底层的 HTTP 协议细节。

### 基础 Chat Completions 请求

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的技术文档撰写助手。回答要简洁、准确、有条理。"
            },
            {
                "role": "user",
                "content": "什么是 RESTful API？用三个要点解释。"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stop": ["\n\n"]
    }' | python3 -m json.tool
```

返回结果（经过 `json.tool` 格式化）：

```json
{
    "id": "chatcmpl-a1b2c3d4",
    "object": "chat.completion",
    "created": 1736942400,
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "RESTful API 是一种基于 HTTP 协议的软件架构风格，核心思想是将一切抽象为\"资源\"并通过标准 HTTP 方法操作。\n\n**三个要点：**\n\n1. **资源导向设计**：每个 URL 代表一种资源（如 `/users/123`），而非远程过程调用。\n\n2. **统一接口约束**：仅使用 GET（查询）、POST（创建）、PUT（更新）、DELETE（删除）等标准方法，语义清晰。\n\n3. **无状态通信**：服务器不保存客户端上下文，每次请求包含完整信息，便于水平扩展和缓存。"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 28,
        "completion_tokens": 156,
        "total_tokens": 184
    }
}
```

注意几个关键字段的含义：

- **`id`**: 本次请求的唯一标识符（可用于日志追踪）
- **`choices[0].message.content`**: 模型生成的回复正文
- **`choices[0].finish_reason`**: 结束原因 —— `"stop"` 表示正常结束（遇到停止词或生成完毕）；`"length"` 表示达到了 `max_tokens` 上限
- **`usage.prompt_tokens`**: 输入（System + User messages）的总 token 数
- **`usage.completion_tokens`**: 模型生成的 token 数
- **`usage.total_tokens`**: 两者之和（计费依据）

### 流式输出请求

对于交互式应用来说，流式输出才是常态。在 curl 中添加 `"stream": true` 即可启用 SSE（Server-Sent Events）流式模式：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "用一首短诗描述秋天"}
        ],
        "stream": true,
        "temperature": 0.9,
        "max_tokens": 200
    }'
```

你会看到类似这样的流式输出（每个 `data:` 行是一个 SSE 事件）：

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"role":"assistant","content":""},"index":0}],"object":"chat.completion.chunk"}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"秋"},"index":0}],"object":"chat.completion.chunk"}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"风"},"index":0}],"object":"chat.completion.chunk"}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"起"},"index":0}],"object":"chat.completion.chunk"}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"时"},"index":0}],"object":"chat.completion.chunk"}

...（中间省略 N 个 chunk）

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"。"},"index":0}],"object":"chat.completion.chunk"}

data: {"id":"chatcmpl-xxx","choices":[{}],"finish_reason":"stop","object":"chat.completion.chunk"}

data: [DONE]
```

注意观察流式输出的结构特征：
- **第一个 chunk** 包含 `role: "assistant"` 但 `content` 为空字符串
- **后续每个 chunk** 只包含新增的 `content` 片段（可能只有一个字！）
- **倒数第二个 chunk** 的 `finish_reason` 为 `"stop"`
- **最后一个 chunk** 固定为 `[DONE]`，表示流结束

这种设计使得客户端可以实现"逐字显示"的效果——每收到一个 chunk 就立即渲染到屏幕上，用户感知到的延迟就是 TTFT（首个 chunk 到达的时间），而不是等待整个回复完成的时间。

---

## 3.5 Python OpenAI SDK 调用

虽然 curl 能工作，但在真实项目中你会用编程语言的 SDK。好消息是：**vLLM 完全兼容 OpenAI SDK**，这意味着你可以直接用 `openai` 包来调用，无需学习任何新 API。

### 同步调用

```python
"""
vLLM Python SDK 调用示例 — 同步模式
演示：基础对话、参数控制、错误处理
"""

from openai import OpenAI


def chat_example():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed-for-vllm",  # vLLM 不校验 key
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "你是 Python 专家"},
            {"role": "user", "content": "解释 Python 装饰器的原理并给一个实战例子"}
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.1,
    )

    print(f"📝 回复:\n{response.choices[0].message.content}")
    print(f"\n📊 Token 统计:")
    print(f"   Prompt: {response.usage.prompt_tokens} tokens")
    print(f"   Completion: {response.usage.completion_tokens} tokens")
    print(f"   Total: {response.usage.total_tokens} tokens")
    print(f"   Finish reason: {response.choices[0].finish_reason}")


if __name__ == "__main__":
    chat_example()
```

### 异步调用（高并发场景必备）

当你的应用需要同时发起多个请求时，异步客户端能大幅提升效率：

```python
"""
vLLM 异步调用示例 — AsyncOpenAI
适用于高并发批量处理场景
"""

import asyncio
import time
from openai import AsyncOpenAI


async def batch_ask(client: AsyncOpenClient, prompts: list[str]) -> list[str]:
    """并发发送多个请求"""
    tasks = []
    for prompt in prompts:
        task = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content for r in responses]


async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    prompts = [
        "用一句话解释什么是容器化",
        "列举 Kubernetes 的三个核心概念",
        "解释微服务架构的优缺点",
        "什么是 CI/CD？",
    ]

    start = time.time()
    results = await batch_ask(client, prompts)
    elapsed = time.time() - start

    for i, (prompt, result) in enumerate(zip(prompts results), 1):
        print(f"\n{'='*50}")
        print(f"[问题 {i}] {prompt}")
        print(f"[回答] {result[:200]}...")

    print(f"\n⏱️  {len(prompts)} 个请求并行完成，总耗时 {elapsed:.2f}s")
    print(f"   平均每个请求 {elapsed/len(prompts):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

### 流式输出的 Python 实现

```python
"""
vLLM 流式输出 Python 示例
模拟打字机效果逐 Token 显示
"""

from openai import OpenAI
import sys


def streaming_typerwriter():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "写一段关于 AI 未来发展的展望"}],
        stream=True,
        temperature=0.8,
        max_tokens=500,
    )

    full_text = ""
    first_token_time = None
    total_tokens = 0

    print("🤖 ", end="", flush=True)

    for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.role and not delta.content:
            continue

        if delta.content:
            if first_token_time is None:
                import time
                first_token_time = time.time()

            content = delta.content
            print(content, end="", flush=True)
            full_text += content
            total_tokens += 1

    if first_token_time:
        import time
        ttft = (first_token_time - start_time) * 1000
        total_time = (time.time() - start_time) * 1000
        tpot = (total_time - ttft) / max(total_tokens - 1, 1)

        print(f"\n\n{'─'*40}")
        print(f"📈 性能指标:")
        print(f"   TTFT (首Token延迟): {ttft:.0f}ms")
        print(f"   TPOT (每Token间隔): {tpot:.0f}ms")
        print(f"   总 Token 数: {total_tokens}")
        print(f"   总耗时: {total_time:.0f}ms")


if __name__ == "__main__":
    import time
    start_time = time.time()
    streaming_typerwriter()
```

运行效果：

```
🤖 人工智能的未来发展将呈现几个显著趋势。首先，**通用人工智能（AGI）**的探索将加速，
尽管距离真正实现仍有相当长的路要走，但在推理能力、规划能力和自主学习方面会持续突破。
其次，**多模态融合**将成为标配——AI 将不再局限于文本和图像，而是能够理解和生成视频、
3D 内容甚至感官体验的综合系统。第三，**AI 与物理世界的结合**将深化，
机器人技术和具身智能让 AI 从"大脑"进化为完整的智能体...

────────────────────────────────
📈 性能指标:
   TTFT (首Token延迟): 380ms
   TPOT (每Token间隔): 32ms
   总 Token 数: 187
   总耗时: 6340ms
```

这里我们测量到了两个关键的性能指标：
- **TTFT = 380ms**：用户发出请求到看到第一个字的等待时间（用户体验的核心指标）
- **TPOT = 32ms**：每个后续 Token 的平均间隔（决定"打字速度"的感觉）

这两个指标将在后续的性能优化章节中反复出现。

---

## 3.6 vLLM 支持的热门模型速查

vLLM 不是只支持某一个模型，而是一个通用的推理引擎。以下是截至最新版本的模型支持情况：

### 第一梯队：完美支持 ✅

这些模型经过了 vLLM 团队的充分测试和优化，开箱即用：

| 模型 | 参数量 | 特色 | 启动命令示例 |
|:---|:---|:---|:---|
| **Llama 3.1/3.2/3.3** | 8B / 70B / 405B | Meta 旗舰，生态最大 | `--model meta-llama/Meta-Llama-3.1-8B-Instruct` |
| **Qwen2.5** | 0.5B - 72B | 中文 SOTA，超长上下文 | `--model Qwen/Qwen2.5-7B-Instruct` |
| **Mistral / Mixtral** | 7B / 8x7B / 8x22B | MoE 架构，高效 | `--model mistralai/Mistral-7B-Instruct-v0.3` |
| **DeepSeek-V2/V3** | 16B / 671B | 推理+代码能力强 | `--model deepseek-ai/DeepSeek-V2.5` |
| **Gemma 2** | 2B / 9B / 27B | Google 开源 | `--model google/gemma-2-9b-it` |
| **Phi-3** | mini(3.8B) / medium(14B) | 微软小模型 | `--model microsoft/Phi-3-mini-4k-instruct` |

### 第二梯队：良好支持 ⚠️

这些模型基本可用，但可能需要额外参数或存在已知限制：

| 模型 | 注意事项 |
|:---|:---|
| **Yi 系列** (零一万物) | 需要 `--trust-remote-code` |
| **InternLM2** (书生·浦语) | 需要 `--trust-remote-code` |
| **Baichuan2** (百川) | 部分 version 需指定 tokenizer |
| **ChatGLM3/4** (智谱) | 需要 `--trust-remote-code` + 特殊 chat template |
| **Falcon 180B** | 需要多卡 TP |

### 多模态模型（VLM）

vLLM 原生支持视觉语言模型：

```bash
# LLaVA-OneVision（图像理解）
--model llava-hf/llava-onevision-qwen2-7b-ov-hf

# Qwen2-VL（中文多模态）
--model Qwen/Qwen2-VL-7B-Instruct
```

VLM 的调用方式与纯文本模型相同，只是在 `messages` 中传入图片内容（Base64 编码或 URL），具体用法将在第五章多模态章节详解。

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **最简启动** | `python -m vllm.entrypoints.openai.api_server --model <name> --port 8000` 两行搞定 |
| **启动日志解读** | 模型加载时间 → Block 分配数量 → KV Cache 占用 → 最大并发数 → 监听端口 |
| **关键参数 Top 5** | `--model`(必填) / `--tensor-parallel-size`(多卡) / `--max-model-len`(上下文) / `--gpu-memory-utilization`(显存比) / `--trust-remote-code`(国产模型必开) |
| **API 兼容性** | 完全兼容 OpenAI 规范：`/v1/chat/completions` + `stream:true` SSE + `usage` 计费字段 |
| **三种调用方式** | curl（调试验证）→ openai SDK（开发首选）→ AsyncOpenAI（高并发） |
| **性能指标** | TTFT（首 Token 延迟，影响用户体感）+ TPOT（每 Token 间隔，影响打字速度感） |
| **健康检查** | GET `/health` 返回 `{"status":"ok"}` 后方可发请求；GET `/v1/models` 查看已加载模型 |
| **模型生态** | Llama/Qwen/Mistral/DeepSeek/Gemma/Phi 六大家族完美支持；国产模型记得加 `--trust-remote-code` |

> **一句话总结**：vLLM 的启动和使用体验高度接近 OpenAI API——同样的端点格式、同样的请求/响应结构、同样的 SDK 调用方式。唯一不同的是 `base_url` 从 `https://api.openai.com/v1` 变成了 `http://localhost:8000/v1`，以及 `api_key` 可以随便填（vLLM 不做鉴权）。这种"零迁移成本"的设计是 vLLM 能够快速普及的关键原因之一。
