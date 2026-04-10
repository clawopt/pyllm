# 02-2 RESTful API 完整指南

## API 的三种核心模式

Ollama 的 API 虽然端点不多，但每个端点都承载着不同的设计理念。理解这些理念有助于你在实际开发中选择正确的接口。

```
┌───────────────────────────────────────────────────────────────┐
│                  Ollama API 三大模式                        │
│                                                            │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ /api/chat   │   │ /api/generate │   │ /api/embeddings│   │
│  │             │   │              │   │               │   │
│  │ 对话式      │   │ 补全式        │   │ 向量化         │   │
│  │ 多轮上下文   │   │ 单轮补全     │   │ 纯输入输出     │   │
│  │ 流式友好    │   │ 底层灵活     │   │ RAG 基础       │   │
│  └──────┬──────┘   └──────┬───────┘   └──────┬────────┘    │
│         │                 │                  │             │
│         ▼                 ▼                  ▼             │
│  ┌─────────────────────────────────────────────────┐       │
│  │            你的应用层                           │       │
│  │  聊天机器人 / 对话系统 / Agent                   │       │
│  │  文本生成 / 翻译 / 摘要                          │       │
│  └─────────────────────────────────────────────────┘       │
└───────────────────────────────────────────────────────────────┘
```

简单来说：
- **`/api/chat`**：90% 的场景用这个就够了。它封装了完整的对话逻辑（自动拼接 system prompt、维护历史、处理多模态），是最高层的抽象。
- **`/api/generate`**：当你需要更底层的控制时用。比如你想自己管理对话格式、传入原始 prompt 而不是 message 数组、或者做一些 `/api/chat` 不支持的高级操作。
- **`/api/embeddings`**：完全独立的用途——把文本变成向量，用于搜索和 RAG。

下面我们逐一深入。

## Chat Completions 接口（核心）

这是 Ollama 最重要也是使用频率最高的接口。

### 基本请求与响应

```python
import requests
import json

API_BASE = "http://localhost:11434"

def chat_basic():
    """最简单的聊天请求"""
    
    response = requests.post(
        f"{API_BASE}/api/chat",
        json={
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "user", "content": "什么是递归？"}
            ],
            "stream": False,  # 非流式：一次性返回完整回答
        },
        timeout=60,
    )
    
    result = response.json()
    
    print("=== 完整响应结构 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    # 输出类似:
    # {
    #   "model": "qwen2.5:7b",
    #   "created_at": "2026-04-08T10:00:00Z",
    #   "message": {
    #     "role": "assistant",
    #     "content": "递归是一种编程技术..."
    #   },
    #   "done": true,
    #   "done_reason": "stop",
    #   "total_duration": 1234567890,
    #   "prompt_eval_count": 12,
    #   "prompt_eval_duration": 120000000,
    #   "eval_count": 45,
    #   "eval_duration": 1100000000
    # }
    
    # 提取回答内容
    answer = result["message"]["content"]
    print(f"\n模型回答: {answer}")
    
    # 性能指标
    total_ms = result["total_duration"] / 1_000_000
    input_tokens = result.get("prompt_eval_count", 0)
    output_tokens = result.get("eval_count", 0)
    print(f"耗时: {total_ms:.0f}ms | 输入: {input_tokens} tokens | 输出: {output_tokens} tokens")


chat_basic()
```

### 请求体字段详解

`/api/chat` 的请求体虽然看起来只有几个字段，但每个字段都有值得深挖的细节：

```python
def chat_full_options_demo():
    """展示所有可用参数"""
    
    payload = {
        # ===== 必填字段 =====
        
        "model": "qwen2.5:7b",
        # 模型名称。可以是 ollama list 中显示的任何模型名
        
        "messages": [
            # 消息数组。每条消息有 role 和 content 两个字段
            {"role": "system", "content": "你是一个Python专家"},
            {"role": "user", "content": "写一个快速排序"},
            
            # 注意: 第一条消息的 role 通常应该是 "user" 或 "system"
            # 如果第一条是 "assistant"，某些模型可能行为异常
            
            # 多轮对话示例:
            {"role": "assistant", "content": "好的，这是快速排序代码..."},
            {"role": "user", "content": "请解释第3行的作用"},
            # Ollama 会自动将完整 messages 数组作为上下文送入模型
        ],
        
        # ===== 可选字段 =====
        
        "stream": False,
        # 是否使用流式输出（SSE）
        # False: 一次性返回完整 JSON（适合脚本/批处理）
        # True:  返回 SSE 流，逐 token 推送（适合实时交互）
        
        "options": {
            # 推理参数覆盖（可选）
            # 这些值会临时覆盖模型的默认设置
            # 不传则使用 Modelfile 中定义的默认值或模型内置默认值
            
            "temperature": 0.7,
            # 采样温度。0=确定性，>1=更随机
            
            "top_k": 40,
            # Top-K 采样：只从概率最高的 K 个 token 中选择
            
            "top_p": 0.9,
            # Top-P（核采样）：累积概率达到 P 的最小集合
            
            "num_ctx": 4096,
            # 上下文窗口长度。超过此长度的内容会被截断
            
            "num_predict": 512,
            # 最大生成长度。-1 表示不限制
            
            "repeat_penalty": 1.1,
            # 重复惩罚系数。>1 抑制重复，<1 鼓励重复
            
            "seed": 42,
            # 随机种子。固定 seed 可复现输出
            
            "stop": ["\n\n", "Human:", "Assistant:"],
            # 自定义停止序列。遇到其中任何一个就停止生成
        },
        
        "tools": None,
        # 工具调用定义（Function Calling）
        # 仅支持 tool-calling 能力的模型（如 Llama 3.1, Qwen 2.5）
        # 详细用法在后续章节展开
        
        "format": "",
        # 强制输出格式: "json" → 尝试输出合法 JSON
        # 仅部分模型支持
    }
    
    return payload


def chat_with_tools():
    """Function Calling 示例（需要支持的模型）"""
    
    payload = {
        "model": "qwen2.5:7b",  # 或 llama3.1:8b 等
        "messages": [
            {"role": "user", "content": "北京今天天气怎么样？"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的当前天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ],
        "stream": False,
    }
    
    resp = requests.post(f"{API_BASE}/api/chat", json=payload, timeout=30)
    result = resp.json()
    
    # Function Calling 的响应中，message.content 可能为空
    # 取而代之的是 tool_calls 字段
    if "tool_calls" in result.get("message", {}):
        for tc in result["message"]["tool_calls"]:
            func_name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            print(f"🔧 模型想调用工具: {func_name}({args})")
            # 你需要在代码里执行实际的函数调用
            # 然后把结果通过新消息发回模型
    else:
        print(f"普通回复: {result['message']['content']}")


chat_with_tools()
```

### 响应体字段详解

每次 `/api/chat` 调用的响应中都包含丰富的元数据。即使你只关心 `message.content`，了解这些元数据对调试和性能优化也很有价值：

| 字段 | 类型 | 含义 | 使用场景 |
|------|------|------|---------|
| `model` | string | 实际使用的模型名 | 日志记录 |
| `created_at` | ISO 8601 时间戳 | 请求时间追踪 |
| `message.role` | `"assistant"` | 固定值 | 格式统一 |
| `message.content` | string | 模型的文字回答 | **这是你最需要的字段** |
| `message.tool_calls` | array/null | 工具调用请求 | Function Calling |
| `images` | array/null | 生成的图片数据 | 多模态输出 |
| `done` | bool | 是否生成完毕 | 流式模式下判断结束 |
| `done_reason` | string | 停止原因 | `"stop"` / `"length"` / `"eof"` |
| `total_duration` | nanoseconds | 总耗时（含排队+推理） | 性能基准 |
| `load_duration` | nanoseconds | 模型加载耗时 | 首次请求特别关注 |
| `prompt_eval_count` | int | 输入 token 数 | 计费参考 |
| `prompt_eval_duration` | nanoseconds | 输入处理耗时 | 瓶颈分析 |
| `eval_count` | int | 输出 token 数 | 计费参考 |
| `eval_duration` | nanoseconds | 输出生成耗时 | 瓶颈分析 |

### 流式输出（Streaming）

对于交互式应用（聊天界面、IDE 助手），用户不想等上十秒才看到全部回复——他们希望像 ChatGPT 那样看到文字一个字一个字地蹦出来。这就是流式输出的价值：

```python
def chat_streaming():
    """流式输出：逐 token 实时返回"""
    
    response = requests.post(
        f"{API_BASE}/api/chat",
        json={
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "user", "content": "用三句话解释什么是机器学习"}
            ],
            "stream": True,  # 关键：开启流式
        },
        stream=True,  # requests 库的流式参数
        timeout=60,
    )
    
    print("模型回复:", end=" ", flush=True)
    
    full_response = ""
    
    for line in response.iter_lines():
        if not line.strip():
            continue
        
        # 每一行是一个 SSE (Server-Sent Event) 数据块
        # 格式: data: {...JSON...}
        if line.startswith("data: "):
            data_str = line[6:]  # 去掉 "data: " 前缀
            
            if data_str == "[DONE]":
                print("\n\n✅ 生成完成")
                break
            
            try:
                chunk = json.loads(data_str)
                
                # 流式 chunk 的结构与完整响应略有不同
                # message 可能只包含 content 的增量（delta）
                delta = chunk.get("message", {}).get("content", "")
                
                if delta:
                    print(delta, end="", flush=True)  # 逐字打印，无缓冲
                    full_response += delta
                
                # 其他元数据只在最后一个 chunk 或 done 时出现
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
                    total_ms = chunk.get("total_duration", 0) // 1e6
                    print(f"\n\n[统计] {eval_count} tokens, {total_ms:.0f}ms")
                    
            except json.JSONDecodeError:
                # 偶尔会有非 JSON 行（如 :ping 心跳包）
                continue
    
    return full_response


result = chat_streaming()
print(f"\n完整回复长度: {len(result)} 字符")
```

**流式的关键细节：**

1. **`stream=True` 要设两个地方**：Ollama API 的 `stream: true` 和 Python requests 的 `stream=True`
2. **SSE 协议**：Ollama 使用 Server-Sent Events 标准，每行以 `data: ` 开头，以 `[DONE]` 结束
3. **增量拼接**：每个 chunk 的 `message.content` 是增量（新增的文字），你需要自行累加得到完整回复
4. **心跳包**：偶尔会收到空行或 `: ping` 行，需要跳过不能当 JSON 解析
5. **元数据位置**：`done=true` 的那个 chunk 才包含完整的统计信息

### 错误处理

```python
def chat_with_error_handling(question, model="qwen2.5:7b"):
    """带完整错误处理的聊天函数"""
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": question}]},
            timeout=30,
        )
        response.raise_for_status()  # 非 2xx 状态码抛异常
        
        result = response.json()
        return result["message"]["content"]
        
    except requests.exceptions.Timeout:
        return f"⏰ 请求超时（{model} 可能正在加载或问题太复杂）"
    
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到 Ollama 服务。确认 ollama serve 正在运行？"
    
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        if status == 404:
            return f"❌ 模型 '{model}' 不存在。先用 ollama pull {model}"
        elif status == 500:
            body = e.response.text
            return f"⚠️ 服务器内部错误: {body[:200]}"
        else:
            return f"❌ HTTP {status}: {e.response.text[:200]}"
    
    except json.JSONDecodeError:
        return "❌ 响应不是有效的 JSON（可能是服务版本不兼容）"
    
    except Exception as e:
        return f"❌ 未预期错误: {type(e).__name__}: {e}"


# 测试各种错误场景
print(chat_with_error_handling("你好"))                              # 正常
print(chat_with_error_handling("你好", model="nonexist-model"))    # 404
# print(chat_with_error_handling("你好" * 10000))                   # 超时
```

## Generate 接口（底层补全）

如果说 `/api/chat` 是"帮你打包好了所有细节"的高级接口，那 `/api/generate` 就是"给你最大的灵活性"的底层接口。两者的关系可以这样理解：

```
/api/chat 内部做了什么？
  1. 接收 messages 数组
  2. 将 messages 转换为内部 prompt 格式（加上 system + history）
  3. 自动添加 template（<|begin_of_text|> 等）
  4. 调用 /api/generate
  5. 将 generate 的原始输出解析为 message 格式返回

/api/generate 则跳过了步骤 1-3 和 5，
让你完全控制 prompt 的构造方式。
```

### Generate 与 Chat 的核心区别

```python
def compare_chat_vs_generate():
    """对比同一个任务用两种接口的实现差异"""
    
    question = "将以下 JSON 转为 YAML:\n{\"name\":\"Alice\",\"age\":25,\"skills\":[\"Python\",\"Go\"]}"
    
    print("=" * 60)
    print("方式一：/api/chat（推荐）")
    print("=" * 60)
    
    # Chat 方式：你只需要关心"谁说了什么"
    chat_resp = requests.post(f"{API_BASE}/api/chat", json={
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "你是格式转换助手"},
            {"role": "user", "content": question},
        ],
        "stream": False,
    }, timeout=30).json()
    print(f"回答:\n{chat_resp['message']['content']}\n")
    
    print("=" * 60)
    print("方式二：/api/generate（底层）")
    print("=" * 60)
    
    # Generate 方式：你需要自己组装完整的 prompt
    generate_resp = requests.post(f"{API_BASE}/api/generate", json={
        "model": "qwen2.5:7b",
        "prompt": f"""<<SYS
你是一个格式转换助手。严格按要求的格式输出，不要有多余文字。
<|im_start|>assistant
<|im_end|>
SYS
{question}
""",  # 注意：这里需要手动拼好完整的 prompt！
        "stream": False,
        "options": {
            "temperature": 0.1,  # 格式转换任务用低 temperature 更准确
            "num_predict": 256,
        }
    }, timeout=30).json()
    print(f"回答:\n{generate_resp['response']}\n")
    
    print("=" * 60)
    print("差异总结:")
    print("- chat: messages 数组 → 自动拼接 prompt → 返回 message 对象")
    print("- generate: 手写完整 prompt 字符串 → 返回 response 纯文本")
    print("- generate 支持 images 参数（多模态输入）和 template 参数（自定义模板）")
    print("- generate 更适合：RAG 注入（直接控制 context 位置）/ 特殊格式输出 / 原始实验")


compare_chat_vs_generate()
```

### Generate 的独有能力

有几个事情只有 `/api/generate` 能做：

**1. 多模态图像输入**

```python
def generate_with_image():
    """Generate 接口支持直接传入图片"""
    
    import base64
    from pathlib import Path
    
    # 读取本地图片并编码为 base64
    image_path = Path("/Users/you/screenshot.png")
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    
    response = requests.post(f"{API_BASE}/api/generate", json={
        "model": "llava:13b",  # 视觉语言模型
        "prompt": "Describe this image in detail:",
        "images": [image_b64],  # ← 只有 generate 支持图片输入！
        "stream": False,
    }, timeout=60)
    
    result = response.json()
    print(result["response"])


# Chat 接口目前不支持直接传图（需要先将图片转为 base64 URL）
```

**2. 自定义模板**

```python
def generate_with_custom_template():
    """完全控制提示词模板"""
    
    response = requests.post(f"{API_BASE}/api/generate", json={
        "model": "qwen2.5:7b",
        "template": "{{ .System }}{{ .Prompt }}{{ .Response }}",  # 自定义模板
        "system": "You are a helpful assistant.",
        "prompt": "What is 2+2?",
        "raw": True,  # raw=True 时返回原始 token 序列而非文本
        "stream": False,
    }, timeout=30)
    
    result = response.json()
    if result.get("raw"):
        print(f"Raw tokens: {result['response'][:100]}...")
        # raw 模式返回的是 token ID 列表，可用于高级分析


generate_with_custom_template()
```

**3. Context 管理**

```python
def generate_with_context():
    """Generate 支持传入之前的上下文实现多轮对话"""
    
    # 第一次请求
    resp1 = requests.post(f"{API_BASE}/api/generate", json={
        "model": "qwen2.5:7b",
        "prompt": "我叫小明",
        "stream": False,
    }).json()
    context = resp1.get("context")  # 获取上下文标识符
    
    # 第二次请求：带上上下文
    resp2 = requests.post(f"{API_BASE}/api/generate", json={
        "model": "qwen2.5:7b",
        "prompt": "我叫什么名字？",
        "context": context,  # ← 关键：传入上一轮的上下文
        "stream": False,
    }).json()
    
    print(resp2["response"])
    # 应该能正确回答"你叫小明"，因为上下文包含了第一轮对话


generate_with_context()
```

注意 `context` 字段：它是 Ollama 内部的 KV Cache 句柄。通过传递它，你可以在多次 `generate` 调用之间保持对话状态——这在构建自定义对话系统时非常有用。但要注意 `context` 只在同一个模型内有效，切换模型后 context 失效。

## Embeddings 接口

Embedding 接口的用途和前两者完全不同——它不做"生成"，而是做"转换"：把一段文本变成一个固定长度的数值向量。

```python
def embedding_demo():
    """文本向量化基础演示"""
    
    texts = [
        "I love programming",
        "我喜欢编程",
        "The weather is nice today",
        "天气真好",
    ]
    
    embeddings = []
    
    for text in texts:
        resp = requests.post(f"{API_BASE}/api/embeddings", json={
            "model": "nomic-embed-text",  # 嵌入模型
            "input": text,
        }, timeout=30).json()
        
        embed = resp["embedding"]
        embeddings.append(embed)
        
        print(f"'{text}' → [{embed[0]:.4f}, ..., {embed[-1]:.4f}] (维度={len(embed)})")
    
    # 核心用途：计算语义相似度
    import numpy as np
    
    emb_a = np.array(embeddings[0])
    emb_b = np.array(embeddings[1])  # "I love programming" vs "我喜欢编程"
    emb_c = np.array(embeddings[2])  # vs "The weather..."
    
    sim_ab = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    sim_ac = np.dot(emb_a, emb_c) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_c))
    
    print(f"\n相似度比较:")
    print(f"  'I love programming' ↔ '我喜欢编程': {sim_ab:.4f}  (语义相近 → 高)")
    print(f"  'I love programming' ↔ 'The weather...': {sim_ac:.4f}  (语义无关 → 低)")
    
    # 这就是 RAG（检索增强生成）的基础：
    # 高相似度 → 文档相关 → 作为上下文送给 LLM → 准确回答


embedding_demo()
```

关于 Embedding 模型的选择和使用，我们会在第六章深入展开。这里先记住一点：**Embedding 的质量直接决定了 RAG 系统的上限**——如果向量化器无法区分"苹果是一种水果"和"Apple 是一家公司"这两种含义，那么检索就会出错，后面的 LLM 再强也没用。

## Models 辅助接口

除了上面三个核心操作接口外，还有几个辅助接口在日常开发中很有用：

```python
def auxiliary_apis_demo():
    """辅助接口速查"""
    
    # === 1. 列出模型 ===
    models = requests.get(f"{API_BASE}/api/tags").json()
    print(f"已安装模型数: {len(models.get('models', []))}")
    for m in models.get("models", []):
        size_mb = m.get("size", 0) // (1024*1024)
        print(f"  - {m['name']:<35} {size_mb:>8} MB")
    
    # === 2. 查看模型详情 ===
    details = requests.post(f"{API_BASE}/api/show", json={
        "model": "qwen2.5:7b",
    }).json()
    print(f"\n模型详情:")
    print(f"  架构: {details.get('modelfile', {}).get('architecture')}")
    print(f"  参数量: {details.get('modelfile', {}).get('parameters')}")
    print(f"  上下文: {details.get('modelfile', {}).get('context_length')}")
    
    # === 3. 复制模型 ===
    copy_resp = requests.post(f"{API_BASE}/api/copy", json={
        "source": "qwen2.5:7b",
        "destination": "my-qwen-copy",
    }).json()
    print(f"\n复制结果: {copy_resp.get('status')} → 现在可以用 my-qwen-copy 了")
    
    # === 4. 远程拉取模型 ===
    pull_resp = requests.post(f"{API_BASE}/api/pull", json={
        "name": "deepseek-v2:16b:q4_K_M",
    }).json()
    print(f"拉取状态: {pull_resp.get('status')}")
    print(f"拉取详情: {pull_resp.get('status')}")
    
    # === 5. 删除模型 ===
    del_resp = requests.delete(f"{API_BASE}/api/blob/sha256-xxxxx", json={
        "name": "test-model-to-delete",
    })
    print(f"\n删除 HTTP 状态: {del_resp.status_code}")


auxiliary_apis_demo()
```

这些辅助接口让程序化的模型管理成为可能——你可以通过 API 完成之前只能在终端做的所有操作（pull/rm/cp/show），这意味着你的应用可以自带"模型市场"功能。

## 性能基准测试框架

最后，让我们搭建一个简单的性能测试工具，这在你评估不同模型或调优参数时会反复用到：

```python
import time
import statistics
import requests
import json

API = "http://localhost:11434"

def benchmark_model(model, prompts, runs=3):
    """
    对指定模型执行性能基准测试
    
    Args:
        model: 模型名称
        prompts: 测试问题列表
        runs: 每个问题重复次数（取平均减少偶然误差）
    
    Returns:
        包含详细统计信息的字典
    """
    
    results = []
    
    for prompt in prompts:
        run_times = []
        input_tokens_list = []
        output_tokens_list = []
        
        for _ in range(runs):
            start = time.perf_counter()
            
            resp = requests.post(f"{API}/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }, timeout=120)
            
            elapsed = time.perf_counter() - start
            data = resp.json()
            
            run_times.append(elapsed)
            input_tokens_list.append(data.get("prompt_eval_count", 0))
            output_tokens_list.append(data.get("eval_count", 0))
        
        results.append({
            "prompt": prompt[:50] + "...",
            "avg_time_s": statistics.mean(run_times),
            "min_time_s": min(run_times),
            "max_time_s": max(run_times),
            "p50_time_s": statistics.median(run_times),
            "avg_input_tokens": statistics.mean(input_tokens_list),
            "avg_output_tokens": statistics.mean(output_tokens_list),
            "tokens_per_second": (
                statistics.mean(output_tokens_list) / statistics.mean(run_times)
                if statistics.mean(run_times) > 0 else 0
            ),
        })
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"  Benchmark: {model}")
    print(f"  Prompts: {len(prompts)} × {runs} runs each")
    print(f"{'='*60}")
    print(f"{'Prompt':<35} {'Time(s)':>8} {'In Tok':>7} {'Out Tok':>8} {'t/s':>7}")
    print(f"{'-'*60}")
    
    for r in results:
        print(f"{r['prompt']:<35}{r['avg_time_s']:>8.2f}{r['avg_input_tokens']:>7}{r['avg_output_tokens']:>8}{r['tokens_per_second']:>7.1f}")
    
    total_avg = statistics.mean([r["avg_time_s"] for r in results])
    total_tps = statistics.mean([r["tokens_per_second"] for r in results])
    print(f"{'-'*60}")
    print(f"{'TOTAL AVERAGE':<35}{total_avg:>8.2f}{'':>7}{'':>8}{total_tps:>7.1f}")
    
    return results


# 运行基准测试
if __name__ == "__main__":
    benchmark_model(
        "qwen2.5:7b",
        prompts=[
            "什么是递归？用一句话解释",
            "写一个 Python 快速排序",
            "将 [1,3,5,2,4] 从小到大排序",
            "Explain the difference between process and thread",
            "用中文写一首五言绝句",
        ],
        runs=3,
    )
```

运行这个脚本你会得到类似这样的输出：

```
************************************************************
  Benchmark: qwen2.5:7b
  Prompts: 5 × 3 runs each
************************************************************
Prompt                               Time(s)  In Tok  Out Tok    t/s
--------------------------------------------------------------
什么是递归？用一句话解释...           1.23     12      45      36.5
写一个 Python 快速排序...              2.56     18     156      60.9
将 [1,3,5,2,4] 从小到大排序...          0.89      8       12      13.5
Explain the difference between...        3.12     22     198      63.5
用中文写一首五言绝句...              0.78      6       28      35.9
--------------------------------------------------------------
TOTAL AVERAGE                       1.72     13.2    87.8    51.0
```

这些数据就是你在选型时的决策依据——比如你可以清楚地看到"代码生成类任务比问答类任务慢 2 倍"、"中文输出比英文输出略慢"这类规律。

本章介绍了 API 服务的启动配置和三个核心接口。下一章我们将把这些 API 调用封装成真正好用的 Python/JavaScript 应用。
