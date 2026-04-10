# 其他框架与生态集成

> **白板时间**：除了 LangChain 和 LlamaIndex，vLLM 的 OpenAI 兼容性让它几乎能接入所有主流的 LLM 应用框架和工具链。这一节我们快速覆盖 Haystack、Dify、FastGPT、Flowise、Vercel AI SDK、LiteLLM、One API 等生态工具，让你知道在什么场景下该用什么。

## 一、OpenAI SDK 通用适配（万能钥匙）

### 1.1 核心原理

```
任何支持 base_url 参数的 OpenAI 客户端都能连 vLLM:

┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Client App │────→│  vLLM Server │←────│  OpenAI API  │
│             │     │  :8000       │     │  (兼容)      │
└─────────────┘     └──────────────┘     └─────────────┘
                          ↑
                    所有客户端只需改 base_url!
```

### 1.2 支持的客户端/SDK 列表

| 语言/平台 | 库名 | 安装 | 配置方式 |
|----------|------|------|---------|
| **Python** | `openai` | `pip install openai` | `OpenAI(base_url="...")` |
| **Python (异步)** | `openai` | 同上 | `AsyncOpenAI(base_url="...")` |
| **JavaScript** | `openai` | `npm install openai` | `new OpenAI({baseURL: "..."})` |
| **Go** | `openai-go` | `go get github.com/openai/openai-go` | `OpenAIClient{BaseURL: "..."}` |
| **Java** | `openai-java` | Maven dependency | `OpenAiService.builder().baseUrl("...")` |
| **C# / .NET** | `Azure.AI.OpenAI` | `dotnet add package` | `OpenAIClient(baseUri: "...")` |
| **Rust** | `async-openai` | `cargo add async-openai` | `Client::new().with_base_url("...")` |
| **Ruby** | `ruby-openai` | `gem install ruby-openai` | `OpenAI::Client.new(uri: "...")` |
| **PHP** | `openai-php` | `composer require` | `Client::baseUrl("...")` |

### 1.3 JavaScript / TypeScript 示例

```javascript
// 前端直接调用 vLLM
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'not-needed',  // vLLM 不验证 key
  dangerouslyAllowBrowser: true,  // 开发环境用
});

async function chat(message) {
  const stream = await client.chat.completions.create({
    model: 'Qwen/Qwen2.5-7B-Instruct',
    messages: [{ role: 'user', content: message }],
    stream: true,
    max_tokens: 256,
  });

  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content || '';
    process.stdout.write(text);
  }
}

chat('你好，介绍一下你自己').catch(console.error);
```

### 1.4 Go 示例

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	openai "github.com/openai/openai-go"
)

func main() {
	client := openai.NewClient("http://localhost:8000/v1")
	ctx := context.Background()

	start := time.Now()
	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: "Qwen/Qwen2.5-7B-Instruct",
		Messages: []openai.ChatCompletionMessage{
			{Role: "openai.user", Content: "说你好"},
		},
		MaxTokens: 64,
		Temperature: ptrFloat(0.7),
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("回复: %s\n", resp.Choices[0].Message.Content)
	fmt.Printf("耗时: %v\n", time.Since(start))
}

func ptrFloat(f float32) *float32 { return &f }
```

## 二、统一网关：LiteLLM / One API

### 2.1 LiteLLM — 多模型统一入口

```python
# litellm_config.yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: $OPENAI_API_KEY
  
  - model_name: vllm-qwen
    litellm_params:
      model: openai/qwen
      api_base_url: http://localhost:8000
  
  - model_name: vllm-medical
    litellm_params:
      model: openai/medical-lora
      api_base_url: http://localhost:8000

# 客户端代码不变，通过 LiteLLM 路由到不同后端
from litellm import completion

resp = completion(
    model="vllm-qwen",           # LiteLLM 自动路由到 vLLM
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

### 2.2 One API — API 管理分发平台

One API 是开源的 API 管理/分发平台，支持：
- 多个 AI 提供商统一管理
- Key 分发和限流
- 负载均衡
- 使用量统计

```yaml
# One API 渠道配置示例
channels:
  - name: vllm-local
    type: openai
    base_url: http://localhost:8000/v1
    key: sk-vllm-local
    
models:
  - id: qwen-7b
    channel: vllm-local
    model: Qwen/Qwen2.5-7B-Instruct
    
  - id: qwen-medical
    channel: vllm-local
    model: meta-llama/Llama-3.1-8B-Instruct@medical-lora
```

## 三、应用平台集成

### 3.1 Dify（低代码 AI 平台）

Dify 是开源的 LLM 应用构建平台，可视化编排工作流：

```python
# Dify 中连接 vLLM 的步骤：
#
# 1. 进入「设置」→「模型供应商」
# 2. 添加「OpenAI API 兼容」供应商
# 3. Base URL: http://your-vllm-server:8000/v1
# 4. API Key: any-string (vllm 不验证)
# 5. 模型名: Qwen/Qwen2.5-7B-Instruct
#
# 之后可以在 Dify 中像使用 GPT-4 一样使用 vLLM 后端的模型
```

### 3.2 FastGPT（知识库问答）

FastGPT 是基于 LLM 的知识库问答系统：

```yaml
# fastgpt 配置文件中指定 vLLM
LLM_MODEL: "Qwen/Qwen2.5-7B-Instruct"
LLM_BASE_URL: "http://localhost:8000/v1"
LLM_API_KEY: "not-needed"
LLM_TEMPERATURE: 0.7
EMBEDDING_MODEL: "BAAI/bge-m3"
EMBEDDING_BASE_URL: "http://localhost:8001/v1"
```

### 3.3 Flowise（拖拽式 LangChain UI）

Flowise 是可视化的 LangChain 编排工具：

```
Flowise 连接 vLLM 步骤:

1. 打开 Flowise UI (http://localhost:3000)
2. 左侧面板 → "+" → 选择 "OpenAI" 类型的 Chat Model
3. 配置:
   - Base URL: http://localhost:8000
   - API Key: any-string
   - Model: Qwen/Qwen2.5-7B-Instruct
4. 拖拽组件构建工作流:
   ChatOpenAI → Prompt Template → Output
5. 点击运行测试
```

## 四、前端框架集成

### 4.1 Vercel AI SDK + Next.js

```tsx
// app/api/chat/route.ts
import { streamText } from 'ai';
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: process.env.VLLM_BASE_URL || 'http://localhost:8000/v1',
  apiKey: 'not-needed',  // vLLM 不需要真实 key
});

export async function POST(req: Request) {
  const { messages } = await req.json();
  
  const result = await streamText({
    model: openai('Qwen/Qwen2.5-7B-Instruct'),
    messages,
    async onFinish({ text }) {
      // 可选: 记录对话历史
    },
  });
  
  return result.toDataStreamResponse();
}
```

```tsx
// app/page.tsx (前端)
'use client';

import { useChat } from 'ai/react';

export default function Chat() {
  const { input, setInput, messages, handleSubmit, isLoading } = useChat();

  return (
    <div className="max-w-2xl mx-auto p-4">
      <div className="h-[500px] overflow-y-auto border rounded p-4 mb-4">
        {messages.map(m => (
          <div key={m.id} className={`mb-2 ${m.role === 'user' ? 'text-right' : ''}`}>
            <span className={`inline-block px-3 py-1 rounded-lg ${
              m.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-100'
            }`}>
              {m.content}
            </span>
          </div>
        ))}
      </div>
      
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="输入消息..."
          className="flex-1 border rounded px-3 py-2"
          disabled={isLoading}
        />
        <button 
          type="submit" 
          disabled={isLoading}
          className="bg-blue-500 text-white px-4 py-2 rounded"
        >
          发送
        </button>
      </form>
    </div>
  );
}
```

## 五、监控与可观测性

### 5.1 OpenTelemetry 集成

```python
# 在 vLLM 服务启动前设置 OTel
import os

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://otel-collector:4318"
os.environ["OTEL_SERVICE_NAME"] = "vllm-inference"

# 启动 vLLM 时启用 tracing
# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --enable-opentelemetry \
#     --port 8000
```

---

## 六、总结

本节覆盖了 vLLM 与整个 AI 工具链的集成：

| 工具/平台 | 类型 | 集成方式 | 适用场景 |
|-----------|------|---------|---------|
| **OpenAI SDK (Python/JS/Go)** | 客户端库 | 改 `base_url` | 所有语言的基础接入 |
| **LiteLLM** | 统一网关 | YAML 配置路由 | 多模型统一管理 |
| **One API** | API 管理 | Web UI + YAML | 企业级 API 分发 |
| **LangChain** | 编排框架 | `ChatOpenAI` | Agent / 复杂工作流 |
| **LlamaIndex** | RAG 框架 | `Vllm()` 原生类 | 知识库问答系统 |
| **Dify** | 低代码平台 | UI 配置 | 快速搭建 AI 应用 |
| **FastGPT** | 知识库 QA | 配置文件 | 对话式文档搜索 |
| **Flowise** | 可视化编排 | 拖拽 UI | 无代码 Chain 构建 |
| **Vercel AI SDK** | 前端框架 | React Hook | Next.js 全栈应用 |

**核心要点回顾**：

1. **`base_url` 是唯一的魔法参数**——任何 OpenAI SDK 改这一个值就能连上 vLLM
2. **LiteLLM / One API 是多模型管理的最佳实践**——一个入口管理所有后端
3. **Dify + vLLM = 最快的产品化路径**——非技术人员也能搭建 AI 应用
4. **Next.js + Vercel AI SDK = 最佳前端体验**——流式输出 + SSR 开箱即用
5. **不要重复造轮子**——先看有没有现成工具能满足需求

下一节我们将学习 **生产级架构设计**——把以上所有能力整合成一个可靠的系统。
