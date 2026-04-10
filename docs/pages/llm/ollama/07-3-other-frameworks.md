# 07-3 其他框架集成速查

## OpenAI SDK 兼容：万能适配器

在上一节我们提到过 OpenAI SDK 兼容模式是接入 Ollama 的"万能钥匙"。这里做一个更全面的速查：

```python
# === Python (openai 库) ===
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Chat Completion
resp = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Hello"}]
)
print(resp.choices[0].message.content)

# Embedding
emb = client.embeddings.create(model="nomic-embed-text", input="test")
print(len(emb.data[0].embedding))

# === JavaScript / TypeScript (@ai-sdk/openai) ===
import OpenAI from 'openai';

const ollama = new OpenAI({
  baseURL: 'http://localhost:11434/v1',
  apiKey: 'ollama' // required but unused
});

const chat = await ollama.chat.completions.create({
  model: 'qwen2.5:7b',
  messages: [{ role: 'user', content: '你好' }]
});
console.log(chat.choices[0].message.content);
```

**任何支持自定义 `base_url` 的 OpenAI SDK 客户端都可以通过这种方式连接 Ollama**。这是最通用的集成方式。

## Vercel AI SDK

Vercel AI SDK 是 Next.js 生态中最流行的 AI 集成库，它原生支持 Ollama 作为 Provider：

```typescript
// npm install ai @ai-sdk/ollama

import { generateText } from 'ai';
import { ollama } from '@ai-sdk/ollama';

async function main() {
  const { text } = await generateText({
    model: ollama('qwen2.5:7b'),
    prompt: '解释什么是 RAG',
    temperature: 0.4,
    maxTokens: 500,
  });
  
  console.log(text);
}

// 流式输出
import { streamText } from 'ai';

const result = await streamText({
  model: ollama('qwen2.5:7b'),
  prompt: '写一首关于编程的诗',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

## Haystack（deepset）

Haystack 是 deepset 出品的企业级 NLP 框架，对 RAG 有深度支持：

```python
# pip install haystack-ai

from haystack.components.generators import OllamaGenerator
from haystack.components.embedders import OllamaTextEmbedder, OllamaDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack import Pipeline, Document
from haystack.utils import print_answers

# 构建 Pipeline
pipe = Pipeline()

# 添加组件
pipe.add_component("embedder", OllamaDocumentEmbedder(model="nomic-embed-text"))
pipe.add_component("retriever", InMemoryBM25Retriever())
pipe.add_component("generator", OllamaGenerator(model="qwen2.5:7b"))

# 连接组件
pipe.connect("embedder.documents", "retriever.documents")
pipe.connect("retriever", "generator.documents")

# 运行
result = pipe.run(
    data={"generator": {"prompt": "{query}"; \nDocuments:\n{documents}"},
    params={"query": "什么是 Docker？"}
)
```

## Flowise：可视化拖拽式 AI 工作流

Flowise 是一个开源的 LangChain 可视化编排工具，可以通过 UI 界面拖拽节点来构建 LLM 应用，并原生支持 Ollama 节点：

```bash
# 安装和启动 Flowise
npm install -g flowise
npx flowise start

# 访问 http://localhost:3000
# 在界面中:
# 1. 从左侧面板拖入 "Ollama" 节点
# 2. 配置模型名称 (如 qwen2.5:7b) 和 base_url
# 3. 拖入其他节点 (Prompt Template, Output Parser 等)
# 4. 连接各节点
# 5. 点击运行测试
```

Flowise 特别适合不熟悉编程的产品经理或数据分析师——他们可以通过可视化界面快速搭建原型。

## Dify：开源 LLM 应用开发平台

Dify 是一个功能完整的开源 LLM 应用开发平台，内置了 Ollama 支持：

```bash
# Docker 方式部署 Dify
docker compose -f docker-compose.yaml up -d

# 访问 http://localhost/install
# 在设置中:
# 1. 进入"模型供应商"
# 2. 选择 "Ollama"
# 3. 填写 Base URL: http://host.docker.internal:11434
# 4. 测试连接
# 5. 创建应用时选择 Ollama 模型
```

Dify 提供了比 Flowise 更完整的功能：用户管理、日志追踪、版本管理、API 发布等，适合团队协作场景。

## 各框架对比速查表

| 框架 | 接入方式 | 代码量 | 学习曲线 | 适用场景 |
|------|---------|--------|---------|---------|
| **OpenAI SDK** | 改 base_url | ~5 行 | 极低 | 快速验证、通用 |
| **LangChain** | ChatOllama / ChatOpenAI | ~30 行 | 中等 | 通用 LLM 应用、Agent |
| **LlamaIndex** | llama-index-llms-ollama | ~20 行 | 中等 | RAG/知识库专用 |
| **Vercel AI SDK** | @ai-sdk/ollama | ~10 行 | 低 | Next.js Web 应用 |
| **Haystack** | OllamaGenerator | ~25 行 | 中高 | 企业级 NLP 管线 |
| **Flowise** | GUI 拖拽 | 0 行代码 | 极低 | 原型/非技术人员 |
| **Dify** | GUI + API | 0 行代码 | 低 | 团队/生产环境 |

## 本章小结

这一节快速浏览了 Ollama 与各种主流框架的集成方式：

1. **OpenAI SDK 兼容是万能适配器**——几乎所有框架都支持自定义 `base_url`
2. **Vercel AI SDK** 是 Next.js 项目的最佳选择
3. **Haystack** 适合需要企业级 NLP 管道的场景
4. **Flowise 和 Dify** 提供了零代码的可视化方案
5. **选型建议**：
   - 个人项目 / 快速原型 → OpenAI SDK 或 LangChain
   - RAG 知识库 → LlamaIndex
   - Web 应用 → Vercel AI SDK
   - 企业平台 → Dify
   - 不想写代码 → Flowise

下一节我们将讨论如何将这些框架级集成提升到生产级的架构设计。
