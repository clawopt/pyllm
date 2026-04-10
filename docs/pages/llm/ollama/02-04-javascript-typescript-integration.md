# 02-4 JavaScript / TypeScript 接入

## 为什么前端也需要 Ollama？

你可能觉得"Ollama 是后端工具，前端管它干嘛？"——但现代 AI 应用的架构趋势是**智能向前端下沉**。想想看：

- **ChatGPT 的网页版**：纯前端应用，AI 能力全在浏览器里
- **Cursor / Windsurf**：编辑器里的 AI 助手
- **Notion / Obsidian 插件**：笔记软件中的 AI 对话
- **桌面端工具（TypingM）**：原生应用的 AI 集成

这些场景的共同点是：**用户不想打开终端、不想部署后端服务、只想在当前使用的工具里直接获得 AI 能力。** 而 Ollama 恰好提供了 HTTP API，让这一切成为可能。

## Node.js 官方库

```bash
npm install ollama
```

```javascript
// ===== 基础用法 =====

import ollama from "ollama";

// 同步调用
const response = await ollama.chat({
    model: 'qwen2.5:7b',
    messages: [{ role: 'user', content: '什么是闭包？' }],
});

console.log(response.message.content);
// → "闭包是一种编程技术..."

// 流式输出（逐 token 推送）
const stream = await ollama.chat({
    model: 'qwen2.5:7b',
    messages: [{ role: 'user', content: '写一首关于春天的诗' }],
    stream: true,
});

process.stdout.write("模型回复: ");
for await const chunk of stream) {
    if (chunk.message?.content) {
        process.stdout.write(chunk.message.content);
    }
}
console.log("\n");

// Embedding
const embed = await ollama.embeddings({
    model: 'nomic-embed-text',
    input: 'Hello world',
});

console.log(`Embedding dim=${embed.embeddings[0].length}`);
// → Embedding dim=768

// Generate（底层接口）
const gen = await ollama.generate({
    model: 'qwen2.5:7b',
    prompt: 'What is 2+2?',
});
console.log(gen.response);
```

官方库的 API 设计和 Python 版本高度一致，只是语言从 Python 变成了 JavaScript。核心接口 `chat` / `generate` / `embeddings` 三件套全部支持，流式和非流式也都覆盖。

## 浏览器端直接调用（无构建）

对于不想安装 Node.js 的场景，可以直接在浏览器中用 `fetch` 调用 Ollama API。这在以下场景特别有用：

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ollama in Browser</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; }
        #chat-container { border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 400px; }
        .message { margin: 12px 0; padding: 10px 14px; border-radius: 8px; }
        .user { background: #e3f2fd; color: #1a56db; }
        .assistant { background: #f0f0f0; }
        textarea { width: 100%; height: 80px; padding: 8px; font-size: 14px; resize: vertical; }
        button { background: #1a56db; color: white; border: none; padding: 10px 24px; border-radius: 6px; cursor: pointer; font-size: 14px; }
        button:hover { background: #155eae; }
        #status { font-size: 13px; color: #666; margin-top: 8px; min-height: 20px; }
    </style>
</head>
<body>
    <h1>🦙 Ollama Chat (Browser Edition)</h1>
    
    <div id="chat-container"></div>
    <textarea id="input" placeholder="输入你的问题..." rows="3"></textarea>
    <button onclick="sendMessage()">发送</button>
    <div id="status"></div>

    <script>
    const API = "http://localhost:11434";
    let messages = [];
    
    // 注意: 浏览器有 CORS 限制，默认 localhost 不受影响，
    // 但如果从其他端口或域名访问，需要配置 Ollama 的 CORS
    
    async function sendMessage() {
        const input = document.getElementById('input');
        const msg = input.value.trim();
        if (!msg) return;
        
        // 添加用户消息到界面
        addMessage(msg, 'user');
        input.value = '';
        
        // 禁用按钮，显示状态
        document.querySelector('button').disabled = true;
        document.getElementById('status').textContent = '🔄 思考中...';
        
        try {
            const resp = await fetch(`${API}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: 'qwen2.5:7b',
                    messages: [...messages, { role: 'user', content: msg }],
                    stream: true,
                }),
            });
            
            // 流式读取响应
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let assistantMsg = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                // SSE 格式: data: {...}
                for (const line of chunk.split('\n')) {
                    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                        const data = JSON.parse(line.slice(6));
                        if (data.message?.content) {
                            assistantMsg += data.message.content;
                            // 实时更新显示（不等待完成）
                            updateAssistantMessage(assistantMsg);
                        }
                    }
                }
            }
            
            // 完成
            addMessage(assistantMsg, 'assistant');
            messages.push({ role: 'user', content: msg }, { role: 'assistant', content: assistantMsg });
            
            document.getElementById('status').textContent = `✅ 完成 (${assistantMsg.length} 字符)`;
            
        } catch (err) {
            document.getElementById('status').textContent = `❌ 错误: ${err.message}`;
        } finally {
            document.querySelector('button').disabled = false;
            document.getElementById('input').focus();
        }
    }
    
    function addMessage(content, role) {
        const container = document.getElementById('chat-container');
        const div = document.createElement('div');
        div.className = `message ${role}`;
        div.textContent = content;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }
    
    function updateAssistantMessage(text) {
        const msgs = document.querySelectorAll('.message.assistant');
        const last = msgs[msgs.length - 1];
        if (last) last.textContent = text;
        const container = document.getElementById('chat-container');
        container.scrollTop = container.scrollHeight;
    }
    
    // 回车键发送
    document.getElementById('input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    </script>
</body>
</html>
```

把这个 HTML 文件保存为 `ollama-chat.html`，然后在浏览器中直接打开——一个完全不需要任何构建步骤的 AI 聊天界面就完成了。这就是 Ollama + 前端的魅力所在：**零依赖、零构建、双击即用**。

## React 集成：生产级组件

对于更复杂的应用（比如企业内部工具平台），我们需要把 Ollama 调用封装成可复用的 React 组件：

```tsx
// components/OllamaChat.tsx
import { useState, useRef, useCallback } from 'react';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
}

interface UseOllamaReturn {
    response: string;
    stats: {
        inputTokens: number;
        outputTokens: number;
        durationMs: number;
        tokensPerSecond: number;
    };
}

export function useOllama(
    defaultModel: string = 'qwen2.5:7b',
    apiUrl: string = 'http://localhost:11434',
): UseOllamaReturn & {
    const [response, setResponse] = useState('');
    const [stats, setStats] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const abortController = useRef<AbortController | null>(null);
    
    const chat = useCallback(async (
        message: string,
        options?: { model?: string; temperature?: number },
    ): Promise<string> => {
        setIsLoading(true);
        setStats(null);
        
        // 取消上一次请求
        abortController.current?.abort();
        abortController.current = new AbortController();
        
        try {
            const controller = abortController.current;
            const startTime = performance.now();
            
            const resp = await fetch(`${apiUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: options?.model || defaultModel,
                    messages: [{ role: 'user', content: message }],
                    stream: False,
                    signal: controller.signal,
                }),
            });
            
            const data = await resp.json();
            const endTime = performance.now();
            
            const result: UseOllamaReturn = {
                response: data.message.content,
                stats: {
                    inputTokens: data.prompt_eval_count || 0,
                    outputTokens: data.eval_count || 0,
                    durationMs: Math.round(endTime - startTime),
                    tokensPerSecond:
                        data.eval_count && (endTime - startTime) > 0
                            ? Math.round(data.eval_count / ((endTime - startTime) / 1000))
                            : 0,
                },
            };
            
            setResponse(result.response);
            setStats(result.stats);
            return result.response;
            
        } catch (err: any) {
            if (err.name === 'AbortError') return '[已取消]';
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, [defaultModel, apiUrl]);
    
    const chatStream = useCallback(async (
        message: string,
        onChunk: (chunk: string) => void,
        options?: { model?: string; temperature?: number },
    ): Promise<void> => {
        setIsLoading(true);
        
        const resp = await fetch(`${apiUrl}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: options?.model || defaultModel,
                messages: [{ role: 'user', content: message }],
                stream: true,
            }),
        });
        
        const reader = resp.body!.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            for (const line of decoder.decode(value).split('\n')) {
                if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                    const data = JSON.parse(line.slice(6));
                    if (data.message?.content) {
                        onChunk(data.message.content);
                    }
                }
            }
        }
        
        setIsLoading(false);
    }, [defaultModel, apiUrl]);
    
    return {
        response,
        stats,
        isLoading,
        chat,
        chatStream,
    };
}

// ===== 使用示例 =====

function ChatPage() {
    const { response, isLoading, chat } = useOllama('qwen2.5:7b');
    const [messages, setMessages] = useState<Message[]>([]);
    
    const handleSend = async () => {
        const input = /* 从输入框获取 */;
        const answer = await chat(input);
        setMessages(prev => [
            ...prev,
            { role: 'user' as const, content: input, timestamp: Date.now() },
            { role: 'assistant' as const, content: answer, timestamp: Date.now() },
        ]);
    };
    
    return (
        <div className="chat-interface">
            {/* 输入区域 */}
            <div className="messages">
                {messages.map(m => (
                    <div key={m.timestamp} className={`msg ${m.role}`}>{m.content}</div>
                ))}
            </div>
            {/* 状态栏 */}
            {isLoading ? <Spinner /> : null}
            {stats && (
                <div className="stats">
                    {stats.tokensPerSecond.toFixed(1)} t/s |
                    {stats.durationMs}ms
                </div>
            )}
        </div>
    );
}
```

这个 `useOllama Hook` 封装了：
- **状态管理**：loading / response / stats
- **取消机制**：AbortController 支持请求取消
- **双模式**：同步 `chat()` 和流式 `chatStream()`
- **类型安全**：完整的 TypeScript 类型定义

## Next.js / Nuxt SSR 集成

如果你的项目是 Next.js（特别是 App Router），服务端调用 Ollama 有一些特殊考虑：

```typescript
// app/api/chat/route.ts
import { NextRequest } from 'next/server';
import ZAI from '@/lib/zai';  // 或直接 fetch

export async function POST(request: NextRequest) {
    const { message, model = 'qwen2.5:7b' } = await request.json();
    
    // 服务端调用 Ollama —— 注意是在 server component 中执行
    const start = Date.now();
    
    const resp = await fetch('http://localhost:11434/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model,
            messages: [{ role: 'user', content: message }],
            stream: false,
        }),
    });
    
    const data = await resp.json();
    const latency = Date.now() - start;
    
    return Response.json({
        answer: data.message.content,
        model,
        latency_ms: latency,
        tokens: {
            input: data.prompt_eval_count,
            output: data.eval_count,
        },
    });
}
```

Next.js App Router 的优势在于：**API 调用在服务器端完成，客户端只接收最终结果**。这意味着：
- 用户看不到 API 地址（安全性更好）
- 可以做中间处理（日志记录、缓存、RAG 注入）
- SEO 友好（搜索引擎能索引到结果）

## Electron 桌面应用集成

Electron 让你能把 Web 技术打包成跨平台桌面应用。结合 Ollama，你可以做出真正"离线可用"的 AI 工具：

```typescript
// main.ts（Electron 主进程）
import { app, BrowserWindow, ipcMain } from 'electron';
import { handleChat } from './ollama-service';

let mainWindow: BrowserWindow | null = null;

async function createWindow() {
    mainWindow = new BrowserWindow({
        width: 900,
        height: 700,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: true,
        },
    });

    await mainWindow.loadFile('index.html');
}

app.whenReady().then(createWindow);

ipcMain.on('chat:message', (_event, msg) => {
    // 收到渲染进程的消息，转发给主进程处理
    handleChat(msg).then(reply => {
        mainWindow?.webContents.send('chat:reply', reply);
    });
});
```

```typescript
// ollama-service.ts（IPC 通信层）
import fetch from 'node-fetch';

const API = process.env.OLLAMA_URL || 'http://localhost:11434';

export async function handleChat(message: string): Promise<string> {
    const resp = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: 'qwen2.5:7b',
            messages: [{ role: 'user', content: message }],
            stream: false,
        }),
    });
    
    const data = await resp.json();
    return data.message.content;
}
```

这个架构的核心思路是：**Electron 主进程负责 UI 渲染和窗口管理，通过 IPC 把消息发送给后台 worker（或直接 HTTP 调用 Ollama），拿到结果后再更新 UI**。这样即使 Ollama 在做耗时较长的推理，UI 也不会卡顿。

## VS Code 扩展开发预览

VS Code 是开发者最常驻留的工具，在里面集成 Ollama 作为 AI 编程助手是非常自然的需求。VS Code 扩展可以用 TypeScript 开发：

```json
// package.json
{
    "name": "ollama-vscode",
    "displayName": "Ollama Assistant",
    "version": "1.0.0",
    "engines": ["*"],
    "categories": ["Other", "Programming Languages"],
    "activationEvents": [],
    "main": "./out/extension.js",
    "contributes": {
        "commands": [
            {
                "command": "ollama.ask",
                "title": "Ask Ollama",
                "command": "ollama.ask",
            },
            {
                "command": "ollama.explain",
                "title": "Explain Selected Code",
                "command": "ollama.explain",
            },
            {
                "command": "ollama.review",
                "title": "Review Current File",
                "command": "ollama.review",
            },
        ],
        "slashCommands": [
            {
                "name": "ollama.ask",
                "description": "Ask Ollama a question",
                "command": "ollama.ask ${input}",
            },
        ],
    },
}
```

```typescript
// src/extension.ts
import * as vscode from 'vscode';
import fetch from 'node-fetch';

const MODEL = 'qwen2.5:7b';
const API = 'http://localhost:11434';

async function askOllama(question: string): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        void vscode.showErrorMessage('没有打开的文件');
        return;
    }
    
    const selection = editor.selection;
    const selectedText = selection.document.getText();  // 用户选中的代码
    
    const prompt = selectedText || question || '请帮我解释这段代码';
    
    const resp = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: MODEL,
            messages: [
                { role: 'system', content: '你是 VS Code 的 AI 助手。请简洁回答，直接给出答案，不要多余解释。' },
                { role: 'user', content: `${prompt}\n\`\`\`\n${selectedText}\n\`\`\``` },
            ],
            stream: false,
        }),
    });
    
    const data = await resp.json();
    const reply = data.message.content;
    
    // 在新建文档中展示回答
    const doc = await vscode.workspace.openTextDocument({
        content: `# Ollama 回复\n\n${reply}`,
        language: 'plaintext',
    });
    
    await vscode.window.showTextDocument(doc.uri);
}
```

这个扩展让开发者可以：
- 选中代码后按快捷键直接问"这段代码有什么问题"
- 用 `/ollama.ask` 斜开侧边栏进行对话
- 用 `/ollama.explain` 让 AI 解释选中代码
- 所有操作都在 VS Code 内完成，不需要切换窗口

## 常见前端集成陷阱

### CORS 跨域问题

这是浏览器端调用 Ollama 最常遇到的问题：

```bash
# 现象：你的 Web 应用运行在 http://localhost:3000
# Ollama 运行在 http://localhost:11434
# 浏览器向 11434 发起 fetch → CORS preflight 失败

# 解决方案一（推荐用于开发环境）：
# Ollama 默认允许所有来源的请求（*），所以 localhost 不受限制
# 但如果你修改了 OLLAMA_HOST 为非 localhost 地址，就需要处理 CORS

# 解决方案二（生产环境）：通过反向代理统一处理
# Nginx 配置:
location /v1/ {
    proxy_pass http://127.0.0.1:11434;
    add_header Access-Control-Allow-Origin *;
    add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
    add_header Access-Control-Allow-Headers "Content-Type";
}
```

### 大响应体阻塞 UI

```javascript
// ❌ 问题：一次性返回完整响应导致 UI 卡住
const resp = await fetch(url, { ... });
const data = await resp.json();  // 如果模型思考了 30 秒，这里就卡 30 秒

// ✅ 正确：始终使用流式
const resp = fetch(url, { ..., stream: true });
const reader = resp.body.getReader();
// 逐 token 更新 UI，用户立即看到内容开始出现
```

### 连接状态管理

```javascript
// ❌ 问题：每次请求都创建新连接（HTTP/1.1 无状态）
// 对于高频场景（如 IDE 自动补全，每秒可能触发多次），连接开销很大

// ✅ 正确：保持长连接（HTTP/2 或 WebSocket）
// Ollama 本身支持 keep-alive
// 前端可以通过 connection pooling 复用连接
// 或者使用 SSE 的自动重连机制
```

本章我们涵盖了从浏览器原生 fetch 到 React/Vue/Next.js/Electron/VS Code 全栈的前端接入方案。下一章将快速覆盖 Go/Java/Rust 等其他语言的接入方法。
