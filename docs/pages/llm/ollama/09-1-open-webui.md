# 09-1 Open WebUI

## 为什么需要 Web UI

到目前为止，我们与 Ollama 的交互方式只有两种：**终端命令行**（`ollama run`）和 **API 调用**（`curl` / Python / JavaScript）。这两种方式对开发者来说足够了，但对于非技术人员——产品经理想试用模型效果、设计师想测试文案风格、老板想快速问一个问题——终端和 API 都是巨大的门槛。

Web UI 填平了这个鸿沟。它让任何人都能通过浏览器与本地大模型交互，就像使用 ChatGPT 一样自然。

```
┌─────────────────────────────────────────────────────────────┐
│              Ollama 生态的用户层级                              │
│                                                             │
│  Layer 0: 纯命令行                                          │
│  ┌─────────────┐                                           │
│  │ $ ollama   │  → 开发者、运维工程师                      │
│  │   run ...   │    需要懂命令行操作                           │
│  └─────────────┘                                           │
│                                                             │
│  Layer 1: API 调用                                         │
│  ┌─────────────┐                                           │
│  │ requests/  │  → 应用开发者、集成工程师                    │
│  │ fetch()   │    需要会写代码                               │
│  └─────────────┘                                           │
│                                                             │
│  Layer 2: Web UI ★ ← 本章重点                             │
│  ┌─────────────┐                                           │
│  │ �────────┐│  → 产品经理、设计师、普通用户                 │
│  │ │ 浏览器   │    无需安装任何东西                          │
│  │ └────────┘│    打开网页就能用                             │
│  └─────────────┘                                           │
│                                                             │
│  💡 每一层都依赖下一层，但每层都扩大了用户群体            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Open WebUI 简介

Open WebUI（原名 Ollama WebUI）是目前功能最丰富、社区最活跃的开源 ChatGPT 风格 Web 界面，对 Ollama 的支持也是所有第三方工具中最好的。

### 核心特性一览

| 功能类别 | 具体能力 |
|---------|---------|
| **对话** | 多模型切换、多轮对话、Markdown 渲染、代码高亮 |
| **RAG** | 文件上传（PDF/图片/代码）、知识库管理、自动检索增强 |
| **图像生成** | DALL·E / Stable Diffusion 集成（需额外配置） |
| **Function Calling** | 工具调用 / 函数执行（需支持 tools 的模型） |
| **Modelfile 支持** | 在界面中创建和管理自定义模型 |
| **用户系统** | 多用户、权限管理、角色控制 |
| **插件系统** | 可扩展的插件架构 |
| **主题定制** | 多种内置主题 + 自定义 CSS |
| **API 兼容** | 同时支持多个后端（Ollama/OpenAI/Azure） |

### 安装方式

#### 方式一：Docker（推荐，最简单）

```bash
# 一键启动，包含所有依赖
docker run -d \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main

# 访问 http://localhost:3000
# 首次访问需要创建管理员账户
```

#### 方式二：pip 安装

```bash
pip install open-webui

# 启动
open-webui

# 默认监听 http://localhost:8080
```

### 初始配置：连接 Ollama

首次打开 Open WebUI 后，你需要告诉它 Ollama 服务在哪里：

```
步骤:
1. 打开 http://localhost:3000
2. 创建管理员账户（设置用户名和密码）
3. 登录后 → 点击右上角头像 → Settings
4. 找到 "Connections" 或 "External Connections"
5. 选择 "Ollama"
6. 填写 URL: http://host.docker.internal:11434
7. 点击 "Save" 或 "Connect"

验证: 左侧边栏应该能看到你已安装的 Ollama 模型列表
```

### 对话功能深度体验

Open WebUI 的对话界面设计得非常接近 ChatGPT：

```markdown
## 对话界面功能

### 基础操作
- **模型选择**: 左侧边栏下拉选择不同模型（qwen2.5:7b / llama3.1:8b 等）
- **新对话**: 点击 "+" 按钮
- **对话历史**: 自动保存，左侧显示历史列表
- **搜索历史**: 支持关键词搜索过往对话

### 高级功能
- **参数调节**: 每次对话可调整 Temperature / Top-P / Max Tokens
- **系统提示词**: 可为每个模型预设不同的 System Prompt
- **Markdown 输出**: 完整支持代码块、表格、列表渲染
- **代码高亮**: 自动识别语言并语法着色
- **流式输出**: 打字机效果，实时显示生成过程
- **复制/重新生成**: 每条回复都可单独操作
- **导出对话**: 支持 Markdown / JSON / 图片 格式导出
```

### RAG 稡式：文档问答

这是 Open WebUI 最强大的功能之一：

```
RAG 使用流程:

1. 点击界面上方的 "#" 图标或 "Documents" 标签
2. 上传你的文档（支持 PDF / TXT / MD / CSV / 图片）
3. 选择 Embedding 模型（如 nomic-embed-text）
4. 选择 LLM 模型（如 qwen2.5:7b）
5. 输入问题 → 系统自动：
   a. 将文档分块
   b. 用 Embedding 模型向量化
   c. 存入内置向量库
   d. 检索相关内容
   e. 注入到 Prompt 中
   f. 用 LLM 生成回答（引用来源标注）

💡 整个过程无需写一行代码！
```

### Modelfile 管理

在 Open WebUI 中直接创建和管理自定义模型：

```
Settings → Workspace → Modelfiles → Create New

填写或粘贴你的 Modelfile 内容:

FROM qwen2.5:7b
SYSTEM """你是专业的技术文档翻译器"""
PARAMETER temperature 0.3

点击 Save → 新模型立即出现在左侧边栏
可以直接像使用官方模型一样运行
```

### 插件系统

Open WebUI 通过插件扩展功能：

```python
# 内置常用插件示例

# 1. 网页搜索插件
# Settings → Plugins → Web Search
# 启用后模型可以联网搜索最新信息

# 2. 图像生成插件
# Settings → Plugins → Image Generation
# 配置 DALL·E API 后即可在对话中生成图片

# 3. 代码解释器
# 任何代码回复都会附带详细的逐行注释

# 4. SQL 查询助手
# 自然语言转 SQL，并在沙箱中安全执行
```

### 生产部署配置

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    volumes:
      - open-webui-data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - WEBUI_SECRET_KEY=your-secret-key-here
      - SCRAPE_BLOCK_BROWSING=True
      - ENABLE_RAG_WEB_SEARCH=True
      - RAG_EMBEDDING_MODEL=nomic-embed-text
      - RAG_LLM_MODEL=qwen2.5:7b
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
    networks:
      - webui-net

networks:
  webui-net:
```

```nginx
# 反向代理配置（添加 HTTPS 和域名绑定）
server {
    listen 443 ssl;
    server_name chat.yourcompany.com;

    ssl_certificate     /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    location / {
        proxy_pass http://open-webui:8080;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

## 与官方 Ollama WebUI 的对比

| 特性 | Open WebUI | Ollama 官方 WebUI |
|------|-----------|---------------|
| 安装复杂度 | 中等（Docker/pip） | **零**（内置） |
| 功能丰富度 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| RAG 能力 | ⭐⭐⭐⭐⭐ | ❌ |
| 用户管理 | ⭐⭐⭐⭐ | ❌ |
| 插件系统 | ⭐⭐⭐⭐ | ❌ |
| Modelfile 支持 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 多后端支持 | ⭐⭐⭐⭐ | ❌ |
| 资源占用 | 较大（Python + 数据库） | 极轻 |
| 适用场景 | 团队共享 / 生产环境 | 快速测试 / 个人使用 |

**选型建议**：
- 个人开发/快速测试 → **Ollama 内置 WebUI**（零配置）
- 团队内部知识库 / 演示给非技术人员 → **Open WebUI**
- 面向用户的正式产品 → **Open WebUI + Nginx + HTTPS**

## 本章小结

这一节全面介绍了 Open WebUI：

1. **Open WebUI 是目前最强的 Ollama 第三方 Web 界面**——功能覆盖对话/RAG/图像/插件/用户管理
2. **Docker 一键部署**是最简单的安装方式：一个 `docker run` 命令
3. **核心价值在于 RAG 模式**——上传文档即可问答，无需编写任何代码
4. **Modelfile 支持让非程序员也能创建和使用自定义模型**
5. **生产部署**需要配合 Nginx 反向代理和 HTTPS
6. **与官方 WebUI 的定位差异**：官方版轻量适合个人使用，Open WebUI 功能更全适合团队协作

下一节我们将介绍 Ollama 官方内置的 Web 界面。
