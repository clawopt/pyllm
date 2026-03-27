# 本地模型部署（Ollama）

如果你对数据隐私有极高要求，或者需要在断网环境下使用OpenClaw，部署本地模型是最佳选择。这一章，我们介绍如何使用Ollama部署和运行本地大模型。

## 为什么选择本地模型

**优势：**

| 特性 | 云端模型 | 本地模型 |
|------|---------|---------|
| 数据隐私 | 数据上传云端 | 数据不出本地 |
| 网络依赖 | 需要稳定网络 | 完全离线可用 |
| 使用成本 | 按量付费 | 一次性硬件投入 |
| 使用限制 | 有API限流 | 无限制 |
| 响应速度 | 受网络影响 | 本地计算，稳定 |

**劣势：**

| 特性 | 云端模型 | 本地模型 |
|------|---------|---------|
| 硬件要求 | 无 | 较高（GPU推荐） |
| 模型能力 | 最强 | 相对较弱 |
| 维护成本 | 无 | 需要自行维护 |
| 首次部署 | 简单 | 需要下载模型 |

**适用场景：**

- 处理敏感数据（财务、医疗、法律）
- 离线环境（内网、出差、户外）
- 大量重复任务（避免API成本）
- 学习和研究（了解模型原理）

## Ollama安装与守护进程启动

Ollama是目前最流行的本地模型运行工具，支持一键安装和运行。

### 安装Ollama

**MacOS：**

```bash
# 方式一：官网下载
# 访问 https://ollama.com/download 下载安装包

# 方式二：Homebrew安装
brew install ollama
```

**Linux：**

```bash
# 一键安装
curl -fsSL https://ollama.com/install.sh | sh

# 或手动安装
# 1. 下载最新版本
curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama
chmod +x /usr/local/bin/ollama
```

**Windows：**

```powershell
# 访问 https://ollama.com/download 下载安装包
# 或使用winget
winget install Ollama.Ollama
```

**验证安装：**

```bash
ollama --version
# 输出：ollama version is 0.1.xx
```

### 启动Ollama服务

**MacOS/Windows：**

安装后Ollama会自动启动，你可以在菜单栏/系统托盘看到图标。

**Linux：**

```bash
# 前台启动（调试用）
ollama serve

# 后台启动
ollama serve &

# 或使用systemd服务
sudo systemctl start ollama
sudo systemctl enable ollama  # 开机自启
```

**验证服务状态：**

```bash
# 检查服务是否运行
curl http://localhost:11434

# 预期输出
Ollama is running
```

### 配置Ollama

**修改监听地址（允许远程访问）：**

```bash
# 设置环境变量
export OLLAMA_HOST=0.0.0.0:11434

# 或在启动时指定
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

**修改模型存储位置：**

```bash
# 默认存储在 ~/.ollama
# 可以修改到其他位置
export OLLAMA_MODELS=/data/ollama_models
```

**配置systemd服务：**

```bash
# 创建服务配置
sudo tee /etc/systemd/system/ollama.service <<EOF
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=$USER
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=/data/ollama_models"
ExecStart=/usr/local/bin/ollama serve
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 重载并启动
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
```

## 拉取与运行模型

### 查看可用模型

访问 https://ollama.com/library 查看所有支持的模型。

**推荐模型：**

| 模型 | 参数量 | 磁盘空间 | 内存需求 | 特点 |
|------|--------|---------|---------|------|
| qwen2.5:7b | 7B | 4.7GB | 8GB | 中文能力强，推荐 |
| qwen2.5:14b | 14B | 9GB | 16GB | 能力更强 |
| llama3.1:8b | 8B | 4.7GB | 8GB | 英文能力强 |
| deepseek-coder:6.7b | 6.7B | 3.8GB | 8GB | 代码专用 |
| mistral:7b | 7B | 4.1GB | 8GB | 平衡性好 |
| gemma2:9b | 9B | 5.5GB | 12GB | Google开源 |

### 拉取模型

```bash
# 拉取Qwen2.5 7B（推荐中文用户）
ollama pull qwen2.5:7b

# 拉取Llama3.1 8B
ollama pull llama3.1:8b

# 拉取DeepSeek Coder（代码专用）
ollama pull deepseek-coder:6.7b

# 拉取指定版本
ollama pull qwen2.5:7b@sha256:xxx
```

**下载过程：**

```
pulling manifest
pulling 6e4f0b5a... 100% |████████████████████| 4.7 GB / 4.7 GB
pulling 3b8d7d2a... 100% |████████████████████|  12 KB /  12 KB
pulling 4bb0c3ea... 100% |████████████████████|  25 KB /  25 KB
verifying sha256 digest
writing manifest
success
```

### 运行模型

**交互式对话：**

```bash
ollama run qwen2.5:7b

# 进入对话模式
>>> 你好，请介绍一下你自己
你好！我是Qwen，由阿里云开发的大语言模型。我可以帮助你回答问题、编写代码、翻译文本等。有什么我可以帮助你的吗？

>>> /bye  # 退出
```

**单次问答：**

```bash
ollama run qwen2.5:7b "用Python写一个快速排序"
```

**通过API调用：**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "你好",
  "stream": false
}'
```

### 管理本地模型

```bash
# 列出已安装的模型
ollama list

# 输出示例
NAME                ID              SIZE    MODIFIED
qwen2.5:7b          6e4f0b5a...     4.7 GB  2 hours ago
llama3.1:8b         4e8a6d2c...     4.7 GB  1 day ago

# 查看模型详情
ollama show qwen2.5:7b

# 删除模型
ollama rm llama3.1:8b

# 更新模型
ollama pull qwen2.5:7b
```

### 模型参数调整

```bash
# 设置温度参数
ollama run qwen2.5:7b --temperature 0.7

# 设置上下文长度
ollama run qwen2.5:7b --num-ctx 4096

# 设置GPU层数
ollama run qwen2.5:7b --num-gpu 35
```

**创建自定义模型：**

```bash
# 创建Modelfile
cat > Modelfile <<EOF
FROM qwen2.5:7b

# 设置参数
PARAMETER temperature 0.7
PARAMETER num_ctx 4096

# 设置系统提示
SYSTEM 你是一个专业的编程助手，擅长Python和JavaScript。
EOF

# 创建模型
ollama create my-coder -f Modelfile

# 运行自定义模型
ollama run my-coder
```

## 修改配置文件指向本地服务

配置OpenClaw使用本地Ollama模型。

### 基本配置

**编辑OpenClaw配置：**

```bash
openclaw config edit
```

**添加本地模型配置：**

```json
{
  "gateway": {
    "model": {
      "provider": "ollama",
      "model": "qwen2.5:7b",
      "baseUrl": "http://localhost:11434"
    }
  }
}
```

### 多模型配置

```json
{
  "models": {
    "default": {
      "provider": "ollama",
      "model": "qwen2.5:7b",
      "baseUrl": "http://localhost:11434"
    },
    "coding": {
      "provider": "ollama",
      "model": "deepseek-coder:6.7b",
      "baseUrl": "http://localhost:11434"
    },
    "cloud": {
      "provider": "alibaba",
      "model": "qwen-plus",
      "apiKey": "${ALIBABA_API_KEY}"
    }
  }
}
```

### 混合配置（本地+云端）

```json
{
  "gateway": {
    "model": {
      "provider": "ollama",
      "model": "qwen2.5:7b",
      "baseUrl": "http://localhost:11434",
      "fallback": [
        {
          "provider": "alibaba",
          "model": "qwen-plus",
          "apiKey": "${ALIBABA_API_KEY}"
        },
        {
          "provider": "deepseek",
          "model": "deepseek-chat",
          "apiKey": "${DEEPSEEK_API_KEY}"
        }
      ]
    }
  }
}
```

**执行逻辑：**

```
请求 → 本地Ollama → 成功 → 返回结果
           ↓ 失败/离线
       云端Qwen-Plus → 成功 → 返回结果
           ↓ 失败
       云端DeepSeek → 成功 → 返回结果
```

### 验证配置

```bash
# 测试本地模型
openclaw models test --model default

# 预期输出
Testing model connection...

Provider: ollama
Model: qwen2.5:7b
Base URL: http://localhost:11434

Sending test request...
✓ Connection successful
✓ Response time: 2.35s
✓ Model responded: "你好！我是Qwen..."

Local model is ready!
```

## 性能优化

### GPU加速

**检查GPU状态：**

```bash
# NVIDIA GPU
nvidia-smi

# Mac Metal
system_profiler SPDisplaysDataType
```

**配置GPU使用：**

```bash
# 设置GPU层数（越多越快，但需要更多显存）
ollama run qwen2.5:7b --num-gpu 35

# 完全使用CPU（显存不足时）
ollama run qwen2.5:7b --num-gpu 0
```

### 量化模型

量化可以显著减少内存占用：

```bash
# 4-bit量化（默认）
ollama pull qwen2.5:7b

# 更高量化（更小、更快、精度略降）
ollama pull qwen2.5:7b-q4_0

# 更低量化（更大、更准、更慢）
ollama pull qwen2.5:7b-q8_0
```

### 批处理优化

```json
{
  "gateway": {
    "model": {
      "provider": "ollama",
      "model": "qwen2.5:7b",
      "options": {
        "num_batch": 512,
        "num_ctx": 4096,
        "num_thread": 8
      }
    }
  }
}
```

## 常见问题

### 内存不足

```
Error: CUDA out of memory
```

**解决方案：**

1. 使用更小的模型（7B → 3B）
2. 降低量化级别（q8 → q4）
3. 减少GPU层数
4. 关闭其他占用内存的程序

### 响应速度慢

**解决方案：**

1. 确保使用GPU加速
2. 使用更小的模型
3. 减少上下文长度
4. 升级硬件（更多内存/更好GPU）

### 模型下载失败

**解决方案：**

```bash
# 使用代理
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
ollama pull qwen2.5:7b

# 或使用镜像站
# 修改 /etc/hosts 添加镜像解析
```

---

通过Ollama部署本地模型，你的OpenClaw现在可以完全离线运行了。无论是处理敏感数据，还是在无网络环境下使用，本地模型都提供了可靠的保障。结合云端模型的Fallback配置，你可以同时享受本地模型的隐私优势和云端模型的能力优势。
