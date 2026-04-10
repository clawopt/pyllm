# 02-01 启动 API 服务

## 从终端聊天到 API 服务：思维的转变

上一章我们在终端里用 `ollama run` 和模型聊了天，这很好——但如果你想把 Ollama 的能力集成到你自己的应用中（一个 Python 脚本、一个 Web 服务、一个 VS Code 插件），终端交互模式就不够用了。你需要的是**程序化的接口**，这就是 API。

Ollama 在你安装并运行后，就已经自动在后台启动了一个 HTTP API 服务。你可能完全没有意识到这一点——因为上一节我们一直在用 CLI 和它对话，但这个 CLI 本质上就是 API 的一个薄包装。当你执行 `ollama run qwen2.5:1.5b "你好"` 时，Ollama 实际上做了这样一件事：

```
你的终端
   │
   ▼ ollama (CLI)
   │  把你的输入封装成 HTTP POST 请求
   │  发送到 http://localhost:11434/api/chat
   │
   ▼ Ollama Server（后台进程）
   │  接收请求 → 加载模型 → 推理 → 返回响应
   │
   ◀ ollama (CLI)
      解析响应 → 打印到终端
```

所以理解 API 服务的关键洞察是：**CLI 和 API 是同一个东西的两面。** 你在终端里能做的所有事情，通过 API 也都能做；而且通过 API 还能做很多 CLI 做不到的事情（比如流式输出到 Web 页面、并发处理多个请求、集成到更大的系统里）。

## 验证服务状态

在开始写代码之前，先确认服务确实在跑着：

```bash
# 方法一：检查 API 版本信息
curl http://localhost:11434/api/version
# {"version":"0.5.7"}

# 方法二：查看已加载的模型列表
curl http://localhost:11434/api/tags
# {"models":[]}  ← 刚安装完可能还没有任何模型

# 方法三：健康检查（综合验证）
curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags
# 应该返回 200
```

如果以上任何一个命令报 `Connection refused` 或超时，说明 Ollama 服务没有正常运行。排查步骤：

```bash
# 1. 确认 ollama 进程存在
ps aux | grep ollama

# 2. 如果没进程，手动启动
ollama serve &

# 3. 检查端口是否被占用
lsof -i :11434

# 4. macOS 用户注意：Ollama app 应该在菜单栏有图标
# 图标存在 = 后台服务正在运行
```

## 核心配置项：让服务按你的方式工作

Ollama 服务的行为可以通过环境变量精细控制。这些变量可以在启动服务前设置，也可以写在 shell 配置文件中持久化。

### 监听地址：谁可以访问？

```bash
# 默认行为：只监听 localhost（127.0.0.1）
# 这意味着只有本机能访问，其他电脑连不上

# 开放局域网访问（开发/测试环境常用）
OLLAMA_HOST=0.0.0.0:11434 ollama serve
# 现在同一 WiFi 下的其他设备也能调用你的 Ollama 了

# ⚠️ 生产环境不要用 0.0.0.0！应该在反向代理层做认证
```

### 模型目录：模型文件放哪里？

```bash
# 默认：~/.ollama/models/
# 如果你想把模型放到更大的磁盘分区：
OLLAMA_MODELS=/data/ollama-models ollama serve
# 所有后续的 pull/create 操作都会使用新路径
```

### 并发与性能参数

```bash
# 并行处理请求数量（默认自动）
OLLAMA_NUM_PARALLEL=4 ollama serve
# 同时处理多少个并发请求。设太大会导致内存不足，
# 设太小会排队等待。推荐值：CPU核数的一半到全部

# 同时加载的最大模型数
OLLAMA_MAX_LOADED_MODELS=3 ollama serve
# 内存够用时可以同时保持多个模型热加载
# 超过限制时最久未使用的会被卸载

# 模型空闲后多久卸载（默认 5m）
OLLAMA_KEEP_ALIVE=30m ollama serve
# 设短节省内存，设长避免重复加载开销

# 单次请求超时时间（秒）
OLLAMA_REQUEST_TIMEOUT=120 ollama serve
# 处理大模型+长文本时可能需要更长
```

### 调试选项

```bash
# 开启详细日志（排错必备）
OLLAMA_DEBUG=1 ollama serve
# 会打印每个请求的详细信息：输入token数、推理耗时、错误详情

# verbose 模式
OLLAMA_VERBOSE=1 ollama serve
# 比 DEBUG 少一些细节，但能看到关键中间状态
```

### 持久化配置的最佳实践

每次手动 export 环境变量很麻烦，推荐写入配置文件：

```bash
# ~/.ollama/env  （Ollama 会自动读取这个文件）
# 或者写入你的 shell profile:

# ~/.zshrc 或 ~/.bashrc 中添加:
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_DEBUG=0

# 写入后重新加载 shell 或 source 一次即可生效
source ~/.zshrc
```

## Docker 场景的服务配置

如果你的 Ollama 跑在 Docker 里（这在服务器部署中很常见），配置方式略有不同：

```yaml
# docker-compose.yml — 完整的生产级服务配置
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    environment:
      # 通过环境变量传入配置
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_NUM_PARALLEL=8
      - OLLAMA_KEEP_ALIVE=15m
      - OLLAMA_MAX_LOADED_MODELS=5
      - OLLAMA_DEBUG=${OLLAMA_DEBUG:-0}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
        limits:
          memory: 32G
    networks:
      - ollama-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  ollama-net:
    driver: bridge
```

几个 Docker 特有的注意事项：

**1. 健康检查很重要**

上面配置中的 `healthcheck` 让 Docker 能自动检测 Ollama 是否真的在正常工作（而不只是容器"活着"）。当健康检查失败时，Docker 可以自动重启容器。这对于无人值守的服务器部署至关重要。

**2. GPU 直通的常见坑**

```bash
# ❌ 错误：只写了 GPU 声明但宿主机没装 nvidia-docker-runtime2
# 结果：容器启动成功但内部用的是 CPU 模式（不会报错！）

# ✅ 正确的验证流程：
docker exec -it ollama-server nvidia-smi
# 应该看到 GPU 信息
# 如果显示 "command not found" → GPU 配置未生效

docker logs ollama-server 2>&1 | grep -i "gpu\|cuda\|metal"
# 看到 "using cuda" = GPU 加载成功
# 什么都没看到或只有 "using cpu" = 在用纯 CPU
```

**3. 数据卷必须持久化**

```yaml
volumes:
  - ollama-data:/root/.ollama  # ✅ 必须有这一行
# 没有 volume → 容器重建 = 所有下载的模型全没了
# 这是最常见的 Docker 新手错误之一
```

## systemd 服务配置（Linux 原生安装）

如果你不用 Docker 而是 Linux 上直接安装的 Ollama，systemd 服务管理是最可靠的方式：

```ini
# /etc/systemd/system/ollama.service
[Unit]
Description=Ollama Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ollama
Group=ollama
ExecStart=/usr/local/bin/ollama serve
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_KEEP_ALIVE=10m"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Restart=always
RestartSec=5

# 安全加固
NoNewPrivileges=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/usr/share/ollama:/root/.ollama

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# 查看状态
sudo systemctl status ollama
# 应该显示 active (running)

# 查看日志
journalctl -u ollama -f
# 实时跟踪日志输出

# 重启服务（修改配置后）
sudo systemctl restart ollama
```

systemd 方案相比直接 `ollama serve &` 后台运行的优点：
- **开机自启**：服务器重启后自动恢复
- **崩溃自愈**：进程异常退出后自动重启（`Restart=always`）
- **日志集中**：统一通过 journalctl 管理
- **资源隔离**：以专用用户运行，权限最小化

## 服务端点速查表

Ollama API 服务除了后面章节会深入讲解的 `/api/chat`、`/api/generate`、`/api/embeddings` 这些核心端点外，还有一些辅助端点值得了解：

| 端点 | 方法 | 用途 | 典型用途 |
|------|------|------|---------|
| `/api/tags` | GET | 列出所有可用模型 | 前端展示模型列表 / 健康检查 |
| `/api/version` | GET | 获取版本信息 | 兼容性检查 / 问题上报 |
| `/api/ps` | GET | 查看当前运行状态 | 运维监控 / 性能调试 |
| `/api/chat` | POST | 对话式生成 | 主要接口，流式+非流式 |
| `/api/generate` | POST | 补全式生成 | 底层接口，支持多模态 |
| `/api/embeddings` | POST | 文本向量化 | RAG / 语义搜索 |
| `/api/copy` | POST | 创建模型副本 | Modelfile 操作 |
| `/api/pull` | POST | 拉取模型 | 远程触发模型下载 |
| `/api/delete` | DELETE | 删除模型 | 远程清理 |
| `/api/blobs/:sha256` | HEAD/Blobs | Blob 操作 | 高级用法 |

其中 `/api/ps` 特别有用——它能告诉你当前服务器上有哪些模型被加载到了内存中、各自占用了多少显存、正在处理多少请求：

```bash
$ curl -s http://localhost:11434/api/ps | python3 -m json.tool
{
    "models": [
        {
            "name": "qwen2.5:7b",
            "model": "qwen2.5:7b",
            "size": 4663911328,
            "vram": 4696168448,
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen2",
                "parameter_size": "7.0B",
                "quantization_level": "Q4_K_M"
            },
            "expires_at": "2026-04-08T12:35:00Z",
            "working": false
        }
    ]
}
```

字段解读：
- **name/model**: 模型标识
- **size**: 模型文件大小（字节）
- **vram**: 当前占用的显存/内存（字节）。注意 `working: false` 时 vram 可能为 0（已卸载到磁盘但 manifest 仍注册）
- **details**: 模型的架构细节
- **expires_at**: 自动卸载的时间点（由 `OLLAMA_KEEP_ALIVE` 控制）
- **working**: 是否正在处理请求

这个端点是构建监控 Dashboard 的核心数据源——我们在第十章会回到这里。

## 常见问题

### Q: 改了环境变量为什么不生效？
**A**: 如果你是在已经运行的 `ollama serve` 进程之前 export 的变量，那对已有进程无效。需要先停止旧进程（`Ctrl+C` 或 `ollama stop`），重新设置变量后再启动。或者如果用的是 systemd 管理，`sudo systemctl restart ollama` 即可。

### Q: 多个 Ollama 实例会不会冲突？
**A**: 取决于是否使用相同端口。两个实例监听不同端口（如 11434 和 11435）可以共存，各有各的模型空间。但如果都抢 11434，第二个会启动失败并报 "address already in use"。生产环境中通常只需要一个实例 + 反向代理。

### Q: 服务启动后一段时间变慢了怎么办？
**A**: 最常见原因是内存碎片化或模型累积过多。试试 `ollama list` 看有多少模型被加载，然后 `ollama rm` 不用的。也可以缩短 `OLLAMA_KEEP_ALIVE` 让闲置模型更快释放。如果是长时间运行的问题，定期 `systemctl restart ollama`（Linux）或重启 Docker 容器也能缓解。
