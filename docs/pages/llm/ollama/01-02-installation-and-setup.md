# 01-02 安装与环境准备

## 你的电脑能跑大模型吗？

在开始安装之前，我们需要先回答一个很现实的问题：**你的硬件到底能不能跑得动大模型？** 这不是要劝退你，而是因为"能跑"和"跑得好"是两回事，提前了解硬件需求可以避免后面遇到"下载了模型结果内存爆掉"的尴尬。

### 内存：最关键的资源

大模型运行时需要把模型权重加载到内存中，这是硬性需求——不像 CPU 可以用虚拟内存勉强凑合，内存不够就是真的不行。不同大小的模型需要的内存大致如下：

```
┌───────────────────────────────────────────────────────────────┐
│                  模型大小 vs 内存需求（量化后 q4_K_M）           │
├──────────┬────────────┬────────────┬────────────┬────────────┤
│ 模型参数量 │ 最小推荐    │ 流畅运行    │ 舒适体验    │ 备注        │
├──────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
│ 1.5B     │ 2 GB       │ 4 GB       │ 8 GB       │ 极轻量任务  │
│ 3B       │ 3 GB       │ 6 GB       │ 8 GB       │ Phi3/Gemma │
│ 7B       │ 5 GB       │ 8 GB       │ 16 GB      │ 主流选择    │
│ 8B       │ 6 GB       │ 10 GB      │ 16 GB      │ Llama3/Mistral│
│ 13B/14B  │ 9 GB       │ 14 GB      │ 24 GB      │ Qwen2.5    │
│ 32B      │ 20 GB      │ 28 GB      │ 48 GB      │ DeepSeek   │
│ 70B      │ 40 GB      │ 48 GB      │ 80+ GB     │ Llama3-70B │
└──────────┴────────────┴────────────┴────────────┴────────────┘

注意: 以上是 q4_K_M 量化后的估算。fp16 精度下所有数值 × 2~3。
KV Cache 在长对话时会额外占用约 10-30% 的模型大小。
```

这里有几个关键点需要理解：

**第一，表中的数值是"最小可用"，不是"舒适使用"。** 比如 7B 模型标注最小 5GB，意思是你的机器至少要有这么多空闲内存才能把模型加载进去。但实际对话时还需要额外的空间给 KV Cache（保存对话历史）、推理时的中间计算、以及操作系统本身。所以如果你只有 8GB 内存的 Mac，跑 7B 模型虽然能启动，但稍微聊几轮就可能因为内存不足而卡顿甚至崩溃。

**第二，Apple Silicon（M1/M2/M3/M4）的用户有特殊优势。** 这些芯片采用统一内存架构（Unified Memory），CPU 和 GPU 共享同一块内存条。这意味着你 Mac 上标称的 16GB 或 24GB 内存全部可以被 Ollama 使用——不需要像 NVIDIA 显卡那样区分"显存"和"系统内存"。一台 M2 Mac mini（24GB 版本）就能比较舒服地跑 13B 甚至 32B 的模型。

**第三，Linux + NVIDIA GPU 是性能最强但配置最复杂的方案。** 如果你有张 4090（24GB VRAM），那你可以轻松跑 70B 模型的 fp16 版本，而且速度飞快。但前提是你需要正确安装 CUDA 驱动、配置好环境变量、处理各种版本兼容问题——这些我们在安装部分会详细说明。

### CPU：被低估的瓶颈

很多人只关注内存和 GPU，但实际上 **CPU 在本地跑模型时也是重要角色**。Ollama 底层的 llama.cpp 使用 CPU 进行大量的矩阵运算（尤其是在没有 GPU 或者 GPU 只负责部分层的时候），而且模型加载、tokenize 输入文本、采样输出等步骤都依赖 CPU。

一个实用的经验法则是：
- **纯 CPU 模式**（无 GPU）：现代多核 CPU 可以流畅跑 3B 以下的模型，7B 模型也能跑但每秒只能生成 2-5 个 token（大约是打字阅读的速度）
- **GPU 加速模式**：GPU 负责 Transformer 中计算量最大的矩阵乘法部分，CPU 负责其余工作。典型的加速比是 CPU-only 的 5-15 倍
- **Apple Metal 加速**：M1/M2/M3/M4 的 GPU 单元通过 Metal Performance Shaders (MPS) 加速，虽然没有 NVIDIA CUDA 快，但对大多数场景已经足够了

### 存储空间：别忽略这个

模型文件其实不小。一个 q4_K_M 量化的 7B 模型大约占 **4.2GB** 磁盘空间，13B 约 **8.5GB**，70B 约 **40GB**。如果你打算尝试多个模型，或者还要存放自定义转换后的 GGUF 文件，建议预留 **50-100GB** 的磁盘空间给 `~/.ollama/models/` 目录。

## macOS 安装：最丝滑的体验

macOS 是 Ollama 支持最好的平台，安装过程可以用"简单到令人发指"来形容。

```bash
# 方式一：Homebrew（推荐）
brew install ollama

# 安装完成后验证
ollama --version
# 应该显示类似: ollama version is 0.5.7
```

就这么一行命令。Homebrew 会自动完成以下事情：
- 下载 Ollama 二进制文件
- 创建 `/Applications/Ollama.app` 应用
- 配置 `ollama` 命令到 PATH
- 设置开机自启（通过 launchd）

安装完成后，你会在菜单栏看到 Ollama 的图标，说明后台服务已经自动启动了。

```bash
# 验证服务是否正常运行
curl http://localhost:11434/api/tags
# 返回类似 {"models": []} 说明 API 服务正常
# （此时还没有拉取任何模型，所以 models 列表为空）
```

**关于 Apple Silicon 的特别说明：**

如果你用的是 Intel Mac（非 Apple Silicon），Ollama 仍然可以运行，但只能使用 CPU 模式——这意味着速度会比较慢，而且只能跑较小的模型（建议 3B 以下）。如果你的 Intel Mac 只有 8GB 内存，体验可能不太理想。这种情况下建议考虑用远程服务器上的 Ollama（通过 SSH 端口转发），而不是在本地硬跑。

对于 M 系列 Mac 用户（M1/M2/M3/M4），Ollama 会自动检测并启用 Metal 加速。你可以在日志中看到类似 `metal` 字样来确认：

```bash
# 启动任意模型后观察日志
ollama run qwen2.5:1.5b "hi"
# 终端输出中可能会看到:
# > [gemma-2.0] using metal backend
```

## Linux 安装：灵活但有细节需要注意

Linux 是 Ollama 的"原生栖息地"（Ollama 官方主要在 Linux 上开发和测试），功能最完整，尤其是 GPU 相关的能力。但安装比 macOS 多几个步骤。

```bash
# 官方一键安装脚本（推荐大多数用户直接用这个）
curl -fsSL https://ollama.com/install.sh | sh

# 这个脚本会做:
# 1. 创建 ollama 用户（安全隔离）
# 2. 创建 /usr/local/bin 的符号链接
# 3. 设置 systemd 服务（开机自启）
# 4. 启动 ollama serve 后台进程
```

**手动安装方式**（如果你不想用 root 权限执行脚本）：

```bash
# 1. 下载二进制文件
curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama
chmod +x /usr/local/bin/ollama

# 2. 创建服务用户（可选但推荐）
sudo useradd -r -d /usr/share/ollama ollama
sudo mkdir -p /usr/share/ollama
sudo chown -R ollama:ollama /usr/share/ollama

# 3. 启动服务
sudo systemctl start ollama
# 或者前台运行测试:
ollama serve &
```

### NVIDIA GPU 驱动的配置

这是 Linux 安装中最容易出问题的地方，值得单独展开。

**前置条件检查：**

```bash
# 检查 CUDA 是否已安装
nvidia-smi
# 应该能看到你的 GPU 信息和驱动版本
# 需要 CUDA 11.8 或更高版本（Ollama 1.5+ 要求）

# 检查 CUDA 编译工具链
nvcc --version
# 如果没装或版本太旧，需要先安装 CUDA Toolkit
```

**常见 GPU 问题排查：**

| 问题现象 | 可能原因 | 解决方法 |
|---------|---------|---------|
| `ollama run` 报错 "no cuda devices found" | 驱动未安装或版本不匹配 | 安装对应版本的 NVIDIA 驱动 |
| 能启动但极慢（和 CPU 一样慢） | 未加载 CUDA 库 | 检查 `LD_LIBRARY_PATH` 是否包含 CUDA lib |
| 运行一段时间后显存溢出 | GPU 显存不够放模型 | 用更小的模型或开启 offload-gpu-layers |
| 多 GPU 时只用了一张 | `CUDA_VISIBLE_DEVICES` 未设置 | 导出环境变量指定 GPU |

**NVIDIA 驱动安装速查（Ubuntu 22.04）：**

```bash
# 添加 NVIDIA 官方仓库
curl -fsSL https://nvidia.github.io/libnvidia-repo/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia.gpg
curl -s -L https://nvidia.github.io/libnvidia-repo/stable/deb/nvidia.local_55040.pub-key | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-local-55040.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia.gpg] https://libnvidia-ml.repo.nvidia.com/jammy/dev/ubuntu22.04 /$(. /etc/os-release; echo UBUNTU_CODENAME) /main" | sudo tee /etc/apt/sources.list.d/nvidia-ml.list > /dev/null
sudo apt-get update

# 安装驱动和 toolkit
sudo apt-get install -y nvidia-driver-550 nvidia-cuda-toolkit-12-6 nvidia-cudnn

# 重启计算机使驱动生效
sudo reboot
```

### SELinux 问题（RHEL/CentOS/Fedora）

如果你在企业级 Linux 发行版上使用，SELinux 可能会阻止 Ollama 访问 GPU 设备：

```bash
# 临时关闭（用于快速测试）
sudo setenforce 0

# 或者更精细的方式：允许 Ollama 访问 GPU
sudo semanage port -a -t ollama_t -p http_cache_port_t -P 11434
```

## Windows 安装：WSL2 是正道

Windows 原生支持还在实验阶段，目前最稳定的方案是通过 WSL2（Windows Subsystem for Linux 2）运行 Ollama。

```powershell
# 在 PowerShell（管理员）中启用 WSL2
wsl --install -d Ubuntu-22.04
# 安装完成后重启电脑

# 进入 WSL 环境
wsl

# 在 WSL 内部按照上面的 Linux 方式安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

安装完成后，你在 Windows 的浏览器中访问 `http://localhost:11434` 就可以使用 Ollama Web UI 了——因为 WSL2 会自动端口转发。

**Windows 原生安装包（实验性）：**

Ollama 也提供了 `.msi` 和 `.exe` 安装包，可以从 GitHub Releases 页面下载。但目前原生支持有限（主要是缺少 GPU 加速），适合只想用 CPU 跑小模型的场景。

## Docker 安装：服务器部署的首选

如果你要把 Ollama 部署到服务器上（比如公司内部 AI 平台），Docker 是最干净的方式。

```dockerfile
# docker-compose.yml — 生产级 Ollama 服务模板
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"    # API 端口
    volumes:
      # 关键：模型数据必须持久化！否则容器重建就丢失了
      - ollama_data:/root/.ollama
    environment:
      # 允许局域网内其他机器访问（不要在生产环境开 0.0.0.0）
      - OLLAMA_HOST=0.0.0.0
      # GPU 配置（仅当宿主机有 NVIDIA GPU 且安装了 nvidia-docker-runtime2 时）
      - NVIDIA_VISIBLE_DEVICES=all
    # GPU 直通配置
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    # 资源限制（可选，防止 Ollama 吃掉所有资源）
    deploy:
      resources:
        limits:
          memory: 32G  # 根据你的服务器调整
    networks:
      - ollama-net

networks:
  ollama-net:
    driver: bridge
```

```bash
# 启动
docker compose up -d

# 验证
curl http://localhost:11434/api/tags
```

**Docker GPU 支持的关键点：**

要让 Ollama 在 Docker 中使用 NVIDIA GPU，你需要确保：
1. 宿主机已安装 NVIDIA Driver（`nvidia-smi` 可用）
2. 已安装 **NVIDIA Container Toolkit**（不是 CUDA Toolkit！这是两个不同的东西）
3. 已安装 **nvidia-docker-runtime2**
4. `docker-compose.yml` 中包含 GPU 部署声明

如果 GPU 配置有问题，容器内的 Ollama 会回退到 CPU 模式——不会报错，只是会很慢。你可以通过查看日志确认是否成功加载了 GPU：

```bash
docker logs ollama 2>&1 | grep -i "gpu\|cuda\|metal"
# 看到 "using cuda" 或 "using metal" 表示 GPU 加载成功
# 如果没有任何 gpu/cuda/metal 相关输出，说明在用 CPU 模式
```

## 安装后的必做检查清单

不管你用哪种方式安装，完成以下检查可以避免后续 90% 的坑：

```python
#!/usr/bin/env python3
"""Ollama 安装后自检脚本"""
import subprocess
import sys
import shutil
import platform


def check_command(cmd, description):
    """检查命令是否存在"""
    exists = shutil.which(cmd) is not None
    status = "✅" if exists else "❌"
    version = ""
    if exists:
        try:
            result = subprocess.run(
                [cmd, "--version"] if cmd != "nvidia-smi" else ["nvidia-smi"],
                capture_output=True, text=True, timeout=5,
            )
            version = result.stdout.strip().split("\n")[0][:60]
        except Exception:
            version = "(无法获取版本)"
    print(f"{status} {description}: {cmd} {version}")
    return exists


def check_url(url, description):
    """检查 HTTP 服务是否可达"""
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
            capture_output=True, text=True, timeout=5,
        )
        code = result.stdout.strip()
        status = "✅" if code == "200" else "⚠️ "
        print(f"{status} {description}: {url} (HTTP {code})")
        return code == "200"
    except Exception as e:
        print(f"❌ {description}: {url} ({e})")
        return False


def check_memory(min_gb=8):
    """检查可用内存"""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        total_bytes = int(result.stdout.split(":")[1].strip())
        total_gb = total_bytes // (1024 ** 3)
        
        # macOS: vm.stat
        # Linux: /proc/meminfo MemAvailable
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5,
            )
            free_bytes = int(result.stdout.split()[10])
            free_gb = free_bytes // (1024 ** 3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        free_bytes = int(line.split()[1]) * 1024
                        break
            free_gb = free_bytes // (1024 ** 3)
        
        status = "✅" if free_gb >= min_gb else "⚠️ "
        print(f"{status} 可用内存: {free_gb} GB / 总计 {total_gb} GB (建议 ≥{min_gb}GB)")
        return free_gb >= min_gb
    except Exception:
        print("❌ 无法检测内存信息")
        return False


print("=" * 60)
print("Ollama 安装自检工具")
print("=" * 60)
print()

print("--- 基础环境 ---")
check_command("ollama", "Ollama CLI")
check_command("curl", "curl（API 调用必需）")

print()
print("--- 服务状态 ---")
check_url("http://localhost:11434/api/tags", "Ollama API 服务")

print()
print("--- 硬件资源 ---")
check_memory(8)

if platform.system() == "Linux":
    check_command("nvidia-smi", "NVIDIA GPU 驱动")
elif platform.machine().startswith("arm64") or platform.machine().startswith("arm64e"):
    print(f"✅ Apple Silicon: {platform.processor()}")

print()
print("--- 下一步 ---")
print("运行: ollama run qwen2.5:1.5b 开始第一个对话")
print("或访问: http://localhost:11434 打开 WebUI")
```

把这个脚本保存为 `check_installation.py` 并运行 `python3 check_installation.py`，它会自动检查 Ollama 是否正确安装、服务是否正常响应、硬件是否满足基本要求。如果所有项都显示 ✅，恭喜你，环境已经就绪！

## 常见安装问题与解决方案

### 问题 1：`ollama: command not found`

**原因**：安装后 shell 的 PATH 还没刷新，或者安装脚本没有成功创建符号链接。

**解决**：
```bash
# 方法一：重启终端（最简单）
# 完全退出当前终端窗口，重新打开

# 方法二：手动添加 PATH
export PATH="/usr/local/bin:$PATH"
ollama --version  # 验证

# 方法三：确认 Homebrew 安装位置
ls /opt/homebrew/bin/ollama 2>/dev/null && echo "Found in homebrew" || echo "Not in homebrew"
which ollama || echo "ollama not in PATH anywhere"
```

### 问题 2：端口 11434 被占用

**原因**：之前的 Ollama 进程没有正确退出，或者有其他程序恰好用了 11434 端口。

**解决**：
```bash
# 查看谁占了端口
lsof -i :11434

# 如果是残留的 ollama 进程
kill $(lsof -t -i :11434)

# 如果是其他程序（比如另一个 ollama 实例）
# 要么停掉它，要么修改 OLLAMA_HOST 使用其他端口:
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

### 问题 3：Linux 上 `ollama run` 卡住不动

**原因**：首次运行某个模型时需要先下载模型文件（从 HuggingFace Hub），国内网络可能很慢或完全连不上。

**解决**：
```bash
# 方法一：设置代理
export HTTPS_PROXY=http://your-proxy:port
ollama run qwen2.5:7b

# 方法二：手动预下载（先下载再运行，可以看到进度条）
ollama pull qwen2.5:7b
# 下载完后再 run 就秒开了
ollama run qwen2.5:7b

# 方法三：使用镜像站（如果有）
# OLLAMA_ORIGINS 设置为国内镜像源
```

### 问题 4：Docker 中 Ollama 不使用 GPU

**原因**：最常见的三个问题是：(1) 没装 nvidia-docker-runtime2 (2) docker-compose 缺少 GPU 声明 (3) 宿主机本身没装 NVIDIA 驱动。

**诊断步骤**：
```bash
# 1. 确认宿主机 GPU 可见
nvidia-smi
# 如果这里报错，先解决驱动问题

# 2. 确认 nvidia-docker-runtime2 已安装
nvidia-container-cli list
# 应该列出 runtime 列表

# 3. 确认容器内识别到了 GPU
docker exec -it ollama nvidia-smi
# 如果报 command not found → GPU 配置没生效
# 如果显示 GPU 信息 → 正常

# 4. 检查 Ollama 日志中的加载信息
docker logs ollama 2>&1 | grep -i "gpu\|cuda\|metal"
```

### 问题 5：Mac 上运行 70B 模型极其缓慢或崩溃

**原因**：70B 模型即使用 q4_K_M 量化也需要 ~40GB 内存。如果你的 Mac 只有 16GB 或 24GB，系统会大量使用 swap（交换内存），导致速度下降几十倍。

**现实建议**：Mac 上合理的选择是 ≤13B 的模型。70B 及以上请考虑使用 Linux + NVIDIA GPU 的服务器。

---

下一节，我们将在确认环境无误的基础上，真正地运行第一个模型，并学习如何与它进行交互式对话。
