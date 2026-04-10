# 安装与环境准备

## 白板导读

上一节我们回答了"为什么需要 vLLM"这个问题，现在轮到最实际的一步了：把 vLLM 跑起来。与 Ollama 的 `brew install ollama` 一行命令搞定不同，vLLM 的安装涉及更多的环境依赖——因为它直接操作 GPU 显存、自定义 CUDA Kernel、运行分布式通信，所以对底层环境的要求也更严格。但别担心，本节会带你从零开始，一步步完成从操作系统准备到验证安装成功的全部流程。我们会覆盖 Linux（主力生产环境）、Docker（容器化部署）、以及 macOS（开发测试）三种场景，并详细记录每个环节可能遇到的问题和解决方案。

---

## 2.1 系统要求总览

在开始安装之前，先确认你的硬件和软件是否满足 vLLM 的最低要求。vLLM 是一个重度依赖 NVIDIA GPU CUDA 生态的工具，这意味着它目前主要运行在 Linux + NVIDIA GPU 的组合上。

### 硬件要求

| 组件 | 最低要求 | 推荐配置 | 说明 |
|:---|:---|:---|:---|
| **GPU** | NVIDIA GTX 1060 6GB+ (Pascal) | A100 80GB / H100 80GB / RTX 4090 | 需要 Compute Capability ≥ 6.0 |
| **显存（VRAM）** | ≥ 8GB | ≥ 24GB（7B 模型） / ≥ 80GB（70B 模型） | 模型权重 + KV Cache + 运行时开销 |
| **系统内存（RAM）** | ≥ 16GB | ≥ 64GB | CPU offload 和预处理需要 |
| **磁盘空间** | ≥ 50GB SSD | ≥ 200GB NVMe SSD | 存放模型文件（7B FP16 ≈ 14GB） |
| **CPU** | 4 核以上 | 16 核+（多卡场景） | 数据预处理和调度线程 |

### 软件要求

| 组件 | 版本要求 | 说明 |
|:---|:---|:---|
| **操作系统** | Ubuntu 20.04+ / CentOS 7.9+ / WSL2 | Linux 原生最佳；macOS 实验性支持 |
| **Python** | 3.9 - 3.12 | 推荐 3.10 或 3.11 |
| **CUDA Toolkit** | 11.8+（推荐 12.1+） | 必须与驱动版本匹配 |
| **NVIDIA 驱动** | ≥ 525.60.13 | `nvidia-smi` 可查看当前版本 |
| **cuDNN** | ≥ 8.6 | 通常随 CUDA Toolkit 自动安装 |

### 各模型的最小显存需求

这是大家最关心的问题——我的显卡能跑多大的模型？以下数据基于 FP16 精度：

| 模型规模 | 参数量 | FP16 权重大小 | 最小显存（含 KV Cache） | 推荐显存 |
|:---|:---|:---|:---|:---|
| **1B-3B** | 1-3 Billion | 2-6 GB | 8 GB | 12 GB |
| **7B** | 7 Billion | ~14 GB | 20 GB | 24 GB |
| **13B-14B** | 13-14 Billion | ~28 GB | 36 GB | 48 GB |
| **32B-34B** | 32-34 Billion | ~64 GB | 72 GB | 80 GB (A100) |
| **70B-72B** | 70-72 Billion | ~140 GB | 4×80GB = 320 GB | 4×A100/H100 |
| **405B** | 405 Billion | ~810 GB | 8×H100 + NVLink | 超算集群 |

> **重要提示**：如果使用量化模型（AWQ/GPTQ INT4），上述显存需求可以降低约 **60-75%**。例如 7B INT4 模型只需约 5-6 GB 显存，一张 RTX 3060 12GB 就能跑。这就是为什么量化在生产环境中如此重要——我们将在第七章深入讨论。

---

## 2.2 Linux 环境：完整安装流程

Linux 是 vLLM 的主战场，绝大多数生产部署都运行在 Linux 上。下面以 Ubuntu 22.04 为例，演示完整的安装过程。

### 第一步：检查 GPU 环境

```bash
# 查看 NVIDIA 驱动版本
nvidia-smi

# 预期输出类似：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.154.05   Driver Version: 535.154.05   CUDA Version: 12.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  NVIDIA RTX 4090      Off | 00000000:01:00.0 Off |                  N/A |
# | 30%    42C    P8    18W / 450W |      4MiB / 24564MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# 查看 CUDA 版本（编译时用的工具链版本）
nvcc --version
# nvcc release 12.1, V12.1.105
```

如果 `nvidia-smi` 报错 "command not found"，说明没有安装 NVIDIA 驱动。你需要先安装驱动：

```bash
# Ubuntu 安装 NVIDIA 驱动（推荐使用官方 PPA）
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot  # 重启后生效
```

### 第二步：安装 Python 和基础依赖

```bash
# 确认 Python 版本（需要 3.9-3.12）
python3 --version
# Python 3.11.9

# 如果没有或版本过低：
sudo apt update
sudo apt install -y python3 python3-pip python3-venv build-essential

# 创建虚拟环境（强烈推荐！不要污染系统 Python）
python3 -m vllm-env venv
source venv/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel
```

> **为什么一定要用虚拟环境？** vLLM 依赖的 PyTorch 版本非常严格（必须与你机器上的 CUDA 版本匹配）。如果你系统上已经装了其他版本的 PyTorch（比如用于训练的），直接 pip install vllm 可能导致版本冲突。虚拟环境可以完美隔离这些依赖。

### 第三步：安装 vLLM

```bash
# 方法一：标准安装（自动检测 CUDA 版本并下载匹配的 PyTorch）
pip install vllm

# 这个命令会做很多事情：
# 1. 检测你的 CUDA 版本（通过 nvidia-smi 或 nvcc）
# 2. 下载对应版本的 PyTorch（如 torch==2.3.0+cu121）
# 3. 下载 vllm 及其所有依赖（transformers, sentencepiece, prometheus_client 等）
# 4. 编译 vLLM 自定义的 CUDA kernels

# 安装过程通常需要 3-10 分钟，取决于网络速度
```

安装过程中你可能会看到大量输出，最后出现 `Successfully installed vllm-x.x.x` 就表示成功了。

```bash
# 验证安装
python -c "
import vllm
import torch
print(f'vLLM version: {vllm.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} '
              f'({torch.cuda.get_device_properties(i).total_mem // 1024**3}GB)')
"
```

预期输出：

```
vLLM version: 0.6.1
PyTorch version: 2.3.0+cu121
CUDA available: True
CUDA version: 12.1
GPU count: 1
  GPU 0: NVIDIA RTX 4090 (24GB)
```

---

## 2.3 Docker 安装方式

对于生产环境和团队协作来说，Docker 是更推荐的部署方式。它消除了"在我机器上能跑"的经典问题，确保所有环境完全一致。

### 使用官方镜像

vLLM 提供了预构建好的 Docker 镜像，内置了 CUDA runtime、PyTorch 和 vLLM 本体：

```bash
# 拉取最新版官方镜像
docker pull vllm/vllm-openai:latest

# 或者指定版本
docker pull vllm/vllm-openai:v0.6.1

# 查看可用的镜像标签
# https://hub.docker.com/r/vllm/vllm-openai/tags
```

### 启动 vLLM 容器

```bash
# 基础启动（单卡）
docker run --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --trust-remote-code
```

参数解释：

| 参数 | 含义 |
|:---|:---|
| `--gpus all` | 将所有 GPU 传递给容器（也可指定 `device=0` 只传第一张卡） |
| `-p 8000:8000` | 端口映射：宿主机 8000 → 容器 8000 |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | **关键！** 挂载 HuggingFace 缓存目录，避免每次启动都重新下载模型文件 |
| `--model` | 要加载的模型（HuggingFace ID 或本地路径） |
| `--trust-remote-code` | 信任模型的远程代码（Qwen/Llama 等模型需要） |

### 多卡 Docker 启动

```bash
# 指定使用 GPU 0 和 GPU 1（张量并行 TP=2）
docker run --gpus '"device=0,1"' \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-70B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 16384 \
    --trust-remote-code
```

### docker-compose 方式（推荐）

对于包含多个服务的生产环境，用 docker-compose 编排更加清晰：

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-server
    ports:
      - "8000:8000"
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0', '1']
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --tensor-parallel-size 2
      --max-model-len 16384
      --gpu-memory-utilization 0.92
      --trust-remote-code
      --disable-log-requests
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

启动：

```bash
docker compose up -d
# 查看日志
docker compose logs -f vllm
```

> **⚠️ Docker 注意事项**：NVIDIA Container Toolkit 必须正确安装才能让容器访问 GPU。如果没有安装，运行 `--gpus all` 时容器内看不到任何 GPU。安装方法见下文。

### 安装 NVIDIA Container Toolkit

```bash
# 配置 NVIDIA Docker 仓库
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 重启 Docker 使配置生效
sudo systemctl restart docker

# 验证：容器内能看到 GPU
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

---

## 2.4 macOS 开发环境（实验性支持）

vLLM 主要面向 NVIDIA GPU，但在 Apple Silicon（M1/M2/M3/M4）芯片上也有实验性的 MPS（Metal Performance Shaders）后端支持。适合开发调试和小规模实验，但不推荐用于生产环境。

### 安装步骤

```bash
# macOS 安装（需要 Homebrew）
# 先确认芯片类型
sysctl -n machdep.cpu.brand_string
# Apple M2 Pro / M3 Max 等

# 创建虚拟环境
python3 -m venv vllm-mac-env
source vllm-mac-env/bin/activate

# 安装 PyTorch MPS 版本
pip install torch torchvision torchaudio

# 安装 vLLM
pip install vllm

# 验证
python -c "import vllm; print(vllm.__version__)"
```

### macOS 上的限制

| 特性 | 支持状态 | 说明 |
|:---|:---|:---|
| **基本推理** | ✅ 支持 | 单卡模式可用 |
| **MPS 加速** | ⚠️ 部分 | Metal 后端，性能约为同级别 GPU 的 30-50% |
| **张量并行（TP）** | ❌ 不支持 | Mac 只有统一内存，无多 GPU |
| **FP8/INT4 量化** | ⚠️ 有限 | AWQ/GPTQ 部分支持 |
| **CUDA Kernels** | ❌ 不适用 | 无 NVIDIA GPU |
| **Speculative Decoding** | ❌ 不支持 | |

### macOS 实际体验参考

在 Mac Studio M2 Max（96GB 统一内存）上运行 Qwen2.5-7B-Instruct：

```
模型加载时间: ~45 秒（首次，后续有缓存约 15 秒）
Token 生成速度: ~25-40 tok/s（取决于上下文长度）
适用场景: 开发调试、代码验证、小批量评估
不推荐: 高并发服务、延迟敏感的生产应用
```

---

## 2.5 常见安装问题排查

安装 vLLM 时，90% 的问题都逃不出以下几类。这里给出完整的排查清单。

### 问题一：CUDA 版本不匹配

**症状**：
```
ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
```
或者：
```
RuntimeError: CUDA is required but the system doesn't have a supported GPU.
```

**原因**：安装的 PyTorch CUDA 版本与你机器上的实际 CUDA Toolkit 版本不一致。

**解决方法**：

```bash
# 1. 先确认你的 CUDA 版本
nvcc --version
# 输出: release 12.1, V12.1.105 → 你的 CUDA 是 12.1

# 2. 卸载当前的 vllm 和 pytorch
pip uninstall vllm torch -y

# 3. 指定正确的 CUDA 版本重新安装 PyTorch
# CUDA 12.1 对应的 PyTorch
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 4. 再安装 vLLM
pip install vllm
```

### 问题二：GPU 显存不足

**症状**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.00 GiB...
```

**原因**：模型太大，超出显卡容量。

**解决方案（按优先级排序）**：

```bash
# 方案 1：使用更小的模型
# 把 Qwen2.5-7B 换成 Qwen2.5-1.5B 或 Qwen2.5-0.5B
--model Qwen/Qwen2.5-1.5b-Instruct

# 方案 2：启用量化（减少 60-75% 显存占用）
# 使用 AWQ 量化版模型
--model casperhao/Meta-Llama-3.1-8B-Instruct-AWQ \
--quantization awq

# 方案 3：降低 max-model-len（减少 KV Cache 预留）
--max-model-len 2048   # 默认值可能很大

# 方案 4：降低 gpu-memory-utilization
--gpu-memory-utilization 0.85   # 默认 0.90

# 方案 5：启用 CPU offload（将部分层放到 CPU 上）
--cpu-offload-gb 10   # 将约 10GB 的层卸载到 CPU 内存
```

### 问题三：模型下载失败或超时

**症状**：
```
OSError: We couldn't connect to 'https://huggingface.co' to load this file...
```

**原因**：国内网络访问 HuggingFace Hub 不稳定。

**解决方案**：

```bash
# 方案 1：设置 HuggingFace 镜像站
export HF_ENDPOINT=https://hf-mirror.com
# 然后重启 vLLM 服务

# 方案 2：提前手动下载模型到本地
# 先安装 huggingface_hub
pip install huggingface_hub

# 用 mirror 下载
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /data/models/Qwen2.5-7B-Instruct

# 然后启动时指定本地路径
--model /data/models/Qwen2.5-7B-Instruct

# 方案 3：使用 Docker 时挂载已下载好的模型
-v /data/models:/models
--model /models/Qwen2.5-7B-Instruct
```

### 问题四：Docker 中 GPU 不可见

**症状**：
```
docker run --gpus all ... 
# 容器内执行 nvidia-smi 报错：
# couldn't find libnvidia-ml.so.1
```

**解决方法**：

```bash
# 1. 确认主机上 nvidia-smi 正常
nvidia-smi

# 2. 确认 nvidia-container-runtime 已安装
which nvidia-container-runtime
# 如果没有，按上文 2.4 节安装 NVIDIA Container Toolkit

# 3. 确认 Docker 默认 runtime 包含 nvidia
docker info | grep -i runtime
# 应该看到 nvidia 出现在 Default Runtime 中

# 4. 手动指定 runtime 测试
docker run --runtime nvidia --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### 问题五：共享内存不足（Docker）

**症状**：
```
OSError: [Errno 12] Cannot allocate memory
```
或者：
```
NCCL error: unhandled system error in NCCL_P2P level, syscall = shmget
```

**原因**：Docker 默认的 `/dev/shm`（共享内存）只有 64MB，而某些模型初始化时需要更多共享内存来交换中间数据。

**解决方法**：

```bash
# 方法一：增大 shm-size（推荐）
docker run --gpus all --shm-size=8gb ... vllm/vllm-openai:latest ...

# 方法二：挂载 /dev/shm
docker run --gpus all -v /dev/shm:/dev/shm ... 

# 方法三：docker-compose 中配置
services:
  vllm:
    shm_size: '8gb'
```

### 问题六：权限错误

**症状**：
```
PermissionError: [Errno 13] Permission denied: '/root/.cache/huggingface'
```

**原因**：Docker 容器内的进程用户与挂载目录的所有者不一致。

**解决方法**：

```bash
# 确保挂载目录有写权限
chmod -R 777 ~/.cache/huggingface

# 或者在 docker-compose 中指定用户
user: "${UID}:${GID}"
```

---

## 2.6 安装验证清单

完成安装后，按照以下清单逐一验证，确保所有组件正常工作：

```bash
#!/bin/bash
# vllm_install_check.sh — vLLM 安装环境检查脚本

echo "========================================="
echo "       vLLM 环境健康检查"
echo "========================================="

# 1. Python 版本
echo -n "[1/7] Python 版本: "
python3 --version || { echo "❌ 未找到 Python"; exit 1; }

# 2. CUDA 可用性
echo -n "[2/7] CUDA: "
python3 -c "import torch; print('✅ CUDA', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ 不可用')" 2>/dev/null

# 3. vLLM 导入
echo -n "[3/7] vLLM: "
python3 -c "import vllm; print('✅ vLLM', vllm.__version__)" 2>/dev/null || echo "❌ 导入失败"

# 4. GPU 信息
echo "[4/7] GPU 详情:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  ❌ nvidia-smi 不可用"

# 5. 磁盘空间
echo -n "[5/7] 磁盘空间: "
df -h . | awk 'NR==2{printf "%.0fGB 可用 / %.0fGB 总计\n", $4, $2}'

# 6. 共享内存（Docker 环境）
echo -n "[6/7] 共享内存: "
if [ -f /proc/meminfo ]; then
    shmall=$(grep Shmall /proc/meminfo | awk '{print $2}')
    shmmax=$(grep Shmmax /proc/meminfo | awk '{printf "%.0fMB", $2/1024/1024}')
    echo "✅ $shmmax"
else
    echo "⚠️ 非 Linux 环境"
fi

# 7. 网络连通性
echo -n "[7/7] HuggingFace 连通: "
curl -sf --connect-timeout 5 https://huggingface.co > /dev/null 2>&1 && echo "✅ 正常" || echo "⚠️ 无法直连（建议设 HF_ENDPOINT 镜像）"

echo "========================================="
echo "  全部检查完成，可以开始使用 vLLM 了！"
echo "========================================="
```

保存为脚本后运行：

```bash
chmod +x vllm_install_check.sh
./vllm_install_check.sh
```

预期输出：

```
=========================================
       vLLM 环境健康检查
=========================================
[1/7] Python 版本: Python 3.11.9
[2/7] CUDA: ✅ CUDA 12.1 | GPU: NVIDIA RTX 4090
[3/7] vLLM: ✅ vLLM 0.6.1
[4/7] GPU详情:
  NVIDIA RTX 4090, 24576 MiB, 535.154.05
[5/7] 磁盘空间: 180GB 可用 / 500GB 总计
[6/7] 共享内存: ✅ 8192MB
[7/7] HuggingFace 连通: ✅ 正常
=========================================
  全部检查完成，可以开始使用 vLLM 了！
=========================================
```

当所有 7 项都显示 ✅ 或 ⚠️（不影响使用的警告），你的 vLLM 环境就已经准备好了。接下来我们将在下一节中启动第一个模型服务并进行实际的 API 调用。

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **目标平台** | Linux + NVIDIA GPU 是主力；Docker 是生产首选；macOS 仅适合开发实验 |
| **最小硬件** | GTX 1060 6GB+ 可跑 1-3B 小模型；RTX 4090 24GB 可跑 7B 模型；70B 模型需 4×A100 |
| **安装命令** | `pip install vllm`（自动匹配 CUDA 版本的 PyTorch） |
| **虚拟环境** | **必须用 venv** 隔离依赖，避免与其他项目的 PyTorch 冲突 |
| **Docker 关键点** | NVIDIA Container Toolkit + `--gpus all` + 模型缓存 Volume 挂载 + `--shm-size=8gb` |
| **Top 3 错误** | ①CUDA 版本不匹配 → 重装匹配的 PyTorch ②OOM → 量化/降参/CPU offload ③HF 下载失败 → 设镜像站 |
| **验证方法** | `import vllm` + `torch.cuda.is_available()` + `nvidia-smi` 三连确认 |

> **一句话总结**：vLLM 的安装比 Ollama 复杂得多，因为它直接操作 CUDA Kernel 和 GPU 显存管理。但一旦搭建好 Linux + CUDA + PyTorch + vLLM 这条工具链，你就拥有了一个能榨干 GPU 性能的企业级推理引擎。记住三个黄金法则：**永远用虚拟环境、永远挂载模型缓存目录、永远先跑安装检查脚本再开始调试问题**。
