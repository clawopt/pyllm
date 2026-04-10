# Docker 容器化部署

> **白板时间**：你在开发机上用 `python -m vllm.entrypoints.openai.api_server` 跑通了 vLLM 服务，现在要把它搬到生产环境。生产环境不会有人工登录服务器敲命令——你需要把一切都打包进容器，用声明式配置管理，让系统可以一键启动、自动恢复、水平扩展。这就是 Docker 容器化的价值。

## 一、vLLM 官方镜像

### 1.1 镜像选择

| 镜像名 | 用途 | 大小 | 说明 |
|--------|------|------|------|
| `vllm/vllm-openai:latest` | API Server | ~15 GB | **推荐** - 包含完整依赖 |
| `vllm/vllm-serve:latest` | 轻量服务 | ~8 GB | 精简版 |
| `vllm/vllm-dev:latest` | 开发调试 | ~20 GB | 包含调试工具 |

### 1.2 最简 Docker 启动

```bash
# 拉取镜像
docker pull vllm/vllm-openai:latest

# 运行 vLLM 服务
docker run --gpus all \
    -p 8000:8000 \
    -v $(pwd)/models:/root/.cache/huggingface/hub \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct
```

## 二、完整 Docker Compose 配置

### 2.1 生产级 docker-compose.yml

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-service
    restart: unless-stopped
    
    ports:
      - "8000:8000"
    
    volumes:
      # 模型文件持久化（避免每次重启重新下载）
      - ${MODEL_CACHE_DIR:-./models}:/root/.cache/huggingface/hub
      # LoRA 适配器目录（可选）
      - ${LORA_DIR:-./adapters}:/adapters
      # 数据目录
      - ./data:/data
    
    # GPU 直通配置
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: ${GPU_COUNT:-1}
              device_ids: ${GPU_IDS:-all}
    
    # 环境变量
    environment:
      - NVIDIA_VISIBLE_DEVICES=${GPU_IDS:-all}
      - VLLM_LOGGING_LEVEL=INFO
      - VLLM_HOST=0.0.0.0
      - VLLM_PORT=8000
    
    # 命令参数
    command: >
      --model ${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
      --tensor-parallel-size ${TP_SIZE:-1}
      --max-model-len ${MAX_MODEL_LEN:-8192}
      --gpu-memory-utilization ${GPU_UTILIZATION:-0.90}
      --dtype auto
      --trust-remote-code
      --enable-prefix-caching
      --port 8000
      ${EXTRA_ARGS:-}
    
    # 健康检查
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
    # 资源限制
    shm_size: '8gb'                    # 共享内存（重要！）
    ulimits:
      memlock: -1                       # 锁定内存页
      nproc: 65535                     # 进程数限制
      nofile:
        soft: 1048576
        hard: 1048576
    
    # 日志驱动
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"
        tag: "vllm"
    
    networks:
      - vllm-net

networks:
  vllm-net:
    driver: bridge
```

### 2.2 .env 环境变量文件

```bash
# .env.production — 不要提交到版本控制！

# ===== 模型配置 =====
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
# MODEL_PATH=/path/to/local/model  # 如果使用本地模型

# ===== 硬件配置 =====
GPU_COUNT=2
GPU_IDS="0,1"
TP_SIZE=2

# ===== 性能调优 =====
MAX_MODEL_LEN=8192
GPU_UTILIZATION=0.92
ENABLE_PREFIX_CACHING=true

# ===== 功能开关 =====
ENABLE_LORA=false
# LORA_MODULES=
MAX_LORAS=4

# ===== 量化 (可选) =====
QUANTIZATION=awq
# QUANTIZATION=gptq
# QUANTIZATION=fp8

# ===== 路径配置 =====
MODEL_CACHE_DIR=./models
LORA_DIR=./adapters

# ===== 额外启动参数 =====
EXTRA_ARGS=

# ===== 监控 =====
METRICS_PORT=9090
```

## 三、多阶段构建优化

### 3.1 多阶段 Dockerfile

```dockerfile
# Dockerfile.vllm-optimized
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# === Stage 1: 安装 Python 依赖 ===
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir /root/.cache/pip \
    -r requirements.txt

# === Stage 2: 运行时镜像 ===
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=/root/.cache/pip
ENV DEBIAN_FRONTEND=noninteractive

# 复制 venv 和已安装的包
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/pip /root/.cache/pip

# 创建非 root 用户（安全最佳实践）
RUN useradd -m -s /bin/bash vllm
USER vllm
WORKDIR /app

# 预下载常用模型到缓存层（可选，减少首次启动时间）
# RUN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('Qwen/Qwen2.5-7B-Instruct')"

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]
```

### 3.2 requirements.txt

```txt
# requirements.txt
vllm>=0.6.6
torch>=2.5.0
transformers>=4.40.0
accelerate>=0.30.0
safetensors>=0.4.0
outlines>=0.7.0
```

## 四、镜像构建与验证

### 4.1 构建脚本

```bash
#!/bin/bash
set -e

IMAGE_NAME="my-vllm-service"
IMAGE_TAG="v1.0.0"
REGISTRY="registry.example.com"

echo "===== 构建 vLLM Docker 镜像 ====="
docker build \
    -f Dockerfile.vllm-optimized \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "===== 镜像信息 ====="
docker images ${IMAGE_NAME}:${IMAGE_TAG}

echo "===== 扫描安全漏洞 ====="
# trivy image --severity HIGH,CRITICAL ${IMAGE_NAME}:${IMAGE_TAG}

echo "===== 本地测试运行 ====="
docker run --rm -it \
    --gpus '"device=0"' \
    -p 18000:8000 \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-model-len 1024 &

sleep 15

echo "===== 健康检查 ====="
curl -sf http://localhost:18000/health && echo "✅ 服务正常" || echo "❌ 服务异常"

echo "===== 推送到仓库 ====="
# docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
```

## 五、GPU 直通配置详解

### 5.1 NVIDIA Container Toolkit 配置检查清单

```bash
#!/bin/bash
# gpu_setup_check.sh — Docker GPU 环境检查

echo "===== NVIDIA 驱动检查 ====="
nvidia-smi || { echo "❌ NVIDIA 驱动未安装"; exit 1; }
echo "✅ 驱动版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"

echo ""
echo "===== NVIDIA Container Toolkit ====="
which nvidia-container-toolkit && echo "✅ 已安装" || {
    echo "⚠️  未安装，正在安装..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb .*#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg]#' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null \
    && sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit
    echo "✅ 安装完成"
}

echo ""
echo "===== 运行时配置检查 ====="
sudo nvidia-ctk runtime --runtime=docker || true
echo "✅ 运行时已配置"

echo ""
echo "===== 测试 GPU 可见性 ====="
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 六、常见 Docker 问题排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `Could not select device driver` | 未安装 nvidia-container-toolkit | `apt-get install nvidia-container-toolkit` |
| `CUDA out of memory` | GPU 显存不足或未指定 GPU | 加 `--gpus "\"device=0\""` 或减小模型 |
| `Error response from daemon: unknown` | Docker 版本过旧 | 升级 Docker 到 24+ |
| `Shared memory limit` | 默认 /dev/shm 太小 | 添加 `shm_size: '8gb'` |
| `OOM killed` | 模型太大或 batch size 过大 | 减小模型/量化/减少并发 |
| 首次启动极慢 | 首次需下载模型 | 使用 volume 挂载本地模型 |

---

## 七、总结

本节完成了 Docker 容器化部署的全部内容：

| 主题 | 核心要点 |
|------|---------|
| **官方镜像** | `vllm/vllm-openai:latest`，包含完整依赖链 |
| **Docker Compose** | 一键管理 GPU/卷/网络/健康检查/重启策略 |
| **环境变量** | `.env` 文件管理所有可配置项，不硬编码 |
| **GPU 直通** | `--gpus all` + nvidia-container-toolkit + `shm_size` |
| **多阶段构建** | Builder 层安装依赖 → Runtime 层精简镜像 |
| **非 root 用户** | 安全最佳实践，最小权限原则 |
| **模型持久化** | Volume 挂载 → 避免重复下载 |
| **健康检查** | Docker 内置 probe + `/health` 端点 |

**核心要点回顾**：

1. **`proxy_buffering off` 在 Nginx 中是第一铁律**——但在 Docker 中 `shm_size: '8gb'` 同样关键
2. **Volume 挂载模型目录是必须的**——否则每次重启都要重新下载 GB 级的模型
3. **多阶段构建可以显著减小镜像大小**——分离构建依赖和运行时
4. **`.env` 文件管理配置是生产标准实践**——不同环境只需改 env 文件
5. **先在本地 `docker run --rm` 测试通过** 再推到生产环境

下一节我们将学习 **Kubernetes 编排**——让 vLLM 服务实现真正的企业级弹性伸缩。
