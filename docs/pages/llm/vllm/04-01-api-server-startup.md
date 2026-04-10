# API Server 启动与配置

## 白板导读

vLLM 最具工程价值的特性之一，就是它开箱即用的 **OpenAI 兼容 HTTP API**。这意味着你在任何已经对接过 OpenAI GPT-4 的应用中，只需要修改一行 `base_url`，就能无缝切换到本地 vLLM 服务——不需要学习新的 SDK，不需要修改请求/响应的解析逻辑，甚至不需要告诉你的团队"我们换了后端"。这一节将从启动命令的最简形式出发，逐步展开到生产级完整配置，覆盖模型相关、并行分布式、性能调优、功能开关和服务端参数的全部细节。你将学会如何用一份 `docker-compose.yml` 文件定义一个生产级的 vLLM API 服务。

---

## 1.1 最小可运行启动

让我们从最简单的情况开始——在一台有 GPU 的机器上，用最少的参数启动一个能响应请求的 API Server：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

就这一行。执行后你会看到终端输出类似：

```
INFO 15:30:00.00 api_server.py:168] Initializing vLLM with config:
  model='Qwen/Qwen2.5-7B-Instruct'
  dtype=auto
INFO 15:30:00.00 config.py:975] Setting max_model_len to 32768 for model Qwen/Qwen2.5-7B-Instruct
INFO 15:30:00.00 config.py:1002] Setting RoPE scaling to 'linear' for model Qwen/Qwen2.5-7B-Instruct
INFO 15:30:00.01 model_runner.py:317] Loading model weights from Qwen/Qwen2.5-7B-Instruct
  Loading safetensors checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
  Loading safetensors checkpoint shards:  25%|██▊       | 1/4 [00:03<..., ?it/s]
  Loading safetensors checkpoint shards:  50%|█████     | 2/4 [00:07<..., ?it/s]
  Loading safetensors checkpoint shards: 100%|██████████| 4/4 [00:14<..., ?it/s]
INFO 15:30:00.14 model_runner.py:340] Model loading took 14.32 seconds
INFO 15:30:00.14 engine.py:296] # GPU blocks: 42831, # CPU blocks: 2048
INFO 15:30:00.14 engine.py:302] Maximum concurrency for 32K context: 69 sequences
INFO 15:30:00.14 engine.py:306] KV cache memory: 12.42 GiB
INFO 15:30:00.14 gpu_mapping.py:123] Found 1 GPU(s).
INFO 15:30:00.14 gpu_executor.py:89] Initializing VLLM on GPU 0...
INFO 15:30:00.16 llm_engine.py:161] init engine (profile=False) took 16.21 seconds
INFO 15:30:00.16 api_server.py:172] Application startup complete.
INFO 15:30:00.16 api_server.py:174] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

当看到 **`Uvicorn running on http://0.0.0.0:8000`** 这一行时，服务就已经准备好了。打开另一个终端验证：

```bash
# 健康检查
curl -s http://localhost:8000/health

# 预期返回: {"status":"ok"}

# 模型列表
curl http://localhost:8000/v1/models | python3 -m json.tool

# 发送第一个请求
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"你好"}]}'
```

三个端点全部正常工作 → 你已经有了一个可用的 LLM API 服务。

---

## 1.2 启动参数全景图

上一章（01-03）我们已经介绍过启动参数的分类。这里以**参考卡片**的形式给出快速查阅版：

### 模型与数据类

| 参数 | 默认值 | 说明 | 生产建议 |
|:---|:---|::---|:---|
| `--model` | *必填* | HuggingFace ID 或本地路径 | 使用具体版本 tag（如 `Qwen/Qwen2.5-7B-Instruct`） |
| `--dtype` | `auto` | 权重精度：auto/half/bfloat16/float16/float32 | FP16 足够；BF16 训练兼容性更好 |
| `--trust-remote-code` | `false` | 是否信任模型自定义代码 | **国产模型必须开启**（Qwen/DeepSeek/Yi 等） |
| `--revision` | `main` | 模型的 git branch/tag | 锁定版本避免更新导致的不一致 |
| `--download-dir` | HF 缓存目录 | 模型文件下载位置 | 生产环境设为共享存储（NFS） |
| `--load-format` | `auto` | 加载格式：pt/safetensors/dummy/dtensor/sharded_state | 一般不用改 |

### 并行与分布式类

| 参数 | 默认值 | 说明 | 注意事项 |
|:---|:---|:|:---|
| `--tensor-parallel-size` | `1` | 张量并行 GPU 数量 | 7B→1, 13B→2, 70B→4, 100B+→8 |
| `--pipeline-parallel-size` | `1` | 流水线并行 Stage 数 | 通常配合 TP 使用（TP×PP=总GPU数） |
| `--worker-use-ray` | `false` | 用 Ray 管理多 Worker | 多机部署时启用 |
| `--distributed-executor-backend` | `None` | 分布式后端: ray/mp | Ray 更成熟 |

### 性能与资源类（最重要！）

| 参数 | 默认值 | 含义 | 调优指南 |
|:---|:---|:---|:---|
| `--max-model-len` | 模型默认值 | 单个请求最大上下文长度 | **核心参数！** 直接决定 KV Cache 大小和并发能力 |
| `--gpu-memory-utilization` | `0.90` | GPU 显存给模型+KV 的比例 | 0.85(保守) / 0.92(推荐) / 0.95(激进) |
| `--max-num-seqs` | `256` | 同时运行的最大序列数 | 根据实际 QPS 设置 |
| `--max-num-batched-tokens` | `自动` | 每 iteration 最大 token 数 | 自动计算通常够用 |
| `--scheduler-delay-factor` | *未设置* | 调度延迟因子 | 见 Ch03 决策树 |
| `--swap-space` | `4` | CPU swap 空间 (GiB) | 0=禁用抢占 / 4=标准 / 8+=重度 |
| `--cpu-offload-gb` | `0` | CPU 卸载模型层数 (GiB) | 模型太大单卡不够时使用 |

### 功能开关类

| 参数 | 默认 | 功能 |
|:---|:---|:---|
| `--enable-lora` | `false` | LoRA 适配器支持（Ch08 详细讲解） |
| `--max-loras` | `1` | 同时加载的最大 LoRA 数量 |
| `--enable-auto-tool-choice` | `false` | 自动工具选择 |
| `--enable-prefix-caching` | `false` | 前缀缓存（PagedAttention 高级特性） |
| `--enforce-eager` | `false` | 强制禁用 CUDA Graphs（调试用） |
| `--speculative-model` | `None` | Draft 模型路径（Ch10 Speculative Decoding） |

### 服务端类

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--host` | `0.0.0.0` | 监听地址（0.0.0.0 = 所有网卡） |
| `--port` | `8000` | 监听端口 |
| `--ssl-keyfile` | - | SSL 私钥文件路径 |
| `--ssl-certfile` | - | SSL 证书文件路径 |
| `--disable-log-requests` | `false` | 关闭请求日志（高吞吐推荐开启） |

---

## 1.3 生产级 docker-compose.yml 完整模板

这是可以直接用于生产的配置，包含 Nginx 反向代理、健康检查、资源限制：

```yaml
# docker-compose.prod.yml — vLLM 生产级部署
# 包含: vLLM + Nginx + 可选监控组件

version: '3.8'

services:
  # ===== vLLM 推理引擎 =====
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-inference
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ${MODEL_CACHE:-/data/models}:/root/.cache/huggingface:rw
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['${GPU_IDS:-0}']
    command: >
      --model ${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
      --tensor-parallel-size ${TP_SIZE:-1}
      --max-model-len ${MAX_LEN:-16384}
      --gpu-memory-utilization ${GPU_UTIL:-0.92}
      --max-num-seqs ${MAX_SEQS:-128}
      --scheduler-delay-factor ${SCHED_DELAY:-0.1}
      --swap-space ${SWAP_SPACE:-4}
      --trust-remote-code
      --enable-prefix-caching
      --disable-log-requests
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 30s
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"
        compress: "true"

  # ===== Nginx 反向代理（可选但强烈推荐）=====
  nginx:
    image: nginx:1.25-alpine
    container_name: vllm-nginx
    depends_on:
      - vllm
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
    depends_on:
      - vllm
    restart: unless-stopped

  # ===== Prometheus（监控，可选）=====
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: vllm-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - vllm
    restart: unless-stopped

  # ===== Grafana（可视化，可选）=====
  grafana:
    image: grafana/grafana:11.0.0
    container_name: vllm-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PW:-changeme}
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

networks:
  default:
    driver: bridge
```

配套的 Nginx 配置文件 `nginx/conf.d/vllm.conf`：

```nginx
# nginx/conf.d/vllm.conf — vLLM 反代 + 安全加固

limit_req_zone $binary_remote_addr zone=api_limit:10m rate=20r/s;
limit_conn_zone conn_per_ip_zone=10 burst=20 nodelay;

server {
    listen 80;
    server_name _;

    location / {
        # SSE 流式响应必须关闭缓冲!
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_pass http://vllm:8000;

        # 推理请求超时时间
        proxy_read_timeout 300s;
        proxy_send_timeout 60s;

        # 限流（引用上面定义的限制区）
        limit_req zone=api_limit burst=30 nodelay;
        limit_conn zone=conn_per_ip burst=20 nodelay;

        # 安全头
        add_header X-Content-Type-Options "nosniff";
        add_header X-Frame-Options "SAMEORIGIN";
    }

    location /health {
        access_log off;
        return 200 '{"status":"ok"}';
        content_type application/json;
    }

    location /metrics {
        proxy_pass http://prometheus:9090/metrics;
    }
}
```

---

## 1.4 EngineArgs 配置文件方式

除了命令行参数，vLLM 还支持通过 YAML/TOML 文件来管理复杂配置：

```yaml
# vllm_config.yaml — vLLM 引擎配置文件

model: "Qwen/Qwen2.5-7B-Instruct"
tokenizer: null
revision: "main"
tokenizer_mode: auto
trust_remote_code: true
dtype: auto
download_dir: "/data/models"
load_format: auto

tensor_parallel_size: 1
pipeline_parallel_size: 1
worker_use_ray: false
distributed_executor_backend: null

max_model_len: 16384
gpu_memory_utilization: 0.92
swap_space: 4
cpu_offload_gb: 0
num_gpu_blocks_override: null
max_num_seqs: 128
max_num_batched_tokens: null
scheduler_delay_factor: 0.1
  
enable_lora: false
max_loras: 1
max_lora_rank: 16
max_lora_size: 64

enable_prefix_caching: true
enforce_eager: false
speculative_model: null
num_speculative_tokens: 5

host: "0.0.0.0"
port: 8000
ssl_keyfile: null
ssl_certfile: null
disable_log_requests: true
```

启动时指定配置文件：

```bash
python -m vllm.entrypoints.openai.api_server \
    --config vllm_config.yaml
```

> **配置文件的优点**：
> 1. **可读性好**：YAML 的层级结构比长命令行更清晰
> 2. **版本控制**：可以用 Git 管理 `vllm_config.yaml` 的变更历史
> 3. **环境隔离**：不同环境（dev/staging/prod）用不同配置文件
> 4. **减少人为错误**：不会因为漏写某个 `--` 或引号错误导致参数丢失

---

## 1.5 多模型同时加载

vLLM 支持在同一个 API Server 上**同时加载多个模型**并通过路由分发请求到不同模型：

```bash
# 方式一：启动时指定多个模型（不推荐，所有模型同时占用显存）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8b-instruct \
    --model Qwen/Qwen2.5-7b-instruct \
    --model mistral/Mistral-7B-Instruct-v0.3 \
    --port 8000
```

⚠️ **注意**：这种模式下所有模型的权重都会被加载到 GPU 中，总显存需求是各模型之和。只适合模型都很小的场景（如都是 1-3B 的小模型组合）。

```bash
# 方式二（推荐）：动态切换模型
# 先启动一个轻量级模型作为默认
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5b-instruct \
    --port  --trust-remote-code
    
# 然后在请求中动态切换：
#   POST /v1/chat/completions {"model": "meta-llama/Llama-3.1-8b-instruct", ...}
#   vLLM 会自动下载并加载 Llama 3.1 8B（首次请求会有冷启动延迟）
#   后续对 Llama 的请求直接命中缓存
```

```bash
 # 方式三（高级）：vLLM 不直接加载模型，完全按需
# 通过 --model 引用一个"模型路由器"服务
# 这个模式需要额外的 router 组件（详见 Ch09 架构设计）
```

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **最简启动** | `python -m vllm.entrypoints.openai.api_server --model <name> --port 8000` 一行搞定 |
| **五大参数类别** | 模型数据 / 并行分布式 / 性能资源 / 功能开关 / 服务端（共 30+ 参数） |
| **Top 5 关键参数** | `--model`(必填) / `--tp`(多卡) / `--max-model-len`(上下文) / `--gpu-mem-util`(显存比) / `--delay-factor`(调度延迟) |
| **生产部署** | Docker Compose + Nginx 反代 + Health Check + Prometheus/Grafana + 日志持久化 |
| **配置文件** | 支持 YAML 格式 (`--config`)，优于超长命令行；支持 Git 版本管理 |
| **多模型** | 同进程多模型需谨慎（显存叠加）；推荐动态按需加载或外部路由器模式 |

> **一句话总结**：API Server 是 vLLM 对外暴露能力的窗口。它的启动虽然只需一行命令，但生产级部署需要精心调优 30+ 个参数——其中 `--max-model-len`（决定 KV Cache 大小和并发上限）、`--gpu-memory-utilization`（决定显存分配策略）、`--max-num-seqs`（控制并发水位线）是最关键的三个"黄金参数"。配合 Docker Compose 和 Nginx 反向代理，你可以构建一个既高性能又安全可靠的 LLM API 服务。
