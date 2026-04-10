# 10-1 Docker 部署深度实践

## Docker 部署：从开发到生产

在前面的章节中我们已经多次使用过 Docker 来运行 Ollama，但那些都是"够用就好"的临时配置。这一节我们将构建一个**生产级**的 Docker 部署方案——包含 GPU 直通、数据持久化、健康检查、安全加固和日志管理。

## 官方镜像解析

```
ollama/ollama:latest 镜像的层次结构:

ollama/ollama:latest
├── 基于 ubuntu:22.04 (或 rockylinux:9)
│
├── ollama 二进制文件 (/usr/local/bin/ollama)
│   ├── 这是 Go 编译的单体二进制
│   ├── 包含完整的 HTTP API 服务 + llama.cpp 推理引擎
│   └── 入口点: serve / run / create / pull ...
│
│
├── 模型默认存储路径: /root/.ollama/
│   ├── blobs/     → GGUF 权重文件（实际模型数据）
│   └── manifests/ → 模型清单文件（recipe）
│
└── 环境变量默认值:
    ├── OLLAMA_HOST=11434
    ├── OLLAMA_MODELS=/root/.ollama/models
    └── OLLAMA_KEEP_ALIVE=5m
```

## 完整生产级 docker-compose.yml

```yaml
# ============================================================
#  Ollama 生产环境部署 - 完整配置模板
#  适用场景: 团队共享 / 内网部署 / GPU 服务器
# ============================================================

version: '3.8'

services:
  # ==================== Ollama 主服务 ====================
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-prod
    restart: unless-stopped
    ports:
      - "11434:11434"
      # 注意：如果用 host network 模式，不需要暴露端口到宿主机
      # 因为 Nginx 反代会处理外部流量
    
    volumes:
      # 核心数据持久化：模型权重不可丢失！
      - ollama_models:/root/.ollama/models
      # 可选：预加载常用模型
      - ./preloaded:/preloaded:ro
      
    environment:
      # === 基本配置 ===
      - OLLAMA_HOST=0.0.0.0          # 允许所有接口访问（由 Nginx 控制外部访问）
      - OLLAMA_PORT=11434
      - OLLAMA_ORIGINS=*              # 允许所有来源（开发阶段）
      
      # === 性能调优 ===
      - OLLAMA_NUM_PARALLEL=4           # 并行批处理数
      - OLLAMA_MAX_LOADED_MODELS=3       # 最大同时加载模型数
      - OLLAMA_KEEP_ALIVE=10m            # 模型空闲后保留时间
      - OLLAMA_REQUEST_TIMEOUT=120         # 单请求超时(秒)
      - OLLAMA_DEBUG=false               # 生产环境关闭 debug
      
      # === NVIDIA GPU 配置 ===
      - NVIDIA_VISIBLE_DEVICES=0        # 使用第一张 GPU
      - CUDA_VISIBLE_DEVICES=0             # (同上)
      - OLLAMA_GPU_LAYERS=-1             # 自动决定 GPU 层数量
      - OLLAMA_GPU_OVERHEAD=0              # GPU 显存开销预留
      
      # === 安全相关 ===
      - OLLAMA_NOPRUNE=false              # 不跳过安全检查
      - OLLAMA_LOAD_FORMAT="auto"
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']
        limits:
          memory: 24g                    # 容器内存上限
    
    # === 健康检查 ===
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
    
    # === 日志配置 ===
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
        labels: "service=ollama,env=production"
    
    networks:
      - ollama-network

  # ==================== Nginx 反向代理 ====================
  nginx:
    image: nginx:alpine
    container_name: ollama-nginx
    restart: unless-stopped
    depends_on:
      - ollama
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./certs:/etc/nginx/certs:ro
    networks:
      - ollama-network

  # ==================== Redis 缓存（可选）====================
  redis:
    image: redis:7-alpine
    container_name: ollama-redis
    restart: unless-stopped
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - ollama-network

networks:
  ollama-network:
    driver: bridge
```

## Nginx 配置：生产级反代

```nginx
# nginx/conf.d/ollama.conf

# 上游 Ollama 服务
upstream ollama_backend {
    server ollama:11434;
    keepalive 32;
    keepalive_timeout 60;
}

# 限流配置
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/s;
limit_conn_zone conn_limit_zone:10;

server {
    listen 80;
    server_name _;  # 先监听 HTTP（测试用）
    
    # 安全头
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # 只暴露必要端点到 Ollama
    location /api/chat {
        limit_req zone=api_limit burst=50 nodelay;
        limit_conn conn_limit_zone 5;
        
        proxy_pass http://ollama:11434/api/chat;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;
        proxy_read_timeout 180s;
        
        # 请求体大小限制
        client_max_body_size 20m;
        
        # 禁用缓存（Ollama 的响应不应该被缓存）
        proxy_no_cache $http_cache_x_purge $request_uri $request_method;
        add_header Cache-Control "no-store";
        add_header Pragma "no-cache";
    }
    
    location /api/generate {
        limit_req zone=api_limit burst=30 nodelay;
        proxy_pass http://ollama:11434/api/generate;
        # ... 同上配置
    }
    
    location /api/embeddings {
        limit_req zone=api_limit burst=100 nodelay;
        proxy_pass http://ollama://11434/api/embeddings;
        # ... 同上配置
    }
    
    # 健康检查端点（供监控使用）
    location /health {
        access_log off;
        return 200 '{"status":"ok"}';
        content_type application/json;
    }
    
    # 其他所有路径返回 404
    location / {
        return 404 '{"error":"not found"}';
        content_type application/json;
    }
}
```

## Apple Silicon 特殊处理

在 Apple Silicon Mac 上部署 Ollama 有一些需要注意的特殊情况：

```yaml
# apple-silicon.docker-compose.yml

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-mac
    restart: unless-stopped
    
    # ⚠️ Apple Silicon Docker 的 GPU 直通需要特殊配置
    # 方案 A: 使用 --platform linux/amd64 + QEMU 模拟
    platform: linux/amd64
    
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_NUM_PARALLEL=4
      # Metal 加速自动启用（Ollama 在 Mac 上会自动检测）
    
    volumes:
      - ollama_mac_data:/root/.ollama
    
    deploy:
      resources:
        limits:
          memory: 16g  # Mac 内存要足够大
    
    # 如果确实需要 GPU 直通（实验性）
    # device_nvidia: true 不适用于 Mac
    # 需要通过特殊方式实现
```

**注意**: 在 Mac 上通常**不推荐使用 Docker 运行 Ollama**——直接 `brew install ollama && ollama serve` 更简单、性能更好。Docker 版本在 Mac 上会有额外的虚拟化开销。

## 多容器编排：Kubernetes Deployment

如果你的组织已经使用 Kubernetes，这里是一个基础的 K8s Deployment：

```yaml
# k8s/ollama-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-service
  namespace: ai
spec:
  replicas: 2  # 两个副本做负载均衡
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
          - containerPort: 11434
        env:
          - name: OLLAMA_HOST
            value: "0.0.0.0"
          - name: OLLAMA_NUM_PARALLEL
            value: "4"
          - name: OLLAMA_MAX_LOADED_MODELS
            value: "2"
          - name: OLLAMA_KEEP_ALIVE
            value: "10m"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2000m"
          limits:
            memory: "16Gi"
            cpu: "4000m"
        volumeMounts:
          - name: model-data
            mountPath: /root/.ollama
            persistentVolumeClaimName: ollama-pvc
      readinessProbe:
        httpGet:
          path: /api/tags
          port: 11434
          initialDelaySeconds: 15
          periodSeconds: 30
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ollama-pvc
  namespace: ai
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 100Gi
  volumeMode: Filesystem
---
apiVersion: v1
kind: Service
metadata:
  name: ollama-service
  namespace: ai
spec:
  selector:
    app: ollama
  ports:
    - port: 11434
      targetPort: 11434
  type: ClusterIP
```

## 本章小结

这一节完成了生产级 Docker 部署的全部关键配置：

1. **完整 docker-compose.yml** 包含 Ollama + Nginx + Redis 三件套
2. **Nginx 反代** 提供了限流、超时控制和安全头
3. **GPU 直通** 通过 NVIDIA Container Toolkit 和 `deploy.resources.reservations.devices` 实现
4. **Apple Silicon** 上建议直接原生运行而非 Docker（避免虚拟化开销）
5. **Kubernetes Deployment** 提供了多副本负载均衡的生产级方案
6. **核心原则：模型数据必须挂载为 Volume**——绝不能放在容器层

下一节我们将讨论安全加固。
