---
title: 容器化部署：Docker + Kubernetes
description: Dockerfile 最佳实践、多阶段构建、docker-compose 本地编排、K8s 生产部署清单、Helm Chart、滚动更新与回滚
---
# 容器化部署：Docker + Kubernetes

上一节我们把 LangChain 应用封装成了 FastAPI 服务。现在需要把它**容器化**，让它可以在任何环境中一致地运行——开发者的笔记本、测试服务器、生产集群。

## 为什么需要容器化

传统的部署方式（直接在服务器上 `pip install && python main.py`）有以下痛点：

| 痛点 | 具体表现 |
|------|---------|
| **环境不一致** | 开发环境 Python 3.11，服务器 3.9 → 依赖冲突 |
| **依赖地狱** | `requirements.txt` 装了但系统库版本不对 |
| **部署复杂** | 手动 SSH 上传代码 → 安装依赖 → 重启服务 →祈祷不报错 |
| **扩缩容困难** | 加一台新服务器要重复所有步骤 |
| **回滚困难** | 出问题了想回到上一个版本？没有快照 |

Docker 把应用及其全部依赖打包成一个**不可变的镜像**，确保"在我机器上能跑 = 在任何地方都能跑"。Kubernetes 则在此基础上提供**自动扩缩容、健康检查、滚动升级、故障自愈**等生产级能力。

## Dockerfile：从零到优化

### 基础版 Dockerfile（能用但不推荐）

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

这个 Dockerfile 能工作，但有严重问题：
- 每次改一行代码都要重新安装所有依赖（构建慢）
- 镜像体积大（包含 build 工具）
- 以 root 用户运行（安全风险）

### 生产级 Dockerfile

```dockerfile
# ===== 第一阶段：构建 =====
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== 第二阶段：运行 =====
FROM python:3.11-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000",
     "--workers", "2", "--log-level", "info"]
```

关键优化点解析：

| 优化 | 说明 | 效果 |
|------|------|------|
| **多阶段构建** | builder 阶段装依赖，runtime 阶段只复制产物 | 镜像缩小 60%+ |
| **非 root 用户** | 创建专用用户运行 | 安全性提升 |
| **最小化 apt 安装** | 只装 curl（用于 healthcheck），装完清理 | 减少攻击面 |
| **HEALTHCHECK** | Docker 层面的存活检测 | K8s 能自动重启不健康的容器 |
| **--workers=2** | uvicorn 多 worker 并发处理请求 | 提升吞吐量 |

### requirements.txt 优化

```txt
# 核心依赖
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
sse-starlette>=1.8.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# LangChain 生态
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
langchain-chroma>=0.1.0

# 数据库
chromadb>=0.4.0
sqlalchemy>=2.0.0

# 可观测性
langsmith>=0.1.0

# 工具
python-multipart>=0.0.6   # FastAPI 文件上传支持
httpx>=0.25.0             # 异步 HTTP 客户端
```

构建并运行：

```bash
cd api_service
docker build -t langchain-api:latest .
docker run -d \
    --name langchain-api \
    -p 8000:8000 \
    -e OPENAI_API_KEY="sk-xxx" \
    -e ENV="production" \
    -v ./data:/app/data \
    langchain-api:latest

# 验证
curl http://localhost:8000/health
```

## docker-compose：本地开发与多服务编排

单容器用 `docker run` 就够了，但真实场景通常需要多个服务协同工作：API 服务 + Redis（缓存/会话存储）+ 向量数据库 + 可能还有前端。

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: langchain-api
    ports:
      - "${API_PORT:-8000}:8000"
    environment:
      - ENV=${ENV:-development}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379/0
      - LANGCHAIN_TRACING_V2=${LANGCHAIN_TRACING_V2:-false}
      - LANGCHAIN_PROJECT=langchain-api-local
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M

  redis:
    image: redis:7-alpine
    container_name: langchain-redis
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  redis_data:

networks:
  default:
    name: langchain-network
```

启动完整栈：

```bash
# 一键启动所有服务
docker compose up -d

# 查看日志
docker compose logs -f api

# 扩展 API 到 3 个实例（配合负载均衡）
docker compose up -d --scale api=3

# 停止并清理
docker compose down -v
```

## Kubernetes 生产部署

当流量增长到单机无法承载时，就需要 Kubernetes 了。K8s 不是 Docker 的替代品，而是**容器的编排系统**——它管理着数百个容器的生命周期。

### 核心资源清单

```yaml
# k8s/langchain-api/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-api
  labels:
    app: langchain-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langchain-api
  template:
    metadata:
      labels:
        app: langchain-api
    spec:
      containers:
      - name: api
        image: your-registry/langchain-api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENV
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: langchain-secrets
              key: openai-api-key
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: langchain-data-pvc
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - langchain-api
              topologyKey: kubernetes.io/hostname
```

### Service 与 Ingress

```yaml
# k8s/langchain-api/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: langchain-api-svc
spec:
  type: ClusterIP
  selector:
    app: langchain-api
  ports:
  - port: 80
    targetPort: 8000
  sessionAffinity: None
```

```yaml
# k8s/langchain-api/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: langchain-api-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "120"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: langchain-api-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: langchain-api-svc
            port:
              number: 80
```

### HPA：自动水平扩缩容

```yaml
# k8s/langchain-api/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langchain-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langchain-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization:70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

HPA 配置解读：
- **最小 3 副本**：保证基本可用性和故障容忍
- **最大 20 副本**：控制成本上限
- **CPU 目标 70%**：平均 CPU 超过 70% 时触发扩容
- **快速扩容、缓慢缩容**：流量突增时快速响应（30秒内翻倍），流量下降时保守缩容（5分钟才减少），避免抖动

### Secrets 管理

绝对不要把 API Key 写在 YAML 文件中提交到 Git：

```bash
# 创建 Secret
kubectl create secret generic langchain-secrets \
  --from-literal=openai-api-key='sk-xxx' \
  --from-literal=langsmith-key='lsv2_xxx' \
  --dry-run=client -o yaml > k8s/secrets.yaml

# 确认 secrets.yaml 不包含明文密钥后提交
git add k8s/secrets.yaml
```

对于更严格的场景，使用外部密钥管理系统：
- **HashiCorp Vault**: 企业级密钥管理
- **AWS Secrets Manager / GCP Secret Manager**: 云原生方案
- **Sealed Secrets**: Kubernetes 原生加密方案

### 部署命令速查

```bash
# 应用所有资源配置
kubectl apply -f k8s/

# 查看部署状态
kubectl get pods -l app=langchain-api -w
kubectl describe deployment langchain-api

# 查看 HPA 状态
kubectl get hpa langchain-api-hpa

# 查看日志
kubectl logs -f deployment/langchain-api

# 手动扩容到 5 副本
kubectl scale deployment langchain-api --replicas=5

# 滚动更新到 v1.1.0
kubectl set image deployment/langchain-api \
    api=your-registry/langchain-api:v1.1.0

# 回滚到上一版本
kubectl rollout undo deployment/langchain-api

# 查看发布历史
kubectl rollout history deployment/langchain-api

# 强制重启所有 Pod（配置变更后）
kubectl rollout restart deployment/langchain-api
```

## 从开发到生产的部署流水线

一个完整的 CI/CD 流水线应该包含以下阶段：

```
Git Push
    │
    ▼ [CI: 测试]
    ├── 单元测试 (pytest)
    ├── 类型检查 (mypy)
    ├── Lint (ruff)
    └── 构建镜像 (docker build)
         │
    ▼ [推送到仓库]
    Docker Registry (ECR/GCR/Harbor)
         │
    ▼ [CD: 部署到 Staging]
    ├── kubectl apply (staging namespace)
    ├── 运行回归测试 (13.4 的离线评估)
    └── 人工审批（可选）
         │
    ▼ [CD: 部署到 Production]
    ├── kubectl apply (production namespace)
    ├── 滚动更新（逐步替换 Pod）
    └── 监控验证（LangSmith + Prometheus）
```

## 常见误区

**误区一：把 .env 文件打包进镜像**。环境变量应该通过 Docker `-e` 参数或 K8s `env` / `secret` 注入，而不是写死在镜像里。否则一旦 Key 泄露就需要重新构建和推送镜像。

**误区二：不设置资源限制（limits）**。没有 CPU/memory limit 的 Pod 可能会吃掉整台机器的资源，导致其他服务被驱逐（OOM Kill）。**必须为每个容器设置 request 和 limit**。

**误区三：用 latest 标签做生产部署**。`latest` 是不可控的——你不知道当前跑的是哪个版本的代码。生产环境必须使用**语义化版本号标签**（如 `v1.2.3`）或 Git commit hash。
