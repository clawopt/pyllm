# Kubernetes 编排

> **白板时间**：Docker Compose 适合单机部署。但当你的服务需要跨多节点、自动故障恢复、按流量弹性伸缩时——你需要 **Kubernetes**。K8s 是容器编排的事实标准，它让 vLLM 服务具备企业级的可靠性和可扩展性。

## 一、K8s 部署架构

### 1.1 架构概览

```
                    Internet
                        │
              ┌─────────▼─────────┐
              │   Ingress Controller │
              │   (TLS + 路由)      │
              └─────────┬─────────┘
                        │
        ┌───────────────▼───────────────┐
        │      Service: vllm-svc         │
        │      (ClusterIP, port 80)     │
        └──────────┬────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │     Deployment: vllm-deploy    │
    │     (ReplicaSet 管理)          │
    │     replicas: 2               │
    │     ┌──────────┬──────────┐    │
    │     │ Pod vllm  │ Pod vllm  │    │
    │     │ :8000    │ :8000    │    │
    │     │ GPU 0,1  │ GPU 2,3  │    │
    │     └──────────┴──────────┘    │
    └────────────────────────────────┘
```

### 1.2 资源需求

| 组件 | 资源请求 | 说明 |
|------|---------|------|
| **Pod** | GPU: 1-2 卡; CPU: 4-8 核; Memory: 32-64GB | 取决于模型大小 |
| **GPU** | NVIDIA A100 80GB / H100 80GB / RTX 4090 | 需要支持 MIG 或直通 |
| **PVC** | 50-200 GB (模型权重+KV Cache) | 模型文件持久化 |
| **Node** | 需要安装 NVIDIA Device Plugin 和 Container Toolkit | GPU 节点前置条件 |

## 二、Deployment YAML

### 2.1 完整 Deployment 配置

```yaml
# k8s/vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: llm-system
  labels:
    app: vllm
    component: inference
spec:
  replicas: 2                    # 初始副本数（可根据负载自动扩缩）
  selector:
    matchLabels:
      app: vllm
      component: inference
  template:
    metadata:
      labels:
        app: vllm
        component: inference
    spec:
      # ===== 节点选择（仅调度到 GPU 节点）=====
      nodeSelector:
        gpu-type: "nvidia-a100"       # 或 nvidia-h100 / rtx4090
      
      # ===== 容忍度（允许在 GPU 节点运行）=====
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: NoSchedule
      
      # ===== 容器配置 =====
      initContainers:
        # Init Container: 预下载模型到共享存储
        - name: model-downloader
          image: busybox:latest
          command:
            - sh
            - -c |
              set -ex;
              echo "[Init] 开始预下载模型...";
              if [ -n "$MODEL_HF_NAME" ]; then
                echo "[Init] 从 HuggingFace 下载: $MODEL_HF_NAME";
                python3 -c "
import subprocess
subprocess.run([
    'python3', '-m', 'huggingface_hub.cli',
    'download', '--local-dir', '/mnt/models',
    '$MODEL_HF_NAME'
]);
                " || echo "[Init] 下载失败，使用本地模型";
              else
                echo "[Init] 使用本地模型: $MODEL_LOCAL_PATH";
              fi;
              
              # 验证模型文件存在
              ls -la /mnt/models/ 2>/dev/null || {
                echo "[ERROR] 模型目录为空!";
                exit 1;
              }
              
              echo "[Init] 模型就绪!";
            "
          env:
            - name: MODEL_HF_NAME
              value: "${MODEL_HF_NAME:-Qwen/Qwen2.5-7B-Instruct}"
            - name: MODEL_LOCAL_PATH
              value: "/mnt/models"
          volumeMounts:
            - name: model-storage
              mountPath: /mnt/models
              readOnly: false
      
      containers:
        - name: vllm-server
          image: vllm/vllm-openai:${VLLM_VERSION:-latest}
          ports:
            - containerPort: 8000
          env:
            - name: NVIDIA_VISIBLE_DEVICES
              value: "${NVIDIA_VISIBLE_DEVICES:-all}"
            - name: VLLM_LOGGING_LEVEL
              value: "INFO"
            - name: VLLM_HOST
              value: "0.0.0.0"
            - name: VLLM_PORT
              value: "8000"
          
          # 启动命令
          command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
          args:
            - "--model=/mnt/models/${MODEL_FILENAME}"
            - "--tensor-parallel-size=${TP_SIZE:-1}"
            - "--max-model-len=${MAX_MODEL_LEN:-8192}"
            - "--gpu-memory-utilization=${GPU_UTILIZATION:-0.90}"
            - "--dtype=auto"
            - "--trust-remote-code"
            - "--enable-prefix-caching=${ENABLE_PREFIX_CACHE:-true}"
            - "--port=8000"
          
          # ===== 资源请求 =====
          resources:
            limits:
              nvidia.com/gpu: ${GPU_COUNT:-1}
              memory: "${MEMORY_LIMIT:-64Gi}"
            requests:
              nvidia.com/gpu: ${GPU_REQUEST:-1}
              cpu: "${CPU_REQUEST:-4000m}"
              memory: "${MEMORY_REQUEST:-32Gi}"
          
          # ===== Liveness Probe (存活检查) =====
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120     # 模型加载需要时间
              periodSeconds: 15
              timeoutSeconds: 5
              failureThreshold: 3
          
          # ===== Readiness Probe (就绪检查) =====
          readinessProbe:
            httpGet:
              path: /v1/models
              port: 8000
            initialDelaySeconds: 130
              periodSeconds: 10
              timeoutSeconds: 5
            failureThreshold: 3
          
          # ===== Volume 挂载 =====
          volumeMounts:
            - name: model-storage
              mountPath: /mnt/models
              readOnly: true             # 只读，模型由 init container 写入
            - name: lora-adapters
              mountPath: /adapters
              readOnly: true
            - name: shm-volume
              mountPath: /dev/shm
          
          securityContext:
            privileged: false
            runAsNonRoot: true
            allowPrivilegeEscalation: false
            capabilities:
              add: ["SYS_ADMIN"]
      
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: vllm-model-pvc
        - name: lora-adapters
          persistentVolumeClaim:
            claimName: vllm-lora-pvc
        - name: shm-volume
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi

---
# PVC: 模型存储
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-model-pvc
  namespace: llm-system
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi                  # 根据实际模型大小调整

---
# PVC: LoRA 存储较小的适配器
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-lora-pvc
  namespace: llm-system
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
# Service: Cluster 内部访问
apiVersion: v1
kind: Service
metadata:
  name: vllm-svc
  namespace: llm-system
spec:
  selector:
    app: vllm
    component: inference
  ports:
    - port: 8000
      targetPort: 8000
      name: http
  type: ClusterIP

---
# Ingress: 外部访问（可选）
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-ingress
  namespace: llm-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  rules:
    - host: llm.yourcompany.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: vllm-svc
                port:
                  number: 8000
```

## 三、部署操作

### 3.1 一键部署

```bash
#!/bin/bash
# deploy-vllm-k8s.sh — K8s 一键部署脚本

set -e

NAMESPACE="llm-system"

echo "===== 1. 创建 Namespace ====="
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

echo ""
echo "===== 2. 应用 PV/PVC/Deployment/Service ====="
kubectl apply -f k8s/vllm-deployment.yaml

echo ""
echo "===== 3. 等待 Pod 就绪 ====="
echo "等待 Init Container 完成模型下载..."
kubectl -n $NAMESPACE rollout status deployment/vllm-inference --timeout=600s

echo ""
echo "===== 4. 验证部署状态 ====="
kubectl -n $NAMESPACE get pods -l app=vllm
kubectl -n $NAMESPACE get svc
kubectl -n $namespace get pvc

echo ""
echo "===== 5. 端口转发测试 ====="
kubectl -n $namespace port-forward svc/vllm-svc 8000:8000 &
PF_PID=$!
sleep 5

curl -sf http://localhost:8000/health && echo "✅ vLLM 服务正常运行" || echo "❌ 服务异常"

kill $PF_PID 2>/dev/null
wait $PF_ID 2>/dev/null

echo ""
echo "===== 部署完成! ====="
echo "Service: vllm-svc.$NAMESPACE.svc.cluster.local"
echo "外部访问: 配置 Ingress 或使用 NodePort"
```

## 四、HPA 自动伸缩

### 4.1 基于 Custom Metrics 的 HPA

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: llm-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-inference
  
  minReplicas: 1
  maxReplicas: 6                 # 最大扩展到 6 个副本
  replicas: 2                 # 默认副本数
  
  metrics:
    # 基于 GPU 利用率扩缩
    - type: Pods
      pods:
        metricName: gpu_utilization
        target:
          type: AverageValue
          averageValue: "80"        # GPU 利用率 > 80% 时扩容
    
    # 基于队列深度扩缩
    - type: Pods
      pods:
        metricName: queue_depth
        target:
          type: AverageValue
          averageValue: "20"        # 平均排队 > 20 时扩容
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300   # 冷却 5 分钟再缩容
      policies:
        - type: Percent
          value: 20                     # 每次最多缩容 20%
        - type: Pods
          value: 1                      # 至少保留 1 个副本
```

**注意**：`gpu_utilization` 和 `queue_depth` 是自定义指标，需要配合 Prometheus Adapter：

```yaml
# k8s/prometheus-adapter.yaml (ConfigMap)
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-prometheus-adapter
  namespace: llm-system
data:
  config.yml: |
    rules:
      - pattern: "vllm:num_requests_running"
        source: { __name__: { __address__ }:8000/metrics }
        metric: num_requests_running
        type: Gauge
      - pattern: "vllm:gpu_cache_usage_perc"
        source: { __name__: { __address__ }:8000/metrics }
        metric: gpu_cache_usage_perc
        type: Gauge
```

## 五、Node 亲和性与拓扑约束

### 5.1 确保 Pod 调度到正确的 GPU 节点

```yaml
# 在 Deployment spec 中添加：
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              key: nvidia.com/gpu.product
                operator: In
                values:
                  - "A100-SXM4-80GB"
                  - "H100-80G"
                  - "RTX-4090"
      preferredDuringSchedulingIgnoredDuringExecution:
        weight: 100
        preference:
          - matchExpressions:
              key: topology.kubernetes.io/zone
                operator: In
                values:
                  - "us-west-2a"
```

---

## 六、总结

本节完成了 Kubernetes 生产级部署：

| 主题 | 核心要点 |
|------|---------|
| **Deployment** | ReplicaSet 管理 Pod 生命周期，支持滚动更新 |
| **Init Container** | 预下载模型到共享 PVC，避免启动时下载延迟 |
| **Probe** | Liveness（存活）+ Readiness（就绪）双保险 |
| **PVC** | 模型和 LoRA 的持久化存储，Pod 重启不丢数据 |
| **HPA** | 基于自定义指标（GPU利用率/队列深度）的自动伸缩 |
| **NodeSelector** | 确保只调度到有正确 GPU 类型的节点 |
| **Ingress** | TLS 终结 + 域名路由的外部访问 |

**核心要点回顾**：

1. **Init Container 是生产环境的必备组件**——它把模型下载从"启动时"提前到"调度时"
2. **`initialDelaySeconds: 120`** 是 Probe 的关键参数——模型加载需要 1-2 分钟
3. **HPA 让服务能应对流量波动**——但 GPU 场景下要谨慎设置 maxReplicas（每张卡都很贵）
4. **PVC 用 `ReadWriteOnce`**——模型文件只需要写一次，之后所有 Pod 只读挂载
5. **Node Affinity 确保大模型调度到有足够显存的节点**

下一节我们将学习 **监控、日志与告警体系**。
