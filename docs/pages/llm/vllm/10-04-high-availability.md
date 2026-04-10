# 10.4 高可用与弹性伸缩

> 当你的 vLLM 服务从"能跑"升级到"永不宕机"，高可用架构和智能伸缩就成了必修课。本节将带你构建生产级的高可用方案——从 Pod 防亲和到多区域灾备，从 HPA 弹性伸缩到蓝绿发布，确保你的 LLM 服务在故障面前依然稳如磐石。

## 10.4.1 故障分类与影响分析

在讨论高可用之前，我们需要先搞清楚**可能出什么问题**。vLLM 生产环境中的故障可以按影响范围分为五个等级：

| 故障类型 | 影响范围 | 典型原因 | 恢复时间 | 自动恢复？ |
|---------|---------|---------|---------|-----------|
| **Pod 崩溃** | 单个实例 | OOM、CUDA 错误、进程异常 | 秒级（重启） | ✅ K8s RestartPolicy |
| **OOM Kill** | 单个实例 | 请求量激增、内存泄漏 | 秒级（重启） | ✅ K8s + 资源限制 |
| **节点故障** | 1-N 个实例 | 硬件故障、内核崩溃、断电 | 分钟级（驱逐+重建） | ⚠️ 需配置 Node Controller |
| **网络分区** | 区域级别 | 交换机故障、DNS 异常 | 分钟~小时 | ❌ 需人工介入 |
| **数据中心 outage** | 全部服务 | 自然灾害、电力中断 | 小时~天 | ❌ 需多区域部署 |

### 故障场景模拟

比如下面的程序展示了一个完整的故障注入测试框架：

```python
"""
vLLM 高可用故障注入测试框架
用于验证系统在各种故障模式下的表现
"""

import asyncio
import aiohttp
import time
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class FaultType(Enum):
    POD_CRASH = "pod_crash"
    OOM_KILL = "oom_kill"
    NODE_FAILURE = "node_failure"
    NETWORK_PARTITION = "network_partition"
    HIGH_LATENCY = "high_latency"


@dataclass
class FaultScenario:
    fault_type: FaultType
    description: str
    inject_method: str  # kubectl / API / network
    expected_impact: str
    recovery_time_sla: int  # seconds


FAULT_SCENARIOS = [
    FaultScenario(
        fault_type=FaultType.POD_CRASH,
        description="随机删除一个 vLLM Pod",
        inject_method="kubectl delete pod",
        expected_impact="短暂 5xx，自动恢复 <30s",
        recovery_time_sla=30,
    ),
    FaultScenario(
        fault_type=FaultType.OOM_KILL,
        description="触发 OOM（发送超长请求）",
        inject_method="API stress test",
        expected_impact="单 Pod 重启，其他 Pod 承接流量",
        recovery_time_sla=60,
    ),
    FaultScenario(
        fault_type=FaultType.NODE_FAILURE,
        description="模拟节点不可达",
        inject_method="iptables DROP",
        expected_impact="该节点所有 Pod 迁移",
        recovery_time_sla=120,
    ),
    FaultScenario(
        fault_type=FaultType.NETWORK_PARTITION,
        description="切断一半 Pod 的网络",
        inject_method="NetworkPolicy deny",
        expected_impact="部分请求失败，需客户端重试",
        recovery_time_sla=300,
    ),
]


class HAValidator:
    """高可用性验证器"""

    def __init__(
        self,
        base_url: str = "http://vllm-service:8000",
        num_requests: int = 100,
        concurrency: int = 10,
    ):
        self.base_url = base_url
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.results: List[dict] = []

    async def send_request(
        self, session: aiohttp.ClientSession, request_id: int
    ) -> dict:
        """发送单个请求并记录指标"""
        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 50,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                latency = (time.time() - start_time) * 1000
                return {
                    "request_id": request_id,
                    "status": response.status,
                    "latency_ms": latency,
                    "success": 200 <= response.status < 300,
                    "error": None if response.status == 200 else await response.text(),
                }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                "request_id": request_id,
                "status": 0,
                "latency_ms": latency,
                "success": False,
                "error": str(e),
            }

    async def run_load_test(self) -> dict:
        """运行负载测试"""
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.send_request(session, i) for i in range(self.num_requests)
            ]
            self.results = await asyncio.gather(*tasks)

        return self._analyze_results()

    def _analyze_results(self) -> dict:
        """分析测试结果"""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]

        latencies = [r["latency_ms"] for r in successful]
        error_codes = {}

        for r in failed:
            code = r.get("status", "timeout")
            error_codes[code] = error_codes.get(code, 0) + 1

        return {
            "total": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) * 100,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p50_latency_ms": (
                statistics.median(latencies) if latencies else 0
            ),
            "p99_latency_ms": (
                sorted(latencies)[int(len(latencies) * 0.99)]
                if len(latencies) > 1
                else 0
            ),
            "error_distribution": error_codes,
        }


async def chaos_test_demo():
    """混沌工程演示"""
    print("=" * 60)
    print("🔥 vLLM 高可用混沌工程测试")
    print("=" * 60)

    validator = HAValidator(num_requests=200, concurrency=20)

    # Phase 1: 基线测试（无故障）
    print("\n[Phase 1] 基线测试（正常状态）")
    baseline = await validator.run_load_test()
    print(f"  成功率: {baseline['success_rate']:.1f}%")
    print(f"  平均延迟: {baseline['avg_latency_ms']:.1f}ms")
    print(f"  P99延迟: {baseline['p99_latency_ms']:.1f}ms")

    # Phase 2: 模拟 Pod Crash（实际执行时取消注释）
    # print("\n[Phase 2] 注入故障: Pod Crash")
    # os.system("kubectl delete pod -l app=vllm --random")
    # await asyncio.sleep(5)  # 等待故障生效
    # during_fault = await validator.run_load_test()
    # print(f"  成功率: {during_fault['success_rate']:.1f}%")
    # print(f"  恢复时间: {measure_recovery_time()}s")

    # Phase 3: 对比分析
    print("\n[Phase 3] SLA 合规检查")
    sla_targets = {
        "success_rate": 99.9,  # 99.9% 可用性
        "p99_latency_ms": 5000,  # P99 < 5秒
        "recovery_time_s": 30,   # 故障恢复 < 30秒
    }
    print(f"  目标成功率: ≥{sla_targets['success_rate']}%")
    print(f"  实际成功率: {baseline['success_rate']:.1f}%")
    print(f"  {'✅ PASS' if baseline['success_rate'] >= sla_targets['success_rate'] else '❌ FAIL'}")


if __name__ == "__main__":
    asyncio.run(chaos_test_demo())
```

运行这个程序，你会得到类似这样的输出：

```
============================================================
🔥 vLLM 高可用混沌工程测试
============================================================

[Phase 1] 基线测试（正常状态）
  成功率: 100.0%
  平均延迟: 245.3ms
  P99延迟: 892.1ms

[Phase 3] SLA 合规检查
  目标成功率: ≥99.9%
  实际成功率: 100.0%
  ✅ PASS
```

## 10.4.2 Kubernetes 高可用配置

### Pod 反亲和性：避免单点故障

最基础也最重要的一步：**确保多个 vLLM Pod 不在同一台机器上**。如果两个 Pod 都在 node-1 上，那 node-1 宕机就等于全挂。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-ha
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      # 关键配置 1: Pod 反亲和性
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values: [vllm]
                topologyKey: kubernetes.io/hostname
        # 可选: GPU 节点亲和性（只调度到有特定 GPU 的节点）
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values: ["NVIDIA-A100-SXM4-80GB"]
      # 关键配置 2: 拓扑分布约束（跨可用区分布）
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: vllm
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - --model=Qwen/Qwen2.5-7B-Instruct
            - --port=8000
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "32Gi"
            requests:
              nvidia.com/gpu: 1
              memory: "28Gi"
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120  # 模型加载需要时间
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 1
```

这段 YAML 的关键设计：

| 配置项 | 作用 | 为什么重要 |
|-------|------|-----------|
| `podAntiAffinity` | 尽量让不同 Pod 在不同节点 | 避免"一挂全挂" |
| `topologySpreadConstraints` | 跨可用区均匀分布 | 机房级容灾 |
| `livenessProbe` | 健康检查，不健康则重启 | 自动故障恢复 |
| `readinessProbe` | 就绪检查，未就绪不接收流量 | 防止请求打到半成品 |
| `initialDelaySeconds: 120` | 给模型加载留足时间 | 避免误判为 unhealthy |

### PodDisruptionBudget：保护最小可用副本

当你需要做节点维护或版本升级时，K8s 可能会同时驱赶多个 Pod。PDB 确保任何时候都有足够的副本在线：

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: vllm-pdb
spec:
  minAvailable: 2  # 至少保持 2 个 Pod 在线（总共 3 个副本时）
  # 或者使用百分比形式：
  # minAvailable: 65%  # 保留至少 65% 的 Pod
  selector:
    matchLabels:
      app: vllm
```

**PDB 工作原理示意**：

```
正常状态:  [Pod-1 ✓] [Pod-2 ✓] [Pod-3 ✓]  → 3/3 在线

维护节点 node-1:
           [Pod-1 ✗ 驱逐中] [Pod-2 ✓] [Pod-3 ✓]
           PDB 检查: minAvailable=2, 当前在线=2 ✅ 允许驱逐

再想驱逐 Pod-2?
           [Pod-1 ✗] [Pod-2 ? ] [Pod-3 ✓]
           PDB 检查: minAvailable=2, 当前在线=1 ❌ 拒绝！
```

### RollingUpdate：零停机滚动更新

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1          # 滚动更新时最多多启动 1 个 Pod
    maxUnavailable: 0     # 滚动更新时不允许有任何 Pod 不可用
```

**滚动更新流程**：

```
T=0s:   [v1.0-Pod1 ✓] [v1.0-Pod2 ✓] [v1.0-Pod3 ✓]
        ↓ 开始滚动更新到 v2.0

T=60s:  [v1.0-Pod1 ✓] [v1.0-Pod2 ✓] [v1.0-Pod3 ✓] [v2.0-Pod4 启动中...]
        ↑ maxSurge=1 允许临时多一个

T=180s: [v1.0-Pod1 ✓] [v1.0-Pod2 ✓] [v2.0-Pod3 ✓] [v2.0-Pod4 ✓]
        ↑ 新 Pod 就绪，开始替换旧 Pod

T=240s: [v1.0-Pod1 ✓] [v2.0-Pod2 ✓] [v2.0-Pod3 ✓] [v2.0-Pod4 ✓]
        ↑ maxUnavailable=0 保证始终有 3 个可用

T=300s: [v2.0-Pod1 ✓] [v2.0-Pod2 ✓] [v2.0-Pod3 ✓] [v2.0-Pod4 终止]
        ↑ 更新完成
```

## 10.4.3 HPA 弹性伸缩

### 基于 Custom Metrics 的智能伸缩

vLLM 的负载不能仅靠 CPU/内存来判断。真正的瓶颈是 **GPU 利用率** 和 **请求队列深度**。

首先安装 Prometheus Adapter 来暴露自定义指标：

```yaml
# prometheus-adapter-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-adapter-config
data:
  config.yaml: |
    rules:
      # GPU 利用率指标
      - seriesQuery: 'vllm_gpu_utilization{namespace!="",pod!=""}'
        resources:
          overrides:
            namespace: {resource: "namespace"}
            pod: {resource: "pod"}
        name:
          matches: "vllm_gpu_utilization"
          as: "gpu_utilization"
        metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'

      # 请求队列深度
      - seriesQuery: 'vllm_num_requests_waiting{namespace!="",pod!=""}'
        resources:
          overrides:
            namespace: {resource: "namespace"}
            pod: {resource: "pod"}
        name:
          matches: "vllm_num_requests_waiting"
          as: "queue_depth"
        metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

然后定义 HPA：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-ha
  minReplicas: 2       # 最少 2 个副本（保证高可用）
  maxReplicas: 10      # 最多扩展到 10 个
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # 稳定 60 秒后再扩容
      policies:
        - type: Percent
          value: 100                   # 每次最多翻倍
          periodSeconds: 60
        - type: Pods
          value: 2                     # 或每次最多增加 2 个
          periodSeconds: 60
      selectPolicy: Min
    scaleDown:
      stabilizationWindowSeconds: 300  # 缩容前等 5 分钟（防止抖动）
      policies:
        - type: Percent
          value: 10                    # 每次最多缩 10%
          periodSeconds: 60
  metrics:
    # 主要指标: GPU 利用率
    - type: Pods
      pods:
        metric:
          name: gpu_utilization
        target:
          type: AverageValue
          averageValue: "70"         # GPU 利用率超过 70% 时扩容
    # 辅助指标: 请求队列深度
    - type: Pods
      pods:
        metric:
          name: queue_depth
        target:
          type: AverageValue
          averageValue: "5"          # 每个 Pod 平均等待 >5 个请求时扩容
    # 保底指标: CPU 使用率
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### 伸缩行为调优参数说明

| 参数 | 推荐值 | 说明 |
|-----|--------|------|
| `scaleUp.stabilizationWindowSeconds` | 30-90 | 防止瞬时流量 spike 导致过度扩容 |
| `scaleDown.stabilizationWindowSeconds` | 180-300 | 避免频繁缩容导致的服务波动 |
| `minReplicas` | ≥2 | 高可用的最低要求 |
| `maxReplicas` | 取决于 GPU 资源池大小 | 不能超过集群中可用 GPU 数量 |
| `gpu_utilization target` | 60-80% | 太低浪费资源，太高响应慢 |

## 10.4.4 预测性伸缩与 KEDA

对于有明显流量模式的业务（如工作时间高峰、夜间低谷），可以使用 **KEDA (Kubernetes Event-driven Autoscaling)** 实现基于预测的伸缩：

```yaml
# keda-vllm-scaler.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-keda-scaler
spec:
  scaleTargetRef:
    name: vllm-ha
  minReplicaCount: 2
  maxReplicaCount: 10
  cooldownPeriod: 300  # 冷却期 5 分钟
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 60
        scaleDown:
          stabilizationWindowSeconds: 300
  triggers:
    # 触发器 1: Prometheus 指标
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: vllm_request_rate
        query: |
          sum(rate(vllm_request_success_total[2m]))
        threshold: "100"  # QPS > 100 时开始扩容
    # 触发器 2: Cron 定时（工作日早高峰提前预热）
    - type: cron
      metadata:
        timezone: Asia/Shanghai
        start: "0 8 * * 1-5"    # 工作日 8:00
        end: "0 22 * * 1-5"     # 工作日 22:00
        desiredReplicas: "6"    # 工作时间维持 6 个副本
    # 触发器 3: Cron 定时（夜间低峰）
    - type: cron
      metadata:
        timezone: Asia/Shanghai
        start: "0 23 * * *"     # 每天 23:00
        end: "0 7 * * *"       # 次日 7:00
        desiredReplicas: "2"    # 夜间缩到 2 个副本
```

**KEDA vs HPA 对比**：

| 特性 | HPA | KEDA |
|-----|-----|------|
| 触发方式 | 指标阈值 | 事件驱动（Prometheus/Cron/Kafka/Redis 等） |
| 定时伸缩 | ❌ 不支持 | ✅ Cron trigger |
| 外部事件 | ❌ 仅支持内置指标 | ✅ 支持任意数据源 |
| 复杂度 | 低（原生 K8s） | 中（需安装 KEDA） |
| 适用场景 | 一般 Web 服务 | 有明显流量模式的 LLM 服务 |

## 10.4.5 蓝绿部署与金丝雀发布

### 蓝绿部署脚本

```bash
#!/bin/bash
# deploy-blue-green.sh
# vLLM 蓝绿部署脚本 — 零停机切换

set -e

NAMESPACE="vllm-production"
BLUE_DEPLOY="vllm-blue"
GREEN_DEPLOY="vllm-green"
SERVICE="vllm-service"
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
NEW_IMAGE="${2:-vllm/vllm-openai:v0.6.6}"

echo "=========================================="
echo "🔄 vLLM 蓝绿部署开始"
echo "=========================================="

# Step 1: 判断当前活跃版本
CURRENT=$(kubectl get svc $SERVICE -n $NAMESPACE \
  -o jsonpath='{.spec.selector.version}' 2>/dev/null || echo "unknown")

if [ "$CURRENT" = "blue" ]; then
  ACTIVE=$BLUE_DEPLOY
  STANDBY=$GREEN_DEPLOY
  NEW_VERSION="green"
elif [ "$CURRENT" = "green" ]; then
  ACTIVE=$GREEN_DEPLOY
  STANDBY=$BLUE_DEPLOY
  NEW_VERSION="blue"
else
  # 首次部署，默认 blue 为活跃
  ACTIVE=$BLUE_DEPLOY
  STANDBY=$GREEN_DEPLOY
  NEW_VERSION="green"
fi

echo "当前活跃版本: $ACTIVE"
echo "待部署版本: $STANDBY ($NEW_VERSION)"

# Step 2: 更新待部署版本的镜像
echo ""
echo "[Step 2] 更新 $STANDBY 镜像 → $NEW_IMAGE"
kubectl set image deployment/$STANDBY \
  vllm=$NEW_IMAGE -n $NAMESPACE

kubectl set env deployment/$STANDBY \
  VLLM_MODEL=$MODEL \
  -n $NAMESPACE

# Step 3: 等待新版本就绪
echo ""
echo "[Step 3] 等待 $STANDBY 就绪..."
kubectl rollout status deployment/$STANDBY -n $NAMESPACE --timeout=600s

# Step 4: 健康检查
echo ""
echo "[Step 4] 执行健康检查..."
STANDBY_POD=$(kubectl get pods -n $NAMESPACE \
  -l app=vllm,version=$NEW_VERSION \
  -o jsonpath='{.items[0].metadata.name}')

# 等待 liveness probe 通过
for i in $(seq 1 30); do
  STATUS=$(kubectl exec $STANDBY_POD -n $NAMESPACE \
    -- curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")

  if [ "$STATUS" = "200" ]; then
    echo "✅ 健康检查通过 (尝试 #$i)"
    break
  fi

  echo "⏳ 等待就绪... (尝试 #$i, status=$STATUS)"
  sleep 10
done

# Step 5: 流量切换（更新 Service Selector）
echo ""
echo "[Step 5] 切换流量 → $NEW_VERSION"
kubectl patch svc $SERVICE -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"version\":\"$NEW_VERSION\"}}}"

echo ""
echo "[Step 6] 验证流量切换..."
sleep 5
curl -s http://vllm-service.$NAMESPACE.svc.cluster.local:8000/v1/models | jq .

# Step 6: （可选）关闭旧版本
if [ "${3:-}" = "--cleanup" ]; then
  echo ""
  echo "[Step 7] 清理旧版本 $ACTIVE..."
  kubectl scale deployment/$ACTIVE --replicas=0 -n $NAMESPACE
  echo "✅ 旧版本已缩容至 0"
fi

echo ""
echo "=========================================="
echo "✅ 蓝绿部署完成！当前活跃版本: $NEW_VERSION"
echo "=========================================="
```

### 金丝雀发布（灰度发布）

如果你不想一次性切全部流量，可以用 Istio 或 Nginx Ingress 做金丝雀发布：

```yaml
# canary-virtualservice.yaml (Istio)
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vllm-canary
spec:
  hosts:
    - vllm.example.com
  http:
    # 95% 流量走稳定版
    - route:
        - destination:
            host: vllm-service
            subset: stable
          weight: 95
        # 5% 流量走金丝雀版
        - destination:
            host: vllm-service
            subset: canary
          weight: 5
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: vllm-destination
spec:
  host: vllm-service
  subsets:
    - name: stable
      labels:
        version: blue
    - name: canary
      labels:
        version: green
```

**金丝雀发布渐进式流量调整计划**：

| 时间 | Canary 流量 | Stable 流量 | 动作 |
|-----|------------|------------|------|
| T+0h | 5% | 95% | 初步放行，观察错误率 |
| T+2h | 15% | 85% | 无异常，逐步放量 |
| T+6h | 30% | 70% | 持续观察 TTFT/TPOT |
| T+12h | 50% | 50% | 半开状态 |
| T+24h | 100% | 0% | 全量切换，下线旧版 |

## 10.4.6 多区域灾备架构

对于要求 **99.99% 以上可用性** 的核心业务，单区域部署是不够的。你需要跨区域的多活或主备架构：

### 架构总览

```
                        ┌─────────────┐
                        │   DNS/GSLB   │
                        │  (CloudFlare │
                        │   / Route53) │
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │   Region A     │ │   Region B     │ │   Region C     │
     │  (主区域)      │ │  (热备)        │ │  (冷备)        │
     │                │ │                │ │                │
     │ ┌────────────┐ │ │ ┌────────────┐ │ │ ┌────────────┐ │
     │ │ Ingress    │ │ │ │ Ingress    │ │ │ │ 待命...    │ │
     │ │ (Nginx)    │ │ │ │ (Nginx)    │ │ │ │            │ │
     │ └─────┬──────┘ │ │ └─────┬──────┘ │ │ └────────────┘ │
     │       │        │ │       │        │ │                │
     │ ┌─────▼──────┐ │ │ ┌─────▼──────┐ │ │                │
     │ │ vLLM ×3    │ │ │ │ vLLM ×2    │ │ │                │
     │ │ (TP×2)     │ │ │ │ (TP×2)     │ │ │                │
     │ └─────┬──────┘ │ │ └─────┬──────┘ │ │                │
     │       │        │ │       │        │ │                │
     │ ┌─────▼──────┐ │ │ ┌─────▼──────┐ │ │                │
     │ │ Model Store│◄┼─┼─►│Model Store│ │ │                │
     │ │ (NFS/PVC)  │ │ │ │(NFS/PVC)   │ │ │                │
     │ └────────────┘ │ │ └────────────┘ │ │                │
     └────────────────┘ └────────────────┘ └────────────────┘
              │                │
              └───────┬────────┘
                      ▼
              ┌──────────────┐
              │ Cross-Region │
              │ Replication  │
              │ (rsync/S3)   │
              └──────────────┘
```

### 主备切换自动化

```python
"""
vLLM 多区域故障转移控制器
监控主区域健康状态，自动切换 DNS 到备用区域
"""

import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class RegionStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class RegionEndpoint:
    name: str
    health_url: str
    service_url: str
    priority: int  # 1=primary, 2=secondary, 3=tertiary


REGIONS = [
    RegionEndpoint(
        name="us-east-1",
        health_url="https://vllm-us-east.example.com/health",
        service_url="https://vllm-us-east.example.com",
        priority=1,
    ),
    RegionEndpoint(
        name="ap-southeast-1",
        health_url="https://vllm-ap-southeast.example.com/health",
        service_url="https://vllm-ap-southeast.example.com",
        priority=2,
    ),
    RegionEndpoint(
        name="eu-west-1",
        health_url="https://vllm-eu-west.example.com/health",
        service_url="https://vllm-eu-west.example.com",
        priority=3,
    ),
]


class FailoverController:
    """故障转移控制器"""

    def __init__(
        self,
        check_interval: int = 10,
        failure_threshold: int = 3,
        recovery_threshold: int = 5,
        dns_provider: str = "cloudflare",  # cloudflare / route53 / alidns
    ):
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold  # 连续 N 次失败才判定为 DOWN
        self.recovery_threshold = recovery_threshold  # 连续 N 次成功才判定为 RECOVERED
        self.dns_provider = dns_provider
        self.region_status: dict[str, list[bool]] = {
            r.name: [] for r in REGIONS
        }
        self.active_region: Optional[str] = None
        self.logger = logging.getLogger("failover")

    async def check_region_health(self, region: RegionEndpoint) -> bool:
        """检查单个区域的健康状态"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    region.health_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def update_region_status(self):
        """更新所有区域的状态"""
        check_tasks = [
            self.check_region_health(r) for r in REGIONS
        ]
        results = await asyncio.gather(*check_tasks)

        for region, is_healthy in zip(REGIONS, results):
            history = self.region_status[region.name]
            history.append(is_healthy)

            # 只保留最近的 N 条记录
            if len(history) > max(self.failure_threshold, self.recovery_threshold):
                history.pop(0)

    def determine_active_region(self) -> Optional[RegionEndpoint]:
        """根据各区域状态决定活跃区域"""
        candidates = []

        for region in REGIONS:
            history = self.region_status[region.name]
            if not history:
                continue

            recent_failures = sum(1 for h in history[-self.failure_threshold:] if not h)
            recent_successes = sum(h for h in history[-self.recovery_threshold:] if h)

            if recent_failures >= self.failure_threshold:
                status = RegionStatus.DOWN
            elif recent_successes == self.recovery_threshold and len(history) >= self.recovery_threshold:
                status = RegionStatus.HEALTHY
            elif recent_failures > 0:
                status = RegionStatus.DEGRADED
            else:
                status = RegionStatus.HEALTHY

            self.logger.info(
                f"Region {region.name}: status={status.value}, "
                f"recent_history={history[-5:]}"
            )

            if status in (RegionStatus.HEALTHY, RegionStatus.DEGRADED):
                candidates.append((region.priority, region))

        if not candidates:
            self.logger.error("❌ 所有区域都不可用！")
            return None

        # 选择优先级最高的健康区域
        candidates.sort(key=lambda x: x[0])
        best_region = candidates[0][1]

        if self.active_region != best_region.name:
            self.logger.warning(
                f"🔄 故障转移: {self.active_region} → {best_region.name}"
            )
            self.active_region = best_region.name
            await self.switch_dns(best_region)

        return best_region

    async def switch_dns(self, region: RegionEndpoint):
        """切换 DNS 记录到目标区域"""
        self.logger.info(f"🌐 切换 DNS → {region.service_url}")

        if self.dns_provider == "cloudflare":
            await self._switch_cloudflare(region)
        elif self.dns_provider == "route53":
            await self._switch_route53(region)
        else:
            self.logger.warning(f"不支持的 DNS 提供商: {self.dns_provider}")

    async def _switch_cloudflare(self, region: RegionEndpoint):
        """Cloudflare DNS 切换（示例）"""
        # 实际实现需要 Cloudflare API Token
        zone_id = "your_zone_id"
        record_name = "vllm.example.com"

        async with aiohttp.ClientSession() as session:
            # 获取记录 ID
            async with session.get(
                f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records?"
                f"name={record_name}",
                headers={"Authorization": "Bearer YOUR_API_TOKEN"},
            ) as resp:
                data = await resp.json()
                record_id = data["result"][0]["id"]

            # 更新 DNS（CNAME 指向新的负载均衡器）
            new_content = region.service_url.replace("https://", "")
            async with session.put(
                f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records/{record_id}",
                headers={"Authorization": "Bearer YOUR_API_TOKEN"},
                json={
                    "type": "CNAME",
                    "name": record_name,
                    "content": new_content,
                    "proxied": True,
                },
            ) as resp:
                result = await resp.json()
                if result.get("success"):
                    self.logger.info(f"✅ DNS 已更新 → {new_content}")
                else:
                    self.logger.error(f"❌ DNS 更新失败: {result}")

    async def _switch_route53(self, region: RegionEndpoint):
        """Route53 DNS 切换（示例）"""
        import boto3

        client = boto3.client("route53")

        # 解析目标域名获取 IP
        # 实际生产中应该用加权路由策略
        client.change_resource_record_sets(
            HostedZoneId="your_hosted_zone_id",
            ChangeBatch={
                "Changes": [
                    {
                        "Action": "UPSERT",
                        "ResourceRecordSet": {
                            "Name": "vllm.example.com.",
                            "Type": "CNAME",
                            "TTL": 60,
                            "ResourceRecords": [{"Value": region.service_url.replace("https://", "")}],
                        },
                    }
                ]
            },
        )
        self.logger.info("✅ Route53 DNS 已更新")

    async def run(self):
        """运行故障转移控制循环"""
        self.logger.info("🛡️ 故障转移控制器启动")
        self.logger.info(f"   检查间隔: {self.check_interval}s")
        self.logger.info(f"   故障阈值: {self.failure_threshold} 次连续失败")
        self.logger.info(f"   恢复阈值: {self.recovery_threshold} 次连续成功")

        while True:
            try:
                await self.update_region_status()
                await self.determine_active_region()
            except Exception as e:
                self.logger.error(f"检查循环异常: {e}")

            await asyncio.sleep(self.check_interval)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    controller = FailoverController(
        check_interval=10,
        failure_threshold=3,
        recovery_threshold=5,
        dns_provider="cloudflare",
    )

    await controller.run()


if __name__ == "__main__":
    asyncio.run(main())
```

### RTO/RPO 目标设定

| 灾难级别 | RTO (恢复时间目标) | RPO (恢复点目标) | 方案 |
|---------|------------------|----------------|------|
| Pod 故障 | < 30s | 0 | K8s RestartPolicy + PDB |
| 节点故障 | < 2min | 0 | Pod 反亲和 + 自动迁移 |
| 可用区故障 | < 5min | 0 | 跨 AZ 部署 + GSLB |
| 区域故障 | < 10min | < 1min | 多区域热备 + DNS 切换 |
| 数据中心毁损 | < 1hour | < 5min | 跨地域冷备 + S3 同步 |

## 10.4.7 高可用 Checklist

上线前逐一确认：

- [ ] **Pod 反亲和性**: 不同 Pod 调度到不同节点
- [ ] **拓扑分布**: 跨可用区均匀分布（maxSkew ≤ 1）
- [ ] **PDB**: minAvailable ≥ ⌈replicas × 65%⌉
- [ ] **探针**: liveness + readiness，initialDelaySeconds ≥ 模型加载时间
- [ ] **HPA**: minReplicas ≥ 2，基于 GPU 利用率 + 队列深度
- [ ] **RollingUpdate**: maxUnavailable = 0，maxSurge = 1
- [ ] **资源限制**: memory limit 设置（防 OOM 影响节点）
- [ ] **优雅终止**: preStop hook + SIGTERM 处理（完成进行中的请求）
- [ ] **DNS TTL**: ≤ 60s（加快故障切换速度）
- [ ] **故障演练**: 每季度一次 Chaos Testing
- [ ] **告警覆盖**: 所有故障类型都有对应的 alert rule
- [ ] **Runbook**: 每种故障场景都有标准操作手册

---

**下一节预告**：完成了高可用架构后，最后一节 **10.5 性能调优终极指南** 将汇总所有优化手段——从 Speculative Decoding 到 CUDA Graphs，从 FlashAttention 到 NUMA 优化，帮你把 vLLM 的性能压榨到最后 1%。
