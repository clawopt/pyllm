# 高可用与扩展

## 白板导读

当你的 Ollama 服务从"个人玩具"升级为"团队基础设施"后，一个无法回避的问题摆在面前：**单点故障（Single Point of Failure）**。Ollama 默认以单进程模式运行，一旦进程崩溃、机器宕机或 GPU 过热降频，所有依赖它的下游服务（客服机器人、代码助手、文档问答系统）将全部中断。

本节将从三个维度解决这个问题的工程化方案：**水平扩展（Scale Out）**——通过多实例部署 + 负载均衡分散流量；**垂直扩展（Scale Up）**——硬件升级路线图与容量规划；**故障恢复（Failover）**——健康检查、自动重启、数据备份与灾难恢复策略。最后还会给出 SLA（服务等级协议）定义模板，帮助你与业务方明确"可用性承诺"的边界。

---

## 10.4.1 水平扩展：多实例 + 负载均衡

### 为什么 Ollama 需要负载均衡？

Ollama 的推理请求是 **CPU/GPU 密集型任务**——一个 7B 模型的完整对话可能占用 GPU 数十秒，期间该 GPU 无法处理其他请求（除非你用 `OLLAMA_NUM_PARALLEL` 开启有限并行）。这意味着：

```
单实例瓶颈：
  User A ──┐
  User B ──┼──▶ Ollama :11434 ──▶ GPU ──▶ 串行排队等待
  User C ──┘
```

解决方案是在前面加一层 **Load Balancer（负载均衡器）**，将请求分发到多个 Ollama 实例：

```
多实例 + LB：
  User A ──┐
  User B ──┼──▶ Nginx / HAProxy ──┬──▶ Ollama-A :11434 ──▶ GPU-A
  User C ──┘                    ├──▶ Ollama-B :11435 ──▶ GPU-B
                                  └──▶ Ollama-C :11436 ──▶ GPU-C
```

### Nginx 七层负载均衡配置

Nginx 是最常用的软件负载均衡器，支持多种调度算法：

```nginx
upstream ollama_backend {
    # 最少连接数算法：将新请求发给当前连接数最少的服务器
    # 这对长耗时推理请求特别重要——避免某个实例堆积过多请求
    least_conn;

    server 192.168.1.11:11434 weight=1 max_fails=3 fail_timeout=30s;
    server 192.168.1.12:11434 weight=1 max_fails=3 fail_timeout=30s;
    server 192.168.1.13:11434 weight=2 max_fails=3 fail_timeout=30s;
    # weight=2 表示这台机器性能更强，分到双倍流量

    # 预留一台备用节点，正常不参与调度
    server 192.168.1.14:11434 backup;
}

server {
    listen 80;
    server_name ollama.internal.example.com;

    # SSE 流式响应必须关闭缓冲
    proxy_buffering off;
    proxy_cache off;

    location / {
        proxy_pass http://ollama_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 推理请求超时时间设为 5 分钟
        proxy_read_timeout 300s;
        proxy_send_timeout 60s;
        proxy_connect_timeout 10s;

        # 支持 SSE 的关键头部
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        chunked_transfer_encoding on;
    }

    # 健康检查端点
    location /health {
        access_log off;
        return 200 'OK';
        add_header Content-Type text/plain;
    }
}
```

> **⚠️ 关键细节**：`proxy_buffering off;` 是 Ollama 负载均衡的**生命线**。如果开启缓冲，Nginx 会等后端完整响应后才返回给客户端——但 Ollama 的 `/api/chat` 和 `/api/generate` 使用 SSE 流式传输，永远不会"完成"，客户端会一直等到超时。关闭缓冲后 Nginx 即时转发每个 SSE 数据块。

### Sticky Sessions（会话亲和性）

对于**多轮对话场景**，同一个用户的连续请求最好路由到同一台 Ollama 实例。原因有两个：

1. **KV Cache 复用**：同一台机器上如果上一个请求的 System Prompt 相同，可以复用 KV Cache，大幅减少 Prompt Eval 时间
2. **上下文一致性**：不同实例之间没有共享状态，但同一实例的模型权重和运行状态是一致的

```nginx
upstream ollama_backend_sticky {
    least_conn;

    # 用 IP Hash 确保同一用户总是打到同一台机器
    ip_hash;

    server 192.168.1.11:11434;
    server 192.168.1.12:11434;
    server 192.168.1.13:11434;
}
```

> **IP Hash 的局限**：如果你的用户经过 CDN 或代理（所有请求的 `$remote_addr` 都一样），IP Hash 就退化为固定路由了。更可靠的方案是用 **Cookie-based sticky session**（需要 Nginx Plus 商业版）或在应用层实现 session affinity。

### HAProxy 四层负载均衡（更高性能）

如果你追求极致的转发性能（特别是在千兆内网环境下），HAProxy 比 Nginx 更适合做四层 TCP 转发：

```
global
    log /dev/log local0
    maxconn 4096

defaults
    mode tcp
    timeout connect 5000ms
    timeout client 300000ms
    timeout server 300000ms

frontend ollama_front
    bind *:11434
    default_backend ollama_servers

backend ollama_servers
    balance roundrobin
    option httpchk GET /
    server ollama-1 192.168.1.11:11434 check inter 5s fall 3 rise 2
    server ollama-2 192.168.1.12:11434 check inter 5s fall 3 rise 2
    server ollama-3 192.168.1.13:11434 check inter 5s fall 3 rise 2
```

### OllamaClusterManager：集群状态管理

下面是一个完整的 Python 工具类，用于管理多实例 Ollama 集群的状态同步、模型分发和健康监控：

```python
"""
Ollama Cluster Manager
管理多实例部署的 Ollama 集群：
  - 实例注册与健康发现
  - 模型同步分发
  - 负载均衡决策辅助
  - 集群状态看板
"""

import json
import time
import hashlib
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import requests


class NodeState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    DRAINING = "draining"


@dataclass
class ClusterNode:
    id: str
    host: str
    port: int = 11434
    state: NodeState = NodeState.HEALTHY
    weight: int = 1
    models: List[str] = field(default_factory=list)
    gpu_memory_total_mb: int = 0
    gpu_memory_used_mb: int = 0
    active_requests: int = 0
    last_heartbeat: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def gpu_utilization(self) -> float:
        if self.gpu_memory_total_mb > 0:
            return self.gpu_memory_used_mb / self.gpu_memory_total_mb
        return 0.0


@dataclass
class ClusterStatus:
    total_nodes: int = 0
    healthy_nodes: int = 0
    total_models: List[str] = field(default_factory=list)
    total_active_requests: int = 0
    timestamp: str = ""


class OllamaClusterManager:
    """Ollama 多实例集群管理器"""

    def __init__(self, heartbeat_interval: int = 10):
        self.nodes: Dict[str, ClusterNode] = {}
        self.heartbeat_interval = heartbeat_interval
        self._lock = threading.Lock()
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

    def register_node(self, host: str, port: int = 11434,
                      weight: int = 1) -> str:
        node_id = hashlib.md5(f"{host}:{port}".encode()).hexdigest()[:8]
        with self._lock:
            if node_id not in self.nodes:
                self.nodes[node_id] = ClusterNode(
                    id=node_id, host=host, port=port, weight=weight
                )
        return node_id

    def unregister_node(self, node_id: str):
        with self._lock:
            self.nodes.pop(node_id, None)

    def start_heartbeat(self):
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self):
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=15)

    def _heartbeat_loop(self):
        while self._running:
            self._check_all_nodes()
            time.sleep(self.heartbeat_interval)

    def _check_all_nodes(self):
        with self._lock:
            for node_id, node in list(self.nodes.items()):
                try:
                    start = time.time()
                    resp = requests.get(f"{node.url}/", timeout=5)
                    latency_ms = (time.time() - start) * 1000

                    if resp.status_code == 200:
                        tags_resp = requests.post(
                            f"{node.url}/api/tags", json={}, timeout=5
                        )
                        models = [m['name'] for m in tags_resp.json().get('models', [])]

                        node.state = NodeState.HEALTHY
                        node.models = models
                        node.last_heartbeat = datetime.now().isoformat()
                        node.metadata["latency_ms"] = latency_ms
                    else:
                        node.state = NodeState.DEGRADED
                except Exception:
                    node.state = NodeState.DOWN
                    node.active_requests = 0

    def get_status(self) -> ClusterStatus:
        with self._lock:
            all_models = set()
            for node in self.nodes.values():
                all_models.update(node.models)

            return ClusterStatus(
                total_nodes=len(self.nodes),
                healthy_nodes=sum(
                    1 for n in self.nodes.values()
                    if n.state == NodeState.HEALTHY
                ),
                total_models=list(all_models),
                total_active_requests=sum(n.active_requests for n in self.nodes.values()),
                timestamp=datetime.now().isoformat()
            )

    def select_node_for_request(self, model: str,
                                prefer_low_load: bool = True) -> Optional[ClusterNode]:
        """为请求选择最优节点"""
        candidates = []
        with self._lock:
            for node in self.nodes.values():
                if node.state not in (NodeState.HEALTHY, NodeState.DEGRADED):
                    continue
                if model and model not in node.models:
                    continue
                candidates.append(node)

        if not candidates:
            return None

        if prefer_low_load:
            candidates.sort(key=lambda n: (
                n.active_requests,
                n.gpu_utilization,
                n.metadata.get("latency_ms", 99999)
            ))
        else:
            import random
            weighted = []
            for n in candidates:
                weighted.extend([n] * n.weight)
            return random.choice(weighted)

        return candidates[0]

    def sync_model(self, model_name: str,
                   source_node_id: Optional[str] = None) -> dict:
        """将模型同步到所有健康节点"""
        results = {"synced": [], "failed": [], "skipped": []}

        with self._lock:
            for node_id, node in self.nodes.items():
                if node.state != NodeState.HEALTHY:
                    results["skipped"].append(f"{node_id} ({node.state.value})")
                    continue

                if model_name in node.models and node_id != source_node_id:
                    results["skipped"].append(f"{node_id} (already has)")
                    continue

                try:
                    pull_url = f"{node.url}/api/pull"
                    resp = requests.post(pull_url, json={
                        "name": model_name, "stream": False
                    }, timeout=300)

                    if resp.status_code == 200:
                        node.models.append(model_name)
                        results["synced"].append(node_id)
                    else:
                        results["failed"].append(f"{node_id}: {resp.status_code}")
                except Exception as e:
                    results["failed"].append(f"{node_id}: {e}")

        return results

    def get_dashboard_data(self) -> dict:
        status = self.get_status()
        nodes_info = []
        with self._lock:
            for node in self.nodes.values():
                nodes_info.append(asdict(node))

        return {
            "cluster_status": asdict(status),
            "nodes": nodes_info,
            "model_distribution": self._get_model_distribution()
        }

    def _get_model_distribution(self) -> Dict[str, List[str]]:
        dist = {}
        with self._lock:
            for node in self.nodes.values():
                for model in node.models:
                    if model not in dist:
                        dist[model] = []
                    dist[model].append(node.id)
        return dist

    def drain_node(self, node_id: str):
        """优雅下线节点：停止接收新请求，等待现有请求完成"""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].state = NodeState.DRAINING


if __name__ == "__main__":
    cluster = OllamaClusterManager()

    cluster.register_node("192.168.1.11", weight=1)
    cluster.register_node("192.168.1.12", weight=1)
    cluster.register_node("192.168.1.13", weight=2)

    cluster.start_heartbeat()

    time.sleep(3)

    print(json.dumps(cluster.get_dashboard_data(), indent=2, ensure_ascii=False))

    best = cluster.select_node_for_request("qwen2.5:7b")
    if best:
        print(f"\n推荐节点: {best.id} ({best.host}:{best.port})")

    cluster.stop_heartbeat()
```

---

## 10.4.2 垂直扩展：硬件升级路线图

### 从笔记本到服务器机柜的演进路径

水平扩展是"加更多机器"，垂直扩展是"让每台机器更强"。对于 LLM 推理而言，**内存带宽是第一瓶颈**（第八章已详细分析），所以垂直扩展的核心目标是提升内存/显存带宽：

| 阶段 | 配置 | 适用规模 | 月活用户 | 成本参考 |
|------|------|---------|---------|---------|
| **Tier 1 - 开发测试** | Mac M1/M2 16GB | 个人使用 | 1-5 | ¥0（已有设备） |
| **Tier 2 - 小团队** | Mac M2/M3 Max 64-96GB 或 RTX 4090 24GB | 小组内部 | 5-20 | ¥15,000-25,000 |
| **Tier 3 - 部门级** | RTX 4090×2 或 A5000 48GB | 单部门 | 20-100 | ¥50,000-100,000 |
| **Tier 4 - 企业级** | A100 80GB × 2-4 或 H100 | 全公司 | 100-1000+ | ¥200,000-800,000 |
| **Tier 5 - 数据中心** | H100/H200 集群 + InfiniBand | SaaS 产品 | 10000+ | 百万级 |

### 各阶段的关键决策点

#### Tier 1 → Tier 2：什么时候需要独立 GPU？

判断标准不是"用户数量"，而是**并发请求数 × 模型大小**：

```python
def need_dedicated_gpu(
    concurrent_users: int,
    avg_requests_per_hour: float,
    model_size_gb: float,
    current_device: str = "mac_m1_16gb"
) -> dict:
    """评估是否需要升级到独立 GPU"""
    device_specs = {
        "mac_m1_16gb": {"memory_bandwidth_gbps": 68.25, "unified_memory_gb": 16},
        "mac_m2_max_96gb": {"memory_bandwidth_gbps": 400, "unified_memory_gb": 96},
        "rtx_4090_24gb": {"memory_bandwidth_gbps": 1008, "vram_gb": 24},
        "a100_80gb": {"memory_bandwidth_gbps": 2039, "vram_gb": 80},
        "h100_80gb": {"memory_bandwidth_gbps": 3350, "vram_gb": 80},
    }

    current = device_specs[current_device]
    peak_concurrent = concurrent_users * (avg_requests_per_hour / 3600) * 30
    estimated_vram_needed = model_size_gb * 1.5 * min(peak_concurrent, 4)

    recommendations = []

    if estimated_vram_needed > current.get("unified_memory_gb",
                                             current.get("vram_gb", 0)) * 0.8:
        recommendations.append({
            "level": "critical",
            "message": f"预估峰值显存需求 {estimated_vram_needed:.1f}GB "
                      f"超过设备容量 {current.get('unified_memory_gb', current.get('vram_gb'))}GB"
        })

    qps_target = concurrent_users * avg_requests_per_hour / 3600
    if qps_target > 1 and current["memory_bandwidth_gbps"] < 400:
        recommendations.append({
            "level": "warning",
            "message": f"目标 QPS {qps_target:.1f} 需要"
                      f"≥400GB/s 内存带宽（当前 {current['memory_bandwidth_gbps']}GB/s）"
        })

    return {
        "current_device": current_device,
        "estimated_peak_vram_gb": round(estimated_vram_needed, 1),
        "target_qps": round(qps_target, 2),
        "recommendations": recommendations,
        "suggested_upgrade": "rtx_4090_24gb" if any(
            r["level"] == "critical" for r in recommendations
        ) else "keep_current"
    }


print(json.dumps(
    need_dedicated_gpu(concurrent_users=20, avg_requests_per_hour=120,
                       model_size_gb=4.5, current_device="mac_m1_16gb"),
    indent=2, ensure_ascii=False
))
```

### Apple Silicon 升级路径详解

对于 macOS 用户（这是 Ollama 最大量的用户群体），Apple Silicon 的统一内存架构有独特的优势：

| 芯片 | 统一内存 | 内存带宽 | 可跑最大模型 | 参考价格 |
|------|---------|---------|------------|---------|
| M1 | 16GB | 68 GB/s | ~7B Q4 | 已停产 |
| M1 Pro | 32GB | 200 GB/s | ~13B Q4 | 已停产 |
| M1 Max | 64GB | 400 GB/s | ~27B Q4 | 已停产 |
| M2 | 16/24GB | 100 GB/s | ~7B Q4 | ¥4,500 起 |
| M2 Pro | 16/32GB | 200 GB/s | ~13B Q4 | ¥15,000 起 |
| M2 Max | 96/128GB | 400 GB/s | ~70B Q4 | ¥20,000 起 |
| M3 | 24GB | 100 GB/s | ~7B Q4 | ¥4,500 起 |
| M3 Pro | 18/36GB | 150 GB/s | ~13B Q4 | ¥15,000 起 |
| M3 Max | 36/48/64/128GB | 300/400 GB/s | ~70B Q4 | ¥25,000 起 |

> **性价比之选**：M2 Max 96GB 版本是目前跑本地大模型的"甜点配置"——能流畅运行 70B Q4_K_M 模型（约 42GB 显存），内存带宽 400 GB/s 接近 RTX 4090 的水平，价格却只有 A100 的零头。唯一的缺点是不支持 CUDA 生态（但这不影响 Ollama，因为 Ollama 底层用的是 Metal 后端）。

---

## 10.4.3 故障恢复机制

### 故障分类与应对策略

生产环境的故障不会只有一种形态，我们需要针对不同类型的故障设计不同的恢复策略：

| 故障类型 | 检测方式 | 恢复策略 | RTO 目标 |
|---------|---------|---------|---------|
| **Ollama 进程崩溃** | 进程消失 | systemd/supervisor 自动重启 | < 30 秒 |
| **GPU 驱动异常** | nvidia-smi 无输出 | 重启驱动 / 切换备用节点 | < 2 分钟 |
| **磁盘空间满** | df 检测 | 清理旧模型缓存 / 扩容 | < 10 分钟 |
| **单机硬件故障** | 心跳丢失 | 流量切换到其他节点 | < 1 分钟 |
| **机房级故障** | 所有节点失联 | 切换到异地灾备 | < 5 分钟（需提前规划） |

### systemd 服务化管理（Linux）

在 Linux 上使用 systemd 管理 Ollama 进程是实现自动重启的标准做法：

```ini
# /etc/systemd/system/ollama.service
[Unit]
Description=Ollama LLM Service
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
ExecStartPre=/bin/bash -c 'docker pull ollama/ollama:latest'
ExecStart=/usr/bin/docker run --rm \
  --gpus all \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  --name ollama \
  ollama/ollama:latest
ExecStop=/usr/bin/docker stop ollama
Restart=always
RestartSec=5
StartLimitIntervalSec=300
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
```

关键参数解释：

- **`Restart=always`**：无论退出码是什么都自动重启
- **`RestartSec=5`**：重启前等待 5 秒（给端口释放留时间）
- **`StartLimitBurst=5`**：5 分钟内最多重启 5 次，超过则放弃（防止无限重启循环导致系统卡死）

启动和管理命令：

```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama     # 开机自启
sudo systemctl start ollama      # 启动服务
sudo systemctl status ollama     # 查看状态
journalctl -u ollama -f          # 实时查看日志
```

### macOS launchd 等效方案

macOS 上没有 systemd，但可以用 launchd 达到同样的效果：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.ollama</string>

    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/ollama</string>
        <string>serve</string>
    </array>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/var/log/ollama.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/ollama.stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>OLLAMA_HOST</key>
        <string>0.0.0.0</string>
        <key>OLLAMA_NUM_PARALLEL</key>
        <string>2</string>
        <key>OLLAMA_KEEP_ALIVE</key>
        <string>24h</string>
    </dict>
</dict>
</plist>
```

安装路径：`~/Library/LaunchAgents/com.user.ollama.plist`

```bash
launchctl load ~/Library/LaunchAgents/com.user.ollama.plist
launchctl list | grep ollama
```

### OllamaFailoverController：完整故障切换控制器

```python
"""
Ollama Failover Controller
实现主备切换、健康探测、流量迁移的完整故障恢复控制器。
"""

import os
import time
import json
import subprocess
import smtplib
from email.mime.text import MIMEText
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from datetime import datetime
from enum import Enum
import requests


class FailoverAction(Enum):
    NOOP = "noop"
    RESTART_SERVICE = "restart_service"
    SWITCH_TO_BACKUP = "switch_to_backup"
    SCALE_OUT = "scale_out"
    ALERT_ONLY = "alert_only"


@dataclass
class FailoverEvent:
    timestamp: str
    event_type: str
    node_id: str
    action_taken: FailoverAction
    details: str
    resolved: bool = False


class OllamaFailoverController:
    """
    Ollama 故障恢复控制器
    
    核心逻辑：
    1. 定期健康检查所有节点
    2. 发现故障 → 根据严重程度选择恢复动作
    3. 执行恢复动作并记录事件
    4. 通知相关人员
    """

    def __init__(
        self,
        primary_url: str = "http://localhost:11434",
        backup_urls: Optional[List[str]] = None,
        check_interval: int = 15,
        failure_threshold: int = 3,
        notification_email: Optional[str] = None
    ):
        self.primary_url = primary_url
        self.backup_urls = backup_urls or []
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self.notification_email = notification_email

        self.consecutive_failures = 0
        self.current_active = "primary"
        self.event_log: List[FailoverEvent] = []
        self._running = False
        self._recovery_callbacks: List[Callable] = []

    def on_failover(self, callback: Callable):
        """注册故障切换回调"""
        self._recovery_callbacks.append(callback)

    def start(self):
        self._running = True
        while self._running:
            try:
                self._check_cycle()
            except Exception as e:
                self._log_event("check_error", "_system",
                                FailoverAction.ALERT_ONLY, str(e))
            time.sleep(self.check_interval)

    def stop(self):
        self._running = False

    def _check_cycle(self):
        is_primary_healthy = self._health_check(self.primary_url)

        if is_primary_healthy:
            self.consecutive_failures = 0
            if self.current_active != "primary":
                self._failback_to_primary()
            return

        self.consecutive_failures += 1

        if self.consecutive_failures >= self.failure_threshold:
            action = self._decide_action()
            self._execute_failover(action)

    def _health_check(self, url: str) -> bool:
        try:
            resp = requests.get(url, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _decide_action(self) -> FailoverAction:
        if self.backup_urls:
            return FailoverAction.SWITCH_TO_BACKUP
        return FailoverAction.RESTART_SERVICE

    def _execute_failover(self, action: FailoverAction):
        now = datetime.now().isoformat()

        if action == FailoverAction.SWITCH_TO_BACKUP:
            backup = self._find_healthy_backup()
            if backup:
                self.current_active = backup
                self._log_event("failover", self.primary_url, action,
                              f"切换到备份: {backup}")
                for cb in self._recovery_callbacks:
                    try:
                        cb("switch", backup)
                    except Exception:
                        pass
            else:
                self._log_event("failover_failed", self.primary_url,
                              FailoverAction.ALERT_ONLY,
                              "无可用备份节点")

        elif action == FailoverAction.RESTART_SERVICE:
            restart_result = self._restart_ollama_service()
            self._log_event("service_restart", self.primary_url, action,
                          restart_result)
            self.consecutive_failures = 0

        self._send_notification(action)

    def _find_healthy_backup(self) -> Optional[str]:
        for url in self.backup_urls:
            if self._health_check(url):
                return url
        return None

    def _failback_to_primary(self):
        self.current_active = "primary"
        self._log_event("failback", self.primary_url,
                       FailoverAction.NOOP, "主节点已恢复，切回主节点")

    @staticmethod
    def _restart_ollama_service() -> str:
        platform = os.uname().sysname.lower()
        try:
            if platform == "darwin":
                subprocess.run(["launchctl", "kickstart",
                               "-k", "gui/$(id -u)/com.user.ollama"],
                              timeout=30)
                return "launchd restart triggered"
            elif platform == "linux":
                subprocess.run(["systemctl", "restart", "ollama"], timeout=30)
                return "systemd restart triggered"
            else:
                subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"],
                              timeout=15, shell=True)
                time.sleep(3)
                subprocess.Popen(["ollama", "serve"])
                return "Windows process restarted"
        except Exception as e:
            return f"restart failed: {e}"

    def _log_event(self, event_type: str, node_id: str,
                   action: FailoverAction, details: str):
        event = FailoverEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            node_id=node_id,
            action_taken=action,
            details=details
        )
        self.event_log.append(event)
        print(f"[FAILOVER] [{event_type}] {details}")

    def _send_notification(self, action: FailoverAction):
        if not self.notification_email or action == FailoverAction.NOOP:
            return

        recent_events = [e for e in self.event_log[-5:]
                        if not e.resolved]
        body = f"""Ollama Failover Alert

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Action: {action.value}
Active Node: {self.current_active}

Recent Events:
"""
        for e in recent_events:
            body += f"  - [{e.event_type}] {e.details}\n"

        try:
            msg = MIMEText(body)
            msg["Subject"] = f"[Ollama] {action.value} - {datetime.now()}"
            msg["From"] = "ollama-monitor@example.com"
            msg["To"] = self.notification_email

            with smtplib.SMTP("localhost", 25) as s:
                s.send_message(msg)
        except Exception:
            pass

    def get_report(self) -> dict:
        return {
            "status": "running" if self._running else "stopped",
            "active_node": self.current_active,
            "consecutive_failures": self.consecutive_failures,
            "total_events": len(self.event_log),
            "recent_events": [
                {"time": e.timestamp, "type": e.event_type,
                 "action": e.action_taken.value, "details": e.details}
                for e in self.event_log[-10:]
            ]
        }


if __name__ == "__main__":
    controller = OllamaFailoverController(
        primary_url="http://localhost:11434",
        backup_urls=["http://192.168.1.12:11434"],
        failure_threshold=2,
        check_interval=10
    )

    def on_switch_callback(action: str, target: str):
        print(f">>> 回调触发: {action} -> {target}")

    controller.on_failover(on_switch_callback)

    try:
        controller.start()
    except KeyboardInterrupt:
        controller.stop()
        print("\n" + json.dumps(controller.get_report(), indent=2))
```

### 数据备份策略

Ollama 的核心数据存储在 `~/.ollama/models/` 目录中，包含下载的模型文件（GGUF 格式）和 blob 数据。备份策略如下：

```bash
#!/bin/bash
# ollama-backup.sh — Ollama 模型数据备份脚本

OLLAMA_DIR="$HOME/.ollama"
BACKUP_DIR="/backup/ollama"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

echo "[${TIMESTAMP}] 开始备份..."

# 创建增量备份（rsync 只传输变化的部分）
rsync -avz --delete \
  "$OLLAMA_DIR/models/" \
  "$BACKUP_DIR/models_$TIMESTAMP/"

# 记录当前已安装的模型列表
curl -s http://localhost:11434/api/tags | \
  jq -r '.models[].name' > "$BACKUP_DIR/models_$TIMESTAMP/model_list.txt"

# 创建最新备份的软链接（方便恢复）
rm -f "$BACKUP_DIR/latest"
ln -s "$BACKUP_DIR/models_$TIMESTAMP" "$BACKUP_DIR/latest"

# 清理过期备份
find "$BACKUP_DIR" -maxdepth 1 -type d -name "models_*" \
  -mtime +$RETENTION_DAYS -exec rm -rf {} \;

# 备份大小统计
TOTAL_SIZE=$(du -sh "$BACKUP_DIR/models_$TIMESTAMP" | cut -f1)
MODEL_COUNT=$(wc -l < "$BACKUP_DIR/models_$TIMESTAMP/model_list.txt")

echo "[${TIMESTAMP}] 备份完成: ${TOTAL_SIZE}, ${MODEL_COUNT} 个模型"
```

设置定时任务（每天凌晨 3 点执行）：

```bash
# crontab -e
0 3 * * * /path/to/ollama-backup.sh >> /var/log/ollama-backup.log 2>&1
```

---

## 10.4.4 容量规划

### 容量规划公式

容量规划的核心问题是："未来 N 个月，随着用户增长，我需要多少算力？"

```
所需 GPU 数量 = ceil(
    (日活跃用户 × 人均日请求量 × 峰值系数)
    / (单个 GPU 日处理能力 × 利用率上限)
)
```

各参数说明：

| 参数 | 典型值 | 说明 |
|------|--------|------|
| **DAU（日活）** | 业务预测 | 来自产品侧的增长曲线 |
| **人均日请求量** | 5-50 次 | 取决于产品形态（聊天 vs 偶尔查询） |
| **峰值系数** | 3-5x | 峰值 QPS / 平均 QPS，通常早高峰和晚高峰明显 |
| **单 GPU 日处理能力** | ~50,000-200,000 请求 | 取决于模型大小和量化级别 |
| **利用率上限** | 70-80% | 留余量应对突发流量 |

### CapacityPlanner 工具类

```python
"""
Ollama Capacity Planner
基于增长预测计算硬件需求，生成采购计划。
"""

import math
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


@dataclass
class HardwareSpec:
    name: str
    gpu_model: str
    vram_gb: int
    memory_bandwidth_gbps: float
    price_cny: float
    power_watts: int
    daily_request_capacity: float = 0.0

    def __post_init__(self):
        self.daily_request_capacity = self.vram_gb * 2500


@dataclass
class CapacityPlan:
    month: int
    projected_dau: int
    projected_daily_requests: int
    required_gpus: int
    recommended_hardware: str
    estimated_cost_cny: float
    notes: str = ""


HARDWARE_CATALOG = [
    HardwareSpec("RTX 4090 Workstation", "RTX 4090", 24, 1008, 18000, 450),
    HardwareSpec("A5000 Server", "RTX A5000", 48, 768, 35000, 230),
    HardwareSpec("A100 80G", "A100 80GB", 80, 2039, 140000, 300),
    HardwareSpec("H100 80G", "H100 80GB", 80, 3350, 280000, 700),
]


class OllamaCapacityPlanner:
    """Ollama 容量规划器"""

    def __init__(
        self,
        current_dau: int = 100,
        requests_per_user_per_day: float = 10.0,
        peak_factor: float = 4.0,
        target_utilization: float = 0.75,
        monthly_growth_rate: float = 0.15
    ):
        self.current_dau = current_dau
        self.requests_per_user = requests_per_user_per_day
        self.peak_factor = peak_factor
        self.utilization = target_utilization
        self.growth_rate = monthly_growth_rate

    def calculate_gpu_needs(self, dau: int, hardware: HardwareSpec) -> float:
        daily_requests = dau * self.requests_per_user
        peak_qps = (daily_requests / 86400) * self.peak_factor
        gpu_capacity = hardware.daily_request_capacity * self.utilization
        return math.ceil(daily_requests / gpu_capacity)

    def plan_months_ahead(self, months: int = 12) -> List[CapacityPlan]:
        plans = []
        base_date = datetime.now()

        for month_offset in range(1, months + 1):
            future_dau = int(
                self.current_dau * ((1 + self.growth_rate) ** month_offset)
            )
            daily_reqs = future_dau * self.requests_per_user

            best_hw = HARDWARE_CATALOG[0]
            gpus_needed = self.calculate_gpu_needs(future_dau, best_hw)

            for hw in HARDWARE_CATALOG[1:]:
                needed = self.calculate_gpu_needs(future_dau, hw)
                total_cost = needed * hw.price_cny
                best_cost = gpus_needed * best_hw.price_cny

                if needed <= 4 and total_cost < best_cost * 1.3:
                    best_hw = hw
                    gpus_needed = needed

            plan_date = base_date + relativedelta(months=month_offset)
            plans.append(CapacityPlan(
                month=month_offset,
                projected_dau=future_dau,
                projected_daily_requests=int(daily_reqs),
                required_gpus=gpus_needed,
                recommended_hardware=f"{best_hw.name} x{gpus_needed}",
                estimated_cost_cny=gpus_needed * best_hw.price_cny,
                notes=self._generate_notes(month_offset, future_dau, gpus_needed)
            ))

        return plans

    def _generate_notes(self, month: int, dau: int, gpus: int) -> str:
        notes = []
        if month == 3 and gpus > 1:
            notes.append("建议开始引入负载均衡")
        if month == 6 and dau > 500:
            notes.append("考虑引入模型缓存层降低重复计算")
        if gpus >= 4:
            notes.append("建议采用 GPU 集群方案")
        if gpus >= 8:
            notes.append("建议评估云 GPU 弹性伸缩方案")
        return "; ".join(notes) if notes else ""

    def get_summary(self, months: int = 12) -> dict:
        plans = self.plan_months_ahead(months)
        total_investment = sum(p.estimated_cost_cny for p in plans[-3:])
        max_gpus = max(p.required_gpus for p in plans)

        return {
            "planning_horizon_months": months,
            "starting_dau": self.current_dau,
            "projected_final_dau": plans[-1].projected_dau,
            "growth_rate": f"{self.growth_rate * 100:.0f}%/月",
            "max_gpu_requirement": max_gpus,
            "total_investment_last_3months_cny": f"¥{total_investment:,.0f}",
            "monthly_plan": [asdict(p) for p in plans]
        }


if __name__ == "__main__":
    planner = OllamaCapacityPlanner(
        current_dau=100,
        requests_per_user_per_day=10,
        peak_factor=4.0,
        monthly_growth_rate=0.15
    )

    summary = planner.get_summary(12)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
```

输出示例（截取关键月份）：

```json
{
  "projected_final_dau": 535,
  "max_gpu_requirement": 4,
  "monthly_plan": [
    {"month": 1, "projected_dau": 115, "required_gpus": 1, ...},
    {"month": 3, "projected_dau": 152, "required_gpus": 1,
     "notes": "建议开始引入负载均衡"},
    {"month": 6, "projected_dau": 231, "required_gpus": 2,
     "notes": "考虑引入模型缓存层降低重复计算"},
    {"month": 9, "projected_dau": 352, "required_gpus": 3,
     "notes": "建议采用 GPU 集群方案"},
    {"month": 12, "projected_dau": 535, "required_gpus": 4,
     "notes": "建议评估云 GPU 弹性伸缩方案"}
  ]
}
```

---

## 10.4.5 SLA 定义与服务等级协议模板

### 什么是 SLA？

SLA（Service Level Agreement，服务等级协议）是你与业务方之间的**契约**，明确约定了服务的质量标准和违约后果。没有 SLA 的服务就像没有合同的合作——出了问题互相推诿。

### Ollama 服务 SLA 关键指标

| 指标 | 定义 | 企业级目标 | 内部工具目标 |
|------|------|-----------|-------------|
| **可用性（Availability）** | 服务可正常响应的时间比例 | 99.9%（月宕机 ≤43 分钟） | 99%（月宕机 ≤7.3 小时） |
| **RTO（Recovery Time Objective）** | 故障发生后恢复服务的目标时间 | < 5 分钟 | < 30 分钟 |
| **RPO（Recovery Point Objective）** | 可接受的数据丢失量 | 0（实时同步） | < 1 小时 |
| **P50 延迟** | 50% 请求的响应时间 | < 2 秒 | < 5 秒 |
| **P99 延迟** | 99% 请求的响应时间 | < 15 秒 | < 30 秒 |
| **错误率** | HTTP 5xx 错误占比 | < 0.1% | < 1% |

### SLA 文档模板

```markdown
# Ollama LLM 推理服务等级协议 (SLA)

## 1. 服务概述
- **服务名称**: Ollama LLM Inference API
- **服务范围**: 提供基于本地部署的大语言模型推理能力
- **支持模型**: qwen2.5:7b, llama3.1:8b, codestral:22b 等
- **服务窗口**: 7×24 小时（计划维护除外）

## 2. 服务等级指标

### 2.1 可用性
| 等级 | 月度可用性目标 | 允许月度停机时间 |
|------|--------------|-----------------|
| Premium | 99.9% | ≤ 43.2 分钟 |
| Standard | 99.5% | ≤ 3.65 小时 |
| Basic | 99% | ≤ 7.31 小时 |

### 2.2 性能指标
| 指标 | Premium | Standard | Basic |
|------|---------|----------|-------|
| P50 延迟 (TTFT) | ≤ 1s | ≤ 3s | ≤ 5s |
| P99 延迟 | ≤ 10s | ≤ 20s | ≤ 30s |
| Token 生成速度 | ≥ 30 tok/s | ≥ 15 tok/s | ≥ 8 tok/s |

### 2.3 故障恢复
| 场景 | RTO 目标 | 说明 |
|------|---------|------|
| 单实例崩溃 | ≤ 30s | 自动重启 |
| 单机故障 | ≤ 2min | 切换至备用节点 |
| 机房故障 | ≤ 10min | 切换至异地灾备 |

## 3. 维护窗口
- **定期维护**: 每周日 02:00-04:00 (UTC+8)
- **紧急维护**: 提前 4 小时通知
- **维护期间不计入可用性考核**

## 4. 违约赔偿
| 违约程度 | 赔偿方式 |
|---------|---------|
| 可用性低于 99% | 当月服务费减免 10% |
| 可用性低于 95% | 当月服务费减免 30% |
| 连续两个月低于 95% | 免除当月费用 + 免费架构优化咨询 |

## 5. 联系方式
- **技术支持**: ollama-team@example.com
- **紧急热线**: +86-xxx-xxxx-xxxx
- **状态页**: https://status.example.com/ollama
```

### SLATracker：SLA 合规追踪器

```python
"""
Ollama SLA Tracker
追踪实际服务水平是否满足 SLA 承诺，
生成月度合规报告。
"""

import json
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class IncidentSeverity(Enum):
    P1_PARTIAL = "partial_outage"
    P2_DEGRADED = "degraded_performance"
    P3_MINOR = "minor_issue"


@dataclass
class AvailabilityRecord:
    timestamp: str
    available: bool
    response_time_ms: float = 0.0
    error: str = ""


@dataclass
class Incident:
    id: str
    start_time: str
    end_time: Optional[str] = None
    severity: IncidentSeverity = IncidentSeverity.P2_DEGRADED
    description: str = ""
    root_cause: str = ""
    resolution: str = ""


class OllamaSLATracker:
    """SLA 合规追踪器"""

    def __init__(self, sla_target: float = 0.999):
        self.sla_target = sla_target
        self.records: List[AvailabilityRecord] = []
        self.incidents: List[Incident] = []
        self.monthly_reports: Dict[str, dict] = {}

    def record_check(self, available: bool,
                     response_time_ms: float = 0.0,
                     error: str = ""):
        self.records.append(AvailabilityRecord(
            timestamp=datetime.now().isoformat(),
            available=available,
            response_time_ms=response_time_ms,
            error=error
        ))

    def report_incident(
        self,
        severity: IncidentSeverity,
        description: str,
        root_cause: str = ""
    ) -> str:
        incident_id = f"INC-{len(self.incidents)+1:04d}"
        incident = Incident(
            id=incident_id,
            start_time=datetime.now().isoformat(),
            severity=severity,
            description=description,
            root_cause=root_cause
        )
        self.incidents.append(incident)
        return incident_id

    def resolve_incident(self, incident_id: str, resolution: str):
        for inc in self.incidents:
            if inc.id == incident_id:
                inc.end_time = datetime.now().isoformat()
                inc.resolution = resolution
                break

    def calculate_availability(
        self, start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> dict:
        if not self.records:
            return {"availability": 1.0, "total_checks": 0, "downtime_minutes": 0}

        filtered = self.records
        if start or end:
            filtered = [
                r for r in self.records
                if (not start or datetime.fromisoformat(r.timestamp) >= start)
                and (not end or datetime.fromisoformat(r.timestamp) <= end)
            ]

        total = len(filtered)
        if total == 0:
            return {"availability": 1.0, "total_checks": 0, "downtime_minutes": 0}

        available_count = sum(1 for r in filtered if r.available)
        availability = available_count / total

        downtime_checks = total - available_count
        assumed_interval_sec = 60
        downtime_minutes = (downtime_checks * assumed_interval_sec) / 60

        latencies = [r.response_time_ms for r in filtered
                     if r.response_time_ms > 0]
        p50 = sorted(latencies)[len(latencies)//2] if latencies else 0
        p99 = sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) > 1 else (latencies[0] if latencies else 0)

        return {
            "availability": round(availability, 6),
            "sla_target": self.sla_target,
            "met_sla": availability >= self.sla_target,
            "total_checks": total,
            "successful_checks": available_count,
            "failed_checks": downtime_checks,
            "downtime_minutes": round(downtime_minutes, 2),
            "allowed_downtime_minutes": round(
                (1 - self.sla_target) * 43200, 2
            ),
            "p50_latency_ms": round(p50, 1),
            "p99_latency_ms": round(p99, 1),
            "incidents_count": len([
                i for i in self.incidents
                if i.start_time >= (start or datetime.min).isoformat()
            ])
        }

    def generate_monthly_report(self, year: int, month: int) -> dict:
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)

        availability_data = self.calculate_availability(start, end)
        month_incidents = [
            i for i in self.incidents
            if start <= datetime.fromisoformat(i.start_time) < end
        ]

        report = {
            "period": f"{year}-{month:02d}",
            "availability": availability_data,
            "incidents": [
                {
                    "id": i.id,
                    "severity": i.severity.value,
                    "description": i.description,
                    "duration_hours": round(
                        (datetime.fromisoformat(i.end_time or i.start_time) -
                         datetime.fromisoformat(i.start_time)).total_seconds() / 3600,
                        2
                    ) if i.end_time else None,
                    "root_cause": i.root_cause,
                    "resolution": i.resolution
                }
                for i in month_incidents
            ],
            "compensation_eligible": not availability_data["met_sla"],
            "compensation_tier": self._calculate_compensation(
                availability_data["availability"]
            )
        }

        self.monthly_reports[f"{year}-{month:02d}"] = report
        return report

    @staticmethod
    def _calculate_compensation(availability: float) -> str:
        if availability >= 0.999:
            return "none"
        elif availability >= 0.99:
            return "tier_1_10_percent"
        elif availability >= 0.95:
            return "tier_2_30_percent"
        else:
            return "tier_3_full_refund_plus_consulting"


if __name__ == "__main__":
    tracker = OllamaSLATracker(sla_target=0.999)

    for i in range(10000):
        available = random.random() > 0.001
        rt = random.uniform(200, 3000) if available else 0
        tracker.record_check(available, rt)

    tracker.report_incident(
        IncidentSeverity.P1_PARTIAL,
        "GPU 驱动崩溃导致服务不可用",
        "NVIDIA 驱动版本 550.90 与内核不兼容"
    )

    report = tracker.generate_monthly_report(2026, 1)
    print(json.dumps(report, indent=2, ensure_ascii=False))
```

---

## 要点回顾

| 维度 | 关键要点 |
|------|---------|
| **水平扩展** | Nginx/HAProxy 做七层/四层负载均衡，`proxy_buffering off` 对 SSE 流式响应至关重要 |
| **Sticky Session** | 多轮对话场景用 `ip_hash` 保证同一用户路由到同一实例（复用 KV Cache） |
| **集群管理** | OllamaClusterManager 实现节点注册、心跳检测、模型同步、智能选节点 |
| **垂直扩展** | 内存带宽是第一瓶颈；M2 Max 96GB 是 macOS 甜点配置；企业级选 A100/H100 |
| **故障恢复** | systemd（Linux）/ launchd（macOS）实现进程守护 + 自动重启 |
| **Failover** | 主备切换三步曲：健康检测失败累积 → 超过阈值触发切换 → 执行恢复动作 + 通知 |
| **数据备份** | rsync 增量备份 `~/.ollama/models/`，配合 crontab 定时执行，保留 7 天历史 |
| **容量规划** | 公式：`GPU数 = ceil(DAU × 人均请求 × 峰值系数 / (单GPU能力 × 利用率))` |
| **SLA** | 99.9% 可用性意味着每月允许停机 ≤43 分钟；明确 RTO/RPO/P50/P99 指标和违约赔偿 |

> **一句话总结**：高可用的本质不是"永不故障"，而是**故障发生时能在承诺的时间内自动恢复**，并且通过 SLA 与业务方建立合理的预期管理。从单实例 systemd 守护 → 多实例 Nginx 负载均衡 → FailoverController 主备切换 → SLATracker 合规追踪，这四层防护构成了 Ollama 生产环境的高可用保障体系。
