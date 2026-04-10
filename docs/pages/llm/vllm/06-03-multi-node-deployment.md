# 多节点分布式部署

> **白板时间**：你已经掌握了单机多卡（TP）和流水线并行（PP）。但现实是——很多公司没有一台 8 卡 H100 的机器。他们有的是一个集群，每台机器 2 张 A100。这时候你需要把 vLLM 的推理服务分布到多台机器上运行。这就是**多节点分布式部署**。它涉及网络配置、Ray 集群管理、模型文件同步、故障恢复等一系列工程问题。

## 一、为什么需要多节点？

### 1.1 典型场景

| 场景 | 为什么需要多节点 |
|------|----------------|
| **70B+ 模型服务** | 单机显存不够（即使 TP=8 也需要 8×80GB=640GB） |
| **超高吞吐需求** | 单机无法满足 QPS 要求（需要水平扩展） |
| **硬件异构集群** | 不同型号的 GPU 分布在不同节点上 |
| **高可用要求** | 避免单点故障 |
| **成本优化** | 使用多个较小的实例代替一个大实例 |

### 1.2 架构概览

```
                    ┌──────────────┐
                    │   Client     │
                    │ (API 调用)    │
                    └──────┬───────┘
                           │ HTTP
                    ┌──────▼───────┐
                    │  Load Balancer│
                    │  (Nginx/Envoy)│
                    └──┬───────┬───┘
                       │       │
              ┌────────▼─┐ ┌──▼────────┐
              │ Node 0   │ │  Node 1   │
              │ (Head)   │ │ (Worker)  │
              │          │ │           │
              │ GPU 0,1  │ │ GPU 2,3   │
              │ Stage 0  │ │ Stage 1   │
              │ TP rank  │ │ TP rank   │
              │  0,1     │ │  0,1      │
              └──────────┘ └───────────┘
                   Ray Cluster (自动管理)
```

## 二、Ray 集成模式（推荐）

### 2.1 环境准备

```bash
# ===== 所有节点执行 =====

# 1. 安装依赖（所有节点版本必须一致）
pip install "vllm[ray]"
pip install ray[default]

# 2. 验证安装
python -c "import ray; print(f'Ray {ray.__version__}')"
python -c "import vllm; print(f'vLLM {vllm.__version__}')"

# 3. 确认 GPU 可见
nvidia-smi
```

### 2.2 启动 Ray 集群

```bash
# ===== Head Node =====
ray start --head \
    --port=6379 \
    --dashboard-port=8265 \
    --dashboard-host="0.0.0.0" \
    --num-cpus=16 \
    --num-gpus=2

# 输出示例：
# 2024-xx-xx xx:xx:xx, INFO head_node.py:xxx -- 
# Ray runtime started.
# Next, run the following command at another terminal on this machine:
#   ray start --address='192.168.1.100:6379'
# 
# Then, run the following command on a different machine:
#   ray start --address='192.168.1.100:6379'

# ===== Worker Nodes =====（在每台 worker 上执行）
ray start --address="192.168.1.100:6379" \
    --num-cpus=16 \
    --num-gpus=2

# ===== 验证集群状态 =====
# 在任意节点执行:
ray status

# 预期输出:
# =========
# Node table
# =========
# # Resources
# ...
# Alive
# -----
# 2/2   (head node)
# 2/2   (worker node)
# --------
# 2 nodes: 4 CPUs, 4 GPUs
```

### 2.3 启动 vLLM 服务

```bash
# ===== 在 Head Node 上启动 vLLM =====
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --worker-use-ray \
    --ray-address auto \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--worker-use-ray` | 启用 Ray 作为分布式后端 |
| `--ray-address auto` | 自动检测 Ray head 地址 |
| `--tensor-parallel-size 2` | 每个 node 内部用 2 卡做 TP |
| `--pipeline-parallel-size 2` | 跨 2 个 node 做 PP |

### 2.4 完整部署脚本

比如下面的脚本自动化整个部署流程：

```bash
#!/bin/bash
# deploy_vllm_cluster.sh — vLLM 多节点集群部署脚本

set -e

# ========== 配置区 ==========
MODEL="Qwen/Qwen2.5-72B-Instruct"
TP_SIZE=2
PP_SIZE=2
PORT=8000
DTYPE="bfloat16"
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.90

# ========== 检查环境 ==========
echo "[1/5] 检查环境..."

if ! command -v ray &> /dev/null; then
    echo "❌ Ray 未安装"
    exit 1
fi

if ! python -c "import vllm" &> /dev/null; then
    echo "❌ vLLM 未安装"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  ✅ GPU 数量: ${GPU_COUNT}"
echo "  ✅ Ray 版本: $(ray --version)"
echo "  ✅ vLLM 版本: $(python -c 'import vllm; print(vllm.__version__)')"

# ========== 初始化/连接 Ray ==========
echo ""
echo "[2/5] 初始化 Ray 集群..."

if [ "$RAY_HEAD" != "" ]; then
    echo "  连接到已有集群: $RAY_HEAD"
    ray start --address="$RAY_HEAD" --num-gpus=$GPU_COUNT || true
else
    echo "  启动新的 Head 节点..."
    ray start --head --port=6379 --dashboard-host="0.0.0.0" --num-gpus=$GPU_COUNT
fi

echo "  ✅ Ray 集群状态:"
ray status | head -20

# ========== 启动 vLLM ==========
echo ""
echo "[3/5] 启动 vLLM 服务..."
echo "  模型: $MODEL"
echo "  并行: TP=$TP_SIZE, PP=$PP_SIZE"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE" \
    --worker-use-ray \
    --ray-address auto \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" &

VLLM_PID=$!
sleep 5

# ========== 健康检查 ==========
echo ""
echo "[4/5] 健康检查..."

for i in $(seq 1 12); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "  ✅ vLLM 服务已就绪 (http://localhost:$PORT)"
        break
    fi
    
    if [ $i -eq 12 ]; then
        echo "  ❌ 服务启动超时"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    
    echo "  ⏳ 等待服务启动... ($i/12)"
    sleep 5
done

# ========== 验证 API ==========
echo ""
echo "[5/5] 验证 API..."
RESPONSE=$(curl -s http://localhost:$PORT/v1/models | python -c "
import sys, json
data = json.load(sys.stdin)
print(f'模型: {[m[\"id\"] for m in data[\"data\"]]}')
")
echo "  $RESPONSE"

echo ""
echo "==========================================="
echo "🚀 vLLM 集群部署完成!"
echo "   API: http://$(hostname -I | awk '{print $1}'):$PORT"
echo "   Dashboard: http://$(hostname -I | awk '{print $1}'):8265"
echo "==========================================="

wait $VLLM_PID
```

## 三、手动分布式模式（无 Ray）

如果因为某些原因不能使用 Ray，vLLM 也支持手动模式：

### 3.1 手动模式架构

```
Node 0 (Head/Rank 0):
  CUDA_VISIBLE_DEVICES=0,1
  python -m vllm.entrypoints.openai.api_server ... --pp-partition 0
  
Node 1 (Worker/Rank 1):
  CUDA_VISIBLE_DEVICES=0,1  
  python -m vllm.entrypoints.openai.api_server ... --pp-partition 1
```

### 3.2 手动启动命令

```bash
# ===== Node 0 (Head) =====
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend multiprocess \
    --pp-partition 0 \
    --port 8000

# ===== Node 1 (Worker) =====
export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend multiprocess \
    --pp-partition 1 \
    --port 8001
```

## 四、网络与安全配置

### 4.1 防火墙规则

```bash
# vLLM + Ray 集群需要开放的端口:

# Ray 内部通信
sudo ufw allow 6379/tcp    # Ray 默认端口
sudo ufw allow 8265/tcp    # Ray Dashboard

# vLLM API
sudo ufw allow 8000/tcp    # vLLM API Server

# NCCL (GPU 通信)
sudo ufw allow 29500:29600/tcp  # NCCL 端口范围
export NCCL_SOCKET_IFNAME=eth0
export NCCL_PORT_RANGE=29500-29600
```

### 4.2 SSH 免密登录（多节点必需）

```bash
# 在 Head Node 上生成密钥（如果没有）
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# 复制公钥到所有 Worker 节点
ssh-copy-id user@worker-node-1
ssh-copy-id user@worker-node-2

# 验证免密登录
ssh user@worker-node-1 "hostname"
ssh user@worker-node-2 "hostname"
```

## 五、模型存储策略

### 5.1 方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **各节点独立下载** | 最简单 | 占用多倍磁盘空间 | 小模型 / 磁盘充裕 |
| **NFS 共享** | 一份存储即可 | NFS I/O 可能成为瓶颈 | 中等规模 |
| **对象存储 (S3/OSS)** | 弹性扩展 | 下载速度取决于带宽 | 云原生环境 |
| **分布式文件系统** | 高性能共享 | 配置复杂 | 大规模生产 |

### 5.2 推荐方案：NFS + 缓存

```bash
# ===== Head Node: 设置 NFS 共享 =====
sudo apt install nfs-kernel-server

# 创建共享目录
sudo mkdir -p /shared/models
sudo chown nobody:nogroup /shared/models

# 配置 exports
echo "/shared/models *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a
sudo systemctl restart nfs-kernel-server

# ===== Worker Nodes: 挂载 NFS =====
sudo apt install nfs-common
sudo mkdir -p /shared/models
sudo mount -t nfs <head-node-ip>:/shared/models /shared/models

# 验证
ls /shared/models/

# ===== 启动 vLLM 时指定模型路径 =====
python -m vllm.entrypoints.openai.api_server \
    --model /shared/models/Qwen2.5-72B-Instruct \
    ...
```

## 六、多节点 Checklist

### 6.1 部署前检查

```python
def preflight_check():
    """多节点部署前检查清单"""
    
    checks = [
        ("Python 版本一致", _check_python_version),
        ("vLLM 版本一致", _check_vllm_version),
        ("PyTorch 版本一致", _check_torch_version),
        ("CUDA 版本一致", _check_cuda_version),
        ("GPU 驱动一致", _check_driver_version),
        ("NCCL 版本一致", _check_nccl_version),
        ("网络连通性", _check_network),
        ("SSH 免密登录", _check_ssh),
        ("时间同步 (NTP)", _check_ntp),
        ("模型文件可访问", _check_model_access),
        ("防火墙端口开放", _check_firewall),
        ("磁盘空间充足", _check_disk_space),
    ]
    
    print("=" * 60)
    print("vLLM 多节点部署前检查")
    print("=" * 60)
    
    all_pass = True
    for name, check_fn in checks:
        try:
            result = check_fn()
            status = "✅" if result else "❌"
            print(f"  {status} {name}")
            if not result:
                all_pass = False
        except Exception as e:
            print(f"  ⚠️  {name}: {e}")
            all_pass = False
    
    print("\n" + "=" * 60)
    if all_pass:
        print("🎉 所有检查通过！可以开始部署。")
    else:
        print("⚠️  部分检查未通过，请先修复后再部署。")
    return all_pass


def _check_python_version():
    import sys
    return sys.version_info >= (3, 9)

def _check_vllm_version():
    import vllm
    return hasattr(vllm, '__version__')

def _check_network():
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except:
        return False

# ... 其他检查函数类似实现 ...

preflight_check()
```

### 6.2 运维监控脚本

```python
#!/usr/bin/env python3
"""vLLM 集群监控面板"""

import requests
import time
import json

class VllmClusterMonitor:
    """vLLM 多节点监控"""
    
    def __init__(self, nodes: list[str]):
        self.nodes = nodes
    
    def health_check_all(self) -> dict:
        """检查所有节点的健康状态"""
        results = {}
        
        for node in nodes:
            try:
                r = requests.get(f"http://{node}:8000/health", timeout=3)
                r_models = requests.get(f"http://{node}:8000/v1/models", timeout=3)
                
                results[node] = {
                    "status": "healthy" if r.status_code == 200 else "unhealthy",
                    "latency_ms": r.elapsed.total_seconds() * 1000,
                    "models": [m["id"] for m in r_models.json().get("data", [])],
                }
            except Exception as e:
                results[node] = {
                    "status": "unreachable",
                    "error": str(e),
                }
        
        return results
    
    def monitor_loop(self, interval: int = 30):
        """持续监控循环"""
        
        while True:
            results = self.health_check_all()
            
            print(f"\n{'='*60}")
            print(f"集群监控 | {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            for node, info in results.items():
                if info.get("status") == "healthy":
                    icon = "🟢"
                    detail = f"{info['latency_ms']:.0f}ms | {', '.join(info['models'][:2])}"
                elif info.get("status") == "unreachable":
                    icon = "🔴"
                    detail = info.get("error", "unknown")[:40]
                else:
                    icon = "🟡"
                    detail = info.get("status", "unknown")
                
                print(f"  {icon} {node:<20} {detail}")
            
            time.sleep(interval)


if __name__ == "__main__":
    nodes = [
        "192.168.1.100",
        "192.168.1.101",
    ]
    
    monitor = VllmClusterMonitor(nodes)
    monitor.monitor_loop(interval=15)
```

---

## 七、总结

本节覆盖了多节点分布式部署的全部工程实践：

| 主题 | 核心要点 |
|------|---------|
| **为什么多节点** | 显存不足 / 高吞吐 / 高可用 / 成本优化 |
| **推荐方式** | Ray 集群管理 (`--worker-use-ray`) |
| **核心参数** | `--tp` × `--pp` = 总 GPU 数；`--ray-address auto` |
| **模型存储** | NFS 共享为首选，各节点独立下载最简单 |
| **网络要求** | 开放 6379(Ray)/8000(vLLM)/29500-29600(NCCL) |
| **SSH 免密** | 多节点间必须配置 |
| **版本一致性** | Python/vLLM/PyTorch/CUDA/Driver 全部节点相同 |
| **健康检查** | 部署前 Checklist + 运行时 Monitor Loop |

**核心要点回顾**：

1. **Ray 是 vLLM 多节点的事实标准**——自动处理 worker 注册、心跳检测和故障恢复
2. **TP × PP = 总 GPU 数**——这是最基本的资源计算公式
3. **模型存储策略影响首次加载速度**——NFS 共享避免重复下载但可能有 I/O 瓶颈
4. **版本一致性是多节点最大的坑**——不同版本的库会导致莫名其妙的错误
5. **先跑 preflight checklist 再部署**——能节省大量排错时间

下一节我们将学习 **推理性能基准测试**——如何科学地测量和比较不同配置的性能。
