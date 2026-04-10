# 08-4 模型预加载与热管理

## 预加载：消灭"首次请求慢"问题

Ollama 的一个经典体验痛点是：**第一次向某个模型发请求时特别慢**（可能需要 5-30 秒），后续请求就快了。这是因为模型需要从磁盘加载到内存。对于生产环境来说，这个"冷启动"延迟是不可接受的。

### 预加载机制

```bash
# 方法一: ollama pull (仅下载，不一定加载到内存)
ollama pull qwen2.5:7b
# → 文件在 ~/.ollama/models/blobs/ 中，但可能不在内存里

# 方法二: ollama ps + ollama run --preload (查看和预加载)
ollama ps
# 如果模型不在列表中，说明它没在内存中

# 方法三: 启动时自动预加载关键模型
# 在 systemd 服务或启动脚本中:
for model in qwen2.5:7b nomic-embed-text; do
    ollama run "$model" "hi" &
done
wait
```

### 更优雅的预热方案

```python
#!/usr/bin/env python3
"""Ollama 模型预热器"""

import requests
import time
import threading
from typing import List


class OllamaWarmer:
    """启动时预热 Ollama 模型"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def warmup(self, models: List[str], verbose=True):
        """
        预加载指定模型到内存
        
        Args:
            models: 要预热的模型名称列表，如 ["qwen2.5:7b", "nomic-embed-text"]
        """
        
        print(f"\n🔥 开始预热 {len(models)} 个模型...\n")
        
        results = []
        
        for model in models:
            start = time.time()
            
            if verbose:
                print(f"  ⏳ 预热: {model}", end=" ", flush=True)
            
            try:
                # 发送一个最小请求来触发模型加载
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "options": {"num_predict": 1}
                    },
                    timeout=300  # 首次加载可能很慢
                )
                
                elapsed = time.time() - start
                
                if resp.status_code == 200:
                    data = resp.json()
                    load_time = data.get("total_duration", 0) / 1e9
                    
                    status = "✅"
                    detail = f"{elapsed:.1f}s"
                else:
                    status = "❌"
                    detail = f"HTTP {resp.status_code}: {resp.text[:50]}"
                
            except requests.Timeout:
                status = "⏱️"
                detail = f">300s 超时"
                elapsed = 300
            except Exception as e:
                status = "❌"
                detail = str(e)[:50]
                elapsed = time.time() - start
            
            results.append({
                "model": model,
                "status": status,
                "detail": detail,
                "time": round(elapsed, 1)
            })
            
            if verbose:
                print(f"{status} ({detail})")
        
        # 输出汇总
        success = sum(1 for r in results if r["status"] == "✅")
        
        print(f"\n{'='*50}")
        print(f"  完成! 成功: {success}/{len(results)}")
        
        total_time = sum(r["time"] for r in results)
        print(f"  总耗时: {total_time:.1f}s")
        
        return results
    
    def verify_warm_models(self):
        """验证哪些模型当前已加载"""
        
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=5)
            
            if resp.status_code != 200:
                print("⚠️ 无法获取模型状态")
                return []
            
            models = resp.json().get("models", [])
            
            print(f"\n📊 当前已加载的模型 ({len(models)} 个):")
            for m in models:
                name = m.get("name", "?")
                size = m.get("size", 0) / (1024**3)
                vm = m.get.get("size_vram", 0) / (1024**3)
                details = m.get("details", {})
                
                params = details.get("parameter_size", "?")
                family = details.get("family", "?")
                
                print(f"  📦 {name:<25s} "
                      f"参数:{params:<6s} "
                      f"大小:{size:>6.1f}G "
                      f"显存:{vm:>6.1f}G")
            
            return [m["name"] for m in models]
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            return []


def create_systemd_warmup_service():
    """生成 systemd 服务配置用于开机自动预热"""
    
    config = """\
[Unit]
Description=Ollama Model Warmer
After=network.target ollama.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/ollama-warmup.py
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
"""
    
    print("\n📋 systemd 服务配置:")
    print(config)


if __name__ == "__main__":
    warmer = OllamaWarmer()
    
    # 预热核心模型
    warmer.warmup([
        "qwen2.5:7b",
        "nomic-embed-text",
    ])
    
    # 验证
    loaded = warmer.verify_warm_models()
```

## 热管理：温度与散热

### MacBook 跑大模型的发热现实

这是每一个用笔记本跑 70B 模型的人都经历过的场景：

```
┌─────────────────────────────────────────────────────────────┐
│              MacBook 热管理生存指南                          │
│                                                             │
│  运行 70B 模型时的典型症状:                                │
│                                                             │
│  🔥 温度飙升:                                               │
│  ┌────────────────┐                                        │
│  │ CPU: 95-100°C   │ ← 接近热节流保护阈值               │
│  │ GPU: 90-95°C    │                                    │
│  │ 内存附近: 75°C  │                                    │
│  │ 表面: 熨得不能摸 │                                    │
│  └────────────────┘                                        │
│                                                             │
│  🌀 风扇全速运转:                                             │
│  ┌──────┐  ┌──────┐                                         │
│  │ 左风扇│  │右风扇│  ~6000 RPM (最大噪音)              │
│  │ 声音如 │  │如飞机│                                   │
│  │  起飞  │  │ 准备  │                                   │
│  └──────┘  └──────┘                                         │
│                                                             │
│  🐌 系统降频保护:                                           │
│  CPU 频率从 3.5GHz → 2.8GHz → 2.2GHz                   │
│  (性能下降 30-40%，但防止硬件损坏)                         │
│                                                             │
│  💡 解决方案:                                               │
│  1. 用小模型代替 (7B vs 70B)                               │
│  2. 散热底座 / 外接风扇                                     │
│  3. 降低 num_ctx 和 num_parallel                            │
│  4. 在空调房间使用                                          │
│  5. 使用外接显卡 eGPU                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 温度监控脚本

```python
#!/usr/bin/env python3
"""MacBook 温度监控与降频检测"""

import subprocess
import time
import os


class ThermalMonitor:
    """macOS 温度监控器"""
    
    def __init__(self):
        self.is_macos = os.uname().sysname == "Darwin"
    
    def get_temperatures(self):
        """获取各组件温度"""
        
        if not self.is_macos:
            return {"error": "Only macOS supported"}
        
        try:
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "smc"],
                capture_output=True, text=True, timeout=10
            )
            
            temps = {}
            output = result.stdout
            
            lines = output.split("\n")
            for line in lines:
                if "temperature" in line.lower() or "die" in line.lower():
                    key = line.split(":")[0].strip()
                    val = line.split(":")[-1].strip()
                    temps[key] = val
            
            return {
                "cpu": temps.get("CPU die temperature", "N/A"),
                "gpu": temps.get("GPU die temperature", "N/A"),
                "memory": temps.get("memory proximity temperature", "N/A"),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def monitor(self, interval_sec=10, duration_min=60):
        """持续监控并报警"""
        
        print(f"\n🌡️ 温度监控开始 (每{interval_sec}s检查一次, 共{duration_min}分钟)\n")
        
        iterations = (duration_min * 60) // interval_sec
        
        for i in range(iterations):
            temps = self.get_temperatures()
            
            cpu_t = temps.get("cpu", "?")
            gpu_t = temps.get("gpu", "N/A")
            
            # 解析温度值
            def parse_temp(t_str):
                if t_str == "N/A":
                    return None
                import re
                nums = re.findall(r'[\d.]+', t_str)
                return float(nums[0]) if nums else None
            
            cpu_val = parse_temp(cpu_t)
            gpu_val = parse_temp(gpu_t)
            
            # 状态判断
            if cpu_val and cpu_val > 95:
                emoji = "🔴"
                alert = "⚠️ 过热!"
            elif cpu_val and cpu_val > 85:
                emoji = "🟠"
                alert = ""
            elif cpu_val:
                emoji = "🟢"
                alert = ""
            else:
                emoji = "⚪"
                alert = ""
            
            bar_len = int(min(cpu_val or 0, 100) / 4)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            
            print(f"[{emoji}] CPU: {cpu_t or '?':>8s} | "
                  f"GPU: {gpu_t or 'N/A':>8s} | {bar}")
            
            if alert:
                print(f"      {alert}")
            
            time.sleep(interval_sec)


if __name__ == "__main__":
    monitor = ThermalMonitor()
    monitor.monitor(interval_sec=5, duration_min=5)
```

## 长时间运行的稳定性

### 内存泄漏检测

```python
#!/usr/bin/env python3
"""长时间运行稳定性监控"""

import requests
import time
import threading
import tracemalloc


class StabilityMonitor:
    """Ollama 长时间运行稳定性监控"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.stats = []
        self.running = True
    
    def _loop(self, interval_sec=60):
        """后台监控循环"""
        
        while self.running:
            snapshot = {}
            
            # 1. Ollama 进程状态
            try:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
                snapshot["api_status"] = resp.status_code
                snapshot["model_count"] = len(resp.json().get("models", []))
            except:
                snapshot["api_status"] = "ERROR"
            
            # 2. 内存占用
            import subprocess
            try:
                pid_result = subprocess.run(
                    ["pgrep", "-f", "ollama"],
                    capture_output=True, text=True, timeout=5
                )
                pids = [p.strip() for p in pid_result.stdout.strip().split('\n') if p.strip()]
                
                total_rss_mb = 0
                for pid in pids[:5]:  # 只看前几个进程
                    rss = subprocess.run(
                        ["ps", "-o", "rss=", "-p", pid],
                        capture_output=True, text=True, timeout=5
                    )
                    rss_kb = int(rss.stdout.strip().split('\n')[-1])
                    total_rss_mb += rss_kb // 1024
                
                snapshot["rss_mb"] = total_rss_mb
                snapshot["pid_count"] = len(pids)
            except:
                snapshot["rss_mb"] = "N/A"
            
            # 3. Python 进程内存 (如果本脚本是长期运行的)
            current, peak = tracemalloc.get_traced_memory()
            snapshot["python_mem_mb"] = current / (1024**2)
            
            snapshot["timestamp"] = time.strftime("%H:%M:%S")
            self.stats.append(snapshot)
            
            time.sleep(interval_sec)
    
    def start_monitoring(self):
        """启动后台监控"""
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()
    
    def stop_monitoring(self):
        self.running = False
    
    def get_report(self):
        """生成稳定性报告"""
        
        if not self.stats:
            return "无数据"
        
        first = self.stats[0]["timestamp"]
        last = self.stats[-1]["timestamp"]
        
        rss_values = [s.get("rss_mb", 0) for s in self.stats 
                     if isinstance(s.get("rss_mb"), (int, float))]
        
        report = f"""
╔═════════════════════════════════════════════════════════╗
║           Ollama 稳定性监控报告                              ║
╠══════════════════════════════════════════════════════════╣
║ 监控时段: {first} → {last}
║ 数据点数: {len(self.stats)}
║
║ 内存趋势:
║   初始 RSS: {rss_values[0] if rss_values else 'N/A':>8} MB
║   当前 RSS: {rss_values[-1] if rss_values else 'N/A':>8} MB
║   峰值 RSS: {max(rss_values) if rss_values else 'N/A':>8} MB
║   内存增长: {(rss_values[-1] - rss_values[0]) if len(rss_values)>1 else 0:>+8} MB
║
║ API 可用性:
║   成功请求: {sum(1 for s in self.stats if s.get('api_status')==200):>5d}/{len(self.stats)}
╚═════════════════════════════════════════════════════════╝
"""
        return report


if __name__ == "__main__":
    monitor = StabilityMonitor()
    monitor.start_monitoring()
    
    try:
        input("\n按 Enter 停止监控...")
    except KeyboardInterrupt:
        pass
    
    monitor.stop_monitoring()
    print(monitor.get_report())
```

## 本章小结

这一节讨论了运行时管理的三个重要话题：

1. **预加载（Warm-up）**可以消除首次请求的 5-30 秒冷启动延迟——通过启动时发送最小请求触发模型加载
2. **热管理是 MacBook 用户的核心痛点**——70B 模型会让 CPU 接近 100°C、风扇全速、系统降频；解决方案包括换小模型、散热底座、eGPU
3. **温度监控脚本**实时追踪 CPU/GPU/内存温度并在过热时告警
4. **长时间运行需要关注内存泄漏**——定期重启进程、监控系统资源消耗、设置日志轮转
5. **systemd `restart=always`** 配置确保 Ollama 进程崩溃后自动恢复

下一节我们将讨论量化深度指南和系统级优化。
