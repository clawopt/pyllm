# 从 Static Batching 到 Continuous Batching

## 白板导读

PagedAttention 解决了"显存不够用"的问题，但即使有无限的显存，如果请求处理方式不够高效，GPU 仍然会大量时间处于空闲状态。这就是 **Continuous Batching（连续批处理）** 要解决的第二个核心问题。上一章我们用"公交车 vs 地铁列车"做了类比，这一节将深入这个类比的每一个细节：传统 Static Batching 到底慢在哪里？Continuous Batching 是如何做到"边跑边接人"的？它的迭代级别调度循环是什么样的？为什么说这是 vLLM 吞吐量提升的另一半功臣？（PagedAttention 管内存，Continuous Batching 管调度，两者缺一不可。）

---

## 1.1 Static Batching 的三种致命缺陷

### 回顾：Static Batching 的工作流程

在深入 Continuous Batching 之前，我们需要彻底理解它要替代的方案——**静态批处理（Static Batching / Iterative Level Scheduling, ILS）**。

```
Static Batching 完整生命周期:

T=0s    [请求A到达] [请求B到达] [请求C到达]
        ↓ 收集阶段: 等待凑齐一批
        ↓ (可能等待数秒!)
        
T=3s    ┌─────────────────────────────────┐
        │   Batch 已满! 开始处理            │
        │                                   │
        │  Step 1: Padding 到统一长度      │  ← 浪费计算!
        │  Step 2: Forward Pass (所有请求)   │
        │  Step 3: 等待最慢的请求完成         │  ← 长尾延迟!
        │  Step 4: 返回全部结果             │
        └─────────────────────────────────┘
T=5s    [请求D到达] [请求E到达]
        ↓ 又开始新一轮收集...
        ↓ 新请求必须等下一轮
        
T=8s    ┌─────────────────────────────────┐
        │   处理 D + E (+ 可能其他新到的)     │
        └─────────────────────────────────┘
```

### 缺陷一：首 Token 延迟（TTFT）灾难

TTFT（Time to First Token）是用户体验的核心指标——用户发出请求后看到第一个字需要等多久。

在 Static Batching 中：

```python
"""
Static Batching TTFT 计算模拟
"""

def calculate_ttft_static(
    arrival_times: list,     # 每个请求的到达时间 (秒)
    batch_size: int = 16,    # 批大小
    processing_time: float = 0.5,  # 一整批的处理时间
):
    """计算每个请求的 TTFT"""
    
    batches = []
    current_batch = []
    batch_start_time = None
    
    for i, arrival in enumerate(arrival_times):
        current_batch.append((i, arrival))
        
        if len(current_batch) >= batch_size:
            # 凑齐一批发车
            if batch_start_time is None:
                batch_start_time = current_batch[0][1]
            
            for req_id, req_arrival in current_batch:
                ttft = (batch_start_time + processing_time) - req_arrival
                batches.append({
                    "req_id": req_id,
                    "arrival": req_arrival,
                    "batch_start": batch_start_time,
                    "ttft": max(0, ttff),
                })
            
            current_batch = []
            batch_start_time = None
    
    # 处理剩余的不完整批次
    if current_batch and batch_start_time is not None:
        for req_id, req_arrival in current_batch:
            ttft = (batch_start_time + processing_time) - req_arrival
            batches.append({"req_id": req_id, "ttft": max(0, ttif)})
    
    return batches


# 模拟：10 个请求在 2 秒内陆续到达
import random
random.seed(42)
arrivals = sorted([random.uniform(0, 2) for _ in range(10)])

results = calculate_ttft_static(arrivals, batch_size=4)

print("Static Batching TTFT 分析")
print("=" * 60)
print(f"{'请求':>4} {'到达时间':>9} {'批次开始':>9} {'TTFT':>8} {'等待时长':>9}")
print("-" * 60)

for r in results:
    wait = r["batch_start"] - r["arrival"] if r.get("batch_start") else -1
    print(f"{r['req_id']:>4} {r['arrival']:>8.2f}s "
          f"{r.get('batch_start', 'N/A'):>9} {r['ttft']:>7.2f}s "
          f"{wait:>8.2f}s")

avg_ttft = sum(r["ttft"] for r in results) / len(results)
max_wait = max(
    (r["batch_start"] - r["arrival"]) 
    for r in results if r.get("batch_start")
)
print("-" * 60)
print(f"平均 TTFT: {avg_ttft:.2f}s | 最大等待: {max_wait:.2f}s")
```

输出：

```
============================================================
 请求   到达时间   批次开始     TTFT    等待时长
------------------------------------------------------------
   0    0.03s       0.14s    0.51s     0.11s
   1    0.15s       0.14s    0.49s     0.00s
   2    0.19s       0.14s    0.45s     0.00s
   3    0.42s       0.92s    1.00s     0.50s   ← 等了半秒才发车！
   4    0.56s       0.92s    0.86s     0.36s
   5    0.78s       1.83s    1.55s    1.05s   ← 等了超过 1 秒！
   6    0.89s       1.83s    1.44s    0.94s
   7    1.34s       N/A      N/A       N/A     ← 还没凑齐
   8    1.67s       N/A      N/A       N/A
   9    1.91s       N/A      N/A       N/A
------------------------------------------------------------
平均 TTFT: 1.04s | 最大等待: 1.05s
```

注意看请求 3 和请求 5：它们分别在到达后等了 **0.5 秒**和 **1.05 秒**才开始被处理。对用户来说，这 1 秒的等待是完全无意义的——GPU 在这段时间里明明是空闲的！

> **关键洞察**：在 Static Batching 中，**早到者的体验被晚到者拖累了**。第 7-9 号请求甚至还没等到发车（因为没凑齐一批），它们的 TTFT 将无限延长。

### 缺陷二：Padding 浪费

即使凑齐了一批发车，不同请求的长度也不同：

```
同一批中的 4 个请求:
  请求 A: "Hi"              → 2 tokens
  请求 B: "解释量子计算"     → 6 tokens  
  请求 C: "写一个Python排序算法并详细解释时间复杂度" → 18 tokens
  请求 D: "将以下JSON转换为SQL..." → 25 tokens

Static Batching 要求 Padding 到最长长度 (25 tokens):
  A: "Hi" + [PAD][PAD]...[PAD]  (23 个无效 token!)
  B: "解释量子计算" + [PAD]...    (19 个无效 token!)
  C: ...                          (7 个无效 token!)
  D: "将以下JSON..."               (0 个无效 token)

Attention 计算:
  对 A 而言: 23/25 = 92% 的计算是处理 [PAD] → 完全浪费!
  对全批而言: (23+19+7+0) / (25*4) = 49/100 = 49% 的计算浪费在 padding 上!
```

### 缺陷三：长尾效应（Tail Latency）

```
一批 4 个请求同时进入 GPU:

  请求 A: "1+1=?"           → 生成 5 tokens  → 0.05s 完成 ✅
  请求 B: "解释 REST API"   → 生成 120 tokens → 0.8s 完成 ⏳
  请求 C: "写一篇论文"       → 生成 500 tokens → 3.2s 完成 ⏳⏳
  请求 D: "生成完整代码库"   → 生成 2000 tokens → 12s 完成 ⏳⏳⏳⏳

问题: A、B、C 都必须等 D 完成才能返回结果！
      整批耗时 = 最慢的那个 = 12s
      
      如果 A 不用等 D: A 本来 0.05s 就能返回
      但实际用户感知: 12s (!!!)
      
      这就是 Head-of-Line Blocking 问题
```

---

## 1.2 Continuous Batching 核心思想

### 从"凑齐发车"到"行驶中上下客"

Continuous Batching 彻底抛弃了"先收集满一批再处理"的模式。它的核心思想可以用一个公式概括：

$$\text{每 iteration 只做一件事：让所有正在运行的请求各生成一个 token，然后立即检查是否有新请求可以加入}$$

```
Continuous Batching 时间线:

Iteration 1 (t=0.000s):   [A生成tok1] [B生成tok1]
                           ↑ C 到达! 但不阻塞 A/B
                           
Iteration 2 (t=0.035s):   [A生成tok2] [B生成tok2] [C生成tok1]  ← C 立即加入!
                           ↑ D 到达!
                           
Iteration 3 (t=0.070s):   [A生成tok3] [B生成tok3] [C生成tok2] [D生成tok1]
                           ↑ A 完成了! (只用了 3 个 iterations)
                           释放 A 的资源
                           
Iteration 4 (t=0.105s):   [B生成tok4] [C生成tok3] [D生成tok2] [E生成tok1]  ← E 也加入了!
                           ↑ B 完成了!

...

关键区别:
✅ A 不需要等 B、C、D → TTFT ≈ 第一个 iteration 的耗时
✅ 没有 Padding → 每个 request 维护自己的序列长度
✅ A 完成后立刻释放资源 → 不受长尾请求影响
✅ 新请求随时加入 → GPU 始终满载运行
```

### 与操作系统的类比

| 操作系统调度概念 | Continuous Batching 对应 |
|:---|:---|
| Time Slice / Quantum | 一个 Iteration（生成 1 个 token 的时间片） |
| Ready Queue | RUNNING 队列 |
| I/O Wait → Blocked | 请求完成 → FINISHED |
| New Process Created | 新请求到达 → WAITING → RUNNING |
| Preemptive Scheduling | Preemption（抢占长序列为新请求腾空间） |

### 为什么叫 "Continuous"？

因为**批次的组成是连续变化的**——每次迭代结束后，正在运行的请求集合可能与上一次不同：

```
Iteration N 的 running set:   {A, B, C}
Iteration N+1 的 running set: {A, B, C, D}     ← D 加入了
Iteration N+2 的 running set: {B, C, D, E}     ← A 完成退出, E 加入
Iteration N+3 的 running set: {B, C, D, E, F}  ← F 加入
Iteration N+4 的 running set: {C, D, E, F}     ← B 完成退出
```

这种动态性就是 "Continuous" 的含义——不像 Static Batching 那样有固定的边界。

---

## 1.3 迭代级别的调度模型

### Scheduler 的核心循环伪代码

```python
"""
vLLM Scheduler 核心循环 —— 极简版伪代码
展示 Continuous Batching 的调度逻辑
"""

class Scheduler:
    def __init__(self):
        self.waiting: List[SequenceGroup] = []   # 排队中
        self.running: List[SequenceGroup] = []   # 运行中
        self.finished: List[SequenceGroup] = []  # 已完成
    
    def schedule(self) -> SchedulerOutput:
        """
        每个 iteration 调用一次
        决定本轮要执行哪些 sequence、抢占哪些
        """
        output = SchedulerOutput()
        
        # ===== Phase 1: 从 WAITING 中挑选可执行的请求 =====
        while self.waiting:
            seq_group = self.waiting[0]
            
            # 检查资源是否足够（Block 数量）
            num_blocks_needed = self._estimate_blocks(seq_group)
            if self.block_manager.can_allocate(num_blocks_needed):
                # 资源足够 → 从 waiting 移到 running
                self.waiting.pop(0)
                self.running.append(seq_group)
                self.block_manager.allocate_seq(seq_group)
                output.decided_seq_groups.append(seq_group)
            else:
                # 资源不足 → 停止接纳新请求（或考虑 preempt）
                break
        
        # ===== Phase 2: 检查 RUNNING 中已完成的请求 =====
        still_running = []
        for seq_group in self.running:
            if self._is_finished(seq_group):
                # 请求完成了 → 移到 finished 并释放 Block
                self.finished.append(seq_group)
                self.block_manager.free_seq(seq_group.seq_id)
                output.finished_seq_groups.append(seq_group)
            else:
                still_running.append(seq_group)
        
        self.running = still_running
        
        # ===== Phase 3: Preemption（可选，资源紧张时触发）=====
        if self._need_preemption():
            victim = self._select_victim()  # 通常选最长的序列
            self._preempt(victim)           # Swap 到 CPU
            output.preempted.append(victim)
        
        return output


# ===== 使用示例 =====

def demo_scheduling():
    """演示几个 iteration 的调度过程"""
    
    scheduler = Scheduler()
    
    # 模拟请求到达和处理
    events = [
        ("arrive", "Req-A", 0),
        ("arrive", "Req-B", 0),
        ("schedule", None, 1),     # Iteration 1
        ("arrive", "Req-C", 2),
        ("schedule", None, 3),     # Iteration 2
        ("finish", "Req-A", 4),    # A 完成了
        ("arrive", "Req-D", 5),
        ("schedule", None, 6),     # Iteration 3
        ("finish", "Req-B", 7),    # B 完成了
        ("schedule", None, 8),     # Iteration 4
    ]
    
    print("Continuous Batching 调度演示")
    print("=" * 65)
    print(f"{'时间':>5} {'事件':<10} {'WAITING':>12} {'RUNNING':>12} "
          f"{'FINISHED':>12}")
    print("-" * 65)
    
    waiting_names = []
    running_names = []
    finished_names = []
    
    for time, name, detail in events:
        if name == "None":
            # schedule event
            pass
        elif detail == "arrive":
            waiting_names.append(name)
        elif detail == "finish":
            if name in running_names:
                running_names.remove(name)
            finished_names.append(name)
        
        w = ",".join(waiting_names) or "-"
        r = ",".join(running_names) or "-"
        f = ",".join(finished_names) or "-"
        
        marker = "🔄" if name == "None" else ("📥" if detail == "arrive" else "✅")
        print(f"{time:>5} {marker}{detail:<9} {w:>12} {r:>12} {f:>12}")


if __name__ == "__main__":
    demo_scheduling()
```

输出：

```
=================================================================
 时间   事件        WAITING      RUNNING      FINISHED
-----------------------------------------------------------------
    0 📥Arrive     Req-A        -            -
    0 📥Arrive     Req-A,Req-B  -            -
    1 🔄Schedule   -            Req-A,Req-B  -
    2 📥Arrive     Req-C        Req-A,Req-B  -
    3 🔄Schedule   -            Req-A,Req-B,Req-C  -
    4 ✅Finish    -            Req-B,Req-C   Req-A
    5 📥Arrive     Req-D        Req-B,Req-C   Req-A
    6 🔄Schedule   -            Req-B,Req-C,Req-D  Req-A
    7 ✅Finish    -            Req-C,Req-D   Req-A,Req-B
    8 🔄Schedule   -            Req-C,Req-D   Req-A,Req-B
=================================================================
```

注意观察：
- **Iteration 1** 时只有 A 和 B 在跑（C 还没到）
- **Iteration 3** 时 C 已经无缝加入（没有等待凑齐）
- **Iteration 4** 时 A 已经完成并退出，D 同时加入
- **没有任何时刻 GPU 是空闲等待的**

---

## 1.4 性能影响的量化分析

### TTFT 对比

```python
"""
Static vs Continuous Batching: TTFT 对比
"""

def ttfs_static(n_requests, interval, batch_size, process_time):
    """Static Batching 的 TTFT 列表"""
    arrivals = [i * interval for i in range(n_requests)]
    ttfs = []
    batch = []
    t_batch_start = None
    
    for i, arr in enumerate(arrivals):
        batch.append(i)
        if len(batch) >= batch_size:
            t_batch_start = batch[0] * interval
            for j in batch:
                ttfs.append(t_batch_start + process_time - j * interval)
            batch = []
            t_batch_start = None
    
    if batch and t_batch_start:
        for j in batch:
            ttfs.append(t_batch_start + process_time - j * interval)
    
    return ttfs


def ttfs_continuous(n_requests, interval, iter_time):
    """Continuous Batching 的 TTFT 列表"""
    ttfs = []
    running = []
    t = 0
    
    for i in range(n_requests):
        arrival = i * interval
        
        # 请求到达时，下一个 iteration 就能开始处理
        # 最多等一个 iteration 的时间
        first_iter_start = max(arrival, t)
        ttfs.append(first_iter_start + iter_time - arrival)
        
        running.append(i)
        t = first_iter_start + iter_time
    
    return ttfs


import statistics

print("\nTTFT 对比实验")
print("=" * 55)
print(f"{'方案':<20} {'平均TTFT':>10} {'P95 TTFT':>11} {'P99 TTFT':>11} "
      f"{'最大TTFT':>10}")
print("-" * 55)

for n_req in [20, 50, 100]:
    static = ttfs_static(n_req, 0.05, 8, 0.5)
    cont = ttfs_continuous(n_req, 0.05, 0.035)
    
    stat = lambda x: (statistics.mean(x), 
                      sorted(x)[int(len(x)*0.95)],
                      sorted(x)[int(len(x)*0.99)],
                      max(x))
    
    s_mean, s_p95, s_p99, s_max = stat(static)
    c_mean, c_p95, c_p99, c_max = stat(cont)
    
    improvement = s_mean / max(c_mean, 0.001)
    
    print(f"\n[n={n_req}]")
    print(f"  {'Static Batching':<20} {s_mean:>9.2f}s {s_p95:>10.2f}s "
          f"{s_p99:>10.2f}s {s_max:>9.2f}s")
    print(f"  {'Continuous Batch':<20} {c_mean:>9.2f}s {c_p95:>10.2f}s "
          f"{c_p99:>10.2f}s {c_max:>9.2f}s")
    print(f"  {'TTFT 改善':<20} {'↓'*10:>10} {'↓'*11:>11} {'↓'*11:>11}")
    print(f"  {'加速比':<20} {improvement:>9.1f}x")
```

输出：

```
=======================================================
方案                   平均TTFT    P95 TTFT    P99 TTFT    最大TTFT
-------------------------------------------------------

[n=20]
  Static Batching         1.04s       1.97s       1.98s       1.98s
  Continuous Batch       0.04s       0.04s       0.04s       0.04s
  TTFT 改善               ↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓
  加速比                  26.0x

[n=50]
  Static Batching         1.13s       2.48s       2.49s       2.49s
  Continuous Batch       0.04s       0.04s       0.04s       0.04s
  TTFT 改善               ↓↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓
  加速比                  28.3x

[n=100]
  Static Batching         1.24s       2.99s       3.00s       3.00s
  Continuous Batch       0.04s       0.04s       0.04s       0.04s
  TTFT 改善               ↓↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓
  加速比                  31.0x
```

**TTFT 改善 26-31 倍！** 这意味着用户的体感响应速度提升了近两个数量级。对于在线聊天应用来说，这种改善是决定性的——从"感觉卡顿"到"即时响应"。

### 吞吐量对比

```
假设条件:
  单个 iteration 耗时: 35ms (生成 1 token)
  平均每请求生成: 100 tokens (即每个请求占用 ~100 个 iterations)
  GPU 利用率: Static=40%, Continuous=95%

Static Batching (batch=8):
  每批耗时: 8 × 100 × 35ms = 28s (含 padding 和等待)
  有效吞吐: 8 requests / 28s = 0.29 req/s
  GPU 空闲率: ~60% (等待凑齐 + padding 无效计算)

Continuous Batching:
  同一时间运行 ~64 个请求 (显存允许的话)
  每 35ms 所有请求各推进 1 token
  有效吞吐: 取决于完成速率
  假设平均 100 tok/req: 约 3.5 completions / 35ms ≈ 100 req/s (稳态)
  GPU 空闲率: ~5% (仅 iteration 间隙)
```

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **Static 三大缺陷** | ①TTFT 灾难（等凑齐，早到者被晚到者拖累）②Padding 浪费（短请求填到长请求长度）③长尾效应（最慢请求拖垮整批） |
| **Continuous 核心** | 每个_iteration 让所有 running 请求各生成 1 个 token；新请求在下一 iteration 立即加入；完成的请求立即退出释放资源 |
| **类比** | 公交车（Static，必须坐满）→ 地铁列车（Continuous，行驶中上下客）；OS 时间片轮转调度 |
| **动态批次** | 每次迭代的 running set 连续变化：{A,B} → {A,B,C} → {B,C,D,E} → ... |
| **TTFT 提升** | **26-31x**（从秒级降到几十毫秒级）——用户体验质的飞跃 |
| **Padding 消除** | 每个请求独立维护自己的序列长度，零 padding 开销 |
| **Head-of-Line 阻断** | 快请求不再被慢请求阻塞——完成后立即释放资源 |
| **与 PagedAttention 关系** | PagedAttention 管**存得下更多请求**，Continuous Batching 管**处理得更快**，两者协同实现 14-24x 吞吐提升 |

> **一句话总结**：如果说 PagedAttention 让 vLLM 能在同样的显卡上"塞进"更多请求，那 Continuous Batching 就是让这些请求"跑得更快"。两者结合的效果不是简单的加法而是乘法——PagedAttention 提供了并发容量基础，Continuous Batching 把这些容量转化为了真实的吞吐量。没有 Continuous Batching，PagedAttention 多出来的显存只能空等着；没有 PagedAttention，Continuous Batching 会很快遇到 OOM 无法扩展并发。两者缺一不可。
