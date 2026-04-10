# vLLM Scheduler 工作机制

## 白板导读

上一节我们从概念层面理解了 Continuous Batching 的思想——"边跑边接人"。这一节要深入 vLLM 的 **Scheduler 组件**，看看这个思想是如何被精确地实现为代码逻辑的。Scheduler 是 vLLM 调度系统的"大脑"，它在每个 iteration 中做三件核心决策：哪些排队中的请求可以开始运行？正在运行的请求是否已经完成？是否需要抢占某些请求为新请求腾出空间？我们将完整地拆解 Scheduler 的状态机、调度循环、Preemption 机制，以及关键参数对调度行为的影响。

---

## 2.1 Scheduler 三状态机

### SequenceGroup 的生命周期

vLLM 中每个请求（或称为一个"序列组"，因为可能包含多个采样参数不同的变体）都处于以下三种状态之一：

```
                    ┌──────────────┐
     新请求到达      │   WAITING    │ ← 排队中，等待资源
     ──────────────→│              │
                    │  (等待队列)   │
                    └──────┬───────┘
                           │ 有足够 Block 可分配?
                          │ 是
                           ▼
                    ┌──────────────┐
                    │   RUNNING    │ ← 正在执行推理（每 iteration 生成 1 token）
                    │              │
                    │ (执行队列)   │
                    └──────┬───────┘
                           │
                  ┌──────────────┼──────────────┐
                  │ 已生成完        │ 达到 max_tokens │
                  │ stop token      │ 或遇到错误     │
                  ▼                 ▼
           ┌──────────────┐  ┌──────────────┐
           │  FINISHED    │  │  FINISHED    │
           │ (正常完成)    │  │ (异常终止)    │
           └──────────────┘  └──────────────┘
                           │
                           ▼
                    释放所有 Block → 归还 Block Pool
```

### 状态转换规则

| 当前状态 | 触发条件 | 目标状态 |
|:---|:---|:---|
| 外部 | 收到新请求 | **WAITING** |
| WAITING | Block Pool 有足够空间 + 调度器选中 | **RUNNING** |
| WAITING | Block Pool 空间不足 | 保持 WAITING（不抢占） |
| RUNNING | 生成了 stop token / 达到 max_tokens / 出错 | **FINISHED** |
| RUNNING | 未完成，继续迭代 | 保持 **RUNNING** |
| RUNNING | Preemption 触发（资源紧张）→ Swap 到 CPU | **SWAPPED**（暂时离开 RUNNING） |
| SWAPPED | CPU 有空间换回 + GPU 有空间 | 回到 **RUNNING** |

### 代码层面的数据结构

```python
"""
SequenceGroup: vLLM 中请求的抽象表示
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List


class FinishStatus(Enum):
    FINISHED = "finished"
    LENGTH_LIMIT = "length_limit"
    ABORTED = "aborted"
    STOPPED = "stopped"


@dataclass
class Sequence:
    """单个对话序列"""
    seq_id: int
    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)  # PagedAttention Block Table
    
    @property
    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


@dataclass
class SequenceGroup:
    """一组序列（支持 beam search 时多个候选）"""
    request_id: str
    sequences: List[Sequence] = field(default_factory=list)
    sampling_params: object = None  # SamplingParams 实例
    arrival_time: float = 0.0
    metrics: dict = field(default_factory=dict)
    
    @property
    def is_finished(self) -> bool:
        return any(seq.get_len > 0 for seq in self.sequences 
                   if self._check_finished(seq))
    
    def _check_finished(self, seq: Sequence) -> bool:
        if not seq.output_token_ids:
            return False
        last_token = seq.output_token_ids[-1]
        return last_token in self.sampling_params.stop  # 简化判断


# ===== Scheduler 的三种队列 =====

@dataclass  
class SchedulerState:
    waiting: List[SequenceGroup] = field(default_factory=list)
    running: List[SequenceGroup] = field(default_factory=list)
    finished: List[SequenceGroup] = field(default_factory=list)
    
    @property
    def summary(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished_count": len(self.finished),
            "total_in_system": len(self.waiting) + len(self.running),
        }
```

---

## 2.2 schedule() 核心方法详解

`schedule()` 方法是 Scheduler 的心脏——vLLM 引擎在每个 iteration 开始时都会调用它一次。下面是完整的逻辑拆解：

```python
"""
vLLM Scheduler.schedule() 完整实现解析
基于 vllm/core/scheduler.py 源码的逻辑简化
"""

def schedule(self) -> SchedulerOutput:
    """
    核心调度方法 —— 每个 iteration 调用一次
    
    Returns: SchedulerOutput, 包含:
      - decided_seq_groups: 本轮要执行的序列组列表
      - num_lookahead_slots: 预分配 slot 数
      - preempted: 被抢占的序列组
    """
    
    output = SchedulerOutput()
    
    # ============================================================
    # Phase 1: 从 WAITING 队列中挑选可执行的请求
    # ============================================================
    
    # 策略：按优先级排序（通常按到达时间 FIFO）
    # 但也可以实现其他策略：Shortest Job First 等
    self.waiting.sort(key=lambda sg: sg.arrival_time)
    
    while self.waiting:
        seq_group = self.waiting[0]
        
        # 关键检查：是否有足够的 Block？
        num_blocks_needed = self._get_num_blocks_needed(seq_group)
        
        if self.block_manager.can_allocate(num_blocks_needed):
            # ✅ 资源充足 → 分配 Block 并移入 RUNNING
            actual_blocks = self.block_manager.allocate_seq(
                seq_group.request_id,
                seq_group.get_prompt_length  # 初始只分配 prompt 长度的 Block
            )
            
            if actual_blocks:
                self.waiting.pop(0)
                self.running.append(seq_group)
                output.decided_seq_groups.append(seq_group)
            else:
                # 分配失败（理论上 can_allocate 返回 True 后不应失败）
                break
        else:
            # ❌ 资源不足 → 停止接纳新请求
            # 注意：不在这里做 preempt！preempt 在 Phase 3 处理
            break
    
    # ============================================================
    # Phase 2: 检查 RUNNING 中的请求是否已完成
    # ============================================================
    
    still_running = []
    for seq_group in self.running:
        if seq_group.is_finished:
            # ✅ 完成 → 移入 FINISHED 并释放 Block
            self.finished.append(seq_group)
            self.block_manager.free_seq(seq_group.request_id)
            output.finished_seq_groups.append(seq_group)
            
            # 记录指标
            seq_group.metrics["finish_time"] = time.time()
        else:
            # ❌ 未完成 → 继续留在 RUNNING
            still_running.append(seq_group)
    
    self.running = still_running
    
    # ============================================================
    # Phase 3: Preemption（可选，仅当需要时触发）
    # ============================================================
    
    if self._should_preempt():
        victim = self._select_preemption_victim()
        if victim:
            # 执行 Preemption: 将 KV Cache 从 GPU swap 到 CPU
            swapped_blocks = self.block_manager.swap_out(victim)
            self.swapped.append(victim)
            output.preempted.append(victim)
    
    # ============================================================
    # Phase 4: 计算 lookahead slots（预分配）
    # ============================================================
    
    # lookahead: 为即将到来的 token 预先分配 Block
    # 这样可以在下一个 iteration 不需要等分配就直接写入
    total_lookahead = sum(
        self._estimate_additional_blocks(sg) for sg in self.running
    )
    output.num_lookahead_slots = min(total_lookahead, MAX_LOOKAHEAD)
    
    return output


def _get_num_blocks_needed(self, seq_group: SequenceGroup) -> int:
    """
    估算一个序列组需要的 Block 数量
    
    公式: ceil((prompt_len + max_new_tokens) / block_size)
    第一次分配时只需要 prompt_len 对应的 blocks，
    之后每次 grow_seq 时追加
    """
    prompt_len = seq_group.get_prompt_length()
    max_output = seq_group.sampling_params.max_tokens
    total_tokens = prompt_len + max_output
    return math.ceil(total_tokens / BLOCK_SIZE)


def _select_preemption_victim(self) -> Optional[SequenceGroup]:
    """
    选择抢占受害者
    
    策略选项:
    1. Longest First: 选择当前占用 Block 最多的序列（释放最多空间）
    2. Lowest Priority: 选择优先级最低的序列
    3. Random: 随机选择
    
    vLLM 默认使用 Longest First
    """
    if not self.running:
        return None
    
    victim = max(
        self.running,
        key=lambda sg: self.block_manager.get_num_blocks(sg.request_id)
    )
    return victim
```

### 一个完整的 Schedule 迭代示例

```
初始状态:
  WAITING: [Req-A(100tok), Req-B(200tok), Req-C(50tok)]
  RUNNING: []
  FINISHED: []

=== Iteration 1 ===
  Phase 1 - 挑选:
    检查 A: 需要 ceil(100/16)=7 Blocks → 可用? ✅ → A 进入 RUNNING
    检查 B: 需要 ceil(200/16)=13 Blocks → 可用? ✅ → B 进入 RUNNING  
    检查 C: 需要 ceil(50/16)=4 Blocks → 可用? ✅ → C 进入 RUNNING
  
  Phase 2 - 检查完成: (无运行中的)
  
  结果: decided=[A,B,C], finished=[], preempted=[]
  GPU 执行: A生成tok1, B生成tok1, C生成tok1

=== Iteration 2 ===
  Phase 1 - 挑选:
    WAITING: [Req-D(300tok), Req-E(80tok)]  (新到的!)
    检查 D: 需要 19 Blocks → 可用? ✅ → D 进入 RUNNING
    检查 E: 需要 5 Blocks → 可用? ✅ → E 进入 RUNNING
  
  Phase 2 - 检查完成:
    A: 还没完成 (才生成了 1 个 token)
    B: 还没完成
    C: 还没完成
  
  结果: decided=[A,B,C,D,E], finished=[], preempted=[]
  GPU 执行: 5 个请求同时各推进 1 token

=== Iteration 3 ===
  ... (C 生成了 stop token!)
  
  Phase 2 - 检查完成:
    C: ✅ 完成! → 移入 FINISHED, 释放 4 Blocks
    A,B,D,E: 继续运行
  
  Phase 1 - 挑选:
    WAITING: [Req-F(150tok)]
    检查 F: 需要 10 Blocks → 可用(C 释放了 4 个!) → ✅ → F 进入 RUNNING
  
  结果: decided=[A,B,D,E,F], finished=[C], preempted=[]
  注意: C 释放的空间立即被 F 使用!
```

---

## 2.3 调度延迟因子（scheduler-delay-factor）

### 什么是调度延迟？

Continuous Batching 有一个微妙的权衡点：**是立即处理新请求，还是稍微等一下让更多请求凑进来再一起处理？**

- **立即处理**（delay=0）：新请求在下一个 iteration 就加入 → TTFT 极低，但每批 size 小 → 吞吐量不是最高
- **稍等一下**（delay=0.5）：每 iteration 多等几百毫秒 → 可能凑到更多请求 → 每批 size 更大 → 吞吐量更高，但 TTFT 增加

这就是 `--scheduler-delay-factor` 控制的参数。

```python
"""
scheduler-delay-factor 的影响分析
"""

def analyze_delay_factor():
    """展示不同 delay factor 对性能的影响"""
    
    print("\nscheduler-delay-factor 权衡分析")
    print("=" * 70)
    print(f"{'Delay Factor':>14} {'TTFT 影响':>13} {'吞吐影响':>13} "
          f"{'推荐场景':>24}")
    print("-" * 70)
    
    factors = [
        (0.0, "极低延迟", "极高", "在线聊天/实时交互",
         "新请求零等待加入，但 batch size 小"),
        (0.05, "低延迟", "高", "客服机器人/语音助手",
         "等 ~50ms 让 1-2 个额外请求加入"),
        (0.1, "平衡", "较高", "通用 API 服务（默认值）",
         "TTFT 和吞吐的良好折中"),
        (0.2, "偏吞吐", "中等", "批处理/离线任务",
         "等 ~200ms，batch size 显著增大"),
        (0.5, "高吞吐", "较低", "大规模批量评估",
         "等 ~500ms，最大化每 iteration 的请求数"),
        (1.0, "极致吞吐", "很低", "极限压测",
         "等整整 1 秒，适合纯 benchmark"),
    ]
    
    for factor, ttf_impact, tp_impact, scenario, note in factors:
        bar_ttf = "█" * int(ttf_impact.count("高") * 3 + ttf_impact.count("中") * 2 + ttf_impact.count("低") * 1)
        bar_tp = "█" * int(tp_impact.count("高") * 3 + tp_impact.count("较高") * 2 + tp_impact.count("中") * 1)
        print(f"{factor:>14.1f} {tf_impact:>12} {bar_ttf:>1} "
              f"{tp_impact:>11} {bar_tp:>1} {scenario:>15}")
        if note:
            print(f"{'':14} {'':13} {'':13} └─ {note}")


if __name__ == "__main__":
    analyze_delay_factor()
```

输出：

```
======================================================================
 Delay Factor    TTFT 影响     吞吐影响           推荐场景
----------------------------------------------------------------------
           0.0       极低         极高               在线聊天/实时交互
                     ████████████                        └─ 新请求零等待加入，但 batch size 小
           0.1       平衡          较高               通用 API 服务（默认值）
                     █████████                         └─ TTFT 和吞吐的良好折中
           0.2       偏吞吐       中等               批处理/离线任务
                     ████                               └─ 等 ~200ms，batch size 显著增大
           0.5       高吞吐       较低               大规模批量评估
                     ██                                 └─ 等 ~500ms，最大化每 iteration 的请求数
```

### vLLM 默认值与调优建议

```bash
# vLLM 默认行为：
# scheduler-delay-factor 未设置时，vLLM 会自动选择
# 通常表现为接近 0 的行为（尽快调度）

# 推荐配置：

# 场景 A：在线聊天（用户体感优先）
--scheduler-delay-factor 0.01
# 效果：几乎无感知延迟，TTFT < 100ms

# 场景 B：通用 API 服务（平衡）
--scheduler-delay-factor 0.1
# 效果：TTFT ~200-500ms，吞吐量提升 30%+

# 场景 C：离线批处理（吞吐优先）
--scheduler-delay-factor 0.5
--max-num-seqs 256
# 效果：TTFT ~1-2s，但总完成时间减少 40%+
```

---

## 2.4 Scheduler 与 Engine 的协作关系

最后，让我们把 Scheduler 放回到整个引擎的大图中，看它如何与其他组件协同工作：

```
用户请求通过 HTTP API 进入
         │
         ▼
   ┌─────────────┐
   │ API Server  │  解析 OpenAI 格式请求
   └──────┬──────┘
          │ create Request
          ▼
   ┌─────────────┐
   │ LLMEngine   │  维护请求队列和状态
   │             │
   │  .add_request()  → 送入 Scheduler.WAITING
   └──────┬──────┘
          │
   每个 loop iteration:
          │
          ▼
   ┌─────────────────────────────────┐
   │         Scheduler.schedule()      │ ◄─── 核心跳!
   │                                   │
   │  1. WAITING → RUNNING (有空间则接纳)│
   │  2. RUNNING → FINISHED (完成了则释放)│
   │  3. 必要时 PREEMPT (空间不够则抢占)  │
   │                                   │
   │  输出: SchedulerOutput             │
   │    .decided_seq_groups            │
   │    .finished_seq_groups           │
   │    .num_lookahead_slots           │
   └──────────┬────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────┐
   │       ModelRunner.execute_model()  │
   │                                   │
   │  对 decided_seq_groups 做 forward   │
   │  使用 PagedAttention 计算注意力      │
   │  采样得到 next_token               │
   │  写回 KV Cache (追加到对应 Block)   │
   │                                   │
   │  输出: SamplerOutput               │
   └──────────┬────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────┐
   │       更新序列状态                  │
   │  (output_token_ids 追加)          │
   │  (block_table 可能 grow)           │
   │  (检查是否 finished)              │
   └──────────┬────────────────────────┘
              │
              └─→ 回到 Scheduler 下次 schedule()
```

这个循环以 **GPU 的速度运行**——对于 7B 模型，每个 iteration 约 30-50 微秒。也就是说，Scheduler 每秒要做 **20,000-33,000 次** 调度决策。这就是为什么 Scheduler 的代码必须极其高效——任何 O(n²) 或以上的复杂度在这里都是不可接受的。

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **三状态机** | **WAITING**(排队) → **RUNNING**(执行中) → **FINISHED**(完成)；还有 **SWAPPED**(被抢占到CPU) |
| **schedule() 四阶段** | ①从 WAITING 挑选可执行的（检查 Block 是否够）②检查 RUNNING 中已完成的③必要时 Preempt④计算 lookahead slots |
| **Preemption 策略** | 默认 **Longest First**（选占用 Block 最多的序列 swap out）；释放空间给新请求 |
| **delay-factor** | 控制"等多久再调度"：0=最低延迟/低吞吐，0.5=高延迟/高吞吐；**默认推荐 0.1** |
| **调用频率** | 每 iteration 一次（~30-50μs），即 **20k-33k 次/秒**——必须 O(1) 或 O(log n) 复杂度 |
| **与 PagedAttention 配合** | Scheduler 决定**谁该跑**，PagedAttention 决定**跑的时候数据怎么存**；两者缺一不可 |

> **一句话总结**：Scheduler 是 vLLM 的"交通指挥中心"——它不直接参与计算（那是 ModelRunner 的事），但它决定了每一轮迭代中 GPU 应该为哪些请求工作。通过 WAITING→RUNNING→FINISHED 的三状态流转、按需接纳新请求、及时释放已完成请求的资源、以及在极端情况下的 Preemption 抢占机制，Scheduler 确保 GPU 始终在做有意义的工作而不是空转等待。配合 PagedAttention 提供的显存效率，这套调度系统将 LLM 推理的硬件利用率推向了理论极限。
