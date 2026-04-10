# PagedAttention 实现细节

## 白板导读

前两节我们从"为什么需要 PagedAttention"讲到了它的设计思想——OS 虚拟内存类比、Block 切分、Block Table 映射。现在要进入最硬核的部分：**vLLM 是如何在代码层面实现 PagedAttention 的？** 这涉及到三个层面的工作：第一，Attention 计算本身如何适配非连续的 Block 存储；第二，多个请求共享相同前缀时如何避免重复存储（Prefix Caching）；第三，自定义 CUDA Kernel 如何高效地完成分散-收集式的注意力计算。这些内容既是理解 vLLM 内部机制的关键，也是面试中区分"用过 vLLM"和"懂 vLLM"的分水岭。

---

## 3.1 Attention 计算中的 Block 收集

### 传统 Attention vs PagedAttention 的计算流程对比

在传统方案中，KV Cache 是一个连续的张量：

```python
# 传统方案的 KV Cache 形状
# kv_cache: [num_layers, 2, num_kv_heads, seq_len, head_dim]
# 这是一个大矩阵，所有 token 的 K 和 V 连续排列

# Attention 计算（简化版）
def attention_traditional(query, kv_cache):
    """
    query: [num_heads, head_dim]          # 当前位置的 Q
    kv_cache: [2, num_kv_heads, seq_len, head_dim]  # 所有历史位置 K,V
    """
    keys = kv_cache[0]    # [num_kv_heads, seq_len, head_dim]
    values = kv_cache[1]  # [num_kv_heads, seq_len, head_dim]
    
    # 直接做矩阵乘法 —— 因为是连续的！
    scores = torch.matmul(query, keys.transpose(-1, -2)) / (head_dim ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, values)
    return output
```

但在 PagedAttention 中，KV Cache 不再连续——它分散在多个物理 Block 中。计算时需要先**收集（gather）**数据，再执行 Attention：

```python
"""
PagedAttention 的核心计算逻辑
展示如何从分散的 Block 中重建用于计算的 KV 张量
"""

import torch
from typing import List


class PagedAttentionCompute:
    """PagedAttention 计算引擎（简化版）"""
    
    def __init__(self, block_size: int = 16, num_layers: int = 32,
                 num_kv_heads: int = 8, head_dim: int = 128):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # 物理存储：所有 Block 的实际数据
        # shape: [total_blocks, num_layers, 2(K+V), num_kv_heads, block_size, head_dim]
        # 这是 GPU 上的一块预分配的大显存区域
        self.kv_cache_storage: torch.Tensor = None
    
    def gather_kv_from_blocks(
        self,
        block_table: List[int],       # 该序列的 Block Table
        layer_id: int,                 # 当前 Transformer 层
        num_valid_tokens: int,         # 有效 token 数量（可能 < len(block_table)*block_size）
    ) -> tuple:
        """
        从分散的 Block 中收集 KV 数据，组成连续张量
        
        Args:
            block_table: 如 [3, 7, 12, 21] 表示 4 个物理块
            layer_id: 第几层
            num_valid_tokens: 实际有效的 token 数
            
        Returns:
            (keys, values): 连续的 [num_kv_heads, total_tokens, head_dim] 张量
        """
        all_keys = []
        all_values = []
        tokens_collected = 0
        
        for block_idx, physical_block_id in enumerate(block_table):
            start_token = block_idx * self.block_size
            end_token = min(start_token + self.block_size, num_valid_tokens)
            
            if start_token >= num_valid_tokens:
                break
            
            # 从物理存储中取出这个 Block 的数据
            # 实际 vLLM 中使用自定义 CUDA kernel 做高效 indexing
            block_data = self.kv_cache_storage[
                physical_block_id,     # 第几个物理块
                layer_id,             # 第几层
                :,                    # K 和 V (dim=2)
                :,                    # 所有 KV 头
                :end_token - start_token,  # 这个块中有效的部分
                :                     # head_dim
            ]
            
            keys_block = block_data[0]  # K 部分
            values_block = block_data[1]  # V 部分
            
            all_keys.append(keys_block)
            all_values.append(values_block)
            tokens_collected += (end_token - start_token)
        
        if all_keys:
            keys = torch.cat(all_keys, dim=1)   # 沿 seq_len 维度拼接
            values = torch.cat(all_values, dim=1)
        else:
            keys = torch.empty(self.num_kv_heads, 0, self.head_dim)
            values = torch.empty(self.num_kv_heads, 0, self.head_dim)
        
        return keys, values
    
    def compute_attention(
        self,
        query: torch.Tensor,           # [num_heads, head_dim] 当前 Q
        block_table: List[int],
        layer_id: int,
        num_valid_tokens: int,
    ) -> torch.Tensor:
        """完整的 PagedAttention 计算流程"""
        
        # Step 1: Gather —— 从 Block 中收集分散的 K,V
        keys, values = self.gather_kv_from_blocks(
            block_table, layer_id, num_valid_tokens
        )
        
        # Step 2: Attention 计算（与传统方式完全相同的数学运算）
        scale = self.head_dim ** -0.5
        scores = torch.matmul(query.unsqueeze(1), keys.transpose(-1, -2)) * scale
        weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.matmul(weights, values)
        
        return output.squeeze(1)  # [num_heads, head_dim]


# ===== 使用示例 =====

def demo_paged_attn_compute():
    """演示 PagedAttention 的完整计算过程"""
    
    compute = PagedAttentionCompute(
        block_size=16, num_layers=32, num_kv_heads=8, head_dim=128
    )
    
    # 模拟一个序列: 长度 50 tokens, 占用 4 个 Block
    block_table = [3, 7, 12, 21]  # 物理 Block ID
    num_tokens = 50
    
    print("PagedAttention 计算演示")
    print("=" * 55)
    print(f"Block Table: {block_table}")
    print(f"序列长度: {num_tokens} tokens")
    print(f"占用 Block 数: {len(block_table)} (最后一块利用率: {(num_tokens % 16)/16*100:.0f}%)")
    print("-" * 55)
    
    for layer in [0, 15, 31]:  # 展示不同层的处理
        print(f"\nLayer {layer}:")
        print(f"  收集 Block {block_table} 中的 KV Cache...")
        print(f"  执行 Attention 计算 (Q × K^T → softmax → × V)")
        print(f"  输出: [{compute.num_kv_heads} heads × {compute.head_dim} dim]")


if __name__ == "__main__":
    demo_paged_attn_compute()
```

### 关键优化：不需要真的物理拼接！

上面的代码为了可读性展示了"收集后拼接"的过程，但 vLLM 实际实现中**并不需要真的把数据拷贝成连续张量**。它使用了更聪明的做法：

```python
# vLLM 实际使用的策略：indexing-based attention

def paged_attention_kernel(
    query,              # [num_heads, head_dim]
    key_cache,           # [num_blocks, num_kv_heads, block_size, head_dim] 全部 K cache
    value_cache,         # [num_blocks, num_kv_heads, block_size, head_dim] 全部 V cache
    block_table,         # [seq_len_in_blocks] 该序列的物理块ID列表
    context_lens,        # [batch] 每个序列的有效长度
    head_dim,
):
    """
    自定义 CUDA kernel 实现：
    不拼接，直接用 index 操作访问分散的 Block 数据
    
    伪代码:
    for each head h:
        for each query_position q_pos:
            q = query[h, q_pos]
            
            # 遍历该序列的所有已缓存位置
            score_sum = 0
            max_score = -inf
            for each cached position c_pos in [0..context_len):
                # 通过 block_table[c_pos // block_size] 找到物理块号
                block_id = block_table[c_pos // block_size]
                offset = c_pos % block_size
                
                k = key_cache[block_id, h, offset]  # 直接索引！不拷贝！
                score = dot(q, k) / sqrt(head_dim)
                
                # 在 kernel 内维护 softmax 的数值稳定版本
                max_score = max(max_score, score)
                score_sum += exp(score - max_score)
            
            # 第二遍：归一化并加权求和
            output = 0
            for each cached position c_pos:
                block_id = block_table[c_pos // block_size]
                offset = c_pos % block_size
                v = value_cache[block_id, h, offset]
                weight = exp(dot(q, k) - max_score) / score_sum
                output += weight * v
            
            output[h, q_pos] = output
    """
    pass  # 实际实现在 C++/CUDA 中
```

> **性能影响分析**：这种 index-based 方式避免了数据拷贝，但引入了**非连续内存访问**（GPU 的 global memory latency 约 200-800 cycles）。vLLM 通过以下手段缓解这个问题：
> 1. 将 Block 数据保持在 L2 cache 友好的布局中
> 2. 利用 CUDA Shared Memory 缓存频繁访问的 Block
> 3. 对同一 Block 内的 16 个 token 做向量化加载
> 
> 最终结果：PagedAttention 的 kernel 效率比传统连续 Attention 低约 **2-5%**，但换来了 **显存利用率提升 30-40%**——这是极其划算的 trade-off。

---

## 3.2 Prefix Caching（前缀缓存共享）

### 动机：System Prompt 的重复开销

在实际应用中，大量请求共享相同的 System Prompt：

```
请求 A: [System: "你是Python专家"] + "解释装饰器"
请求 B: [System: "你是Python专家"] + "写一个快速排序"
请求 C: [System: "你是Python专家"] + "什么是 GIL"
请求 D: [System: "你是Python专家"] + "推荐几个库"
...
```

每个请求的 System Prompt 都一样，对应的 KV Cache 也完全相同。如果不做任何优化，每个请求都要独立计算和存储一份 System Prompt 的 KV Cache——这不仅是空间浪费，更是**计算浪费**（Prompt Eval 阶段需要为每个请求重新计算一遍）。

### Prefix Caching 方案

PagedAttention 天然支持前缀缓存共享，因为：

1. **Block 是独立分配的**——多个 Sequence 可以指向同一个物理 Block
2. **引用计数机制**——知道有多少个 Sequence 在用同一个 Block

```python
"""
Prefix Caching 示意图解与实现
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CachedPrefix:
    """缓存的前缀（如 System Prompt 的 KV Cache）"""
    prefix_hash: str          # 内容哈希值（用于快速匹配）
    block_ids: List[int]      # 占用的物理 Block ID 列表
    ref_count: int = 0        # 引用计数（多少个序列在使用）
    content_length: int = 0   # 前缀的 token 数


class PrefixCacheManager:
    """前缀缓存管理器"""
    
    def __init__(self):
        self.cache: Dict[str, CachedPrefix] = {}
    
    def _hash_prompt(self, prompt_tokens: list) -> str:
        """将 prompt token 序列哈希化为缓存键"""
        import hashlib
        content = str(prompt_tokens)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_or_create(
        self,
        system_prompt_tokens: list,
        block_allocator,  # BlockAllocator 实例
        block_size: int = 16,
    ) -> List[int]:
        """
        获取或创建前缀缓存
        
        Returns: 该前缀占用的物理 Block ID 列表
        """
        prefix_hash = self._hash_prompt(system_prompt_tokens)
        
        # 命中缓存！
        if prefix_hash in self.cache:
            cached = self.cache[prefix_hash]
            cached.ref_count += 1
            return cached.block_ids
        
        # 未命中 → 分配新 Block 并计算 KV Cache
        num_tokens = len(system_prompt_tokens)
        num_blocks = (num_tokens + block_size - 1) // block_size
        
        new_blocks = block_allocator.allocate(num_blocks)
        if new_blocks is None:
            raise RuntimeError("无法分配前缀缓存 Block")
        
        # TODO: 这里会调用模型计算 System Prompt 的 KV Cache
        # 并写入到 new_blocks 对应的物理存储位置
        # _compute_and_store_kv(new_blocks, system_prompt_tokens, ...)
        
        self.cache[prefix_hash] = CachedPrefix(
            prefix_hash=prefix_hash,
            block_ids=new_blocks,
            ref_count=1,
            content_length=num_tokens,
        )
        
        return new_blocks
    
    def release(self, prefix_hash: str, block_allocator):
        """释放对一个前缀缓存的引用"""
        if prefix_hash not in self.cache:
            return
        
        cached = self.cache[prefix_hash]
        cached.ref_count -= 1
        
        if cached.ref_count <= 0:
            # 没有人用了 → 真正释放 Block
            block_allocator.free(cached.block_ids)
            del self.cache[prefix_hash]


# ===== 性能影响示例 =====

def demo_prefix_caching():
    """演示 Prefix Caching 的效果"""
    
    print("\nPrefix Caching 效果演示")
    print("=" * 60)
    
    system_tokens = ["你", "是", "一", "个", "有", "帮", "助", "的", "助手"]
    system_len = len(system_tokens)
    block_size = 16
    
    # 场景：100 个请求共享同一个 System Prompt
    num_requests = 100
    avg_user_tokens = 30  # 用户问题平均长度
    
    # 无 Prefix Caching
    blocks_without_cache = num_requests * (
        (system_len + block_size - 1) // block_size +
        (avg_user_tokens + block_size - 1) // block_size
    )
    
    # 有 Prefix Caching
    prefix_blocks = (system_len + block_size - 1) // block_size  # 只需 1 次
    per_request_extra = (avg_user_tokens + block_size - 1) // block_size
    blocks_with_cache = prefix_blocks + num_requests * per_request_extra
    
    saved = blocks_without_cache - blocks_with_cache
    saving_pct = saved / blocks_without_cache * 100
    
    print(f"请求数: {num_requests}")
    print(f"System Prompt: {system_len} tokens ({prefix_blocks} Blocks)")
    print(f"平均用户输入: ~{avg_user_tokens} tokens ({per_request_extra} Blocks)")
    print("-" * 60)
    print(f"无缓存总 Block 数:  {blocks_without_cache:,}")
    print(f"有缓存总 Block 数:  {blocks_with_cache:,}")
    print(f"节省 Block 数:      {saved:,} ({saving_pct:.1f}%)")
    print(f"\n💡 如果每个 Block = 128KB:")
    print(f"   节省显存: {saved * 128 / 1024:.1f} MB")
    print(f"   Prompt Eval 加速: ~{num_requests}x (System Prompt 只算一次!)")


if __name__ == "__main__":
    demo_prefix_caching()
```

输出：

```
============================================================
Prefix Caching 效果演示
============================================================
请求数: 100
System Prompt: 9 tokens (1 Blocks)
平均用户输入: ~30 tokens (2 Blocks)
------------------------------------------------------------
无缓存总 Block 数:  300
有缓存总 Block 数:  201
节省 Block 数:      99 (33.0%)

💡 如果每个 Block = 128KB:
   节省显存: 12.4 MB
   Prompt Eval 加速: ~100x (System Prompt 只算一次!)
```

> **注意**：33% 的节省看起来不大，这是因为我们的例子中 System Prompt 很短（9 tokens）。在生产环境中，System Prompt 通常包含详细的角色定义、规则约束、领域知识等，可能长达 **500-2000 tokens**。此时 Prefix Caching 的节省率可以轻松达到 **60-90%**。

---

## 3.3 Copy-on-Write（写时复制）

### 问题场景

当两个请求 A 和 B 共享了同一个前缀 Block（比如它们都有相同的 System Prompt），如果 B 需要在某个位置修改自己的 KV Cache（虽然这种情况较少见，但在某些变长 attention 或 speculative decoding 场景下会发生），直接修改会影响 A。

解决方案：**Copy-on-Write（COW）**

```python
"""
Copy-on-Write 机制示意
"""

class CopyOnWriteManager:
    """COW 管理器"""
    
    def __init__(self, block_allocator):
        self.allocator = block_allocator
        self.cow_records: dict = {}  # original_block_id → copy_block_id
    
    def write_to_shared_block(
        self,
        block_id: int,
        offset: int,
        new_value,
        writer_seq_id: str,
    ) -> int:
        """
        尝试写入一个被共享的 Block
        
        如果 ref_count > 1（被其他序列引用）:
            → 触发 COW：创建副本，让当前序列指向副本
            → 写入副本（不影响原始 Block）
            
        如果 ref_count == 1（只有自己在用）:
            → 直接写入（无需 COW）
        
        Returns: 实际写入到的 Block ID（可能是原块或新副本）
        """
        ref_count = self.allocator.get_ref_count(block_id)
        
        if ref_count > 1:
            # 需要 Copy-on-Write
            new_block = self.allocator.allocate(1)
            if new_block is None:
                raise OOMError("COW 失败：无法分配新 Block")
            
            # 复制原始 Block 的全部内容到新 Block
            self._copy_block(block_id, new_block[0])
            
            # 更新引用计数
            self.allocator.decrement_ref(block_id)
            self.allocator.set_ref(new_block[0], 1)
            
            # 记录 COW 关系
            self.cow_records[block_id] = new_block[0]
            
            # 写入副本
            self._write_value(new_block[0], offset, new_value)
            return new_block[0]
        else:
            # 直接写入
            self._write_value(block_id, offset, new_value)
            return block_id
    
    def _copy_block(self, src_id: int, dst_id: int):
        """GPU 上的高效 Block 复制（使用 cudaMemcpyDeviceToDevice）"""
        pass  # 实际调用 cudaMemcpy
    
    def _write_value(self, block_id, offset, value):
        """写入指定 Block 的指定偏移位置"""
        pass


# COW 流程图解
"""
时间线:

T1: Seq-A 和 Seq-B 共享 Block #5 (ref_count=2)

T2: Seq-B 尝试修改 Block #5 的 offset=8
    │
    ├─ 检查 ref_count(#5) = 2 > 1 → 需要 COW!
    │
    ├─ 分配新 Block #42
    │
    ├─ cudaMemcpy: Block #5 → Block #42 (完整复制)
    │
    ├─ ref_count(#5)-- → 1 (只剩 Seq-A)
    │  ref_count(#42)=1 (Seq-B 独占)
    │
    └─ 写入 Block #42 的 offset=8 ✅

最终状态:
  Seq-A: [..., #5, ...]      # 仍指向原始 Block
  Seq-B: [..., #42, ...]     # 指向 COW 副本
  Block #5 和 #42 内容相同（除了 offset=8 处）
"""
```

---

## 3.4 vLLM 源码关键类走读

让我们看看 vLLM 实际源码中与 PagedAttention 相关的核心类的职责划分：

```
vllm/core/
│
├── cache_engine.py           # CacheEngine: 管理 GPU/CPU 上的 Block 存储池
│   ├── gpu_cache: GPUCache  # GPU 端的 Block 存储（CUDA Tensor）
│   └── cpu_cache: CPUCache  # CPU 端的 Block 存储（用于 swap）
│
├── block_manager.py          # BlockSpaceManager: 统一的块空间管理
│   ├── allocate()             # 为新序列分配 Block
│   ├── can_allocate()         # 检查是否有足够空间
│   ├── free()                 # 释放序列的所有 Block
│   └── get_num_free_gpu_blocks()  # 查询空闲 Block 数
│
├── block_table.py            # BlockTable 数据结构
│   └── 存储每层的 block_table tensor
│
└── attention/backends/
    ├── flash_attn.py          # FlashAttention 后端（非分页模式）
    ├── paged_attn.py          # ★ PagedAttention 后端（本章重点）
    │   ├── PagedAttentionKernel  # CUDA Kernel 封装
    │   ├── forward()            # 前向传播入口
    │   └── 支持多种数据类型 (fp16/bf16/fp32/int8)
    └── torch_attn.py           # PyTorch 原生 attention（fallback）

vllm/worker/
│
└── model_runner.py
    ├── _add_kv_cache()        # 将新生成的 KV 写入对应 Block
    ├── _model_exec_wrapper()   # 包装 model forward，注入 Block Table
    └── execute_model()         # 每个 iteration 的主循环
```

### 核心调用链（一次推理 iteration）

```
Scheduler.schedule()
    │
    ▼ 决定本轮要执行的 sequences
    │
LLMEngine._run_engine_steps()
    │
    ▼ 对于每个 sequence group:
    │
ModelRunner.execute_model(scheduler_output)
    │
    ├─ 1. 准备输入
    │     └─ 根据 scheduler_output.seq_group.meta_data 中的 block_tables
    │        构建 input metadata（包含每层的 block_table tensor）
    │
    ├─ 2. Model Forward Pass
    │     └─ model.forward(
    │            input_ids,
    │            positions,
    │            kv_caches=block_tables,  ← ★ PagedAttention 在这里生效
    │            ...
    │        )
    │     其中每一层的 Attention:
    │     └─ PagedAttention.forward(
    │            query,
    │            key_cache,      # [num_blocks, kv_heads, block_size, head_dim]
    │            value_cache,
    │            block_tables,   # [batch, max_blocks_per_seq] 每行的 block table
    │            context_lens,   # [batch] 每个序列的有效长度
    │        )
    │
    ├─ 3. 采样得到 next_token
    │     └─ sampler(next_token_logits) → next_token
    │
    ├─ 4. 写回 KV Cache
    │     └─ model_runner._add_kv_cache(
    │            next_token,
    │            block_tables,    # 追加到对应 Block
    │            ...
    │        )
    │
    └─ 5. 返回 SamplerOutput
```

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **计算适配** | PagedAttention 用 index-based kernel 替代了连续张量的 matmul；通过 `block_table` 做间接寻址 |
| **Gather 操作** | 从分散的物理 Block 中按 block_table 收集 K/V，理论上需要拼接，实际上用 CUDA index 避免拷贝 |
| **性能代价** | 非连续访问导致 ~2-5% 的 kernel 效率损失，换来 30-40% 显存利用率的提升 |
| **Prefix Caching** | 多请求共享相同 System Prompt 时只存一份 KV Cache（通过 ref_count 共享 Block）；100 个共享请求可节省 33-90% Block |
| **Copy-on-Write** | 当共享 Block 需要被修改且 ref_count > 1 时触发 COW：分配新 Block → 复制内容 → 写入副本 |
| **源码结构** | `CacheEngine`(存储) → `BlockSpaceManager`(分配) → `PagedAttentionKernel`(计算) 三层协作 |
| **调用链** | Scheduler → Engine → ModelRunner → Model.forward → 各层 Attention → PagedAttention kernel |

> **一句话总结**：PagedAttention 的实现是一个精心设计的工程系统——Block 作为固定大小的原子单位消除了碎片化，Block Table 提供了灵活的非连续映射，Prefix Caching 让共享前缀零成本复用，Copy-on-Write 保护了共享数据的一致性，而自定义 CUDA Kernel 则在保持计算正确性的同时最小化了非连续访问的性能损失。这套组合拳使得 vLLM 能够在有限的 GPU 显存上实现前所未有的并发推理能力。
