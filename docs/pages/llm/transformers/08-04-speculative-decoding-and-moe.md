# 推测解码、MoE 与高级生成技术

## 这一节讲什么？

前面三节我们学习了从 logits 到文本的完整生成流程，以及各种解码策略和参数控制。但所有这些方法都有一个共同的瓶颈：**自回归生成的串行特性决定了它无法并行化——每个 token 都必须等前一个 token 生成完毕后才能开始计算**。

这一节我们将突破这个限制，学习几种能够显著提升生成速度的前沿技术：

1. **推测解码（Speculative Decoding）**——用小模型"打草稿"，大模型"审核"，实现 2~5x 加速
2. **Mixture of Experts (MoE)**——只激活部分专家网络，大幅减少每次推理的计算量
3. **非自回归生成**——一次性输出整个序列，彻底打破串行限制
4. **动态批处理（Continuous Batching）**——服务端的高吞吐优化策略

---

## 一、推测解码（Speculative Decoding）：小模型加速大模型

## 1.1 核心直觉

推测解码的思想非常优雅，可以用一个日常类比来理解：

> 你是一位资深编辑（大模型），你的实习生（小模型）先帮你草拟一篇文章。你不需要从头写起，只需要快速浏览实习生的草稿：**大部分内容直接采纳，只在不对的地方修改或重写**。这样你的工作效率就大幅提升了！

在 LLM 的语境下：
- **Draft Model（草稿模型）**：一个小而快的模型（如 GPT-2 small），快速生成 K 个候选 token
- **Target Model（目标模型）**：我们要用的大模型（如 LLaMA-70B），一次验证这 K 个 token
- **验证机制**：如果 Target Model 同意 Draft 的预测 → 全部保留（1 次前向传播得到 K 个 token）
- **回退机制**：如果在第 i 个位置不同意 → 丢弃 i 及之后的所有 token，从位置 i 重新采样

## 1.2 算法流程图解

```
输入: "The meaning of life is"
        ↓
    ┌─────────────────────────────┐
    │ Step 1: Draft Model (小型)   │
    │ 输入: "The meaning of life is"│
    │ 快速生成: "to find happiness"│ (K=4 个新 token)
    └──────────┬──────────────────┘
               ↓ 候选序列
    ┌─────────────────────────────┐
    │ Step 2: Target Model (大型)   │
    │ 输入: 完整候选序列            │
    │ 一次性前向传播                │
    │                              │
    │ 检查每个位置的预测一致性:      │
    │   pos 0: "to"     ✅ 匹配     │
    │   pos 1: "find"   ✅ 匹配     │
    │   pos 2: "happiness" ❌ 不匹配 │ ← 在这里停止!
    └──────────┬──────────────────┘
               ↓
    ┌─────────────────────────────┐
    │ Step 3: 接受 + 重新采样       │
    │ 接受: "to find" (2 tokens)   │ ← 用 1 次 Target 推理得到 2 个有效 token!
    │ 回退到 pos 2, 从 Target 的分布中重新采样
    └──────────┬──────────────────┘
               ↓
        继续下一轮...
```

## 1.3 手写推测解码实现

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpeculativeDecodingResult:
    """推测解码的结果"""
    accepted_tokens: int           # 本轮接受的 token 数
    total_draft_tokens: int       # 草稿模型生成的总 token 数
    target_forward_passes: int    # 目标模型的前向传播次数
    speedup_ratio: float          # 相比纯 Target 生成的加速比


class SpeculativeDecoder:
    """
    推测解码器
    
    使用一个小型 Draft Model 加速大型 Target Model 的推理
    """

    def __init__(
        self,
        draft_model,              # 小型草稿模型
        target_model,             # 大型目标模型
        tokenizer,
        draft_max_tokens: int = 5, # 草稿模型每次生成的最大 token 数
        max_speculative_tokens: int = 8, # 最大推测步数
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.draft_max_tokens = draft_max_tokens
        self.max_speculative_tokens = max_speculative_tokens
        
        self.device = next(target_model.parameters()).device
        self.draft_device = next(draft_model.parameters()).device

        # 将两个模型都设为 eval 模式
        draft_model.eval()
        target_model.eval()

    @torch.no_grad()
    def _generate_draft(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        用 Draft Model 生成候选 token 序列
        
        Returns:
            draft_ids: 草稿序列 (包含原始 input)
            draft_probs: 每个 position 的概率分布
        """
        current_ids = input_ids.clone()
        all_probs = []

        for _ in range(num_tokens):
            outputs = self.draft_model(current_ids)
            next_logits = outputs.logits[0, -1, :]
            probs = F.softmax(next_logits / 1.0, dim=-1)

            # 采样 (可以用 greedy 或 sampling)
            next_token = torch.multinomial(probs.unsqueeze(0), num_samples=1)[0]

            all_probs.append(probs)
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        draft_probs = torch.stack(all_probs)  # (num_drafted, vocab_size)
        return current_ids, draft_probs

    @torch.no_grad()
    def _verify_and_accept(
        self,
        original_input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, int]:
        """
        用 Target Model 验证草稿并决定接受多少个 token
        
        Returns:
            n_accepted: 接受的 token 数量
            sampled_token: 如果在某个位置不接受, 重新采样的 token
            accept_position: 第一个不被接受的位置 (-1 表示全部接受)
        """
        # Target Model 的一次性前向传播 (关键优化!)
        target_outputs = self.target_model(draft_ids)
        target_logits = target_outputs.logits[0]  # (seq_len, vocab_size)

        # 只检查新生成的部分 (排除原始 input)
        num_new_tokens = draft_ids.shape[1] - original_input_ids.shape[1]
        
        n_accepted = 0
        accept_position = -1
        sampled_token = None

        for i in range(num_new_tokens):
            pos = original_input_ids.shape[1] + i  # 在完整序列中的位置
            
            # Target Model 在该位置的预测
            target_probs = F.softmax(target_logits[pos], dim=-1)
            
            # Draft Model 在该位置的预测
            draft_prob = draft_probs[i]
            
            # 草稿选择的 token
            drafted_token = draft_ids[pos].item()

            # 验证: Target 是否也认为这是好选择?
            # 方法: 检查 drafted_token 在 target 分布中的概率是否足够高
            target_prob_of_drafted = target_probs[drafted_token].item()
            draft_prob_value = draft_prob[drafted_token].item()

            # 简单的接受准则: 两个模型的预测差异不大
            # 更严格的准则是使用精确的概率比值检验
            acceptance_threshold = 0.0  # 可以调整

            if target_prob_of_drafted > acceptance_threshold:
                # Target 也认可这个 token → 接受
                n_accepted += 1
            else:
                # Target 不认可 → 在此位置拒绝, 重新采样
                accept_position = i
                # 从 Target 的分布中重新采样
                sampled_token = torch.multinomial(
                    target_probs.unsqueeze(0), num_samples=1
                )[0]
                break

        return n_accepted, sampled_token, accept_position

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        verbose: bool = True,
    ) -> str:
        """
        执行推测解码生成
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        total_accepted = 0
        total_target_forwards = 0
        total_draft_tokens = 0

        if verbose:
            print("=" * 70)
            print("推测解码 (Speculative Decoding)")
            print("=" * 70)
            print(f"\nPrompt: '{prompt}'")
            print(f"Draft Model: {self.draft_model.config.model_type}")
            print(f"Target Model: {self.target_model.config.model_type}")

        while input_ids.shape[1] < inputs["input_ids"].shape[1] + max_new_tokens:

            # Step 1: Draft Model 生成草稿
            num_to_draft = min(self.draft_max_tokens, 
                               max_new_tokens - (input_ids.shape[1] - inputs["input_ids"].shape[1]))
            
            draft_ids, draft_probs = self._generate_draft(input_ids, num_to_draft)
            num_drafted = draft_ids.shape[1] - input_ids.shape[1]
            total_draft_tokens += num_drafted

            # Step 2: Target Model 验证
            n_accepted, resampled_token, reject_pos = self._verify_and_accept(
                input_ids, draft_ids, draft_probs
            )
            total_target_forwards += 1

            # Step 3: 构建新的输入序列
            if n_accepted == num_drafted:
                # 全部接受! 直接使用草稿结果
                input_ids = draft_ids
            else:
                # 部分接受: 接受前 n_accepted 个, 加上重新采样的 token
                accepted_part = draft_ids[:, :input_ids.shape[1] + n_accepted]
                if resampled_token is not None:
                    input_ids = torch.cat([accepted_part, resampled_token.unsqueeze(0).unsqueeze(0)], dim=1)
                else:
                    input_ids = accepted_part

            total_accepted += n_accepted

            if verbose:
                print(f"  [Round] Draft={num_drafted}, "
                      f"Accepted={n_accepted}/{num_drafted}, "
                      f"Total accepted so far={total_accepted}")

            # 检查 EOS
            last_token = input_ids[0, -1].item()
            if last_token == self.tokenizer.eos_token_id:
                break

        result = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if verbose:
            speedup = total_target_forwards / max(total_accepted, 1)  # 理想情况下的加速比
            print(f"\n✅ 生成完成!")
            print(f"   总接受 token: {total_accepted}")
            print(f"   目标模型前向次数: {total_target_forwards}")
            print(f"   草稿模型总生成: {total_draft_tokens}")
            print(f"   理论加速比: ~{speedup:.1f}x")

        return result


def demo_speculative_decoding():
    """演示推测解码"""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Draft model: 小型 GPT-2
    draft_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Target model: 同样用 GPT-2 演示 (实际应用中应该是更大的模型)
    target_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        draft_max_tokens=4,
    )

    result = decoder.generate(
        prompt="The future of artificial intelligence",
        max_new_tokens=40,
        verbose=True,
    )

    print(f"\n📝 最终输出:\n{result}")


if __name__ == "__main__":
    demo_speculative_decoding()
```

## 1.4 推测解码的效果分析

```python
def speculative_decoding_analysis():
    """推测解码的性能分析"""

    analysis = {
        "最佳场景": {
            "描述": "Draft 和 Target 的'意见一致率高'",
            "典型": "同一系列的不同规模模型 (如 LLaMA-7B + LLaMA-65B)",
            "加速比": "2~5x",
            "原因": "大部分草稿被直接接受, Target 只需少量回退",
        },
        " "一般场景": {
            "描述": "两个模型有一定差异但方向一致",
            "典型": "不同架构的模型 (GPT + LLaMA)",
            "加速比": "1.3~2x",
            "原因": "部分草稿需要修正, 但仍有不少直接接受",
        },
        "最差场景": {
            "描述": "Draft 和 Target 差异很大",
            "典型": "Draft 太弱或任务不匹配",
            "加速比": "< 1.2x (可能更慢!)",
            "原因": "大量回退导致 Draft 的计算白费了",
        },
    }

    print("=" * 75)
    print("推测解码性能分析")
    print("=" * 75)
    for scenario, info in analysis.items():
        print(f"\n📊 {scenario}")
        for k, v in info.items():
            print(f"   {k}: {v}")

speculative_decoding_analysis()
```

---

## 二、Mixture of Experts (MoE)：稀疏激活的大模型

## 2.1 从 Dense 到 Sparse 的思维转变

传统的大语言模型是**密集的（Dense）**——每一层、每一个神经元在处理每一个 token 时都会被激活。这意味着：

```
Dense Transformer (如 LLaMA-7B):
  Token "apple" → Layer 1 (全部激活) → Layer 2 (全部激活) → ... → Layer N (全部激活)
  
  参数量: 7B (全部参与计算)
  计算量: 7B × seq_len (每个 token 都触发全量计算)
```

MoE（混合专家模型）的核心思想是：**不是所有知识都需要用来处理每一个 token**。就像一家医院不需要每位医生都看同一个病人一样，我们可以让不同的"专家"处理不同类型的信息。

```
MoE Transformer (如 Mixtral-8x7B):
  Token "apple" → Router → 选择 Expert 2 和 Expert 5 → 只有这两个专家参与计算
  
  总参数量: 8×7B = 56B (看起来很大!)
  每次激活参数: ~12.9B (只有 2/8 的专家被激活)
  计算量: ~12.9B × seq_len (远小于 Dense 的 7B×seq_len? 不一定, 取决于实现)
```

## 2.2 MoE 架构详解

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExpertLayer(nn.Module):
    """单个专家层 (标准的 FFN)"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """路由器: 决定将每个 token 分配给哪些专家"""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            router_weights: (batch, seq_len, top_k) 归一化的权重
            expert_indices: (batch, seq_len, top_k) 选中的专家索引
        """
        router_logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Top-K 选择
        router_weights, expert_indices = torch.topk(
            F.softmax(router_logits.float(), dim=-1),
            self.top_k,
            dim=-1,
        )

        # 重新归一化权重
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        return router_weights, expert_indices


class MoELayer(nn.Module):
    """
    Mixture of Experts 层
    替代 Transformer 中的标准 FFN 层
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()

        self.router = Router(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertLayer(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE 前向传播
        
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape

        # Step 1: 路由决策
        router_weights, expert_indices = self.router(x)
        # router_weights: (batch, seq_len, top_k)
        # expert_indices: (batch, seq_len, top_k)

        # Step 2: 初始化输出
        output = torch.zeros_like(x)

        # Step 3: 对每个专家, 处理分配给它的 token
        for expert_idx, expert in enumerate(self.experts):

            # 找出哪些位置选择了当前专家
            expert_mask = (expert_indices == expert_idx)  # (batch, seq_len, top_k)

            if expert_mask.any():
                # 收集所有分配给当前专家的 token
                flat_mask = expert_mask.any(dim=-1)  # (batch, seq_len)
                indices = torch.where(flat_mask)  # 需要处理的 (batch, seq) 位置
                
                if len(indices[0]) > 0:
                    selected_tokens = x[indices]  # (n_selected, hidden)
                    
                    # 专家处理
                    expert_output = expert(selected_tokens)  # (n_selected, hidden)
                    
                    # 获取对应的路由权重
                    weights = router_weights[indices]  # (n_selected, top_k)
                    # 找到当前专家对应的权重
                    weight_mask = expert_mask[indices]  # (n_selected, top_k)
                    final_weights = weights.where(weight_mask, torch.zeros_like(weights))
                    final_weights = final_weights.sum(dim=-1)  # (n_selected,)
                    
                    # 加权累加到输出
                    output.index_put_(
                        indices,
                        expert_output * final_weights.unsqueeze(-1),
                        accumulate=True,
                    )

        return output


def demo_moe_architecture():
    """演示 MoE 架构"""

    print("=" * 70)
    print("Mixture of Experts (MoE) 架构演示")
    print("=" * 70)

    moe_layer = MoELayer(
        hidden_size=256,         # 缩小的隐藏维度用于演示
        intermediate_size=512,
        num_experts=4,           # 4 个专家
        top_k=2,                 # 每个token 激活 2 个专家
    )

    # 模拟一个 batch 的输入
    x = torch.randn(2, 10, 256)  # (batch=2, seq_len=10, hidden=256)

    output = moe_layer(x)

    total_params = sum(p.numel() for p in moe_layer.parameters())
    expert_params = sum(p.numel() for p in moe_layer.experts.parameters())

    print(f"\n📐 MoE 配置:")
    print(f"   隐藏维度: {256}")
    print(f"   专家数量: {moe_layer.num_experts}")
    print(f"   每次激活专家数: {moe_layer.top_k}")
    print(f"   总参数量: {total_params:,}")
    print(f"   专家层参数: {expert_params:,} ({expert_params/total_params*100:.1f}%)")

    print(f"\n🔢 输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")

    # 分析路由分布
    with torch.no_grad():
        weights, indices = moe_layer.router(x)

    print(f"\n📊 路由分析 (前 3 个 token):")
    for t in range(min(3, x.shape[1])):
        selected = indices[0, t].tolist()
        w = weights[0, t].tolist()
        experts_str = ", ".join([f"E{e}({w_i:.3f})" for e, w_i in zip(selected, w)])
        print(f"   Token {t}: → [{experts_str}]")

demo_moe_architecture()
```

## 2.3 Load Balancing Loss：防止专家坍塌

MoE 有一个著名的训练问题：**专家坍塌（Expert Collapse）**。如果路由器发现某个专家效果特别好，它可能会把所有 token 都发给那个专家，其他专家永远得不到训练。

解决方案是在损失函数中加入 **Load Balancing Loss**：

$$\mathcal{L}_{balance} = N \times \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是专家 $i$ 被选中的比例，$P_i$ 是路由器分配给专家 $i$ 的概率比例。这个损失鼓励均匀分配。

```python
def load_balancing_loss_demo():
    """Load Balancing Loss 演示"""

    import numpy as np

    scenarios = [
        ("理想均衡", [0.25, 0.25, 0.25, 0.25]),
        ("轻微偏斜", [0.35, 0.28, 0.22, 0.15]),
        ("严重坍塌", [0.85, 0.10, 0.03, 0.02]),
    ]

    print("=" * 60)
    print("Load Balancing Loss 示例")
    print("=" * 60)

    for name, distribution in scenarios:
        f = np.array(distribution)  # 实际分配比例
        p = np.array(distribution)  # 路由概率 (简化: 假设相等)
        loss = len(distribution) * np.sum(f * p)

        bar = "█" * int(loss * 50)
        status = "✅" if loss < 1.5 else ("⚠️" if loss < 3.0 else "❌")

        print(f"\n[{status} {name}] Loss = {loss:.4f}")
        print(f"   分布: {distribution}")
        print(f"   {' ' * 6}{bar}")

load_balancing_loss_demo()
```

---

## 三、非自回归生成：打破串行限制

## 3.1 自回归的根本瓶颈

传统 LLM 的自回归生成有一个根本性的限制：

$$T_{total} = \sum_{t=1}^{L} T_{forward}(t) \approx L \times T_{forward\_avg}$$

生成 L 个 token 需要 L 次前向传播，**无法并行化**。

## 3.2 非自回归方法的探索

```python
def non_autoregressive_approaches():
    """非自回归生成方法概览"""

    approaches = [
        {
            "方法": "迭代精炼 (Iterative Refinement)",
            "原理": "先生成完整的粗糙序列, 然后多轮逐步精炼",
            "代表": "BERT-based Seq2Seq, Insertion Transformer",
            "优点": "可以并行生成初始序列",
            "缺点": "多轮精炼的总开销可能更大; 存在 multimodality 问题",
        },
        {
            "方法": "完全并行 (Fully Parallel)",
            "原理": "一次性输出整个序列的所有位置",
            "代表": "CAT (Covering Auto-encoder for Translation)",
            "优点": "理论上的 O(1) 延迟",
            "缺点": "质量通常不如自回归; 长序列上问题严重",
        },
        {
            "方法": "混合方法 (Hybrid)",
            "原理": "先用非自回归快速出初稿, 再用自回归精炼",
            "代表": "Medusa, EAGLE, Lookahead Decoding",
            "优点": "结合两者优势; 实际加速效果好",
            "缺点": "实现复杂; 需要额外的训练或适配",
        },
        {
            "方法": "推测解码 (Speculative Decoding)",
            "原理": "小模型草稿 + 大模型验证",
            "代表": "DeepMind Speculative Decoding, NVIDIA Medusa",
            "优点": "无需修改模型; 即插即用",
            "缺点": "依赖 Draft 和 Target 的一致性",
        },
    ]

    print("=" * 80)
    print("非自回归生成方法对比")
    print("=" * 80)

    for a in approaches:
        print(f"\n🚀 {a['方法']}")
        print(f"   原理: {a['原理']}")
        print(f"   代表: {a['代表']}")
        print(f"   ✅ 优点: {a['优点']}")
        print(f"   ❌ 缺点: {a['缺点']}")

non_autoregressive_approaches()
```

---

## 四、Continuous Batching：服务端的高吞吐策略

## 4.1 Static Batching vs Continuous Batching

传统的静态批处理（Static Batching）有严重的效率问题：

```
Static Batching (静态批处理):
  Batch 1: [Request A (50 steps), Request B (20 steps)]
  → Request A 还没完, 但 Request B 已经完成了
  → Request B 必须等待 A (资源浪费!)

  时间线:
  |A: ███████████████████████████████████████████████████|
  |B: ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|  ← 空闲等待!

Continuous Batching (连续/动态批处理):
  Request B 完成后立即移除, 放入 Request C:
  |A: ███████████████████████████████████████████████████|
  |B: ██████████████████|
  |C:                   |███████████████████████████████████|

  GPU 利用率大幅提升!
```

## 4.2 vLLM 和 PagedAttention

vLLM 是目前最流行的 LLM 推理引擎之一，它的核心创新是 **PagedAttention**——借鉴操作系统的虚拟内存分页思想来管理 KV Cache：

```python
def explain_pagedattention():
    """PagedAttention 原理解释"""

    explanation = """
传统 KV Cache 管理:
┌─────────────────────────────────────────────────────┐
│ Request A: [Block 1024 tokens, 连续内存]            │
│ Request B: [Block 512 tokens, 连续内存]             │
│                                                     │
│ 问题:                                               │
│ 1. 必须预分配最大长度的连续内存 (浪费!)              │
│ 2. 内存碎片严重 (请求完成后释放的空间难以复用)        │
│ 3. 无法支持超过预设最大长度的请求                     │
└─────────────────────────────────────────────────────┘

PagedAttention (vLLM 的方案):
┌─────────────────────────────────────────────────────┐
│ 物理内存: 分成固定大小的 Page (如每页 16 tokens)     │
│                                                     │
│ Page 1: [Req A tokens 0-15]                         │
│ Page 2: [Req A tokens 16-31]                        │
│ Page 3: [Req B tokens 0-15]                         │
│ Page 4: [Req B tokens 16-31]                        │
│ ...                                                 │
│                                                     │
│ 优势:                                               │
│ 1. 按需分配 Page (初始只需几个页)                    │
│ 2. Page 可以不连续 (像 OS 的虚拟内存)                 │
│ 3. 请求释放的 Page 可立即复用                       │
│ 4. 理论上支持任意长度 (动态添加 Page)               │
└─────────────────────────────────────────────────────┘
"""
    print(explanation)

explain_pagedattention()
```

---

## 五、本章小结

这一节我们学习了多种高级生成技术，它们从不同角度解决"生成太慢"的问题：

| 技术 | 核心思想 | 加速方式 | 适用场景 |
|------|---------|---------|---------|
| **推测解码** | 小模型草稿 + 大模型验证 | 减少大模型调用次数 | 有可用的小模型时 |
| **MoE** | 稀疏激活专家子集 | 减少每次计算量 | 大规模模型训练/部署 |
| **非自回归** | 并行/迭代生成 | 打破串行限制 | 研究前沿，生产慎用 |
| **Continuous Batching** | 动态调度请求 | 提高 GPU 利用率 | 高并发在线服务 |
| **PagedAttention** | 分页管理 KV Cache | 减少显存碎片 | vLLM/TGI 内置 |

**核心要点**：
1. **推测解码是最实用的即插即用加速方案**——不需要修改模型，只需准备一个 Draft Model
2. **MoE 是大模型架构的重要趋势**——Mixtral、DeepSeek-MoE、Grok 等都在采用
3. **生产环境推荐使用 vLLM 或 TGI**——它们内置了 Continuous Batching + PagedAttention
4. **没有免费的午餐**——每种加速技术都有其适用范围和权衡

下一节我们将学习 **自定义生成策略**——如何通过 Logits Processor 和 Guided Decoding 实现对生成过程的精细控制。
