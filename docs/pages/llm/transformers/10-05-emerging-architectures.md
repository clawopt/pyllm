# 10-05 新兴架构

## 为什么需要新架构？

如果你一路读到这里，已经对 Transformer 的注意力机制、位置编码、各种变体（从原始 Transformer 到 ViT、从 BERT 到 GPT、从 LoRA 到 QLoRA）有了相当深入的理解。那么你可能会问：Transformer 已经这么强大了，为什么还需要研究新的架构？答案藏在几个越来越明显的瓶颈里。

第一个瓶颈是**序列长度的二次方复杂度**。标准自注意力的计算量是 `O(n²)`，其中 n 是序列长度。当 n=2048 时这还好，但当 n=128K 甚至更长时（比如处理整本书、长视频、基因组序列），这个二次方开销就变得难以承受了。虽然 FlashAttention 和各种稀疏注意力方案在一定程度上缓解了这个问题，但它们本质上还是在"打补丁"——架构本身并没有改变。

第二个瓶颈是**推理时的 KV Cache 内存增长**。每生成一个新 token，就需要把之前的所有 KV 缓存都保留下来，这意味着内存占用随着生成长度线性增长。对于一个 7B 模型生成 8K token 的场景，仅 KV Cache 就可能占用数 GB 显存。

第三个瓶颈是**对"无限上下文"的真正需求**。很多实际应用（法律文档分析、代码库理解、长期对话记忆）需要的不是"很长的固定窗口"，而是真正的"可以一直记住之前所有内容"的能力。

这些瓶颈催生了几个非常有前景的新架构方向，其中最引人注目的就是 **Mamba（状态空间模型）**、**RWKV**、以及 **混合架构（如 Jamba = Mamba + Transformer）**。下面我们逐一深入。

## Mamba 与状态空间模型（SSM）

## 从 RNN 到 SSM：一条被重新发现的路

要理解 Mamba，我们需要先回到一个更基础的问题：如何建模序列数据？在 Transformer 统治 NLP 之前，RNN（循环神经网络）是主流方案。RNN 的核心思想非常直觉——维护一个隐藏状态 `h_t`，每来一个新的输入 `x_t`，就更新这个状态并产生输出：

```
h_t = f(h_{t-1}, x_t)    # 状态更新
y_t = g(h_t)              # 输出
```

RNN 的优点是推理时内存恒定（`O(1)`），因为只需要保留当前状态；缺点是难以并行训练（第 t 步必须等 t-1 步完成）和长程依赖问题（梯度消失/爆炸）。Transformer 用自注意力完美解决了这两个问题，但引入了 `O(n²)` 的计算复杂度和线性增长的 KV Cache。

**状态空间模型（State Space Models, SSM）** 试图结合两者的优势。它的数学形式看起来像是一个连续系统：

```
h'(t) = A·h(t) + B·x(t)    # 状态微分方程
y(t)   = C·h(t) + D·x(t)    # 输出方程
```

经过离散化（用零阶保持 ZOH 或双线性变换）后，变成了可计算的递推形式：

```
h_t = Ā·h_{t-1} + B̄·x_t     # 离散化后的状态更新
y_t = C·h_t + D·x_t          # 输出
```

这看起来和 RNN 非常像！但关键区别在于参数 A、B、C 的初始化和学习方式——特别是 Mamba 引入了**输入依赖的动态参数**，让模型可以根据当前输入自适应地调整状态更新行为。

```python
import torch
import torch.nn as nn
import math


class S4Layer(nn.Module):
    """
    结构化状态空间模型（S4）的基础实现
    
    S4 是 SSM 家族中最重要的先驱工作之一。
    它的核心创新在于使用 HiPPO（High-order Polynomial 
    Projection Operators）矩阵来初始化状态转移矩阵 A，
    使得模型天然具有长程记忆能力。
    
    数学原理:
    ─────────
    连续形式: h'(t) = Ah(t) + Bx(t), y(t) = Ch(t) + Dx(t)
    
    离散化 (ZOH): 
      Ā = exp(ΔA)       Δ 是步长（可学习或固定）
      B̄ = (ΔA)^{-1}(exp(ΔA) - I) · ΔB
    
    递推计算:
      h_t = Ā·h_{t-1} + B̄·x_t
      y_t = C·h_t + D·x_t
    """
    
    def __init__(
        self,
        d_model: int,        # 模型维度
        d_state: int = 64,   # 状态维度（越大，记忆能力越强）
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 投影层：将输入投影到 SSM 参数空间
        self.in_proj = nn.Linear(d_model, 3 * d_model)  # 产出 x, B, C
        
        # 状态转移矩阵 A: (d_state, d_state)
        # 使用 HiPPO 初始化 - 这是 S4 的核心！
        A = self._hippo_matrix(d_state)
        self.A_log = nn.Parameter(torch.log(A))  # 以 log 形式存储保证正定/稳定
        
        # 步长 Δ: (d_model,) — 控制离散化的粒度
        self.D_log = nn.Parameter(torch.randn(d_model))
        
        # D: 跳跃连接 (d_model,)
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.dropout = nn.Dropout(dropout)
    
    def _hippo_matrix(self, N: int) -> torch.Tensor:
        """
        HiPPO 矩阵初始化
        
        HiPPO (High-order Polynomial Projection Operator) 的思想是：
        让状态空间模型能够最优地"记住"历史输入的多项式系数。
        
        对于长度为 L 的序列，HiPPO 矩阵使得前 k 个状态能够
        完美重建前 k 个 Legendre 多项式的系数。
        
        这意味着 SSM 天然具有对不同时间尺度的信息的编码能力。
        """
        P = torch.arange(1, N + 1, dtype=torch.float32)
        A = torch.outer(P, P)  # A[i,j] = (i+1)(j+1)
        
        # HiPPO 的特定结构
        A = -A + torch.diag(P)  # 对角线加上衰减项
        A = A / (N + 1)         # 归一化
        
        return A
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        
        # 投影得到 x_proj, B_proj, C_proj
        uBC = self.in_proj(x)  # (B, L, 3D)
        x_proj, B_proj, C_proj = uBC.chunk(3, dim=-1)
        
        # 解析 A 和 Δ（从 log 空间）
        A = -torch.exp(self.A_log)  # 负值保证稳定性
        Delta = torch.exp(self.D_log)  # (D,)
        
        # 离散化: ZOH 方法
        # Ā = exp(ΔA), B̄ = (exp(ΔA)-I)/(ΔA) * ΔB
        DA = Delta.unsqueeze(-1) * A.unsqueeze(0)  # (B, D, N)
        A_bar = torch.matrix_exp(DA)  # (B, D, N, N) — 这里简化为对角假设
        
        # 实际实现中 A 通常是对角矩阵（diagonal SSM）
        # 这样 exp(ΔA) 就是逐元素操作，效率极高
        dA = torch.exp(Delta.unsqueeze(-1) * A)  # (D, N) — 对角化版本
        dB = (dA - 1) / (Delta.unsqueeze(-1) * A + 1e-7) * Delta.unsqueeze(-1) * B_proj.transpose(-1, -2)  # (D, N, D)
        
        # 递推计算（RNN 模式 — 推理时用这个，O(L) 时间，O(N) 内存）
        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        
        for l in range(L):
            # h_t = Ā·h_{t-1} + B̄·x_t
            # 注意: 这里的矩阵乘法在对角假设下简化为逐元素操作
            h = dA * h + (B_proj[:, l:l+1, :].transpose(-1, -2) * x_proj[:, l:l+1, :])
            
            # y_t = C·h_t + D·x_t
            y = torch.einsum('bdn,bdn->bd', C_proj[:, l, :, :], h)
            y = y + self.D * x_proj[:, l, :]
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)  # (B, L, D)
        output = self.dropout(output)
        
        return output


def ssm_intuition_demo():
    """
    SSM 直觉演示: 展示 SSM 如何自然地处理不同尺度的信息
    """
    
    print("=" * 60)
    print("SSM 核心直觉演示")
    print("=" * 60)
    
    # 模拟一个简单的 1D SSM
    # 设 A = 0.9（接近1但小于1，表示缓慢衰减的长程记忆）
    # 设 B = 1.0, C = 1.0, D = 0
    
    A_val = 0.9
    B_val = 1.0
    C_val = 1.0
    
    # 输入信号: 一个脉冲 + 后续的零
    input_signal = [5.0] + [0.0] * 49  # 第一步给一个脉冲，后面都是零
    
    # SSM 递推
    h = 0.0
    outputs = []
    
    print("\n脉冲响应演示 (A=0.9):")
    print("步\t输入\t状态h\t输出y")
    print("-" * 35)
    
    for t, x in enumerate(input_signal):
        h = A_val * h + B_val * x  # 状态更新
        y = C_val * h               # 输出
        outputs.append(y)
        
        if t < 15 or abs(y) > 0.01:  # 只打印有意义的部分
            print(f"{t}\t{x:.1f}\t{h:.4f}\t{y:.4f}")
    
    print(f"... (共 {len(input_signal)} 步)")
    print(f"\n关键观察:")
    print(f"  • 即使输入在第1步后就变为0，输出仍然持续了很长时间")
    print(f"  • 这是因为状态 h 在'记忆'过去的输入（以速率 A^t 衰减）")
    print(f"  • A 越接近 1，记忆越长；A 越小，遗忘越快")
    print(f"  • 这就是 SSM 天然具有长程依赖建模能力的原理")


ssm_intuition_demo()
```

运行上面的直觉演示你会看到一个非常优雅的性质：即使输入只在一个时刻非零，SSM 的输出会在之后很长一段时间内持续响应——这就是"状态记忆"的直观表现。参数 A 控制着记忆的衰减速度：A=0.9 意味着每一步状态保留 90% 的信息，遗忘 10%；A=0.99 则意味着极慢的衰减，对应超长程的记忆能力。

## Mamba：选择性状态空间模型

S4 虽然很有趣，但它有一个根本性的局限：**参数 A、B、C、Δ 都是全局共享的**，不依赖于具体输入。这意味着无论当前输入是什么，模型都用相同的方式更新状态——这在某些场景下是合理的（比如对整个序列做统一的特征提取），但在需要"选择性关注"的场景下就不够灵活了。

**Mamba** 的核心创新正是引入了**输入依赖的动态参数**。它让 Δ（离散化步长）、B、C 这些关键参数都可以根据当前输入 x_t 动态计算出来：

```python
class MambaBlock(nn.Module):
    """
    Mamba Block 完整实现概念
    
    Mamba 相对于 S4/S5 的关键改进：
    1. 输入依赖的动态参数: B, C, Δ 都由输入 x 通过线性层动态生成
    2. 选择性机制: 模型可以决定"记住什么"和"忘记什么"
    3. 硬件感知的高效实现: 使用 IO-aware 的并行扫描算法
    
    架构流程:
    ┌──────┐     ┌──────────┐     ┌─────────┐     ┌────┐
    │  x_t  │────▶│ In_Proj  │────▶│ Conv1D  │────▶│ SiLU│
    └──────┘     └──────────┘     └─────────┘     └────┘
                                                      │
                   ┌──────────────────────────────────┘
                   ▼
           ┌───────────────┐
           │  动态参数生成   │
           │  Δ_t = proj_Δ(z_t) │
           │  B_t = proj_B(z_t) │
           │  C_t = proj_C(z_t) │
           └───────┬───────┘
                   ▼
           ┌───────────────┐
           │   SSM 核心     │
           │  h_t = Ā_t·h_{t-1} + B̄_t·x_t │
           │  y_t = C_t·h_t        │
           └───────┬───────┘
                   ▼
           ┌───────────────┐
           │   Out_Proj    │
           └───────┬───────┘
                   ▼
                 输出
    """
    
    def __init__(
        self,
        d_model: int = 2560,
        d_state: int = 16,      # Mamba 的状态维度通常较小
        d_conv: int = 4,        # 卷积核大小
        expand: int = 2,        # 内部扩展倍数
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_dim = d_model * expand
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.expand_dim * 2)
        
        # 卷积层（用于局部特征提取 + 打破逐token处理的限制）
        self.conv1d = nn.Conv1d(
            in_channels=self.expand_dim,
            out_channels=self.expand_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expand_dim,  # 深度卷积
        )
        
        # ★★★ 核心创新: 输入依赖的动态参数投影 ★★★
        # 这些层根据当前输入 z_t 动态生成 SSM 参数
        self.x_proj = nn.Linear(self.expand_dim, d_state * 2 + 1)  # → (B, C, Δ)
        self.dt_proj = nn.Linear(d_state, self.expand_dim)         # Δ 投影
        
        # SSM 参数: A 矩阵（仍然是固定的/半固定的）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A))  # (1, d_state)
        
        # D: 跳跃连接
        self.D = nn.Parameter(torch.ones(self.expand_dim))
        
        # 输出投影
        self.out_proj = nn.Linear(self.expand_dim, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        
        # 输入投影 + SiLU 激活
        xz = self.in_proj(x)  # (B, L, 2*expand_dim)
        x_proj, z = xz.chunk(2, dim=-1)  # 各 (B, L, expand_dim)
        
        # 卷积 + SiLU（局部特征增强）
        x_conv = x_proj.transpose(1, 2)  # (B, expand_dim, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # 截断到原始长度
        x_conv = x_conv.transpose(1, 2)  # (B, L, expand_dim)
        x_conv = torch.nn.functional.silu(x_conv)
        
        # ★ 动态参数生成 ★
        x_dbl = self.x_proj(x_conv)  # (B, L, d_state*2+1)
        delta, B_ssm, C_ssm = x_dbl.split([self.d_state, self.d_state, 1], dim=-1)
        delta = torch.nn.functional.softplus(self.dt_proj(delta.squeeze(-1)))  # 正值约束
        
        # SSM 核心（简化版，实际 Mamba 使用高效的并行扫描算法）
        A = -torch.exp(self.A_log)  # (1, d_state)
        y = self._ssm_recursive(x_conv, A, B_ssm, C_ssm, delta)
        
        # 门控: y * silu(z)
        y = y * torch.nn.functional.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output
    
    def _ssm_recursive(
        self,
        u: torch.Tensor,      # (B, L, expand_dim)
        A: torch.Tensor,      # (1, d_state)
        B: torch.Tensor,      # (B, L, d_state)
        C: torch.Tensor,      # (B, L, d_state)
        delta: torch.Tensor,  # (B, L, expand_dim)
    ) -> torch.Tensor:
        """简化的 SSM 递推计算（教学用途，生产环境应使用并行扫描）"""
        B, L, D = u.shape
        N = A.shape[-1]
        
        # 离散化
        dA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, N)
        dB = delta.unsqueeze(-1) * B             # (B, L, N)
        
        # 递推
        h = torch.zeros(B, D, N, device=u.device)
        ys = []
        
        for i in range(L):
            ui = u[:, i, :]  # (B, D)
            di = dA[:, i, :]  # (B, N)
            dbi = dB[:, i, :] # (B, N)
            ci = C[:, i, :]   # (B, N)
            
            h = di * h + dbi.unsqueeze(1) * ui.unsqueeze(-1)  # (B, D, N)
            y_i = (ci * h).sum(dim=-1)  # (B, D)
            ys.append(y_i)
        
        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + u * self.D  # 跳跃连接
        
        return y


class MambaModel(nn.Module):
    """
    完整的 Mamba 模型（类似 GPT 的堆叠结构）
    
    Mamba 可以直接替换 Transformer 作为语言模型的骨干网络
    """
    
    def __init__(
        self,
        vocab_size: int = 50277,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 堆叠 Mamba Block（类似 Transformer 的 Encoder Layer）
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])
        
        # 归一化
        self.norm = nn.RMSNorm(d_model)
        
        # 输出头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定（减少参数量）
        self.lm_head.weight = self.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        注意: Mamba 不需要 attention_mask!
        因为它是基于 RNN 式的递推，天然按顺序处理
        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


MambaModel
```

Mamba 的设计中有几个值得深思的细节：

**为什么需要卷积层？** 纯 SSM 是逐 token 处理的，每个 token 只能看到自己。加入一个小的 1D 卷积（kernel_size 通常为 4）后，每个位置的输出会融合邻近几步的信息，这相当于给 SSM 加上了"局部感受野"，类似于 CNN 中的做法。实验表明这个小改动能显著提升性能。

**为什么 Δ 要通过 softplus？** Δ 控制 SSM 的离散化粒度，物理意义上必须是正数。softplus(x) = log(1 + e^x) 是一个平滑的非负激活函数，比 ReLU 更适合这里（ReLU 在 0 点不可微且梯度可能不稳定）。

**门控机制 y * silu(z) 的作用？** 这借鉴了 GLU（Gated Linear Unit）的设计思路。z 分支可以看作是"门控信号"，控制哪些信息应该通过、哪些应该被抑制。这让 Mamba 具备了一定的选择性注意力能力——虽然不如 Transformer 的精确注意力那样灵活，但在大多数任务上已经足够好。

## Mamba vs Transformer：何时选哪个？

```python
def mamba_vs_transformer_comparison():
    """
    Mamba vs Transformer 全面对比
    
    这个对比表可以帮助你在实际项目中选择合适的架构
    """
    
    comparison = {
        "维度": {
            "训练并行性": {
                "Transformer": "★★★★★ 完全并行（自注意力一次算完所有位置）",
                "Mamba": "★★☆☆☆ 训练时可用并行扫描算法，但不如 Attention 成熟",
                "优势方": "Transformer",
            },
            "推理效率（短序列）": {
                "Transformer": "★★★☆☆ 需要 KV Cache，每次生成都要 O(n) 计算",
                "Mamba": "★★★★★ O(1) 内存 + O(1) 每步计算，恒定开销",
                "优势方": "Mamba",
            },
            "推理效率（长序列 > 10K）": {
                "Transformer": "★☆☆☆☆ KV Cache 爆炸，显存成为瓶颈",
                "Mamba": "★★★★★ 依然 O(1)，无 KV Cache 问题",
                "优势方": "Mamba（压倒性优势）",
            },
            "长程依赖建模": {
                "Transformer": "★★★★★ 自注意力直接建模任意距离的关系",
                "Mamba": "★★★★☆ 通过状态传递间接建模，理论上有信息压缩损失",
                "优势方": "Transformer（略优）",
            },
            "上下文外推能力": {
                "Transformer": "★★☆☆☆ 受限于训练时的位置编码/窗口大小",
                "Mamba": "★★★★★ 无硬性长度限制，理论上支持无限长度",
                "优势方": "Mamba",
            },
            "指令跟随 / 复杂推理": {
                "Transformer": "★★★★★ 在复杂推理任务上经验证的最强架构",
                "Mamba": "★★★☆☆ 快速追赶中，但大型 Mamba 模型仍较少",
                "优势方": "Transformer（目前）",
            },
            "生态成熟度": {
                "Transformer": "★★★★★ HuggingFace 全套支持，海量预训练权重",
                "Mamba": "★★★☆☆ 官方实现可用，HF 集成进行中",
                "优势方": "Transformer",
            },
            "参数效率": {
                "Transformer": "★★★☆☆ 注意力模块参数量大（Q/K/V/O 投影）",
                "Mamba": "★★★★★ 更紧凑，同参数量下通常效果更好",
                "优势方": "Mamba",
            },
            "吞吐量（tokens/sec）": {
                "Transformer": "★★★☆☆ 受限于 KV Cache 大小",
                "Mamba": "★★★★★ 无 KV Cache，高吞吐潜力大",
                "优势方": "Mamba",
            },
        }
    }
    
    print("=" * 75)
    print("Mamba vs Transformer 决策指南")
    print("=" * 75)
    
    for dimension, data in comparison["维度"].items():
        print(f"\n【{dimension}】")
        for name, value in data.items():
            if name != "优势方":
                print(f"  {name}: {value}")
            else:
                print(f"  ➜ {name}: {value}")
    
    # 场景推荐
    recommendations = """

【场景推荐】

✅ 选 Mamba 的场景:
  • 需要处理超长文本（>32K tokens）：法律文档分析、全书摘要、基因序列
  • 对推理延迟敏感的在线服务：实时对话、边缘设备部署
  • 显存受限的环境：消费级 GPU 上运行大模型
  • 需要高吞吐量的批处理任务

✅ 选 Transformer 的场景:
  • 复杂的多轮推理任务：数学证明、代码生成、逻辑链推理
  • 需要利用现有生态和预训练权重的快速原型开发
  • 多模态融合任务（目前多模态模型大多基于 Transformer）
  • 需要精细的注意力控制的任务（如特定位置的聚焦）

✅ 选混合架构（Jamba 等）的场景:
  • 既想要 Mamba 的长文本效率，又想要 Transformer 的推理能力
  • 愿意承担稍复杂的架构设计和工程实现成本
"""
    print(recommendations)


mamba_vs_transformer_comparison()
```

## RWKV：结合 RNN 效率与 Transformer 表达力

**RWKV（Receptance Weighted Key Value）** 是另一个试图打破 Transformer 局限的重要尝试。它的名字本身就揭示了核心思想：把 Transformer 中 Q/K/V 的概念改造成一种可以在 RNN 模式下高效运行的形态。

## RWKV 的核心公式

RWKV 的每个 token 位置的计算可以写成如下形式（以第 t 个位置为例）：

```
r_t = W_r · x_t + b_r          # Receptance（接收门控）
k_t = W_k · x_t + b_k          # Key
v_t = W_v · x_t + b_v          # Value
w_t = exp(W_w · x_t + b_w)     # 权重（指数衰减）

o_t = r_t ⊙ (Σ_{i=1}^{t} w_{t-i}^{(t)} ⊙ k_i ⊙ v_i) / (Σ_{i=1}^{t} w_{t-i}^{(t)} ⊙ k_i)
```

这个公式看起来有点复杂，但拆解后会变得清晰：

```python
class RWKVTokenMixer(nn.Module):
    """
    RWKV Token Mixer（Time-Mixing 版本）
    
    RWKV 的核心创新: 将注意力中的 Q/K/V 替换为 R/W/K/V 四个向量，
    其中 W（Weight）向量的指数形式实现了类似"衰减记忆"的效果，
    使得整个运算可以用 RNN 方式递推计算。
    
    关键洞察:
    ─────────
    标准 Attention:  attn(q_t, k_i) = softmax(q_t · k_i / √d)
    RWKV:           attn(r_t, k_i, v_i, w_i) = (w_i · k_i · v_i) / (w_i · k_i)
    
    区别:
    1. RWKV 没有 query！而是用 receptance r_t 做"接收门控"
    2. 分母只有 k（没有 q），避免了 softmax 的全局归一化
    3. w 提供了位置相关的衰减权重
    """
    
    def __init__(self, embed_dim: int, head_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size
        num_heads = embed_dim // head_size
        
        # R/W/K/V 四个线性投影
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weight = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key = nn.Linear(embed_dim, embed_dim, bias=True)
        self.value = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # 时间衰减参数（可学习，初始化为负值使初始衰减较快）
        self.time_decay = nn.Parameter(torch.randn(num_heads))
        self.time_first = nn.Parameter(torch.randn(num_heads))
    
    def forward(
        self,
        x: torch.Tensor,
        state: tuple = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            state: 上一步的 RNN 状态 (numerator, denominator, max_length)
                  用于推理时的增量计算
        
        Returns:
            output: (batch, seq_len, embed_dim)
            new_state: 更新后的状态（用于下一步）
        """
        B, T, C = x.shape
        
        # 计算 R/W/K/V
        r = torch.sigmoid(self.receptance(x))  # (B,T,C) — sigmoid 门控
        k = self.key(x)                         # (B,T,C)
        v = self.value(x)                       # (B,T,C)
        w = torch.exp(-torch.exp(self.weight))  # (B,T,C) — 双重exp确保正值+衰减
        
        # === 训练模式: 并行计算（类似 Transformer）===
        if state is None:
            output = self._parallel_forward(r, k, v, w, T)
            new_state = None
        else:
            # === 推理模式: RNN 递推计算（O(1) 每步）===
            output, new_state = self._rnn_forward(r, k, v, w, state)
        
        output = self.output_proj(output)
        return output, new_state
    
    def _parallel_forward(
        self, r, k, v, w, T
    ) -> torch.Tensor:
        """并行模式（训练时使用）— 类似 Transformer Attention 的并行计算"""
        # 简化的并行 RWKV 计算
        # 实际实现需要使用自定义 CUDA kernel 来高效计算累积加权
        # 这里展示概念流程
        
        # 逐位置计算加权和
        ws = w.cumsum(dim=1)  # 衰减权重的累积
        
        wk = (w * k).cumsum(dim=1)  # 加权 key 的累积
        wkv = (w * k * v).cumsum(dim=1)  # 加权 key*value 的累积
        
        # RWKV 输出
        output = r * (wkv / (wk + 1e-8))  # 避免除零
        
        return output
    
    def _rnn_forward(
        self, r, k, v, w, state
    ) -> tuple[torch.Tensor, tuple]:
        """RNN 模式（推理时使用）— O(1) 内存，O(1) 每步计算"""
        B, T, C = r.shape
        
        num, den, max_len = state  # 从上一步继承的状态
        
        outputs = []
        for t in range(T):
            rt = r[:, t:t+1, :]  # (B,1,C)
            kt = k[:, t:t+1, :]
            vt = v[:, t:t+1, :]
            wt = w[:, t:t+1, :]
            
            # 更新分子分母（RNN 式递推）
            num = wt * kt * vt + (wt.detach()) * num  # detach 避免梯度问题
            den = wt * kt + (wt.detach()) * den
            
            ot = rt * (num / (den + 1e-8))
            outputs.append(ot)
        
        output = torch.cat(outputs, dim=1)
        new_state = (num, den, max_len + T)
        
        return output, new_state


class RWKVModel(nn.Module):
    """
    RWKV 语言模型完整骨架
    
    RWKV 的完整架构包含两种 mixing 类型:
    1. Time-Mixing: 类似上面的 Token Mixer，处理跨时间的信息交互
    2. Channel-Mixing: 类似 FFN，处理通道维度的非线性变换
    
    每个 RWKV Block = TimeMix + ChannelMix + LayerNorm
    """
    
    def __init__(
        self,
        vocab_size: int = 50277,
        embed_dim: int = 768,
        n_layers: int = 24,
        head_size: int = 64,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(embed_dim),
                'time_mix': RWKVTokenMixer(embed_dim, head_size),
                'ln2': nn.LayerNorm(embed_dim),
                'channel_mix': RWKVChannelMix(embed_dim),
            }))
        
        self.ln_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        state: list = None,
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            input_ids: (batch, seq_len)
            state: 可选的 RNN 状态列表（推理时传入）
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            new_state: 更新后的状态列表
        """
        x = self.embedding(input_ids)
        new_states = []
        
        for i, layer in enumerate(self.layers):
            prev_state = state[i] if state else None
            
            # Time Mixing
            residual = x
            x = layer['ln1'](x)
            x, time_state = layer['time_mix'](x, state=prev_state[:2] if prev_state else None)
            x = residual + x
            
            # Channel Mixing
            residual = x
            x = layer['ln2'](x)
            x, channel_state = layer['channel_mix'](x, state=prev_state[2:] if prev_state else None)
            x = residual + x
            
            new_states.append(time_state + channel_state)
        
        x = self.ln_out(x)
        logits = self.head(x)
        
        return logits, new_states


# Channel Mix 的简化实现
class RWKVChannelMix(nn.Module):
    """RWKV 的 Channel Mixing（类似 FFN 但有门控）"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.receptance = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, state=None):
        r = torch.sigmoid(self.receptance(x))
        k = torch.square(torch.relu(self.key(x)))  # ReLU + Square（RWKV 特有的激活）
        v = self.value(x)
        return r * (k * v), state


RWKVModel
```

RWKV 最精妙的地方在于它证明了**同一个数学表达式可以同时以 Transformer 式的并行模式和 RNN 式的串行模式运行**。训练时用并行模式获得与 Transformer 相当的训练速度；推理时切换到 RNN 模式，每一步的计算量和内存都是常数级别。这种"两全其美"的特性让 RWKV 在部署场景中极具吸引力。

## 混合架构：Jamba 与 Mamba-Transformer 结合

既然 Mamba 在长序列和高吞吐方面有优势，而 Transformer 在复杂推理和多模态方面更强，那能不能把它们结合起来？**Jamba**（AI21 Labs 发布）给出了肯定的回答——它在 Transformer 的层之间交替插入 Mamba 层，形成了一个混合架构。

```python
class JambaBlock(nn.Module):
    """
    Jamba 混合块的概念实现
    
    Jamba 的核心设计: 在 Transformer 层和 Mamba 层之间交替堆叠
    
    典型的 Jamba 配置（以 Jamba-1.5-Large 为例）:
    - 总层数: 32 层
    - 每 8 层为一个 block
    - 每个 block 内部: [Mamba × 6 + Attention × 1 + Mamba × 1]
    - 即 8 层中只有 1 层是 Transformer Attention，其余 7 层都是 Mamba
    
    这样做的效果:
    - 保留了约 87% 的 Mamba 高效层（长序列友好、低 KV Cache）
    - 保留了约 13% 的 Transformer 层（复杂推理能力强）
    - MoE（混合专家）进一步放大参数量而不增加计算量
    """
    
    def __init__(
        self,
        d_model: int = 4096,
        mamba_config: dict = None,
        attn_config: dict = None,
        moe_config: dict = None,
    ):
        super().__init__()
        
        # Mamba 子层（占大部分层数）
        self.mamba_layer = (
            MambaBlock(**mamba_config) if mamba_config else None
        )
        
        # Transformer Attention 子层（少量穿插）
        self.attn_layer = (
            nn.MultiheadAttention(**attn_config) if attn_config else None
        )
        
        # MoE FFN（可选，Jamba 默认启用）
        self.moe_ffn = (
            MoEFFN(**moe_config) if moe_config else None
        )
        
        # 层归一化
        self.norm_mamba = nn.RMSNorm(d_model)
        self.norm_attn = nn.RMSNorm(d_model)
        self.norm_ffn = nn.RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        layer_type: str = "mamba",  # "mamba" 或 "attention"
    ) -> torch.Tensor:
        """
        根据 layer_type 选择走哪条路径
        """
        if layer_type == "mamba":
            # Mamba 路径（无 KV Cache，高效率）
            residual = x
            x = self.norm_mamba(x)
            x = self.mamba_layer(x)
            x = residual + x
        else:
            # Attention 路径（有 KV Cache，强表达能力）
            residual = x
            x = self.norm_attn(x)
            attn_output, _ = self.attn_layer(x, x, x)
            x = residual + attn_output
        
        # 共享的 MoE FFN
        residual = x
        x = self.norm_ffn(x)
        x = self.moe_ffn(x)
        x = residual + x
        
        return x


class HybridArchitectureScheduler:
    """
    混合架构的层调度器
    
    定义每一层的类型（Mamba vs Attention）
    """
    
    # Jamba 官方的层配置模式
    JAMBA_PATTERN = (
        ["mamba"] * 6 + ["attention"] + ["mamba"]
    )  # 8层一个周期
    
    @classmethod
    def get_layer_type(cls, layer_idx: int, total_layers: int = 32) -> str:
        """根据层索引返回该层应该是 Mamba 还是 Attention"""
        pattern = cls.JAMBA_PATTERN
        return pattern[layer_idx % len(pattern)]
    
    @classmethod
    def analyze_architecture(cls, total_layers: int = 32) -> dict:
        """分析给定总层数下的架构组成"""
        mamba_count = 0
        attn_count = 0
        
        for i in range(total_layers):
            if cls.get_layer_type(i, total_layers) == "mamba":
                mamba_count += 1
            else:
                attn_count += 1
        
        return {
            "total_layers": total_layers,
            "mamba_layers": mamba_count,
            "attention_layers": attn_count,
            "mamba_ratio": f"{mamba_count/total_layers*100:.1f}%",
            "attention_ratio": f"{attn_count/total_layers*100:.1f}%",
            "estimated_kv_cache_savings": f"~{(1-attn_count/total_layers)*100:.0f}%",
        }


print("Jamba 架构分析:")
analysis = HybridArchitectureScheduler.analyze_architecture(32)
for k, v in analysis.items():
    print(f"  {k}: {v}")
```

Jamba 的这种"少量 Attention 层 + 大量 Mamba 层"的设计背后有一个有趣的直觉：**并不是每一个 Transformer 层都在做同样的事情**。研究表明，浅层 Transformer 主要在做局部的模式匹配（类似 n-gram），深层才真正需要全局注意力来做复杂的语义推理。因此，只在最关键的几层保留完整的注意力机制，其余层用更高效的 Mamba 替代，是一种非常务实的工程选择。

## MoE 深入：混合专家架构的生产实践

我们在前面的章节中简要介绍过 MoE（Mixture of Experts），这里我们从新兴架构的角度做一个更深度的探讨，特别是关注 MoE 在最新的大模型（如 Mixtral、DeepSeek-MoE、Qwen-MoE）中的应用细节。

## MoE 的路由策略演进

```python
class ExpertRouterV1(nn.Module):
    """
    第一代路由器: Top-K 硬路由
    
    每个输入 token 选择 top-k 个最相关的专家进行处理，
    其余专家完全跳过（不参与计算也不参与梯度更新）。
    
    问题: 专家负载不平衡 —— 有些专家被频繁选中，
          有些专家几乎从未被选中（"专家坍塌"问题）
    """
    
    def __init__(self, num_experts: int, hidden_dim: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            expert_weights: (batch*seq_len, top_k) — 每个token对各专家的权重
            expert_indices: (batch*seq_len, top_k) — 选中专家的索引
        """
        logits = self.gate(x)  # (N, num_experts)
        
        # Top-K 选择
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)  # 只对选中的 K 个做 softmax
        
        return weights, indices


class ExpertRouterV2(nn.Module):
    """
    第二代路由器: 带 Load Balancing Loss 的路由
    
    解决专家负载不平衡问题的经典方法:
    在损失函数中加入辅助损失项，鼓励均匀分配
    """
    
    def __init__(self, num_experts: int, hidden_dim: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple:
        logits = self.gate(x)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        
        # 计算负载均衡损失
        lb_loss = self._compute_load_balancing_loss(logits, indices)
        
        return weights, indices, lb_loss
    
    def _compute_load_balancing_loss(
        self, logits: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Load Balancing Loss = num_experts * Σ p_i * f_i
        
        其中:
        - p_i = 分配给专家 i 的 token 比例（实际分布）
        - f_i = 专家 i 的平均路由概率（理想分布）
        
        当两者一致时 loss 最小（即路由概率和实际分配匹配）
        """
        N = logits.shape[0]
        
        # f_i: 每个专家的平均路由概率
        expert_probs = torch.softmax(logits, dim=-1).mean(dim=0)  # (num_experts,)
        
        # p_i: 每个专家实际收到的 token 比例
        mask = torch.zeros_like(logits).scatter_(
            1, indices, 1.0
        )  # one-hot 编码选中的专家
        expert_counts = mask.sum(dim=0) / N  # (num_experts,)
        
        # 负载均衡损失
        lb_loss = self.num_experts * (expert_probs * expert_counts).sum()
        
        return lb_loss


class ExpertRouterV3(nn.Module):
    """
    第三代路由器: 共享专家隔离路由 (DeepSeek-MoE 风格)
    
    DeepSeek-MoE 的核心创新:
    - 把专家分为两类:
      1. shared_experts: 所有 token 都必须经过（共享知识）
      2. routed_experts: 只有被路由器选中的 token 才经过（专业化知识）
    
    这种分离带来了两个好处:
    1. 共享专家保证了基础知识不会丢失（解决纯 MoE 的知识碎片化问题）
    2. 路由专家可以更专注地学习细分领域的知识
    """
    
    def __init__(
        self,
        num_shared_experts: int = 1,
        num_routed_experts: int = 8,
        hidden_dim: int = 4096,
        intermediate_dim: int = 14336,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_shared = num_shared_experts
        self.num_routed = num_routed_experts
        self.top_k = top_k
        
        # 共享专家（始终激活）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_dim, hidden_dim, bias=False),
            )
            for _ in range(num_shared_experts)
        ])
        
        # 路由专家（选择性激活）
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_dim, hidden_dim, bias=False),
            )
            for _ in range(num_routed_experts)
        ])
        
        # 路由器（只为路由专家服务）
        self.gate = nn.Linear(hidden_dim, num_routed_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DeepSeek-MoE 风格的前向传播
        
        流程:
        1. 所有 token 先过共享专家 → base_output
        2. 路由器选择 top-k 个路由专家 → routed_output
        3. 最终输出 = base_output + routed_output
        """
        B, seq_len, D = x.shape
        
        # Step 1: 共享专家（所有 token 都经过）
        shared_outputs = []
        for expert in self.shared_experts:
            shared_outputs.append(expert(x))
        base_output = sum(shared_outputs) / len(shared_outputs)  # 平均融合
        
        # Step 2: 路由专家（选择性激活）
        flat_x = x.reshape(-1, D)  # (B*seq_len, D)
        logits = self.gate(flat_x)  # (B*seq_len, num_routed)
        
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        
        # 专家计算（只计算被选中的）
        routed_output = torch.zeros_like(flat_x)
        for i in range(self.num_routed_experts):
            mask = (indices == i)  # 哪些 token 选择了专家 i
            if mask.any():
                selected_tokens = flat_x[mask.any(dim=-1)]
                expert_out = self.routed_experts[i](selected_tokens)
                # 加权累加
                expert_weights = weights.masked_fill(~mask, 0).sum(dim=-1, keepdim=True)
                routed_output = routed_output.scatter_add(
                    0, mask.nonzero(as_tuple=True)[0].unsqueeze(-1).expand_as(expert_out),
                    expert_out * expert_weights[mask.any(dim=-1)],
                )
        
        routed_output = routed_output.reshape(B, seq_len, D)
        
        # Step 3: 合并
        final_output = base_output + routed_output
        
        return final_output, logits


ExpertRouterV1, ExpertRouterV2, ExpertRouterV3
```

DeepSeek-MoE 的这种"共享专家 + 路由专家"分离设计在生产环境中特别有价值。纯 MoE 模型（如原始 Switch Transformer）的一个常见问题是**不同专家之间可能出现知识冗余或知识空洞**——有些专家学了重复的东西，有些重要知识却没有任何专家覆盖。共享专家的存在就像一个"安全网"，确保基础的语言理解能力不会因为路由器的随机性而丢失。

## RoPE v2 与位置编码的新进展

我们在前面章节中详细讨论过 RoPE（Rotary Position Embedding），它是目前大多数主流 LLM（LLaMA、Mistral、Qwen 等）的标准位置编码方案。RoPE v2 及后续变体主要针对两个问题做了改进：**外推能力**和**长序列效率**。

## RoPE 的外推问题回顾

标准 RoPE 的频率设置是 `θ_i = base^(-2i/d)`，其中 base 通常是 10000。这意味着高频分量（对应小的 i）变化快，能精确区分相邻位置；低频分量（对应大的 i）变化慢，负责编码远距离的位置关系。问题是当序列长度超过训练时的最大长度后，低频分量无法区分超出范围的位置——这就是**外推失效**的根本原因。

## 几种改进方案

```python
def rope_variants_comparison():
    """
    RoPE 变体全面对比
    """
    
    import numpy as np
    
    variants = {
        "Original RoPE": {
            "base": 10000,
            "formula": "θ_i = base^(-2i/d)",
            "max_train_len": 4096,
            "extrapolation": "差 — 超出训练长度后迅速退化",
            "used_by": "LLaMA 1, GPT-NeoX",
            "key_feature": "最基础的旋转位置编码",
        },
        "NTK-Aware Scaling": {
            "base": None,  # 动态调整
            "formula": "θ_i = (α·base)^(-2i/d), α = L'/L_train",
            "max_train_len": 4096,
            "extrapolation": "较好 — 通过增大 base 来'拉伸'频率",
            "used_by": "社区广泛采用",
            "key_feature": "推理时动态缩放 base，无需重新训练",
        },
        "YaRN (Yet another RoPE)": {
            "base": 10000,
            "formula": "NTK缩放 + 频率插值 + 注意力温度调整",
            "max_train_len": 4096,
            "extrapolation": "很好 — 在 128K 长度上仍有不错表现",
            "used_by": "CodeLlama, 长文本 LLaMA 变体",
            "key_feature": "三管齐下：NTK + 插值 + temp scaling",
        },
        "RoPE v2 (Dynamic NTK)": {
            "base": "动态",
            "formula": "根据当前序列长度自适应调整 base",
            "max_train_len": "不限（理论上）",
            "extrapolation": "优秀 — 自适应调整",
            "used_by": "Mistral, Qwen 2, DeepSeek",
            "key_feature": "推理时自动检测是否需要外推并调整",
        },
        "AliBi (Alternative)": {
            "base": "N/A",
            "formula": "attention_bias = -s·|i-j|, s随head递减",
            "max_train_len": "不限（天然支持任意长度）",
            "extrapolation": "优秀 — 偏置随距离线性增长，无突变",
            "used_by": "BLOOM, mT5",
            "key_feature": "不需要位置 embedding，偏置直接加到 attention score",
        },
    }
    
    print("=" * 80)
    print("RoPE 变体对比一览")
    print("=" * 80)
    print()
    
    for name, info in variants.items():
        print(f"【{name}】")
        for key, value in info.items():
            if key != "key_feature":
                print(f"  {key}: {value}")
        print(f"  ➜ 核心特点: {info['key_feature']}")
        print()


rope_variants_comparison()


def dynamic_ntk_rope_implementation():
    """
    Dynamic NTK-Aware RoPE 的 Python 实现
    
    这是目前工业界最常用的长文本扩展方案
    """
    
    import torch
    import math
    
    class DynamicNTKRoPE:
        """
        动态 NTK-Aware RoPE
        
        核心思想:
        当检测到输入序列超过训练时的最大长度时，
        自动增大 base 值来"拉伸"频率分布，
        使得原本的高频成分能够覆盖更大的位置范围。
        
        α (scaling factor) 的计算:
        - 如果 seq_len <= max_train_len: α = 1.0（不变）
        - 如果 seq_len > max_train_len: α = (seq_len / max_train_len)^(某个系数)
        """
        
        def __init__(
            self,
            dim: int,              # 位置编码维度（通常是 head_dim 的一半）
            base: float = 10000,   # 原始 base
            max_train_len: int = 4096,  # 训练时的最大长度
            scaling_factor: float = 1.0,  # 额外的缩放因子
        ):
            self.dim = dim
            self.base = base
            self.max_train_len = max_train_len
            self.scaling_factor = scaling_factor
        
        def compute_freqs(
            self,
            seq_len: int,
            device: torch.device = "cpu",
            dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            """
            计算旋转频率
            
            Returns:
                freqs: (seq_len, dim/2) — 每个位置的复数旋转角度
            """
            # 动态调整 base
            if seq_len > self.max_train_len:
                # NTK-Aware 缩放
                scale = seq_len / self.max_train_len
                alpha = scale ** (self.scaling_factor / self.dim)
                effective_base = self.base * alpha
            else:
                effective_base = self.base
            
            # 计算频率: θ_i = base^(-2i/d)
            i = torch.arange(0, self.dim, 2, device=device, dtype=dtype)
            freqs = 1.0 / (effective_base ** (i / self.dim))  # (dim/2,)
            
            # 生成位置索引
            t = torch.arange(seq_len, device=device, dtype=dtype)
            
            # 外积: (seq_len, dim/2)
            angles = torch.outer(t, freqs)
            
            return angles
        
        def apply_rotary_emb(
            self,
            x: torch.Tensor,  # (..., seq_len, head_dim)
            freqs: torch.Tensor,  # (seq_len, head_dim/2)
        ) -> torch.Tensor:
            """
            应用旋转位置编码
            
            将输入 x 拆分为偶数位和奇数位，
            分别应用 cos 和 sin 旋转
            """
            # x_0 = x[..., ::2], x_1 = x[..., 1::2]
            x0 = x[..., ::2].float()
            x1 = x[..., 1::2].float()
            
            cos_f = torch.cos(freqs).to(x.dtype)
            sin_f = torch.sin(freqs).to(x.dtype)
            
            # 旋转: [x0, x1] @ [[cos, -sin], [sin, cos]]
            out0 = x0 * cos_f - x1 * sin_f
            out1 = x0 * sin_f + x1 * cos_f
            
            # 交错合并
            out = torch.stack([out0, out1], dim=-1).flatten(-2)
            
            return out.type_as(x)
    
    # ===== 使用示例 =====
    rope = DynamicNTKRoPE(
        dim=64,               # head_dim=128 时的一半
        base=10000,
        max_train_len=4096,
        scaling_factor=0.67,   # YaRN 推荐的经验值
    )
    
    # 测试不同长度下的行为
    test_lengths = [1024, 4096, 8192, 32768, 128000]
    
    print("Dynamic NTK-RoPE 行为测试:")
    print("-" * 50)
    print(f"{'序列长度':<12} {'是否外推':<10} {'有效Base':<15} {'频率范围'}")
    print("-" * 50)
    
    for length in test_lengths:
        is_extrapolating = length > rope.max_train_len
        freqs = rope.compute_freqs(length)
        effective_base = rope.base * (length / rope.max_train_len)**(rope.scaling_factor/rope.dim) if is_extrapolating else rope.base
        
        freq_min = freqs.min().item()
        freq_max = freqs.max().item()
        
        status = "✅ 正常" if not is_extrapolating else "🔄 外推"
        print(f"{length:<12} {status:<10} {effective_base:<15.1f} "
              f"[{freq_min:.6f}, {freq_max:.4f}]")
    
    return DynamicNTKRoPE


dynamic_ntk_rope_implementation()
```

## 新兴架构的实际落地建议

## 如何在你的项目中评估和选择新架构？

```python
class ArchitectureEvaluator:
    """
    新架构评估框架
    
    在决定是否将 Mamba/RWKV/MoE 等新架构投入生产之前，
    需要从多个维度进行全面评估
    """
    
    EVALUATION_DIMENSIONS = [
        "task_fit",         # 任务适配度
        "data_efficiency",  # 数据效率
        "training_cost",    # 训练成本
        "inference_cost",   # 推理成本
        "latency_requirement",  # 延迟要求
        "memory_constraint",    # 内存限制
        "ecosystem_support",    # 生态支持
        "team_familiarity",     # 团队熟悉度
        "maintenance_cost",     # 维护成本
        "risk_tolerance",       # 风险容忍度
    ]
    
    @staticmethod
    def evaluate_for_scenario(scenario: str) -> dict:
        """
        根据典型场景给出架构推荐
        
        Args:
            scenario: 应用场景名称
        
        Returns:
            推荐结果字典
        """
        
        scenarios = {
            "chatbot_service": {
                "description": "在线聊天机器人服务，需要低延迟、高并发",
                "primary_concern": "inference_latency + throughput",
                "recommended": [
                    {"arch": "Transformer + KV Cache优化", "confidence": "high"},
                    {"arch": "Mamba (如果上下文很长)", "confidence": "medium"},
                    {"arch": "MoE (如果预算允许)", "confidence": "medium"},
                ],
                "avoid": ["纯 RWKV (复杂推理不够强)", "未经大规模验证的新架构"],
            },
            "document_analysis": {
                "description": "长文档分析（合同、报告、论文），需要处理超长文本",
                "primary_concern": "context_length + comprehension",
                "recommended": [
                    {"arch": "Mamba / Mamba-Transformer 混合", "confidence": "high"},
                    {"arch": "Transformer + Long Context (YaRN/NTK)", "confidence": "high"},
                    {"arch": "RAG (检索增强生成)", "confidence": "very_high"},
                ],
                "avoid": ["标准 Transformer (受限于上下文窗口)"],
            },
            "edge_device": {
                "description": "边缘设备部署（手机、IoT），资源极度受限",
                "primary_concern": "model_size + memory + latency",
                "recommended": [
                    {"arch": "量化后的 Transformer (INT4/INT8)", "confidence": "high"},
                    {"arch": "RWKV (原生 O(1) 推理)", "confidence": "medium"},
                    {"arch": "Mamba + 量化", "confidence": "medium"},
                ],
                "avoid": ["未量化的任何大模型", "需要大量 KV Cache 的模型"],
            },
            "real_time_video": {
                "description": "实时视频理解（监控、直播），需要持续处理视频流",
                "primary_concern": "throughput + streaming capability",
                "recommended": [
                    {"arch": "专用视频模型 (VideoMAE/InternVideo)", "confidence": "high"},
                    {"arch": "VideoLLaVA (轻量版)", "confidence": "medium"},
                ],
                "avoid": ["通用 LLM 直接处理视频帧 (太慢)"],
            },
            "code_generation": {
                "description": "代码生成和补全，需要精确的语法理解和长程依赖",
                "primary_concern": "accuracy + long_range_dependency",
                "recommended": [
                    {"arch": "Transformer (CodeLlama/StarCoder/Qwen-Coder)", "confidence": "very_high"},
                    {"arch": "MoE Transformer (DeepSeek-Coder)", "confidence": "high"},
                ],
                "avoid": ["纯 Mamba/RWKV (代码任务验证不足)"],
            },
            "multimodal_agent": {
                "description": "多模态智能体，同时处理图像、文本、音频",
                "primary_concern": "modality_coverage + reasoning",
                "recommended": [
                    {"arch": "Transformer-based (LLaVA/GPT-4V风格)", "confidence": "very_high"},
                    {"arch": "ImageBind + LLM", "confidence": "medium"},
                ],
                "avoid": ["纯文本架构 (不支持多模态)"],
            },
        }
        
        return scenarios.get(scenario, {"error": f"Unknown scenario: {scenario}"})
    
    @staticmethod
    def migration_checklist(current_arch: str, target_arch: str) -> list[str]:
        """
        架构迁移检查清单
        
        从现有架构迁移到新架构前需要确认的事项
        """
        return [
            f"[ ] 目标架构是否有成熟的预训练权重可供微调？({target_arch})",
            f"[ ] 微调数据格式是否兼容？（{current_arch} → {target_arch} 的数据转换）",
            f"[ ] 推理框架是否支持目标架构？（vLLM/TGI/ONNX 对 {target_arch} 的支持情况）",
            f"[ ] 团队是否熟悉目标架构的训练和调试方法？",
            f"[ ] 是否有回退方案？如果新架构效果不好能否快速切回旧架构？",
            f"[ ] A/B 测试计划是否制定？如何公平比较新旧架构？",
            f"[ ] 监控指标是否就绪？除了准确率还要监控哪些指标？",
            f"[ ] 服务基础设施是否需要升级？（GPU型号、内存、网络等）",
        ]


# 使用示例
evaluator = ArchitectureEvaluator()

print("=" * 60)
print("新架构选型指南")
print("=" * 60)

scenarios_to_check = ["chatbot_service", "document_analysis", "edge_device"]

for scenario in scenarios_to_check:
    result = evaluator.evaluate_for_scenario(scenario)
    print(f"\n【{scenario}】{result['description']}")
    print(f"  核心关注: {result['primary_concern']}")
    print(f"  推荐:")
    for rec in result['recommended']:
        print(f"    • {rec['arch']} (置信度: {rec['confidence']})")
    print(f"  避免:")
    for avoid in result['avoid']:
        print(f"    ✗ {avoid}")


print("\n\n架构迁移检查清单 (Transformer → Mamba):")
for item in ArchitectureEvaluator.migration_checklist("Transformer", "Mamba"):
    print(item)
```

## 面试常考问题

**Q1: Mamba 和 Transformer 本质区别是什么？**
Transformer 的核心是自注意力机制，通过 Q/K/V 的点积来建模序列中任意两个位置之间的关系，计算复杂度 `O(n²)`，需要 KV Cache。Mamba 的核心是状态空间模型（SSM），通过隐状态的递推传递来建模序列依赖，计算复杂度 `O(n)`（训练时可并行化为 `O(n log n)`），推理时 `O(1)` 内存和计算。Mamba 的关键是引入了输入依赖的动态参数（Δ、B、C），使其具备了选择性记忆能力。

**Q2: RWKV 如何做到既能并行训练又能 RNN 式推理？**
RWKV 的数学表达式中，Attention 的分子和分母都可以写成**累积求和（cumulative sum）**的形式，而累积求和是可以用并行 scan 算法高效计算的（类似前缀和的并行算法）。因此在训练时，RWKV 利用并行 scan 实现 Transformer 式的全序列并行处理；在推理时，同一组参数又可以改写成 RNN 式的递推形式，每步只需 `O(1)` 的计算和存储。

**Q3: 为什么 Jamba 选择混合架构而不是纯 Mamba？**
实验表明，Mamba 在大多数任务上已经接近甚至超越同规模的 Transformer，但在**复杂的多步推理任务**（如数学证明、逻辑链条推理）上仍有差距。混合架构用少量的 Transformer 层（约 12%）来弥补这一短板，同时保留 Mamba 层（约 88%）带来的长序列效率和低 KV Cache 优势。这是一种务实的"取长补短"策略。

**Q4: DeepSeek-MoE 的共享专家设计解决了什么问题？**
纯 MoE 模型存在"专家坍塌"和"知识碎片化"风险——某些专家可能永远不被选中（浪费容量），或者不同专家学到重复的知识（效率低下）。DeepSeek-MoE 将专家分为共享专家（所有 token 必经）和路由专家（选择性激活）。共享专家确保基础语言能力不会因路由随机性而丢失，路由专家则专注于细粒度的领域知识。这种设计在同等参数量下通常能达到更好的效果。

**Q5: RoPE 的外推问题本质原因是什么？有哪些解决方案？**
本质原因是 RoPE 的频率按几何级数衰减（`base^(-2i/d)`），高频分量只能编码有限范围内的精细位置差异。当序列超出训练长度后，高频分量不足以区分远处位置，导致位置信息模糊化。主要解决方案包括：(1) NTK-Aware Scaling — 推理时动态增大 base 来拉伸频率；(2) YaRN — 结合 NTK 缩放 + 线性插值 + 注意力温度调整；(3) 动态 RoPE — 根据实际序列长度自适应调整；(4) AliBi — 放弃 RoPE，改用与距离成正比的注意力偏置（天然支持任意长度）。
