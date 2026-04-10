# 1.4 nn.Module——构建神经网络的标准方式

> 前面我们用裸 Tensor 和 Autograd 实现了简单的计算。但当模型有几十亿参数、上百层时，手动管理每个权重张量是不现实的。`nn.Module` 是 PyTorch 的"容器"——它自动收集所有可训练参数、管理设备移动、支持序列化保存。每一个 LLM 架构（从 GPT-2 到 LLaMA）都是 `nn.Module` 的子类。

## 从裸参数到 nn.Module

```python
"""
nn.Module 演进：从裸参数到标准写法
"""

import torch
import torch.nn as nn


def demo_module_evolution():
    """展示为什么需要 nn.Module"""

    print("=" * 60)
    print("📦 nn.Module 演进")
    print("=" * 60)

    # === 方式一：裸参数（不推荐） ===
    print("\n--- 方式一: 裸参数 (❌ 不推荐) ---")

    # 手动定义权重
    w1 = torch.randn(768, 3072, requires_grad=True)   # FFN 第一层
    b1 = torch.zeros(3072, requires_grad=True)
    w2 = torch.randn(3072, 768, requires_grad=True)    # FFN 第二层
    b2 = torch.zeros(768, requires_grad=True)

    # 手动前向传播
    x = torch.randn(2, 10, 768)
    h = x @ w1 + b1
    h = torch.relu(h)
    out = h @ w2 + b2

    # 手动收集参数（容易遗漏！）
    params = [w1, b1, w2, b2]
    total_params = sum(p.numel() for p in params)

    print(f"  参数量: {total_params:,}")
    print(f"  问题:")
    print(f"    - 必须手动列出所有参数（容易遗漏）")
    print(f"    - .to(device) 要逐个调用")
    print(f"    - state_dict 要自己构建")
    print(f"    - 无法使用 model.train() / model.eval()")

    # === 方式二：nn.Module（✅ 标准方式） ===
    print("\n--- 方式二: nn.Module (✅ 推荐) ---")

    class FeedForward(nn.Module):
        def __init__(self, n_embed, expansion=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embed, n_embed * expansion),
                nn.GELU(),
                nn.Linear(n_embed * expansion, n_embed),
            )

        def forward(self, x):
            return self.net(x)

    ff = FeedForward(768)
    x = torch.randn(2, 10, 768)
    out = ff(x)

    print(f"  参数量: {sum(p.numel() for p in ff.parameters()):,}")
    print(f"  输出形状: {out.shape}")
    print(f"  优势:")
    print(f"    - parameters() 自动收集所有参数 ✅")
    print(f"    - .to('cuda') 一行搞定设备转移 ✅")
    print(f"    - state_dict() 自动生成 ✅")
    print(f"    - train()/eval() 切换模式 ✅")


if __name__ == "__main__":
    demo_module_evolution()
```

## nn.Module 核心机制

### 1. 参数自动收集

```python
class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 3072)     # 在 __init__ 中注册
        self.linear2 = nn.Linear(3072, 768)
        self.my_param = nn.Parameter(torch.randn(64))  # 也可以直接注册 Parameter

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


model = DemoModel()

# 自动收集所有参数（包括子模块的参数）
for name, param in model.named_parameters():
    print(f"{name:<25} shape={str(param.shape):<20} requires_grad={param.requires_grad}")
```

输出：

```
linear1.weight              shape=torch.Size([3072, 768])  requires_grad=True
linear1.bias                shape=torch.Size([3072])         requires_grad=True
linear2.weight              shape=torch.Size([768, 3072])    requires_grad=True
linear2.bias                shape=torch.Size([768])          requires_grad=True
my_param                   shape=torch.Size([64])           requires_grad=True
```

### 2. 设备递归移动

```python
model = DemoModel()
print(f"默认设备: {next(model.parameters()).device}")  # cpu

model = model.to("cuda")
print(f"to(cuda) 后: {next(model.parameters()).device}")  # cuda:0
# 所有子模块的参数都自动移到了 GPU！
```

### 3. train() / eval() 模式切换

```python
model = DemoModel()
model.train()   # 训练模式：Dropout 开启，BatchNorm 更新统计量
model.eval()    # 推理模式：Dropout 关闭，BatchNorm 使用运行均值/方差

# LLM 中主要影响 Dropout:
dropout_layer = nn.Dropout(p=0.1)
x = torch.ones(10)

model.train()
print(f"train 模式 dropout: {dropout_layer(x)}")  # 部分元素变为 0！

model.eval()
print(f"eval 模式 dropout:  {dropout_layer(x)}")  # 全部保持不变
```

### 4. state_dict / load_state_dict

```python
# 保存模型
torch.save(model.state_dict(), "my_model.pt")

# 加载模型
new_model = DemoModel()
new_model.load_state_dict(torch.load("my_model.pt", weights_only=True))
```

## 常用层一览——LLM 的积木

### Embedding 层

```python
# 词嵌入：将离散 token ID 映射为连续向量
token_embedding = nn.Embedding(num_embeddings=32000, embedding_dim=4096)
token_ids = torch.tensor([15496, 2159, 287])
embeddings = token_embedding(token_ids)  # (3, 4096)

# 位置嵌入：编码位置信息
position_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=4096)
positions = torch.arange(3)
pos_embs = position_embedding(positions)  # (3, 4096)
```

### Linear 层

```python
# 全连接层 —— Attention 中的 Q/K/V/O 投影、FFN、LM Head 都用它
q_proj = nn.Linear(in_features=4096, out_features=4096, bias=False)
x = torch.randn(2, 128, 4096)
q = q_proj(x)  # (2, 128, 4096)
```

### LayerNorm / RMSNorm

```python
# LayerNorm — Transformer Block 中的归一化
layer_norm = nn.LayerNorm(normalized_shape=4096)
x_normed = layer_norm(x)  # 每个样本独立归一化，mean≈0, var≈1

# RMSNorm — LLaMA/Qwen 使用（更高效）
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

### Dropout

```python
# Dropout —— 正则化，防止过拟合
dropout = nn.Dropout(p=0.1)
model.train()
print(dropout(torch.ones(10)))   # 约 10% 的元素变成 0
model.eval()
print(dropout(torch.ones(10)))   # 保持不变
```

> ⚠️ **LLM 训练中 Dropout 通常设为 0**。因为预训练数据量极大（万亿级 token），过拟合风险很低。但微调小数据集时可以适当开启。

## Container 模块：Sequential vs ModuleList vs ModuleDict

这是初学者最容易混淆的地方：

```python
def demo_containers():
    """三种 Container 的对比"""

    print("=" * 60)
    print("📦 Container 模块对比")
    print("=" * 60)

    # === Sequential: 顺序容器 ===
    # 适用: 固定层数、固定顺序的前馈网络
    sequential_net = nn.Sequential(
        nn.Linear(768, 2048),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 768),
    )
    x = torch.randn(2, 10, 768)
    print(f"\n[Sequential]")
    print(f"  输入: {x.shape}")
    print(f"  输出: {sequential_net(x).shape}")
    print(f"  适用: 固定结构的前馈网络（如 FFN）")

    # === ModuleList: 动态长度列表 ===
    # 适用: 层数由配置决定（如 N 个 Transformer Block）
    class TransformerStack(nn.Module):
        def __init__(self, n_layers):
            super().__init__()
            # ❌ 不能用 Sequential! 因为层数是动态的
            # ✅ 用 ModuleList
            self.layers = nn.ModuleList([
                nn.Linear(768, 768) for _ in range(n_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x) + x  # Residual Connection
            return x

    stack_4layer = TransformerStack(n_layers=4)
    stack_8layer = TransformerStack(n_layers=8)
    print(f"\n[ModuleList]")
    print(f"  4层模型参数量: {sum(p.numel() for p in stack_4layer.parameters()):,}")
    print(f"  8层模型参数量: {sum(p.numel() for p in stack_8layer.parameters()):,}")
    print(f"  适用: 动态层数（Transformer Block 列表）")

    # === ModuleDict: 按名称索引 ===
    # 适用: 多个子模块按名字访问（如多任务学习）
    task_heads = nn.ModuleDict({
        "classification": nn.Linear(768, 10),
        "regression": nn.Linear(768, 1),
        "embedding": nn.Linear(768, 256),
    })
    cls_out = task_heads["classification"](x[:, 0, :])
    print(f"\n[ModuleDict]")
    print(f"  可用 key: {list(task_heads.keys())}")
    print(f"  分类头输出: {cls_out.shape}")
    print(f"  适用: 多任务学习 / 按名访问子模块")

    # === 选择指南 ===
    print(f"\n{'='*60}")
    print(f"选择指南:")
    print(f"  Sequential → 固定结构、纯顺序执行")
    print(f"  ModuleList → 动态数量、循环遍历、需要 residual connection")
    print(f"  ModuleDict → 按名称访问、条件分支选择子模块")


if __name__ == "__main__":
    demo_containers()
```

## 参数初始化策略

PyTorch 有默认的初始化策略，但对于 Transformer 来说，了解并自定义初始化很重要：

```python
def demo_initialization():
    """LLM 相关的初始化策略"""

    print("=" * 60)
    print("🎲 参数初始化策略")
    print("=" * 60)

    # PyTorch 默认初始化
    linear_default = nn.Linear(768, 768)
    w_default = linear_default.weight.data
    print(f"\n[默认 Kaiming Uniform]")
    print(f"  范围: [{w_default.min():.4f}, {w_default.max():.4f}]")
    print(f"  标准差: {w_default.std():.4f}")

    # GPT-2 风格: 截断正态分布
    linear_trunc = nn.Linear(768, 768)
    nn.init.trunc_normal_(linear_trunc.weight, std=0.02)
    w_trunc = linear_trunc.weight.data
    print(f"\n[截断正态分布 (TruncNormal)]")
    print(f"  std=0.02, 范围限制在 [-2σ, 2σ]")
    print(f"  实际范围: [{w_trunc.min():.4f}, {w_trunc.max():.4f}]")
    print(f"  使用者: GPT-2 / LLaMA / Qwen / 几乎所有现代 LLM")

    # Xavier/Glorot 初始化
    linear_xavier = nn.Linear(768, 768)
    nn.init.xavier_uniform_(linear_xavier.weight)
    print(f"\n[Xavier Uniform]")
    print(f"  适用于 tanh/sigmoid 激活")
    print(f"  公式: U(-√(6/(in+out)), √(6/(in+out)))")

    # Kaiming/He 初始化
    linear_kaiming = nn.Linear(768, 768)
    nn.init.kaiming_normal_(linear_kaiming.weight, mode='fan_in', nonlinearity='relu')
    print(f"\n[Kaiming Normal (He Init)]")
    print(f"  适用于 ReLU / GELU 激活")
    print(f"  PyTorch Linear 默认使用的就是 Kaiming Uniform")

    # Embedding 特殊初始化
    emb = nn.Embedding(32000, 4096)
    nn.init.normal_(emb.weight, mean=0.0, std=0.02)
    print(f"\n[Embedding 初始化]")
    print(f"  正态分布, mean=0, std=0.02")
    print(f"  这是 LLM Token Embedding 的常见初始化")


if __name__ == "__main__":
    demo_initialization()
```

## register_buffer：非参数张量的注册

有些张量不是"可训练参数"，但需要随模型一起移动设备和保存。比如 Causal Mask：

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, max_seq_len=2048):
        super().__init__()
        # 注册 buffer: 不是参数（不会参与梯度更新），但随模型移动和保存
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))  # 下三角因果掩码
        )

    def forward(self, x):
        B, T, C = x.shape
        attn = torch.matmul(x, x.transpose(-2, -1)) / (C ** 0.5)
        # mask 会自动在正确的 device 上
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        return F.softmax(attn, dim=-1)


model = CausalSelfAttention(max_seq_len=512)
model = model.to("cuda")  # mask 也自动搬到 cuda 了！
print(f"mask device: {model.mask.device}")  # cuda:0

# 保存时 buffer 也会被包含
state = model.state_dict()
print(f"'mask' in state_dict: {'mask' in state}")  # True
```

下一节我们将完成第 1 章的最后一个小项目：**从零实现 MiniGPT**。
