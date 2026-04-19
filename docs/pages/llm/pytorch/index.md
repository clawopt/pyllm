---
title: "PyTorch Lightning教程 - 深度学习模型训练实战 | PyLLM"
description: "PyTorch Lightning深度学习训练完整教程：张量操作、Transformer实现、训练循环、Lightning框架、PEFT LoRA微调、分布式训练、模型导出"
head:
  - meta: {name: 'keywords', content: 'PyTorch,Lightning,深度学习,Transformer,LoRA,分布式训练,模型微调'}
  - meta: {property: 'og:title', content: 'PyTorch Lightning教程 - 深度学习模型训练实战 | PyLLM'}
  - meta: {property: 'og:description', content: 'PyTorch Lightning深度学习训练完整教程：张量操作、Transformer实现、训练循环、Lightning框架、PEFT LoRA微调、分布式训练、模型导出'}
  - meta: {name: 'twitter:title', content: 'PyTorch Lightning教程 - 深度学习模型训练实战 | PyLLM'}
  - meta: {name: 'twitter:description', content: 'PyTorch Lightning深度学习训练完整教程：张量操作、Transformer实现、训练循环、Lightning框架、PEFT LoRA微调、分布式训练、模型导出'}
---

# PyTorch 教程大纲

---

## 总体设计思路

**PyTorch 是大语言模型的"母语"**——无论是 HuggingFace Transformers、vLLM、LangChain 还是任何 LLM 框架，底层都是 PyTorch。不理解 PyTorch，就永远只是在"调包"，遇到问题只能盲猜。

但传统的 PyTorch 教程有两个通病：一是从 MNIST/CIFAR 开始讲起（与大模型脱节），二是把 Lightning 当作唯一正道（忽略了 LLM 生态的多样性）。本教程的设计原则是：

1. **从第一天就面向大模型**：所有示例围绕 Transformer、注意力机制、语言建模展开，不用 MNIST
2. **三种训练范式并重**：手动训练循环 → PyTorch Lightning → HuggingFace Trainer，让读者理解各自的优劣和适用场景
3. **从原理到源码级深度**：不仅会调用 API，还要能从零手写一个 GPT-2 级别的模型
4. **覆盖全链路**：张量操作 → 自动微分 → 模型构建 → 数据管道 → 训练循环 → 分布式训练 → 推理优化 → 生产部署

### 本教程与已有教程的关系

```
你的学习路径:

Python 基础 ──→ NumPy 科学计算 ──→ 【本教程: PyTorch】
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
            HuggingFace            vLLM 推理引擎         LangChain/LlamaIndex
            Transformers           (高性能推理)          (应用框架)
            (模型/微调/部署)
```

- **Transformers 教程**（已存在）：聚焦 HF 生态（Tokenizer、Model Hub、Trainer、PEFT）
- **vLLM 教程**（已存在）：聚焦推理优化（PagedAttention、Continuous Batching）
- **本教程（PyTorch）**：聚焦底层能力——**如何用纯 PyTorch 从零构建和训练一个大模型**

### 关于 PyTorch Lightning 的定位

> 用户常问："学 PyTorch 是不是就是学 Lightning？"

答案是：**不完全是**。在大模型时代，训练工具有三个层次：

| 层次 | 工具 | 适用场景 | LLM 相关度 |
|-----|------|---------|-----------|
| **L1 原生 PyTorch** | `nn.Module` + 手写循环 | 理解原理 / 自定义需求极高 / 论文复现 | ⭐⭐⭐⭐⭐ 必须掌握 |
| **L2 PyTorch Lightning** | `LightningModule` + `Trainer` | 复杂训练流程 / 多任务学习 / 研究 / 生产管线 | ⭐⭐⭐⭐ 重要桥梁 |
| **L3 HuggingFace Trainer** | `Trainer` + `TrainingArguments` | LLM 微调（LoRA/全量）/ 快速迭代 / 工程落地 | ⭐⭐⭐⭐⭐ **实际最常用** |

本教程将 Lightning 作为 **L2 层的核心内容**来讲（第 5 章），同时对比 L1 和 L3，帮助读者建立完整的认知地图。

---

## 第01章：PyTorch 核心基础——为大模型打地基（5节）

### 定位
零基础入门。假设读者有 Python 和 NumPy 基础，但从未接触过 PyTorch。所有示例使用**与大模型相关的数据**（token IDs、embedding matrices、attention scores），而不是 MNIST 图像。

### 01-01 为什么是 PyTorch？Tensor vs NumPy 的本质区别
- **PyTorch 在 LLM 技术栈中的位置**：
  - vLLM 底层是 PyTorch + CUDA Kernel
  - HuggingFace Transformers 封装了 PyTorch 的 nn.Module
  - LoRA/QLoRA 的矩阵运算依赖 PyTorch 的 autograd
  - torch.compile (PyTorch 2.0+) 正在改变推理加速格局
- **Tensor vs ndarray 的关键差异**：
  - GPU 支持（`.to("cuda")`）— 这是 LLM 的生命线
  - 自动微分（Autograd）— 反向传播自动完成
  - 动态计算图（eager mode）— 调试友好
  - 与 NumPy 的互操作性（`.numpy()` / `torch.from_numpy()`）
- **第一个 Tensor 操作**：
  ```python
  import torch
  # 模拟 LLM 中的一个 token embedding 查找
  vocab_size, embed_dim = 32000, 4096
  embedding_table = torch.randn(vocab_size, embed_dim)
  token_ids = torch.tensor([15496, 2159, 287])  # "Hello world"
  embeddings = embedding_table[token_ids]  # shape: [3, 4096]
  ```
- **环境安装与验证**：
  - CPU 版 vs GPU 版（CUDA 版本匹配）
  - Apple Silicon (MPS) 支持
  - 验证 GPU 可用性：`torch.cuda.is_available()`
- **dtype 大全**：
  - FP32 / FP16 / BF16 / FP8 — 各自何时使用
  - INT8 / INT4 — 量化的基础
  - BFloat16 为什么是大模型的新宠（比 FP16 更宽的动态范围）

### 01-02 Tensor 操作精要——LLM 中最常用的 20 个操作
- **创建 Tensor**：
  - `torch.tensor()`, `torch.zeros()`, `torch.randn()`, `torch.arange()`, `torch.ones_like()`
  - 模拟模型权重初始化、位置编码、注意力掩码
- **形状变换（Shape Manipulation）— LLM 中最高频的操作**：
  - `.view()`, `.reshape()`, `.permute()`, `.transpose()`, `.unsqueeze()`, `.squeeze()`
  - 实战：Transformer 中 Q/K/V 的投影维度变换
  - 实战：Multi-head Attention 的 head 分割与合并
  - ⚠️ 常见坑：contiguous 的含义与何时需要
- **索引与切片**：
  - 高级索引：Boolean mask（实现 attention mask）、gather 操作（实现 token embedding lookup）
  - 实战：KV Cache 的索引操作
- **数学运算**：
  - `matmul` / `@` — 注意力得分计算的核心
  - `softmax` — Attention 归一化
  - `layer_norm` / `rms_norm` — Transformer 归一化
  - `einsum` — Einstein 求和约定（简洁表达复杂张量运算）
- **广播机制（Broadcasting）**：
  - 与 NumPy 广播规则一致
  - 实战：Positional Encoding 加法、Residual Connection
- **归约操作**：
  - `.mean()`, `.sum()`, `.max()`, `.argmax()` — 用于 loss 计算、采样策略
- **拼接与分割**：
  - `cat`, `stack`, `split`, `chunk` — 批处理、序列拼接

### 01-03 Autograd 自动微分——理解反向传播的本质
- **为什么必须理解 Autograd？**
  - 所有 LLM 训练都依赖它
  - 调试梯度问题时（梯度消失/爆炸）必须懂
  - 自定义算子时需要手动实现 backward
- **计算图（Computational Graph）概念**：
  - Tensor 的 `.grad_fn` 属性
  - 叶子节点（Leaf Tensor）vs 中间节点
  - 前向传播构建图，反向传播遍历图
- **手动求导 vs Autograd 对比**：
  ```python
  # Autograd 方式:
  x = torch.tensor([1., 2., 3.], requires_grad=False)
  w = torch.tensor([0.5], requires_grad=True)
  y = torch.tensor([2., 3., 4.])
  pred = w * x          # 前向
  loss = ((pred - y) ** 2).mean()   # loss
  loss.backward()       # 反向！一行搞定
  print(w.grad)         # tensor([-3.6667])
  ```
- **梯度管理**：
  - `.zero_grad()` — 为什么每步都要清零？
  - `.detach()` — 截断梯度流（用于 KV Cache、停止梯度等场景）
  - `torch.no_grad()` — 推理时关闭梯度计算（节省显存）
  - `torch.enable_grad()` / `torch.inference_mode()` — 细粒度控制
- **常见 Autograd 问题排查**：
  - `RuntimeError: element 0 of tensors does not require grad`
  - 梯度累积时的 grad 处理
  - in-place 操作导致的报错（`_` 后缀函数的问题）
- **高阶 Autograd（进阶）**：
  - 二阶导数（Hessian）：`create_graph=True`
  - Jacobian-vector product（JVP）：用于某些高级优化器
  - 自定义 Function 的 forward/backward（为写自定义 CUDA kernel 打基础）

### 01-04 nn.Module——构建神经网络的标准方式
- **从裸参数到 nn.Module 的进化**：
  ```python
  # ❌ 不规范的方式
  w1 = torch.randn(784, 256, requires_grad=True)
  b1 = torch.zeros(256, requires_grad=True)

  # ✅ PyTorch 标准方式
  class MyModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = nn.Linear(784, 256)
      def forward(self, x):
          return self.linear(x)
  ```
- **nn.Module 的核心机制**：
  - `parameters()` / `named_parameters()` — 自动收集所有可训练参数
  - `state_dict()` / `load_state_dict()` — 权重的保存与加载
  - `.train()` / `.eval()` — 训练/推理模式切换（影响 Dropout/BatchNorm）
  - `.to(device)` — 递归移动所有子模块到指定设备
  - `children()` / `modules()` — 模型树遍历
- **常用层（Layers）一览**：
  - `nn.Linear` — 全连接层（FFN、Output Projection、LM Head）
  - `nn.Embedding` — 词嵌入层（Token Embedding、Position Embedding）
  - `nn.LayerNorm` / `nn.RMSNorm` — 归一化层（Transformer 必备）
  - `nn.Dropout` — 正则化
  - `nn.Sequential` — 顺序容器
- **Container 模块**：
  - `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`
  - 什么时候用 Sequential vs ModuleList？（动态层数时必须用 ModuleList）
  - 实战：用 ModuleList 构建 N 层 Transformer Block
- **参数初始化**：
  - 默认初始化策略（Kaiming Uniform for Linear/Xavier for others）
  - 自定义初始化：`nn.init.xavier_uniform_`, `nn.init.kaiming_normal_`, `nn.init.trunc_normal_`
  - Transformer 特有的初始化（截断正态分布、分层学习率的基础）
- **注册 Buffer**：
  - `register_buffer()` — 非参数但需要随模型移动的张量
  - 实战：causal mask（因果注意力掩码）、position_ids、rotary_freqs

### 01-05 第一个小项目：从零实现 MiniGPT
- **目标**：用本章所学知识，从零构建一个字符级语言模型
- **架构设计**：
  - TokenEmbedding → PositionEmbedding → TransformerBlock(N) × 4 → LayerNorm → LM Head
  - 参数量：~10 万（极小，CPU 即可训练）
  - 数据集：tiny_shakespeare 或中文小说片段
- **完整代码结构**：
  ```
  MiniGPT (
    (token_embed): Embedding(vocab_size, n_embed)
    (position_embed): Embedding(block_size, n_embed)
    (blocks): ModuleList(
      (0-3): TransformerBlock(...)
    )
    (ln_f): LayerNorm(n_embed)
    (lm_head): Linear(n_embed, vocab_size)
  )
  ```
- **训练循环（手动版，为下一章铺垫）**：
  - 数据准备：文本 → token IDs → 滑动窗口生成样本
  - 前向传播 → CrossEntropyLoss → Backward → SGD 优化
  - 训练过程可视化（loss 曲线）
  - 生成：贪心解码 / 温度采样
- **效果展示**：生成的文本从乱码到逐渐出现英文/中文单词
- **本项目作为后续章节的基准**：后面每一章都在这个基础上增加功能

---

## 第02章：数据管道——LLM 训练的数据基石（4节）

### 定位
数据决定了模型的上限。对于 LLM 来说，数据管道不仅仅是"加载文件"，更涉及**超长序列的处理、动态 padding、高效 collate、多进程加载**等工程挑战。本章教你构建生产级的数据管道。

### 02-01 Dataset 与 DataLoader 基础
- **Dataset 两种模式**：
  - Map-style Dataset：`__getitem__`, `__len__`（适用于预加载数据）
  - Iterable-style Dataset：`__iter__`（适用于流式/超大数据集）
  - 如何选择？
- **DataLoader 核心参数**：
  - `batch_size`：批次大小对 GPU 利用率的影响
  - `shuffle`：训练集 shuffle / 验证集不 shuffle
  - `num_workers`：多进程数据加载（速度提升 2-5x）
  - `pin_memory`：锁定页内存（加速 CPU→GPU 传输）
  - `drop_last`：是否丢弃最后不完整的 batch
  - `persistent_workers`：避免每个 epoch 重建 worker 进程
- **Sampler 体系**：
  - `RandomSampler`, `SequentialSampler`, `WeightedRandomSampler`
  - `DistributedSampler`（DDP 训练时必用，第 7 章详解）
- **实战：构建 LLM 预训练数据集**：
  - 从 JSONL 文件读取文本
  - Tokenize 并截断/填充到统一长度
  - 创建 input_ids + labels（shifted by 1）

### 02-02 Collate Function 与变长序列处理
- **为什么需要自定义 Collate？**
  - LLM 的输入序列长度天然不同（短句 vs 长文档）
  - 默认 collate 要求所有样本 shape 相同 → 不适合文本数据
- **Padding 策略**：
  - 左 padding vs 右 padding（因果模型通常左 pad）
  - `pad_token_id` 的选择（0 vs -100 vs 特殊 token）
  - Padding 到 batch 内最大长度 vs 固定 max_length
- **自定义 Collate Function**：
  ```python
  def llm_collate_fn(batch):
      max_len = max(item['input_ids'].shape[0] for item in batch)
      padded = []
      for item in batch:
          pad_len = max_len - item['input_ids'].shape[0]
          padded_input = F.pad(item['input_ids'], (0, pad_len), value=pad_token_id)
          padded_mask = F.pad(item['attention_mask'], (0, pad_len), value=0)
          padded_labels = F.pad(item['labels'], (0, pad_len), value=-100)
          padded.append({...})
      return default_collate(padded)
  ```
- **Attention Mask 的构造**：
  - Padding mask（区分真实 token 和 padding）
  - Causal mask（自回归模型的下三角掩码）
  - 组合掩码：padding_mask | causal_mask
- **Packed Sequences（进阶）**：
  - `pack_padded_sequence` / `pad_packed_sequence`
  - 消除 padding 的无效计算

### 02-03 高效数据处理技巧
- **Tokenization 集成**：
  - 使用 HuggingFace Tokenizer 的 `batch_encode_plus`
  - `return_tensors="pt"` 直接返回 PyTorch Tensor
  - `truncation=True`, `padding=False`（延迟 padding 到 collate 阶段）
- **内存优化**：
  - Generator / IterableDataset 处理 TB 级语料
  - 内存映射（memmap）技术
  - Arrow / Parquet 格式存储预处理后的数据
  - WebDataset：shards 流式读取
- **数据增强（针对 LLM）**：
  - 数据混合策略（高质量 : 低质量 = 1 : 9 等）
  - 数据去重（MinHash / SimHash 近似去重）
- **性能基准测试**：
  - 不同 num_workers 下的 throughput 对比
  - pin_memory 的加速效果实测
  - 预 tokenize vs 运行时 tokenize 的对比

### 02-04 完整数据管道实战：从原始文本到 Training Batch
- **端到端 Pipeline**：
  ```
  原始文本 (.txt/.jsonl)
      ↓ 读取 & 清洗
  文本列表
      ↓ Tokenizer 批量编码
  token_ids 列表
      ↓ 滑窗切分 (stride < block_size 重叠)
  训练样本 (input_ids, labels)
      ↓ DataLoader + Custom Collate
  Batch Tensor (GPU ready)
  ```
- **完整代码实现**：
  - `TextDataset` 类：支持多种格式输入
  - `LLMCollator` 类：动态 padding + mask 构造
  - `build_dataloader()` 工厂函数
  - 数据统计与可视化：序列长度分布、词表覆盖率
- **与 HuggingFace Dataset 的对比**：
  - HF `map()` / `filter()` / `shuffle()` 的便捷性
  - 什么情况下该用 HF Dataset，什么情况自己写

---

## 第03章：用纯 PyTorch 构建 Transformer（5节）

### 定位
这是整个教程的**高潮章节**。你将从零开始，一行一行地实现一个完整的 GPT-2 架构。完成后，你将真正理解"Attention Is All You Need"的每一个细节——这不仅是炫技，更是阅读任何 LLM 源码（LLaMA、Qwen、Mistral）的前提。

### 03-01 单头 Self-Attention 手写实现
- **Attention 的直觉引入**：
  - 从信息检索类比：Query = "我在找什么"，Key = "每条信息的标签"，Value = "信息本身"
  - 公式推导：$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
  - 缩放因子 $\frac{1}{\sqrt{d_k}}$ 的作用（防止 softmax 饱和）
- **逐步实现**：
  ```python
  class SingleHeadAttention(nn.Module):
      def __init__(self, n_embed, head_dim):
          super().__init__()
          self.q_proj = nn.Linear(n_embed, head_dim, bias=False)
          self.k_proj = nn.Linear(n_embed, head_dim, bias=False)
          self.v_proj = nn.Linear(n_embed, head_dim, bias=False)
          self.proj = nn.Linear(head_dim, n_embed)

      def forward(self, x, mask=None):
          B, T, C = x.shape
          q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
          attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
          if mask is not None:
              attn = attn.masked_fill(mask == 0, float('-inf'))
          attn = F.softmax(attn, dim=-1)
          out = attn @ v
          return self.proj(out)
  ```
- **Causal Mask（因果掩码）的实现**：
  - 下三角矩阵：`torch.tril(torch.ones(T, T))`
  - 为什么自回归模型必须 causal？
  - Mask 的正确位置：softmax 之前填 `-inf`
- **数值稳定性陷阱**：
  - Softmax 的溢出/下溢问题
  - 注意力分数的数值范围监控

### 03-02 Multi-Head Attention 与残差连接
- **从单头到多头**：
  - 多头的本质：并行运行多个单头注意力，然后 concat
  - 两种实现方式：
    1. 多个独立的 Q/K/V 投影（直观但慢）
    2. 单次大投影 + reshape 分割（高效，实际采用）
  - `num_heads` × `head_dim` = `n_embed`
- **高效实现**：
  ```python
  class MultiHeadAttention(nn.Module):
      def __init__(self, n_embed, num_heads):
          super().__init__()
          self.num_heads = num_heads
          self.head_dim = n_embed // num_heads
          self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
          self.proj = nn.Linear(n_embed, n_embed)

      def forward(self, x, mask=None):
          B, T, C = x.shape
          qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
          q, k, v = qkv.unbind(2)  # 各 (B, T, heads, head_dim)
          q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

          attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
          if mask is not None:
              attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
          attn = F.softmax(attn, dim=-1)

          out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
          return self.proj(out)
  ```
- **残差连接（Residual Connection）**：
  - `x = x + sublayer(x)` — 为什么有效？
  - Pre-Norm vs Post-Norm：
    - Post-Norm（原始 Transformer）：LayerNorm 在 Add 之后
    - Pre-Norm（GPT-2/LLaMA/Qwen）：LayerNorm 在 Sublayer 之前 → 训练更稳定
- **LayerNorm vs RMSNorm**：
  - LayerNorm: $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
  - RMSNorm (LLaMA 用): $\hat{x}_i = \frac{x_i}{\sqrt{\bar{x}^2 + \epsilon}} \cdot \gamma$ （无均值中心化，更快）
  - 手写 RMSNorm 实现

### 03-03 Feed-Forward Network 与 Transformer Block 组装
- **FFN（Feed-Forward Network）**：
  - 结构：`Linear(up) → Activation → Linear(down)`
  - Expansion ratio：通常 4x（如 GPT-2: 768 → 3072 → 768）
  - 激活函数演进：ReLU → GELU → SwiGLU/SiLU（现代 LLM 标准）
  - SwiGLU 实现：$FFN(x) = (xW_1 \odot SiLU(xW_2))W_3$
  - FFN 占模型参数量的 ~2/3（Attention 只占 ~1/3）
- **完整 Transformer Block**：
  ```python
  class TransformerBlock(nn.Module):
      def __init__(self, n_embed, num_heads, dropout=0.1):
          super().__init__()
          self.ln1 = RMSNorm(n_embed)
          self.attn = MultiHeadAttention(n_embed, num_heads)
          self.dropout1 = nn.Dropout(dropout)
          self.ln2 = RMSNorm(n_embed)
          self.ffn = FeedForward(n_embed, expansion=4)
          self.dropout2 = nn.Dropout(dropout)

      def forward(self, x, mask=None):
          x = x + self.dropout1(self.attn(self.ln1(x), mask))
          x = x + self.dropout2(self.ffn(self.ln2(x)))
          return x
  ```
- **Dropout 的正确使用**：
  - 训练时开启 (`model.train()`)，推理时关闭 (`model.eval()`)
  - LLM 训练中 dropout 通常设为 0（因为数据量大，过拟合风险低）

### 03-04 Positional Encoding——给 Transformer 位置感
- **为什么需要 Positional Encoding？**
  - Self-Attention 是置换不变的（Permutation Invariant）
  - "我 爱 你" 和 "你 爱 我" 的 Attention 结果相同 → 需要注入位置信息
- **方案一：绝对位置编码（Sinusoidal / Learned）**：
  - Sinusoidal（原始 Transformer）
  - Learned Embedding：GPT-2 采用
  - 局限性：外推性差
- **方案二：RoPE（旋转位置编码）——现代 LLM 标准**：
  - 核心思想：将位置信息编码到 Query-Key 的相对旋转中
  - 数学推导：复数乘法 = 平面旋转
  - 实现：外积生成旋转矩阵 → 应用 Q/K
  - 优势：天然支持相对位置 + 一定的长度外推能力
  - LLaMA/Qwen/Mistral 全部采用 RoPE
- **方案三：ALiBi（Attention with Linear Biases）**：
  - 用静态偏置替代位置编码
  - 天然支持任意长度外推
- **实战：实现 RoPE**：
  ```python
  class RotaryEmbedding(nn.Module):
      def __init__(self, dim, max_seq_len=2048, base=10000.0):
          super().__init__()
          inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
          self.register_buffer('inv_freq', inv_freq)
          t = torch.arange(max_seq_len).float()
          freqs = torch.outer(t, inv_freq)
          self.register_buffer('cos_cached', freqs.cos())
          self.register_buffer('sin_cached', freqs.sin())

      def forward(self, x):  # x: (B, T, head_dim)
          ...
  ```

### 03-05 组装完整 GPT 模型与前向传播追踪
- **完整模型组装**：
  ```
  GPT (
    wte: Embedding(vocab_size, n_embed)     # Token Embedding
    wpe: Embedding(block_size, n_embed)     # Position Embedding (或 RoPE)
    drop: Dropout(p)
    h: ModuleList(                           # N 个 Transformer Block
      [TransformerBlock(...) for _ in range(n_layer)]
    )
    ln_f: LayerNorm(n_embed)                 # 最终 Layer Norm
    lm_head: Linear(n_embed, vocab_size)     # 输出层（通常与 wte 权重共享）
  )
  ```
- **权重共享（Weight Tying）**：
  - `lm_head.weight = wte.weight` — 减少参数 + 正则化效果
  - GPT-2 / LLaMA 都采用
- **完整 Forward Pass 逐行追踪**：
  - Input: token_ids `(B, T)`
  - Step 1-6：Embedding → Blocks → Norm → LM Head → logits `(B, T, vocab_size)`
- **损失函数**：
  - `CrossEntropyLoss` + `ignore_index=-100`
  - label shift: `labels = input_ids[:, 1:]`
- **参数量统计脚本**：
  - 按组件分类统计：Embedding / Attention / FFN / LayerNorm / LM Head
  - 各部分占比分析
- **模型验证**：
  - 随机输入 → shape 检查
  - Gradient Check：数值梯度 vs autograd 梯度
  - 过拟合单个 batch（Sanity Check）

---

## 第04章：训练循环与优化策略（4节）

### 定位
模型搭建好了，怎么训练？这一章深入训练循环的每一个环节：损失函数、优化器、学习率调度、梯度裁剪、混合精度训练。这些知识无论用手写循环、Lightning 还是 HF Trainer 都通用。

### 04-01 手写标准训练循环
- **最小可行训练循环（~30 行）**：
  ```python
  model = GPT(config).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
  criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

  for epoch in range(num_epochs):
      for batch in train_loader:
          input_ids = batch['input_ids'].to(device)
          labels = batch['labels'].to(device)
          logits = model(input_ids)
          loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
  ```
- **逐步完善训练循环**：
  - 添加 Gradient Clipping
  - 添加 Gradient Accumulation
  - 添加 Learning Rate Scheduling
  - 添加 Evaluation
  - 添加 Logging
  - 添加 Checkpoint Saving
  - 添加 Early Stopping
- **TrainingState 数据类**：封装 epoch / global_step / best_loss / patience
- **Epoch vs Iteration vs Step 的术语辨析**

### 04-02 优化器详解——AdamW 及其变体
- **SGD → Adam → AdamW 的进化**：
  - AdamW：Decoupled Weight Decay — **LLM 训练的事实标准**
- **AdamW 关键参数**：
  - `lr=3e-4`, `betas=(0.9, 0.999)`, `weight_decay=0.01`, `eps=1e-8`
- **其他优化器（了解即可）**：
  - AdamW8bit（8-bit 量化状态，省显存）
  - Adafactor（参数少，适合超大规模模型）
  - Lion（动量 + 符号函数，简洁高效）
- **优化器状态字典**：
  - `optimizer.state_dict()` 包含什么？
  - 断点续训的关键

### 04-03 学习率调度与梯度技巧
- **Learning Rate Scheduler**：
  - CosineAnnealingLR（LLM 最常用）
  - CosineWithWarmups（HF Transformers 提供）
  - Warmup 步数选择：总步数的 1-5%
- **Gradient Accumulation（梯度累积）**：
  - 目标：模拟更大的 batch size
  - Loss 缩放：`loss = loss / accumulation_steps`
- **Gradient Clipping（梯度裁剪）**：
  - `clip_grad_norm_`（按范数裁剪，推荐）
  - `max_norm=1.0`
- **Mixed Precision Training（混合精度训练）**：
  - FP16 vs BF16（推荐 BF16）
  - `torch.cuda.amp.autocast` + `GradScaler`
  - PyTorch 2.0+ 的 `torch.amp` 新接口

### 04-04 训练监控与调试
- **Logging 方案**：
  - Python logging + tqdm（最简方案）
  - TensorBoard / W&B（可视化）
  - JSON Lines（轻量离线）
- **关键监控指标**：
  - Train/Val Loss 曲线、Learning Rate、Gradient Norm
  - Throughput (tokens/sec)、GPU Utilization & Memory
- **常见训练异常诊断**：
  - Loss = NaN → lr 过大 / FP16 溢出 / 数据异常值
  - Loss 不下降 → lr 过小 / 未 shuffle / 模型 bug
  - Val Loss >> Train Loss → 过拟合
  - Loss 震荡剧烈 → batch 太小 / lr 太大
  - GPU OOM → 减小 batch / gradient checkpointing / 量化
- **Checkpoint 管理**：
  - 保存频率、保存内容、Top-K 保留策略
  - 早停逻辑：patience + min_delta

---

## 第05章：PyTorch Lightning——训练编排利器（4节）

### 定位
当你的训练循环变得复杂（多任务、多优化器、复杂 logging、分布式训练），手写循环的维护成本急剧上升。PyTorch Lightning 将**训练逻辑与工程代码分离**，让你专注于模型本身。

### 05-01 Lightning 核心哲学与第一个模型
- **Lightning 解决什么问题？**
  - 手写循环的痛点：设备管理、分布式、checkpoint、logging、半精度……样板代码太多
  - 设计哲学：`LightningModule` 封装模型+优化+逻辑，`Trainer` 封装训练循环+硬件
- **从手写到 Lightning 的迁移对照**：
  ```python
  # 手写版 (~80 行样板代码)
  # Lightning 版 (~25 行核心逻辑)
  class LitModel(LightningModule):
      def training_step(self, batch, batch_idx):
          loss = self.compute_loss(batch)
          self.log("train_loss", loss)
          return loss

  trainer = Trainer(max_epochs=10, accelerator="gpu")
  trainer.fit(lit_model, train_loader, val_loader)
  ```
- **LightningModule vs nn.Module**：
  - 继承关系：`LightningModule` → `nn.Module`（完全兼容）
  - 新增方法：`training_step()`, `validation_step()`, `configure_optimizers()`
  - 新增钩子：20+ lifecycle hooks
- **Trainer 核心参数速览**：
  - `max_epochs`, `gradient_clip_val`, `accumulate_grad_batches`
  - `accelerator` ("cpu"/"gpu"/"tpu"/"mps"), `devices`
  - `precision` ("16-mixed"/"bf16-mixed"/"32"), `logger`

### 05-02 Lightning 生命周期与回调系统
- **完整生命周期（Lifecycle Hooks）**：
  ```
  prepare_data() → setup()
  → [epoch_start → [batch_start → training_step → ...] × N batches → epoch_end] × M epochs
  ```
  - 每个 hook 的触发时机与典型用途表格
  - 重点 hook 详解：`on_fit_start/end`, `on_train_batch_start`, `on_before/after_backward`
- **Callback（回调）系统**：
  - 内置 Callbacks：
    - `ModelCheckpoint`：自动保存最佳模型（Top-K / 按 metric 选择）
    - `EarlyStopping`：早停（monitor + patience + min_delta）
    - `LearningRateMonitor`：记录 lr 变化
    - `StochasticWeightAveraging(SWA)`：权重平均提升泛化
  - 自定义 Callback 示例：GradientMonitoring（每 N 步记录梯度范数）
  - Callback 组合与执行顺序
- **Logger 系统**：
  - `TensorBoardLogger`, `CSVLogger`, `WandbLogger`
  - `self.log()` vs `self.log_dict()` vs `self.experiment.log()`

### 05-03 Lightning 中的高级训练特性
- **多优化器 / 多 LR Scheduler**：
  - GAN 训练（Generator + Discriminator 各自优化器）
  - 分层学习率：head 层 lr×10，body 层 lr×1
  - `configure_optimizers()` 返回多种形式
- **多 DataLoader**：`train_dataloader()` + `val_dataloader()` + `test_dataloader()`
- **分布式策略**：
  - `strategy="ddp"` — DDP（多 GPU）
  - `strategy="fsdp"` — FSDP（超大模型）
  - `strategy="deepspeed"` — DeepSpeed ZeRO
  - `strategy="auto"` — 自动选择最优策略
- **Gradient Checkpointing**：
  - `gradient_checkpointing=True`
  - 显存节省 ~30-50%，训练时间增加 ~20%
  - 大模型训练必备（70B+ 几乎必须启用）
- **半精度配置**：
  - `precision="16-mixed"` (FP16 + GradScaler)
  - `precision="bf16-mixed"` (BF16，H100/Ampere 推荐)

### 05-04 Lightning 实战：用 Lightning 训练我们的 GPT 模型
- **将第 3 章的手写 GPT 改造为 LightningModule**：
  ```python
  class LitGPT(LightningModule):
      def __init__(self, config):
          super().__init__()
          self.model = GPT(config)
          self.config = config

      def training_step(self, batch, batch_idx):
          input_ids = batch['input_ids']
          labels = batch['labels']
          logits = self.model(input_ids)
          loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
          self.log('train_loss', loss, prog_bar=True)
          return loss

      def configure_optimizers(self):
          return torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

      def configure_callbacks(self):
          return [ModelCheckpoint(monitor='val_loss', save_top_k=3), EarlyStopping(monitor='val_loss', patience=5)]
  ```
- **完整训练启动**：
  ```python
  trainer = Trainer(
      max_epochs=10, accelerator="gpu", devices=1,
      precision="bf16-mixed", gradient_clip_val=1.0,
      accumulate_grad_batches=4,
      logger=WandbLogger(project="lit-gpt"),
  )
  trainer.fit(model, train_loader, val_loader)
  ```
- **Lightning vs 手写循环完整对比表**：

  | 维度 | 手写循环 | Lightning |
  |-----|---------|-----------|
  | 代码量 | 80-150 行 | 25-40 行核心逻辑 |
  | 设备管理 | 手动 .to(device) | 自动 |
  | 分布式 | 手写 DDP | `strategy="ddp"` |
  | 半精度 | 手动 autocast | `precision="bf16"` |
  | Checkpoint | 手动 save/load | ModelCheckpoint callback |
  | Logging | 手动集成 | `self.log()` + Logger |
  | 灵活性 | 100% | 90%（极端情况需 override） |

---

## 第06章：HuggingFace Trainer 与 LLM 微调实战（5节）

### 定位
这是**最贴近实战的一章**。虽然我们学会了手写循环和 Lightning，但在 LLM 微调领域，HuggingFace `Trainer` + `PEFT` 是工业界的事实标准。

### 06-1 Trainer API 速成——三步微调一个模型
- **为什么 LLM 微调首选 Trainer？**
  - 与 Model Hub 无缝集成
  - 内置 PEFT/LoRA 支持
  - 内置 BitsAndBytes 量化
  - 社区生态成熟
- **最快上手（3 步）**：
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
  from peft import LoraConfig, get_peft_model

  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

  lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj"])
  model = get_peft_model(model, lora_config)

  args = TrainingArguments(output_dir="./output", per_device_train_batch_size=4, ...)
  trainer = Trainer(model=model, args=args, train_dataset=train_data)
  trainer.train()
  ```
- **TrainingArguments 完整参数指南**：
  - 训练控制：`num_train_epochs`, `max_steps`, `per_device_train/eval_batch_size`
  - 学习率：`learning_rate`, `weight_decay`, `warmup_ratio/steps`, `lr_scheduler_type`
  - 优化：`optim` ("adamw_torch"/"adamw_hf"/"adafactor"), `gradient_checkpointing`
  - 精度：`fp16`, `bf16`, `tf32`
  - 日志：`logging_steps`, `report_to`
  - 保存：`save_strategy`, `save_total_limit`
  - 分布式：`fsdp`, `deepspeed`
- **DataCollator 系列**：
  - `DataCollatorForLanguageModeling`：语言模型训练（MLM / CLM）
  - `DataCollatorForSeq2Seq`：Seq2Seq 任务
  - 自定义 DataCollator

### 06-2 PEFT 与 LoRA 微调深度实践
- **LoRA 原理回顾**（从 PyTorch 角度重新审视）：
  - $W' = W + BA$，其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$
  - rank $r \ll d$（通常 r=8/16/32/64，而 d=4096）
  - 可训练参数占比：≈ 0.1%-1%
- **PEFT 库核心 API**：
  - `LoraConfig` 配置项详解：
    - `r`, `lora_alpha` (通常=2r)
    - `target_modules`（**最重要！**哪些层加 LoRA）
    - `lora_dropout`, `bias`, `modules_to_save`, `task_type`
  - `get_peft_model()`, `PeftModel.save_pretrained()`, `merge_and_unload()`
- **target_modules 选择指南**：
  - LLaMA/Qwen/Mistral 架构：`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - 如何查看层名：`for name, param in model.named_parameters(): print(name)`
  - 推荐：仅 Attention 层（性价比最高）或 Attention + FFN（效果最好）
- **完整 LoRA 微调流水线**：
  - 数据准备：指令格式化（Alpaca / ShareGPT 格式）
  - Tokenization：`tokenizer.apply_chat_template()`
  - 训练配置最佳实践模板
  - 监控：W&B 集成
  - 评估：PPL / Benchmark / 人工评估
  - 导出：合并 LoRA 权重

### 06-3 量化训练与 QLoRA
- **为什么需要量化训练？**
  - 7B FP14 = 14GB（单卡 A100 可以）
  - 70B FP16 = 140GB（即使 4×A100 也紧张）
  - 70B INT4 = ~37GB（单卡可以！）
- **BitsAndBytes 量化配置**：
  ```python
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
  )
  model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config, device_map="auto")
  ```
- **QLoRA（Quantized LoRA）**：
  - 4-bit Base Model + LoRA 微调
  - QLoRA 性能接近全量 16-bit 微调
  - RTX 4090 可微调 7B 模型！48GB 可微调 34B
- **NF4 vs FP4 vs INT4** 对比
- **双重量化（Double Quantization）**：额外节省 ~0.4 bits/param

### 06-4 三种训练方式终极对比
- **同一任务的三种实现**：任务 = Alpaca 数据集微调 Qwen2.5-7B
  - 方式 A：纯 PyTorch 手写循环（~200 行）
  - 方式 B：PyTorch Lightning（~120 行）
  - 方式 C：HuggingFace Trainer + PEFT（~60 行）
- **全方位对比表**：

  | 维度 | 手写 PyTorch | Lightning | HF Trainer |
  |-----|-------------|-----------|------------|
  | 代码量 | 200 行 | 120 行 | 60 行 |
  | 上手难度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
  | 灵活性 | 100% | 85% | 60% |
  | LoRA 支持 | 需自己实现 | 需配合 PEFT | **内置** |
  | 量化训练 | 需自己实现 | 需配合 BnB | **内置** |
  | Model Hub | 需自己处理 | 需自己处理 | **一键 push** |
  | 适用场景 | 研究/论文/极致定制 | 复杂流程/多任务 | **LLM 微调首选** |

- **选型决策指南**：
  - Research / 发论文 → Lightning
  - LLM 微调 / 产品落地 → **HF Trainer + PEFT**
  - 学习理解原理 → 先手写 → 再 Lightning → 最后 Trainer
  - 极致性能优化 / 自定义 CUDA kernel → 手写 PyTorch

### 06-5 微调后的模型评估与导出
- **评估方法**：
  - Perplexity（困惑度）
  - 自动 Benchmark：MMLU / C-Eval / CMMLU / HumanEval
  - 生成质量评估：人工评分 / GPT-as-Judge / Arena
- **模型导出格式**：
  - HF 格式：`save_pretrained()` → transformers/vLLM/Ollama 兼容
  - 合并 LoRA：`model.merge_and_unload()`
  - GGUF 格式（Ollama 兼容）
- **模型上传 HuggingFace Hub**：
  - `push_to_hub()` 一键上传
  - Model Card 编写

---

## 第07章：分布式训练——突破单卡极限（4节）

### 定位
单卡跑不动 70B 模型？训练太慢？分布式训练是必经之路。本章覆盖 DDP、FSDP、DeepSpeed 三大主流方案。

### 07-1 DDP（DistributedDataParallel）基础
- **DP vs DDP vs FSDP**：
  - DataParallel (DP)：主卡瓶颈，已废弃
  - DistributedDataParallel (DDP)：每卡一份模型副本，AllReduce 同步梯度
  - Fully Sharded Data Parallel (FSDP)：模型分片到各卡，显存效率最高
- **DDP 核心原理**：
  - 每个进程持有完整模型副本
  - 反向后 AllReduce 汇聚平均梯度
  - 通信开销 ∝ 模型参数量
- **手写 DDP 训练**：
  ```python
  dist.init_process_group(backend="nccl")
  local_rank = int(os.environ["LOCAL_RANK"])
  model = GPT(config).to(local_rank)
  model = DDP(model, device_ids=[local_rank])

  sampler = DistributedSampler(train_dataset, rank=local_rank)
  loader = DataLoader(dataset, sampler=sampler, ...)

  for batch in loader:
      loss = model(batch['input_ids'].to(local_rank))
      loss.backward(); optimizer.step()

  dist.destroy_process_group()
  ```
  - 启动命令：`torchrun --nproc_per_node=4 train.py`
- **DDP 注意事项**：
  - Batch Size 是 per-device
  - Learning Rate 线性缩放
  - `DistributedSampler.set_epoch()` 必须 each epoch 调一次
  - `find_unused_parameters`（LoRA 微调时可能需要 True）

### 07-2 FSDP（Fully Sharded Data Parallel）
- **FSDP 解决什么问题？**
  - 70B FP16 = 140GB，4 卡 A80 存不下
  - FSDP 将参数、梯度、优化器状态全部切分
  - 显存 ≈ 总量 / num_gpus
- **FSDP 工作原理**：
  - All-Gather（前向时收集当前层完整参数）
  - Compute
  - Reduce-Scatter（反向时分散梯度）
- **Sharding Strategies**：
  - FULL_SHARD：全部分片（最省显存）
  - SHARD_GRAD_OP：切分梯度和优化器状态
  - NO_SHARD：等价于 DDP
- **在 Lightning 中使用 FSDP**：
  ```python
  trainer = Trainer(strategy=FSDPStrategy(
      sharding_strategy="FULL_SHARD",
      auto_wrap_policy=transformer_auto_wrap_policy(GPTBlock),
      state_dict_type="full",
  ), devices=4)
  ```
- **FSDP vs DDP 选型**：
  - 7B 以下 → DDP（更快）
  - 13B+ → **FSDP**（必须用）

### 07-3 DeepSpeed ZeRO——微软的大规模训练方案
- **ZeRO 三个阶段**：
  | 阶段 | 分片内容 | 显存节省 | 通信开销 |
  |-----|---------|---------|---------|
  | ZeRO-1 | 仅优化器状态 | ~4x | 低 |
  | ZeRO-2 | 优化器状态 + 梯度 | ~8x | 中 |
  | ZeRO-3 | 全部切分 | **~N×** | 高 |
- **ZeRO-Offload / ZeRO-Infinity**：CPU/NVMe Offloading
- **DeepSpeed 配置文件 (ds_config.json)**：
  ```json
  {
    "fp16": {"enabled": true},
    "zero_optimization": {
      "stage": 2,
      "offload_optimizer": {"device": "cpu"}
    },
    "gradient_clipping": 1.0
  }
  ```
- **DeepSpeed vs FSDP**：
  - 一般情况 → FSDP（PyTorch 原生）
  - 超大规模 (100B+) 或需 Offload → **DeepSpeed**

### 07-4 分布式训练实战与排错
- **端到端实战：4×A100 训练 7B 模型**
  - 环境准备：NCCL / SSH / CUDA
  - 启动方式对比：`torchrun` vs `accelerate launch` vs `deepspeed`
  - 完整训练脚本模板（支持 DDP/FSDP/DeepSpeed 切换）
- **性能基准**：
  - 弱扩展 / 强扩展效率
  - MFU (Model FLOPs Utilization)
  - TFLOPS/GPU 实测参考
- **常见错误排查**：
  - NCCL timeout → 网络/防火墙
  - Device mismatch → DDP 包装时机错误
  - Address already in use → 端口冲突
  - OOM on Rank 0 only → 数据分布不均
  - Hang/deadlock → Collective 操作不匹配
- **成本估算公式**：`total_time = (steps × time_per_step) / num_gpus × scaling_efficiency`

---

## 第08章：推理优化——让模型跑得更快（4节）

### 定位
训练好的模型要投入生产，推理速度和资源消耗至关重要。本章覆盖从 `torch.compile` 到量化、从导出到部署的全链路优化。

### 08-1 torch.compile 与图编译优化
- **PyTorch 2.0 的革命性变化**：
  - `torch.compile()`：TorchDynamo + TorchInductor
  - **无需修改模型代码**，一行 `model = torch.compile(model)` 即可
- **工作原理**：
  - Dynamo：捕获 bytecode → FX Graph
  - Inductor：FX Graph → Triton/C++ kernels（融合算子）
- **使用方式与编译模式**：
  - `"default"` / `"reduce-overhead"` / `"max-autotune"`
- **性能实测**：
  - GPT-2 Small：加速 10-30%
  - LLaMA 7B：加速 15-50%
  - 注意：首次编译耗时 1-5 分钟
- **限制与调试**：不支持 dynamic shapes 的操作、`TORCH_LOGS="+dynamo"`

### 08-2 模型量化——减小体积与加速推理
- **PyTorch 原生量化支持**：
  - `torch.quantization`（动态/静态量化）
  - `torchao`（最新量化库，FP8/INT4/int4_weight_only）
- **INT8 动态量化（最简单）**：
  ```python
  model_q = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
  ```
- **INT4 权重量化（LLM 主流）**：
  ```python
  from torchao.quantization import quantize, int4_weight_only
  model_int4 = quantize(model, int4_weight_only())  # 7B: 14GB → ~4.5GB
  ```
- **FP8 量化（Hopper GPU 原生）**：
  - E4M3fn / E5M2
  - `torch.float8` + `torchao`
- **量化质量对比表**（FP16 / INT8 Dynamic / INT4 Weight-only / FP8 / AWQ INT4）

### 08-3 模型导出与部署格式
- **ONNX 导出**：`torch.onnx.export()` + opset_version + dynamic_axes
- **ExportedProgram（PyTorch 2.x 新格式）**：`torch.export()`
- **SafeTensors 格式**：HF 标准，安全快速
- **GGUF 格式（Ollama 兼容）**：llama.cpp 转换链路
- **部署目标平台**：
  - vLLM：HF 格式直接用
  - Ollama：GGUF 格式
  - TensorRT：NVIDIA GPU 极致推理
  - ONNX Runtime：跨平台

### 08-4 推理性能分析与优化 Checklist
- **性能剖析工具**：
  - PyTorch Profiler：`torch.profiler.profile()` + TensorBoard
  - `torch.utils.benchmark`：微观 benchmark
  - `nsys`（NVIDIA System Profiler）：GPU 级剖析
- **推理优化决策树**：
  ```
  模型太大？→ 量化(INT4/FP8) → 还不够？→ FSDP 推理 / 多卡
  推理慢？→ torch.compile → FlashAttention → 量化 → vLLM
  CPU 推理？→ ONNX Runtime / OpenVINO
  移动端？→ CoreML / TFLite / ExecuTorch
  ```
- **端到端优化 Checklist**（15 项检查清单）

---

## 第09章：生产部署与 MLOps（3节）

### 定位
模型训练好了，怎么变成可靠的服务？本章将 PyTorch 模型推向生产环境，涵盖 TorchServe、FastAPI 服务化、Docker 化和监控。

### 09-1 TorchServe 与模型服务化
- **TorchServe 简介**：
  - AWS + Meta 出品的官方模型服务框架
  - 支持多模型并发、动态批处理、自动扩缩容
- **模型打包与部署**：
  - `torch-model-archiver` 创建 .mar 文件
  - `torchserve --start --models my_model=model.mar`
  - 自定义 Handler：`handle()` 方法处理推理请求
- **Inference API**：
  - RESTful API：POST /predictions/{model_name}
  - gRPC API：高性能调用
  - Management API：模型注册/注销/缩放
- **TorchServe vs 自建 FastAPI 服务**：
  - TorchServe：开箱即用，适合标准场景
  - FastAPI：灵活可控，适合自定义需求
  - vLLM：LLM 专用，性能最强（详见 vLLM 教程）

### 09-2 FastAPI + PyTorch 构建推理服务
- **最小可用服务**：
  ```python
  from fastapi import FastAPI
  import torch

  app = FastAPI()
  model = GPT(config).to("cuda").eval()

  @app.post("/generate")
  async def generate(prompt: str, max_tokens: int = 100):
      with torch.no_grad():
          input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
          output = model.generate(input_ids, max_new_tokens=max_tokens)
          return {"text": tokenizer.decode(output[0])}
  ```
- **生产级增强**：
  - 异步处理：`asyncio` + `BackgroundTask`
  - 批量推理：请求队列 + 动态 batching
  - 模型热加载：无需重启更新模型
  - 连接池管理
- **Docker 化**：
  - Dockerfile 最佳实践（多阶段构建）
  - NVIDIA Container Toolkit 配置
  - docker-compose 编排
- **性能优化**：
  - `torch.inference_mode()` 替代 `torch.no_grad()`
  - `torch.compile()` 加速
  - 半精度推理
  - 流式输出（SSE）

### 09-3 监控、日志与持续集成
- **Prometheus + Grafana 监控栈**：
  - 自定义 Metrics：请求延迟、QPS、GPU 使用率、错误率
  - Prometheus Python client 集成
  - Grafana Dashboard 模板
- **结构化日志**：
  - JSON 格式日志 + request_id 追踪
  - Loguru / structlog 库推荐
  - ELK / Loki 聚合
- **CI/CD 流水线**：
  - GitHub Actions 自动化训练
  - 模型版本管理（MLflow / DVC）
  - 自动化测试：单元测试 / 集成测试 / 回归测试
  - 自动化部署：Docker → K8s
- **A/B 测试框架**：
  - 同时运行多个模型版本
  - 流量分配与结果收集
  - 统计显著性检验

---

## 附录

### A. PyTorch 版本演进与 LLM 相关新特性
- PyTorch 1.13 → 2.0 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5
- 每个版本的 LLM 相关重要更新
- `torch.compile` 成熟历程
- SDPA (Scaled Dot Product Attention) 内置
- `torchao` 量化库

### B. 常用代码模板速查
- 标准 training loop 模板
- LightningModule 模板
- HF Trainer + PEFT 模板
- DDP 训练模板
- FSDP 配置模板
- DeepSpeed ds_config.json 模板
- FastAPI 推理服务模板

### C. GPU 内存计算速查
- 模型参数量 → FP16/BF16/INT4 显存占用
- KV Cache 显存估算
  - 公式：`layers × 2(KV) × num_kv_heads × head_dim × seq_len × dtype_bytes`
  - 各模型（7B/13B/34B/70B）在不同 seq_len 下的 KV Cache 大小表
- Batch Size × Seq Length → 激活值显存估算
- Gradient Checkpointing 显存节省计算

### D. 常见错误速查表
- `CUDA out of memory` → 解决方案树
- `RuntimeError: Expected all tensors to be on the same device` → 设备不一致排查
- `NCCL timeout` → 网络/防火墙排查
- `Loss is NaN` → 逐步定位（数据/学习率/数值类型）
- DDP hang → Collective operation 不匹配
- `torch.compile` 报错 → dynamo debug 模式

### E. PyTorch → Lightning → HF Trainer → vLLM 技术选型决策树
```
你要做什么？
├── 从零理解 LLM 原理
│   └── 🎯 纯 PyTorch 手写 Transformer（Ch3）
│
├── 训练/微调一个 LLM
│   ├── 快速微调（LoRA/QLoRA）
│   │   └── 🎯 HF Trainer + PEFT（Ch6）
│   ├── 复杂训练流程（多任务/GAN/研究实验）
│   │   └── 🎯 PyTorch Lightning（Ch5）
│   └── 极致定制（自定义 CUDA kernel / 新架构）
│       └── 🎯 纯 PyTorch 手写循环（Ch4）
│
├── 单卡跑不动 / 要加速训练
│   └── 🎯 分布式训练 DDP/FSDP/DeepSpeed（Ch7）
│
├── 推理部署（高性能服务）
│   ├── < 100 并发 / 单模型
│   │   └── FastAPI + torch.compile（Ch9）
│   ├── > 100 并发 / 生产环境
│   │   └── 🎯 vLLM（详见 vLLM 教程）
│   └── 边缘设备 / 移动端
│       └── ONNX Runtime / CoreML / ExecuTorch
│
└── 学习路线建议
    ├── 初学者: Ch1 → Ch2 → Ch3 → Ch4（手写循环）→ Ch6（HF Trainer）
    ├── 进阶者: Ch1-3 → Ch5（Lightning）→ Ch6 → Ch7（分布式）
    └── 专家: Ch3（源码级）→ Ch7（FSDP/DeepSpeed）→ Ch8（推理优化）
```

### F. 学习路线图
- **第 1-2 周（入门）**：Ch1 张量基础 + Ch2 数据管道 + Ch3 构建 MiniGPT
- **第 3-4 周（进阶）**：Ch4 训练循环 + Ch5 Lightning + 用 Lightning 重训 MiniGPT
- **第 5-6 周（实战）**：Ch6 HF Trainer + LoRA 微调真实 7B 模型
- **第 7-8 周（高级）**：Ch7 分布式训练 + Ch8 推理优化 + Ch9 生产部署
- **持续（精通）**：阅读 LLaMA/Qwen 源码 → 自定义算子 → 参与开源贡献

---

## 统计信息

| 统计项 | 数量 |
|-------|------|
| 总章数 | 9 章 |
| 总节数 | **38 节** |
| 附录 | 6 个 |
| 预计总字数 | ~12-15 万字 |
| 核心代码示例 | 50+ 个完整可运行程序 |
| 覆盖工具 | PyTorch / Lightning / HF Trainer / PEFT / DeepSpeed / TorchServe / FastAPI |
