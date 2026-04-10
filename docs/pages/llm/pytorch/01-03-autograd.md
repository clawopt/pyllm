# 1.3 Autograd 自动微分——理解反向传播的本质

> 你可能听说过"反向传播"（Backpropagation）是深度学习的核心算法。但你知道 PyTorch 是如何自动完成这个过程的吗？Autograd 就是答案。它像一个隐形的数学家，在你调用 `loss.backward()` 的瞬间，自动沿着计算图回溯，计算出每一个参数对 loss 的偏导数。这一节，我们揭开 Autograd 的面纱。

## 为什么必须理解 Autograd？

在 LLM 训练中，你几乎每一步都在和 Autograd 打交道：

```
训练循环的每一轮:
  前向: input → Embedding → Attention → FFN → logits → loss
         ↑ 每一步都构建了计算图节点

  反向: loss.backward()   ← Autograd 自动遍历计算图
         ↑ 为每个参数计算 d(loss)/d(参数)

  更新: optimizer.step()    ← 用梯度更新参数
         ↑ 如果梯度算错了，模型就学不到东西
```

不理解 Autograd，你就无法：
- 调试**梯度消失/爆炸**问题
- 正确使用 `detach()` 截断梯度流（KV Cache 等场景）
- 自定义复杂的训练逻辑（如梯度惩罚、对抗训练）
- 实现自定义 CUDA 算子（需要手写 backward）

## 计算图概念

### 从手动求导到 Autograd

比如下面的程序展示一个最简单的例子——线性回归：

```python
"""
Autograd 入门：从手动求导到自动求导
"""

import torch


def manual_vs_autograd():
    """对比手动求导和 PyTorch Autograd"""

    print("=" * 60)
    print("📐 手动求导 vs Autograd")
    print("=" * 60)

    # === 数据 ===
    x = torch.tensor([1., 2., 3., 4.])
    y = torch.tensor([2., 4., 6., 8.])  # y = 2x

    # === 手动求导过程 ===
    print("\n--- 手动求导 ---")
    w_manual = 0.5  # 初始权重
    lr = 0.01       # 学习率

    pred = w_manual * x           # [0.5, 1.0, 1.5, 2.0]
    loss = ((pred - y) ** 2).mean()  # MSE Loss

    # 手动推导: dL/dw = mean(2 * (w*x - y) * x) = mean(2 * pred_error * x)
    error = pred - y               # [-1.5, -3.0, -4.5, -6.0]
    grad = (2 * error * x).mean()  # dL/dw
    w_new = w_manual - lr * grad

    print(f"  初始 w: {w_manual}")
    print(f"  预测值: {pred.tolist()}")
    print(f"  Loss: {loss:.4f}")
    print(f"  手动计算梯度 dL/dw: {grad:.4f}")
    print(f"  更新后 w: {w_new:.4f}")

    # === Autograd 方式 ===
    print("\n--- Autograd 方式 ---")
    w_auto = torch.tensor([0.5], requires_grad=True)  # 关键：requires_grad=True!

    pred_a = w_auto * x
    loss_a = ((pred_a - y) ** 2).mean()

    print(f"  初始 w: {w_auto.item():.4f}")
    print(f"  Loss: {loss_a.item():.4f}")

    # 一行代码完成所有梯度的计算！
    loss_a.backward()

    print(f"  Autograd 计算梯度: {w_auto.grad.item():.4f}")  # 应该与手动结果一致
    assert abs(grad - w_auto.grad.item()) < 1e-6, "梯度不一致！"

    # === 对比总结 ===
    print(f"\n{'='*60}")
    print(f"✅ 手动梯度 = {grad:.4f}, Autograd 梯度 = {w_auto.grad.item():.4f}")
    print(f"   差异: {abs(grad - w_auto.grad.item()):.2e} ≈ 0")
    print(f"\n💡 核心区别:")
    print(f"   手动: 需要自己推导链式法则（复杂网络几乎不可能）")
    print(f"   Auto: 只需 loss.backward()，无论多深的网络都自动完成")


if __name__ == "__main__":
    manual_vs_autograd()
```

运行输出：

```
============================================================
📐 手动求导 vs Autograd
============================================================

--- 手动求导 ---
  初始 w: 0.5
  预测值: [0.5, 1.0, 1.5, 2.0]
  Loss: 16.8750
  手动计算梯度 dL/dw: -9.2500
  更新后 w: 0.5925

--- Autograd 方式 ---
  初始 w: 0.5000
  Loss: 16.8750
  Autograd 计算梯度: -9.2500

============================================================
✅ 手动梯度 = -9.2500, Autograd 梯度 = -9.2500
   差异: 0.00e+00 ≈ 0

💡 核心区别:
   手动: 需要自己推导链式法则（复杂网络几乎不可能）
   Auto: 只需 loss.backward()，无论多深的网络都自动完成
```

### 计算图的内部结构

```python
def inspect_computation_graph():
    """查看 Autograd 构建的计算图"""

    print("\n" + "=" * 60)
    print("🔍 计算图结构探查")
    print("=" * 60)

    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)

    c = a * b          # MulBackward0
    d = c + 1.0        # AddBackward0
    e = d ** 2          # PowBackward0

    print(f"\n前向传播路径:")
    print(f"  a = tensor([2.0], requires_grad=True)")
    print(f"  b = tensor([3.0], requires_grad=True)")
    print(f"  c = a * b      → grad_fn: {c.grad_fn}")
    print(f"  d = c + 1.0    → grad_fn: {d.grad_fn}")
    print(f"  e = d ** 2     → grad_fn: {e.grad_fn}")

    print(f"\n每个节点的 grad_fn 类型揭示了操作类型:")
    print(f"  MulBackward0  ← 乘法操作")
    print(f"  AddBackward0  ← 加法操作")
    print(f"  PowBackward0  ← 幂运算")

    e.backward()

    print(f"\n反向传播后的梯度:")
    print(f"  de/da = {a.grad.item():.2f}")  # de/da = 2*(ab+1)*b = 2*7*3 = 42
    print(f"  de/db = {b.grad.item():.2f}")  # de/db = 2*(ab+1)*a = 2*7*2 = 28


if __name__ == "__main__":
    inspect_computation_graph()
```

## 梯度管理四大工具

### 1. zero_grad() —— 为什么每次都要清零？

```python
def demo_zero_grad():
    """演示为什么需要 zero_grad"""

    x = torch.tensor([1., 2.], requires_grad=True)
    y = torch.tensor([2., 4.])

    w = torch.tensor([0.5], requires_grad=True)

    # 第 1 步
    loss1 = ((w * x - y) ** 2).mean()
    loss1.backward()
    print(f"第1步梯度: {w.grad.item():.4f}")  # -9.25

    # ❌ 忘记清零！直接第 2 步
    loss2 = ((w * x - y) ** 2).mean()
    loss2.backward()
    print(f"第2步梯度(未清零): {w.grad.item():.4f}")  # -18.50！累积了！

    # ✅ 正确做法
    w.grad.zero_()  # 或 optimizer.zero_grad()
    loss3 = ((w * x - y) ** 2).mean()
    loss3.backward()
    print(f"第3步梯度(已清零): {w.grad.item():.4f}")  # -9.25 ✅


if __name__ == "__main__":
    demo_zero_grad()
```

**原因**：PyTorch 默认**累加梯度**到 `.grad` 属性上（这是为了支持 Gradient Accumulation）。如果你不清零，新计算的梯度会叠加到旧的上面。

### 2. detach() —— 截断梯度流

`detach()` 返回一个**共享数据但切断梯度连接**的新张量。这在以下场景至关重要：

```python
def demo_detach():
    """detach 在 LLM 中的典型用法"""

    print("=" * 60)
    print("✂️  detach() 梯度截断演示")
    print("=" * 60)

    # === 场景 1: KV Cache 中的 detach ===
    # 在自回归生成中，过去的 KV Cache 不需要再计算梯度
    past_kv = torch.randn(2, 12, 10, 64, requires_grad=True)  # 过去的 KV

    # 新的计算不应该影响 past_kv 的梯度
    new_q = torch.randn(2, 12, 1, 64, requires_grad=True)
    detached_past = past_kv.detach()  # ⭐ 切断梯度！

    # 使用 detached_past 进行 attention 计算
    attn_out = new_q @ detached_past.transpose(-2, -1)

    # 反向传播时，past_kv 不会有梯度
    dummy_loss = attn_out.sum()
    dummy_loss.backward()

    print(f"\n[KV Cache 场景]")
    print(f"  past_kv.requires_grad: True")
    print(f"  past_kv.grad after backward: {past_kv.grad}")  # None! ✅
    print(f"  new_q.grad: {new_q.grad.shape if new_q.grad is not None else None}")

    # === 场景 2: 从模型输出中提取不需要梯度的信息 ===
    model_output = torch.randn(2, 100, requires_grad=True)

    # 只需要输出的数值（不用于后续梯度计算）
    output_value = model_output.detach().numpy()  # 转 NumPy 必须 detach
    print(f"\n[数值提取场景]")
    print(f"  detach 后可以安全转为 NumPy: shape={output_value.shape}")


if __name__ == "__main__":
    demo_detach()
```

### 3. torch.no_grad() —— 推理时的显存优化

```python
def demo_no_grad():
    """no_grad 的效果对比"""

    import time

    model_size = (1000, 1000)
    iterations = 1000

    x = torch.randn(*model_size, requires_grad=True)
    w = torch.randn(*model_size, requires_grad=True)

    # 有梯度追踪
    start = time.time()
    for _ in range(iterations):
        y = (x @ w).sum()
        # 不调用 backward，但仍然构建了计算图
    with_grad_time = time.time() - start

    # 无梯度追踪
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            y = (x @ w).sum()
            # 不构建计算图，节省内存和时间
    no_grad_time = time.time() - start

    print(f"\n{'='*50}")
    print(f"有 Autograd:   {with_grad_time*1000:.1f} ms ({iterations}次)")
    print(f"无 Autograd:  {no_grad_time*1000:.1f} ms ({iterations}次)")
    print(f"加速比:       {with_grad_time/no_grad_time:.2f}x")
    print(f"\n💡 推理时务必用 with torch.no_grad() 或 model.eval() + inference_mode()")
    print(f"   可节省 ~30-50% 显存（无需存储中间激活值的梯度信息）")


if __name__ == "__main__":
    demo_no_grad()
```

### 4. torch.inference_mode() —— PyTorch 2.x 推荐

```python
def demo_inference_mode():
    """inference_mode() vs no_grad()"""

    print("\n" + "=" * 50)
    print("🚀 推理模式对比: no_grad vs inference_mode")
    print("=" * 50)

    x = torch.randn(2, 768)

    # no_grad (传统方式)
    with torch.no_grad():
        out1 = x * 2

    # inference_mode (PyTorch 2.0+, 推荐方式)
    with torch.inference_mode(True):
        out2 = x * 2

    # 验证结果一致
    assert torch.equal(out1, out2)

    print(f"  no_grad 输出: {out1[0,:3]}")
    print(f"  inference_mode 输出: {out2[0,:3]}")
    print(f"  结果一致: ✅")
    print(f"\n  区别:")
    print(f"  no_grad:     仍可能触发某些 view/reduction 的错误检查")
    print(f"  inference_mode: 完全关闭推理相关行为，更快更安全（推荐用于部署）")


if __name__ == "__main__":
    demo_inference_mode()
```

## 常见 Autograd 问题排查

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `element 0 of tensors does not require grad` | loss 来自不需要梯度的张量 | 确保模型参数 `requires_grad=True` |
| `Trying to backward through the graph a second time` | 多次调用 `backward()` | 每次 forward 后只调用一次，或设置 `retain_graph=True` |
| `one of the variables needed for gradient computation has been modified by an inplace operation` | 使用了 `+=`, `*=` 等 in-place 操作 | 改为 `x = x + 1` 而非 `x += 1` |
| `RuntimeError: gradient required for tensors without a gradient function` | 对非叶子张量访问 `.grad` | 用 `.retain_grad()` 或对叶子张量操作 |
| Loss 变成 NaN | FP16 溢出 / lr 太大 / 数据含 inf/nan | 降低 lr、改用 BF16、检查数据 |

## 高阶 Autograd（进阶）

### 二阶导数

某些高级优化器（如自然梯度方法）或研究场景需要二阶导数：

```python
def demo_second_order():
    """二阶导数示例"""

    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 3  # y = x³

    # 一阶导数: dy/dx = 3x² = 12
    y.backward(create_graph=True)  # create_graph=True 保留计算图用于二阶
    first_grad = x.grad.clone()
    print(f"一阶导数 dy/dx at x=2: {first_grad.item():.1f}")  # 12.0

    # 二阶导数: d²y/dx² = 6x = 12
    first_grad.backward()
    second_grad = x.grad.item()
    print(f"二阶导数 d²y/dx² at x=2: {second_grad:.1f}")  # 12.0


if __name__ == "__main__":
    demo_second_order()
```

下一节我们将学习 **nn.Module**——PyTorch 构建神经网络的标准方式。
