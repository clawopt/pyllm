# 训练过程监控与调试：让模型训练不再"黑盒"

## 这一节讲什么？

上一节我们用 Trainer 成功跑通了微调流程，loss 在下降，accuracy 在上升——一切看起来都很完美。但实际项目中，事情往往不会这么顺利。你可能会遇到：**loss 突然爆炸变成 NaN**、**训练 loss 持续下降但验证 loss 反而上升**、**模型对训练集完美拟合但测试集一塌糊涂**、**显存溢出（OOM）**……

这些问题的根源各不相同，但解决它们的方法有迹可循。这一节，我们将系统性地学习如何**监控训练过程**和**诊断常见问题**，让你从"盲人摸象式调参"进化到"有理有据地调试"。

---

## 一、训练曲线的解读艺术

## 1.1 四种典型模式

训练过程中最直观的信息来源就是 loss 曲线。学会读懂不同形状的曲线，是调试的第一步：

```python
import numpy as np
import matplotlib.pyplot as plt

def show_typical_training_curves():
    """展示四种典型的训练曲线模式"""

    steps = np.arange(200)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ===== 模式 1: 正常收敛 (理想情况) =====
    np.random.seed(42)
    train_loss_normal = 1.0 * np.exp(-steps / 40) + 0.02 * np.random.randn(len(steps)) + 0.05
    val_loss_normal = 0.95 * np.exp(-steps / 45) + 0.03 * np.random.randn(len(steps)) + 0.08

    axes[0, 0].plot(steps, train_loss_normal, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(steps, val_loss_normal, 'r--', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('✅ 正常收敛\nTrain 和 Val 同步下降，最终趋于平稳')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ===== 模式 2: 过拟合 (Overfitting) =====
    train_loss_overfit = 1.0 * np.exp(-steps / 20) + 0.01 * np.random.randn(len(steps))
    val_loss_overfit = np.minimum(
        0.9 * np.exp(-steps / 60) + 0.02,
        0.15 + 0.003 * steps + 0.00002 * steps ** 1.5 + 0.02 * np.random.randn(len(steps))
    )

    axes[0, 1].plot(steps, train_loss_overfit, 'b-', label='Train Loss', linewidth=2)
    axes[0, 1].plot(steps, val_loss_overfit, 'r--', label='Val Loss', linewidth=2)
    overfit_point = 80
    axes[0, 1].axvline(x=overfit_point, color='orange', linestyle=':', alpha=0.8,
                    label=f'过拟合开始 (~step {overfit_point})')
    axes[0, 1].fill_between(steps[overfit_point:],
                         train_loss_overfit[overfit_point:], val_loss_overfit[overfit_point:],
                         alpha=0.2, color='red')
    axes[0, 1].set_title('⚠️ 过拟合\nTrain 继续下降但 Val 开始上升\n→ 需要早停/正则化/更多数据')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ===== 模式 3: 欠拟合 (Underfitting) =====
    train_loss_underfit = 1.5 - 0.002 * steps + 0.05 * np.random.randn(len(steps))
    val_loss_underfit = 1.6 - 0.0018 * steps + 0.06 * np.random.randn(len(steps))

    axes[1, 0].plot(steps, train_loss_underfit, 'b-', label='Train Loss', linewidth=2)
    axes[1, 0].plot(steps, val_loss_underfit, 'r--', label='Val Loss', linewidth=2)
    axes[1, 0].axhline(y=0.5, color='green', linestyle='--', alpha=0.7,
                    label='理想最低值')
    axes[1, 0].set_title('⚠️ 欠拟合\nLoss 几乎不降或下降极慢\n→ 增大模型/增加训练时间/降低正则化')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ===== 模式 4: 训练不稳定 =====
    train_loss_unstable = []
    val = 1.0
    for s in steps:
        noise_scale = 0.3 * (1 + 0.5 * np.sin(s / 10))
        val = val - 0.005 + noise_scale * np.random.randn() + 0.02 * np.sin(s / 30)
        val = max(0.1, min(val, 3.0))
        train_loss_unstable.append(val)

    val_loss_unstable = [v + 0.05 + 0.04 * np.random.randn() for v in train_loss_unstable]

    axes[1, 1].plot(steps, train_loss_unstable, 'b-', label='Train Loss', linewidth=1.5)
    axes[1, 1].plot(steps, val_loss_unstable, 'r--', label='Val Loss', linewidth=1.5)
    axes[1, 1].set_title('❌ 不稳定/Loss 爆炸\n剧烈震荡或持续上升\n→ 降低 lr / 检查数据 / 梯度裁剪')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

show_typical_training_curves()
```

## 1.2 如何判断你的曲线属于哪种？

| 观察项 | 正常 | 过拟合 | 欠拟合 | 不稳定 |
|--------|------|--------|--------|--------|
| Train Loss | 持续下降 ✓ | 持续下降到接近 0 | 下降很慢或不降 | 剧烈波动 |
| Val Loss | 持续下降 ✓ | 先降后升 ✗ | 一直很高 | 波动或上升 |
| Train-Val Gap | 小且稳定 | 越来越大 | 都高但 gap 小 | 无规律 |
| 典型原因 | — | 数据少/模型大/epochs 多 | 模型太小/lr 太小/正则化太强 | lr 太大/数据异常 |

---

## 二、TensorBoard 与 Weights & Biases 可视化

## 2.1 TensorBoard —— 本地可视化

TensorBoard 是深度学习领域最经典的监控工具，HF Trainer 原生支持：

```python
# 在 TrainingArguments 中启用
args = TrainingArguments(
    ...,
    report_to="tensorboard",
    logging_dir="./logs",  # TensorBoard 日志目录
)

trainer = Trainer(..., args=args)

# 启动训练后，在终端运行:
# tensorboard --logdir ./logs --port 6006
# 然后浏览器打开 http://localhost:6006
```

Trainer 会自动记录以下内容到 TensorBoard：

| 标签页 | 内容 | 用途 |
|--------|------|------|
| Scalars | Loss / Learning Rate / Epoch | 监控基本指标趋势 |
| Histograms | 权重/梯度分布 | 检查梯度消失/爆炸 |
| Distributions | 同上（更平滑的视图） | 同上 |
| Projector | t-SNE / UMAP / PCA | 高维特征可视化 |

## 2.2 Weights & Biases —— 云端协作监控

对于团队项目，W&B 更实用——它把所有实验记录在云端，方便对比：

```python
# 安装: pip install wandb
# 登录: wandb login

args = TrainingArguments(
    ...,
    report_to="wandb",
    run_name="sentiment-roberta-v1",      # 实验名称
    # wandb 的额外配置可以通过环境变量设置:
    # WANDB_PROJECT=my-project
    # WANDB_ENTITY=my-team
)

trainer = Trainer(..., args=args)
```

W&B 的 Dashboard 会展示：
- **实时的 loss 曲线图**
- **超参数对比表**（不同实验并排显示）
- **系统资源监控**（GPU 利用率、显存、温度）
- **Gradient histograms**

## 2.3 自定义日志记录

除了 Trainer 自动记录的内容，你还可以通过回调函数记录自定义指标：

```python
from transformers import TrainerCallback
import numpy as np

class DetailedMetricsCallback(TrainerCallback):
    """记录更详细的训练指标"""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        if hasattr(args, "_wandb"):
            import wandb
            wandb.log({
                "custom/eval_accuracy_per_class": self._per_class_acc(metrics),
                "custom/eval_loss_gap": metrics.get("eval_loss", 0) -
                                   state.history[-1].get("train_loss", 0),
                "custom/samples_per_second": metrics.get("eval_samples_per_second", 0),
            })

    def _per_class_acc(self, metrics):
        return metrics.get("accuracy_0", 0), metrics.get("accuracy_1", 0)


callbacks = [DetailedMetricsCallback()]
trainer = Trainer(..., callbacks=callbacks)
```

---

## 三、常见问题诊断手册

## 问题 1：Loss 变成 NaN

这是最常见的训练崩溃现象之一。

```python
def diagnose_nan_loss():
    """NaN Loss 的诊断流程"""
    print("=" * 60)
    print("🔍 NaN Loss 诊断清单")
    print("=" * 60)

    checks = [
        ("数据中是否包含 NaN/Inf？",
         "检查: dataset 中是否有空字符串、None 值、除零操作",
         "修复: ds.filter(lambda x: x['text'] and len(x['text'].strip()) > 0)"),

        ("是否使用了 fp16 但没有梯度缩放？",
         "fp16 的值域有限（max=65504），大梯度可能溢出",
         "修复: 确保 Trainer 自动处理了 amp（默认开启）\n"
         "       或手动设置: args.fp16_backend='amp'"),

        ("学习率是否太大？",
         "BERT 微调用 1e-3 是自杀行为",
         "修复: 降低到 2e-5 ~ 5e-5"),

        ("是否忘记做 gradient clipping？",
         "某些层的梯度可能突然爆炸",
         "修复: max_grad_norm=1.0（Trainer 默认已开启）"),

        ("Batch Normalization / Layer Norm 是否有问题？",
         "如果 batch size=1 且用了 BatchNorm，方差为 0 → 除零 → NaN",
         "修复: BERT 使用 LayerNorm，不受此影响；但如果自定义模块用了 BN，注意 bs"),

        ("损失函数是否合理？",
         "CrossEntropyLoss with logits=True 不要再接 Softmax！",
         "检查: model 输出的是 logits 还是 probabilities"),
    ]

    for i, (question, diagnosis, fix) in enumerate(checks, 1):
        print(f"\n[{i}] {question}")
        print(f"    可能原因: {diagnosis}")
        print(f"    解决方案: {fix}")

diagnose_nan_loss()
```

## 问题 2：Loss 不下降

```python
def diagnose_stale_loss():
    """Loss 不下降的诊断"""
    print("\n" + "=" * 60)
    print("🔍 Loss 不下降诊断")
    print("=" * 60)

    causes = [
        ("学习率太小",
         "lr=1e-7 时每步更新量 ≈ 0，几乎等于没训练",
         "尝试: 逐步增大 lr: 1e-7 → 5e-6 → 1e-5 → 2e-5"),
        
        ("优化器选择不当",
         "SGD 对 Transformer 效果差；AdamW 是标准选择",
         "确保: optimizer=AdamW (Trainer 默认)"),

        ("模型头部未正确初始化",
         "新增的分类头权重如果是全零，Softmax 后输出均匀分布",
         "检查: model.classifier.weight 初始化是否正常\n"
         "       HF 默认用 truncated normal，应该没问题"),

        ("标签是否正确？",
         "label 从 0 开始还是从 1 开始？id2label 映射是否正确？",
         "打印一个 batch 的 labels 确认范围和分布"),

        ("数据是否有问题？",
         "标签是否和文本匹配？是否存在标注错误？",
         "抽样人工检查 20 条 data 的 text-label 对应关系"),

        ("Tokenizer 是否和预训练一致？",
         "用 bert-base-chinese 的权重 + roberta 的 tokenizer = 灾难",
         "确保: model 和 tokenizer 来自同一个 checkpoint"),
    ]

    for i, (q, d, f) in enumerate(causes, 1):
        print(f"\n[{i}] {q}")
        print(f"    分析: {d}")
        print(f"    行动: {f}")

diagnose_stale_loss()
```

## 问题 3：显存不足（OOM）

```python
def diagnose_and_fix_oom():
    """OOM 诊断与解决方案"""

    solutions = [
        ("减小 batch_size",
         "最直接有效的方法",
         "per_device_train_batch_size: 16 → 8 → 4 → 2 → 1"),

        ("开启梯度累积",
         "小 batch + 多步累积 = 大 batch 等效效果",
         "gradient_accumulation_steps: 1 → 4 → 8"),

        ("使用混合精度 (fp16)",
         "将参数和激活值从 FP32 降到 FP16，显存减半",
         "fp16: True (需要 GPU 支持)"),

        ("使用梯度检查点",
         "用计算换空间，减少约 60-70% 显存",
         "gradient_checkpointing: True"),

        ("冻结部分层",
         "只训练最后 N 层（适合数据少的场景）",
         "for param in model.encoder.layer[:8].parameters():\n"
         "    param.requires_grad = False"),

        ("使用 DeepSpeed / FSDP",
         "多 GPU 分布式策略",
         "deepspeed='./ds_config.json' 或 accelerate launch"),
    ]

    print("💾 OOM 解决方案 (按推荐顺序):")
    print("=" * 60)
    for i, (method, desc, how) in enumerate(solutions, 1):
        print(f"\n{i}. {method}")
        print(f"   原理: {desc}")
        print(f"   操作: {how}")

    print(f"\n{'='*60}")
    print("显存估算公式:")
    print(f"{'='*60}")
    print("""
    近似显存占用 (FP32):
      = 模型参数 × 4 bytes × 2 (参数+梯度)
      + 中间激活值 × batch_size × seq_len × hidden_size × 4
      + Optimizer states (Adam: 2× 参数量)
      + CUDA 上下文开销 (~10-20%)

    以 BERT-base (110M 参数), batch=32, seq_len=128 为例:
      模型参数+梯度: 110M × 4B × 2 ≈ 880 MB
      Adam 状态:       110M × 4B × 2   ≈ 880 MB
      激活值:          约 1-4 GB (取决于层数)
      总计:            约 3-6 GB (保守估计)
    """)

diagnose_and_fix_oom()
```

## 问题 4：训练速度太慢

```python
def speed_up_checklist():
    """训练加速检查清单"""
    items = [
        ("数据加载是否是瓶颈？",
         "dataloader_num_workers > 0? dataloader_pin_memory = True?",
         "如果 CPU 利用率 100% 而 GPU 利用率低 → 加快数据加载"),

        ("是否使用了 Fast Tokenizer?",
         "tokenizer 使用 Rust 实现 vs Python 实现，速度差 ~24x",
         "确认: AutoTokenizer 默认就是 Fast Tokenizer"),

        ("batch_size 是否足够大？",
         "GPU 利用率 < 50% 说明算力浪费",
         "增大 batch 或使用 gradient_accumulation"),

        ("是否开启了 fp16?",
         "现代 GPU 上 fp16 可以提速 2-3x",
         "fp16: True (需 A100/V100/RTX 30xx+)"),

        ("是否频繁评估？",
         "每个 step 都 eval 会严重拖慢训练",
         "改为 eval_strategy='epoch' 或 eval_steps=500"),

        ("模型是否过大？",
         "bert-large (340M) 比 bert-base (110M) 慢 3x",
         "考虑用更小的基线模型先跑通"),
    ]

    print("⚡ 训练加速检查:")
    for q, check, tip in items:
        print(f"\n☐ {q}")
        print(f"  检查: {check}")
        print(f"  提示: {tip}")

speed_up_checklist()
```

---

## 四、Gradient Accumulation 深度解析

前面我们多次提到梯度累积，这里做一个完整的数学解释：

```python
def explain_gradient_accumulation_math():
    """梯度累积的数学原理"""

    print("=" * 60)
    print("梯度累积 (Gradient Accumulation) 数学原理")
    print("=" * 60)

    print("""
标准 SGD 更新规则:
  θ_{t+1} = θ_t - η × ∇L(θ_t; D)

其中:
  θ: 模型参数
  η: 学习率
  ∇L: 在数据集 D 上计算的损失梯度
  D: 完整的数据集

问题: 当 |D| 很大时，无法一次性放入内存
     或者想模拟更大的有效 batch size

解决方案: Mini-batch SGD
  把 D 分成多个 mini-batch: D = {B₁, B₂, ..., Bₙ}
  每个 mini-batch 单独计算梯度，然后平均

梯度累积的本质:
  不是每个 mini-batch 都立即更新参数，
  而是"累积"多个 mini-batch 的梯度后再统一更新:

  累积 g = 0
  for k in 1..K:
      g += ∇L(θ; B_k)          # 只累加梯度，不更新参数
  θ ← θ - η × (g / K)           # K 步后才更新一次

等效于:
  θ ← θ - η × ∇L(θ; B₁ ∪ B₂ ∪ ... ∪ B_K)

即: 有效 batch size = K × per_device_batch_size
""")

    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    torch.manual_seed(42)
    param = torch.tensor([5.0], requires_grad=True)

    actual_bs = 4
    target_bs = 16
    n_accumulate = target_bs // actual_bs

    print(f"\n演示:")
    print(f"  目标有效 batch size: {target_bs}")
    print(f"  实际每步 batch size:  {actual_bs}")
    print(f"  梯度累积步数:         {n_accumulate}")

    accumulated_grad = 0
    history = []

    for step in range(n_accumulate):
        dummy_input = torch.randn(actual_bs, 1)
        dummy_target = torch.ones(actual_bs, 1)

        loss = ((param * dummy_input - dummy_target) ** 2).mean()
        loss.backward()

        accumulated_grad += param.grad.item()
        history.append(param.item())

        print(f"\n  Step {step+1}:")
        print(f"    Loss = {loss.item():.4f}, Grad = {param.grad.item():.4f}")
        print(f"    累积梯度 = {accumulated_grad:.4f}")
        print(f"    参数值   = {param.item():.4f} (尚未更新)")

        param.grad.zero_()

    effective_lr = 0.01
    update = effective_lr * (accumulated_grad / n_accumulate)
    param.data -= update

    print(f"\n  {'='*40}")
    print(f"  执行更新: θ -= {effective_lr} × ({accumulated_grad:.4f}/{n_accumulate})")
    print(f"  更新量: {update:.4f}")
    print(f"  最终参数: {param.item():.4f}")

explain_gradient_accumulation_math()
```

---

## 五、Early Stopping —— 自动防止过拟合

```python
from transformers import EarlyStoppingCallback, TrainerCallback
import copy

class BestModelSaverCallback(TrainerCallback):
    """保存最佳模型的回调（比 load_best_model_at_end 更灵活）"""

    def __init__(self, save_path="./best_model"):
        super().__init__()
        self.save_path = save_path
        self.best_metric = None
        self.best_state_dict = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return

        current_metric = metrics["eval_loss"]

        if self.best_metric is None or current_metric < self.best_metric:
            self.best_metric = current_metric
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            print(f"  🏆 新最佳! eval_loss={current_metric:.4f}")


# 使用示例
callbacks = [
    EarlyStoppingCallback(
        early_stopping_patience=2,     # 连续 2 个 epoch 无改善就停
        early_stopping_threshold=0.001,  # 改善幅度小于此值视为无改善
    ),
]

args = TrainingArguments(
    ...,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,  # loss 越小越好
    save_strategy="epoch",
)
```

> **注意 `early_stopping_patience` 的设置**：
> - patience=1：只要 1 个 epoch 没提升就停止（激进，可能欠拟合）
> - patience=2~3：大多数任务的推荐值
> - patience=5+：非常宽容，除非你有充足的理由

---

## 小结

这一节我们建立了完整的训练监控与调试能力体系：

1. **四种典型曲线**：正常收敛（train/val 同步下降）、过拟合（train↓ val↑）、欠拟合（都不降）、不稳定（震荡/爆炸）。每种都有明确的原因和对策
2. **可视化工具**：TensorBoard（本地免费，Scalars/Histograms/Distributions）、W&B（云端协作，实验对比）。Trainer 通过 `report_to` 参数一键集成
3. **四大问题诊断手册**：
   - NaN Loss → 数据异常/fp16溢出/lr太大/重复Softmax
   - Loss 不降 → lr太小/优化器不对/标签错位/tokenizer不匹配
   - OOM → 减小bs/梯度累积/fp16/梯度检查点/冻结层/DeepSpeed
   - 速度慢 → num_workers/fast_tokenizer/fp16/增大bs/减少eval频率
4. **梯度累积数学原理**：K步累积梯度后再更新 = 等效于 K×bs 的大batch训练。`warmup_ratio` 会自动考虑累积步数
5. **Early Stopping**：`EarlyStoppingCallback(patience=2)` 自动防止过拟合，配合 `load_best_model_at_end=True` 自动恢复最佳模型

下一节我们将整合所有知识，提供一个**生产级的完整微调脚本模板**，涵盖分类、NER、QA 三种任务类型。
