# 高级训练技巧：从"能跑通"到"跑得好"

## 这一节讲什么？

前面的内容已经让你能够成功微调一个模型了——loss 在下降，accuracy 在提升，模型也保存好了。但如果你追求的是**生产级的效果和效率**，还需要掌握一些进阶技巧。

这一节，我们将覆盖以下高级主题：
- **分层学习率（Layer-wise LR）**：不同层用不同的学习率
- **冻结层策略（Freezing Layers）**：只训练部分层来节省资源
- **对比学习与监督对比损失**：小数据场景下的利器
- **多 GPU 训练策略**：DP / DDP / FSDP / DeepSpeed
- **混合精度与量化感知训练**
- **超参数搜索基础**

---

## 一、分层学习率 —— 让底层学得慢、顶层学得快

## 1.1 直觉

回顾一下我们在第3章学到的知识：BERT 的不同层编码了不同抽象级别的信息。底层的 Layer 0-3 学到的是通用特征（字符模式、短语结构），这些在预训练阶段已经学得很好了；而顶层的 Layer 8-12 是任务相关的组合特征，需要根据你的具体任务做较大调整。

那么一个自然的想法是：**让底层的小幅调整、顶层的大幅调整**——这就是分层学习率（Layer-wise Learning Rate Decay）。

## 1.2 实现方式

```python
def setup_layerwise_lr(model, base_lr=2e-5, decay_factor=0.9):
    """
    设置分层学习率：越深的层学习率越大
    
    decay_factor: 每层的学习率乘以这个因子
              = 0.9 表示每层比上一层多 10%
    """
    
    # 获取所有参数组
    param_groups = []
    
    # 方法 1: 按 Encoder 层分组（推荐）
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        n_layers = len(model.encoder.layer)
        
        # Embedding 层：最小学习率（几乎不更新）
        embedding_params = [
            {"params": model.embeddings.parameters(), "lr": base_lr * decay_factor ** (n_layers + 1)}
        ]
        param_groups.extend(embedding_params)
        
        # 每个 Transformer 层
        for i, layer in enumerate(model.encoder.layer):
            layer_lr = base_lr * decay_factor ** (n_layers - i)
            param_groups.append({
                "params": layer.parameters(),
                "lr": layer_lr,
                "name": f"layer_{i}_lr_{layer_lr:.2e}"
            })
        
        # Pooler/Head 层：最大学习率
        if hasattr(model, "pooler"):
            param_groups.append({
                "params": model.pooler.parameters(),
                "lr": base_lr,
                "name": f"pooler_lr_{base_lr:.2e}"
            })
        if hasattr(model, "classifier"):
            param_groups.append({
                "params": model.classifier.parameters(),
                "lr": base_lr,
                "name": f"classifier_lr_{base_lr:.2e}"
            })
    
    return param_groups


# 使用示例
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "hfl/chinese-roberta-wwm-ext", num_labels=3
)

param_groups = setup_layerwise_lr(model, base_lr=2e-5, decay_factor=0.95)

optimizer = torch.optim.AdamW(param_groups)

print("各层学习率:")
for pg in param_groups:
    if "name" in pg:
        n_params = sum(p.numel() for p in pg["params"])
        print(f"  {pg['name']}: {n_params:,} 参数")
```

输出示例：
```
各层学习率:
  layer_11_lr_2.00e-05: 4,718,592 参数
  layer_10_lr_1.90e-05: 4,718,592 参数
  layer_09_lr_1.81e-05: 4,718,592 参数
  ...
  layer_0_lr_1.15e-05: 4,718,592 参数
  embeddings_lr_1.08e-05: 2,466,816 参数
  pooler_lr_2.00e-05: 590,848 参数
  classifier_lr_2.00e-05: 2,303 参数
```

## 1.3 HF Trainer 中使用分层学习率

Trainer 本身不直接支持分层 lr，但可以通过回调或自定义 optimizer 实现：

```python
from transformers import TrainerCallback

class LayerWiseLRCallback(TrainerCallback):
    """在 on_optimizer_step 前修改各层学习率"""
    
    def __init__(self, base_lr=2e-5, decay=0.95, model=None):
        super().__init__()
        self.base_lr = base_lr
        self.decay = decay
        self.model = model
        
    def on_optimizer_step(self, args, state, control, **kwargs):
        if not hasattr(self, "_layer_ids"):
            # 首次调用时建立层索引映射
            self._layer_ids = {}
            current_layer = -1
            
            for name, param in self.model.named_parameters():
                if "encoder.layer." in name:
                    layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_num != current_layer:
                        current_layer = layer_num
                
            # 设置各层学习率
            n_layers = len(self.model.encoder.layer)
            for pg in self.optimizer.param_groups:
                if hasattr(pg, "layer_id"):
                    pg["lr"] = self.base_lr * (self.decay ** (n_layers - pg["layer_id"]))
```

> **更简单的替代方案**：HF 的 `get_linear_schedule_with_warmup` 已经提供了足够好的默认行为。对于大多数任务，统一的线性衰减 + warmup 就够了。分层 LR 主要在**预训练从头开始**或**极大领域差异**的场景下才有显著收益。

---

## 二、冻结层策略 —— 小数据的救命稻草

## 2.1 为什么冻结？

当你的标注数据很少（比如只有几百条）时，更新全部 110M 个参数几乎必然导致过拟合。一种有效的策略是：**冻结预训练模型的底部 N 层，只训练顶部几层 + 分类头**。

```python
def freeze_bottom_layers(model, n_freeze=8):
    """
    冻结底部 N 层的 Transformer encoder
    
    Args:
        model: 预训练模型
        n_freeze: 冻结多少层（从底层算起）
                    0 = 不冻结任何层
                    8 = 冻结前 8 层（保留后 4 层可训练）
                    12 = 冻结全部 encoder 层（只训练 head）
    """
    frozen_count = 0
    trainable_count = 0
    total_params = 0
    
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        all_layers = model.encoder.layer
        n_total = len(all_layers)
        
        for i, layer in enumerate(all_layers):
            is_frozen = i < n_freeze
            
            for param in layer.parameters():
                param.requires_grad = not is_frozen
                
                if is_frozen:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
                total_params += param.numel()
    
    # 始终保持 head 可训练
    for name, param in model.named_parameters():
        if "encoder" not in name and "embeddings" not in name:
            param.requires_grad = True
            trainable_count += param.numel()
            total_params += param.numel()
    
    print(f"冻结策略: 底部 {n_freeze}/{n_total} 层被冻结")
    print(f"  可训练参数: {trainable_count:,} ({trainable_count/total_params*100:.1f}%)")
    print(f"  冻结参数:   {frozen_count:,} ({frozen_count/total_params*100:.1f}%)")
    
    return model


# 使用示例
model = AutoModelForSequenceClassification.from_pretrained(
    "hfl/chinese-roberta-wwm-ext", num_labels=3
)

# 场景 A: 数据极少 (< 500条) → 只训练最后 2 层 + head
model = freeze_bottom_layers(model, n_freeze=10)

# 场景 B: 数据较少 (500-2000条) → 训练最后 6 层 + head  
model = freeze_bottom_layers(model, n_freeze=6)

# 场景 C: 数据充足 (> 5000条) → 全部可训练
model = freeze_bottom_layers(model, n_freeze=0)
```

## 2.2 渐进式解冻（Gradual Unfreezing）

更精细的策略是**先冻结大部分层训练几个 epoch，然后逐步解冻**：

```python
def gradual_unfreeze_training(model, train_dataset, eval_dataset, tokenizer,
                                unfreeze_schedule=[(0, 10), (1, 8), (2, 6), (3, 4)]):
    """
    渐进式解冻训练
    
    unfreeze_schedule: list of (epoch, n_frozen_layers)
        例如 [(0, 10), (1, 8), (2, 6)]
        表示 epoch 0 冻结 10 层, epoch 1 解冻为 8 层, ...
    """
    
    from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
    
    for phase_idx, (start_epoch, n_frozen) in enumerate(unfreeze_schedule):
        print(f"\n{'='*50}")
        print(f"Phase {phase_idx+1}: Epoch {start_epoch}+, 冻结 {n_frozen} 层")
        print(f"{'='*50}")
        
        model = freeze_bottom_layers(model, n_freeze=n_frozen)
        
        args = TrainingArguments(
            output_dir=f"./output_phase{phase_idx}",
            num_train_epochs=start_epoch + 3,  # 每个 phase 训练 3 个 epoch
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=True,
            report_to="none",
            logging_steps=50,
        )
        
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        
        trainer.train()
```

---

## 三、多 GPU 训练策略

## 3.1 四种并行方式的对比

| 策略 | 缩写 | 数据并行？ | 模型并行？ | 适用场景 |
|------|------|-----------|-----------|---------|
| Data Parallel | DP | ✅ | ❌ | 单机多卡，简单场景 |
| Distributed Data Parallel | DDP | ✅ | ❌ | 多机多卡，推荐方案 |
| Fully Sharded Data Parallel | FSDP | ✅ | ✅ | 大模型单机多卡 |
| DeepSpeed | DS | ✅ | ✅✅ | 超大模型（7B+） |

## 3.2 使用 accelerate 启动分布式训练

HF 的 `accelerate` 库让分布式训练变得非常简单：

```bash
# 安装
pip install accelerate

# 启动单卡训练
python my_train.py

# 启动多卡 DDP 训练
accelerate launch --num_processes 2 --multi_gpu my_train.py

# 启动 FSDP 训练 (shard across GPUs)
accelerate launch --num_processes 4 --fsdp "full_shard auto_wrap" my_train.py

# DeepSpeed ZeRO-2
accelerate launch --deepspeed ds_config_z2.json my_train.py
```

`accelerate_config.yaml` 配置文件示例：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 4
mixed_precision: fp16

# FSDP 配置 (用于 7B-20B 模型)
fsdp_config:
  sharding_strategy: FULL_SHARD
  offload_params: false
  auto_wrap_policy: TRANSFORMER_BASED

# DeepSpeed 配置 (用于 20B+ 模型)
deepspeed_config:
  zero_optimization:
    stage: 2
    offload_optimizer: true
    offload_param: true
```

---

## 四、混合精度训练详解

## 4.1 FP16 的完整工作流

```python
def demonstrate_mixed_precision():
    """展示混合精度训练的工作流程"""

    print("=" * 60)
    print("混合精度 (Mixed Precision) 工作流程")
    print("=" * 60)

    steps = [
        ("前向传播 (FP16)", 
         "模型权重: FP32 → 自动转换为 FP16\n"
         "中间激活值: FP16 计算\n"
         "速度: ~2x 加速"),
        
        ("Loss 计算 (FP32)",
         "Loss 在 FP32 下计算（保证精度）\n"
         "防止 loss 下溢出"),
        
        ("反向传播 (FP16)",
         "梯度初始为 FP16\n"
         "可能溢出 → 用 GradScaler 缩放"),
        
        ("梯度缩放 (GradScaler)",
         "动态调整缩放因子 S\n"
         "如果检测到 inf/nan: S /= 2\n"
         "如果连续正常: S *= 2\n"
         "范围通常: [2^0, 2^16]"),
        
        ("权重更新 (FP32)",
         "梯度 × S 后转为 FP32\n"
         "optimizer 更新 FP32 权重\n"
         "再转回 FP16 用于下次前向"),
    ]

    for step_name, details in steps:
        print(f"\n📍 {step_name}")
        print(f"   {details}")

    print(f"\n{'='*60}")
    print("为什么能加速?")
    print(f"{'='*60}")
    print("""
    1. Tensor Core 加速 (A100/V100): FP16 矩阵运算比 FP32 快 8-16x
    2. 显存减半: 参数 FP16 存储 (但优化器状态仍为 FP32)
    3. 内存带宽: 传输 2 bytes/sample vs 4 bytes/sample
    4. 通信量减少: AllReduce 通信量减半
    """)

demonstrate_mixed_precision()
```

## 4.2 BF16 vs FP16 的选择指南

| 特性 | FP16 | BF16 |
|------|------|------|
| 最大值 | 65504 | ~3.4×10³⁸ (同 FP32) |
| 最小正值 | 6.1×10⁻⁵ | ~5.9×10⁻²⁰ |
| 精度 (尾数) | 10 bits | 7 bits |
| 上溢风险 | ⚠️ 高（>65504 变 inf） | ✅ 极低 |
| 下溢风险 | ⚠️ 有（极小值变 0） | ⚠️ 有（但比 FP16 好） |
| 推荐硬件 | V100/A100 | A100 (新架构) / H100 |
| 推荐场景 | 推理 / 有 clip 的训练 | 训练（尤其是 LLM） |

---

## 五、常见高级问题速查

## Q1: 微调后效果不如直接用 Pipeline 怎么办？

```python
# 可能原因及解决方案:

# 1. 数据质量问题
# → 抽查 50 条样本人工验证 label 是否正确
# → 检查类别分布是否严重不平衡

# 2. 学习率不合适
# → 尝试 [1e-5, 2e-5, 5e-5] 三个值分别跑一次
# → 观察哪个 loss 下降最稳定

# 3. 模型选择不当
# → BERT-base 对短文本可能过强（overkill）
# → 尝试更小的模型或领域相关模型

# 4. 预处理错误
# → tokenize 时是否用了错误的 max_length？
# → 是否忘记对齐 labels 和 input_ids？
# → NER 任务是否正确设置了 -100 标签？

# 5. 评估指标误导
# → Accuracy 在不平衡数据上没有意义！
# → 改用 F1-score / macro-F1 / weighted-F1
```

## Q2: 如何判断是否需要继续训练还是换方向？

```python
def should_continue_or_pivot(eval_history):
    """基于历史指标决定下一步行动"""
    if not eval_history:
        return "需要更多数据"

    recent_losses = [m.get("eval_loss") for m in eval_history[-3:] if "eval_loss" in m]
    recent_accs = [m.get("eval_accuracy", m.get("eval_f1", 0)) 
                  for m in eval_history[-3:]]

    if recent_accs and max(recent_accs) > 0.85:
        return "效果已经很好，可以部署"

    if recent_losses and (recent_losses[-1] > recent_losses[0]):
        if recent_losses[-1] > recent_losses[-2]:
            return "⚠️ 过拟合趋势! 减少 epochs 或增加正则化"

    if len(recent_losses) >= 3 and all(recent_losses[i] <= recent_losses[i-1] + 0.001 
                   for i in range(1, len(recent_losses))):
        return "⚠️ Loss 不再下降，考虑降低 lr 或停止训练"

    return "继续观察，当前趋势正常"
```

---

## 小结

这一节覆盖了将微调从"能用"提升到"好用"的关键技巧：

1. **分层学习率**：底层用较小 lr（保护预训练特征），顶层用较大 lr（快速适配任务）。通过 `decay_factor` 控制逐层放大比例。HF Trainer 不原生支持，可通过自定义 optimizer 或 callback 实现
2. **冻结层策略**：数据少时冻结底部 N 层，只训练顶部几层 + head。可训练参数可降到 1-10%。支持渐进式解冻（gradual unfreezing）实现更平滑的迁移
3. **多 GPU 并行**：DP（简单但低效）→ DDP（推荐标准方案）→ FSDP（大模型单机多卡）→ DeepSpeed ZeRO（超大模型跨节点）
4. **混合精度**：FP16 推理首选（速度快），BF16 训练首选（数值稳定）。核心组件 `GradScaler` 动态管理缩放因子防溢出
5. **决策框架**：效果不好时按顺序排查：数据质量 → lr → 模型选择 → 预处理 → 评估指标；通过趋势分析决定继续/转向/停止

下一节是第6章最后一节——微调后的效果评估与生产部署。
