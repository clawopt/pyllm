# PEFT 生产最佳实践：从训练到部署的全链路指南

## 这一节讲什么？

前面四节我们已经系统学习了 PEFT 的原理（LoRA/Adapter/Prefix/(IA)³）、不同任务的配置策略、以及 Adapter 的合并与切换。这一节我们将把这些知识串联起来，形成一份**从训练到部署到监控的完整生产指南**。

具体来说，你将学到：
1. **PEFT 训练全流程最佳实践**——数据准备、超参选择、训练技巧
2. **效果评估与调优**——如何判断 LoRA 是否真的有效、何时该换 Full FT
3. **性能优化**——显存、速度、吞吐量的全方位优化
4. **故障排查手册**——常见问题的诊断与解决
5. **持续迭代策略**——模型版本管理、A/B 测试、回滚机制

---

## 一、PEFT 训练全流程最佳实践

## 1.1 训练前 Checklist

在开始任何一次 PEFT 微调之前，先回答以下问题：

```python
def pre_training_checklist():
    """PEFT 训练前检查清单"""

    checklist = [
        {
            "类别": "数据质量",
            "项目": "数据量是否足够? (LoRA: >500, Full FT: >2000)",
            "通过标准": "✅ / ⚠️ / ❌",
            "说明": "太少会导致过拟合或欠拟合",
        },
        {
            "类别": "数据质量",
            "项目": "数据格式是否正确? (Alpaca/ShareGPT/OpenAI)",
            "通过标准": "✅ / ❌",
            "说明": "格式错误会导致训练完全失败",
        },
        {
            "类别": "基座模型",
            "项目": "基座模型能力是否覆盖任务需求?",
            "通过标准": "✅ / ❌",
            "说明": "7B 模型做不了复杂推理; 代码任务用 Code LLM",
        },
        {
            "类别": "硬件资源",
            "项目": "显存是否满足需求? (参考估算公式)",
            "通过标准": "✅ / ⚠️ (需梯度累积) / ❌",
            "说明": "7B QLoRA 需 ~10GB; 7B LoRA FP16 需 ~16GB",
        },
        {
            "类别": "PEFT 配置",
            "项目": "target_modules 名称是否匹配模型结构?",
            "通过标准": "✅ / ❌",
            "说明": "BERT 用 query/value, LLaMA 用 q_proj/v_proj",
        },
        {
            "类别": "评估方案",
            "项目": "是否有独立的测试集和评估指标?",
            "通过标准": "✅ / ❌",
            "说明": "没有 eval 就不知道训练是否有效",
        },
        {
            "类别": "Baseline",
            "项目": "是否建立了基线性能 (zero-shot 或 few-shot)?",
            "通过标准": "✅ / ❌",
            "说明": "需要知道微调前后的对比才能判断效果",
        },
    ]

    print("=" * 80)
    print("PEFT 训练前检查清单")
    print("=" * 80)

    for item in checklist:
        status = item["通过标准"]
        icon = "✅" if status.startswith("✅") else ("⚠️" if "⚠️" in status else "❌")
        print(f"\n[{icon}] [{item['类别']}] {item['项目']}")
        print(f"    标准: {status} | 说明: {item['说明']}")

pre_training_checklist()
```

## 1.2 数据准备的黄金法则

PEFT 微调的数据准备有几个容易被忽视的关键点：

```python
def data_preparation_rules():
    """PEFT 数据准备的黄金法则"""

    rules = [
        {
            "规则": "数据多样性 > 数据数量",
            "解释": "1000 条高质量、覆盖多种场景的数据,"
                   "优于 10000 条高度重复的数据",
            "反例": "1000 条都是 '翻译: Hello → 你好' 的数据",
            "正例": "包含问候/问答/指令/推理/创作等多种类型",
        },
        {
            "规则": "去除 PII (个人隐私信息)",
            "解释": "训练数据中的姓名/电话/地址等可能被模型记忆并在推理时泄露",
            "工具": "presidio (Microsoft 的 PII 检测库)",
        },
        {
            "规则": "控制输入长度分布",
            "解释": "如果 90% 的数据都 < 256 token 但有 10% 超过 2048,"
                   "长样本会拖慢训练且可能导致 OOM",
            "建议": "截断到合理长度 (如 512/1024/2048),"
                   "或对超长样本做分段处理",
        },
        {
            "规则": "平衡正负样本比例",
            "解释": "对于分类任务, 99:1 的不平衡会让模型学会'总是预测多数类'",
            "建议": "过采样少数类 / 次采样多数类 / 使用 class_weight / focal loss",
        },
        {
            "规则": "验证 tokenizer 的一致性",
            "解释": "训练时用的 tokenizer 和推理时必须完全一致,"
                   "包括 special_tokens 和 vocab",
            "检查": "保存 adapter 时必须同时保存 tokenizer!",
        },
    ]

    print("=" * 75)
    print("PEFT 数据准备黄金法则")
    print("=" * 75)

    for i, r in enumerate(rules, 1):
        print(f"\n📏 法则 {i}: {r['规则']}")
        print(f"   解释: {r['解释']}")
        if '反例' in r:
            print(f"   ❌ 反例: {r['反例']}")
        if '正例' in r:
            print(f"   ✅ 正例: {r['正例']}")
        if '建议' in r:
            print(f"   💡 建议: {r['建议']}")
        if '工具' in r:
            print(f"   🔧 工具: {r['工具']}")
        if '检查' in r:
            print(f"   🔍 检查: {r['检查']}")

data_preparation_rules()
```

## 1.3 超参数选择的系统性方法

很多新手凭感觉设置超参数，这往往导致反复试错。下面是一个系统化的方法：

```python
def systematic_hyperparameter_selection():
    """PEFT 超参数选择的系统性方法"""

    print("=" * 70)
    print("PEFT 超参数选择决策树")
    print("=" * 70)

    # Step 1: 根据 GPU 显存决定量化策略
    def step1_choose_quantization(gpu_memory_gb):
        """根据显存选择量化"""
        if gpu_memory_gb >= 80:
            return "FP16/BF16 (无需量化)"
        elif gpu_memory_gb >= 24:
            return "FP16/BF16 + gradient_checkpointing"
        elif gpu_memory_gb >= 16:
            return "QLoRA (4-bit)"
        else:
            return "QLoRA (4-bit) + deepspeed stage 2"

    # Step 2: 根据数据量选择 rank
    def step2_choose_rank(data_size):
        if data_size < 500:
            return {"rank": 4, "alpha": 8, "reason": "极小数据, 防止过拟合"}
        elif data_size < 2000:
            return {"rank": 8, "alpha": 16, "reason": "小数据集"}
        elif data_size < 10000:
            return {"rank": 16, "alpha": 32, "reason": "中等数据集"}
        elif data_size < 50000:
            return {"rank": 32, "alpha": 64, "reason": "较大数据集"}
        else:
            return {"rank": 64, "alpha": 128, "reason": "大数据集, 充分表达"}

    # Step 3: 根据任务类型选择 learning rate
    def step3_choose_lr(task_type):
        lr_map = {
            "classification": 2e-4,
            "ner": 3e-4,
            "qa": 3e-4,
            "generation": 2e-4,
            "summarization": 2e-4,
        }
        return lr_map.get(task_type, 2e-4)

    # 演示
    scenarios = [
        ("RTX 4090 (24GB)", 24, 3000, "generation"),
        ("RTX 3090 (24GB)", 24, 800, "classification"),
        ("A100 (40GB)", 40, 15000, "ner"),
        ("A100 (80GB)", 80, 50000, "generation"),
    ]

    print("\n场景化推荐:")
    for name, mem, data, task in scenarios:
        quant = step1_choose_quantization(mem)
        rank_cfg = step2_choose_rank(data)
        lr = step3_choose_lr(task)

        print(f"\n🖥️  {name}")
        print(f"   量化策略:   {quant}")
        print(f"   Rank 配置:  r={rank_cfg['rank']}, α={rank_cfg['alpha']} ({rank_cfg['reason']})")
        print(f"   学习率:     {lr}")
        print(f"   任务类型:   {task}")

systematic_hyperparameter_selection()
```

---

## 二、效果评估：PEFT 真的有效吗？

## 2.1 建立正确的评估框架

很多人训练完 LoRA 后只看 training loss 就下结论，这是远远不够的。一个完整的评估应该包含以下层次：

```python
class PeftEvaluator:
    """
    PEFT 效果评估器
    多维度、多层次的评估体系
    """

    def __init__(self, base_model, peft_model, tokenizer, test_data):
        self.base_model = base_model
        self.peft_model = peft_model
        self.tokenizer = tokenizer
        self.test_data = test_data

    def evaluate_zero_shot_baseline(self):
        """Step 1: 基座的 zero-shot/few-shot 性能"""
        print("\n[Step 1] 基座模型 Baseline")
        # 在测试集上运行基座模型 (无任何微调)
        # 这是最重要的基准!
        pass

    def evaluate_peft_model(self):
        """Step 2: PEFT 微调后模型的性能"""
        print("\n[Step 2] PEFT 模型性能")
        # 在同一测试集上运行微调后的模型
        pass

    def compute_improvement(self):
        """Step 3: 计算提升幅度"""
        print("\n[Step 3] 提升分析")
        # 对比 baseline 和 PEFT 结果
        # 关键指标: 绝对提升 / 相对提升 / 统计显著性
        pass

    def evaluate_full_ft_comparison(self, full_ft_model=None):
        """Step 4: 与 Full FT 的差距 (可选)"""
        print("\n[Step 4] vs Full Fine-Tuning")
        # 如果条件允许, 训练一个 Full FT 版本做对照
        # 目标: PEFT 应达到 Full FT 的 90%+ 效果
        pass

    def human_evaluation(self, n_samples=50):
        """Step 5: 人工抽样评估 (不可省略!)"""
        print("\n[Step 5] 人工抽样评估")
        # 自动指标不能反映一切!
        # 至少抽查 50 条生成结果, 人工判断质量
        pass


def evaluation_framework_demo():
    """演示评估框架的使用"""

    print("=" * 70)
    print("PEFT 效果评估完整流程")
    print("=" * 70)

    framework = """
    ┌─────────────────────────────────────────────┐
    │           PEFT 评估金字塔                    │
    ├─────────────────────────────────────────────┤
    │                                             │
    │         🧪 自动指标 (定量)                  │
    │    ┌───────────┬───────────┬───────────┐    │
    │    │ Accuracy  │ F1-Score  │ BLEU/ROUGE│    │
    │    └───────────┴───────────┴───────────┘    │
    │              ↓ 必须通过                      │
    │         📊 对比分析                          │
    │    ┌───────────┬───────────┐               │
    │    │ vs Base   │ vs Full FT│               │
    │    └───────────┴───────────┘               │
    │              ↓ 推荐通过                      │
    │         👁️ 人工审查 (定性)                  │
    │    ┌───────────────────────────┐          │
    │    │ 抽样 50+ 条生成结果       │          │
    │    │ 检查: 流畅性/准确性/安全性│          │
    │    └───────────────────────────┘          │
    │                                             │
    └─────────────────────────────────────────────┘
    """
    print(framework)

evaluation_framework_demo()
```

## 2.2 判断 PEFT 是否足够好的标准

```python
def peft_quality_thresholds():
    """PEFT 效果判断标准"""

    print("=" * 70)
    print("PEFT 效果判断标准")
    print("=" * 70)

    standards = [
        {
            "等级": "🟢 优秀 (可直接部署)",
            "vs Baseline": "+15% 以上提升",
            "vs Full FT": "达到 95%+ 效果",
            "人工评估": "无明显问题",
            "行动": "部署上线",
        },
        {
            "等级": "🟡 良好 (可部署但需监控)",
            "vs Baseline": "+5%~15% 提升",
            "vs Full FT": "达到 85~95% 效果",
            "人工评估": "偶有小问题",
            "行动": "部署 + 加强监控 + 收集反馈后迭代",
        },
        {
            "等级": "🟠 及格 (需改进)",
            "vs Baseline": "0~5% 提升或有波动",
            "vs Full FT": "达到 70~85% 效果",
            "人工评估": "有明显缺陷",
            "行动": "调整 rank/lr/data 后重新训练",
        },
        {
            "等级": "🔴 不合格 (需重大调整)",
            "vs Baseline": "负向提升 (比基座还差!)",
            "vs Full FT": "< 70% 效果",
            "人工评估": "严重问题 (幻觉/偏激/不安全)",
            "行动": "检查数据/配置/基座模型选择",
        },
    ]

    for s in standards:
        print(f"\n{s['等级']}")
        for k, v in s.items():
            if k != "等级":
                print(f"   {k}: {v}")

peft_quality_thresholds()
```

## 2.3 常见的"假阳性"陷阱

有些情况下 PEFT 看起来效果好但实际上有问题：

```python
def false_positive_traps():
    """PEFT 评估中的假阳性陷阱"""

    traps = [
        {
            "陷阱": "Training Loss 下降但 Eval Loss 上升",
            "真相": "严重的过拟合! LoRA 参数虽然少但仍可能过拟合小数据集",
            "检测": "必须同时监控 train_loss 和 eval_loss",
            "修复": "增大 lora_dropout; 减小 rank; 增加数据量; 早停",
        },
        {
            "陷阱": "Eval Accuracy 很高但实际使用很差",
            "真相": "测试集和真实分布不一致 (distribution shift); "
                   "或测试集太简单/有泄露",
            "检测": "用真实生产数据做 held-out 测试",
            "修复": "收集更多真实场景数据; 做对抗性测试",
        },
        {
            "陷阱": "Few 个样本上效果惊艳",
            "真相": "可能是记忆了这几个样本而非真正学会了泛化",
            "检测": "在完全未见过的数据上测试",
            "修复": "扩大验证集; 做交叉验证",
        },
        {
            "陷阱": "合并后精度下降",
            "真相": "合并过程中的数值误差 (FP32→BF16 截断)",
            "检测": "对比合并前后在同一输入上的输出差异",
            "修复": "保持 FP32 合并后再转换; 使用 safe_merge=True",
        },
    ]

    print("=" * 75)
    print("PEFT 评估假阳性陷阱")
    print("=" * 75)

    for t in traps:
        print(f"\n🚨 陷阱: {t['陷阱']}")
        print(f"   真相: {t['真相']}")
        print(f"   检测: {t['检测']}")
        print(f"   修复: {t['修复']}")

false_positive_traps()
```

---

## 三、性能优化全攻略

## 3.1 显存优化

显存是 PEFT 训练中最常见的瓶颈。以下是按成本从低到高的优化手段：

```python
def memory_optimization_pyramid():
    """显存优化金字塔 (从低成本到高成本)"""

    optimizations = [
        {
            "层级": 1, "名称": "Gradient Checkpointing",
            "节省": "~30-50% 激活值显存",
            "代价": "训练速度降低 ~20%",
            "配置": "TrainingArguments(gradient_checkpointing=True)",
            "适用": "几乎所有场景 (默认开启!)",
        },
        {
            "层级": 2, "名称": "混合精度 (BF16/FP16)",
            "节省": "~50% 模型权重显存",
            "代价": "可能的数值不稳定 (FP16)",
            "配置": "TrainingArguments(bf16=True) 或 fp16=True",
            "适用": "支持 BF16 的 GPU (Ampere+); FP16 给老显卡",
        },
        {
            "层级": 3, "名称": "梯度累积",
            "节省": "允许更小的 batch size",
            "代价": "训练步数增加",
            "配置": "gradient_accumulation_steps=4/8/16",
            "适用": "batch size 受限时",
        },
        {
            "层级": 4, "名称": "QLoRA (4-bit 量化)",
            "节省": "~75% 模型权重显存",
            "代价": "~1-3% 精度损失",
            "配置": "BitsAndBytesConfig(load_in_4bit=True)",
            "适用": "消费级 GPU (< 24GB) 上跑大模型",
        },
        {
            "层级": 5, "名称": "DeepSpeed ZeRO",
            "节省": "跨卡分布式显存",
            "代价": "需要多 GPU + 通信开销",
            "配置": "accelerate launch --config ds_config.json",
            "适用": "多卡环境; 单卡无法容纳时",
        },
        {
            "层级": 6, "名称": "FSDP (Fully Sharded Data Parallel)",
            "节省": "最彻底的分布式显存切分",
            "代价": "配置复杂度高",
            "配置": "accelerate config 中设置 fsdp=True",
            "适用": "超大模型 (30B+) 的多卡训练",
        },
    ]

    print("=" * 85)
    print("显存优化金字塔 (按推荐优先级排序)")
    print("=" * 85)

    print(f"\n{'层级':<4} {'方法':<28} {'节省':<18} {'代价':<20} {'适用'}")
    print("-" * 95)
    for opt in optimizations:
        print(f"{opt['层级']:<4} {opt['名称']:<28} {opt['节省']:<18} {opt['代价']:<20} {opt['适用']}")

memory_optimization_pyramid()
```

## 3.2 训练速度优化

```python
def speed_optimization_guide():
    """训练速度优化指南"""

    techniques = [
        {
            "技术": "FlashAttention-2",
            "提速": "2-4x Attention 计算",
            "实现": "model = FlashAttention2Model.from_pretrained(...)",
            "要求": "Ampere+ GPU; PyTorch >= 2.0",
            "备注": "目前最重要的单点加速技术",
        },
        {
            "技术": "Torch Compile (torch.compile)",
            "提速": "10-30% 端到端",
            "实现": "model = torch.compile(model)",
            "要求": "PyTorch >= 2.0; 可能遇到 graph break",
            "备注": "新功能, 可能有兼容性问题",
        },
        {
            "技术": "DataLoader num_workers",
            "提速": "减少 CPU 瓶颈",
            "实现": "dataloader = DataLoader(..., num_proc=4)",
            "要求": "足够的 CPU 核心",
            "备注": "num_workers=CPU核心数//2 是好起点",
        },
        {
            "技术": "TF32 (NVIDIA Ampere)",
            "提速": "矩阵运算加速",
            "实现": "自动启用 (Ampere GPU 默认)",
            "要求": "A100/RTX 3090/4090",
            "备注": "几乎零成本的免费加速",
        },
        {
            "技术": "编译后的 CUDA kernels",
            "提速": "算子融合消除 kernel launch 开销",
            "实现": "xformers / Triton",
            "要求": "额外安装",
            "备注": "vLLM/TGI 内部已集成",
        },
    ]

    print("=" * 80)
    print("训练速度优化技术一览")
    print("=" * 80)

    for t in techniques:
        print(f"\n⚡ {t['技术']}")
        print(f"   提速: {t['提速']}")
        print(f"   实现: {t['实现']}")
        print(f"   要求: {t['要求']}")
        print(f"   备注: {t['备注']}")

speed_optimization_guide()
```

## 3.3 推理性能优化

```python
def inference_optimization():
    """推理阶段的性能优化"""

    strategies = [
        {
            "阶段": "模型加载",
            "优化": "懒加载 + 预热",
            "详情": "首次请求时加载模型; 启动后跑几轮 dummy inference"
                   "让 JIT 编译完成",
            "预期收益": "消除冷启动延迟 spike",
        },
        {
            "阶段": "模型格式",
            "优化": "导出为 ONNX / TensorRT",
            "详情": "合并后的 LoRA 权重可以像普通模型一样导出;"
                   "ONNX Runtime 比 PyTorch Eager 快 2-3x",
            "预期收益": "延迟降低 30-60%",
        },
        {
            "阶段": "批处理",
            "优化": "动态批处理 (Dynamic Batching)",
            "详情": "等待窗口内收集多个请求再一起推理;"
                   "比固定 batch 更高效利用 GPU",
            "预期收益": "吞吐量提升 2-5x",
        },
        {
            "阶段": "KV Cache",
            "优化": "PagedAttention (vLLM)",
            "详情": "解决 KV Cache 内存碎片问题;"
                   "支持更高的并发和更长的上下文",
            "预期收益": "显存利用率提升 50%+",
        },
        {
            "阶段": "服务框架",
            "优化": "使用 vLLM / TGI 替代原生 HF",
            "详情": "vLLM 和 TGI 已经内置了上述所有优化;"
                   "开箱即用的高性能推理引擎",
            "预期收益": "端到端 3-10x 加速",
        },
    ]

    print("=" * 80)
    print("推理性能优化全链路")
    print("=" * 80)

    for s in strategies:
        print(f"\n🚀 [{s['阶段']}] {s['优化']}")
        print(f"   详情: {s['详情']}")
        print(f"   预期收益: {s['预期收益']}")

inference_optimization()
```

---

## 四、故障排查手册

## 4.1 训练阶段常见问题

```python
def training_troubleshooting():
    """训练阶段故障排查"""

    problems = [
        {
            "现象": "CUDA Out of Memory (OOM)",
            "可能原因": [
                "batch_size 太大",
                "序列长度过长",
                "没有开启 gradient_checkpointing",
                "没有使用量化 (非 QLoRA)",
                "激活值积累过多",
            ],
            "解决方案": [
                "减小 per_device_train_batch_size 到 1 或 2",
                "减小 max_length (512→256→128)",
                "开启 gradient_checkpointing=True",
                "改用 QLoRA (4-bit)",
                "增加 gradient_accumulation_steps 补偿",
            ],
            "快速诊断": "nvidia-smi 观察峰值显存占用",
        },
        {
            "现象": "Loss 为 NaN",
            "可能原因": [
                "学习率太大",
                "FP16 数值溢出",
                "数据中有异常值 (无穷大/NaN)",
                "某些层权重爆炸",
            ],
            "解决方案": [
                "降低学习率 (2e-4 → 5e-5 → 1e-5)",
                "改用 BF16 (范围更大)",
                "清洗数据: 检查是否有空文本/超长文本",
                "添加梯度裁剪: max_grad_norm=1.0",
            ],
            "快速诊断": "打印每个 batch 的 loss, 找出 NaN 出现的位置",
        },
        {
            "现象": "Loss 不下降 (Stale Loss)",
            "可能原因": [
                "学习率太小",
                "LoRA 没有被正确应用 (target_modules 错误)",
                "数据标签全部相同",
                "优化器状态异常",
            ],
            "解决方案": [
                "增大学习率到 2e-4",
                "print_trainable_parameters() 确认可训练参数 > 0",
                "检查数据中 label 分布是否均匀",
                "重启训练, 清除 optimizer 缓存",
            ],
            "快速诊断": "打印 model 的 requires_grad 状态",
        },
        {
            "现象": "Loss 震荡剧烈",
            "可能原因": [
                "学习率太大",
                "batch size 太小",
                "数据质量差 (噪声多)",
                "warmup 不够",
            ],
            "解决方案": [
                "降低学习率",
                "增大 batch size 或梯度累积步数",
                "清洗数据, 移除低质量样本",
                "增大 warmup_ratio (0.03 → 0.1)",
            ],
            "快速诊断": "画 loss 曲线观察震荡模式",
        },
        {
            "现象": "eval_loss >> train_loss (严重过拟合)",
            "可能原因": [
                "数据量太小",
                "rank 太大",
                "训练 epochs 太多",
                "lora_dropout 太小",
            ],
            "解决方案": [
                "增加数据或数据增强",
                "减小 rank (64→16→8)",
                "早停 (EarlyStoppingCallback)",
                "增大 lora_dropout (0.05→0.1→0.2)",
            ],
            "快速诊断": "train/eval loss 曲线分离点",
        },
    ]

    print("=" * 85)
    print("PEFT 训练故障排查手册")
    print("=" * 85)

    for p in problems:
        print(f"\n🔴 现象: {p['现象']}")
        print(f"   可能原因:")
        for i, reason in enumerate(p['可能原因'], 1):
            print(f"     {i}. {reason}")
        print(f"   解决方案:")
        for i, sol in enumerate(p['解决方案'], 1):
            print(f"     {i}. {sol}")
        print(f"   快速诊断: {p['快速诊断']}")

training_troubleshooting()
```

## 4.2 推理阶段常见问题

```python
def inference_troubleshooting():
    """推理阶段故障排查"""

    problems = [
        {
            "现象": "生成内容全是重复循环",
            "原因": "repetition_penalty 设置不当或未设置",
            "解决": "generate() 时加 repetition_penalty=1.15",
            "进阶": "使用 contrastive search 或 typical sampling",
        },
        {
            "现象": "生成内容与训练数据无关 (灾难性遗忘)",
            "原因": "LoRA 的 alpha/r 太大, 过度修改原始权重",
            "解决": "降低 alpha 或增大 rank (等效于减小缩放)",
            "预防": "始终保留基座模型做 A/B 对比",
        },
        {
            "现象": "生成内容出现幻觉 (Hallucination)",
            "原因": "模型在不确定时倾向于编造信息",
            "解决": "降低 temperature (0.7→0.3); RAG 增强; 后处理校验",
            "注意": "这是 LLM 的固有问题, 无法完全消除",
        },
        {
            "现象": "推理速度很慢 (>2s/token)",
            "原因": "没使用 KV Cache; CPU 推理; 模型太大",
            "解决": "确保 use_cache=True; 用 GPU; 模型量化/蒸馏",
        },
        {
            "现象": "不同 Adapter 切换后输出变差",
            "原因": "set_adapter() 后模型内部状态未正确刷新",
            "解决": "确认每次 generate 前都调用了 set_adapter();"
                   "尝试 reload 后再 set_adapter",
        },
    ]

    print("\n" + "=" * 80)
    print("PEFT 推理故障排查手册")
    print("=" * 80)

    for p in problems:
        print(f"\n🔴 现象: {p['现象']}")
        print(f"   原因: {p['原因']}")
        print(f"   解决: {p['解决']}")
        if '进阶' in p:
            print(f"   进阶: {p['进阶']}")
        if '注意' in p:
            print(f"   ⚠️  注意: {p['注意']}")
        if '预防' in p:
            print(f"   💡 预防: {p['预防']}")

inference_troubleshooting()
```

---

## 五、持续迭代与版本管理

## 5.1 模型版本管理最佳实践

```python
def model_versioning_best_practices():
    """模型版本管理最佳实践"""

    practices = [
        {
            "实践": "语义化版本号",
            "格式": "{task}-{base_model}-peft-{method}-v{MAJOR}.{MINOR}.{PATCH}",
            "示例": "sentiment-bert-base-chinese-lora-v1.2.3",
            "规则": "MAJOR: 架构变更; MINOR: 数据/超参更新; PATCH: bug fix",
        },
        {
            "实践": "完整的 Model Card",
            "内容": "训练数据/超参数/评估结果/已知限制/使用示例",
            "位置": "保存在模型目录下的 README.md",
            "重要性": "让其他人 (包括未来的自己) 能复现你的结果",
        },
        {
            "实践": "保留训练日志",
            "内容": "完整的 TrainingArguments + loss 曲线 + eval 指标历史",
            "格式": "JSON / TensorBoard events / W&B run",
            "用途": "问题排查和效果对比的基础",
        },
        {
            "实践": "Adapter + Tokenizer 打包保存",
            "命令": "model.save_pretrained() + tokenizer.save_pretrained()",
            "原因": "Tokenizer 不一致是推理出错的最常见原因之一",
        },
        {
            "实践": "Git LFS 管理大文件",
            "工具": "git lfs track '*.bin' '*.safetensors'",
            "好处": "版本控制 + 协作 + 回滚",
        },
    ]

    print("=" * 80)
    print("PEFT 模型版本管理最佳实践")
    print("=" * 80)

    for p in practices:
        print(f"\n📋 {p['实践']}")
        for k, v in p.items():
            if k != "实践":
                print(f"   {k}: {v}")

model_versioning_best_practices()
```

## 5.2 A/B 测试框架

```python
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class AdapterVariant(Enum):
    BASE = "base"
    V1 = "adapter_v1"
    V2 = "adapter_v2"


@dataclass
class ExperimentResult:
    variant: AdapterVariant
    latency_ms: float
    success: bool
    user_rating: Optional[int] = None  # 1-5 分


class ABTestFramework:
    """
    PEFT Adapter A/B 测试框架
    用于在线对比不同版本的 Adapter 效果
    """

    def __init__(
        self,
        router,  # MultiAdapterRouter 实例
        traffic_split: Dict[AdapterVariant, float] = None,
    ):
        self.router = router
        self.traffic_split = traffic_split or {
            AdapterVariant.BASE: 0.10,   # 10% 流量给基座 (control)
            AdapterVariant.V1: 0.45,      # 45% 流量给 V1
            AdapterVariant.V2: 0.45,      # 45% 流量给 V2
        }
        self.results: List[ExperimentResult] = []

    def route_request(self, user_id: str) -> AdapterVariant:
        """根据用户 ID 和流量分配决定路由到哪个版本"""
        hash_val = hash(user_id) % 100
        cumulative = 0

        for variant, ratio in self.traffic_split.items():
            cumulative += ratio * 100
            if hash_val < cumulative:
                adapter_name = variant.value
                if adapter_name != "base":
                    self.router.set_adapter(adapter_name)
                return variant

        return AdapterVariant.BASE

    def record_result(self, result: ExperimentResult):
        """记录一次实验结果"""
        self.results.append(result)

    def get_statistics(self) -> dict:
        """计算各版本的统计指标"""
        from collections import defaultdict

        stats = defaultdict(lambda: {"count": 0, "total_latency": 0, "errors": 0, "ratings": []})

        for r in self.results:
            s = stats[r.variant]
            s["count"] += 1
            s["total_latency"] += r.latency_ms
            if not r.success:
                s["errors"] += 1
            if r.user_rating:
                s["ratings"].append(r.user_rating)

        report = {}
        for variant, s in stats.items():
            report[variant.value] = {
                "requests": s["count"],
                "avg_latency_ms": s["total_latency"] / max(s["count"], 1),
                "error_rate": f"{s['errors']/max(s['count'],1)*100:.1f}%",
                "avg_rating": sum(s["ratings"]) / len(s["ratings"]) if s["ratings"] else "N/A",
            }

        return report


def demo_ab_testing():
    """演示 A/B 测试流程"""

    print("=" * 65)
    print("PEFT Adapter A/B 测试演示")
    print("=" * 65)

    print("\n📊 流量分配:")
    print("   基座模型 (Control): 10%")
    print("   Adapter V1:         45%")
    print("   Adapter V2:         45%")

    print("\n🔄 路由逻辑 (基于 user_id hash):")
    test_users = ["user_001", "user_002", "user_003", "user_004", "user_005"]
    for uid in test_users:
        hash_val = hash(uid) % 100
        variant = "V1" if hash_val < 55 else ("V2" if hash_val < 90 else "BASE")
        print(f"   {uid} (hash={hash_val:>3}) → {variant}")

    print("\n📈 运行一周后的统计报告 (模拟):")
    mock_report = {
        "base": {"requests": 1000, "avg_latency_ms": 120.5, "error_rate": "2.1%", "avg_rating": 3.2},
        "adapter_v1": {"requests": 4500, "avg_latency_ms": 125.3, "error_rate": "1.8%", "avg_rating": 4.1},
        "adapter_v2": {"requests": 4500, "avg_latency_ms": 123.8, "error_rate": "1.5%", "avg_rating": 4.3},
    }

    for variant, stats in mock_report.items():
        print(f"\n   [{variant}]")
        for k, v in stats.items():
            print(f"      {k}: {v}")


if __name__ == "__main__":
    demo_ab_testing()
```

## 5.3 回滚机制

```python
class AdapterRollbackManager:
    """
    Adapter 版本回滚管理器
    支持快速回滚到之前的稳定版本
    """

    def __init__(self, storage_path: str = "./adapter_versions"):
        import os
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.version_history = []  # [(version, path, timestamp, metrics)]

    def save_version(self, model, tokenizer, version: str, metrics: dict = None):
        """保存一个新版本"""
        import time
        version_path = f"{self.storage_path}/{version}"
        model.save_pretrained(version_path)
        tokenizer.save_pretrained(version_path)

        entry = {
            "version": version,
            "path": version_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics or {},
        }
        self.version_history.append(entry)

        print(f"💾 已保存版本: {version} → {version_path}")

    def rollback(self, target_version: str = None):
        """
        回滚到指定版本 (默认回滚到上一个稳定版本)
        """
        if not self.version_history:
            raise ValueError("没有可回滚的版本")

        if target_version is None:
            target_version = self.version_history[-2]["version"]

        for entry in reversed(self.version_history):
            if entry["version"] == target_version:
                print(f"↩️  回滚到版本: {target_version}")
                print(f"   路径: {entry['path']}")
                print(f"   保存时间: {entry['timestamp']}")
                if entry["metrics"]:
                    print(f"   当时指标: {entry['metrics']}")
                return entry["path"]

        raise ValueError(f"版本 {target_version} 不存在")

    def list_versions(self):
        """列出所有已保存的版本"""
        print(f"\n📜 Adapter 版本历史:")
        print(f"{'版本':<20} {'时间':<22} {'主要指标'}")
        print("-" * 60)
        for entry in self.version_history:
            metrics_str = str(entry.get("metrics", {}))[:30]
            print(f"{entry['version']:<20} {entry['timestamp']:<22} {metrics_str}")


def demo_rollback():
    """演示回滚机制"""

    manager = AdapterRollbackManager()

    # 模拟保存几个版本
    versions = [
        ("v1.0.0", {"accuracy": 0.89, "f1": 0.88}),
        ("v1.1.0", {"accuracy": 0.91, "f1": 0.90}),
        ("v1.2.0", {"accuracy": 0.87, "f1": 0.86}),  # 这个版本出了问题
        ("v1.2.1-hotfix", {"accuracy": 0.92, "f1": 0.91}),
    ]

    for ver, metrics in versions:
        manager.save_version(None, None, ver, metrics)

    manager.list_versions()
    print("\n发现 v1.2.0 有问题, 执行回滚...")
    manager.rollback("v1.1.0")


if __name__ == "__main__":
    demo_rollback()
```

---

## 六、本章小结

这是 PEFT 章节的最后一节，让我们回顾整个 PEFT 知识体系的要点：

| 层次 | 内容 | 核心要点 |
|------|------|---------|
| **原理** | LoRA / Adapter / Prefix / (IA)³ | ΔW=BA 低秩分解; bottleneck MLP; soft prompt; 缩放向量 |
| **配置** | 不同任务的 rank/target_modules/lr | 分类 r 小, 生成 r 大; NER 需要 seqeval |
| **操作** | 合并 / 切换 / 管理 | merge_and_unload() 消除延迟; set_adapter() 动态切换 |
| **优化** | 显存 / 速度 / 推理 | GC + BF16 + QLoRA + FlashAttention + vLLM |
| **运维** | 监控 / A/B / 回滚 | 全链路评估; 流量分割; 快速回滚 |

**PEFT 的终极价值**：用不到 1% 的参数增量，达到 Full Fine-Tuning 90%+ 的效果，同时支持一套基座服务多个任务。这使得大模型微调从"只有大公司能玩"变成了"每个人都能参与"的技术。

至此，**第 7 章 PEFT 参数高效微调** 全部完成！下一章我们将进入 **第 8 章 文本生成与解码策略**，深入探索大语言模型是如何把 logits 变成人类可读的文本的。
