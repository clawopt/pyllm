# LoRA 最佳实践与性能优化

> **白板时间**：LoRA 能跑起来了——但生产环境还有一堆问题等着你：LoRA 不生效怎么办？多 LoRA 并发时显存爆了怎么处理？切换 LoRA 有延迟吗？和量化一起用有什么坑？这一节我们逐一解决这些实战中的关键问题。

## 一、LoRA 训练与服务衔接

### 1.1 训练时的关键配置

```python
def lora_training_config():
    """vLLM 兼容的 LoRA 训练配置模板"""
    
    config = """
    ════════════════════════════════════════════════
          vLLM 兼容的 LoRA 训练配置 (PEFT)
    ════════════════════════════════════════════════
    
    from peft import LoraConfig
    
    lora_config = LoraConfig(
        # === 必须与 vLLM 兼容的参数 ===
        task_type="CAUSAL_LM",           # ✅ 必须是这个值
        r=16,                           # rank (推荐 8/16/32)
        lora_alpha=32,                   # alpha = 2 × r (推荐)
        target_modules=[                # ⭐ 最重要！必须正确
            "q_proj",
            "k_proj", 
            "v_proj", 
            "o_proj",
            "gate_proj",                  # FFN 层
            "up_proj",
            "down_proj",
        ],
        
        # === 推荐设置 ===
        lora_dropout=0.05,               # 训练时 dropout
        bias="none",                     # 不训练 bias（节省参数）
        modules_to_save=None,            # 不冻结额外模块
        
        # === 高级选项 ===
        use_dora=False,                   # 不用 DoRA（vLLM 暂不支持）
        use_rslora=False,                 # 不用 RsLoRA
        
        # === 推理相关（不影响训练，但建议设好）===
        inference_mode=True,              # 标记为推理模式
    )
    
    ⚠️ 常见错误:
    ├── target_modules 写错名字 → LoRA 加载后不生效
    │   ❌ ["attn.q_proj"] → ✅ ["q_proj"]
    │   ❌ ["linear_q"]   → ✅ ["q_proj"]
    │
    ├── r 太大 → LoRA 文件太大，加载慢
    │   推荐: r=16 (平衡) / r=32 (高质量) / r=64 (极致)
    │
    └── 用了 DoRA / RsLoRA → vLLM 可能不支持
        解决: 设置 use_dora=False, use_rslora=False 后重新导出
    """
    print(config)

lora_training_config()
```

### 1.2 导出为 vLLM 兼容格式

```python
def export_for_vllm(model, tokenizer, output_dir: str):
    """将 PEFT LoRA 导出为 vLLM 兼容格式"""
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 验证输出文件
    import os
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    
    for f in required_files:
        fpath = os.path.join(output_dir, f)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"缺少必要文件: {f}\n"
                f"请确保使用 model.save_pretrained() 导出"
            )
    
    size_mb = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(('.safetensors', '.bin', '.json'))
    ) / 1024 / 1024
    
    print(f"[导出成功] {output_dir}")
    print(f"  大小: {size_mb:.1f} MB")
    print(f"  vLLM 启动参数:")
    print(f"    --lora-modules my-lora={output_dir}")
```

## 二、常见问题排查

### 2.1 问题诊断清单

```python
def lora_troubleshooting():
    """LoRA 问题排查指南"""
    
    guide = """
    ════════════════════════════════════════════════════
              vLLM LoRA 常见问题排查
    ════════════════════════════════════════════════════
    
    ┌─────────────────────────────────────────────────────┐
    │ 问题1: LoRA 不生效（输出和基础模型一样）         │
    ├─────────────────────────────────────────────────────┤
    │ 排查步骤:                                        │
    │  ① 检查 adapter_config.json 的 target_modules     │
    │     cat adapter_config.json | grep target_modules  │
    │     应包含 q_proj/k_proj/v_proj/o_proj 等        │
    │                                                     │
    │  ② 检查 API 调用的 model 字段格式                 │
    │     ✅ base-model@lora-name                       │
    │     ❌ 仅写 lora-name                             │
    │     ❌ 仅写 base-model                            │
    │                                                     │
    │  ③ 检查启动日志是否显示 LoRA 加载成功             │
    │     grep -i "lora" logs/vllm.log                    │
    │                                                     │
    │  ④ 检查 safetensors 文件是否完整                   │
    │     ls -la adapter_model.safetensors               │
    └─────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────┐
    │ 问题2: 加载 LoRA 报错 "OSError" 或 "KeyError"      │
    ├─────────────────────────────────────────────────────┤
    │ 可能原因:                                         │
    │  • target_modules 与实际模型层名不匹配             │
    │  • 使用了 vLLM 不支持的变体 (DoRA/RsLoRA)          │
    │  • safetensors 版本不兼容                          │
    │                                                     │
    │ 解决方案:                                         │
    │  1. 确认模型架构对应的正确 target_modules          │
    │     Llama/Qwen/Mistral: q/k/v/o/gate/up/down     │
    │     Baichuan/Yi: 同上                              │
    │                                                     │
    │  2. 重新导出，确保 use_dora=False                 │
    │     from peft import LoraConfig                    │
    │     LoraConfig(..., use_dora=False)                │
    │                                                     │
    │  3. 升级 safetensors                               │
    │     pip install --upgrade safetensors              │
    └─────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────┐
    │ 问题3: 多 LoRA 显存不足 (OOM)                      │
    ├─────────────────────────────────────────────────────┤
    │ 分析:                                            │
    │  每个 LoRA 虽然小 (~25MB/rank-16)，但运行时有额外开销│
    │  max_loras 决定了预分配的缓存空间                   │
    │                                                     │
    │ 解决方案:                                         │
    │  1. 减少 --max-loras (默认 1，按需增加)           │
    │  2. 减小单个 LoRA 的 rank                         │
    │  3. 使用动态注册/卸载而非全部预加载               │
    │  4. 基础模型启用量化 (AWQ/GPTQ) 腾出空间           │
    └─────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────┐
    │ 问题4: LoRA 切换有延迟 (>100ms)                    │
    ├─────────────────────────────────────────────────────┤
    │ 正常行为:                                         │
    │  首次调用某 LoRA: ~10-50ms (初始化缓存)            │
    │  后续调用同一 LoRA: <5ms (直接命中)               │
    │                                                     │
    │ 如果持续高延迟:                                   │
    │  → 检查是否触发了频繁的 register/unregister         │
    │  → 增加 --max-loras 让更多 LoRA 常驻内存          │
    │  → 启用 Prefix Caching 加速 prompt 处理           │
    └─────────────────────────────────────────────────────┘
    """
    print(guide)

lora_troubleshooting()
```

## 三、LoRA + 量化组合

### 3.1 最佳组合方案

```python
def lora_quantization_combo():
    """LoRA 与量化的组合策略"""
    
    info = """
    ══════════════════════════════════════════════════
          LoRA + 量化组合指南
    ══════════════════════════════════════════════════
    
    ✅ 推荐组合:
    ┌─────────────┬──────────┬──────────┬────────────────┐
    │ Base Model   │ LoRA 格式  │ 效果       │ 适用场景       │
    ├─────────────┼──────────┼──────────┼────────────────┤
    │ AWQ INT4     │ FP16      │ ⭐⭐⭐⭐⭐   │ 生产首选 🏆    │
    │ GPTQ INT4    │ FP16      │ ⭐⭐⭐⭐    │ GPTQ 生态丰富  │
    │ FP16/BF16    │ FP16      │ ⭐⭐⭐⭐⭐   │ 追求最高质量  │
    │ FP8          │ FP16      │ ⭐⭐⭐⭐    │ H100 用户     │
    │ BitsAndBytes│ NF4       │ ⭐⭐⭐⭐    │ QLoRA 原生搭档 │
    └─────────────┴──────────┴──────────┴────────────────┘
    
    ⚠️ 注意事项:
    1. LoRA 权重始终以 FP16 存储（不受基础模型量化影响）
    2. AWQ/GPTQ 量化的 base + FP16 LoRA 是最稳定组合
    3. 不要对 LoRA 本身做 INT4 量化（质量损失大）
    4. BitsAndBytes NF4 base + LoRA = QLoRA 完整流程
    
    🚫 不推荐的组合:
    • INT4 Base + INT4 LoRA → 双重量化误差累积
    • FP8 Base + FP8 LoRA → vLLM 支持有限
    • SqueezeLLM Base + LoRA → 兼容性未验证
    """
    print(info)

lora_quantization_combo()
```

### 3.2 AWQ + LoRA 启动示例

```bash
# Base 模型用 AWQ INT4 量化 + LoRA FP16 适配器
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-3.1-8B-Instruct-AWQ \
    --quantization awq \
    --enable-lora \
    --lora-modules \
        domain-expert=/models/lora/domain-expert-v1 \
    --max-loras 8 \
    --port 8000
```

## 四、性能优化

### 4.1 LoRA 场景的调优参数

```bash
# LoRA 场景专用调优
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules lora1=/path/to/lora1 ... \
    --max-loras 16 \
    --max-lora-rank 64 \
    --enable-prefix-caching \        # ⭐ LoRA 共享 system prompt 时加速显著
    --scheduler-delay-factor 0.1 \   # 低延迟优先
    --max-num-seqs 128 \             # 支持足够并发
    --gpu-memory-utilization 0.90 \
    --port 8000
```

### 4.2 性能基准参考

| 配置 | TTFT | TPOT | QPS (单 LoRA) | QPS (4 LoRA 轮转) |
|------|------|------|---------------|-------------------|
| 无 LoRA, FP16 | 450ms | 42ms | 45 | 44 |
| 无 LoRA, AWQ | 280ms | 28ms | 78 | 76 |
| **1 LoRA, FP16** | **460ms** | **43ms** | **44** | **44** |
| **4 LoRA, FP16** | **462ms** | **44ms** | **43** | **43** |
| **4 LoRA + PrefixCache** | **380ms** | **40ms** | **52** | **51** |

**关键发现**：
- LoRA 开销极小（TTFT 增加 < 5ms）
- 多 LoRA 轮换几乎无额外开销
- **Prefix Caching 在 LoRA 场景下收益更大**（共享 system prompt 时）

---

## 五、总结

本节完成了 LoRA 服务化的最佳实践：

| 主题 | 核心要点 |
|------|---------|
| **target_modules** | 必须含 q/k/v/o/gate/up/down_proj，这是 LoRA 生效的前提 |
| **导出格式** | `model.save_pretrained()` → PEFT 标准 → vLLM 直接加载 |
| **不生效排查** | 检查 model 字段格式、target_modules 匹配、日志确认 |
| **+ 量化组合** | AWQ INT4 + FP16 LoRA = 生产首选；NF4 + LoRA = QLoRA 原生 |
| **显存管理** | `--max-loras` 控制常驻数量；动态注册/卸载按需调整 |
| **性能开销** | LoRA 切换 < 10ms；Prefix Cache 在 LoRA 场景加速显著 |

**全 Chapter 8 核心要点回顾**：

1. **LoRA 让一个模型服务 N 个领域成为可能**——这是多租户 AI 服务的基石
2. **`base-model@lora-name` 是唯一的切换方式**——简单、透明、零侵入
3. **target_modules 是第一 debug 点**——90% 的 LoRA 问题都源于此
4. **AWQ + LoRA 是生产的黄金组合**——最小显存 + 最大灵活性
5. **Prefix Caching 对 LoRA 场景特别有价值**——多个请求共享相同的 system prompt + LoRA

至此，**Chapter 8（LoRA 适配器服务化）全部完成**！接下来进入 **Chapter 9：LangChain / LlamaIndex 集成**。
