# 08-5 量化深度指南

## 量化本质回顾

在前面多个章节中我们都提到了"量化"（Quantization），但从未深入讨论过它的技术细节。这一节将填补这个空白——理解量化的原理和权衡，是做出正确模型选择决策的基础。

### 从 FP16 到 INT4：到底发生了什么

```
┌─────────────────────────────────────────────────────────────┐
│              量化：精度换大小的艺术                           │
│                                                             │
│  原始权重 (FP16 - 半精度浮点):                            │
│  ┌─────┬─────┬─────┬─────┬─────┐                       │
│  │0.12 │-0.34│ 0.56│ 0.78│-0.23│  每个数 16 bit     │
│  └─────┴─────┴─────┴─────┴─────┘                       │
│  总大小: N × 2 bytes (N=参数数量)                        │
│                                                             │
│  Q4_K_M 量化后 (4-bit 整数):                               │
│  ┌─────┬─────┬─────┬─────┬─────┐                       │
│  │  3  │  7  │  4  │  6  │  1  │  每个数 ~4 bit     │
│  └─────┴─────┴─────┴─────┴─────┘                       │
│  总大小: N × 0.5 bytes (压缩 4x!)                         │
│  + 量化查找表 (codebook): ~N×0.01 bytes                    │
│                                                             │
│  推理时:                                                  │
│  原始: weight × input → 精确浮点运算                      │
│  量化: dequantize(weight) × input → 近似浮点运算              │
│        (从 codebook 中查表恢复近似值)                     │
│                                                             │
│  质量影响:                                                 │
│  f16 → q8_0:   几乎无感知 (准确率损失 <1%)                   │
│  f16 → q4_K_M: 可感知但可接受 (损失 3-5%)                  │
│  f16 → q2_K:  明显质量下降 (损失 >10%)                     │
│                                                             │
│  为什么不直接用 1-bit / 2-bit?                              │
│  → 极端量化后某些语义信息永久丢失                          │
│  → 但对于简单分类任务可能仍然够用                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Ollama 中的量化选择：完整对比

### 同一模型不同量化级别的 A/B 测试

```python
#!/usr/bin/env python3
"""量化级别对比测试"""

import requests
import time


def test_quantization_levels():
    """对同一问题用不同量化级别的模型回答，比较质量"""
    
    model_variants = [
        ("qwen2.5:7b-f16", "半精度 (基准)"),
        ("qwen2.5:7b-q8_0", "8-bit"),
        ("qwen2.5:7b-q5_K_M", "5-bit K-Medium (推荐)"),
        ("qwen2.5:7b-q4_K_M", "4-bit K-Medium (最常用)"),
        ("qwen2.5:7b-q4_0", "4-bit legacy (旧版)"),
        ("qwen2.5:7b-q3_K_S", "3-bit K-Small (极限)"),
        ("qwen2.5:7b-q2_K", "2-bit (极端)"),
    ]
    
    test_questions = [
        {
            "q": "请解释量子纠缠现象",
            "keywords": ["量子", "纠缠", "爱因斯坦", "贝尔", "超距"],
            "type": "科学"
        },
        {
            "q": "写一个 Python 快速排序算法",
            "keywords": ["def", "quicksort", "partition", "return", "list"],
            "type": "代码"
        },
        {
            "q": "将以下英文翻译成中文：'The deployment of large language models at the edge presents unique challenges in terms of memory bandwidth and thermal management.'",
            "keywords": ["部署", "大语言模型", "边缘", "内存带宽"],
            "type": "翻译"
        },
        {
            "q": "计算斐波那契数列的第10项",
            "keywords": ["55", "斐波那契", "fibonacci"],
            "type": "数学"
        },
    ]
    
    print(f"\n{'='*75}")
    print(f"  量化级别 A/B 对比测试")
    print(f"{'='*75}\n")
    
    for model_name, label in model_variants:
        
        # 检查模型是否存在
        try:
            resp = requests.post("http://localhost:11434/api/chat", json={
                "model": model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "options": {"num_predict": 5}
            }, timeout=30)
            
            if resp.status_code != 200:
                print(f"  ⚠️ {label}: 模型不可用")
                continue
                
        except Exception as e:
            print(f"  ⚠️ {label}: 错误 {e}")
            continue
        
        # 运行测试
        scores = []
        total_time = 0
        
        for tq in test_questions:
            start = time.time()
            resp = requests.post("http://localhost:11434/api/chat", json={
                "model": model_name,
                "messages": [{"role": "user", "content": tq["q"]}],
                "stream": False,
                "options": {"temperature": 0.1, "seed": 42}
            }, timeout=120)
            
            elapsed = time.time() - start
            total_time += elapsed
            
            content = resp.json()["message"]["content"].lower()
            
            # 关键词命中率
            hits = sum(1 for kw in tq["keywords"] if kw.lower() in content)
            score = hits / len(tq["keywords"])
            scores.append(score)
            
            # Token 数
            tokens = resp.json().get("eval_count", 0)
        
        avg_score = sum(scores) / len(scores)
        avg_tps = sum(
            t["eval_count"] for t in [resp.json()] 
            if isinstance(t, dict)
        ) / total_time if total_time > 0 else 0
        
        size_resp = requests.post("http://localhost:11434/api/show", 
                                json={"name": model_name}, timeout=10)
        size_info = size_resp.json()
        model_size = size_info.get("sizeinfo", {}).get("parameter_size", "?")
        size_bytes = size_info.get("size", 0)
        size_gb = size_bytes / (1024**3)
        
        bar = "█" * int(avg_score * 20) + "░" * (20 - int(avg_score * 20))
        
        print(f"  {label:<28s} | 质量:{avg_score:.2f} {bar} | "
              f"{avg_tps:>5.1f} t/s | {size_gb:.1f}GB")


if __name__ == "__main__":
    test_quantization_levels()
```

### 典型输出解读

```
===========================================================================
  量化级别 A/B 对比测试
===========================================================================

  半精度 (基准)          | 质量:0.95 ████████████████░░ |  12.3 t/s | 15.0GB
  8-bit                 | 质量:0.94 ███████████████░░░ |  18.5 t/s |  7.5GB
  5-bit K-Medium (推荐) | 质量:0.91 █████████████░░░░░ |  22.1 t/s |  5.2GB ★
  4-bit K-Medium (最常用) | 质量:0.87 ████████████░░░░░░ |  25.8 t/s |  4.3GB ★★
  4-bit legacy           | 质量:0.83 ███████████░░░░░░░ |  26.2 t/s |  4.2GB
  3-bit K-Small         | 质量:0.71 █████████░░░░░░░░ |  31.4 t/s |  2.9GB
  2-bit (极端)           | 质量:0.52 ██████░░░░░░░░░░░ |  38.2 t/s |  1.5GB
```

**关键发现**：
- **Q4_K_M 是甜点**——质量损失仅 8% 左右（0.95→0.87），体积缩小 3.5 倍，速度提升 2 倍+
- **Q2_K 质量下降明显**——数学/代码类任务可能出错或产生幻觉
- **翻译任务对量化最敏感**——细微的语义差别可能被丢失
- **代码任务相对鲁棒**——即使 q3_K_S 也能生成基本正确的语法结构

## "够用就好"原则的决策框架

```python
#!/usr/bin/env python3
"""基于场景的量化级别推荐引擎"""


def recommend_quantization(available_ram_gb, task_type, quality_requirement):
    """
    根据实际条件推荐量化级别
    
    Args:
        available_ram_gb: 可用内存 (GB)
        task_type: creative/code/translation/embedding/rqa/math
        quality_requirement: low/medium/high/critical
    """
    
    recommendations = []
    
    # 嵌入模型：始终使用高精度
    if task_type == "embedding":
        if available_ram_gb >= 2:
            return [("nomic-embed-text", "f16/bf16", "嵌入模型需要高精度")]
        else:
            return [("tinyembed-1b", "q4_0", "极小嵌入模型")]
    
    # 数学/推理任务：不低于 q5_K_M
    if task_type in ["math", "reasoning"]:
        if available_ram_gb >= 32:
            return [("qwen2.5:32b-q5_K_M", "Q5_K_M", "数学需要精度")]
        elif available_ram_gb >= 16:
            return [("qwen2.5:14b-q5_K_M", "Q5_K_M", "14B + Q5")]
        elif available_ram_gb >= 8:
            return [("qwen2.5:7b-q5_K_M", "Q5_K_M", "7B + Q5 (最小推荐)")]
        elif available_ram_gb >= 4:
            return [("qwen2.5:3b-q5_K_M", "Q5_K_M", "3B + Q5 (勉强)")]
        else:
            return [("phi3:mini-q4_0", "Q4_0", "资源极度有限")]
    
    # 代码生成：q4_K_M 通常足够
    if task_type == "code":
        if quality_requirement in ["high", "critical"] and available_ram_gb >= 24:
            return [("deepseek-coder:33b-q4_K_M", "Q4_K_M", "高质量代码")]
        elif available_ram_gb >= 8:
            return [("deepseek-coder:6.7b-q4_K_M", "Q4_K_M", "标准代码质量")]
        else:
            return [("starcoder2:3b-q4_0", "Q4_0", "轻量代码")]
    
    # 创意写作：可以接受较低精度
    if task_type == "creative":
        if quality_requirement == "critical":
            return [("llama3.1:8b-q5_K_M", "Q5_K_M", "高质量创作")]
        elif quality_requirement == "low":
            if available_ram_gb < 4:
                return [("gemma2:2b-q4_0", "Q4_0", "快速创意")]
            return [("qwen2.5:7b-q4_K_M", "Q4_K_M", "标准创作")]
        else:
            return [("qwen2.5:7b-q5_K_M", "Q5_K_M", "平衡创作")]
    
    # 通用对话/翻译：q4_K_M 是默认推荐
    default_rec = [
        ("qwen2.5:7b-q4_K_M", "Q4_K_M (通用推荐)", True),
        ("qwen2.5:7b-q5_K_M", "Q5_K_M (高质量)", False),
        ("qwen2.5:7b-q8_0", "Q8_0 (高保真)", False),
    ]
    
    for rec, _, _ in default_rec:
        # 简单检查内存是否够用
        estimated_size_gb = {"f16": 15, "q8_0": 7.5, "q5_K_M": 5.2, 
                           "q4_K_M": 4.3, "q4_0": 4.2}.get(rec[1].split()[-1], 5)
        if estimated_size_gb * 1.2 <= available_ram_gb:
            return [(rec[0], rec[1], rec[2])]
    
    return default_rec[:1]


if __name__ == "__main__":
    scenarios = [
        (8, "chat", "medium"),      # 8GB Mac, 通用对话
        (16, "code", "high"),       # 16GB, 高质量代码
        (4, "creative", "low"),      # 4GB, 随便写写
        (48, "math", "critical"),    # 48GB, 数学推理
        (64, "embedding", "high"),   # 64GB, 向量嵌入
    ]
    
    for ram, task, quality in scenarios:
        result = recommend_quantization(ram, task, quality)
        print(f"\nRAM={ram}GB | Task={task:10s} | Quality={quality}")
        print(f"  推荐: {result[0]} ({result[1]}) — {result[2]}")
```

## 混合精度：注意力层 fp16 + FFN 层 q4

这是 llama.cpp 社区的一个高级优化方向——**不是所有层都需要同样的精度**：

- **注意力层（Attention）**: 对精度极其敏感——微小的权重误差会被 softmax 放大，导致 attention 分布失真
- **前馈网络层（FFN）**: 相对鲁棒——它主要做非线性变换，一定的量化误差不会造成灾难性后果

因此最优策略是：**Attention 用 fp16 或 q8_0，FFN 用 q4_K_M**。

Ollama 目前不支持这种混合精度配置（需要在编译 llama.cpp 时手动设置），但了解这个方向有助于你理解为什么某些量化方案效果更好。

## 量化感知训练 vs 事后量化

| 维度 | 事后量化（Ollama 默认） | 量化感知训练 (QAT) |
|------|------------------------|------------------|
| **流程** | 训练好的模型 → 直接量化 | 在训练过程中模拟量化 |
| **工具** | `llama-quantize` 一条命令 | 需要 PyTorch + 特殊训练脚本 |
| **灵活性** | ✅ 任意时间、任意粒度 | ❌ 必须重新训练 |
| **质量** | 好（大多数场景） | 更好（特别是低 bit） |
| **适用性** | ★★★★★ | ★★☆☆☆ |
| **谁在做** | 所有 Ollama 用户 | Meta/Mistral/Qwen 团队 |

**结论**: 对于 99% 的 Ollama 用户，事后量化完全够用。只有当你有特殊需求（如 2-3bit 极限压缩且要求保持质量）时才需要考虑 QAT。

## 本章小结

这一节完成了对量化技术的深度探讨：

1. **量化本质是用精度换大小**——FP16 权重通过查找表映射为 4-bit 整数，推理时再恢复为近似浮点值
2. **Q4_K_M 是绝大多数场景的最优解**——3-5% 的质量损失换来 4x 的体积缩减和 2x+ 的速度提升
3. **A/B 测试证明**：翻译 > 数学 > 代码 > 创作写作（对量化的敏感度递减）
4. **嵌入式任务应该用 f16/bf16**——向量相似度对精度极其敏感
5. **混合精度（Attention fp16 + FFN q4）** 是前沿方向，能获得更好的精度-速度平衡
6. **"够用就好"原则**：根据任务类型、硬件限制和质量要求动态选择量化级别

至此，第八章"性能优化与资源管理"全部完成。
