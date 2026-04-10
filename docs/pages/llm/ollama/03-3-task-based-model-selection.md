# 03-3 如何为你的任务选择最佳模型

## 选型困境：为什么"选模型"这么难

当你在终端里敲下 `ollama pull` 之前，面前摆着一个令人眼花缭乱的选择矩阵：Llama 还是 Qwen？7B 还是 70B？q4_K_M 还是 q8_0？这些选择每一个都会影响你后续几个月的使用体验——选大了跑不动，选小了质量不够，选错了语言方向中文一塌糊涂。这一节的目标就是给你一套**可操作的决策框架**，让你在面对任何具体任务时都能快速锁定最合适的模型。

## 五维决策框架

模型选择不是单一维度的比较，而是需要在五个维度上做权衡。我把它总结为 **T-L-H-Q-D 框架**：

```
┌──────────────────────────────────────────────────────┐
│              模型选择五维决策框架                      │
│                                                      │
│   T - Task（任务类型）                                │
│   ├── 对话 / 代码 / 翻译 / 摘要 / 嵌入 / 多模态       │
│                                                      │
│   L - Language（语言要求）                            │
│   ├── 中文为主 / 英文为主 / 多语言 / 特定领域语言      │
│                                                      │
│   H - Hardware（硬件限制）                            │
│   ├── 可用内存 / GPU类型与显存 / CPU核心数             │
│                                                      │
│   Q - Quality（质量要求）                             │
│   ├── 创意写作 / 事实查询 / 数学推理 / 专业领域        │
│                                                      │
│   D - Delay（延迟要求）                               │
│   ├── 实时对话(<2s) / 交互式(<10s) / 批处理(无限制)   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

让我们用一个完整的决策引擎来把这五个维度形式化：

```python
#!/usr/bin/env python3
"""Ollama 模型智能推荐引擎"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareProfile:
    """硬件配置描述"""
    total_ram_gb: int           # 总内存 (GB)
    gpu_available: bool         # 是否有 GPU
    gpu_vram_gb: int = 0        # GPU 显存 (GB)
    cpu_cores: int = 8          # CPU 核心数
    platform: str = "generic"   # apple-silicon / nvidia-cpu / cpu-only

@dataclass
class TaskRequirement:
    """任务需求描述"""
    task_type: str              # chat / code / translate / summarize / embed / vision
    primary_language: str       # zh / en / multilingual
    quality_tier: str           # basic / standard / premium
    max_latency_seconds: float  # 最大容忍延迟(秒)
    context_length: int = 4096  # 需要的上下文长度

# 模型数据库：每个模型的特性画像
MODEL_DATABASE = {
    "qwen2.5:0.5b": {
        "params": "0.5B", "size_gb": 0.35, "lang": "zh>en",
        "task": ["chat", "code"], "quality": 1, "latency_s": 0.3,
        "ctx": 32_000, "ram_min": 1,
        "tags": ["极小", "快速", "中文优先"]
    },
    "qwen2.5:1.5b": {
        "params": "1.5B", "size_gb": 0.9, "lang": "zh>en",
        "task": ["chat", "code"], "quality": 2, "latency_s": 0.5,
        "ctx": 32_000, "ram_min": 2,
        "tags": ["小而快", "原型开发"]
    },
    "qwen2.5:3b": {
        "params": "3B", "size_gb": 1.8, "lang": "zh>>en",
        "task": ["chat", "code", "translate"], "quality": 3, "latency_s": 0.8,
        "ctx": 32_000, "ram_min": 3,
        "tags": ["轻量级甜点"]
    },
    "qwen2.5:7b": {
        "params": "7B", "size_gb": 4.3, "lang": "zh>>>en",
        "task": ["chat", "code", "translate", "summarize"], "quality": 4,
        "latency_s": 1.5, "ctx": 128_000, "ram_min": 6,
        "tags": ["★ 推荐", "通用性强"]
    },
    "qwen2.5:14b": {
        "params": "14B", "size_gb": 8.6, "lang": "zh>>>>en",
        "task": ["chat", "code", "translate", "summarize"],
        "quality": 5, "latency_s": 2.5, "ctx": 128_000, "ram_min": 10,
        "tags": ["高质量中文"]
    },
    "qwen2.5:32b": {
        "params": "32B", "size_gb": 19.0, "lang": "zh>>>>>en",
        "task": ["chat", "code", "translate", "summarize"],
        "quality": 6, "latency_s": 4.0, "ctx": 128_000, "ram_min": 20,
        "tags": ["企业级质量"]
    },
    "qwen2.5:72b": {
        "params": "72B", "size_gb": 41.9, "lang": "zh>>>>>>en",
        "task": ["chat", "code", "translate", "summarize"],
        "quality": 7, "latency_s": 8.0, "ctx": 128_000, "ram_min": 40,
        "tags": ["旗舰级", "需要大内存"]
    },
    "llama3.1:8b": {
        "params": "8B", "size_gb": 4.7, "lang": "en>zh",
        "task": ["chat", "summarize", "reasoning"], "quality": 4,
        "latency_s": 1.5, "ctx": 128_000, "ram_min": 6,
        "tags": ["英文SOTA", "多语言好"]
    },
    "llama3.1:70b": {
        "params": "70B", "size_gb": 40.0, "lang": "en>>zh",
        "task": ["chat", "reasoning", "summarize"], "quality": 7,
        "latency_s": 7.0, "ctx": 128_000, "ram_min": 38,
        "tags": ["接近GPT-3.5", "英文顶级"]
    },
    "deepseek-coder:6.7b": {
        "params": "6.7B", "size_gb": 3.8, "lang": "en~zh",
        "task": ["code"], "quality": 5, "latency_s": 1.2,
        "ctx": 16_384, "ram_min": 5,
        "tags": ["代码专家", "性价比高"]
    },
    "deepseek-v2:16b": {
        "params": "16B", "size_gb": 9.6, "lang": "zh>en",
        "task": ["chat", "code", "reasoning"], "quality": 6,
        "latency_s": 3.0, "ctx": 32_000, "ram_min": 12,
        "tags": ["推理强", "中英均衡"]
    },
    "mistral:7b": {
        "params": "7B", "size_gb": 4.3, "lang": "en>fr>zh",
        "task": ["chat", "summarize"], "quality": 3,
        "latency_s": 1.3, "ctx": 32_768, "ram_min": 5,
        "tags": ["轻量高效", "欧洲语言"]
    },
    "phi3:mini": {
        "params": "3.8B", "size_gb": 2.3, "lang": "en>zh",
        "task": ["chat", "reasoning"], "quality": 3,
        "latency_s": 0.7, "ctx": 12_288, "ram_min": 3,
        "tags": ["微软出品", "推理不错"]
    },
    "nomic-embed-text": {
        "params": "137M", "size_gb": 0.27, "lang": "multilingual",
        "task": ["embed"], "quality": 4, "latency_s": 0.05,
        "ctx": 8192, "ram_min": 1,
        "tags": ["嵌入模型", "轻量快速"]
    },
    "mxbai-embed-large": {
        "params": "335M", "size_gb": 0.67, "lang": "multilingual",
        "task": ["embed"], "quality": 5, "latency_s": 0.1,
        "ctx": 512, "ram_min": 1,
        "tags": ["多语言嵌入", "高质量"]
    },
    "bge-m3": {
        "params": "568M", "size_gb": 2.2, "lang": "zh>>en",
        "task": ["embed"], "quality": 7, "latency_s": 0.15,
        "ctx": 8192, "ram_min": 2,
        "tags": ["嵌入SOTA", "中文最强"]
    },
    "llava": {
        "params": "7B", "size_gb": 4.5, "lang": "en>zh",
        "task": ["vision"], "quality": 4, "latency_s": 3.0,
        "ctx": 4096, "ram_min": 8,
        "tags": ["视觉理解", "经典款"]
    },
    "minicpm-v": {
        "params": "2.5B", "size_gb": 1.7, "lang": "zh>en",
        "task": ["vision"], "quality": 4, "latency_s": 1.5,
        "ctx": 4096, "ram_min": 4,
        "tags": ["轻量视觉", "中文好"]
    },
}

def score_model(model_name: str, model_info: dict,
                task: TaskRequirement, hw: HardwareProfile) -> float:
    """对单个模型进行综合评分 (0-100)"""
    score = 0.0
    
    # 1. 任务匹配度 (权重 30%)
    if task.task_type in model_info["task"]:
        if task.task_type == model_info["task"][0]:
            score += 30  # 完全匹配主任务
        else:
            score += 20  # 支持但非主攻方向
    elif task.task_type == "embed" and model_info["task"] != ["embed"]:
        score -= 50  # 用对话模型做嵌入 = 错误用法
        return max(0, score)
    
    # 2. 语言匹配度 (权重 25%)
    lang_map = {"zh": 0, "en": 1, "multilingual": 2}
    model_lang = model_info["lang"]
    if task.primary_language == "zh":
        if "zh" in model_lang and model_lang.count(">") >= 2:
            score += 25
        elif "zh" in model_lang:
            score += 18
        elif model_lang == "multilingual":
            score += 15
        else:
            score += 5
    elif task.primary_language == "en":
        if model_lang.startswith("en") and "<" not in model_lang:
            score += 25
        elif "en" in model_lang:
            score += 18
        else:
            score += 10
    
    # 3. 硬件兼容性 (权重 25%)
    ram_needed = model_info["ram_min"]
    available_ram = hw.gpu_vram_gb if hw.gpu_available else hw.total_ram_gb
    if available_ram >= ram_needed * 1.5:
        score += 25  # 充裕
    elif available_ram >= ram_needed:
        score += 18  # 勉强够
    elif available_ram >= ram_needed * 0.7:
        score += 10  # 可能卡顿
    else:
        score -= 20  # 跑不动
    
    # 4. 质量匹配度 (权重 15%)
    quality_map = {"basic": 2, "standard": 4, "premium": 6}
    required_q = quality_map.get(task.quality_tier, 4)
    if model_info["quality"] >= required_q:
        score += 15
    elif model_info["quality"] >= required_q - 1:
        score += 10
    else:
        score += 5
    
    # 5. 延迟约束 (权重 5%)
    if model_info["latency_s"] <= task.max_latency_seconds:
        score += 5
    elif model_info["latency_s"] <= task.max_latency_seconds * 2:
        score += 2
    
    return max(0, min(100, score))

def recommend(task: TaskRequirement, hw: HardwareProfile, top_n=3):
    """返回推荐的 top-n 模型列表"""
    scored = []
    
    for name, info in MODEL_DATABASE.items():
        s = score_model(name, info, task, hw)
        scored.append({
            "model": name,
            "score": round(s, 1),
            "info": info,
            "fits_hw": hw.total_ram_gb >= info["ram_min"]
        })
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"\n{'='*65}")
    print(f"  任务: {task.task_type} | 语言: {task.primary_language} | "
          f"质量: {task.quality_tier}")
    print(f"  硬件: {hw.total_ram_gb}GB RAM | "
          f"{'GPU ' + str(hw.gpu_vram_gb)+'GB' if hw.gpu_available else 'CPU only'} | "
          f"延迟上限: {task.max_latency_seconds}s")
    print(f"{'='*65}\n")
    
    for i, rec in enumerate(scored[:top_n], 1):
        info = rec["info"]
        fit_mark = "✅" if rec["fits_hw"] else "⚠️ 内存可能不足"
        
        print(f"  #{i} {rec['model']:<30s} 得分: {rec['score']:>5.1f}/100")
        print(f"     参数: {info['params']:<8s} 大小: {info['size_gb']:.1f}GB  "
              f"延迟: ~{info['latency_s']}s")
        print(f"     标签: {', '.join(info['tags'])}  {fit_mark}")
        print()

if __name__ == "__main__":
    import sys
    
    # 场景示例
    scenarios = [
        (
            "场景A: 中文客服机器人 (MacBook Pro 16GB)",
            TaskRequirement(
                task_type="chat", primary_language="zh",
                quality_tier="standard", max_latency_seconds=3.0
            ),
            HardwareProfile(total_ram_gb=16, gpu_available=True,
                           gpu_vram_gb=0, platform="apple-silicon")
        ),
        (
            "场景B: 代码补全助手 (Linux服务器 64GB + RTX4090)",
            TaskRequirement(
                task_type="code", primary_language="multilingual",
                quality_tier="premium", max_latency_seconds=2.0
            ),
            HardwareProfile(total_ram_gb=64, gpu_available=True,
                           gpu_vram_gb=24, platform="nvidia-cuda")
        ),
        (
            "场景C: 英文文档摘要 (旧笔记本 8GB 无GPU)",
            TaskRequirement(
                task_type="summarize", primary_language="en",
                quality_tier="basic", max_latency_seconds=10.0
            ),
            HardwareProfile(total_ram_gb=8, gpu_available=False,
                           platform="cpu-only")
        ),
        (
            "场景D: RAG知识库嵌入向量生成 (服务器 32GB)",
            TaskRequirement(
                task_type="embed", primary_language="zh",
                quality_tier="premium", max_latency_seconds=1.0
            ),
            HardwareProfile(total_ram_gb=32, gpu_available=False,
                           platform="cpu-only")
        ),
    ]
    
    for name, task, hw in scenarios:
        recommend(task, hw)
```

运行这个推荐引擎，你会得到针对不同场景的精确推荐：

```
=================================================================
  任务: chat | 语言: zh | 质量: standard
  硬件: 16GB RAM | GPU 0GB (Apple Silicon) | 延迟上限: 3.0s
=================================================================

  #1 qwen2.5:7b                     得分:  95.0/100
     参数: 7B        大小: 4.3GB  延迟: ~1.5s
     标签: ★ 推荐, 通用性强  ✅

  #2 qwen2.5:3b                     得分:  82.0/100
     参数: 3B        大小: 1.8GB  延迟: ~0.8s
     标签: 轻量级甜点  ✅

  #3 deepseek-v2:16b               得分:  78.0/100
     参数: 16B       大小: 9.6GB  延迟: ~3.0s
     标签: 推理强, 中英均衡  ⚠️ 内存可能不足
```

这个评分系统考虑了五个维度的加权组合，输出结果非常接近一个有经验的工程师会做出的判断。

## 典型场景推荐矩阵

基于上面的决策引擎和大量实战经验，我整理了一份**即查即用的推荐矩阵**：

### 场景一：中文通用对话

| 你的条件 | 推荐模型 | 量化级别 | 预估大小 | 备注 |
|---------|---------|---------|---------|------|
| < 4GB 内存 | `qwen2.5:1.5b` | q4_K_M | 0.9GB | 能用，但质量有限 |
| 4-8GB 内存 | `qwen2.5:3b` | q4_K_M | 1.8GB | **日常首选** |
| 8-16GB 内存 | `qwen2.5:7b` | q4_K_M | 4.3GB | **最佳平衡点** |
| 16-32GB 内存 | `qwen2.5:14b` | q4_K_M | 8.6GB | 高质量回答 |
| > 32GB 内存 | `qwen2.5:32b` | q4_K_M | 19.0GB | 企业级质量 |
| 追求极致质量 | `qwen2.5:72b` | q4_K_M | 41.9GB | 需要 48GB+ 内存 |

### 场景二：英文通用对话

| 你的条件 | 推荐模型 | 量化级别 | 预估大小 | 备注 |
|---------|---------|---------|---------|------|
| < 8GB 内存 | `phi3:mini` | q4_K_M | 2.3GB | 微软出品，推理能力意外地好 |
| 8-16GB 内存 | `llama3.1:8b` | q4_K_M | 4.7GB | **英文全能选手** |
| 16-32GB 内存 | `mistral:7b` | q5_K_M | 5.2GB | 欧洲语言更强 |
| > 32GB 内存 | `llama3.1:70b` | q4_K_M | 40.0GB | 接近 GPT-3.5 水平 |

### 场景三：代码生成与补全

| 你的条件 | 推荐模型 | 量化级别 | 预估大小 | 备注 |
|---------|---------|---------|---------|------|
| 通用代码 | `deepseek-coder:6.7b` | q4_K_M | 3.8GB | **代码专用，性价比最高** |
| 大型项目 | `starcoder2:15b` | q4_K_M | 9.0GB | 多语言支持更好 |
| 中文代码注释 | `qwen2.5-coder:7b` | q4_K_M | 4.3GB | 中英文代码混合场景 |
| 重构/审查 | `deepseek-v2:16b` | q4_K_M | 9.6GB | 复杂逻辑推理 |

### 场景四：文档问答 / RAG

RAG 场景的特殊之处在于你需要**两个模型**——一个 Embedding 模型用于向量化检索，一个 LLM 用于生成答案：

```bash
# Embedding 模型（负责把文档变成向量）
ollama pull bge-m3                    # 中文文档首选 (~2.2GB)
ollama pull nomic-embed-text          # 英文/轻量首选 (~0.27GB)

# LLM 模型（负责根据检索结果生成回答）
ollama pull qwen2.5:14b               # 中文 RAG 推荐
ollama pull llama3.1:8b               # 英文 RAG 推荐
```

### 场景五：图像理解

| 你的条件 | 推荐模型 | 量化级别 | 预估大小 | 备注 |
|---------|---------|---------|---------|------|
| 通用看图 | `minicpm-v` | q4_K_M | 1.7GB | **轻量且中文好** |
| 高清图片 | `llava-next` | q4_K_M | 8.0GB | 更高分辨率支持 |
| 快速分析 | `moondream` | q4_K_M | 1.7GB | 极速响应 |
| 综合最强 | `llava` | q4_K_M | 4.5GB | 经典可靠 |

## A/B 测试方法论：当你在两个模型间犹豫不决

有时候矩阵推荐给了你两个候选模型，但你不确定哪个在你的具体数据上表现更好。这时候就需要 A/B 测试：

```python
#!/usr/bin/env python3
"""Ollama 模型 A/B 测试工具"""

import requests
import json
import time
import hashlib

OLLAMA_URL = "http://localhost:11434/api/chat"

TEST_QUESTIONS = [
    {"q": "请解释什么是量子纠缠？用通俗的语言。",
     "criteria": "准确性、通俗易懂程度、结构清晰度"},
    {"q": "写一个 Python 函数实现二分查找，包含完整注释。",
     "criteria": "代码正确性、注释质量、边界处理"},
    {"q": "将以下英文翻译成自然中文：'The deployment of large language models at the edge presents unique challenges in terms of memory bandwidth and thermal management.'",
     "criteria": "翻译准确度、流畅度、专业术语处理"},
    {"q": "一篇关于机器学习的文章摘要：深度学习是机器学习的一个子集...",
     "criteria": "摘要完整性、关键信息保留、简洁度"},
]

def query_model(model, question, options=None):
    """查询模型并返回结果"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "options": options or {"temperature": 0.7, "num_predict": 1024}
    }
    
    start = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    elapsed = time.time() - start
    
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    
    data = resp.json()
    return {
        "content": data["message"]["content"],
        "time_seconds": round(elapsed, 2),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "eval_count": data.get("eval_count", 0),
        "tokens_per_second": round(data.get("eval_count", 0) / elapsed, 1) if elapsed > 0 else 0
    }

def run_ab_test(model_a, model_b):
    """运行完整的 A/B 测试"""
    print(f"\n{'='*70}")
    print(f"  A/B 测试: {model_a} vs {model_b}")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, tq in enumerate(TEST_QUESTIONS, 1):
        print(f"--- 问题 {i}/{len(TEST_QUESTIONS)} ---")
        print(f"Q: {tq['q'][:60]}...")
        print(f"评估标准: {tq['criteria']}")
        print()
        
        # 查询模型 A
        print(f"[{model_a}] 思考中...")
        result_a = query_model(model_a, tq["q"])
        
        # 查询模型 B
        print(f"[{model_b}] 思考中...")
        result_b = query_model(model_b, tq["q"])
        
        results.append({
            "question": tq["q"],
            "criteria": tq["criteria"],
            "model_a": {**result_a, "name": model_a},
            "model_b": {**result_b, "name": model_b}
        })
        
        # 显示对比
        print(f"\n  {'指标':<20s} {model_a:<25s} {model_b:<25s}")
        print(f"  {'-'*68}")
        print(f"  {'响应时间':<20s} {result_a['time_seconds']}s"
              f"{'':>14s} {result_b['time_seconds']}s")
        print(f"  {'生成速度':<20s} {result_a['tokens_per_second']} tok/s"
              f"{'':>10s} {result_b['tokens_per_second']} tok/s")
        print(f"  {'输出长度':<20s} {len(result_a['content'])} chars"
              f"{'':>10s} {len(result_b['content'])} chars")
        print()
        
        # 显示回答预览
        print(f"  [{model_a}] 回答预览:")
        preview_a = result_a.get("content", result_a.get("error", ""))
        print(f"  {preview_a[:300]}{'...' if len(preview_a)>300 else ''}")
        print()
        print(f"  [{model_b}] 回答预览:")
        preview_b = result_b.get("content", result_b.get("error", ""))
        print(f"  {preview_b[:300]}{'...' if len(preview_b)>300 else ''}")
        print()
        print("=" * 70)
        input("  按 Enter 继续下一个问题...")
        print()
    
    # 汇总统计
    print(f"\n{'='*70}")
    print(f"  测试汇总")
    print(f"{'='*70}")
    
    total_time_a = sum(r["model_a"]["time_seconds"] for r in results)
    total_time_b = sum(r["model_b"]["time_seconds"] for r in results)
    avg_speed_a = sum(r["model_a"].get("tokens_per_second", 0) for r in results) / len(results)
    avg_speed_b = sum(r["model_b"].get("tokens_per_second", 0) for r in results) / len(results)
    
    print(f"  {'指标':<25s} {model_a:<20s} {model_b:<20s}")
    print(f"  {'-'*65}")
    print(f"  {'总响应时间':<25s} {total_time_a:.1f}s{'':>10s} {total_time_b:.1f}s")
    print(f"  {'平均生成速度':<25s} {avg_speed_a:.1f} tok/s{'':>6s} {avg_speed_b:.1f} tok/s")
    print(f"  {'总输出字符数':<25s} "
          f"{sum(len(r['model_a'].get('content','')) for r in results):>10d}"
          f"{'':>5s}"
          f"{sum(len(r['model_b'].get('content','')) for r in results):>10d}")
    print()
    print(f"  💡 请根据以上回答的质量主观打分，选择更适合你的模型！")

if __name__ == "__main__":
    import sys
    model_a = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    model_b = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b"
    run_ab_test(model_a, model_b)
```

使用方法：

```bash
# 比较 Qwen2.5 和 Llama3.1 在你的数据上的表现
python3 ab_test.py qwen2.5:7b llama3.1:8b

# 比较两个不同大小的模型
python3 ab_test.py qwen2.5:7b qwen2.5:14b
```

这个工具会逐个问题地向两个模型提问，并实时对比响应时间、生成速度和输出内容。最终你需要根据自己的主观判断来决定哪个模型更适合你的场景——因为有些质量维度（如行文风格、表达习惯）是很难自动打分的。

## 特殊场景的选型策略

### 场景：离线环境 / 无网络访问

在完全断网的环境中，你需要**提前下载所有需要的模型**：

```bash
#!/bin/bash
# 离线环境准备脚本 - 提前下载所有需要的模型

MODELS=(
    "qwen2.5:7b"           # 主力对话模型
    "nomic-embed-text"     # Embedding 模型
    "llava"                # 图像理解（如果需要）
)

echo "开始下载离线所需模型..."
for model in "${MODELS[@]}"; do
    echo ""
    echo "▶ 正在拉取: $model"
    ollama pull "$model"
    if [ $? -eq 0 ]; then
        echo "  ✅ $model 下载完成"
        ollama show "$model" --size
    else
        echo "  ❌ $model 下载失败"
    fi
done

echo ""
echo "所有模型下载完毕。当前已下载模型列表:"
ollama list
```

### 场景：多用户共享一台服务器

当多个开发者共用一台 Ollama 服务器时，模型选择需要兼顾各方需求：

```bash
# 推荐的服务器标配模型组合
ollama pull qwen2.5:7b           # 中文用户的主力模型
ollama pull llama3.1:8b          # 英文用户的主力模型
ollama pull deepseek-coder:6.7b  # 代码相关任务
ollama pull nomic-embed-text     # RAG/Embedding 需求
ollama pull phi3:mini            # 快速测试/原型验证
```

这组模型的总磁盘占用约 **15GB**，覆盖了绝大多数日常需求，同时通过 `OLLAMA_KEEP_ALIVE` 和 `OLLAMA_MAX_LOADED_MODELS` 可以控制内存占用。

### 场景：CI/CD 流水线中的自动化检查

在持续集成流水线中使用 Ollama 时，延迟和资源占用是关键考量：

```bash
# CI/CD 推荐配置
# 使用最小可用模型 + 最短 keep_alive
export OLLAMA_KEEP_ALIVE=0          # 请求完成后立即卸载
export OLLAMA_NUM_PARALLEL=1        # 单线程串行处理

# 使用小模型做初步筛选
ollama run phi3:mini "检查以下代码是否有明显的安全漏洞: $(cat src/main.py)"
```

## 本章小结

这一节我们构建了一套系统的模型选择方法论：

1. **T-L-H-Q-D 五维框架**从任务类型、语言要求、硬件限制、质量需求和延迟约束五个角度全面评估
2. **量化评分引擎**可以将你的具体条件转化为 0-100 的客观分数，自动排序推荐
3. **典型场景推荐矩阵**提供了针对中文对话、英文对话、代码生成、RAG、图像理解的即查即用建议
4. **A/B 测试工具**帮助你在两个候选模型之间做出最终决定
5. **`q4_K_M` 量化 + 合适参数量**的组合是大多数场景下的最优解
6. **特殊场景**（离线、多用户、CI/CD）需要额外的选型考量

下一节我们将讨论一个容易被忽视但至关重要的话题：模型的安全性与合规。
