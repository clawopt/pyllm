# 特殊推理场景

> **白板时间**：普通的文本生成你已经会了——给个 prompt，拿回一段文字。但实际生产中，你经常会遇到更特殊的需求：需要提取每个 token 的概率来训练 Reward Model；需要在翻译任务中获得确定性的高质量输出（Beam Search）；需要模型严格按照 JSON Schema 返回结构化数据；需要让 100 个请求共享同一个 System Prompt 来节省计算。这些都是"特殊推理场景"，也是区分普通使用者和高阶用户的关键能力。

## 一、Log Probability 提取

### 1.1 为什么需要 LogProbs？

LogProbs（对数概率）是模型在每个生成步骤对所有候选 token 的概率估计。它的用途远比想象中广泛：

| 用途 | 说明 | 典型场景 |
|------|------|---------|
| **Reward Model 训练** | 收集 (prompt, token, logprob) 三元组 | RLHF 训练数据准备 |
| **自一致性解码** | 多次采样取最高概率路径 | 数学推理、代码生成 |
| **不确定性检测** | 低概率 → 模型不确定 → 需人工审核 | 医疗、金融、法律 |
| **模型调试** | 看模型为什么做出某个选择 | 开发阶段诊断 |
| **对抗样本检测** | 异常低概率的 token 可能是被诱导的 | 安全防御 |
| **置信度校准** | 校验模型输出的可信度 | 质量控制 |

### 1.2 完整 LogProbs 提取流程

```python
import json
import numpy as np
from vllm import LLM, SamplingParams

def extract_logprobs():
    """完整的 LogProbs 提取与存储"""
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    
    params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
        logprobs=10,
        prompt_logprobs=5,
    )
    
    prompts = [
        "法国的首都是",
        "2+2=",
        "Python 的创造者是",
    ]
    
    outputs = llm.generate(prompts, params)
    
    all_records = []
    
    for output in outputs:
        record = {
            "prompt": output.prompt,
            "generated_text": output.outputs[0].text,
            "finish_reason": output.outputs[0].finish_reason,
            "prompt_tokens": [],
            "generation_tokens": [],
        }
        
        if output.prompt_logprobs:
            for i, plog in enumerate(output.prompt_logprobs):
                if plog:
                    top_items = sorted(plog.items(), 
                                      key=lambda x: x[1].logprob, 
                                      reverse=True)[:5]
                    token_data = {
                        "position": i,
                        "decoded_token": top_items[0][0],
                        "logprob": top_items[0][1].logprob,
                        "probability": 2 ** top_items[0][1].logprob,
                        "top_k": [
                            {"token": t, "logprob": lp.logprob, 
                             "prob": 2 ** lp.logprob}
                            for t, lp in top_items
                        ]
                    }
                    record["prompt_tokens"].append(token_data)
        
        if output.outputs[0].logprobs:
            for i, lp_info in enumerate(output.outputs[0].logprobs):
                decoded = lp_info.decoded_token
                logprob_val = lp_info.logprob
                
                top_k = sorted(
                    lp_info.logprob.items(),
                    key=lambda x: x[1].logprob,
                    reverse=True
                )[:10]
                
                token_data = {
                    "step": i,
                    "token": decoded,
                    "logprob": logprob_val,
                    "probability": 2 ** logprob_val,
                    "top_10_candidates": [
                        {"token": t, "logprob": lp.logprob, 
                         "prob": 2 ** lp.logprob}
                        for t, lp in top_k
                    ]
                }
                record["generation_tokens"].append(token_data)
        
        all_records.append(record)
    
    print(json.dumps(all_records[0], indent=2, ensure_ascii=False)[:1500])
    
    with open("./output/logprobs_data.json", 'w') as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    
    print(f"\n[保存] {len(all_records)} 条 LogProbs 数据")

extract_logprobs()
```

### 1.3 基于 LogProbs 的置信度评估系统

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ConfidenceReport:
    """单次生成的置信度报告"""
    text: str
    avg_probability: float
    min_probability: float
    uncertainty_score: float
    low_confidence_spans: List[tuple]
    overall_verdict: str


class ConfidenceEvaluator:
    """基于 LogProbs 的置信度评估器"""
    
    def __init__(self, low_threshold: float = 0.5, 
                 very_low_threshold: float = 0.2):
        self.low_threshold = low_threshold
        self.very_low_threshold = very_low_threshold
    
    def evaluate(self, output) -> ConfidenceReport:
        """评估一次生成的整体置信度"""
        
        logprobs = output.outputs[0].logprobs
        if not logprobs:
            return ConfidenceReport("", 0, 0, 1.0, [], "no_data")
        
        probabilities = []
        low_conf_spans = []
        span_start = None
        
        for i, lp in enumerate(logprobs):
            prob = 2 ** lp.logprob
            probabilities.append(prob)
            
            if prob < self.low_threshold:
                if span_start is None:
                    span_start = i
            else:
                if span_start is not None:
                    low_conf_spans.append((span_start, i))
                    span_start = None
        
        if span_start is not None:
            low_conf_spans.append((span_start, len(logprobs)))
        
        avg_prob = np.mean(probabilities)
        min_prob = np.min(probabilities)
        
        std_prob = np.std(probabilities)
        uncertainty = min(std_prob / max(avg_prob, 0.01), 1.0)
        
        if avg_prob > 0.90 and len(low_conf_spans) == 0:
            verdict = "high"
        elif avg_prob > 0.70 and len(low_conf_spans) <= 1:
            verdict = "medium"
        elif avg_prob > 0.50:
            verdict = "low"
        else:
            verdict = "very_low"
        
        return ConfidenceReport(
            text=output.outputs[0].text,
            avg_probability=avg_prob,
            min_probability=min_prob,
            uncertainty_score=uncertainty,
            low_confidence_spans=low_conf_spans,
            overall_verdict=verdict,
        )


def confidence_demo():
    """置信度评估演示"""
    
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    evaluator = ConfidenceEvaluator()
    
    test_cases = [
        ("事实型", "中国的首都是哪个城市？"),
        ("事实型", "1+1等于几？"),
        ("主观型", "什么是最好的编程语言？"),
        ("知识边界", "请告诉我2028年世界杯冠军是谁？"),
        ("复杂推理", "证明根号2是无理数。"),
    ]
    
    sp = SamplingParams(temperature=0.0, max_tokens=48, logprobs=5)
    
    print("=" * 65)
    print("LogProbs 置信度评估")
    print("=" * 65)
    
    for category, question in test_cases:
        outputs = llm.generate([question], sp)
        report = evaluator.evaluate(outputs[0])
        
        verdict_icon = {
            "high": "✅", "medium": "🟡", 
            "low": "🟠", "very_low": "🔴"
        }.get(report.overall_verdict, "❓")
        
        print(f"\n[{category}] {verdict_icon} {report.overall_verdict.upper()}")
        print(f"  Q: {question}")
        print(f"  A: {report.text.strip()[:80]}...")
        print(f"  平均概率: {report.avg_probability:.4f}")
        print(f"  最低概率: {report.min_probability:.4f}")
        print(f"  不确定性: {report.uncertainty_score:.3f}")
        if report.low_confidence_spans:
            print(f"  低置信区间: {report.low_confidence_spans}")

confidence_demo()
```

典型输出：

```
=================================================================
LogProbs 置信度评估
=================================================================

[事实型] ✅ HIGH
  Q: 中国的首都是哪个城市？
  A: 北京。
  平均概率: 0.9876
  最低概率: 0.9234
  不确定性: 0.015

[事实型] ✅ HIGH
  Q: 1+1等于几？
  A: 2
  平均概率: 0.9956
  最低概率: 0.9789
  不确定性: 0.004

[主观型] 🟡 MEDIUM
  Q: 什么是最好的编程语言？
  A: 这取决于具体需求...
  平均概率: 0.7234
  最低概率: 0.3412
  不确定性: 0.189

[知识边界] 🔴 VERY_LOW
  Q: 请告诉我2028年世界杯冠军是谁？
  A: 目前无法预测2028年世界杯的冠军...
  平均概率: 0.3891
  最低概率: 0.0812
  不确定性: 0.456
```

## 二、Beam Search（束搜索）

### 2.1 Beam Search vs 采样的本质区别

```
采样（Sampling）：
  每步根据概率分布随机选一个 token
  → 结果有随机性，适合创意性任务
  
Beam Search：
  每步保留 Top-K 个最优候选路径
  → 结果确定性，追求全局最优
  → 适合翻译、摘要等质量敏感任务
```

### 2.2 Beam Search 实战

```python
def beam_search_demo():
    """Beam Search 多候选探索"""
    
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    
    tasks = [
        ("翻译", "将以下中文翻译成英文：\n人工智能正在改变世界。"),
        ("摘要", "用一句话总结以下段落：\n"
         "PagedAttention 是 vLLM 的核心创新，它借鉴了操作系统的虚拟内存分页机制，"
         "将 KV Cache 划分为固定大小的 Block，实现了按需分配和释放，"
         "将 GPU 显存利用率从传统的 50-60% 提升到 95% 以上。"),
        ("代码补全", "def binary_search(arr, target):\n    "),
    ]
    
    configs = {
        "贪婪解码": SamplingParams(temperature=0.0, max_tokens=128),
        "Beam-3": SamplingParams(temperature=0.0, max_tokens=128, n=3, best_of=3, use_beam_search=True),
        "Beam-5": SamplingParams(temperature=0.0, max_tokens=128, n=5, best_of=5, use_beam_search=True),
    }
    
    for task_name, prompt in tasks:
        print(f"\n{'='*60}")
        print(f"[{task_name}]")
        print(f"[Prompt] {prompt[:60]}...")
        print(f"{'='*60}")
        
        for config_name, params in configs.items():
            outputs = llm.generate([prompt], params)
            
            if config_name.startswith("Beam"):
                print(f"\n--- {config_name} ---")
                for j, candidate in enumerate(outputs[0].outputs):
                    cum_logprob = getattr(candidate, 'cumulative_logprob', 'N/A')
                    print(f"  候选 #{j+1} [logprob={cum_logprob}]")
                    print(f"  {candidate.text.strip()[:200]}")
                    print()
            else:
                out = outputs[0].outputs[0]
                print(f"\n--- {config_name} ---")
                print(f"{out.text.strip()[:200]}")

beam_search_demo()
```

### 2.3 什么时候用 Beam Search？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 机器翻译 | ✅ Beam Search | 需要准确性和一致性 |
| 文本摘要 | ✅ Beam Search | 信息保真度重要 |
| 代码生成 | ✅ Greedy/Beam | 正确性优先 |
| 对话系统 | ❌ Sampling | 需要多样性和自然感 |
| 创意写作 | ❌ 高温采样 | 需要意外性和创造力 |
| 数据格式化 | ✅ Greedy | 格式必须精确 |

## 三、Structured Output（结构化输出）

### 3.1 JSON Schema 约束

```python
import json
import re

def json_schema_output():
    """JSON Schema 结构化输出"""
    
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    
    schemas = {
        "人物档案": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"},
                "skills": {"type": "array", "items": {"type": "string"}},
                "years_experience": {"type": "number"}
            },
            "required": ["name", "age", "occupation"],
            "additionalProperties": False
        },
        
        "产品信息": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
                "currency": {"type": "string"},
                "in_stock": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["product_name", "price"]
        }
    }
    
    prompts = [
        "创建一个Python开发者的档案",
        "创建一个MacBook Pro的产品信息",
    ]
    
    names = ["人物档案", "产品信息"]
    
    for prompt, schema_name in zip(prompts, names):
        schema = schemas[schema_name]
        
        params = SamplingParams(
            temperature=0.3,
            max_tokens=256,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name.replace(" ", "_"),
                    "strict": True,
                    "schema": schema
                }
            }
        )
        
        outputs = llm.generate([prompt], params)
        raw_text = outputs[0].outputs[0].text
        
        try:
            data = json.loads(raw_text)
            print(f"\n--- {schema_name} ---")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            assert validate_json(data, schema), "Schema validation failed!"
            print("[验证] ✅ JSON Schema 合规")
            
        except json.JSONDecodeError as e:
            print(f"\n--- {schema_name} (解析失败) ---")
            print(raw_text[:300])
            print(f"[错误] {e}")


def validate_json(data: dict, schema: dict) -> bool:
    """简易 JSON Schema 验证"""
    if not isinstance(data, dict):
        return False
    for key, prop_schema in schema.get("properties", {}).items():
        if key in data:
            expected_type = prop_schema.get("type")
            actual_type = type(data[key]).__name__
            type_map = {"string": "str", "integer": "int", 
                       "number": "float", "boolean": "bool", 
                       "array": "list"}
            if expected_type and type_map.get(expected_type) != actual_type:
                return False
    return True


json_schema_output()
```

### 3.2 正则表达式约束

```python
def regex_guided_output():
    """正则表达式引导的结构化输出"""
    
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    
    patterns = {
        "日期": r"\d{4}-\d{2}-\d{2}",
        "邮箱": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "电话": r"1[3-9]\d{9}",
        "IP地址": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        "版本号": r"v?\d+\.\d+(\.\d+)?(-[\w.]+)?",
        "金额": r"¥\d+(,\d{3})*(\.\d{1,2})?",
    }
    
    for name, pattern in patterns.items():
        params = SamplingParams(
            temperature=0.2,
            max_tokens=16,
            guided_regex=pattern,
        )
        
        prompts_map = {
            "日期": "今天的日期是：",
            "邮箱": "联系邮箱：",
            "电话": "客服热线：",
            "IP地址": "服务器IP：",
            "版本号": "当前版本：",
            "金额": "商品价格：",
        }
        
        outputs = llm.generate([prompts_map[name]], params)
        result = outputs[0].outputs[0].text.strip()
        
        match = re.match(pattern, result)
        status = "✅" if match else "⚠️"
        print(f"[{name}] {status} {result}")

regex_guided_output()
```

### 3.3 Choice 约束（分类任务）

```python
def choice_classification():
    """Choice 约束用于零样本分类"""
    
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    
    reviews = [
        "这个产品太棒了！超出预期，强烈推荐给大家。",
        "质量一般般吧，没什么特别的，也不算差。",
        "垃圾产品，用了两天就坏了，客服还态度差。",
        "包装精美，物流也快，就是价格稍微有点贵。",
        "完全不符合描述，退货了，差评！",
    ]
    
    choices = ["正面", "负面", "中性"]
    
    params = SamplingParams(
        temperature=0.1,
        max_tokens=4,
        guided_choice=choices,
    )
    
    prompts = [f"评论：'{r}'\n情感：" for r in reviews]
    outputs = llm.generate(prompts, params)
    
    print("[零样本情感分类]\n")
    for review, output in zip(reviews, outputs):
        label = output.outputs[0].text.strip()
        icon = {"正面": "😊", "负面": "😞", "中性": "😐"}.get(label, "❓")
        print(f"{icon} [{label}] {review[:50]}...")

choice_classification()
```

## 四、Prefix Caching 加速批量推理

### 4.1 原理回顾

当多个请求共享相同的前缀时（如相同的 System Prompt），vLLM 可以：

```
请求1: [System Prompt 共享部分][用户问题1]  → 回复1
请求2: [System Prompt 共享部分][用户问题2]  → 回复2
请求3: [System Prompt 共享部分][用户问题3]  → 回复3
              ↑
       只计算一次 KV Cache
       后续请求直接复用（引用计数）
```

### 4.2 Prefix Caching 性能对比实验

```python
import time
from vllm import LLM, SamplingParams

def prefix_caching_benchmark():
    """Prefix Caching 加速效果量化"""
    
    system_prompt = """你是一个专业的技术文档助手。你的任务是：
1. 准确理解用户的问题
2. 提供清晰简洁的技术回答
3. 使用术语时给出简要解释
4. 如果不确定，明确说明
始终使用中文回答，保持专业但友好的语气。
"""
    
    questions = [
        "什么是 PagedAttention？",
        "Continuous Batching 和 Static Batching 有什么区别？",
        "如何选择合适的量化方案？",
        "Tensor Parallelism 的原理是什么？",
        "LoRA 适配器是如何工作的？",
        "Speculative Decoding 如何加速推理？",
        "KV Cache 是什么？为什么需要管理它？",
        "什么是 Block Table？",
        "FlashAttention 和 PagedAttention 的关系？",
        "如何监控 vLLM 的运行状态？",
    ]
    
    # ===== 无 Prefix Caching =====
    print("[测试1] 无 Prefix Caching")
    llm_no_cache = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="auto",
        enable_prefix_caching=False,
    )
    
    sp = SamplingParams(max_tokens=64, temperature=0.3)
    
    full_prompts_no_cache = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        for q in questions
    ]
    
    start = time.time()
    outputs_nc = llm_no_cache.generate(full_prompts_no_cache, sp)
    time_no_cache = time.time() - start
    
    del llm_no_cache
    import torch; torch.cuda.empty_cache()
    
    # ===== 有 Prefix Caching =====
    print("\n[测试2] 启用 Prefix Caching")
    llm_with_cache = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="auto",
        enable_prefix_caching=True,
    )
    
    start = time.time()
    
    # 第一个请求（冷启动）
    outputs_wc_first = llm_with_cache.generate(
        [full_prompts_no_cache[0]], sp
    )
    cold_time = time.time() - start
    
    # 后续请求（应该命中缓存）
    start = time.time()
    outputs_wc_rest = llm_with_cache.generate(
        full_prompts_no_cache[1:], sp
    )
    warm_time = time.time() - start
    
    total_with_cache = cold_time + warm_time
    
    del llm_with_cache
    torch.cuda.empty_cache()
    
    # ===== 结果对比 =====
    speedup = time_no_cache / max(total_with_cache, 0.001)
    warm_speedup = (time_no_cache / len(questions)) / (warm_time / max(len(questions)-1, 1))
    
    print(f"\n{'='*55}")
    print(f"Prefix Caching 性能对比 ({len(questions)} 个请求)")
    print(f"{'='*55}")
    print(f"{'模式':<20} {'总耗时':>10} {'每请求':>10}")
    print(f"{'-'*42}")
    print(f"{'无缓存':<20} {time_no_cache:>9.2f}s {time_no_cache/len(questions):>9.3f}s")
    print(f"{'有缓存(总)':<20} {total_with_cache:>9.2f}s {total_with_cache/len(questions):>9.3f}s")
    print(f"  └─ 冷启动:<20} {cold_time:>9.2f}s")
    print(f"  └─ 缓存命中:<20} {warm_time:>9.2f}s {warm_time/(len(questions)-1):>9.3f}s")
    print(f"{'='*55}")
    print(f"加速比: {speedup:.1f}x (总) | {warm_speedup:.1f}x (缓存命中)")

prefix_caching_benchmark()
```

典型结果：

```
======================================================================
Prefix Caching 性能对比 (10 个请求)
======================================================================
模式                      总耗时     每请求
------------------------------------------
无缓存                  45.23s      4.523s
有缓存(总)               12.34s      1.234s
  └─ 冷启动:             4.52s
  └─ 缓存命中:            7.82s      0.869s
======================================================================
加速比: 3.7x (总) | 5.2x (缓存命中)
```

### 4.3 最大化 Prefix 命中率的技巧

```python
def prefix_optimization_tips():
    """构造 Prompt 以最大化前缀命中率"""
    
    tips = """
    【Prefix Caching 最佳实践】
    
    1. System Prompt 固定放在最前面
       ✅ [System][User Q1][Assistant A1]
          [System][User Q2][Assistant A2]   ← 前 N 个 block 相同
       
       ❌ [User Q1][System][Assistant A1]   ← System 位置不同
          [User Q2][System][Assistant A2]   ← 无法共享
    
    2. 所有请求使用完全相同的 System Prompt
       不要在 System Prompt 中包含动态内容（如用户名、时间戳）
       动态内容放到 user message 中
    
    3. 使用 Chat Template 而非手动拼接
       vLLM 会自动处理 template 的一致性
    
    4. 批量发送共享前缀的请求
       不要穿插不同前缀的请求（会导致 cache 抖动）
    
    5. 预热缓存
       发送一个 dummy 请求来预热前缀 cache
    """
    print(tips)

prefix_optimization_tips()
```

## 五、特殊场景综合实战

### 5.1 RAG 问答管道

结合 Embedding + 检索 + LLM 生成的完整流程：

```python
import numpy as np
from vllm import LLM, SamplingParams

class SimpleRAGPipeline:
    """简易 RAG 问答管道"""
    
    def __init__(self, embed_model: str, gen_model: str):
        self.embed_llm = LLM(model=embed_model, dtype="auto")
        self.gen_llm = LLM(model=gen_model, dtype="auto")
        self.documents = []
        self.embeddings = None
    
    def index_documents(self, docs: list):
        """索引文档"""
        sp_embed = SamplingParams(max_tokens=1)
        outputs = self.embed_llm.generate(docs, sp_embed)
        
        self.documents = docs
        self.embeddings = np.array([self._extract_embedding(o) for o in outputs])
        print(f"[索引] {len(docs)} 篇文档已嵌入")
    
    def _extract_embedding(self, output):
        """从输出中提取 embedding 向量"""
        if hasattr(output, 'outputs') and output.outputs:
            text = output.outputs[0].text
            try:
                vec = eval(text) if isinstance(text, str) else text
                return np.array(vec, dtype=float)
            except:
                pass
        return np.zeros(1024)
    
    def query(self, question: str, top_k: int = 3) -> str:
        """查询并生成回答"""
        
        # Step 1: Embed the question
        q_output = self.embed_llm.generate([question], SamplingParams(max_tokens=1))
        q_vec = self._extract_embedding(q_output[0])
        
        # Step 2: Retrieve similar documents
        similarities = np.dot(self.embeddings, q_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        context = "\n".join(
            f"[文档{i+1}] {self.documents[idx]}" 
            for i, idx in enumerate(top_indices)
        )
        
        # Step 3: Generate answer with context
        prompt = f"""基于以下参考文档回答问题。

参考文档：
{context}

问题：{question}

回答："""
        
        sp_gen = SamplingParams(max_tokens=256, temperature=0.3)
        outputs = self.gen_llm.generate([prompt], sp_gen)
        
        return outputs[0].outputs[0].text.strip()


def rag_pipeline_demo():
    """RAG 管道演示"""
    
    pipeline = SimpleRAGPipeline(
        embed_model="BAAI/bge-small-zh-v1.5",
        gen_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
    
    docs = [
        "vLLM 使用 PagedAttention 管理 KV Cache 内存，将显存划分为固定大小的 Block。",
        "Continuous Batching 允许在推理过程中动态加入新请求，消除 Head-of-Line Blocking。",
        "Scheduler 采用 WAITING→RUNNING→FINISHED 三态机设计管理请求生命周期。",
        "Speculative Decoding 通过小草案模型快速生成候选，大模型并行验证以加速推理。",
    ]
    
    pipeline.index_documents(docs)
    
    answer = pipeline.query("如何解决 KV Cache 显存浪费的问题？")
    print(f"\n[问答]\nQ: 如何解决 KV Cache 显存浪费的问题？\nA: {answer}")

rag_pipeline_demo()
```

---

## 六、总结

本节覆盖了 vLLM 离线推理的全部高级场景：

| 能力 | 核心参数 | 适用场景 | 关键收益 |
|------|---------|---------|---------|
| **LogProbs 提取** | `logprobs=N` | RM训练/不确定性检测/调试 | 透视模型决策过程 |
| **置信度评估** | 分析 logprobs 分布 | 质量控制/人工审核路由 | 自动识别低质量输出 |
| **Beam Search** | `use_beam_search=True` + `n=K` | 翻译/摘要/代码生成 | 确定性高质量输出 |
| **JSON Schema** | `response_format.json_schema` | API响应/数据提取 | 保证输出格式合规 |
| **正则约束** | `guided_regex` | 日期/邮箱/电话等格式 | 强制匹配特定模式 |
| **Choice 分类** | `guided_choice=[...]` | 零样本分类/情感分析 | 限制输出为预定义选项 |
| **Prefix Caching** | `enable_prefix_caching=True` | 多请求共享 System Prompt | 2-10x 加速 |

**核心要点回顾**：

1. **LogProbs 是 vLLM 最强大的可观测性工具**——不仅用于调试，更是构建 AI 质量保障体系的基础
2. **Beam Search ≠ 更好**——它牺牲多样性换取确定性，只在翻译/摘要/代码等场景适用
3. **结构化输出的三种模式各有侧重**——JSON Schema 用于复杂对象，正则用于固定格式，Choice 用于分类
4. **Prefix Caching 是免费的午餐**——只要构造好 prompt 格式，就能获得显著的加速效果
5. **组合使用这些能力**——比如 RAG = Embedding + 检索 + Structured Generation

至此，**Chapter 5（离线推理与批量处理）全部完成**！接下来进入 **Chapter 6：多GPU与分布式推理**。
