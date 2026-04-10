# 离线推理与批量处理

> **白板时间**：到目前为止，我们一直在用 HTTP API 的方式调用 vLLM——启动一个 API Server，然后通过 OpenAI SDK 发送请求。但 vLLM 还有另一种完全不同的使用方式：**直接在你的 Python 代码中调用 `LLM` 类，像调用一个普通函数一样生成文本**。这种方式叫 **Offline Inference（离线推理）**，它不需要启动 HTTP 服务，没有网络开销，适合批量处理、实验研究、数据处理管道等场景。今天我们就来掌握这种更底层的使用方式。

## 一、为什么需要离线推理？

### 1.1 两种模式对比

| 维度 | API Server 模式 | Offline Inference 模式 |
|------|----------------|---------------------|
| 启动方式 | `python -m vllm.entrypoints.openai.api_server` | 直接在 Python 中 import |
| 调用方式 | HTTP / SDK | `LLM.generate()` 方法 |
| 网络开销 | 有（HTTP 往返） | 无（进程内调用） |
| 适用场景 | 在线服务、多客户端 | 批量处理、实验、数据管道 |
| 并发能力 | 天然支持（多客户端） | 需自行管理 |
| 部署复杂度 | 高（独立进程） | 低（嵌入应用） |
| 延迟 | +2-10ms (HTTP) | 最小化 |

### 1.2 适用场景判断

```
你的需求是什么？
│
├─ 需要为多个外部客户端提供服务
│   └─→ API Server 模式 ✅
│
├─ 需要在 Jupyter/脚本中做实验
│   └─→ Offline Inference ✅
│
├─ 需要批量处理大量文本（如数据增强、翻译）
│   └─→ Offline Inference ✅（无网络开销）
│
├─ 需要嵌入到更大的 Python 应用中
│   └─→ Offline Inference ✅
│
└─ 需要与 FastAPI/Flask 等 Web 框架集成
    ├─ 自己包装一层 → Offline + Web 框架
    └─ 直接用 → API Server 模式
```

## 二、基础用法：LLM 类入门

### 2.1 第一个离线推理程序

比如下面的程序展示了最基础的离线推理用法：

```python
from vllm import LLM, SamplingParams

def hello_offline():
    """第一个离线推理程序"""
    
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="auto",
        gpu_memory_utilization=0.9,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=128,
    )
    
    prompts = [
        "人工智能的定义是什么？请用一句话回答。",
        "列出 Python 的三个主要特性。",
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[输入] {prompt}")
        print(f"[输出] {generated_text}")
        print(f"[完成原因] {output.outputs[0].finish_reason}")
        print()

hello_offline()
```

输出：

```
[输入] 人工智能的定义是什么？请用一句话回答。
[输出] 人工智能(AI)是计算机科学的一个分支，旨在创建能够模拟人类智能（如学习、推理、感知、理解）的技术和系统。
[完成原因] stop

[输入] 列出 Python 的三个主要特性。
[输出] 1. 简洁优雅的语法，易于学习和阅读；2. 动态类型系统，开发效率高；
       3. 丰富的标准库和第三方生态系统。
[完成原因] stop
```

### 2.2 LLM 构造参数详解

`LLM` 类的核心构造参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | 必填 | 模型名称或路径 |
| `dtype` | str | "auto" | 数据类型 (half, bfloat16, float16, auto) |
| `tensor_parallel_size` | int | 1 | 张量并行 GPU 数量 |
| `gpu_memory_utilization` | float | 0.9 | GPU 显存使用率上限 |
| `max_model_len` | int | None | 最大序列长度（None=模型默认） |
| `block_size` | int | 16 | PagedAttention Block 大小 |
| `swap_space` | int | 4 | CPU swap 空间 (GiB) |
| `enforce_eager` | bool | False | 强制使用 eager mode（禁用 CUDA graph） |
| `seed` | int | 0 | 随机种子 |
| `trust_remote_code` | bool | False | 是否信任远程代码 |
| `download_dir` | str | None | 模型下载目录 |
| `load_format` | str | "auto" | 权重加载格式 (auto, safetensors, pt) |
| `enable_prefix_caching` | bool | False | 启用前缀缓存 |
| `quantization` | str | None | 量化方案 (awq, gptq, fp8) |

```python
def advanced_llm_init():
    """高级 LLM 初始化配置"""
    
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        
        # 显存管理
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        
        # 性能优化
        enable_prefix_caching=True,
        enforce_eager=False,
        
        # 可复现性
        seed=42,
        
        # 安全
        trust_remote_code=False,
    )
    
    return llm
```

## 三、SamplingParams：控制生成行为

### 3.1 完整参数参考

`SamplingParams` 是控制模型生成行为的核心类——它决定了模型"怎么说话"而不是"说什么内容"。

```python
from vllm import SamplingParams

params = SamplingParams(
    # === 基础控制 ===
    n=1,                    # 为每个 prompt 生成几个候选
    best_of=1,              # 从几个候选中选最好的（best_of >= n）
    presence_penalty=0.0,   # 存在惩罚 (-2.0 ~ 2.0)
    frequency_penalty=0.0,  # 频率惩罚 (-2.0 ~ 2.0)
    repetition_penalty=1.0, # 重复惩罚 (>1.0 抑制重复)
    temperature=1.0,        # 温度 (0.0 = 贪婪, >0 = 随机采样)
    top_p=1.0,              # 核采样阈值 (0.0 ~ 1.0)
    top_k=-1,               # Top-K 采样 (-1 = 不限制)
    min_p=0.0,              # 最小概率核采样 (v0.6+)
    
    # === 长度控制 ===
    max_tokens=1024,        # 最大生成 token 数
    min_tokens=0,           # 最小生成 token 数
    
    # === 停止条件 ===
    stop=None,              # 停止序列 (str, list[str], 或 list[AllowedTokens])
    stop_token_ids=None,    # 停止 token ID 列表
    include_stop_str_in_output=False,  # 输出中是否包含停止字符串
    
    # === 特殊模式 ===
    skip_special_tokens=True,          # 输出中是否移除特殊 token
    spaces_between_special_tokens=True, # 特殊 token 间是否加空格
    
    # === LogProbs ===
    logprobs=None,         # 返回 top-k logprobs (None = 不返回)
    prompt_logprobs=None,  # 返回 prompt 的 logprobs (None = 不返回)
    detokenize=True,       # 是否将 token ID 反转为文本
    
    # === 结构化输出 ===
    guided_decoding_backend=None,  # 结构化解码后端
    response_format=None,          # JSON Schema 约束输出格式
    guided_json=None,              # JSON Schema 对象
    guided_regex=None,             # 正则表达式约束
    guided_choice=None,            # 选项列表约束
    guided_grammar=None,           # 形式文法约束
    guided_whitelist=None,         # 白名单约束
    guided_grammar_fn=None,        # 自定义文法函数
    
    # === 其他 ===
    seed=None,              # 随机种子
)
```

### 3.2 不同采样策略的效果对比

比如下面的程序展示了不同采样参数对生成结果的影响：

```python
def sampling_comparison():
    """不同采样策略的效果对比"""
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="auto")
    
    prompt = "续写这句话：深夜里，程序员终于找到了那个 Bug。"
    
    strategies = {
        "贪婪解码": SamplingParams(temperature=0.0, max_tokens=128),
        "低温保守": SamplingParams(temperature=0.3, top_p=0.9, max_tokens=128),
        "平衡模式": SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128),
        "高温创意": SamplingParams(temperature=1.2, top_p=0.98, max_tokens=128),
        "TopK=5": SamplingParams(temperature=0.8, top_k=5, max_tokens=128),
        "重复抑制": SamplingParams(temperature=0.7, repetition_penalty=1.15, max_tokens=128),
    }
    
    for name, params in strategies.items():
        outputs = llm.generate([prompt], params)
        text = outputs[0].outputs[0].text.strip()
        print(f"\n--- {name} ---")
        print(text[:200])

sampling_comparison()
```

典型输出差异：

```
--- 贪婪解码 ---
他仔细检查了代码逻辑，发现是一个空指针异常导致的崩溃。修复后重新运行测试，
所有测试用例都通过了。这个 Bug 是由于在多线程环境下对共享资源的访问没有正确同步导致的。

--- 低温保守 ---
原来是变量名拼写错误！把 `user_name` 写成了 `usr_nam`，导致整个用户认证模块都无法正常工作。
经过修复后，系统恢复了正常运行。

--- 平衡模式 ---
屏幕上闪烁的绿色光标仿佛在嘲笑他的执着。当他追踪到第 342 行时，一切豁然开朗——
那是一个竞态条件，只在特定并发模式下才会触发。他在代码旁留下了一行注释：
"愿后来者不再在此处迷失。"

--- 高温创意 ---
那一刻，时间仿佛凝固了。Bug 不是 Bug，而是一扇门——门后是另一个维度。
每行代码都在呼吸，每个变量都是活的。他意识到自己不是在修 Bug，而是在与一个
数字生命对话...

--- TopK=5 ---
问题出在一个边界条件上。当输入为空数组时，索引越界导致了未定义行为。
添加了防御性检查后，问题解决。

--- 重复抑制 ---
原来是一个经典的 off-by-one 错误！循环条件少等了一次迭代，导致最后一个元素被遗漏。
修改后，功能恢复正常。这类错误虽然简单，但在复杂的代码库中往往最难发现。
```

**关键观察**：
- **temperature=0**（贪婪）：结果确定、稳定、但可能单调
- **temperature=0.7**：平衡了创造性和连贯性，适合大多数场景
- **temperature>1.0**：高度随机，适合创意写作但可能不连贯
- **repetition_penalty>1.0**：有效减少重复内容

## 四、LogProbs 与详细输出分析

### 4.1 获取 Token 级别的 LogProbs

离线推理中获取 LogProbs 比 API 更方便——不需要额外解析 SSE 流：

```python
def offline_logprobs_demo():
    """离线推理 LogProbs 分析"""
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="auto")
    
    params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        logprobs=5,
        prompt_logprobs=5,
    )
    
    prompt = "中国的首都是"
    outputs = llm.generate([prompt], params)
    
    output = outputs[0]
    
    print("=" * 70)
    print("LogProbs 详细分析")
    print("=" * 70)
    print(f"\n[Prompt] {prompt}")
    print(f"[完整输出] {output.outputs[0].text}\n")
    
    if output.prompt_logprobs:
        print("[Prompt Tokens]")
        for i, plog in enumerate(output.prompt_logprobs):
            if plog is None:
                continue
            top_token = list(plog.keys())[0]
            logprob = plog[top_token].logprob
            prob = 2 ** logprob
            print(f"  {i}: '{top_token}' (logprob={logprob:.4f}, prob={prob:.4%})")
    
    print("\n[Generated Tokens]")
    for out in output.outputs:
        for i, logprob_info in enumerate(out.logprobs):
            top_token = logprob_info.decoded_token
            top_logprob = logprob_info.logprob
            top_prob = 2 ** top_logprob
            
            print(f"\n  Step {i}: '{top_token}' (prob={top_prob:.4%})")
            print(f"  Top-5 candidates:")
            
            sorted_candidates = sorted(
                logprob_info.logprob.values(), 
                key=lambda x: x.logprob, 
                reverse=True
            )
            for rank, lp in enumerate(sorted_candidates):
                marker = " ← selected" if lp.decoded_token == top_token else ""
                c_prob = 2 ** lp.logprob
                print(f"    {rank+1}. '{lp.decoded_token}' "
                      f"(logprob={lp.logprob:.4f}, prob={c_prob:.4%}){marker}")

offline_logprobs_demo()
```

### 4.2 置信度评估

```python
def confidence_analysis():
    """基于 LogProbs 的置信度评估"""
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="auto")
    
    questions = [
        ("事实型", "1+1等于几？"),
        ("事实型", "Python的发明者是谁？"),
        ("主观型", "什么是最好的编程语言？"),
        ("幻觉测试", "请告诉我2025年世界杯的冠军是谁？"),
    ]
    
    params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
        logprobs=3,
    )
    
    prompts = [f"Q: {q}\nA:" for _, q in questions]
    outputs = llm.generate(prompts, params)
    
    print("[置信度评估]\n")
    
    for (category, question), output in zip(questions, outputs):
        out = output.outputs[0]
        
        if out.logprobs:
            first_lp = out.logprobs[0]
            first_token = first_lp.decoded_token
            first_prob = 2 ** first_lp.logprob
            
            confidence = "高" if first_prob > 0.95 else "中" if first_prob > 0.80 else "低"
            
            print(f"[{category}] Q: {question}")
            print(f"  首Token: '{first_token}' (概率: {first_prob:.4%}, 置信度: {confidence})")
            print(f"  回答: {out.text.strip()}\n")

confidence_analysis()
```

## 五、结构化输出

### 5.1 JSON Mode

vLLM 离线推理同样支持 JSON Schema 约束输出：

```python
import json

def structured_output_demo():
    """结构化输出演示"""
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="auto")
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            },
            "years_experience": {"type": "number"}
        },
        "required": ["name", "age", "skills"],
        "additionalProperties": False
    }
    
    params = SamplingParams(
        temperature=0.3,
        max_tokens=256,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "developer_profile",
                "strict": True,
                "schema": schema
            }
        }
    )
    
    prompts = [
        "创建一个Python后端开发者的档案",
        "创建一个机器学习工程师的档案",
        "创建一个DevOps工程师的档案",
    ]
    
    outputs = llm.generate(prompts, params)
    
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        try:
            data = json.loads(text)
            print(f"\n--- 档案 {i+1} ---")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(f"\n--- 档案 {i+1} (原始) ---")
            print(text)

structured_output_demo()
```

### 5.2 正则表达式约束

```python
def regex_guided_demo():
    """正则表达式引导的生成"""
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="auto")
    
    params = SamplingParams(
        temperature=0.3,
        max_tokens=64,
        guided_regex=r"\d{4}-\d{2}-\d{2}",
    )
    
    prompts = ["今天的日期是：", "项目截止日期："]
    outputs = llm.generate(prompts, params)
    
    for output in outputs:
        text = output.outputs[0].text.strip()
        print(f"输出: {text}")

regex_guided_demo()
```

### 5.3 Choice 约束（选项选择）

```python
def choice_guided_demo():
    """从预定义选项中选择"""
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="auto")
    
    params = SamplingParams(
        temperature=0.1,
        max_tokens=4,
        guided_choice=["正面", "负面", "中性"],
    )
    
    reviews = [
        "这个产品太棒了，强烈推荐！",
        "质量很差，退货了。",
        "一般般吧，没什么特别的。",
    ]
    
    prompts = [f"评论：'{review}'\n情感：" for review in reviews]
    outputs = llm.generate(prompts, params)
    
    for review, output in zip(reviews, outputs):
        sentiment = output.outputs[0].text.strip()
        print(f"'{review[:30]}...' → [{sentiment}]")

choice_guided_demo()
```

## 六、批量处理实战

### 6.1 大规模文本处理

```python
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    """vLLM 批量处理器"""
    
    def __init__(self, model: str, **llm_kwargs):
        self.llm = LLM(model=model, **llm_kwargs)
        self.stats = {"total": 0, "success": 0, "failed": 0, "total_time": 0}
    
    def process_batch(self, prompts: list, sampling_params: SamplingParams,
                     batch_size: int = 256) -> list:
        """分批处理大量 prompt"""
        all_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            start = time.time()
            
            try:
                outputs = self.llm.generate(batch, sampling_params)
                elapsed = time.time() - start
                
                for output in outputs:
                    result = {
                        "prompt": output.prompt,
                        "text": output.outputs[0].text,
                        "finish_reason": output.outputs[0].finish_reason,
                        "latency_s": elapsed / len(batch),
                    }
                    all_results.append(result)
                
                self.stats["success"] += len(batch)
                n_tokens = sum(len(o.outputs[0].text.split()) for o in outputs)
                throughput = len(batch) / elapsed
                
                print(f"[Batch {i//batch_size + 1}] "
                      f"{len(batch)} 条, "
                      f"{elapsed:.2f}s, "
                      f"{throughput:.0f} req/s")
                
            except Exception as e:
                self.stats["failed"] += len(batch)
                print(f"[Batch Error] {e}")
            
            self.stats["total"] += len(batch)
            self.stats["total_time"] += elapsed
        
        return all_results
    
    def get_stats(self) -> dict:
        return {
            **self.stats,
            "avg_latency": self.stats["total_time"] / max(self.stats["success"], 1),
            "overall_throughput": self.stats["total"] / max(self.stats["total_time"], 1),
        }


def batch_processing_demo():
    """大规模批量处理演示"""
    
    processor = BatchProcessor(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="auto",
        enable_prefix_caching=True,
    )
    
    params = SamplingParams(
        temperature=0.3,
        max_tokens=64,
    )
    
    prompts = [f"将以下句子翻译成英文：这是第{i}条测试数据。" 
               for i in range(500)]
    
    start = time.time()
    results = processor.process_batch(prompts, params, batch_size=64)
    total_elapsed = time.time() - start
    
    stats = processor.get_stats()
    
    print(f"\n{'='*50}")
    print(f"[总计] {stats['total']} 条, 总耗时 {total_elapsed:.2f}s")
    print(f"[吞吐] {stats['overall_throughput']:.0f} samples/s")
    print(f"[成功] {stats['success']}, 失败: {stats['failed']}")
    
    if results:
        print(f"\n[示例输出]")
        print(f"  Input:  {results[0]['prompt']}")
        print(f"  Output: {results[0]['text']}")

batch_processing_demo()
```

### 6.2 异步批处理

```python
import asyncio

async def async_batch_process():
    """异步批量处理"""
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="auto")
    params = SamplingParams(temperature=0.3, max_tokens=32)
    
    tasks = []
    prompts = [f"总结关键词：第{i}条数据的内容" for i in range(100)]
    
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(prompts), 10):
            batch = prompts[i:i+10]
            task = loop.run_in_executor(
                executor, 
                lambda b=batch: llm.generate(b, params)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_outputs = sum(len(r) for r in results)
        print(f"[异步批量] 完成 {total_outputs} 条处理")

asyncio.run(async_batch_process())
```

## 七、Prefix Caching 加速

### 7.1 共享 System Prompt 场景

当多个请求共享相同的 System Prompt 时，Prefix Caching 可以大幅加速：

```python
def prefix_caching_demo():
    """Prefix Caching 加速效果演示"""
    
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="auto",
        enable_prefix_caching=True,
    )
    
    system_prompt = """你是一个专业的技术文档助手。你的任务是：
1. 准确理解用户的问题
2. 提供清晰、简洁的回答
3. 使用技术术语时要给出解释
4. 如果不确定，明确说明
始终使用中文回答。"""
    
    user_questions = [
        "什么是 PagedAttention？",
        "Continuous Batching 和 Static Batching 有什么区别？",
        "如何选择合适的量化方案？",
        "Tensor Parallelism 的原理是什么？",
        "LoRA 适配器是如何工作的？",
    ]
    
    full_prompts = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        for q in user_questions
    ]
    
    params = SamplingParams(temperature=0.3, max_tokens=128)
    
    import time
    
    print("[第一次请求 - 冷启动]")
    start = time.time()
    outputs1 = llm.generate(full_prompts[:1], params)
    cold_time = time.time() - start
    print(f"  耗时: {cold_time:.2f}s\n")
    
    print("[后续请求 - Prefix Cache 命中]")
    start = time.time()
    outputs2 = llm.generate(full_prompts[1:], params)
    warm_time = time.time() - start
    print(f"  耗时: {warm_time:.2f}s")
    print(f"  加速比: {cold_time/max(warm_time/len(user_questions[1:]), 0.001):.1f}x\n")
    
    for q, output in zip(user_questions[1:], outputs2):
        print(f"Q: {q}")
        print(f"A: {output.outputs[0].text.strip()[:150]}...")
        print()

prefix_caching_demo()
```

## 八、Beam Search：多候选探索

Beam Search 在需要高质量输出的场景（如翻译、摘要）中非常有用：

```python
def beam_search_demo():
    """Beam Search 多候选搜索"""
    
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="auto")
    
    params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        n=3,
        best_of=3,
        use_beam_search=True,
    )
    
    prompt = "将以下英文翻译成中文：\nThe quick brown fox jumps over the lazy dog."
    
    outputs = llm.generate([prompt], params)
    
    print("[Beam Search 结果 - Top 3 候选]\n")
    for i, candidate in enumerate(outputs[0].outputs):
        score = cumulative_logprob = getattr(candidate, 'cumulative_logprob', None)
        print(f"--- 候选 {i+1} ---")
        print(candidate.text.strip())
        if score is not None:
            print(f"[LogProb] {score:.2f}")
        print()

beam_search_demo()
```

---

## 九、总结

本节我们全面学习了 vLLM 的离线推理能力：

| 能力 | 核心类/方法 | 关键点 |
|------|------------|--------|
| **基本推理** | `LLM.generate()` | 无 HTTP 开销，直接进程内调用 |
| **采样控制** | `SamplingParams` | temperature/top_p/repetition_penalty 等 20+ 参数 |
| **LogProbs 分析** | `logprobs=N` | 透视模型决策过程，用于调试和置信度评估 |
| **结构化输出** | `response_format` | JSON Schema / 正则 / Choice 约束 |
| **批量处理** | 分批 generate() | 支持 10000+ 规模的数据处理 |
| **Prefix Caching** | `enable_prefix_caching=True` | 共享 System Prompt 时加速 2-10x |
| **Beam Search** | `use_beam_search=True` | 高质量翻译/摘要场景的多候选探索 |

**核心要点回顾**：

1. **Offline Inference vs API Server**：前者零网络开销、嵌入应用，后者服务多客户端——根据场景选择
2. **SamplingParams 是控制生成行为的方向盘**：temperature 决定创造性，top_p 控制采样范围，repetition_penalty 抑制重复
3. **LogProbs 让你看到模型的"思考过程"**：每个 token 的概率分布都可见，可用于质量监控和调试
4. **结构化输出确保格式合规**：JSON Schema/正则/Choice 三种约束模式覆盖大部分场景
5. **Prefix Caching 是批量处理的加速利器**：共享 system prompt 时效果显著
6. **BatchProcessor 模式**是生产级批量处理的标准做法

至此，**Chapter 4（OpenAI 兼容 API 服务）全部完成**！接下来我们将进入 **Chapter 5：模型优化与高级特性**。
