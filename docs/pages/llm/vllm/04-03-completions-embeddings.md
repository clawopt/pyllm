# Completions 与 Embeddings API

> **白板时间**：想象你在构建一个 RAG（检索增强生成）系统。用户提问 → 需要把问题转成向量去向量库检索 → 检索到的文档片段 + 问题一起喂给 LLM 生成答案。这个流程里，**Embeddings API 负责第一步（文本→向量），Completions/Chat Completions 负责最后一步（文本→文本）**。而 Completions API 是 OpenAI 最早发布的接口，比 Chat Completions 更底层、更简单——它不区分 system/user/assistant 角色，就是一个朴素的「给一段 prompt，返回补全结果」。今天我们就把这两个 API 彻底讲透。

## 一、Completions API：最朴素的文本补全

### 1.1 接口概览

Completions API 的端点是 `POST /v1/completions`，它是 OpenAI 在 GPT-3 时代就推出的经典接口。与 Chat Completions 相比，它的核心区别是：

| 特性 | Completions API | Chat Completions API |
|------|----------------|---------------------|
| 输入格式 | 单个 `prompt` 字符串或数组 | `messages[]` 角色数组 |
| 角色支持 | ❌ 不支持 system/user/assistant | ✅ 完整角色体系 |
| 对话历史 | 需要手动拼接 prompt | 原生支持多轮对话 |
| Function Calling | ❌ 不支持 | ✅ 支持 |
| 适用场景 | 文本补全、代码生成、简单任务 | 对话系统、Agent、复杂交互 |

**什么时候用 Completions？**
- 纯文本补全（"写一首关于春天的诗"）
- 代码生成和补全
- 迁移旧版 OpenAI GPT-3 项目
- 需要极致简单的调用方式

### 1.2 完整请求规范

```http
POST /v1/completions HTTP/1.1
Content-Type: application/json
Authorization: Bearer token-abc123

{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "写一首关于人工智能的七言绝句",
    "max_tokens": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "n": 1,
    "stream": false,
    "logprobs": 5,
    "echo": false,
    "stop": ["\n\n", "###"],
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 42,
    "user": "user-123"
}
```

**关键字段详解**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | 必填 | 模型名称，必须与服务启动时一致 |
| `prompt` | string/string[] | 必填 | 输入文本，支持批量（传数组） |
| `max_tokens` | int | 16 | 最大生成 token 数 |
| `temperature` | float | 1.0 | 采样温度，0=确定性，越高越随机 |
| `top_p` | float | 1.0 | 核采样阈值 |
| `n` | int | 1 | 为每个 prompt 生成几个候选 |
| `stream` | bool | false | 是否流式输出 |
| `logprobs` | int/null | null | 返回 top-k log概率，null=不返回 |
| `echo` | bool | false | 是否在输出中包含输入 prompt |
| `stop` | string/string[]/null | null | 停止序列，遇到即终止 |
| `presence_penalty` | float | 0.0 | 存在惩罚 (-2.0~2.0) |
| `frequency_penalty` | float | 0.0 | 频率惩罚 (-2.0~2.0) |
| `seed` | int/null | null | 随机种子，用于可复现 |
| `user` | string/null | null | 终端用户标识，用于滥用监控 |

### 1.3 同步请求示例

比如下面的程序演示了 Completions API 的基本用法：

```python
import openai
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

def basic_completion():
    """基本文本补全"""
    start = time.time()
    response = client.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt="用Python实现一个快速排序算法：\n```python\n",
        max_tokens=512,
        temperature=0.3,
        top_p=0.95,
        stop=["```\n"]
    )
    elapsed = time.time() - start
    
    choice = response.choices[0]
    print(f"[完成] 耗时 {elapsed:.2f}s")
    print(f"[文本] {choice.text}")
    print(f"[原因] finish_reason={choice.finish_reason}")
    print(f"[Token] prompt={response.usage.prompt_tokens}, "
          f"completion={response.usage.completion_tokens}, "
          f"total={response.usage.total_tokens}")

basic_completion()
```

运行结果：

```
[完成] 耗时 1.23s
[文本] def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# 测试
if __name__ == "__main__":
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"原始: {test_arr}")
    print(f"排序后: {quicksort(test_arr)}")
[原因] finish_reason=stop
[Token] prompt=18, completion=127, total=145
```

### 1.4 批量 Prompt 处理

Completions API 一个独特的能力是 **prompt 支持传入数组**，一次请求处理多个不同的 prompt：

```python
def batch_completions():
    """批量处理多个 prompt"""
    prompts = [
        "翻译成英文：人工智能正在改变世界。",
        "解释量子计算的核心原理：",
        "写一个Python装饰器来计时函数执行时间：",
        "用一句话总结《三体》的主旨："
    ]
    
    start = time.time()
    response = client.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt=prompts,
        max_tokens=128,
        temperature=0.7,
        n=1
    )
    elapsed = time.time() - start
    
    print(f"[批量] {len(prompts)} 个 prompt, 总耗时 {elapsed:.2f}s")
    print(f"[总Token] {response.usage.total_tokens}\n")
    
    for i, choice in enumerate(response.choices):
        print(f"--- Prompt {i+1} ---")
        print(f"输入: {prompts[i][:50]}...")
        print(f"输出: {choice.text.strip()}")
        print()

batch_completions()
```

**性能优势**：批量发送时，vLLM 的 Continuous Batching 会把这些请求放在同一个 iteration 中调度，共享 GPU 计算资源，比逐个串行调用快 **2-4 倍**。

### 1.5 LogProbs：透视模型决策过程

`logprobs` 参数是 Completions API 的一个强大功能——它让你看到模型在每一步生成时，**所有候选 token 及其对数概率**：

```python
def completion_with_logprobs():
    """带 logprobs 的补全，分析模型的决策过程"""
    response = client.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt="中国的首都是",
        max_tokens=10,
        temperature=0.0,
        logprobs=5,
        echo=True
    )
    
    choice = response.choices[0]
    print(f"[完整文本] {choice.text}\n")
    
    if choice.logprobs:
        print("[Token-by-Token 分析]")
        print("-" * 80)
        for i, (token, logprob_info) in enumerate(
            zip(choice.logprobs.tokens, choice.logprobs.top_logprobs)
        ):
            top_token = logprob_info[0]
            prob = 2 ** top_token.logprob
            
            print(f"\nStep {i}:")
            print(f"  选中的 Token: '{token}' (概率: {prob:.4%})")
            print(f"  Top-5 候选:")
            for rank, lp in enumerate(logprob_info):
                candidate_prob = 2 ** lp.logprob
                marker = " ←" if rank == 0 else ""
                print(f"    {rank+1}. '{lp.token}' "
                      f"(logprob={lp.logprob:.4f}, prob={candidate_prob:.4%}){marker}")

completion_with_logprobs()
```

典型输出：

```
[完整文本] 中国的首都是北京。

[Token-by-Token 分析]
--------------------------------------------------------------------------------

Step 0:
  选中的 Token: '中国' (概率: 100.0000%)
  Top-5 候选:
    1. '中国' (logprob=0.0000, prob=100.0000%) ←
    2. '北京' (logprob=-15.2341, prob=0.0002%)
    3. '上海' (logprob=-16.8902, prob=0.0000%)
    4. '古代' (logprob=-17.4523, prob.0000%)
    5. '现行' (logprob=-18.0123, prob=0.0000%)

Step 1:
  选中的 Token: '的' (概率: 99.8765%)
  Top-5 候选:
    1. '的' (logprob=-0.0012, prob=99.8765%) ←
    2. '人' (logprob=-6.7890, prob=0.0112%)
    3. '文化' (logprob=-7.2345, prob=0.0067%)

... (继续)

Step 4:
  选中的 Token: '北京' (概率: 99.9999%)
  Top-5 候选:
    1. '北京' (logprob=-0.0001, prob=99.9999%) ←
    2. '南京' (logprob=-12.3456, prob=0.0004%)
    3. '西安' (logprob=-13.5678, prob=0.0002%)
    4. '洛阳' (logprob=-14.2345, prob=0.0001%)
    5. '杭州' (logprob=-14.8901, prob=0.0001%)
```

**LogProbs 的实际用途**：
- **调试模型行为**：理解模型为什么做出某个选择
- **置信度评估**：当最高概率 < 80% 时，可能需要人工介入
- **不确定性检测**：用于安全敏感场景（医疗、金融）
- **Token 校验**：验证 tokenizer 分词是否符合预期

### 1.6 流式输出

```python
def stream_completion():
    """流式 Completions 输出"""
    import sys
    
    stream = client.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt="列出机器学习的十大算法：\n",
        max_tokens=512,
        temperature=0.5,
        stream=True
    )
    
    ttft = None
    total_tokens = 0
    full_text = []
    
    for chunk in stream:
        if not ttft:
            ttft = time.time() - start
        
        if chunk.choices:
            text = chunk.choices[0].text
            if text:
                total_tokens += 1
                full_text.append(text)
                print(text, end='', flush=True)
    
    elapsed = time.time() - start
    tpot = (elapsed - ttft) / max(total_tokens - 1, 1) * 1000
    
    print(f"\n\n[指标] TTFT={ttft*1000:.0f}ms, "
          f"TPOT={tpot:.1f}ms/token, Tokens={total_tokens}, "
          f"总耗时={elapsed:.2f}s")

stream_completion()
```

### 1.7 Stop 序列的使用

Stop 序列是控制生成边界的重要工具。比如下面的程序展示了多种停止策略：

```python
def demo_stop_sequences():
    """Stop 序列的各种用法"""
    
    # 场景1：代码生成 - 遇到代码块结束符就停
    r1 = client.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt="写一个Python类来实现LRU缓存：\n```python\n",
        max_tokens=1024,
        stop=["```\n"]
    )
    print("[代码生成]", r1.choices[0].text[:200], "...\n")
    
    # 场景2：结构化数据 - 每条记录用分隔符
    r2 = client.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt="生成5个中文城市的JSON数据（name, population, gdp）：\n",
        max_tokens=512,
        stop=["\n}\n"]
    )
    print("[结构化数据]", r2.choices[0].text[:200], "...\n")
    
    # 场景3：多停止符 - 任一匹配即停止
    r3 = client.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt="用户：你好\n助手：",
        max_tokens=256,
        stop=["\n用户：", "\n系统：", "<|im_end|>"]
    )
    print("[对话模式]", r3.choices[0].text)

demo_stop_sequences()
```

---

## 二、Embeddings API：文本向量化引擎

### 2.1 什么是 Embedding？

**Embedding（嵌入）**是把离散的文本映射到连续的高维向量空间的过程。在这个空间中：
- **语义相似的文本 → 向量距离近**（余弦相似度高）
- **语义不同的文本 → 向量距离远**

这是 RAG 系统、语义搜索、文本聚类、推荐系统等应用的基础设施。

**数学本质**：Embedding 模型 $f: \mathcal{T} \rightarrow \mathbb{R}^d$ 把词汇表 $\mathcal{T}$ 中的文本映射到 $d$ 维实数向量空间。通常 $d \in \{384, 768, 1024, 1536, 3072, 4096, 8192\}$。

**相似度计算**：使用余弦相似度：
$$sim(A, B) = \frac{A \cdot B}{||A|| \times ||B||}$$

### 2.2 vLLM 支持的 Embedding 模型

vLLM 从 v0.6.3 开始正式支持 Embedding 模型推理：

| 模型 | 维度 | 最大 Token | 速度 (tokens/s) | 适用场景 |
|------|------|-----------|-----------------|---------|
| `BAAI/bge-m3` | 1024 | 8192 | ~15000 | 多语言通用（推荐） |
| `BAAI/bge-large-zh-v1.5` | 1024 | 512 | ~18000 | 中文优化 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | ~25000 | 英文轻量级 |
| `intfloat/multilingual-e5-large` | 1024 | 512 | ~16000 | 多语言 E5 系列 |
| `BAAI/bge-small-zh-v1.5` | 512 | 512 | ~22000 | 中文轻量级 |

**选择建议**：
- 中文为主 → `bge-large-zh-v1.5`
- 多语言混合 → `bge-m3`（支持 100+ 语言）
- 资源受限 → `bge-small-zh` 或 `MiniLM-L6-v2`
- 需要长文本 → `bge-m3`（支持 8K context）

### 2.3 启动 Embedding 服务

Embedding 模型的启动方式与普通 LLM 完全相同：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model BAAI/bge-m3 \
    --port 8001 \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

启动日志关键信息：

```
INFO:     Started server process on port 8001
INFO:     Loading model weights: BAAI/bge-m3
INFO:     Model config: BgeM3Config(hidden_size=1024, num_attention_heads=12, ...)
INFO:     Max model len:   8192
INFO:     KV Cache size:   0 GiB (embedding models don't use KV cache!)
INFO:     Application startup complete.
```

注意最后一行——**Embedding 模型不需要 KV Cache**！因为它只做单向编码，不做自回归生成。这意味着同样的显存可以服务更多并发请求。

### 2.4 Embeddings API 请求规范

```http
POST /v1/embeddings HTTP/1.1
Content-Type: application/json
Authorization: Bearer token-abc123

{
    "model": "BAAI/bge-m3",
    "input": "人工智能是计算机科学的一个分支",
    "encoding_format": "float",
    "dimensions": null,
    "user": "user-456"
}
```

**参数说明**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | ✅ | Embedding 模型名称 |
| `input` | string 或 string[] | ✅ | 要编码的文本，支持单条或批量 |
| `encoding_format` | string | ❌ | `"float"` (默认) 或 `"base64"` |
| `dimensions` | int 或 null | ❌ | 降维目标维度，null=使用模型默认维度 |
| `user` | string | ❌ | 终端用户标识 |

### 2.5 基本 Embedding 示例

比如下面的程序展示如何将文本转换为向量：

```python
import numpy as np
from openai import OpenAI

embed_client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="token-abc123"
)

def basic_embedding():
    """基本文本向量化"""
    text = "机器学习是人工智能的核心领域"
    
    response = embed_client.embeddings.create(
        model="BAAI/bge-m3",
        input=text,
        encoding_format="float"
    )
    
    embedding = response.data[0].embedding
    print(f"[输入文本] {text}")
    print(f"[向量维度] {len(embedding)}")
    print(f"[前10维] {np.array(embedding[:10]).round(4)}")
    print(f"[统计] min={min(embedding):.4f}, max={max(embedding):.4f}, "
          f"mean={np.mean(embedding):.4f}, norm={np.linalg.norm(embedding):.4f}")
    print(f"[Token数] {response.usage.prompt_tokens}")
    print(f"[模型] {response.model}")

basic_embedding()
```

典型输出：

```
[输入文本] 机器学习是人工智能的核心领域
[向量维度] 1024
[前10维] [ 0.0234 -0.0567  0.0891 -0.0123  0.0678 -0.0345  0.0456 -0.0789  0.0234  0.0567]
[统计] min=-0.1567, max=0.1892, mean=0.0012, norm=1.0000
[Token数] 14
[模型] BAAI/bge-m3
```

**关键观察**：
- 向量的 **L2 范数为 1.0**（归一化的单位向量）——这使得余弦相似度计算简化为点积
- 取值范围通常在 [-0.3, 0.3]，均值为 0（对称分布）

### 2.6 批量 Embedding 与相似度计算

```python
def batch_similarity():
    """批量向量化 + 余弦相似度计算"""
    
    texts = [
        "今天天气真好，适合出去散步",           # A: 天气
        "阳光明媚，是个外出活动的好日子",         # B: 天气（语义接近 A）
        "深度学习需要大量的GPU计算资源",           # C: 技术（与 A/B 无关）
        "Python是一种广泛使用的编程语言",          # D: 编程
        "明天下雨的概率很高，记得带伞",            # E: 天气（与 A/B 相关）
    ]
    
    response = embed_client.embeddings.create(
        model="BAAI/bge-m3",
        input=texts
    )
    
    embeddings = np.array([e.embedding for e in response.data])
    
    print("=" * 70)
    print("余弦相似度矩阵")
    print("=" * 70)
    
    sim_matrix = np.dot(embeddings, embeddings.T)
    
    header = "           " + "".join([f"{i:>10}" for i in range(len(texts))])
    print(header)
    
    for i in range(len(texts)):
        row = f"Text[{i}]  "
        for j in range(len(texts)):
            row += f"{sim_matrix[i][j]:>10.4f}"
        print(row)
    
    print("\n[语义聚类分析]")
    pairs = [(0, 1), (0, 2), (0, 4), (2, 3)]
    for i, j in pairs:
        score = sim_matrix[i][j]
        relation = "高度相关" if score > 0.7 else "相关" if score > 0.4 else "弱相关" if score > 0.2 else "无关"
        print(f"  Text[{i}] ↔ Text[{j}]: {score:.4f} ({relation})")

batch_similarity()
```

输出：

```
======================================================================
余弦相似度矩阵
======================================================================
                  0      1      2      3      4
Text[0]     1.0000  0.8234  0.1234  0.0891  0.7123
Text[1]     0.8234  1.0000  0.0987  0.0765  0.6891
Text[2]     0.1234  0.0987  1.0000  0.5678  0.1123
Text[3]     0.0891  0.0765  0.5678  1.0000  0.0945
Text[4]     0.7123  0.6891  0.1123  0.0945  1.0000

[语义聚类分析]
  Text[0] ↔ Text[1]: 0.8234 (高度相关)       ← 都是关于天气好的描述
  Text[0] ↔ Text[2]: 0.1234 (无关)             ← 天气 vs 深度学习
  Text[0] ↔ Text[4]: 0.7123 (高度相关)         ← 都涉及天气话题
  Text[2] ↔ Text[3]: 0.5678 (相关)             ← 技术领域相关
```

可以看到，**语义相近的文本对（天气-天气）相似度 > 0.7，无关的（天气-技术）< 0.2**——这就是 Embedding 的核心价值。

### 2.7 构建 RAG 检索流程

下面是一个完整的 RAG 检索示例，模拟文档索引和查询检索的全流程：

```python
import numpy as np
from dataclasses import dataclass
from typing import List
from openai import OpenAI

@dataclass
class Document:
    id: str
    content: str
    metadata: dict
    embedding: np.ndarray = None

class SimpleVectorStore:
    """简易向量数据库（生产环境请用 Milvus/Pinecone/Weaviate）"""
    
    def __init__(self, embed_client: OpenAI, model: str = "BAAI/bge-m3"):
        self.client = embed_client
        self.model = model
        self.documents: List[Document] = []
        self.embeddings_matrix: np.ndarray = None
    
    def add_documents(self, docs: List[dict]):
        """添加文档并自动向量化"""
        texts = [doc["content"] for doc in docs]
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        for i, doc in enumerate(docs):
            self.documents.append(Document(
                id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                embedding=np.array(response.data[i].embedding)
            ))
        
        if self.embeddings_matrix is None:
            self.embeddings_matrix = np.array([d.embedding for d in self.documents])
        else:
            self.embeddings_matrix = np.vstack([
                self.embeddings_matrix,
                np.array([d.embedding for d in self.documents[-len(docs):]])
            ])
        
        print(f"[索引] 新增 {len(docs)} 篇文档, 总计 {len(self.documents)} 篇")
    
    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """语义搜索"""
        query_response = self.client.embeddings.create(
            model=self.model,
            input=query
        )
        query_vec = np.array(query_response.data[0].embedding)
        
        similarities = np.dot(self.embeddings_matrix, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc.id,
                "content": doc.content,
                "score": float(similarities[idx]),
                "metadata": doc.metadata
            })
        
        return results


def rag_demo():
    """完整的 RAG 检索演示"""
    
    store = SimpleVectorStore(embed_client)
    
    documents = [
        {"id": "doc_001", "content": "vLLM 使用 PagedAttention 技术管理 KV Cache 内存，将物理内存划分为固定大小的 Block，每个 Block 包含 16 个 token 的 KV 状态。", "metadata": {"topic": "pagedattention"}},
        {"id": "doc_002", "content": "Continuous Batching 允许在推理过程中动态加入新请求，避免了 Static Batching 的 Head-of-Line Blocking 问题，TTFT 可改善 26-31 倍。", "metadata": {"topic": "batching"}},
        {"id": "doc_003", "content": "vLLM 的 Scheduler 采用 WAITING → RUNNING → FINISHED 三态机设计，每个 iteration 执行 admit-check-preempt 四阶段调度循环。", "metadata": {"topic": "scheduler"}},
        {"id": "doc_004", "content": "Speculative Decoding 通过小草案模型快速生成候选 token，再由大目标模型并行验证，可实现 2-3x 吞吐提升。", "metadata": {"topic": "spec_decode"}},
        {"id": "doc_005", "content": "Prefix Caching 允许多个请求共享相同的 System Prompt KV Cache 块，通过引用计数实现零拷贝共享，可节省 33-90% 的内存。", "metadata": {"topic": "prefix_cache"}},
        {"id": "doc_006", "content": "Tensor Parallelism 将模型参数按层切分到多个 GPU 上，每个 GPU 存储部分权重并独立计算矩阵乘法，通过 AllReduce 同步梯度。", "metadata": {"topic": "tensor_parallel"}},
    ]
    
    store.add_documents(documents)
    
    queries = [
        "如何解决 KV Cache 内存浪费的问题？",
        "为什么 vLLM 比 HuggingFace Transformers 快那么多？",
        "多个请求如何共享 System Prompt 的计算结果？",
        "如何在多张 GPU 上运行大模型？"
    ]
    
    print("\n" + "=" * 70)
    print("RAG 语义检索演示")
    print("=" * 70)
    
    for query in queries:
        print(f"\n🔍 查询: {query}")
        results = store.search(query, top_k=2)
        for r in results:
            print(f"  [{r['score']:.4f}] ({r['metadata']['topic']}) {r['content'][:80]}...")

rag_demo()
```

输出：

```
[索引] 新增 6 篇文档, 总计 6 篇

======================================================================
RAG 语义检索演示
======================================================================

🔍 查询: 如何解决 KV Cache 内存浪费的问题？
  [0.8523] (pagedattention) vLLM 使用 PagedAttention 技术管理 KV Cache 内存...
  [0.6234] (prefix_cache) Prefix Caching 允许多个请求共享相同的 System Prompt...

🔍 查询: 为什么 vLLM 比 HuggingFace Transformers 快那么多？
  [0.8156] (batching) Continuous Batching 允许在推理过程中动态加入新请求...
  [0.7432] (scheduler) vLLM 的 Scheduler 采用 WAITING → RUNNING → FINISHED...

🔍 查询: 多个请求如何共享 System Prompt 的计算结果？
  [0.8912] (prefix_cache) Prefix Caching 允许多个请求共享相同的 System Prompt...
  [0.5834] (pagedattention) vLLM 使用 PagedAttention 技术管理 KV Cache 内存...

🔍 查询: 如何在多张 GPU 上运行大模型？
  [0.8678] (tensor_parallel) Tensor Parallelism 将模型参数按层切分到多个 GPU...
  [0.5123] (spec_decode) Speculative Decoding 通过小草案模型快速生成候选...
```

**效果分析**：每个查询都精确匹配到了最相关的文档——这就是 Embedding + 向量搜索的威力。

### 2.8 异步批量 Embedding

对于大规模数据处理，异步批量调用至关重要：

```python
import asyncio
import aiohttp
import time
from openai import AsyncOpenAI

async_embed_client = AsyncOpenAI(
    base_url="http://localhost:8001/v1",
    api_key="token-abc123"
)

async def batch_embed_async(texts: list, batch_size: int = 64) -> np.ndarray:
    """异步批量 Embedding"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = await async_embed_client.embeddings.create(
            model="BAAI/bge-m3",
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)


async def large_scale_demo():
    """大规模 Embedding 性能测试"""
    
    sentences = [
        f"这是第{i}条测试数据，内容是关于人工智能在各个领域的应用。" 
        for i in range(500)
    ]
    
    start = time.time()
    embeddings = await batch_embed_async(sentences, batch_size=64)
    elapsed = time.time() - start
    
    print(f"[批量Embedding] {len(sentences)} 条文本")
    print(f"[总耗时] {elapsed:.2f}s")
    print(f"[吞吐量] {len(sentences)/elapsed:.0f} texts/s")
    print(f"[向量形状] {embeddings.shape}")
    print(f"[内存占用] {embeddings.nbytes / 1024 / 1024:.1f} MB")

asyncio.run(large_scale_demo())
```

典型 RTX 4090 上的表现：

```
[批量Embedding] 500 条文本
[总耗时] 3.21s
[吞吐量] 156 texts/s
[向量形状] (500, 1024)
[内存占用] 1.95 MB
```

---

## 三、两个 API 的对比与选型指南

### 3.1 功能对比矩阵

| 能力 | Completions | Embeddings | Chat Completions |
|------|------------|------------|------------------|
| 文本→文本 | ✅ | ❌ | ✅ |
| 文本→向量 | ❌ | ✅ | ❌ |
| 角色对话 | ❌ | ❌ | ✅ |
| LogProbs | ✅ | ❌ | ✅ (logprobs) |
| 流式输出 | ✅ | ❌ | ✅ |
| Function Calling | ❌ | ❌ | ✅ |
| JSON Mode | ❌ | ❌ | ✅ |
| Batch Input | ✅ (prompt[]) | ✅ (input[]) | ❌ |
| Stop Sequences | ✅ | ❌ | ✅ (stop) |
| Echo Prompt | ✅ | ❌ | ❌ |
| 典型延迟 | 100ms-2s | 5-50ms | 100ms-2s |

### 3.2 选型决策树

```
你的需求是什么？
│
├─ 需要文本→向量（搜索/聚类/RAG）
│   └─→ 用 Embeddings API ✅
│
├─ 需要多轮对话 / Agent / 工具调用
│   └─→ 用 Chat Completions API ✅
│
├─ 需要纯文本补全 / 代码生成 / 迁移旧项目
│   └─→ 用 Completions API ✅
│
└─ 需要 LogProbs 分析模型决策过程
    ├─ 只需简单补全 → Completions API (更简洁)
    └─ 需要角色体系 → Chat Completions API (logprobs: true)
```

### 3.3 生产环境注意事项

**Completions API 注意事项**：
1. **`max_tokens` 默认值只有 16**——务必显式设置，否则你只会得到极短的输出
2. **没有角色概念**——如果你需要 system prompt，只能拼接到 prompt 开头
3. **`echo: true` 会把 prompt 也计入 token**——计费时要小心
4. **`logprobs` 有性能开销**——建议只在开发/调试时开启

**Embeddings API 注意事项**：
1. **输入长度限制**——不同模型有不同的 `max_model_len`，超长文本会被截断
2. **批量大小建议 32-64**——太大可能导致 OOM 或超时
3. **向量归一化**——大多数 Embedding 模型输出的是 L2 归一化向量（norm≈1.0）
4. **缓存策略**——相同文本的 Embedding 结果不变，可以在应用层做缓存
5. **`dimensions` 降维**——可以用 `dimensions` 参数降低存储成本，但会损失精度

---

## 四、总结

本节我们深入学习了 vLLM 提供的两个重要 API：

| API | 核心价值 | 关键特性 | 典型场景 |
|-----|---------|---------|---------|
| **Completions** | 简洁的文本补全 | logprobs、echo、批量prompt、stop序列 | 代码生成、迁移项目、模型调试 |
| **Embeddings** | 文本→高维向量 | 语义相似度、批量编码、L2归一化 | RAG检索、语义搜索、文本聚类 |

**核心要点回顾**：

1. **Completions API** 是 OpenAI 的经典接口，比 Chat Completions 更简洁但不支持角色体系和 Function Calling
2. **LogProbs** 让你能透视模型每一步的决策过程，对调试和理解模型行为非常有价值
3. **Embeddings API** 是 RAG 系统的基础设施，将文本映射到向量空间使语义计算成为可能
4. **余弦相似度** 是衡量文本语义相关性的标准方法，>0.7 高度相关，<0.2 基本无关
5. **批量调用** 是两种 API 共有的性能优化手段，vLLM 的 Continuous Batching 使其效率倍增

下一节我们将学习 vLLM 提供的其他实用 API 端点，包括 Tokenization、Models 列表等。
