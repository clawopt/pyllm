# 9.1 模型服务化：从训练到 API 的最后一公里

经过前面八章的漫长旅程，你现在拥有了完整的 LLM 工程能力：你能用纯 PyTorch 从零搭建 Transformer，能用 Lightning 或 HF Trainer 高效地训练和微调模型，能在多 GPU 上做分布式训练，还能通过 torch.compile 和量化技术优化推理性能。但所有这些能力都还停留在"实验环境"——模型以 `.pt` 文件的形式躺在磁盘上，只有你自己知道怎么加载它、怎么调用它。当你的产品经理或客户问"这个模型什么时候能上线"的时候，你需要回答的不是一个文件路径，而是一个**可访问的 HTTP 接口地址**。这一节的目标就是完成这"最后一公里"——把一个训练好的 LLM 模型包装成可部署的推理服务。

## 为什么不能直接在训练代码里跑推理？

很多初学者的第一反应是："我训练的时候不是已经能调用 model.generate() 了吗？再加个 Flask 路由不就行了？" 这个想法方向没错，但直接把训练代码改造成服务代码会带来一系列问题：

**依赖冲突**：训练时你安装了 PyTorch Lightning、PEFT、DeepSpeed 等一堆重型依赖（可能几十 GB），但推理时只需要 PyTorch 本身 + tokenizer。如果直接把训练环境打包成 Docker 镜像，镜像体积可能超过 20GB，启动时间长达几分钟。

**资源浪费**：训练代码中包含了 optimizer、scheduler、gradient checkpointing 等推理完全不需要的组件。它们不仅占用磁盘空间，还可能在导入时触发不必要的初始化（比如 CUDA context 创建），白白消耗 GPU 显存。

**缺乏生产特性**：一个可用的推理服务需要的远不止"能跑 forward pass"——它还需要请求批处理（batching）、并发控制、超时处理、错误重试、负载均衡、健康检查、指标监控、日志收集……这些都不是训练代码天然具备的。

**安全风险**：训练代码通常包含敏感信息（数据路径、API key、内部 IP），直接暴露给外部是不可接受的。

所以正确的做法是：把模型导出为轻量级格式 → 用专门的推理框架加载 → 包装为标准化的 API 服务。

## 方案一：FastAPI + 原生 PyTorch

最直接的方案是用 FastAPI（Python 最流行的异步 Web 框架）包装 PyTorch 模型。适合快速原型验证和小规模内部使用：

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import torch
import uvicorn


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float
    model_name: str


class LLMService:
    """LLM 推理服务"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        
        print(f"[Loading model from {model_path}...]")
        self._load_model()
        
        print(f"✓ Model loaded on {self.device}")

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        import time
        
        start_time = time.perf_counter()
        
        messages = [
            {"role": "user", "content": request.prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature if request.do_sample else None,
            top_p=request.top_p if request.do_sample else None,
            top_k=request.top_k if request.do_sample else None,
            do_sample=request.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        tokens_gen = output_ids.shape[1] - inputs['input_ids'].shape[1]
        
        return GenerateResponse(
            text=generated_text,
            tokens_generated=tokens_gen,
            latency_ms=round(latency_ms, 2),
            model_name=self.model_path.split('/')[-1],
        )


app = FastAPI(title="LLM Inference Service")
service: Optional[LLMService] = None


@app.on_event("startup")
async def startup():
    global service
    service = LLMService(model_path="./output/qwen-merged")


@app.post("/v1/chat/completions", response_model=GenerateResponse)
async def chat_completions(request: GenerateRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if request.max_new_tokens > 4096:
        raise HTTPException(status_code=400, detail="max_new_tokens cannot exceed 4096")

    response = await run_in_executor(None, service.generate, (request,))
    return response


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": service is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 关键设计决策解析

**`run_in_executor`**：这是 FastAPI 中最重要的一个细节。默认情况下，FastAPI 在主事件循环（event loop）中运行所有路由函数。但 PyTorch 的前向传播是 CPU/GPU 密集型操作——如果在主线程中执行，它会阻塞事件循环，导致其他请求无法被处理（即使你有多个 worker）。`run_in_executor()` 把推理任务放到线程池中执行，保持主线程空闲来接收新请求。这是实现**伪并发**的最简单方式——注意不是真正的并行（GIL 限制），但对于单卡推理场景足够了。

**Pydantic 模型校验**：`GenerateRequest` 和 `GenerateResponse` 使用 Pydantic 做类型检查和自动文档生成。FastAPI 会自动根据这些模型生成 OpenAPI/Swagger 规范文档，你可以访问 `http://localhost:8000/docs` 看到交互式 API 文档。

**错误处理**：对空 prompt、超长输出等边界情况做了显式的错误返回，而不是让程序崩溃后返回 500 错误。这在生产环境中至关重要——好的错误信息能让前端/客户端做出合理的降级处理。

**`/health` 端点**：供 Kubernetes/Docker 的 liveness probe 使用。K8s 会定期访问这个端点来判断容器是否存活；如果连续多次失败就重启容器。

### 启动与测试

```bash
# 启动服务
python serve_fastapi.py

# 另开终端测试
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好，请介绍一下自己", "max_new_tokens": 128}'

# 响应示例:
# {
#   "text": "你好！我是一个基于Qwen架构的大语言模型...",
#   "tokens_generated": 86,
#   "latency_ms": 1234.56,
#   "model_name": "qwen-merged"
# }

# 健康检查
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true}
```

## 方案二：vLLM / SGLang —— 生产级推理引擎

如果你需要更高的吞吐量（比如每秒处理 10+ 个请求）或者更低的延迟，原生 PyTorch + FastAPI 就不够用了。这时候应该考虑专门的 **LLM 推理引擎**——它们针对 LLM 的自回归生成特性做了大量优化：

**vLLM** 是 UC Berkeley 开源的高性能 LLM 推理和服务引擎，目前是社区最活跃的项目之一。它的核心优化包括：
- **PagedAttention**：解决 KV Cache 内存碎片问题，支持更大的 batch size
- **Continuous Batching**：动态地把不同长度的请求组合到一个 batch 中，最大化 GPU 利用率
- **张量并行**：自动将大模型分割到多张 GPU 上
- **OpenAI 兼容 API**：开箱即用的 `/v1/chat/completions` 接口

```python
# vLLM 服务启动方式（不需要写 Python 代码！）
# pip install vllm

from vllm.entrypoints.openai.api_server import (
    ArgSettings as VllmArgs,
    OpenAIServer,
)

server = OpenAIServer(
    model="./output/qwen-merged",
    tensor_parallel_size=1,       # 张量并行度
    gpu_memory_utilization=0.90, # GPU 显存使用上限
    max_model_len=2048,           # 最大序列长度
    dtype="bfloat16",
)

server.run(host="0.0.0.0", port=8000)
```

更常见的做法是直接用命令行启动 vLLM：

```bash
# 安装
pip install vllm

# 启动服务（一行命令！）
python -m vllm.entrypoints.openai.api_server \
    --model ./output/qwen-merged \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 2048 \
    --dtype bfloat16 \
    --port 8000

# 测试（OpenAI 兼容格式）
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-merged",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

vLLM 相比 FastAPI + 原生 PyTorch 的优势在于：
- **吞吐量提升 5~10×**（通过 Continuous Batching 和优化的 kernel 实现）
- **内存效率更高**（PagedAttention 减少了 KV Cache 碎片）
- **自动管理**（无需手动写 batching / scheduling 逻辑）

代价是需要额外安装 vLLM 及其依赖（包括自定义的 CUDA kernel），且对某些特殊模型架构的支持可能不如原生 PyTorch 完善。

## 方案三：TGI (Text Generation Inference) — HuggingFace 官方方案

**TGI** 是 HuggingFace 官方推出的 LLM 推理服务框架，基于 Rust + Candle（HuggingFace 的 ML 推理框架）构建。它的特点是：
- 极低的内存开销（Rust 零成本抽象 + Candle 轻量级推理引擎）
- 原生支持 HuggingFace Hub 上的模型（一行命令启动任何 HF 模型）
- 内置量化支持（FP8 / INT4 / EXL2）
- OpenAI 兼容 API

```bash
# Docker 方式启动 TGI（推荐）
docker run --gpus all -p 8080:80 \
  -v ~/.cache/huggingface:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --quantize awq \
  --max-total-tokens 4096

# 访问
curl http://localhost:8080/generate \
  -X POST -d '{"inputs":"你好","parameters":{"max_new_tokens":128}}'
```

## 三种方案对比

| 维度 | FastAPI + PyTorch | vLLM | TGI |
|-----|------------------|------|-----|
| **开发速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **吞吐量** | 低 (~1 req/s) | **高** (~10+ req/s) | **高** (~10+ req/s) |
| **延迟** | 中等 | 低 | **最低** |
| **内存效率** | 低 | **高** (PagedAttention) | **最高** |
| **定制灵活性** | **100%** | 中等 | 较低 |
| **部署复杂度** | 低 | 中等 | 低 (Docker) |
| **生态兼容性** | 任意 | HF 模型为主 | **HF 模型优先** |
| **适用场景** | 快速原型 / 内部工具 | **生产服务** | **生产服务** |

**选型建议**：
- 刚开始、只需要一个能跑的接口 → **FastAPI + PyTorch**
- 需要服务真实用户流量 → **vLLM**（社区活跃、文档好、灵活）
- 运行 HuggingFace Hub 上的标准模型 → **TGI**（最省事）
- 对延迟有极致要求 → **TensorRT-LLM**（下一节会提到）

## Docker 化：可复现的部署单元

无论选择哪种方案，最终都应该打包成 Docker 镜像以确保环境一致性：

```dockerfile
# Dockerfile for LLM inference service
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .

EXPOSE 8000

CMD ["python", "serve.py"]
```

```txt
# requirements.txt — 推理专用，不含训练依赖
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
torch>=2.1.0
transformers>=4.36.0
sentencepiece>=0.1.99
protobuf>=4.25.0
```

```bash
# 构建并运行
docker build -t llm-service:v1 .
docker run --gpus all -p 8000:8000 llm-service:v1
```

关键点：requirements.txt 只包含推理所需的依赖（torch, transformers, fastapi 等），不包含 lightning, peft, deepspeed, wandb 等训练专用包。这能把镜像体积从 15GB+ 压缩到 ~5GB，启动时间从分钟级降到秒级。
