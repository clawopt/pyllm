# LoRA 服务化实战

> **白板时间**：上一节我们理解了 LoRA 的原理和 vLLM 的支持能力。现在让我们动手——从启动服务、调用 API、动态管理 LoRA，到构建一个完整的多租户系统。这一节的目标是：**读完你就能在生产环境中部署和管理 LoRA 适配器了**。

## 一、启动带 LoRA 的 vLLM 服务

### 1.1 完整启动命令模板

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules \
        medical-lora=/models/adapters/medical-v1 \
        legal-lora=/models/adapters/legal-v1 \
        code-assistant-lora=/models/adapters/code-v1 \
        creative-writing-lora=/models/adapters/creative-v1 \
    --max-loras 16 \
    --max-lora-rank 64 \
    --max-lora-size 256 \
    --lora-extra-vocab-size 25600 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --trust-remote-code
```

### 1.2 Docker Compose 完整配置

```yaml
# docker-compose.lora.yml
version: '3.8'

services:
  vllm-lora:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models/base:/root/.cache/huggingface/hub
      - ./models/lora:/models/adapters
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']
    command: >
      --model meta-llama/Meta-Llama-3.1-8B-Instruct
      --enable-lora
      --lora-modules
        medical-lora=/models/adapters/medical-v1
        legal-lora=/models/adapters/legal-v1
      --max-loras 16
      --max-lora-rank 64
      --gpu-memory-utilization 0.90
      --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

## 二、API 调用：指定 LoRA

### 2.1 通过 model 参数指定 LoRA

vLLM 使用 `base-model@lora-name` 格式来指定使用哪个 LoRA：

```bash
# 调用 medical-lora 适配器
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct@medical-lora",
        "messages": [
            {"role": "user", "content": "患者出现发热、咳嗽症状三天，请给出初步诊断建议。"}
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }'
```

```bash
# 切换到 legal-lora（同一个请求，换个 model 名即可）
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct@legal-lora",
        "messages": [
            {"role": "user", "content": "分析以下合同条款的法律风险："}
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }'
```

### 2.2 Python SDK 完整示例

比如下面的程序封装了一个支持多 LoRA 切换的客户端：

```python
import time
from openai import OpenAI
from typing import Optional


class LoraClient:
    """vLLM LoRA 多适配器客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 api_key: str = "not-needed",
                 base_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.base_model = base_model
    
    def chat(self, lora_name: str, messages: list,
              max_tokens: int = 512, temperature: float = 0.7,
              stream: bool = False, **kwargs) -> dict:
        """使用指定 LoRA 进行对话"""
        
        model_id = f"{self.base_model}@{lora_name}"
        
        start = time.time()
        
        if stream:
            return self._stream_chat(model_id, messages, max_tokens, 
                                      temperature, **kwargs)
        
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        elapsed = time.time() - start
        
        return {
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "latency_s": elapsed,
            "lora": lora_name,
        }
    
    def _stream_chat(self, model_id, messages, max_tokens, 
                      temperature, **kwargs):
        """流式输出"""
        stream = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        
        full_content = []
        ttft = None
        
        for chunk in stream:
            if not ttft:
                ttft = time.time()
            
            delta = chunk.choices[0].delta.content
            if delta:
                full_content.append(delta)
                yield delta
        
        yield {
            "__metrics__": {
                "ttft_ms": (ttft and (time.time() - ttft) * 1000) or 0,
                "total_chars": sum(len(c) for c in full_content),
            }
        }
    
    def list_available_loras(self) -> list[str]:
        """列出所有可用的 LoRA"""
        try:
            response = self.client.get("/v1/lora")
            return [lora["id"] for lora in response.get("data", [])]
        except:
            return []


def multi_lora_demo():
    """多 LoRA 演示"""
    
    client = LoraClient()
    
    test_cases = [
        ("medical-lora", "患者主诉头痛伴恶心呕吐，体温38.5°C，请分析可能的原因。"),
        ("legal-lora", "甲方逾期未支付货款超过30天，乙方可以采取哪些法律措施？"),
        ("code-assistant-lora", "实现一个Python装饰器来记录函数执行时间和参数。"),
        ("creative-writing-lora", "以第一人称写一段关于星际旅行的描写。"),
    ]
    
    print("=" * 70)
    print("多 LoRA 适配器演示")
    print("=" * 70)
    
    for lora_name, prompt in test_cases:
        result = client.chat(
            lora_name=lora_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.3,
        )
        
        print(f"\n[{lora_name}]")
        print(f"  Prompt: {prompt[:60]}...")
        print(f"  Response: {result['content'][:150]}...")
        print(f"  Tokens: {result['usage']['total_tokens']} "
              f"({result['usage']['prompt_tokens']}+"
              f"{result['usage']['completion_tokens']})")
        print(f"  Latency: {result['latency_s']:.2f}s")

multi_lora_demo()
```

## 三、动态管理 LoRA

### 3.1 运行时注册新 LoRA

无需重启服务，通过 API 动态添加 LoRA：

```python
import requests
import json

def register_lora(lora_name: str, lora_path: str, base_url: str = "http://localhost:8000"):
    """运行时注册新的 LoRA 适配器"""
    
    response = requests.post(
        f"{base_url}/v1/lora/register",
        json={
            "lora_name": lora_name,
            "lora_path": lora_path,
        },
        )
    
    data = response.json()
    print(f"[注册] {lora_name}")
    print(f"  状态码: {response.status_code}")
    print(f"  响应: {json.dumps(data, indent=2)}")
    
    return response.status_code == 200


# 注册新 LoRA
register_lora(
    lora_name="new-domain-lora",
    lora_path="/models/adapters/new-domain-v1"
)
```

### 3.2 卸载 LoRA 释放显存

```python
def unregister_lora(lora_name: str, base_url: str = "http://localhost:8000"):
    """卸载 LoRA 适配器以释放显存"""
    
    response = requests.delete(f"{base_url}/v1/lora/{lora_name}")
    
    print(f"[卸载] {lora_name} → {response.status_code}")
    return response.status_code == 200


unregister_lora("old-unused-lora")
```

### 3.3 查看已加载的 LoRA

```python
def list_loaded_loras(base_url: str = "http://localhost:8000"):
    """查看当前已加载的所有 LoRA"""
    
    response = requests.get(f"{base_url}/v1/lora")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n[已加载的 LoRA] 共 {len(data.get('data', []))} 个\n")
        
        for lora in data.get("data", []):
            print(f"  📌 {lora['id']}")
            if 'path' in lora:
                print(f"     路径: {lora['path']}")
    else:
        print(f"[错误] 无法获取 LoRA 列表 (HTTP {response.status_code})")


list_loaded_loras()
```

## 四、多租户场景实战

### 4.1 多租户路由器

```python
from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI
import time
import hashlib

app = FastAPI(title="Multi-Tenant LLM Gateway")

VLLM_BASE_URL = "http://localhost:8000/v1"

# 租户 → LoRA 映射表
TENANT_LORA_MAP = {
    "tenant_acme_corp": "acme-finance-lora",
    "tenant_hospital_abc": "medical-clinical-lora",
    "tenant_law_firm_xyz": "legal-contract-lora",
    "tenant_edu_platform": "education-tutoring-lora",
}

# API Key 验证 (简化版)
TENANT_KEYS = {
    "sk-acme-12345": "tenant_acme_corp",
    "sk-hospital-67890": "tenant_hospital_abc",
    "sk-law-abcde": "tenant_law_firm_xyz",
    "sk-edu-fghij": "tenant_edu_platform",
}


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    """多租户代理：根据 API Key 自动选择 LoRA"""
    
    body = await request.json()
    
    auth_header = request.headers.get("authorization", "")
    api_key = auth_header.replace("Bearer ", "") if auth_header else ""
    
    tenant = TENANT_KEYS.get(api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    lora_name = TENANT_LORA_MAP.get(tenant)
    if not lora_name:
        raise HTTPException(status_code=404, detail="No LoRA configured for this tenant")
    
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="internal")
    
    model_id = f"meta-llama/Llama-3.1-8B-Instruct@{lora_name}"
    
    start = time.time()
    response = await client.chat.completions.acreate(
        model=model_id,
        messages=body.get("messages", []),
        max_tokens=body.get("max_tokens", 512),
        temperature=body.get("temperature", 0.7),
        stream=body.get("stream", False),
    )
    latency = time.time() - start
    
    result = response.model_dump() if hasattr(response, 'model_dump') else {}
    result["metadata"] = {
        "tenant": tenant,
        "lora_used": lora_name,
        "latency_s": round(latency, 3),
    }
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

### 4.2 A/B 测试框架

```python
import asyncio
from openai import AsyncOpenAI
import json


class LoraABTester:
    """LoRA A/B 测试工具"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.client = AsyncOpenAI(base_url=base_url, api_key="test")
        self.base_model = "meta-llama/Llama-3.1-8B-Instruct"
    
    async def compare_loras(self, prompt: str, lora_a: str, lora_b: str,
                           system_prompt: str = None, **kwargs) -> dict:
        """对比两个 LoRA 在同一 prompt 上的表现"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        common_params = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.0),
        }
        
        # 并发调用两个 LoRA
        r_a, r_b = await asyncio.gather(
            self.client.chat.completions.create(
                model=f"{self.base_model}@{lora_a}", **common_params
            ),
            self.client.chat.completions.create(
                model=f"{self.base_model}@{lora_b}", **common_params
            ),
        )
        
        return {
            "prompt": prompt,
            "lora_a": {
                "name": lora_a,
                "response": r_a.choices[0].message.content,
                "tokens": r_a.usage.total_tokens,
            },
            "lora_b": {
                "name": lora_b,
                "response": r_b.choices[0].message.content,
                "tokens": r_b.usage.total_tokens,
            },
        }


async def run_ab_test():
    """运行 A/B 测试"""
    
    tester = LoraABTester()
    
    prompts = [
        "解释量子纠缠现象。",
        "分析合同中的不可抗力条款。",
        "诊断患者持续高热的可能原因。",
    ]
    
    results = []
    for prompt in prompts:
        result = await tester.compare_loras(
            prompt=prompt,
            lora_a="general-science-lora",
            lora_b="domain-expert-lora",
            system_prompt="你是专家助手，请详细回答。",
        )
        results.append(result)
        
        print(f"\nQ: {prompt}")
        print(f"A ({result['lora_a']['name']}): {result['lora_a']['response'][:120]}...")
        print(f"B ({result['lora_b']['name']}): {result['lora_b']['response'][:120]}...")
    
    with open("./ab_test_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


asyncio.run(run_ab_test())
```

---

## 五、总结

本节完成了 LoRA 服务化的全部实践操作：

| 操作 | 方式 | 说明 |
|------|------|------|
| **启动 LoRA 服务** | `--enable-lora --lora-modules name=path` | 启动时预加载 |
| **调用指定 LoRA** | `model="base@lora-name"` | API 层面零侵入切换 |
| **动态注册** | `POST /v1/lora/register` | 运行时添加新适配器 |
| **动态卸载** | `DELETE /v1/lora/{name}` | 释放显存 |
| **查询列表** | `GET /v1/lora` | 查看已加载的 LoRA |
| **多租户路由** | API Key → Tenant → LoRA 映射 | SaaS 架构标准模式 |
| **A/B 测试** | 并发调用不同 LoRA 对比效果 | 评估适配器质量 |

**核心要点回顾**：

1. **LoRA 切换对客户端完全透明**——只需改 `model` 字段为 `base@lora` 格式
2. **热注册/卸载是生产环境必备能力**——不需要重启服务就能更新模型
3. **多租户架构 = API Key 认证 + LoRA 路由**——一个基础模型服务 N 个客户
4. **A/B 测试是评估 LoRA 质量的科学方法**——不要凭感觉判断哪个更好
5. **`max-loras` 和 `max-lora-rank` 是两个最重要的调参**——决定了能同时服务多少客户

下一节我们将学习 **LoRA 最佳实践与性能优化**。
