# Adapter 合并与切换：多任务 PEFT 的生产实践

## 这一节讲什么？

在实际生产环境中，你经常会遇到这样的场景：**同一个基座模型需要服务多个不同的下游任务或领域**。比如一个 LLaMA-7B 模型，你可能需要它同时具备医疗咨询、法律问答、代码生成、创意写作等能力。

如果为每个任务都训练一个完整的 Full Fine-Tuning 模型，存储和部署成本会呈线性增长。但 PEFT 给了我们一个优雅的解决方案：**只保存每个任务的轻量级适配器（Adapter），在推理时动态切换或合并**。

这一节我们将深入探讨：
1. **LoRA 权重合并的原理与实现**——`merge_and_unload()` 背后发生了什么
2. **多 Adapter 动态切换**——一套基座 + N 个适配器 = N 个专家
3. **Adapter 的增删改查管理**——类似 Git 的版本控制思维
4. **生产部署架构设计**——从单机到分布式的完整方案

---

## 一、权重合并（Merge）深度解析

## 1.1 为什么要合并？

在理解合并之前，先搞清楚 LoRA 在推理时的两种模式：

```
模式一: 分离式推理 (未合并)
─────────────────────────────
输入 X → [原始 W] → 原始输出 ──┐
                                ├──→ (+) ──→ 最终输出
输入 X → [LoRA: B@A] → ΔW·X ──┘

开销: 每次前向传播多一次 B@A 矩阵乘法
延迟: 增加 ~5-15% (取决于 rank 和 hidden_size)

模式二: 合并式推理 (已合并)
─────────────────────────────
输入 X → [W' = W + B@A] ──→ 最终输出

开销: 零额外计算 (和普通模型完全一样)
延迟: 无增加
```

对于在线推理服务来说，**延迟就是生命线**。哪怕 50ms 的额外延迟在高并发场景下也会导致吞吐量显著下降。因此，生产环境几乎总是选择**合并后推理**。

## 1.2 合并的数学原理

LoRA 的合并过程在数学上非常简单：

$$W_{merged} = W_{original} + \frac{\alpha}{r} \times B \times A^T$$

其中：
- $W_{original}$ 是预训练模型的原始权重矩阵（冻结状态）
- $A$ 是 LoRA 的下投影矩阵 $(d_{in} \times r)$
- $B$ 是 LoRA 的上投影矩阵 $(r \times d_{out})$
- $\alpha/r$ 是缩放因子

```python
import torch
import torch.nn as nn

def manual_lora_merge():
    """手动演示 LoRA 权重合并的过程"""

    print("=" * 65)
    print("LoRA 权重合并原理演示")
    print("=" * 65)

    # 模拟一个预训练的线性层
    torch.manual_seed(42)
    d_in, d_out, r = 768, 768, 8

    W_original = nn.Parameter(torch.randn(d_out, d_in) * 0.02)

    # LoRA 的两个低秩矩阵
    A = nn.Parameter(torch.randn(d_in, r) * (1.0 / (r ** 0.5)))
    B = nn.Parameter(torch.zeros(r, d_out))  # 初始为零

    # 模拟训练后的状态 (给 A 和 B 赋一些非零值)
    with torch.no_grad():
        A.copy_(torch.randn(d_in, r) * 0.1)
        B.copy_(torch.randn(r, d_out) * 0.1)

    alpha = 16
    scaling = alpha / r

    # 方法一: 分离式计算 (未合并)
    x = torch.randn(2, 128, d_in)
    original_out = x @ W_original.T
    lora_delta = (x @ A) @ B * scaling
    separated_out = original_out + lora_delta

    # 方法二: 合并后计算
    with torch.no_grad():
        delta_W = (B.T @ A.T) * scaling   # ΔW = B^T @ A^T * scaling
        W_merged = W_original.data + delta_W

    merged_out = x @ W_merged.T

    # 验证两种方法的结果一致
    diff = (separated_out - merged_out).abs().max().item()

    print(f"\n原始权重形状: {W_original.shape}")
    print(f"LoRA-A 形状:    {A.shape}")
    print(f"LoRA-B 形状:    {B.shape}")
    print(f"缩放因子 α/r:   {scaling:.2f}")

    print(f"\n分离式 vs 合并式 输出差异: {diff:.2e}")
    print(f"✅ 差异接近零! 两种方式数学等价")

    print(f"\n💡 合并后的好处:")
    print(f"   - 推理时只需一次矩阵乘法 (而非两次)")
    print(f"   - 可以导出为 ONNX/TensorRT 等格式")
    print(f"   - 不需要 peft 库即可加载和使用")

manual_lora_merge()
```

## 1.3 使用 HF peft 进行合并

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class AdapterManager:
    """
    Adapter 管理器: 合并、切换、导出的统一接口
    """

    def __init__(self, base_model_path):
        self.base_model_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.merged_models_cache = {}

    def load_base_model(self):
        """加载基座模型"""
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        return self.base_model

    def merge_adapter(self, adapter_path, output_path=None):
        """
        合并单个 Adapter 到基座模型

        Args:
            adapter_path: LoRA 适配器的路径
            output_path: 合并后模型的保存路径 (可选)
        """
        if not hasattr(self, 'base_model'):
            self.load_base_model()

        print(f"\n🔧 正在合并 Adapter: {adapter_path}")

        # 加载 PeftModel
        model = PeftModel.from_pretrained(self.base_model, adapter_path)

        # 打印合并前的信息
        model.print_trainable_parameters()

        # 执行合并! 这是最关键的一步
        merged_model = model.merge_and_unload()

        # 验证合并结果
        print(f"\n✅ 合并完成!")
        print(f"   合并后类型: {type(merged_model)}")
        print(f"   是否还是 PeftModel: {isinstance(merged_model, PeftModel)}")

        # 保存合并后的模型
        if output_path:
            merged_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            print(f"   💾 已保存到: {output_path}")

        return merged_model

    def batch_merge_adapters(self, adapter_dict):
        """
        批量合并多个 Adapter

        Args:
            adapter_dict: {
                "medical": "./adapters/lora_medical",
                "legal": "./adapters/lora_legal",
                ...
            }
        """
        results = {}
        for name, path in adapter_dict.items():
            output_path = f"./merged_models/{name}"
            merged = self.merge_adapter(path, output_path=output_path)
            results[name] = merged
        return results


# 使用示例
def demo_merge_workflow():
    """演示完整的工作流程"""

    manager = AdapterManager("meta-llama/Llama-2-7b-hf")
    manager.load_base_model()

    # 合并单个 adapter
    merged = manager.merge_adapter(
        adapter_path="./my_lora_finetune",
        output_path="./merged_llama2_7b_chat",
    )

    # 用合并后的模型进行推理 (不需要 peft 库!)
    inputs = manager.tokenizer("你好，请介绍一下自己", return_tensors="pt").to(merged.device)
    outputs = merged.generate(**inputs, max_new_tokens=256)
    response = manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🤖 模型回复:\n{response}")


if __name__ == "__main__":
    demo_merge_workflow()
```

## 1.4 合并中的常见问题

```python
def merge_gotchas():
    """合并过程中的常见坑"""

    gotchas = [
        {
            "问题": "合并后模型体积变大",
            "原因": "peft 默认保存的是全精度 FP32，即使原模型是 BF16",
            "解决": "合并前确认 base_model 的 dtype; 或手动转换 dtype 后再保存",
        },
        {
            "问题": "合并后效果下降",
            "原因": "可能是精度截断 (FP32→BF16/FP16); "
                   "或缩放因子 α/r 设置不合理导致数值溢出",
            "解决": "保持 FP32 合并后再转目标精度; 检查 alpha/ratio",
        },
        {
            "问题": "merge_and_unload() 报错",
            "原因": "某些特殊层不支持合并 (如 embedding 层加了 LoRA)",
            "解决": "检查 target_modules 是否包含不支持的模块; "
                   "或使用 safe_merge=True 参数",
        },
        {
            "问题": "想保留未合并版本用于继续训练",
            "原因": "merge_and_unload() 会销毁 PeftModel 结构",
            "解决": "在合并之前先 save_pretrained() 保存一份原始 adapter",
        },
        {
            "问题": "QLoRA 合并后精度异常",
            "原因": "4-bit 量化权重合并到 FP16/BF16 时有精度损失",
            "解决": "使用 dequantize_bnb_linear() 先反量化再合并; "
                   "或接受轻微精度损失",
        },
    ]

    print("=" * 75)
    print("LoRA 合并常见问题排查")
    print("=" * 75)

    for g in gotchas:
        print(f"\n⚠️  {g['问题']}")
        print(f"   原因: {g['原因']}")
        print(f"   解决: {g['解决']}")

merge_gotchas()
```

---

## 二、多 Adapter 动态切换

## 2.1 场景描述

假设你有一个智能客服系统，基于同一个 LLaMA-7B 模型，需要处理以下类型的请求：

```
用户: "我的订单什么时候发货?"     → 切换到 [电商] Adapter
用户: "感冒了吃什么药好?"       → 切换到 [医疗] Adapter  
用户: "帮我写一段 Python 代码"   → 切换到 [编程] Adapter
用户: "今天天气怎么样?"         → 切换到 [通用] Adapter (或直接用基座)
```

如果每种场景都加载一个独立的全量模型，4 个场景就需要 4 × ~14GB (FP16) = **56GB 显存**。但用多 Adapter 切换方案，只需要 **1 个基座 (~14GB) + 4 个轻量 Adapter (~各 10MB)** ≈ **14.04GB**。

## 2.2 完整的多 Adapter 切换实现

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Optional
import time


class MultiAdapterRouter:
    """
    多 Adapter 路由器
    支持动态加载、切换、热更新 Adapter
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        default_adapter: Optional[str] = None,
    ):
        self.base_model_name = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

        print(f"📦 加载基座模型: {base_model_name_or_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 将基座模型包装为 PeftModel
        self.model = PeftModel(self.base_model, self.base_model)

        # 已加载的 adapters 字典
        self.adapters: Dict[str, str] = {}  # name -> path
        self.current_adapter = default_adapter
        self._switch_count = 0
        self._latencies = []

        if default_adapter and default_adapter in self.adapters:
            self.set_adapter(default_adapter)

    def add_adapter(self, name: str, path: str):
        """添加一个新的 Adapter"""
        print(f"\n➕ 添加 Adapter: {name} ← {path}")
        self.model.load_adapter(path, adapter_name=name)
        self.adapters[name] = path
        print(f"   ✅ 当前可用 Adapters: {list(self.adapters.keys())}")

    def remove_adapter(self, name: str):
        """移除一个 Adapter"""
        if name in self.adapters:
            self.model.delete_adapter(name)
            del self.adapters[name]
            print(f"🗑️  已移除 Adapter: {name}")
            if self.current_adapter == name:
                self.current_adapter = None

    def set_adapter(self, name: str):
        """切换到指定的 Adapter"""
        start_time = time.time()

        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' 不存在。可用的: {list(self.adapters.keys())}")

        self.model.set_adapter(name)
        self.current_adapter = name
        self._switch_count += 1

        latency_ms = (time.time() - start_time) * 1000
        self._latencies.append(latency_ms)

        print(f"🔄 切换到 [{name}] ({latency_ms:.2f}ms)")

    def auto_route(self, user_input: str, routing_rules: dict) -> str:
        """
        根据输入内容自动路由到合适的 Adapter

        Args:
            user_input: 用户输入文本
            routing_rules: {"关键词列表": "adapter_name", ...}

        Returns:
            路由到的 adapter 名称
        """
        input_lower = user_input.lower()

        for keywords, adapter_name in routing_rules.items():
            if isinstance(keywords, str):
                keywords = [keywords]
            if any(kw in input_lower for kw in keywords):
                if self.current_adapter != adapter_name:
                    self.set_adapter(adapter_name)
                return adapter_name

        # 默认使用当前 adapter 或基座
        return self.current_adapter or "base"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """生成回复"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 移除 prompt 部分 (如果有重复)
        if prompt in response:
            response = response[len(prompt):].strip()

        return response

    def stats(self) -> dict:
        """返回路由器的统计信息"""
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0
        return {
            "total_switches": self._switch_count,
            "avg_switch_latency_ms": f"{avg_latency:.2f}",
            "available_adapters": list(self.adapters.keys()),
            "current_adapter": self.current_adapter,
            "base_model": self.base_model_name,
        }


def demo_multi_adapter_routing():
    """演示多 Adapter 路由"""

    print("=" * 70)
    print("多 Adapter 动态路由演示")
    print("=" * 70)

    router = MultiAdapterRouter(
        base_model_name_or_path="gpt2",  # 用 GPT-2 做演示 (小模型快速)
        default_adapter=None,
    )

    # 添加虚拟的 adapter (实际使用时替换为真实路径)
    # router.add_adapter("ecommerce", "./adapters/lora_ecommerce")
    # router.add_adapter("medical", "./adapters/lora_medical")
    # router.add_adapter("coding", "./adapters/lora_coding")

    # 模拟添加 (仅展示 API)
    print("\n[模拟] 添加三个 Adapter:")
    mock_adapters = [
        ("ecommerce", "./adapters/lora_ecommerce"),
        ("medical", "./adapters/lora_medical"),
        ("coding", "./adapters/lora_coding"),
    ]
    for name, path in mock_adapters:
        print(f"  ➕ router.add_adapter('{name}', '{path}')")

    # 定义路由规则
    routing_rules = {
        ["订单", "发货", "退款", "物流", "购物车"]: "ecommerce",
        ["感冒", "药", "症状", "治疗", "医生"]: "medical",
        ["代码", "python", "函数", "bug", "编程"]: "coding",
    }

    # 测试路由
    test_inputs = [
        "我的订单什么时候发货?",
        "感冒了应该吃什么药?",
        "帮我写一个 Python 快速排序函数",
        "今天天气怎么样?",  # 无匹配, 保持默认
    ]

    print("\n🚦 自动路由测试:")
    for user_input in test_inputs:
        matched = router.auto_route(user_input, routing_rules)
        print(f"  输入: \"{user_input}\"")
        print(f"  → 路由到: [{matched}]")
        print()

    # 统计信息
    print("📊 路由器统计:")
    for k, v in router.stats().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demo_multi_adapter_routing()
```

## 2.3 切换的性能分析

```python
def analyze_switching_performance():
    """分析 Adapter 切换的性能特征"""

    print("=" * 65)
    print("Adapter 切换性能分析")
    print("=" * 65)

    analysis = {
        "操作": ["set_adapter()", "add_adapter()", "remove_adapter()"],
        "耗时": ["< 1ms", "~100-500ms (首次)", "< 1ms"],
        "是否阻塞推理": ["否", "是 (需加载权重)", "否"],
        "显存变化": ["无", "+adapter 大小", "-adapter 大小"],
        "线程安全": ["是", "建议加锁", "是"],
    }

    print(f"\n{'操作':<25} {'耗时':<18} {'阻塞':<12} {'显存':<15} {'线程安全'}")
    print("-" * 85)
    for i in range(len(analysis["操作"])):
        row = [analysis[k][i] for k in analysis.keys()]
        print(f"{row[0]:<25} {row[1]:<18} {row[2]:<12} {row[3]:<15} {row[4]}")

    print("\n💡 生产优化建议:")
    print("   • 启动时预加载所有常用 Adapter (冷启动成本)")
    print("   • 使用 LRU 缓存策略管理内存中的 Adapter 数量")
    print("   • 高频 Adapter 常驻内存, 低频 Adapter 按需加载")
    print("   • set_adapter() 本身极快, 不影响推理吞吐")

analyze_switching_performance()
```

---

## 三、Adapter 版本管理与 Hub 社区

## 3.1 类似 Git 的 Adapter 管理

peft 的 Adapter 管理理念类似于 Git 的分支管理——你可以把每个 Adapter 看作一个"功能分支"，随时创建、切换、删除：

```python
def git_like_adapter_management():
    """Git 风格的 Adapter 管理"""

    commands = [
        ("git branch <name>", "model.add_adapter(name, path)", "创建新 Adapter"),
        ("git checkout <name>", "model.set_adapter(name)", "切换到指定 Adapter"),
        ("git branch -d <name>", "model.delete_adapter(name)", "删除 Adapter"),
        ("git push origin <name>", "model.push_to_hub(repo_id)", "上传到 HuggingFace Hub"),
        ("git pull origin <name>", "PeftModel.from_pretrained(base, hub_id)", "从 Hub 下载"),
        ("git merge", "model.merge_and_unload()", "合并到主分支 (基座)"),
        ("git stash", "model.save_pretrained(local_path)", "本地暂存"),
    ]

    print("=" * 75)
    print("Adapter 管理 ↔ Git 操作对照表")
    print("=" * 75)

    print(f"\n{'Git 命令':<28} {'PEFT 对应操作':<45} {'说明'}")
    print("-" * 95)
    for git_cmd, peft_cmd, desc in commands:
        print(f"{git_cmd:<28} {peft_cmd:<45} {desc}")

git_like_adapter_management()
```

## 3.2 从 HuggingFace Hub 发现和加载社区 Adapter

HuggingFace Hub 上有大量社区共享的 LoRA Adapter，可以直接加载使用：

```python
from huggingface_hub import list_models

def discover_community_adapters():
    """发现 HuggingFace Hub 上的社区 Adapter"""

    print("=" * 65)
    print("发现社区 Adapter")
    print("=" * 65)

    # 搜索 LLaMA 相关的 LoRA adapter
    adapters = list_models(
        filter="lora",
        author="tloen",
        sort="downloads",
        direction=-1,
        limit=10,
    )

    print(f"\n热门社区 Adapter (按下载量排序):")
    for i, model in enumerate(adapters, 1):
        print(f"  {i:>2}. {model.id}")
        print(f"      下载: {model.downloads:,} | 喜欢: {model.likes:,}")
        print()


def load_hub_adapter_demo():
    """从 Hub 直接加载 Adapter"""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    # 加载基座
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 从 Hub 加载社区 Adapter (以 alpaca-lora 为例)
    print("正在从 HuggingFace Hub 加载 Adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        "tloen/alpaca-lora-7b",  # 社区 Adapter 的 repo ID
    )

    model.set_adapter("default")
    print("✅ 社区 Adapter 加载完成!")

    return model


if __name__ == "__main__":
    discover_community_adapters()
```

---

## 四、生产部署架构

## 4.1 单机多 Adapter 服务

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn


class ProductionAdapterService:
    """
    生产级多 Adapter 推理服务
    基于 FastAPI 封装
    """

    def __init__(self, config: dict):
        self.config = config
        self.router = None  # 延迟初始化
        self._initialized = False

    async def initialize(self):
        """懒初始化: 首次请求时加载模型"""
        if self._initialized:
            return

        from .multi_adapter_router import MultiAdapterRouter

        self.router = MultiAdapterRouter(
            base_model_name_or_path=self.config["base_model"],
        )

        # 预加载所有配置的 Adapter
        for name, path in self.config.get("adapters", {}).items():
            self.router.add_adapter(name, path)

        # 设置默认 Adapter
        default = self.config.get("default_adapter")
        if default:
            self.router.set_adapter(default)

        self._initialized = True
        print("✅ Adapter 服务初始化完成")


app = FastAPI(title="Multi-Adapter Inference Service")
service = ProductionAdapterService({
    "base_model": "meta-llama/Llama-2-7b-hf",
    "adapters": {
        "medical": "/models/adapters/lora_medical",
        "legal": "/models/adapters/lora_legal",
        "code": "/models/adapters/lora_code",
    },
    "default_adapter": "general",
})


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    adapter_override: Optional[str] = None  # 手动指定 adapter


class GenerateResponse(BaseModel):
    response: str
    adapter_used: str
    latency_ms: float


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """生成端点"""
    import time
    start = time.time()

    await service.initialize()

    # 如果手动指定了 adapter, 强制切换
    if req.adapter_override:
        service.router.set_adapter(req.adapter_override)

    response = service.router.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )

    latency = (time.time() - start) * 1000

    return GenerateResponse(
        response=response,
        adapter_used=service.router.current_adapter or "base",
        latency_ms=round(latency, 2),
    )


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy" if service._initialized else "initializing",
        "adapters_available": list(service.router.adapters.keys()) if service.router else [],
        "current_adapter": service.router.current_adapter if service.router else None,
    }


@app.get("/adapters")
async def list_adapters():
    """列出所有可用 Adapter"""
    await service.initialize()
    return {
        "adapters": list(service.router.adapters.keys()),
        "current": service.router.current_adapter,
        "stats": service.router.stats(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 4.2 分布式多 Adapter 架构

当单台机器无法承载所有 Adapter 或需要更高并发时，可以采用分布式架构：

```
                    ┌─────────────┐
                    │  Load Balancer│
                    │  (Nginx/GSLB) │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌──────────┐ ┌──────────┐
     │ GPU Server 1 │ │GPU Svr 2 │ │GPU Svr N │
     │ Base+Medical │ │Base+Legal│ │Base+Code │
     │ (Adapter A)  │ │(Adapter B)│ │(Adapter C)│
     └──────────────┘ └──────────┘ └──────────┘
              │            │            │
              └────────────┴────────────┘
                           │
                    ┌──────▼───────┐
                    │ Shared Storage│
                    │ (Adapter Repo)│
                    └──────────────┘
```

```python
def distributed_architecture_guide():
    """分布式多 Adapter 架构指南"""

    strategies = [
        {
            "策略": "Adapter 分片 (Sharding)",
            "描述": "每台 GPU 服务器加载基座 + 一个子集的 Adapter,"
                   "通过网关根据请求类型路由到对应服务器",
            "优点": "每台服务器显存压力小",
            "缺点": "需要额外的路由逻辑; 基座模型被重复加载",
            "适用": "Adapter 数量多 (>10), 每个 Adapter 都很常用",
        },
        {
            "策略": "集中式 + Cache",
            "描述": "一台服务器加载所有 Adapter, 其他服务器做纯推理;"
                   "或所有服务器都加载基座, Adapter 通过高速网络共享",
            "优点": "管理简单; 切换快",
            "缺点": "单点瓶颈; 网络延迟",
            "适用": "Adapter 数量少 (<5), 单机能容纳",
        },
        {
            "策略": "混合部署 (Hybrid)",
            "描述": "高频 Adapter 每台服务器都常驻,"
                   "低频 Adapter 只在一台服务器上按需加载",
            "优点": "平衡了性能和资源利用率",
            "缺点": "实现复杂度高",
            "适用": "大多数生产环境",
        },
    ]

    print("=" * 80)
    print("分布式多 Adapter 部署策略")
    print("=" * 80)

    for s in strategies:
        print(f"\n📐 策略: {s['策略']}")
        print(f"   描述: {s['描述']}")
        print(f"   ✅ 优点: {s['优点']}")
        print(f"   ❌ 缺点: {s['缺点']}")
        print(f"   🎯 适用: {s['适用']}")

distributed_architecture_guide()
```

---

## 五、监控与运维

## 5.1 关键指标监控

```python
class AdapterMonitor:
    """Adapter 运行监控"""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_per_adapter": {},
            "errors_total": 0,
            "switch_count": 0,
            "latencies_p50": [],
            "latencies_p99": [],
            "adapter_load_times": {},
            "memory_usage_gb": {},
        }

    def record_request(self, adapter_name: str, latency_ms: float, success: bool):
        """记录一次请求"""
        self.metrics["requests_total"] += 1
        self.metrics["latencies_p50"].append(latency_ms)

        if adapter_name not in self.metrics["requests_per_adapter"]:
            self.metrics["requests_per_adapter"][adapter_name] = 0
        self.metrics["requests_per_adapter"][adapter_name] += 1

        if not success:
            self.metrics["errors_total"] += 1

    def record_switch(self, adapter_name: str, switch_time_ms: float):
        """记录一次 Adapter 切换"""
        self.metrics["switch_count"] += 1
        self.metrics["adapter_load_times"][adapter_name] = switch_time_ms

    def get_dashboard(self) -> dict:
        """生成监控面板数据"""
        latencies = sorted(self.metrics["latencies_p50"])
        n = len(latencies)

        return {
            "total_requests": self.metrics["requests_total"],
            "error_rate": (
                f"{self.metrics['errors_total']/self.metrics['requests_total']*100:.2f}%"
                if self.metrics["requests_total"] > 0 else "N/A"
            ),
            "p50_latency_ms": latencies[n//2] if n > 0 else 0,
            "p99_latency_ms": latencies[int(n*0.99)] if n > 0 else 0,
            "total_switches": self.metrics["switch_count"],
            "adapter_distribution": self.metrics["requests_per_adapter"],
            "active_adapters": len(self.metrics["requests_per_adapter"]),
        }


def demo_monitoring():
    """演示监控功能"""

    monitor = AdapterMonitor()

    # 模拟一些请求
    scenarios = [
        ("medical", 150.2, True),
        ("legal", 180.5, True),
        ("code", 120.3, True),
        ("medical", 145.8, True),
        ("legal", 200.1, False),
    ]

    for adapter, latency, success in scenarios:
        monitor.record_request(adapter, latency, success)

    dashboard = monitor.get_dashboard()

    print("=" * 60)
    print("Adapter 服务监控面板")
    print("=" * 60)
    for k, v in dashboard.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demo_monitoring()
```

---

## 六、本章小结

这一节我们深入学习了 PEFT 在生产环境中的关键操作——**Adapter 的合并、切换和管理**。核心要点回顾：

| 操作 | API | 耗时 | 生产注意事项 |
|------|-----|------|-------------|
| **合并** | `model.merge_and_unload()` | <1s | 先保存原始 adapter; QLoRA 注意反量化 |
| **切换** | `model.set_adapter(name)` | <1ms | 极快, 可在每个 request 前调用 |
| **添加** | `model.load_adapter(path, name)` | 100-500ms | 启动时预加载 |
| **删除** | `model.delete_adapter(name)` | <1ms | 确认不再需要 |
| **上传** | `model.push_to_hub(repo_id)` | 取决于大小 | 包含 Model Card |

**核心设计原则**：
1. **合并用于部署，分离用于迭代**——开发阶段保持分离方便调试，部署阶段合并消除延迟
2. **一套基座服务多个任务**——这是 PEFT 最大的商业价值所在
3. **预热 + 缓存**——启动时预加载常用 Adapter，避免首次请求的冷启动延迟
4. **监控每个 Adapter 的调用量和质量**——及时发现退化的 Adapter 并重新训练

下一节我们将总结 **PEFT 的生产最佳实践**，包括性能调优、故障排查和持续优化的完整指南。
