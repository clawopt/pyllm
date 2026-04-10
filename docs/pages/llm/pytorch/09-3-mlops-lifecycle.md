# 9.3 模型生命周期管理与 MLOps 实践

前两节我们解决了"怎么把模型变成服务"和"怎么监控服务运行状态"两个问题。但一个 LLM 模型从诞生到退役的完整生命周期远不止这些——它还包括版本管理、A/B 测试、灰度发布、持续评估、回滚策略以及模型注册中心（Model Registry）的建设。这一节我们将系统化地讨论这些话题，并给出一个完整的 **MLOps（Machine Learning Operations）** 实践框架，让模型的整个生命周期的每一个环节都可控、可追溯、可自动化。

## 模型生命周期的全貌

一个典型的生产级 LLM 模型会经历以下阶段：

```
┌──────────────────────────────────────────────────────┐
│              MODEL LIFECYCLE                       │
│                                                      │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐       │
│  │  训练    │ →  │  评估    │ →  │  注册     │       │
│  │ Train   │    │ Evaluate │    │ Register │       │
│  └─────────┘    └─────────┘    └──────────┘       │
│       ↓              ↓               ↓              │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐       │
│  │  版本化  │    │  打标签  │    │  部署     │       │
│  │ Version │    │ Tagging  │    │ Deploy   │       │
│  └─────────┘    └─────────┘    └──────────┘       │
│       ↓              ↓               ↓              │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐       │
│  │  A/B测试 │ →  │ 灰度发布  │ →  │  监控     │       │
│  │ A/B Test│    │ Canary  │    │ Monitor  │       │
│  └─────────┘    └─────────┘    └──────────┘       │
│       ↓              ↓               ↓              │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐       │
│  │  全量发布│ ←  │ 回滚/降级│ ←  │  再训练   │       │
│  │ Promote  │    │ Rollback │    │ Retrain  │       │
│  └─────────┘    └─────────┘    └──────────┘       │
│                                                      │
│  (循环: 当监控指标下降时触发再训练)                  │
└──────────────────────────────────────────────────────┘
```

每个环节都需要对应的工具和流程支撑。下面我们逐一实现。

## 模型注册中心（Model Registry）

**Model Registry** 是 MLOps 的核心基础设施——它是所有已训练模型的中央数据库，记录了每个版本的元数据、性能指标、关联的训练实验和部署状态。没有 Model Registry 的团队最终会陷入混乱：不知道线上跑的是哪个版本、某个 checkpoint 对应哪次超参数配置、出了问题该回滚到哪个版本。

### 基于 MLflow 的实现

MLflow 是 Databricks 开源的最流行的 ML 平台之一，它的 Model Registry 功能非常适合管理 LLM 模型：

```python
import mlflow
from mlflow.tracking import MlflowClient


class LLMModelRegistry:
    """基于 MLflow 的模型注册中心"""

    def __init__(self, registry_uri: str = "sqlite:///mlruns.db"):
        self.client = MlflowClient(registry_uri)
        self.experiment_name = "llm-finetuning"

    def register_model(
        self,
        model_path: str,
        model_name: str,
        version_desc: str,
        metrics: dict,
        hyperparams: dict,
        tags: dict = None,
    ):
        """注册一个新版本的模型"""
        
        run = mlflow.start_run(
            experiment_name=self.experiment_name,
            tags=tags or {},
        )
        
        # 记录超参数
        mlflow.log_params({
            "learning_rate": hyperparams.get("lr", "N/A"),
            "epochs": hyperparams.get("epochs", "N/A"),
            "batch_size": hyperparams.get("batch_size", "N/A"),
            "r": hyperparams.get("lora_r", "N/A"),
            "alpha": hyperparams.get("lora_alpha", "N/A"),
            "quantization": hyperparams.get("quantization", "none"),
            "base_model": hyperparams.get("base_model", "unknown"),
            "dataset": hyperparams.get("dataset", "unknown"),
            "train_tokens": metrics.get("total_train_tokens", 0),
            "gpu_hours": metrics.get("gpu_hours", 0),
        })
        
        # 记录指标
        mlflow.log_metrics({
            "val_loss": metrics.get("val_loss", float("inf")),
            "train_loss": metrics.get("final_train_loss", float("inf")),
            "perplexity": metrics.get("perplexity", float("inf")),
            "tokens_per_sec": metrics.get("inference_tps", 0),
            "model_size_gb": metrics.get("model_size_gb", 0),
            "bleu_score": metrics.get("bleu_score", 0),
            "human_eval": metrics.get("human_eval", 0),
        })
        
        # 记录模型文件
        mlflow.log_artifact(model_path, artifact_path="model")
        
        # 记录示例输出
        if metrics.get("sample_output"):
            mlflow.log_text(metrics["sample_output"], artifact_file="sample_output.txt")
        
        mlflow.end_run()
        
        # 注册到 Model Registry
        model_uri = mlflow.register_model(
            f"runs:{run.info.run_id}/artifacts/model",
            name=model_name,
            tags=tags or {},
        )
        
        # 添加版本描述和别名
        self.client.update_model_version(
            name=model_name,
            version=model_uri.version,
            description=version_desc,
        )
        
        print(f"✓ Model registered: {model_name} v{model_uri.version}")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Val Loss: {metrics.get('val_loss', 'N/A')}")
        
        return model_uri

    def list_versions(self, model_name: str):
        """列出模型的所有版本"""
        versions = self.client.search_model_versions(
            filter_string=f"name='{model_name}'"
        )
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"{'Version':>8s} | {'Stage':>10s} | {'Val Loss':>10s} | "
              f"{'Created':>20s} | {'Description'}")
        print("-" * 85)
        
        for v in versions:
            stage = v.get("current_stage", "None")
            desc = (v.get("description") or "")[:40]
            
            try:
                run = self.client.get_run(v["run_id"])
                val_loss = run.data.metrics.get("val_loss", "N/A")
                created = v["creation_timestamp"][:19]
            except:
                val_loss = "?"
                created = v.get("creation_timestamp", "?")[:19]
            
            marker = "→" if stage in ("Production", "Staging") else " "
            print(f"{v['version']:>8s} | {stage:>10s} | {str(val_loss):>10s} | "
                  f"{created:>20s} | {marker} {desc}")

    def transition_stage(self, model_name: str, version: str, stage: str):
        """切换模型阶段"""
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )
        print(f"✓ {model_name} v{version} → {stage}")


# 使用示例
registry = LLMModelRegistry()

registry.register_model(
    model_path="./output/qwen-lora-final",
    model_name="qwen-alpaca-v2",
    version_desc="LoRA r=16, trained on alpaca-cleaned (52K samples), 3 epochs",
    metrics={
        "val_loss": 1.2345,
        "final_train_loss": 0.8923,
        "perplexity": 12.34,
        "inference_tps": 45.6,
        "model_size_gb": 14.2,
        "sample_output": "人工智能是计算机科学的一个分支...",
    },
    hyperparams={
        "lr": "2e-4",
        "epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "lora_r": 16,
        "lora_alpha": 32,
        "quantization": "bf16",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "alpaca-cleaned-52k",
    },
    tags={"team": "nlp", "env": "prod"},
)

registry.list_versions("qwen-alpaca-v2")

# 输出示例:
# ============================================================
# Model: qwen-alpaca-v2
# ============================================================
# Version |     Stage |  Val Loss |           Created | Description
# -----------------------------------------------------------------------------
#      12 |  None     |    1.2345 | 2024-03-15 14:32:05 | LoRA r=16, trained on...
#      11 | Staging   |    1.2891 | 2024-03-14 09:15:22 | Previous version
#      10 | Production|    1.1567 | 2024-03-12 18:45:00 | Current production
```

## A/B 测试与灰度发布

当你有了一个新版本的模型后，不应该直接把它推到 100% 的流量上。正确的做法是先让一小部分流量访问新版本，确认没有问题后再逐步扩大比例。

### 流量路由实现

```python
import random
import hashlib
from enum import Enum


class RoutingRule(Enum):
    ALL_TO_NEW = "all_to_new"
    PERCENTAGE_BASED = "percentage_based"
    USER_HASH_BASED = "user_hash_based"


class TrafficRouter:
    """A/B 测试流量路由器"""

    def __init__(
        self,
        old_service,           # 当前生产版本的服务实例
        new_service,           # 新版本的服务实例
        rule: RoutingRule = RoutingRule.PERCENTAGE_BASED,
        new_percentage: float = 0.05,  # 初始只给 5% 流量
        seed: int = 42,
    ):
        self.old = old_service
        self.new = new_service
        self.rule = rule
        self.new_pct = new_percentage
        self.rng = random.Random(seed)

    def route(self, request, user_id: str = None):
        """根据规则决定请求发给哪个版本"""
        
        if self.rule == RoutingRule.ALL_TO_NEW:
            return self.new.generate(request)
        
        elif self.rule == RoutingRule.PERCENTAGE_BASED:
            if self.rng.random() < self.new_pct:
                return self.new.generate(request)
            else:
                return self.old.generate(request)
        
        elif self.rule == RoutingRule.USER_HASH_BASED:
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 10000
            if hash_val < self.new_pct * 100:
                return self.new.generate(request)
            else:
                return self.old.generate(request)
    
    def set_new_percentage(self, pct: float):
        self.new_pct = pct


router = TrafficRouter(production_v10, candidate_v11, new_percentage=0.05)


@app.post("/v1/chat/completions")
async def chat_with_ab_test(request: GenerateRequest):
    user_id = request.headers.get("X-User-ID", "anonymous")
    
    response = router.route(request, user_id=user_id)
    
    # 记录哪个版本处理了这个请求（用于分析）
    logger.info("request_routed", extra={
        "extra_data": {
            "version": "new" if router.new_pct > 0 else "old",
            "new_percentage": router.new_pct,
            "user_id": user_id[:8],
        }
    })
    
    return response
```

### 灰度发布流程

灰度发布（Canary Deployment）的标准操作流程：

```
Day 1:  5%  → 新版 (v11)
         95% → 旧版 (v10)
         ↓ 观察 24h：错误率、延迟、用户反馈
         
Day 2:  20% → 新版 (v11)  [指标正常]
         80% → 旧版 (v10)
         ↓ 观察 24h

Day 3:  50% → 新版 (v11)  [指标正常]
         50% → 旧版 (v10)
         ↓ 观察 24h

Day 4: 100% → 新版 (v11) [全部正常]
         0%  → 旧版 (v10)
         ↓ 正式完成发布

Day 5+: 持续监控
         如果发现异常：
         ↓ 立即回滚到 v10
         ↓ 排查原因
```

## 自动化 CI/CD 流水线

把以上所有步骤自动化就构成了完整的 **CI/CD（持续集成/持续部署）流水线**：

```yaml
# .github/workflows/train-and-deploy.yml — GitHub Actions 示例
name: LLM Training & Deployment Pipeline

on:
  push:
    branches: [main]
    paths: ['training/**', 'data/**']
  workflow_dispatch:
    inputs:
      deploy_target:
        description: 'Deployment target'
        required: false
        default: 'staging'
        type: choice
        options: [staging, canary, production]

jobs:
  train:
    runs-on: [self-hosted, linux, gpu]  # GPU runner
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: |
          pip install torch transformers peft bitsandbytes \
                      accelerate datasets wandb mlflow
          
      - name: Run training
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
        run: |
          python train.py --config configs/qwen_lora.yaml
          
      - name: Evaluate model
        run: |
          python evaluate.py --checkpoint output/best_model.pt \
                        --test-data data/test_alpaca.jsonl
          
      - name: Upload to Model Registry
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
        run: |
          python register_model.py \
            --model-path output/best_model \
            --experiment-name llm-finetuning \
            --metrics eval_results.json
          
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: trained-model
          path: output/

  deploy-staging:
    needs: train
    runs: ubuntu-latest
    if: github.event_name == 'push' || github.event.inputs.deploy_target == 'staging'
    
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: trained-model
          path: ./model
          
      - name: Build Docker image
        run: docker build -t llm-service:${{ github.sha }} .
        
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login \
            ${{ secrets.DOCKER_REGISTRY }} -u ${{ secrets.DOCKER_USER }} --password-stdin
          docker push ${{ secrets.DOCKER_REGISTRY }}/llm-service:${{ github.sha }}
          
      - name: Deploy to staging
        run: |
          kubectl set image deployment/llm-service \
            llm-service=${{ secrets.DOCKER_REGISTRY }}/llm-service:${{ github.sha }}

  deploy-production:
    needs: [train, deploy-staging]
    runs: ubuntu-latest
    if: github.event.inputs.deploy_target == 'production'
    environment: production
    
    steps:
      - name: Smoke test on staging
        run: |
          curl -f https://staging.api.example.com/health
          curl -X POST https://staging.api.example.com/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"prompt":"hello"}' | jq '.latency_ms < 2000'
          
      - name: Promote staging to production
        run: |
          kubectl patch deployment llm-service -p '{"spec":{"template":{"spec":{"containers":[{"name":"llm-service","image":"${{ secrets.DOCKER_REGISTRY }}/llm-service:'$(kubectl get deployment llm-service -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d: -f2)'"}]}}}}'
          
      - name: Verify production
        run: sleep 30 && curl -f https://api.example.com/health
```

这个流水线实现了：代码推送 → 自动训练（在 GPU runner 上）→ 自动评估 → 注册到 Model Registry → 构建镜像 → 部署到 staging 环境 → 验证 → （手动触发）部署到生产环境。

## 模型退化检测与自动回滚

即使经过了充分的 A/B 测试，模型上线后仍可能因为数据分布变化（concept drift）而出现退化。你需要一套自动化的机制来检测这种情况：

```python
import numpy as np
from datetime import datetime, timedelta


class ModelDriftDetector:
    """模型退化检测器"""

    def __init__(self, baseline_metrics: dict, threshold: float = 0.1):
        """
        baseline_metrics: 上线时的基线指标
        threshold: 允许退化的阈值比例 (如 0.1 = 允许变差 10%)
        """
        self.baseline = baseline_metrics
        self.threshold = threshold
        self.history = []

    def check(self, current_metrics: dict) -> dict:
        """检查当前指标是否出现退化"""
        
        alerts = []
        
        for metric_name, baseline_val in self.baseline.items():
            current_val = current_metrics.get(metric_name)
            if current_val is None:
                continue
            
            # 对于 loss/perplexity 类指标：越高越差
            if metric_name in ('val_loss', 'perplexity', 'error_rate'):
                degradation = (current_val - baseline_val) / max(baseline_val, 1e-8)
                if degradation > self.threshold:
                    alerts.append({
                        'metric': metric_name,
                        'baseline': baseline_val,
                        'current': current_val,
                        'degradation_pct': f"{degradation*100:.1f}%",
                        'severity': 'critical' if degradation > 0.3 else 'warning',
                    })
            
            # 对于 throughput/accuracy 类指标：越低越差
            elif metric_name in ('tokens_per_sec', 'bleu_score', 'human_eval'):
                degradation = (baseline_val - current_val) / max(baseline_val, 1e-8)
                if degradation > self.threshold:
                    alerts.append({
                        'metric': metric_name,
                        'baseline': baseline_val,
                        'current': current_val,
                        'degradation_pct': f"{degradation*100:.1f}%",
                        'severity': 'critical' if degradation > 0.3 else 'warning',
                    })
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics,
            'alerts': len(alerts),
        })
        
        if alerts:
            self._trigger_alert(alerts)
        
        return {
            'status': 'degraded' if alerts else 'healthy',
            'alerts': alerts,
            'history_len': len(self.history),
        }

    def _trigger_alert(self, alerts):
        """触发告警通知"""
        critical = [a for a in alerts if a['severity'] == 'critical']
        
        msg = f"🚨 Model Drift Detected ({len(critical)} critical):\n"
        for a in alerts:
            msg += f"  [{a['severity'].upper()}] {a['metric']}: " \
                   f"{a['baseline']} → {a['current']} " \
                    f"(+{a['degradation_pct']})\n"
        
        # 在实际项目中这里应该调用 Slack/PagerDuty/webhook
        print(msg)
        
        if len(critical) >= 2:
            print("🔄 Triggering automatic rollback...")
            # self._rollback_to_previous_version()


detector = ModelDriftDetector(
    baseline_metrics={
        'val_loss': 1.2345,
        'perplexity': 12.34,
        'tokens_per_sec': 45.6,
        'empty_response_rate': 0.01,
        'error_rate': 0.002,
    },
    threshold=0.15,
)

# 每小时运行一次检查
current_metrics = {
    'val_loss': 1.4523,  # 升高了！超过 15%
    'perplexity': 13.89,
    'tokens_per_sec': 38.2,  # 下降了！
    'empty_response_rate': 0.035,  # 升高了很多
    'error_rate': 0.008,
}

result = detector.check(current_metrics)
print(result)
# 输出:
# {'status': 'degraded',
#  'alerts': [
#   {'metric': 'val_loss', ... 'severity': 'warning'},
#   {'metric': 'tokens_per_sec', ... 'severity': 'warning'},
#   {'metric': 'empty_response_rate', ... 'severity': 'critical'},
#   {'metric': 'error_rate', ... 'severity': 'warning'}
# ],
#  'history_len': 1}
```

## 第 9 章（及全书）总结

到这里，第 9 章"生产部署与 MLOps"的全部三个小节就完成了，也标志着整个 PyTorch LLM 教程的全部内容正式完结。让我们站在终点回顾一下整条知识路径：

**第 1 章：PyTorch 核心基础**
从 Tensor 与 ndarray 的区别开始，建立了 GPU 编程、Autograd 动态图、nn.Module 模块系统的认知基础，完成了 MiniGPT 项目作为第一次实战。

**第 2 章：数据管道**
深入 Dataset/DataLoader 的 worker 进程池与 Queue 机制，掌握了 num_workers/pin_memory/prefetch_factor 的调优方法；攻克了 Collate Function 中左填充 vs 右填充的因果约束难题；学会了 Memmap/IterableDataset/Parquet 三种 TB 级数据处理方案；最终组装了端到端的完整数据管道。

**第 3 章：手写 Transformer（全书高潮）**
从单头 Self-Attention 的 Query-Key-Value 类比讲起，理解缩放因子 $\frac{1}{\sqrt{d_k}}$ 和因果掩码的本质；用 reshape+transpose 实现 MHA 的高效多头注意力；通过 Pre-Norm + RMSNorm + SwiGLU FFN 组装了完整 Transformer Block；用 RoPE 旋转位置编码赋予模型位置感知能力；最终从 Token Embedding 到 LM Head 完成了 GPT 模型的搭建与验证。

**第 4 章：训练循环与优化**
从 30 行最小可行循环出发，逐步添加梯度裁剪、Cosine Annealing + Warmup 学习率调度、混合精度训练（FP16/BF16）、梯度累积、评估循环、Checkpoint 管理、早停机制；深入 AdamW 优化器的内部机制（动量、方差归一化解耦权值衰减）；建立了完整的训练监控与 NaN/不收敛/过拟合/震荡/OOM 五大异常诊断体系。

**第 5 章：PyTorch Lightning**
体验了声明式 API 如何将 180 行手写循环压缩为 80 行 LightningModule + 8 行 Trainer 配置；掌握了生命周期钩子系统的精确执行顺序和 Callback 可插拔扩展机制；学习了多优化器/多 DataLoader/FSDP/DeepSpeed/Gradient Checkpointing 的高级特性集成。

**第 6 章：HF Trainer + PEFT 微调实战**
三步上手 QLoRA 微调 7B 模型；从数学原理上理解 LoRA 的低秩分解 $\Delta W = BA$ 以及为什么 rank=16 只需 0.44% 参数就能达到接近全量微调的效果；通过 QLoRA（4-bit NF4 + LoRA）实现了 RTX 4090 单卡微调 7B 的能力；完成了三种训练方式的终极对比（手写 PyTorch / Lightning / HF Trainer）。

**第 7 章：分布式训练**
DDP 的 AllReduce 梯度同步与 DistributedSampler 数据分片；FSDP FULL_SHARD 的 AllGather/RScatter 参数分片与三种 Sharding Strategy 权衡；DeepSpeed ZeRO 三阶段的递进式冗余消除与 CPU/NVMe Offloading；完成了环境检查、启动脚本模板、MFU 性能基准和六大类分布式错误的系统化排查手册。

**第 8 章：推理优化**
torch.compile() 的 Dynamo 字节码追踪 + Inductor kernel 生成两阶段流水线与算子融合原理；INT4/FP8/BF16 多种量化方案的质量-速度权衡与 torchao 库的使用；ONNX/GGUF/SafeTensors 三种导出格式的选型决策树；GPU 同步计时、Prefill/Decode 时间分解和系统化优化 Checklist。

**第 9 章：生产部署与 MLOps（本章）**
FastAPI 原生 PyTorch 服务、vLLM PagedAttention 引擎、TGI Rust 推理框架三种方案的对比与选择；结构化 JSON 日志 + Prometheus 指标 + OpenTelemetry 追踪的可观测性三支柱体系；MLflow Model Registry 的版本管理、A/B 测试流量路由、GitHub Actions CI/CD 自动化流水线、模型退化检测与自动回滚。

九个章节，三十余个文件，覆盖了从一行 `import torch` 到生产级 LLM 服务部署的全栈知识。这不仅是教程，更是一份可以随时查阅的参考手册——当你面试时被问到 Attention 的复杂度、当你调试时遇到梯度消失、当你需要选择量化方案或分布式策略时，你都能在这里找到答案。
