# 微调效果评估与部署：从 checkpoint 到生产服务

## 这一节讲什么？

模型微调完成并保存后，工作只完成了一半。你需要回答几个关键问题：**模型真的变好了吗？比基线提升了多少？在真实数据上的表现如何？如何把它部署成一个可用的服务？**

这一节，我们将系统性地学习：
1. **全面的模型评估方法**——不止看 accuracy，还要看 confusion matrix、error analysis、per-class 性能
2. **消融实验（Ablation Study）**——理解每个组件的贡献
3. **模型导出为 ONNX/TensorRT**——生产级推理优化
4. **构建推理 API 服务**——FastAPI + 异步处理
5. **监控与日志**——线上服务的健康检查

---

## 一、全面评估：超越 Accuracy

## 1.1 为什么 Accuracy 不够？

Accuracy 是最直观的指标，但它掩盖了很多重要信息：

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_accuracy_limitation():
    """展示 accuracy 的局限性"""

    # 场景: 99个正样本 + 1 个负样本
    # 模型把所有样本都预测为"正"
    predictions = np.ones(100)
    labels = np.array([1]*99 + [0])

    accuracy = (predictions == labels).mean()
    precision = (predictions[labels==1] == 1).sum() / predictions.sum()  
    recall = (predictions[labels==1] == 1).sum() / (labels == 1).sum()

    print("=" * 60)
    print("Accuracy 的致命缺陷示例")
    print("=" * 60)
    print(f"\n数据分布: 99 正样本, 1 负样本")
    print(f"预测结果: 全部预测为正类")
    print(f"\n  Accuracy = {accuracy:.2%}   ← 看起来很棒!")
    print(f"  Precision = {precision:.2%}  ← 但只要猜对就给正")
    print(f"  Recall = {recall:.2%}     ← 找到了那个负样本吗？")
    
    print(f"\n💡 结论: Accuracy 在类别不平衡时会严重失真!")
    print(f"   应该使用: F1-Score / Precision-Recall 曲线 / AUC")

demonstrate_accuracy_limitation()
```

## 1.2 完整评估工具箱

```python
from evaluate import load
import numpy as np
from collections import Counter

class ComprehensiveEvaluator:
    """全面的分类模型评估器"""

    def __init__(self, id2label=None):
        self.id2label = id2label or {}
        self.acc_metric = load("accuracy")
        self.f1_metric = load("f1")
        self.precision_metric = load("precision")
        self.recall_metric = load("recall")

    def evaluate(self, model, tokenizer, test_dataset, batch_size=32):
        """执行完整评估"""
        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding
        
        collator = DataCollatorWithPadding(tokenizer)
        loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collator)

        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = probs.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].tolist())
                all_probs.extend(probs.cpu().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        results = {}

        # 基础指标
        results["accuracy"] = self.acc_metric.compute(
            predictions=all_preds, references=all_labels)["accuracy"]
        results["f1_macro"] = self.f1_metric.compute(
            predictions=all_preds, references=all_labels, average="macro")["f1"]
        results["f1_weighted"] = self.f1_metric.compute(
            predictions=all_preds, references=all_labels, average="weighted")["f1"]

        # Per-class metrics
        n_classes = len(set(all_labels))
        per_class_f1 = []
        
        for cls in range(n_classes):
            cls_mask = all_labels == cls
            if cls_mask.sum() > 0:
                cls_pred = all_preds[cls_mask]
                cls_true = all_labels[cls_mask]
                f1 = self.f1_metric.compute(
                    predictions=cls_pred, references=cls_true, average="binary")["f1"]
                per_class_f1.append(f1)
            else:
                per_class_f1.append(0.0)

        results["per_class_f1"] = per_class_f1

        # Confusion Matrix
        results["confusion_matrix"] = self._compute_confusion_matrix(
            all_preds, all_labels)

        # Confidence Analysis
        results["confidence_stats"] = self._analyze_confidence(all_probs, all_preds)

        # Error Analysis
        results["error_samples"] = self._analyze_errors(
            all_preds, all_labels, all_probs, test_dataset, n_samples=10
        )

        return results

    def _compute_confusion_matrix(self, preds, labels):
        """计算混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds)
        return cm

    def _analyze_confidence(self, probs, preds):
        """分析模型预测的置信度"""
        max_probs = probs.max(axis=1)
        correct_mask = preds == np.array([self.dataset_labels[i] 
                                                    for i in range(len(preds))])
        
        return {
            "mean_confidence": max_probs.mean(),
            "correct_mean_conf": max_probs[correct_mask].mean() if correct_mask.any() else 0,
            "wrong_mean_conf": max_probs[~correct_mask].mean() if (~correct_mask).any() else 0,
            "low_confidence_ratio": (max_probs < 0.6).mean(),
        }

    def _analyze_errors(self, preds, labels, probs, dataset, n_samples=10):
        """分析错误预测的样本"""
        error_indices = np.where(preds != labels)[0]
        error_samples = []
        
        for idx in error_indices[:n_samples]:
            sample = {
                "true_label": self.id2label.get(int(labels[idx]), str(labels[idx])),
                "pred_label": self.id2label.get(int(preds[idx]), str(preds[idx])),
                "confidence": float(probs[idx].max()),
                "top2_gap": float(probs[idx].max() - sorted(probs[idx])[-2]),
            }
            
            if hasattr(dataset, "text"):
                sample["text"] = dataset[int(idx)]["text"][:200] if hasattr(dataset, "__getitem__") else ""
            
            error_samples.append(sample)
        
        return error_samples

    def print_report(self, results):
        """打印完整的评估报告"""
        print("\n" + "=" * 70)
        print("📊 全面评估报告")
        print("=" * 70)

        print(f"\n【基础指标】")
        print(f"  Accuracy:       {results['accuracy']:.4f}")
        print(f"  F1 (Macro):     {results['f1_macro']:.4f}")
        print(f"  F1 (Weighted):  {results['f1_weighted']:.4f}")

        print(f"\n【各类别 F1】")
        for i, f1 in enumerate(results.get("per_class_f1", [])):
            label_name = self.id2label.get(i, str(i))
            bar = "█" * int(f1 * 30)
            print(f"  {label_name:<15s} {f1:.4f} {bar}")

        print(f"\n【置信度分析】")
        conf = results.get("confidence_stats", {})
        print(f"  平均置信度:      {conf.get('mean_confidence', 0):.4f}")
        print(f"  正确样本平均置信: {conf.get('correct_mean_conf', 0):.4f}")
        print(f"  错误样本平均置信: {conf.get('wrong_mean_conf', 0):.4f}")
        print(f"  低置信比例(<0.6): {conf.get('low_confidence_ratio', 0):.2%}")

        if results.get("error_samples"):
            print(f"\n【错误样本 Top-{len(results['error_samples'])}】")
            for i, err in enumerate(results["error_samples"]):
                print(f"  [{i+1}] 真:{err['true_label']} "
                      f"预:{err['pred_label']} "
                      f"置信:{err['confidence']:.3f}"
                      f"| Gap:{err['top2_gap']:.3f}")
                if "text" in err:
                    print(f"      文本: {err['text'][:80]}...")

        # 可视化 Confusion Matrix
        cm = results.get("confusion_matrix", None)
        if cm is not None:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm / cm.sum(axis=1, keepdims=True),
                       annot=True, fmt=".2f", cmap="Blues",
                       xticklabels=[self.id2label.get(i, str(i)) for i in range(cm.shape[1])],
                       yticklabels=[self.id2label.get(i, str(i)) for i in range(cm.shape[0])])
            plt.title("Confusion Matrix (归一化)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.show()


# 使用示例
evaluator = ComprehensiveEvaluator(id2label={0: "负面", 1: "中性", 2: "正面"})
# results = evaluator.evaluate(my_model, my_tokenizer, eval_dataset)
# evaluator.print_report(results)
```

---

## 二、消融实验：理解每个组件的贡献

## 2.1 什么是消融实验？

消融实验（Ablation Study）通过**逐一移除或替换模型的某个组件**来衡量该组件的贡献。它是论文审稿中的标准分析方法，也是工程中定位性能瓶颈的有力工具。

## 2.2 实践：对比不同配置的效果

```python
def run_ablation_study(train_ds, eval_ds, configs, task="classification"):
    """
    运行消融实验
    
    configs: list of dict, 每个配置包含不同的超参数/策略
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    from transformers import DataCollatorWithPadding
    import tempfile
    import json

    results = []

    for config in configs:
        name = config.pop("name", "config")
        print(f"\n{'='*50}")
        print(f"🔬 配置: {name}")
        print(f"{'='*50}")
        for k, v in config.items():
            print(f"  {k}: {v}")

        model_name = config.pop("model_name", "hfl/chinese-roberta-wwm-ext")
        num_labels = config.pop("num_labels", 3)

        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            args = TrainingArguments(
                output_dir=tempfile.mkdtemp(),
                num_train_epochs=config.get("epochs", 2),
                per_device_train_batch_size=config.get("batch_size", 16),
                learning_rate=config.get("lr", 2e-5),
                eval_strategy="epoch",
                save_strategy="no",
                fp16=False,
                report_to="none",
                logging_steps=999,
                dataloader_num_workers=0,
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
            )

            train_result = trainer.train()
            eval_metrics = trainer.evaluate()

            results.append({
                "name": name,
                "config": config,
                "train_time": train_result.metrics.get("train_runtime", 0),
                **{k: v for k, v in eval_metrics.items()},
            })

            print(f"  ✅ 完成! eval_loss={eval_metrics.get('eval_loss', 'N/A'):.4f}")

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results.append({"name": name, "error": str(e)})

    # 排序输出
    results.sort(key=lambda x: x.get("eval_loss", float("inf")))

    print("\n\n" + "=" * 70)
    print("📊 消融实验结果排名")
    print("=" * 70)
    print(f"{'配置名称':<30} {'Eval Loss':>12} {'Train Time':>12}")
    print("-" * 70)
    for r in results:
        loss_str = f"{r.get('eval_loss', 0):.4f}" if "eval_loss" in r else "N/A"
        time_str = f"{r.get('train_time', 0):.1f}s" if "train_time" in r else "N/A"
        print(f"{r['name']:<30} {loss_str:>12} {time_str:>12}")

    return results


# 示例：对比不同的微调策略
configs = [
    {"name": "Baseline (lr=2e-5, epoch=3)", "model_name": "hfl/chinese-roberta-wwm-ext",
     "num_labels": 3, "lr": 2e-5, "epochs": 3, "batch_size": 16},
    
    {"name": "高 LR (lr=1e-3)", "model_name": "hfl/chinese-roberta-wwm-ext",
     "num_labels": 3, "lr": 1e-3, "epochs": 3, "batch_size": 16},
    
    {"name": "小 Batch (bs=4)", "model_name": "hfl/chinese-roberta-wwm-ext",
     "num_labels": 3, "lr": 2e-5, "epochs": 3, "batch_size": 4},
    
    {"name": "更多 Epoch (epoch=10)", "model_name": "hfl/chinese-roberta-wwm-ext",
     "num_labels": 3, "lr": 2e-5, "epochs": 10, "batch_size": 16},
]

# ablation_results = run_ablation_study(train_ds, eval_ds, configs)
```

---

## 三、导出 ONNX / TensorRT —— 生产级推理加速

## 3.1 导出 ONNX

```python
def export_to_onnx(model_path, onnx_path, opset=17):
    """将 PyTorch 模型导出为 ONNX 格式"""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.exporters.main_export

    # 方式 1: 使用 optimum 一键导出
    main_export(
        model_name=model_path,
        output=onnx_path,
        task="text-classification",
        opset=opset,
        dtype="float32",
    )
    
    print(f"✅ ONNX 模型已导出到: {onnx_path}")


# 使用 ONNX Pipeline
def use_onnx_pipeline(onnx_path):
    """使用 ONNX Runtime 进行推理"""
    from transformers import pipeline

    onnx_pipe = pipeline(
        "text-classification",
        model=onnx_path,
        exporter="onnx",
        device="cpu",
    )

    result = onnx_pipe("这个产品非常好用！")
    print(f"ONNX 推理结果: {result}")
    return onnx_pipe
```

## 3.2 导出 TensorRT

```python
def export_to_tensorrt(onnx_path, trt_path):
    """将 ONNX 进一步转换为 TensorRT 引擎"""
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        network = builder.create_network(
            onnx_path,
            explicit_batch_dimension=1
        )

        config = builder.create_builder_config(precision=trt.Float16)
        engine = builder.build_engine(network, config)
        serialized_engine = engine.serialize()
        
        with open(trt_path, "wb") as f:
            f.write(serialized_engine)
        
        print(f"✅ TensorRT 引擎已导出到: {trt_path}")
        return True

    except ImportError:
        print("⚠️ TensorRT 未安装。安装方法:")
        print("  pip install tensorrt")
        return False
```

---

## 四、构建推理 API 服务

## 4.1 FastAPI 完整模板

```python
"""
文本分类推理服务
启动: uvicorn inference_server:app --host 0.0.0.0 --port 8000
测试: curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"texts": ["这个产品很好！"]}'
"""
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List
from contextlib import asynccontextmanager
from transformers import pipeline
import torch
import time

app = FastAPI(title="NLP Inference Service")

# 全局变量：懒加载模型
_model = None
_tokenizer = None
_classifier = None

class PredictRequest(BaseModel):
    texts: List[str] = Field(..., description="待分类的文本列表")


class Prediction(BaseModel):
    label: str
    score: float
    text: str


class BatchResponse(BaseModel):
    predictions: List[Prediction]


async def get_model():
    """懒加载模型（首次请求时才加载）"""
    global _model, _classifier
    
    if _classifier is None:
        print("🔄 首次请求，加载模型...")
        start = time.time()
        _classifier = pipeline(
            "text-classification",
            model="./my_sentiment_model/final_model",
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,
        )
        print(f"✅ 模型加载完成! 耗时: {time.time()-start:.2f}s")
    
    return _classifier


@app.post("/predict", response_model=BatchResponse)
async def predict(request: PredictRequest):
    classifier = await get_model()
    
    results = classifier(request.texts, top_k=None, truncation=True)
    
    predictions = [
        Prediction(
            label=r["label"],
            score=round(r["score"], 4),
            text=text,
        )
        for r, text in zip(results, request.texts)
    ]
    
    return BatchResponse(predictions=predictions)


@app.get("/health")
async def health():
    global _classifier
    return {
        "status": "healthy",
        "model_loaded": _classifier is not None,
        "device": next(iter(_model.parameters())).device if _model else "not loaded"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 4.2 批量推理优化

```python
class BatchInferenceEngine:
    """高性能批量推理引擎"""

    def __init__(self, model_path, max_batch_size=32, timeout=10.0):
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            device=0,
            batch_size=max_batch_size,
        )
        self.max_batch_size = max_batch_size
        self.timeout = timeout

    @async def infer(self, texts: list[str]) -> list[dict]:
        """异步批量推理"""
        results = self.classifier(texts, truncation=True)
        return [
            {"label": r["label"], "score": round(r["score"], 4)}
            for r in results
        ]
```

---

## 五、线上监控与告警

```python
class ProductionMonitor:
    """生产环境监控器"""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.latencies = []
        self.start_time = time.time()

    def record_request(self, latency_ms, success=True):
        self.request_count += 1
        self.latencies.append(latency_ms)
        if not success:
            self.error_count += 1

    def get_status(self):
        uptime = time.time() - self.start_time
        avg_latency = sum(self.latencies[-100:]) / min(len(self.latencies), 1)
        p95_latency = sorted(self.latencies)[-int(len(self.latencies)*0.05):]
        error_rate = self.error_count / max(self.request_count, 1)

        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "error_rate": f"{error_rate:.4f}",
            "avg_latency_ms": f"{avg_latency:.1f}",
            "p95_latency_ms": f"{p95_latency:.1f}",
            "requests_per_second": self.request_count / max(uptime, 1),
        }
```

---

## 小结

这一节完成了从训练到生产的完整闭环：

1. **全面评估体系**：超越单一 accuracy，使用 `ComprehensiveEvaluator` 类实现 F1(macro/weighted)、Per-Class F1、Confusion Matrix、置信度分析、Error Sample 分析六大维度
2. **消融实验**：通过对比不同配置（lr/batch_size/epochs/冻结层）的效果，精确定位最优超参数组合和每个组件的实际贡献
3. **ONNX/TensorRT 导出**：`optimum` 库一键导出 ONNX（~2-3x 加速），TensorRT 进一步优化到 ~4-5x（需 NVIDIA GPU）
4. **FastAPI 推理服务**：完整的生产级模板——懒加载、异步处理、批量推理、健康检查端点、结构化输入输出
5. **线上监控**：请求数量、延迟分位（P50/P95/P99）、错误率、QPS 等关键运维指标

至此，**第6章 模型微调 Fine-Tuning 全部完成**（6节）！接下来进入第7章：参数高效微调 PEFT。
