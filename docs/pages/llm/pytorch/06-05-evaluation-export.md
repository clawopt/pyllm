# 6.5 微调后的模型评估与导出

经过前面四节的学习，你已经能够用 HF Trainer + PEFT (LoRA/QLoRA) 成功地微调一个 LLM 了——代码不到 60 行，在单张 RTX 4090 上跑完，checkpoint 也保存好了。但训练完成只是第一步。你的模型到底微调得怎么样？它的生成质量如何？它是否真的学会了你期望它学会的东西？这些问题的答案需要通过**评估**来获得。而当你对评估结果满意之后，还需要把模型**导出**为可部署的格式——无论是推送到 HuggingFace Hub 让别人使用，还是转换为 GGUF 格式给 Ollama 加载，或者直接集成到 FastAPI 服务中。这一节覆盖这两个关键环节：评估方法（从自动指标到人工评估）和导出流程（从合并 LoRA 权重到多格式输出）。

## 评估方法一：Perplexity（困惑度）

Perplexity（PPL）是语言模型最基础也最通用的评估指标。它的数学定义是：

$$\text{PPL}(x) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i \mid x_{<i})\right)\right$$

直觉上理解：PPL 衡量的是模型对一段文本的"惊讶程度"——PPL 越低说明模型越"不惊讶"于这段文本（即预测得越好）。对于同一个模型来说，微调后的 PPL 应该比微调前低（至少在你微调目标领域的数据上）。

```python
import torch
from tqdm import tqdm
import math


def evaluate_perplexity(model, tokenizer, texts, batch_size=4, max_length=512):
    """计算给定文本列表的 Perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        input_ids = encodings['input_ids'].to(model.device)
        attention_mask = encodings['attention_mask'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-tokenizer.pad_token_id)
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        mask = attention_mask[..., 1:].contiguous().view(-1).float()
        valid_losses = losses * mask
        
        total_loss += valid_losses.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    return {
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
    }


texts_to_eval = [
    "人工智能是计算机科学的一个重要分支。",
    "The quick brown fox jumps over the lazy dog.",
    "量子计算利用量子力学原理进行信息处理。",
]

result = evaluate_perplexity(merged_model, tokenizer, texts_to_eval)
print(f"PPL: {result['perplexity']:.2f}")
print(f"Avg Loss: {result['avg_loss']:.4f}")
```

关于 PPL 有几个需要注意的点：第一，不同模型的 PPL 不能直接横向比较——因为它们的词表大小、训练数据、架构都不同。PPL 更适合用于**纵向比较**（同一个模型微调前后、或者同一个模型的不同版本之间）。第二，PPL 低不一定意味着生成质量高——一个模型可能通过过度保守的预测（总是输出高频词）获得低 PPL，但生成的文本却很无聊和重复。所以 PPL 通常需要配合其他评估方式一起使用。

## 评估方法二：自动 Benchmark

除了 PPL 之外，还有一系列标准化的 benchmark 可以用来评估 LLM 的能力：

| Benchmark | 测试能力 | 适用场景 |
|-----------|---------|---------|
| **MMLU** | 多学科知识（57 个科目） | 通用知识推理能力 |
| **C-Eval / CMMLU** | 中文多学科知识 | 中文模型评估 |
| **HumanEval / MBPP** | 代码生成 | 编程能力 |
| **GSM8K / MATH** | 数学推理 | 数值推理能力 |
| **HellaSwag / ARC-C** | 常识推理 | 常识判断能力 |
| **TruthfulQA** | 事实正确性 | 减少幻觉 |

```python
def run_benchmark_evaluation(model, tokenizer):
    """使用 lm-eval-harness 运行标准 benchmark"""

    try:
        import lm_eval
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=["winogrande", "hellaswag", "arc_easy", "arc_challenge"],
            batch_size=4,
            device=torch.cuda.current_device(),
        )
        print("\nBenchmark Results:")
        for task, scores in results['results'].items():
            for metric_name, value in scores.items():
                if isinstance(value, float):
                    print(f"  {task}/{metric_name}: {value:.4f}")
        return results

    except ImportError:
        print("lm-eval-harness not installed. Run: pip install lm-eval-harness")
        return None


# run_benchmark_evaluation(merged_model, tokenizer)
```

`lm-eval-harness` 是 EleutherAI 开发的标准 LLM 评估框架，支持上百种 benchmark，是学术研究和工业界的事实标准工具。不过运行完整的 benchmark 需要较长时间（几十分钟到几小时不等），所以在快速迭代阶段通常只跑几个轻量级的任务（如 WinoGrande、HellaSwag）做初步筛选。

## 评估方法三：生成质量评估（人工 / GPT-as-Judge）

对于对话类或指令遵循类的微调任务，自动 benchmark 往往不够用——你需要实际看看模型生成的文本质量如何。最直接的方法是**人工评估**：准备一组测试 prompt，让模型生成回答，然后由人来打分或排序。

```python
@torch.no_grad()
def generate_samples(model, tokenizer, prompts, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """批量生成样本用于质量检查"""
    model.eval()
    samples = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        samples.append({
            'prompt': prompt,
            'generation': generated,
        })

    return samples


test_prompts = [
    "请用三句话解释什么是机器学习。",
    "写一首关于春天的五言绝句。",
    "Python 和 JavaScript 的主要区别是什么？",
    "请将以下英文翻译成中文: The quick brown fox jumps over the lazy dog.",
]

samples = generate_samples(merged_model, tokenizer, test_prompts)

for i, s in enumerate(samples):
    print(f"\n{'='*60}")
    print(f"[Prompt {i+1}] {s['prompt']}")
    print(f"[Generation]\n{s['generation']}")
    print(f"{'='*60}")
```

当测试数量较多时，人工评估成本很高。一种替代方案是 **GPT-as-Judge**——用一个更强的 GPT 模型（如 GPT-4）作为裁判来评估你的模型的输出质量。具体做法是把 prompt + 你的模型输出 + 参考答案一起送给 GPT-4，让它按多个维度（相关性、准确性、流畅性等）打分。

```python
def gpt_as_judge(prompt, model_output, reference=None, judge_model="gpt-4"):
    """使用 GPT 作为裁判评估生成质量"""
    
    judge_prompt = f"""你是一个专业的语言模型评估员。请根据以下标准评估模型的回答:

用户问题: {prompt}

模型回答: {model_output}
{'参考回答: ' + reference if reference else ''}

请从以下维度打分(1-10):
1. 相关性: 回答是否与问题相关
2. 准确性: 信息是否准确无误
3. 完整性: 是否完整回答了问题
4. 流畅性: 语言表达是否自然流畅

请以 JSON 格式输出评分结果。
"""

    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
    )
    
    import json
    return json.loads(response.choices[0].message.content)


score = gpt_as_judge(
    test_prompts[0],
    samples[0]['generation'],
)
print(f"Evaluation scores: {score}")
```

GPT-as-Judge 的好处是成本低、速度快、一致性高（同一个裁判对所有样本使用相同的标准），但它也有局限性——裁判模型本身可能有偏见，而且它不一定能捕捉到特定领域的细微差别。最佳实践是将 GPT-as-Judge 用于初筛，再对有疑问的样本进行人工复核。

## 模型导出：从 LoRA Adapter 到部署就绪

评估通过后，下一步就是把模型导出为各种格式以供部署使用。根据目标平台的不同，有不同的导出路径。

### 路径一：合并 LoRA 权重 → HF 格式

这是最常见的导出方式——把 LoRA 的低秩更新合并回基础模型中，得到一个完整的、可以直接加载的模型：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_export(base_model_path, lora_adapter_path, output_dir):
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print("Saving merged model...")
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✓ Merged model saved to {output_dir}")

    new_model = AutoModelForCausalLM.from_pretrained(output_dir)
    param_count = sum(p.numel() for p in new_model.parameters())
    print(f"Merged model params: {param_count:,}")

    return merged_model


merge_and_export(
    base_model_path="Qwen/Qwen2.5-7B-Instruct",
    lora_adapter_path="./output/qwen-lora-final",
    output_dir="./output/qwen-merged",
)
```

`merge_and_unload()` 的过程是这样的：对于每个被 LoRA 修改的层，计算 $W_{new} = W_0 + B A$（其中 $W_0$ 是原始冻结权重，$B$ 和 $A$ 是 LoRA 的低秩矩阵），然后用合并后的新权重替换原始权重。最终得到的模型不再依赖任何 adapter 文件——它就是一个标准的 HuggingFace 模型，可以用 `AutoModelForCausalLM.from_pretrained()` 直接加载。

### 路径二：导出为 GGUF 格式（Ollama 兼容）

GGUF 是 llama.cpp 项目使用的模型格式，也是 Ollama 推理引擎的原生格式。如果你想把微调后的模型部署到本地或边缘设备上运行，GGUF 是最好的选择之一：

```python
def export_to_gguf(hf_model_dir, output_file, q_method="Q4_K_M"):
    """导出为 GGUF 格式（需要 llama.cpp 的 convert_hf_to_gguf.py 工具）"""
    import subprocess
    
    cmd = [
        "python", "convert_hf_to_gguf.py",
        hf_model_dir,
        "--outfile", output_file,
        "--outtype", q_method,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ GGUF exported to {output_file}")
        import os
        size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"  File size: {size_mb:.1f} MB")
    else:
        print(f"✗ Export failed: {result.stderr}")


# export_to_gguf("./output/qwen-merged", "./output/qwen.Q4_K_M.gguf")
```

`q_method` 参数决定了量化精度：
- `Q4_K_M`：4-bit K-means 量化（推荐，质量和体积的良好平衡）
- `Q5_K_M`：5-bit（质量更好，体积稍大）
- `Q8_0`：8-bit（接近无损）
- `F16`：FP16（不量化）

### 路径三：上传到 HuggingFace Hub

如果你想分享你的模型或者让团队其他成员使用，上传到 Model Hub 是最方便的方式：

```python
from huggingface_hub import login, create_repo, upload_folder


def push_to_hub(local_dir, repo_name, private=False):
    login()

    repo_url = create_repo(
        repo_id=f"your-username/{repo_name}",
        private=private,
        exist_ok=True,
    )

    folder_info = upload_folder(
        folder_path=local_dir,
        repo_id=repo_url.repo_id,
        commit_message="Upload fine-tuned QLoRA model",
    )

    print(f"✓ Model pushed to https://huggingface.co/{repo_url.repo_id}")

    card_content = f"""---
license: apache-2.0
tags:
- generated_from_trainer
- qwen
- lora
language:
- zh
- en
base_model: Qwen/Qwen2.5-7B-Instruct
---

# {repo_name}

This is a fine-tuned version of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

## Training Details
- Method: QLoRA (r=16, alpha=32)
- Data: Alpaca-style instruction dataset
- Hardware: 1× RTX 4090 (24GB)
- Epochs: 3

## Evaluation
- PPL on test set: TBD
"""
    
    with open(f"{local_dir}/README.md", "w") as f:
        f.write(card_content)

    upload_folder(
        folder_path=local_dir,
        repo_id=repo_url.repo_id,
        commit_message="Add model card",
    )


# push_to_hub("./output/qwen-merged", "my-qwen-alpaca")
```

## 第 6 章小结与第 7 章预告

到这里，第 6 章"HuggingFace Trainer 与 LLM 微调实战"的全部五个小节就完成了。我们从 Trainer API 的三步上手开始，深入了 LoRA 的低秩分解原理和 PEFT 库的完整配置选项，学习了 QLoRA 如何通过 4-bit 量化让单卡微调成为现实，然后做了三种训练方式的全面对比实验，最后覆盖了模型评估（PPL、Benchmark、GPT-as-Judge）和多格式导出（HF 合并、GGUF、Hub 上传）的完整流程。

下一章我们将进入分布式训练的世界——当单张 GPU 无法满足需求时（无论是显存不够还是速度太慢），如何利用多台机器、多张 GPU 来并行化训练过程。DDP、FSDP、DeepSpeed ZeRO 三大方案将依次登场，我们还会学习如何在 Lightning 或 HF Trainer 中一键启用它们，以及分布式训练中常见的错误排查方法。
