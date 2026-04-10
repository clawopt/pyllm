# 高效批量处理技巧

> **白板时间**：想象你需要对 100 万条用户评论做情感分析。如果一条条调用，RTX 4090 也要跑好几天。但如果你能找到最优的 batch size、实现断点续传（崩溃了不用从头跑）、利用多 GPU 并行——时间可以从几天缩短到几小时。这一节就是教你如何在生产级规模下高效使用 vLLM 的离线推理能力。

## 一、最优 Batch Size 探索

### 1.1 为什么 Batch Size 很重要？

```
Batch Size 太小 → GPU 算力闲置，吞吐量低
                ↓
        浪费电费和时间

Batch Size 太大 → OOM (CUDA Out of Memory)
                ↓
        程序崩溃，前功尽弃

最优 Batch Size → GPU 利用率 ~95%，无 OOM
                ↓
        最大吞吐，最低成本
```

### 1.2 二分法寻找最优值

比如下面的程序自动探索最优 batch size：

```python
import time
import torch
from vllm import LLM, SamplingParams

def find_optimal_batch_size(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens: int = 64,
    prompt_length: int = 100,
    initial_bs: int = 4,
    max_bs: int = 512,
) -> int:
    """二分法寻找最优 batch size"""
    
    llm = LLM(model=model, dtype="auto")
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.3)
    
    template_prompt = "这是一段测试文本。" * (prompt_length // 8 + 1)
    
    low, high = initial_bs, max_bs
    optimal_bs = initial_bs
    
    print(f"[Batch Size 探索] 模型={model}, max_tokens={max_tokens}")
    print(f"{'Batch Size':>12} | {'状态':>8} | {'耗时':>8} | {'吞吐':>10} | {'VRAM':>8}")
    print("-" * 65)
    
    while low <= high:
        mid = (low + high) // 2
        
        try:
            prompts = [template_prompt for _ in range(mid)]
            
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            outputs = llm.generate(prompts, sp)
            elapsed = time.time() - start
            
            vram_gb = torch.cuda.max_memory_allocated() / 1024**3
            total_tokens = sum(
                len(o.prompt_token_ids) + len(o.outputs[0].token_ids) 
                for o in outputs
            )
            throughput = total_tokens / elapsed
            
            print(f"{mid:>12} | {'✅ OK':>8} | {elapsed:>7.2f}s | {throughput:>9.0f} t/s | {vram_gb:>7.2f}G")
            
            optimal_bs = mid
            low = mid + 1
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{mid:>12} | {'❌ OOM':>8} | {'---':>8} | {'---':>10} | {'---':>8}")
            high = mid - 1
    
    del llm
    torch.cuda.empty_cache()
    
    print(f"\n[结果] 最优 batch size = {optimal_bs}")
    return optimal_bs


bs = find_optimal_batch_size(initial_bs=8, max_bs=256)
```

典型输出：

```
[Batch Size 探索] 模型=Qwen/Qwen2.5-0.5B-Instruct, max_tokens=64
  Batch Size |     状态 |     耗时 |       吞吐 |    VRAM
-----------------------------------------------------------------
          64 |   ✅ OK |    3.21s |    2034 t/s |    6.12G
         128 |   ✅ OK |    5.43s |    2378 t/s |   11.34G
         192 |   ✅ OK |    7.89s |    2456 t/s |   16.78G
         224 |   ❌ OOM |     --- |      --- |     ---
        [结果] 最优 batch size = 192
```

### 1.3 动态 Batch Size 策略

固定 batch size 不够灵活——当剩余显存变化时应该动态调整：

```python
import torch
from vllm import LLM, SamplingParams

class AdaptiveBatchProcessor:
    """自适应批量处理器——根据显存动态调整 batch size"""
    
    def __init__(self, model: str, **kwargs):
        self.llm = LLM(model=model, dtype="auto", **kwargs)
        self.sampling_params = None
        self.total_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        self.target_utilization = 0.90
    
    def _estimate_batch_size(self, sample_prompt: str, 
                              sample_max_tokens: int,
                              current_batch: int = 32) -> int:
        """基于当前显存使用估算最大安全 batch size"""
        
        # 用小 batch 试探
        try:
            test_prompts = [sample_prompt] * min(current_batch, 8)
            torch.cuda.reset_peak_memory_stats()
            self.llm.generate(test_prompts, 
                             SamplingParams(max_tokens=min(sample_max_tokens, 16)))
            peak_per_sample = torch.cuda.max_memory_allocated() / len(test_prompts)
            
            available = self.total_vram * self.target_utilization - 2  # 预留 2GB 安全边际
            estimated = int(available / (peak_per_sample / 1024**3))
            
            return max(1, min(estimated, current_batch * 2))
        except:
            return max(1, current_batch // 2)
    
    def process(self, all_prompts: list, sampling_params: SamplingParams,
                progress_callback=None) -> list:
        """自适应批处理所有 prompt"""
        
        self.sampling_params = sampling_params
        results = []
        idx = 0
        
        # 第一轮探测最优 batch size
        if all_prompts:
            bs = self._estimate_batch_size(
                all_prompts[0], 
                sampling_params.max_tokens or 128
            )
        else:
            bs = 32
        
        while idx < len(all_prompts):
            batch = all_prompts[idx:idx+bs]
            
            try:
                outputs = self.llm.generate(batch, sampling_params)
                results.extend(outputs)
                idx += bs
                
                if progress_callback:
                    progress_callback(idx, len(all_prompts))
                
                # 成功则尝试增大
                bs = min(int(bs * 1.2), len(all_prompts) - idx)
                
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                if bs < 1:
                    raise RuntimeError("即使 batch_size=1 也 OOM")
        
        return results


def adaptive_demo():
    """自适应批处理演示"""
    
    processor = AdaptiveBatchProcessor("Qwen/Qwen2.5-0.5B-Instruct")
    
    prompts = [f"请总结第{i}条数据的内容" for i in range(200)]
    sp = SamplingParams(max_tokens=32, temperature=0.3)
    
    import time
    start = time.time()
    results = processor.process(prompts, sp, 
                                lambda c, t: print(f"\r进度: {c}/{t}", end=""))
    elapsed = time.time() - start
    
    print(f"\n\n[完成] {len(results)} 条, 耗时 {elapsed:.2f}s, "
          f"吞吐 {len(results)/elapsed:.0f} req/s")

adaptive_demo()
```

## 二、断点续传与错误恢复

### 2.1 断点续传机制

对于大规模任务（数万到数百万条），程序可能因各种原因中断。断点续传让你从中断处继续：

```python
import json
import os
import time
from pathlib import Path
from vllm import LLM, SamplingParams

class ResumableProcessor:
    """支持断点续传的批量处理器"""
    
    def __init__(self, model: str, output_dir: str = "./output", 
                 checkpoint_interval: int = 100):
        self.llm = LLM(model=model, dtype="auto")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
    
    def _get_checkpoint_path(self, task_name: str) -> Path:
        return self.output_dir / f"{task_name}_checkpoint.json"
    
    def _get_output_path(self, task_name: str) -> Path:
        return self.output_dir / f"{task_name}_results.jsonl"
    
    def _save_checkpoint(self, path: Path, completed_indices: list):
        with open(path, 'w') as f:
            json.dump({"completed": completed_indices, 
                       "timestamp": time.time()}, f)
    
    def _load_checkpoint(self, path: Path) -> set:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                print(f"[恢复] 从检查点加载，已完成 {len(data['completed'])} 条")
                return set(data['completed'])
        return set()
    
    def _append_result(self, path: Path, result: dict):
        with open(path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def process_with_resume(self, prompts: list, sampling_params: SamplingParams,
                            task_name: str = "batch_task") -> list:
        """带断点续传的批处理"""
        
        ckpt_path = self._get_checkpoint_path(task_name)
        output_path = self._get_output_path(task_name)
        completed_set = self._load_checkpoint(ckpt_path)
        
        pending_indices = [i for i in range(len(prompts)) 
                          if i not in completed_set]
        
        print(f"[任务] 总计 {len(prompts)} 条, "
              f"已完成 {len(completed_set)} 条, 待处理 {len(pending_indices)} 条")
        
        batch_completed_in_round = 0
        
        for i in pending_indices:
            try:
                outputs = self.llm.generate([prompts[i]], sampling_params)
                output = outputs[0]
                
                result = {
                    "index": i,
                    "prompt": prompts[i],
                    "text": output.outputs[0].text,
                    "finish_reason": output.outputs[0].finish_reason,
                    "token_count": len(output.outputs[0].token_ids),
                }
                
                self._append_result(output_path, result)
                completed_set.add(i)
                batch_completed_in_round += 1
                
                if batch_completed_in_round % self.checkpoint_interval == 0:
                    self._save_checkpoint(ckpt_path, sorted(completed_set))
                    pct = len(completed_set) / len(prompts) * 100
                    print(f"[进度] {len(completed_set)}/{len(prompts)} ({pct:.1f}%) - 已保存检查点")
                    
            except Exception as e:
                print(f"[错误] index={i}: {e}")
                error_result = {
                    "index": i, "prompt": prompts[i],
                    "text": "", "error": str(e),
                    "finish_reason": "error"
                }
                self._append_result(output_path, error_result)
                completed_set.add(i)
        
        self._save_checkpoint(ckpt_path, sorted(completed_set))
        print(f"[完成] 全部完成！输出文件: {output_path}")
        
        return list(completed_set)


def resume_demo():
    """断点续传演示"""
    
    processor = ResumableProcessor(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="./batch_output",
        checkpoint_interval=50,
    )
    
    prompts = [f"翻译成英文：第{i}条数据包含重要信息" for i in range(300)]
    sp = SamplingParams(max_tokens=32, temperature=0.3)
    
    completed = processor.process_with_resume(prompts, sp, task_name="translation_v1")

resume_demo()
```

### 2.2 失败样本隔离与重跑

```python
def retry_failed_samples():
    """失败样本的隔离和重跑机制"""
    
    output_file = "./batch_output/translation_v1_results.jsonl"
    failed_indices = []
    
    # 第一步：扫描失败的样本
    with open(output_file) as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line.strip())
            if "error" in record or not record.get("text", "").strip():
                failed_indices.append((record["index"], line_num))
    
    print(f"[扫描] 共发现 {len(failed_indices)} 个失败样本")
    
    if not failed_indices:
        print("[正常] 所有样本都成功处理了")
        return
    
    # 第二步：只重跑失败的
    prompts = [f"翻译成英文：第{i}条数据包含重要信息" 
               for i, _ in failed_indices]
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    sp = SamplingParams(max_tokens=32, temperature=0.3)
    
    outputs = llm.generate(prompts, sp)
    
    # 第三步：更新原文件中的失败记录
    updated_lines = []
    with open(output_file) as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line.strip())
            for fail_idx, (_, orig_line) in enumerate(failed_indices):
                if line_num == orig_line:
                    out = outputs[fail_idx]
                    record.update({
                        "text": out.outputs[0].text,
                        "finish_reason": out.outputs[0].finish_reason,
                        "retried": True,
                        "error": None,
                    })
                    break
            updated_lines.append(json.dumps(record, ensure_ascii=False))
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(updated_lines) + '\n')
    
    print(f"[重跑] 已修复 {len(failed_indices)} 个失败样本")

retry_failed_samples()
```

## 三、多进程并行推理

### 3.1 单机多 GPU 并行

```python
import multiprocessing as mp
import torch
import time
from vllm import LLM, SamplingParams
from typing import List


def worker_process(gpu_id: int, task_queue: mp.Queue, 
                   result_queue: mp.Queue, model: str,
                   sampling_config: dict):
    """Worker 进程：在指定 GPU 上运行 LLM"""
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    llm = LLM(model=model, dtype="auto")
    sp = SamplingParams(**sampling_config)
    
    while True:
        item = task_queue.get()
        if item is None:  # 哨兵值，表示结束
            break
        
        index, prompt = item
        try:
            outputs = llm.generate([prompt], sp)
            result_queue.put({
                "index": index,
                "text": outputs[0].outputs[0].text,
                "status": "success"
            })
        except Exception as e:
            result_queue.put({
                "index": index,
                "text": "",
                "status": "error",
                "error": str(e)
            })
    
    del llm
    torch.cuda.empty_cache()


def multi_gpu_parallel():
    """多 GPU 并行推理演示"""
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"[跳过] 需要至少 2 个 GPU，当前只有 {num_gpus} 个")
        return
    
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    n_samples = 500
    
    prompts = [f"分析第{i}条文本的情感倾向" for i in range(n_samples)]
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    for i, p in enumerate(prompts):
        task_queue.put((i, p))
    
    for _ in range(num_gpus):
        task_queue.put(None)
    
    sampling_config = {"max_tokens": 32, "temperature": 0.3}
    
    start = time.time()
    
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, task_queue, result_queue, model, sampling_config)
        )
        p.start()
        processes.append(p)
    
    results = []
    while len(results) < n_samples:
        result = result_queue.get()
        results.append(result)
        if len(results) % 100 == 0:
            print(f"\r[进度] {len(results)}/{n_samples}", end="")
    
    for p in processes:
        p.join()
    
    elapsed = time.time() - start
    success_count = sum(1 for r in results if r["status"] == "success")
    
    print(f"\n\n[完成] {n_samples} 条样本")
    print(f"[GPU数量] {num_gpus}")
    print(f"[总耗时] {elapsed:.2f}s")
    print(f"[成功率] {success_count}/{n_samples} ({success_count/n_samples:.1%})")
    print(f"[吞吐] {n_samples/elapsed:.0f} samples/s")

multi_gpu_parallel()
```

### 3.2 进度条集成

```python
from tqdm import tqdm

class ProgressableProcessor:
    """带 tqdm 进度条的批量处理器"""
    
    def __init__(self, model: str, **kwargs):
        self.llm = LLM(model=model, dtype="auto", **kwargs)
    
    def process_with_progress(self, prompts: list, 
                               sampling_params: SamplingParams,
                               batch_size: int = 64,
                               desc: str = "Processing") -> list:
        """带进度条的批处理"""
        
        results = []
        
        with tqdm(total=len(prompts), desc=desc, unit="req") as pbar:
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                
                try:
                    outputs = self.llm.generate(batch, sampling_params)
                    results.extend(outputs)
                    pbar.update(len(batch))
                except Exception as e:
                    pbar.write(f"❌ Batch {i//batch_size} 错误: {e}")
                    for j, prompt in enumerate(batch):
                        from dataclasses import dataclass
                        results.append(type('obj', (object,), {
                            'prompt': prompt,
                            'outputs': [type('obj', (object,), {
                                'text': f'[ERROR] {e}',
                                'finish_reason': 'error'
                            })()],
                            'finished': True
                        })())
                    pbar.update(len(batch))
        
        return results


def progress_demo():
    """进度条演示"""
    
    processor = ProgressableProcessor("Qwen/Qwen2.5-0.5B-Instruct")
    
    prompts = [f"生成关于主题{i}的一段描述" for i in range(200)]
    sp = SamplingParams(max_tokens=24, temperature=0.5)
    
    results = processor.process_with_progress(prompts, sp, batch_size=32)
    
    print(f"\n[示例] {results[0].outputs[0].text[:80]}...")

progress_demo()
```

## 四、结果持久化策略

### 4.1 JSON Lines 格式

JSONL 是大批量数据处理的标准格式——每行一个 JSON 对象，支持流式追加：

```python
import json
from datetime import datetime
from pathlib import Path

class ResultWriter:
    """结果持久化管理器"""
    
    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.count = 0
    
    def __enter__(self):
        self.file = open(self.path, 'a', encoding='utf-8')
        return self
    
    def __exit__(self, *args):
        if self.file:
            self.file.close()
    
    def write(self, index: int, prompt: str, output, metadata: dict = None):
        """写入单条结果"""
        record = {
            "index": index,
            "prompt": prompt,
            "generated_text": output.outputs[0].text,
            "finish_reason": output.outputs[0].finish_reason,
            "generated_token_count": len(output.outputs[0].token_ids),
            "prompt_token_count": len(output.prompt_token_ids),
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            record["metadata"] = metadata
        
        self.file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.count += 1
    
    def get_stats(self):
        return {"output_file": str(self.path), "total_records": self.count}


def persistence_demo():
    """结果持久化演示"""
    
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
    sp = SamplingParams(max_tokens=16, temperature=0.3)
    
    prompts = ["什么是AI", "解释机器学习", "深度学习的定义"]
    
    with ResultWriter("./output/batch_results.jsonl") as writer:
        outputs = llm.generate(prompts, sp)
        
        for i, output in enumerate(outputs):
            writer.write(i, output.prompt, output, 
                        metadata={"batch_id": "demo_001"})
    
    stats = writer.get_stats()
    print(f"[写入] {stats['total_records']} 条 → {stats['output_file']}")
    
    # 验证读取
    print("\n[验证读取]")
    with open(stats['output_file']) as f:
        for line in f:
            record = json.loads(line)
            print(f"  [{record['index']}] {record['generated_text'][:50]}...")

persistence_demo()
```

### 4.2 Parquet 格式（大数据友好）

对于超大规模数据集（百万级以上），Parquet 比 JSONL 更高效：

```python
def parquet_export():
    """导出为 Parquet 格式"""
    
    try:
        import pandas as pd
    except ImportError:
        print("[跳过] 需要 pandas: pip install pandas pyarrow")
        return
    
    jsonl_path = "./output/batch_results.jsonl"
    parquet_path = "./output/batch_results.parquet"
    
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, index=False)
    
    original_size = Path(jsonl_path).stat().st_size / 1024
    compressed_size = Path(parquet_path).stat().st_size / 1024
    
    print(f"[Parquet] {len(records)} 条记录")
    print(f"  JSONL: {original_size:.1f} KB")
    print(f"  Parquet: {compressed_size:.1f} KB (压缩比 {original_size/compressed_size:.1f}x)")

parquet_export()
```

## 五、性能基准测试脚本

### 5.1 自动化 Benchmark 工具

```python
import time
import statistics
import torch
from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
from typing import List

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    batch_sizes: List[int] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)
    latencies_p50: List[float] = field(default_factory=list)
    latencies_p99: List[float] = field(default_factory=list)
    vram_usages: List[float] = field(default_factory=list)


def run_throughput_benchmark(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    batch_sizes: List[int] = None,
    max_tokens: int = 64,
    prompt_len: int = 128,
    warmup_rounds: int = 2,
    measure_rounds: int = 5,
) -> BenchmarkResult:
    """自动化吞吐量基准测试"""
    
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    
    llm = LLM(model=model, dtype="auto")
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.3)
    
    template = "这是一个用于性能测试的长提示词。" * (prompt_len // 15 + 1)
    
    result = BenchmarkResult()
    
    print(f"{'='*70}")
    print(f"vLLM 吞吐量基准测试: {model}")
    print(f"max_tokens={max_tokens}, prompt_len≈{prompt_len}, "
          f"warmup={warmup_rounds}, rounds={measure_rounds}")
    print(f"{'='*70}")
    print(f"{'BS':>6} | {'P50(ms)':>8} | {'P99(ms)':>8} | {'Throughput':>12} | {'VRAM(GB)':>9}")
    print("-" * 65)
    
    for bs in batch_sizes:
        latencies = []
        throughputs = []
        
        for round_idx in range(warmup_rounds + measure_rounds):
            prompts = [template for _ in range(bs)]
            
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            
            outputs = llm.generate(prompts, sp)
            
            elapsed = time.time() - start
            vram = torch.cuda.max_memory_allocated() / 1024**3
            
            total_tokens = sum(
                len(o.prompt_token_ids) + len(o.outputs[0].token_ids)
                for o in outputs
            )
            throughput = total_tokens / elapsed
            
            if round_idx >= warmup_rounds:
                latencies.append(elapsed * 1000 / bs)
                throughputs.append(throughput)
        
        p50 = statistics.median(latencies)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        avg_tp = statistics.mean(throughputs)
        
        result.batch_sizes.append(bs)
        result.throughputs.append(avg_tp)
        result.latencies_p50.append(p50)
        result.latencies_p99.append(p99)
        result.vram_usages.append(vram)
        
        print(f"{bs:>6} | {p50:>8.1f} | {p99:>8.1f} | {avg_tp:>11.0f} t/s | {vram:>8.2f}")
    
    del llm
    torch.cuda.empty_cache()
    
    # 找出最优配置
    best_idx = max(range(len(result.throughputs)), 
                   key=lambda i: result.throughputs[i])
    print(f"\n[推荐] batch_size={result.batch_sizes[best_idx]}, "
          f"吞吐量={result.throughputs[best_idx]:.0f} tokens/s")
    
    return result


benchmark = run_throughput_benchmark(
    batch_sizes=[1, 4, 8, 16, 32, 64],
    max_tokens=32,
    warmup_rounds=1,
    measure_rounds=3,
)
```

典型输出：

```
======================================================================
vLLM 吞吐量基准测试: Qwen/Qwen2.5-0.5B-Instruct
max_tokens=32, prompt_len≈128, warmup=2, rounds=5
======================================================================
    BS |   P50(ms) |   P99(ms) |   Throughput |  VRAM(GB)
-----------------------------------------------------------------
     1 |    145.2 |    178.3 |         1102 |     3.21
     4 |    389.5 |    412.1 |         1654 |     4.56
     8 |    678.2 |    701.5 |         1892 |     5.89
    16 |   1234.5 |   1298.2 |         2078 |     8.12
    32 |   2345.6 |   2412.3 |         2189 |    12.34
    64 |   4567.8 |   4698.1 |         2256 |    18.90

[推荐] batch_size=64, 吞吐量=2256 tokens/s
```

---

## 六、生产级批处理框架

把以上所有技术整合成一个完整的框架：

```python
#!/usr/bin/env python3
"""vLLM 生产级批量处理框架"""

import json
import time
import torch
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """批处理配置"""
    model: str
    output_dir: str = "./output"
    batch_size: int = 64
    max_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.95
    enable_checkpoint: bool = True
    checkpoint_interval: int = 500
    num_workers: int = 1
    task_name: str = "batch_task"


@dataclass
class BatchStats:
    """批处理统计"""
    total: int = 0
    success: int = 0
    failed: int = 0
    retried: int = 0
    total_time_s: float = 0.0
    tokens_generated: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.success / max(self.total, 1)
    
    @property
    def throughput(self) -> float:
        return self.total / max(self.total_time_s, 1)
    
    def summary(self) -> str:
        return (
            f"Total: {self.total} | Success: {self.success} ({self.success_rate:.1%}) | "
            f"Failed: {self.failed} | Retried: {self.retried} | "
            f"Time: {self.total_time_s:.1f}s | Throughput: {self.throughput:.0f}/s"
        )


class ProductionBatchProcessor:
    """生产级批量处理器"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = BatchStats()
        self.llm = None
    
    def _init_llm(self):
        if self.llm is None:
            logger.info(f"Loading model: {self.config.model}")
            self.llm = LLM(model=self.config.model, dtype="auto")
            logger.info("Model loaded successfully")
    
    def _get_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
    
    def _load_completed_set(self) -> set:
        ckpt = self.output_dir / f"{self.config.task_name}_checkpoint.json"
        if ckpt.exists():
            with open(ckpt) as f:
                data = json.load(f)
                return set(data["completed"])
        return set()
    
    def _save_checkpoint(self, completed: set):
        if self.config.enable_checkpoint:
            ckpt = self.output_dir / f"{self.config.task_name}_checkpoint.json"
            with open(ckpt, 'w') as f:
                json.dump({"completed": sorted(completed)}, f)
    
    def process(self, prompts: List[str]) -> List[dict]:
        """执行完整批处理流程"""
        
        self._init_llm()
        sp = self._get_sampling_params()
        completed_set = self._load_completed_set()
        
        output_file = self.output_dir / f"{self.config.task_name}_results.jsonl"
        
        pending = [(i, p) for i, p in enumerate(prompts) if i not in completed_set]
        self.stats.total = len(prompts)
        
        logger.info(f"Task: {self.config.task_name}")
        logger.info(f"Total: {len(prompts)}, Already done: {len(completed_set)}, "
                     f"Pending: {len(pending)}")
        
        start_time = time.time()
        round_completed = 0
        
        with tqdm(total=len(pending), desc=self.config.task_name) as pbar:
            for batch_start in range(0, len(pending), self.config.batch_size):
                batch = pending[batch_start:batch_start + self.config.batch_size]
                indices = [item[0] for item in batch]
                batch_prompts = [item[1] for item in batch]
                
                try:
                    outputs = self.llm.generate(batch_prompts, sp)
                    
                    with open(output_file, 'a') as f:
                        for idx, output in zip(indices, outputs):
                            record = {
                                "index": idx,
                                "text": output.outputs[0].text,
                                "finish_reason": output.outputs[0].finish_reason,
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            
                            self.stats.success += 1
                            self.stats.tokens_generated += len(output.outputs[0].token_ids)
                            completed_set.add(idx)
                    
                    pbar.update(len(batch))
                    round_completed += len(batch)
                    
                    if round_completed >= self.config.checkpoint_interval:
                        self._save_checkpoint(completed_set)
                        round_completed = 0
                
                except Exception as e:
                    logger.error(f"Batch error at offset {batch_start}: {e}")
                    for idx in indices:
                        with open(output_file, 'a') as f:
                            f.write(json.dumps({"index": idx, "error": str(e)}) + '\n')
                        self.stats.failed += 1
                        completed_set.add(idx)
                    pbar.update(len(batch))
                    
                    torch.cuda.empty_cache()
        
        self.stats.total_time_s = time.time() - start_time
        self._save_checkpoint(completed_set)
        
        logger.info(f"\n{self.stats.summary()}")
        logger.info(f"Output: {output_file}")
        
        return self.stats


def main():
    """主函数入口"""
    
    config = BatchConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="./production_output",
        batch_size=32,
        max_tokens=48,
        temperature=0.3,
        task_name="sentiment_analysis_v2",
        checkpoint_interval=100,
    )
    
    prompts = [
        f"分析以下评论的情感倾向（正面/负面/中性）：用户评论第{i}条内容"
        for i in range(1000)
    ]
    
    processor = ProductionBatchProcessor(config)
    stats = processor.process(prompts)


if __name__ == "__main__":
    main()
```

---

## 七、总结

本节覆盖了生产级批量处理的全部关键技能：

| 技术 | 解决的问题 | 核心要点 |
|------|-----------|---------|
| **二分法找最优 BS** | 不知道该用多大 batch | 从小到大试探直到 OOM，回退一步 |
| **自适应 Batch Size** | 不同阶段可用显存不同 | 实时估算每样本显存消耗，动态调整 |
| **断点续传** | 大任务中途崩溃 | JSON 格式保存已完成的索引集合 |
| **失败隔离重跑** | 部分样本出错 | 先全量跑完再统一扫描修复失败项 |
| **多 GPU 并行** | 单卡太慢 | 多进程 + `CUDA_VISIBLE_DEVICES` 绑定 |
| **tqdm 进度条** | 不知道跑到哪了 | 自定义 Processor 包装 tqdm |
| **JSONL 持久化** | 结果存储 | 流式追加，每行一个 JSON 对象 |
| **Parquet 导出** | 百万级数据分析 | 列式存储，压缩比 3-5x |
| **Benchmark 工具** | 选型决策依据 | 自动化测量 P50/P99/Throughput/VRAM |

**核心要点回顾**：

1. **最优 batch size 不是固定的**——它取决于模型大小、max_tokens、prompt 长度和可用显存，需要实际测量
2. **断点续传是大规模任务的必备能力**——不要假设你的脚本一定能一次跑完
3. **先完成后完美**——第一遍全量跑完（包括失败的），第二遍专门修复失败项
4. **JSONL > JSON > CSV**——对于结构化日志和结果，JSONL 是最佳选择
5. **Benchmark 数据驱动调优**——用数据而不是直觉来选择配置参数

下一节我们将学习 **特殊推理场景**，包括 LogProb 提取、Beam Search、结构化输出等高级用法。
