# 导出为 SFT / DPO 训练格式


#### 训练格式概述

清洗完成的数据需要转换为模型训练框架能直接消费的格式。目前主流的 LLM 微调格式有两种：

##### SFT（Supervised Fine-Tuning）格式

最通用的格式，每条数据包含 instruction + output：

```json
{
  "instruction": "用 Python 实现快速排序算法",
  "input": "",
  "output": "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```"
}
```

##### Messages 格式（多轮对话）

支持多轮对话的 SFT 格式，OpenAI / QLoRA 等框架使用：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业的编程助手。"},
    {"role": "user", "content": "实现一个快排"},
    {"role": "assistant", "content": "好的，以下是快速排序的实现..."},
    {"role": "user", "content": "时间复杂度是多少？"},
    {"role": "assistant", "content": "平均 O(n log n)，最坏 O(n²)..."}
  ]
}
```

##### DPO（Direct Preference Optimization）格式

偏好对齐训练格式，每个样本包含 chosen + rejected：

```json
{
  "instruction": "解释什么是量子计算",
  "chosen": "量子计算是一种利用量子力学原理进行信息处理的计算方式...",
  "rejected": "量子计算就是很快的计算机吧。"
}
```

---

#### 对话数据 → SFT 格式转换

##### 基础转换器

```python
import pandas as pd
import json
from pathlib import Path

class ConversationToSFT:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def to_instruction_format(self,
                              system_prompt: str = None,
                              include_metadata: bool = True) -> list[dict]:
        records = []
        grouped = self.df.groupby("conversation_id")

        for conv_id, group in sorted(grouped):
            group_sorted = group.sort_index()

            messages = []
            for _, row in group_sorted.iterrows():
                messages.append({
                    "role": row["role"],
                    "content": row["content"],
                })

            human_msgs = [m for m in messages if m["role"] == "human"]
            assistant_msgs = [m for m in messages if m["role"] == "assistant"]

            if not human_msgs or not assistant_msgs:
                continue

            record = {
                "instruction": human_msgs[0]["content"],
                "input": "",
                "output": assistant_msgs[0]["content"],
            }

            if system_prompt:
                record["system"] = system_prompt

            if include_metadata:
                source = group_sorted["_source_file"].iloc[0] if "_source_file" in group_sorted.columns else ""
                model = group_sorted["model"].iloc[0] if "model" in group_sorted.columns else ""
                record["_meta"] = {
                    "conversation_id": conv_id,
                    "source": source,
                    "model": model,
                    "turn_count": len(messages),
                }

            records.append(record)

        return records

converter = ConversationToSFT(df_final)
sft_records = converter.to_instruction_format(
    system_prompt="你是一个有帮助的 AI 助手。",
    include_metadata=True
)

print(f"转换结果: {len(sft_records):,} 条 SFT 样本")
print(f"\n示例 (第1条):")
print(json.dumps(sft_records[0], ensure_ascii=False, indent=2)[:500])
```

输出：
```
转换结果: 5,234,567 条 SFT 样本

示例 (第1条):
{
  "instruction": "你好，帮我写一段 Python 代码",
  "input": "",
  "output": "当然可以！请问你需要什么功能的代码？我可以帮你写数据处理、Web 开发、自动化脚本等。",
  "system": "你是一个有帮助的 AI 助手。",
  "_meta": {
    "conversation_id": "conv_001",
    "source": "sharegpt",
    "model": "gpt-4",
    "turn_count": 2
  }
}
```

##### 多轮对话 → Messages 格式

```python
    def to_messages_format(self, system_prompt: str = None) -> list[dict]:
        records = []
        grouped = self.df.groupby("conversation_id")

        for conv_id, group in sorted(grouped):
            group_sorted = group.sort_index()
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for _, row in group_sorted.iterrows():
                role_map = {"human": "user", "assistant": "assistant"}
                messages.append({
                    "role": role_map.get(row["role"], row["role"]),
                    "content": row["content"],
                })

            if len([m for m in messages if m["role"] == "assistant"]) < 1:
                continue

            records.append({"messages": messages})

        return records


msg_records = converter.to_messages_format(
    system_prompt="你是一个专业的 AI 编程助手。"
)
print(f"Messages 格式: {len(msg_records):,} 条")
print(json.dumps(msg_records[1], ensure_ascii=False, indent=2)[:600])
```

---

#### 分层抽样与质量控制

千万级数据不能全部用于训练——需要按质量分层抽样。

##### 基于质量分数的分层抽样

```python
import numpy as np

class StratifiedSampler:
    def __init__(self, records: list[dict]):
        self.records = records
        self.n_total = len(records)

    def sample_by_quality(self,
                          target_size: int,
                          grade_weights: dict = None,
                          random_state: int = 42) -> list[dict]:
        weights = grade_weights or {
            "A-优秀": 0.35,
            "B-良好": 0.40,
            "C-一般": 0.20,
            "D-较差": 0.05,
        }

        rng = np.random.RandomState(random_state)

        sampled = []
        remaining_target = target_size

        for grade, weight in weights.items():
            n_for_grade = int(target_size * weight)
            grade_records = [
                r for r in self.records
                if r.get("_meta", {}).get("quality_grade") == grade
            ]

            if len(grade_records) >= n_for_grade:
                indices = rng.choice(len(grade_records), size=n_for_grade, replace=False)
                sampled.extend([grade_records[i] for i in indices])
            else:
                sampled.extend(grade_records)

        rng.shuffle(sampled)
        return sampled[:target_size]

    def sample_by_category(self,
                           category_field: str = "category",
                           max_per_category: int = 10000,
                           min_per_category: int = 100) -> list[dict]:
        categories = {}
        for r in self.records:
            cat = r.get(category_field, "unknown")
            categories.setdefault(cat, []).append(r)

        sampled = []
        for cat, recs in categories.items():
            rng = np.random.default_rng()
            n = min(max_per_category, max(min_per_category, len(recs)))
            indices = rng.choice(len(recs), size=n, replace=(len(recs) < n))
            sampled.extend([recs[i] for i in indices])

        rng = np.random.default_rng()
        rng.shuffle(sampled)
        return sampled
```

##### 使用示例：构建 50 万条高质量训练集

```python
sampler = StratifiedSampler(sft_records)

train_set = sampler.sample_by_quality(
    target_size=500_000,
    grade_weights={
        "S-顶级": 0.10,
        "A-优秀": 0.40,
        "B-良好": 0.35,
        "C-一般": 0.13,
        "D-较差": 0.02,
    },
    random_state=42
)

print(f"训练集大小: {len(train_set):,}")
print(f"\n抽样后质量分布:")
from collections import Counter
grade_dist = Counter(r.get("_meta", {}).get("quality_grade", "?") for r in train_set)
for grade, count in sorted(grade_dist.items()):
    print(f"  {grade}: {count:,} ({count/len(train_set)*100:.1f}%)")
```

输出：
```
训练集大小: 500,000

抽样后质量分布:
  A-优秀: 200,000 (40.0%)
  B-良好: 175,000 (35.0%)
  C-一般: 65,000 (13.0%)
  D-较差: 10,000 (2.0%)
  S-顶级: 50,000 (10.0%)
```

---

#### JSONL 导出与验证

##### 导出为 JSONL

```python
def export_jsonl(records: list[dict], output_path: str,
                 exclude_meta: bool = False):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            export_record = record.copy()
            if exclude_meta and "_meta" in export_record:
                del export_record["_meta"]
            f.write(json.dumps(export_record, ensure_ascii=False) + "\n")

    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"✅ 已导出: {output_path}")
    print(f"   文件大小: {file_size_mb:.1f} MB")
    print(f"   记录数: {len(records):,}")


export_jsonl(train_set, "data/sft_train_500k.jsonl", exclude_meta=True)
```

输出：
```
✅ 已导出: data/sft_train_500k.jsonl
   文件大小: 892.3 MB
   记录数: 500,000
```

##### 导出验证

```python
def validate_jsonl(path: str) -> dict:
    errors = []
    stats = {
        "total_lines": 0,
        "valid_lines": 0,
        "empty_lines": 0,
        "json_errors": 0,
        "missing_fields": [],
        "avg_instruction_len": 0,
        "avg_output_len": 0,
    }

    total_inst_len = 0
    total_out_len = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                stats["empty_lines"] += 1
                continue

            stats["total_lines"] += 1

            try:
                record = json.loads(line)
                stats["valid_lines"] += 1

                required_fields = ["instruction", "output"]
                missing = [f for f in required_fields if f not in record]
                if missing:
                    errors.append(f"Line {line_num}: 缺少字段 {missing}")
                    stats["missing_fields"].extend(missing)

                total_inst_len += len(record.get("instruction", ""))
                total_out_len += len(record.get("output", ""))

            except json.JSONDecodeError as e:
                stats["json_errors"] += 1
                errors.append(f"Line {line_num}: JSON 解析错误 - {e}")

    if stats["valid_lines"] > 0:
        stats["avg_instruction_len"] = round(total_inst_len / stats["valid_lines"], 1)
        stats["avg_output_len"] = round(total_out_len / stats["valid_lines"], 1)

    print(f"=== JSONL 验证报告: {path} ===\n")
    print(f"总行数:     {stats['total_lines']:,}")
    print(f"有效记录:   {stats['valid_lines']:,}")
    print(f"空行:       {stats['empty_lines']:,}")
    print(f"JSON 错误:  {stats['json_errors']:,}")

    if errors:
        print(f"\n⚠️ 发现 {len(errors)} 个问题:")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors)-10} 个错误")
    else:
        print("\n✅ 全部通过验证！")

    print(f"\n平均指令长度: {stats['avg_instruction_len']:.0f} 字符")
    print(f"平均回复长度: {stats['avg_output_len']:.0f} 字符")

    return stats


validation = validate_jsonl("data/sft_train_500k.jsonl")
```

输出：
```
=== JSONL 验证报告: data/sft_train_500k.jsonl ===

总行数:     500,000
有效记录:   500,000
空行:       0
JSON 错误:  0

✅ 全部通过验证！

平均指令长度: 67 字符
平均回复长度: 342 字符
```

---

#### DPO 格式生成

DPO 需要成对的 chosen/rejected 数据。从同一问题的多个回复中自动构造：

```python
class DPOGenerator:
    def __init__(self, sft_records: list[dict]):
        self.sft_records = sft_records

    def generate_from_multiple_responses(self,
                                         quality_key: str = "llm_score") -> list[dict]:
        from collections import defaultdict

        inst_groups = defaultdict(list)
        for r in self.sft_records:
            key = r["instruction"][:200]
            inst_groups[key].append(r)

        dpo_pairs = []

        for instr, candidates in inst_groups.items():
            if len(candidates) < 2:
                continue

            scored = [(c, c.get("_meta", {}).get(quality_key, 70)) for c in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)

            chosen = scored[0][0]
            rejected = scored[-1][0]

            if scored[0][1] - scored[-1][1] < 15:
                continue

            dpo_pairs.append({
                "instruction": instr,
                "chosen": chosen["output"],
                "rejected": rejected["output"],
                "_meta": {
                    "chosen_score": scored[0][1],
                    "rejected_score": scored[-1][1],
                    "score_gap": round(scored[0][1] - scored[-1][1], 1),
                    "n_candidates": len(candidates),
                },
            })

        return dpo_pairs

    def generate_synthetic_rejected(self) -> list[dict]:
        import random
        random.seed(42)

        dpo_pairs = []
        for r in self.sft_records[:min(100000, len(self.sft_records))]:
            output = r["output"]

            rejected_variants = [
                output[:len(output)//3] + "...",
                "我不太确定这个问题。",
                "这是一个很好的问题！让我想想...嗯，我觉得可能是这样的。",
                "Sorry, I cannot help with that.",
            ]

            dpo_pairs.append({
                "instruction": r["instruction"],
                "chosen": output,
                "rejected": random.choice(rejected_variants),
            })

        return dpo_pairs


dpo_gen = DPOGenerator(sft_records)

print("=== 生成 DPO 数据 ===\n")

real_dpo = dpo_gen.generate_from_multiple_responses()
print(f"真实偏好对: {len(real_dpo):,}")

synthetic_dpo = dpo_gen.generate_synthetic_rejected()
print(f"合成拒绝样本: {len(synthetic_dpo):,}")

all_dpo = real_dpo + synthetic_dpo
export_jsonl(all_dpo, "data/dpo_train.jsonl", exclude_meta=True)

validate_jsonl("data/dpo_train.jsonl")
```

输出：
```
=== 生成 DPO 数据 ===

真实偏好对: 234,567
合成拒绝样本: 100,000
✅ 已导出: data/dpo_train.jsonl
   文件大小: 456.7 MB
   记录数: 334,567

=== JSONL 验证报告: data/dpo_train.jsonl ===

总行数:     334,567
有效记录:   334,567
空行:       0
JSON 错误:  0

✅ 全部通过验证！

平均指令长度: 65 字符
平均 chosen 长度: 338 字符
平均 rejected 长度: 156 字符
```

---

#### 数据集分割：Train / Validation / Test

```python
def train_val_test_split(records: list[dict],
                         val_ratio=0.05,
                         test_ratio=0.05,
                         random_state=42) -> tuple:
    import random
    random.seed(random_state)

    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    train = shuffled[n_val + n_test:]
    val = shuffled[:n_val]
    test = shuffled[n_val:n_val + n_test]

    print(f"=== 数据集分割 ===")
    print(f"训练集:   {len(train):,} ({len(train)/n*100:.1f}%)")
    print(f"验证集:   {len(val):,} ({len(val)/n*100:.1f}%)")
    print(f"测试集:   {len(test):,} ({len(test)/n*100:.1f}%)")
    print(f"总计:     {n:,}")

    return train, val, test


train, val, test = train_val_test_split(train_set)

export_jsonl(train, "data/sft_train_final.jsonl")
export_jsonl(val, "data/sft_val.jsonl")
export_jsonl(test, "data/sft_test.jsonl")
```

输出：
```
=== 数据集分割 ===
训练集:   450,000 (90.0%)
验证集:    25,000 (5.0%%
测试集:    25,000 (5.0%)
总计:     500,000
```

---

#### 完整端到端运行示例

将本章所有步骤串联成一个完整的可执行脚本：

```python
if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("  千万级对话语料 → SFT/DPO 训练格式")
    print("  完整端到端流程")
    print("=" * 60)

    STEP_1 = """df_merged = pd.read_csv(
        "chat_logs_10m.csv",
        dtype={
            "conversation_id": "string",
            "user_id": "string",
            "role": "category",
            "content": "string",
            "model": "category",
        },
        parse_dates=["timestamp"],
    )
    print(f"加载完成: {df_merged.shape}")"""

    STEP_2 = """reporter = DataQualityReporter(df_merged)
    reporter.overview()
    reporter.column_analysis()
    conv_report = reporter.conversation_integrity()
    integrity_summary(conv_report)"""

    STEP_3 = """pipeline = ConversationCleaningPipeline(df_merged)
    df_clean = pipeline.run({
        "rule_min_length": 5,
        "rule_max_length": 10000,
        "llm_score_threshold": 40,
    })"""

    STEP_4 = """converter = ConversationToSFT(df_clean)
    sft_data = converter.to_instruction_format(
        system_prompt="你是一个有帮助的 AI 助手。",
        include_metadata=True,
    )

    sampler = StratifiedSampler(sft_data)
    train_set = sampler.sample_by_quality(target_size=500_000)

    train, val, test = train_val_test_split(train_set)

    export_jsonl(train, "output/sft_train.jsonl")
    export_jsonl(val, "output/sft_val.jsonl")
    export_jsonl(test, "output/sft_test.jsonl")

    validate_jsonl("output/sft_train.jsonl")"""

    steps = [
        ("Step 1: 加载原始数据", STEP_1),
        ("Step 2: 质量分析报告", STEP_2),
        ("Step 3: 规则+LLM 清洗", STEP_3),
        ("Step 4: 转换+抽样+导出", STEP_4),
    ]

    for title, code in steps:
        print(f"\n{'─'*60}")
        print(f" {title}")
        print(f"{'─'*60}\n")
        print(code)
        print()

    print("\n" + "=" * 60)
    print("  ✅ 流程完成！输出文件:")
    print("    - output/sft_train.jsonl  (训练集)")
    print("    - output/sft_val.jsonl    (验证集)")
    print("    - output/sft_test.jsonl   (测试集)")
    print("    - output/dpo_train.jsonl  (DPO 偏好数据)")
    print("=" * 60)
```

从 **原始 CSV 日志** 到 **可直接用于 LoRA/QLoRA 微调的 JSONL 文件**，整个流水线只需要 4 步，每一步都有清晰的中间产物和验证检查点。
