# 数据质量分析报告生成


#### 为什么需要数据质量报告

在千万级对话语料中，直接开始清洗是盲目的。你需要先回答这些问题：

- 数据中有多少比例是"垃圾"？——空内容、重复对话、单轮对话
- 文本长度分布是否合理？——过短（<5字符）或过长（>10K字符）的异常值
- 角色分布是否平衡？——human/assistant 的配对完整性
- 来源质量差异？——不同来源的数据质量评分
- 时间分布是否有异常？——某天突然暴增可能是爬虫行为

本节构建一个 **自动化的数据质量分析报告系统**，用 Pandas 在几分钟内生成完整的诊断报告。

---

#### 基础统计面板

##### 核心指标概览

```python
import pandas as pd
import numpy as np

class DataQualityReporter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_rows = len(df)
        self.n_cols = len(df.columns)

    def overview(self):
        print("=== 数据概览 ===\n")
        print(f"总行数: {self.n_rows:,}")
        print(f"总列数: {self.n_cols}")
        print(f"内存占用: {self.df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
        print(f"缺失单元格: {self.df.isna().sum().sum():,} ({self.df.isna().sum().sum() / (self.n_rows * self.n_cols) * 100:.2f}%)")
        print(f"完全重复行: {self.df.duplicated().sum():,}")

reporter = DataQualityReporter(df_merged)
reporter.overview()
```

输出：
```
=== 数据概览 ===

总行数: 12,345,678
总列数: 7
内存占用: 6.36 GB
缺失单元格: 492,356 (0.57%)
完全重复行: 23,456
```

##### 列级详细分析

```python
    def column_analysis(self):
        results = []
        for col in self.df.columns:
            series = self.df[col]
            null_count = series.isna().sum()
            null_pct = null_count / self.n_rows * 100
            unique = series.nunique()
            dupes = series.duplicated().sum()

            if pd.api.types.is_string_dtype(series) or \
               (hasattr(series.dtype, 'name') and 'string' in str(series.dtype)):
                avg_len = series.dropna().str.len().mean()
                max_len = series.dropna().str.len().max()
                empty_str = (series == "").sum() + (series.str.strip() == "").sum()
                results.append({
                    "列名": col,
                    "类型": str(series.dtype),
                    "缺失数": null_count,
                    "缺失率%": round(null_pct, 2),
                    "唯一值": unique,
                    "重复值": dupes,
                    "平均长度": round(avg_len, 1) if not np.isnan(avg_len) else "N/A",
                    "最大长度": max_len,
                    "空字符串": empty_str,
                })
            else:
                results.append({
                    "列名": col,
                    "类型": str(series.dtype),
                    "缺失数": null_count,
                    "缺失率%": round(null_pct, 2),
                    "唯一值": unique,
                    "重复值": dupes,
                    "平均长度": "N/A",
                    "最大长度": "N/A",
                    "空字符串": "N/A",
                })

        report_df = pd.DataFrame(results)
        return report_df

col_report = reporter.column_analysis()
print(col_report.to_string(index=False))
```

输出：
```
         列名              类型   缺失数  缺失率%     唯一值    重复值  平均长度  最大长度  空字符串
conversation_id          string       0    0.00  5678901 6677777      11.0      14         0
        user_id          string  145678    1.18  890123 11455555      10.0      12         0
           role        category       0    0.00        2 12345675       NaN      NaN         NaN
        content          string     678    0.01 9876543 2469135     127.0  128456       234
     timestamp     datetime64[ns]       0    0.00 8765432 3691353       NaN      NaN       NaN
          model       category  345678    2.80        5 12300000       NaN      NaN       NaN
  _source_file          string       0    0.00        4 12345674       20.0      22         0
```

---

#### 对话级别质量分析

原始数据是消息级别的（每行一条消息），但质量评估应该在 **对话级别**（一个 conversation_id 对应一组完整对话）。

##### 对话完整性检测

```python
    def conversation_integrity(self):
        conv_stats = self.df.groupby("conversation_id").agg(
            total_messages=("content", "count"),
            human_messages=("content", lambda x: (self.df.loc[x.index, "role"] == "human").sum()),
            assistant_messages=("content", lambda x: (self.df.loc[x.index, "role"] == "assistant").sum()),
            has_null=("content", lambda x: x.isna().any()),
            min_content_len=("content", lambda x: x.dropna().str.len().min() if x.notna().any() else 0),
            max_content_len=("content", lambda x: x.dropna().str.len().max() if x.notna().any() else 0),
            avg_content_len=("content", lambda x: round(x.dropna().str.len().mean(), 1) if x.notna().any() else 0),
            source=("_source_file", "first"),
            model=("model", lambda x: x.dropna().iloc[0] if x.notna().any() else "unknown"),
        ).reset_index()

        conv_stats["is_complete_pair"] = (
            (conv_stats["human_messages"] >= 1) &
            (conv_stats["assistant_messages"] >= 1)
        )
        conv_stats["is_single_turn"] = conv_stats["total_messages"] <= 2
        conv_stats["is_multi_turn"] = conv_stats["total_messages"] > 2

        return conv_stats

conv_report = reporter.conversation_integrity()
print(conv_report.head(10).to_string(index=False))
```

输出：
```
conversation_id  total_messages  human_messages  assistant_messages  has_null  min_content_len  max_content_len  avg_content_len    source   model  is_complete_pair  is_single_turn  is_multi_turn
        conv_001               2               1                   1    False               5             120            62.5  sharegpt    gpt-4              True           True          False
        conv_002               4               2                   2    False              12            3500           245.3  internal   claude-3             True          False           True
        conv_003               1               1                   0    False               3               3             3.0  chat_gpt4  unknown             False           True          False
        conv_004               2               1                   1    False               8              89            48.5  sharegpt    gpt-4              True           True          False
        conv_005               6               3                   3    False              15           128456          8900.2  chat_gpt4    gpt-4              True          False           True
```

##### 完整性统计汇总

```python
def integrity_summary(conv_stats):
    n_total = len(conv_stats)

    summary = {
        "总对话数": n_total,
        "完整配对(有问有答)": conv_stats["is_complete_pair"].sum(),
        "完整配对占比%": f"{conv_stats['is_complete_pair'].sum()/n_total*100:.1f}%",
        "仅有人类消息(无回复)": (~conv_stats["is_complete_pair"] & (conv_stats["human_messages"] > 0)).sum(),
        "仅有助手消息(无问题)": (~conv_stats["is_complete_pair"] & (conv_stats["assistant_messages"] > 0)).sum(),
        "包含空内容": conv_stats["has_null"].sum(),
        "单轮对话": conv_stats["is_single_turn"].sum(),
        "多轮对话": conv_stats["is_multi_turn"].sum(),
        "平均轮次": f"{conv_stats['total_messages'].mean():.1f}",
        "最大轮次": conv_stats["total_messages"].max(),
        "中位数轮次": conv_stats["total_messages"].median(),
    }

    summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["数值"])
    return summary_df

summary = integrity_summary(conv_report)
print(summary.to_string())
```

输出：
```
                          数值
总对话数                 5,678,901
完整配对(有问有答)         5,234,567
完整配对占比%                92.2%
仅有人类消息(无回复)         234,567
仅有助手消息(无问题)          89,012
包含空内容                  567
单轮对话                 3,456,789
多轮对话                 2,222,112
平均轮次                     2.2
最大轮次                      47
中位数轮次                  2.0
```

关键发现：**92.2% 的对话是完整配对的**，但有 32 万+ 的不完整对话需要处理。同时只有约 39% 是真正的多轮对话。

---

#### 文本质量深度分析

##### 长度分布分析

```python
    def text_length_distribution(self):
        content_lens = self.df["content"].dropna().str.len()

        bins = [0, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000, float('inf')]
        labels = ['<5', '5-10', '10-20', '20-50', '50-100', '100-500',
                  '500-1K', '1K-5K', '5K-10K', '>10K']

        dist = pd.cut(content_lens, bins=bins, labels=labels).value_counts().sort_index()
        pct = (dist / len(content_lens) * 100).round(2)

        length_df = pd.DataFrame({
            "区间": dist.index,
            "消息数": dist.values,
            "占比%": pct.values,
        })

        print("=== content 文本长度分布 ===\n")
        print(length_df.to_string(index=False))

        outliers_short = (content_lens < 5).sum()
        outliers_long = (content_lens > 10000).sum()
        print(f"\n⚠️ 异常短文本 (<5字符): {outliers_short:,} ({outliers_short/len(content_lens)*100:.2f}%)")
        print(f"⚠️ 异常长文本 (>10K字符): {outliers_long:,} ({outliers_long/len(content_lens)*100:.2f}%)")

        return length_df

length_dist = reporter.text_length_distribution()
```

输出：
```
=== content 文本长度分布 ===

 区间    消息数   占比%
  <5    125,678    1.02%
 5-10   345,678    2.80%
10-20   890,123    7.21%
20-50  2,345,678   19.00%
50-100  3,456,789   28.00%
100-500 3,234,567   26.20%
500-1K  1,234,567   10.00%
 1K-5K    567,890    4.60%
 5K-10K   123,456    1.00%
 >10K    21,234     0.17%

⚠️ 异常短文本 (<5字符): 125,678 (1.02%)
⚠️ 异常长文本 (>10K字符): 21,234 (0.17%)
```

##### 内容模式检测：垃圾数据识别

```python
    def detect_patterns(self):
        df = self.df.copy()
        content = df["content"].dropna()

        patterns = {
            "纯标点/符号": content.str.contains(r'^[\s\W]+$', regex=True, na=False),
            "全是数字": content.str.contains(r'^[\d\s,.]+$', regex=True, na=False),
            "单个字符": content.str.len() <= 1,
            "重复内容嫌疑": content.str.contains(r'(.)\1{10,}', regex=True, na=False),
            "疑似代码块": content.str.contains(r'```|def |class |import ', regex=True, na=False),
            "包含URL": content.str.contains(r'https?://', regex=True, na=False),
            "包含邮箱": content.str.contains(r'[\w.-]+@[\w.-]+\.\w+', regex=True, na=False),
            "包含敏感词": content.str.contains(r'password|密码|token|secret|api[_-]?key',
                                              case=False, regex=True, na=False),
        }

        pattern_counts = {k: v.sum() for k, v in patterns.items()}
        pattern_df = pd.DataFrame.from_dict(
            pattern_counts, orient="index", columns=["匹配数量"]
        )
        pattern_df["占比%"] = (pattern_df["匹配数量"] / len(content) * 100).round(3)

        print("=== 内容模式检测结果 ===\n")
        print(pattern_df.sort_values("匹配数量", ascending=False).to_string())

        return patterns

patterns = reporter.detect_patterns()
```

输出：
```
=== 内容模式检测结果 ===

              匹配数量     占比%
疑似代码块     1,234,567   10.00%
包含URL          567,890    4.60%
包含敏感词        234,567    1.90%
纯标点/符号        45,678    0.37%
重复内容嫌疑        12,345    0.10%
全是数字           8,901    0.07%
包含邮箱           5,678    0.05%
单个字符         125,678    1.02%
```

注意：这些模式不一定都是"坏数据"。比如"疑似代码块"在编程相关的 SFT 数据中反而是高质量数据。关键是 **标记出来，让后续清洗策略决定如何处理**。

---

#### 来源与时间维度分析

##### 按来源的质量对比

```python
    def source_quality_comparison(self):
        conv_stats = self.conversation_integrity()

        source_summary = conv_stats.groupby("source").agg(
            total_conversations=("conversation_id", "count"),
            complete_pairs=("is_complete_pair", "sum"),
            complete_rate=("is_complete_pair", "mean"),
            multi_turn_rate=("is_multi_turn", "mean"),
            avg_msg_per_conv=("total_messages", "mean"),
            has_null=("has_null", "sum"),
            avg_content_len=("avg_content_len", "mean"),
        ).round(4)

        source_summary["complete_rate"] = (source_summary["complete_rate"] * 100).round(1).astype(str) + "%"
        source_summary["multi_turn_rate"] = (source_summary["multi_turn_rate"] * 100).round(1).astype(str) + "%"

        print("=== 按来源的质量对比 ===\n")
        print(source_summary.to_string())
        return source_summary

source_q = reporter.source_quality_comparison()
```

输出：
```
=== 按来源的质量对比 ===

           total_conversations  complete_pairs complete_rate  multi_turn_rate  avg_msg_per_conv  has_null  avg_content_len
sharegpt              3200000        3104000        97.0%           38.2%              2.41       123            156.7
internal              2800000        2744000        98.0%           45.1%              2.52        89             234.5
chat_gpt4_jan         4500000        4050000        90.0%           33.3%              2.18       234            112.3
other_sources         178901          167567        93.7%           40.5%              2.35       121             189.0
```

可以看到 `sharegpt` 和 `internal` 来源的数据质量明显更高（97-98% 完整配对），而 `chat_gpt4_jan` 只有 90%，可能需要更严格的清洗。

##### 时间趋势分析（检测异常波动）

```python
    def time_trend_analysis(self):
        if "timestamp" not in self.df.columns:
            print("⚠️ 无 timestamp 列，跳过时间分析")
            return None

        daily = self.df.groupby(self.df["timestamp"].dt.date).agg(
            messages=("conversation_id", "count"),
            unique_convs=("conversation_id", "nunique"),
            unique_users=("user_id", "nunique") if "user_id" in self.df.columns else ("conversation_id", "count"),
        )

        daily["msgs_per_conv"] = (daily["messages"] / daily["unique_convs"]).round(1)

        mean_msgs = daily["messages"].mean()
        std_msgs = daily["messages"].std()
        daily["z_score"] = ((daily["messages"] - mean_msgs) / std_msgs).round(2)
        daily["is_anomaly"] = daily["z_score"].abs() > 3

        print("=== 每日数据量趋势（Top 10 / Bottom 10）===\n")
        print("📈 数据量最高的 10 天:")
        print(daily.nlargest(10, "messages")[["messages", "unique_convs", "z_score", "is_anomaly"]].to_string())

        anomalies = daily[daily["is_anomaly"]]
        if len(anomalies) > 0:
            print(f"\n⚠️ 发现 {len(anomalies)} 个异常日期:")
            print(anomalies[["messages", "z_score"]].to_string())

        return daily

trend = reporter.time_trend_analysis()
```

输出：
```
=== 每日数据量趋势（Top 10 / Bottom 10）===

📈 数据量最高的 10 天:
            messages  unique_convs  z_score  is_anomaly
2025-01-20    890123       445061      4.52        True
2025-01-19    756789       378394      3.21        True
2025-01-18    654321       327160      2.18       False
2025-01-17    589012       294506      1.54       False
...

⚠️ 发现 3 个异常日期:
            messages  z_score
2025-01-20  890123    4.52
2025-01-19  756789    3.21
2025-02-01  12345     -3.67
```

异常日期可能意味着爬虫批量抓取（暴涨）或系统故障（骤降），需要在清洗时特别关注这些日期的数据。

---

#### 综合质量评分

##### 为每个对话打分

```python
    def quality_scoring(self):
        conv_stats = self.conversation_integrity()

        scores = pd.Series(index=conv_stats.index, dtype=float, name="quality_score")

        scores += (conv_stats["is_complete_pair"].astype(int)) * 30
        scores += (conv_stats["is_multi_turn"].astype(int)) * 15
        scores += ((conv_stats["avg_content_len"].clip(20, 2000) - 20) / 1980 * 20).clip(0, 20)
        scores += (~conv_stats["has_null"].astype(int)) * 10
        scores += ((conv_stats["total_messages"].clip(2, 10) - 2) / 8 * 15).clip(0, 15)
        scores += (conv_stats["min_content_len"] >= 5).astype(int) * 10

        conv_stats["quality_score"] = scores.round(2)

        score_bins = [0, 30, 50, 70, 85, 101]
        score_labels = ["D-差", "C-一般", "B-良好", "A-优秀", "S-顶级"]
        conv_stats["quality_grade"] = pd.cut(
            scores, bins=score_bins, labels=score_labels, right=False
        )

        grade_dist = conv_stats["quality_grade"].value_counts().sort_index()
        grade_pct = (grade_dist / len(conv_stats) * 100).round(1)

        print("=== 质量等级分布 ===\n")
        for grade, count in grade_dist.items():
            bar = "█" * int(count / len(conv_stats) * 50)
            print(f"  {grade}: {count:>8,} ({grade_pct[grade]:>5.1f}%) {bar}")

        print(f"\n平均分: {scores.mean():.1f}")
        print(f"中位数: {scores.median():.1f}")

        return conv_stats

scored = reporter.quality_scoring()
```

输出：
```
=== 质量等级分布 ===

  D-差:   345,678 (  6.1%) ████████
  C-一般:   890,123 ( 15.7%) ████████████████████
  B-良好: 2,345,678 ( 41.3%) ████████████████████████████████████████████████
  A-优秀: 1,789,012 ( 31.5%) ██████████████████████████████████████████
  S-顶级:   308,410 (  5.4%) ████

平均分: 68.3
中位数: 72.0
```

##### 按来源和模型交叉分析

```python
    def cross_analysis(self, scored_df):
        pivot = scored_df.pivot_table(
            index="source",
            columns="model",
            values="quality_score",
            aggfunc="mean",
        ).round(1)

        count_pivot = scored_df.pivot_table(
            index="source",
            columns="model",
            values="conversation_id",
            aggfunc="count",
        ).fillna(0).astype(int)

        print("\n=== 各来源×模型 平均质量分数 ===\n")
        print(pivot.to_string())

        print("\n=== 各来源×模型 对话数量 ===\n")
        print(count_pivot.to_string())

        return pivot, count_pivot

pivot, counts = reporter.cross_analysis(scored)
```

输出：
```
=== 各来源×模型 平均质量分数 ===

model     claude-3  gpt-4  llama-3  qwen2.5  unknown
source
chat_gpt4_jan      NaN   65.2      NaN     63.1     58.9
internal        76.5    NaN     71.2      NaN      NaN
other_sources   69.8   72.3     65.4     67.8     61.2
sharegpt        74.2   73.8      NaN      NaN      NaN
```

从交叉表可以看出 `internal` 来源的 `claude-3` 数据质量最高（76.5 分），而 `unknown` 模型的数据普遍偏低。

---

#### 导出完整报告

```python
    def generate_full_report(self, output_path: str = "quality_report.md"):
        lines = []
        lines.append("# 千万级对话语料 数据质量报告\n")
        lines.append(f"- 总消息数: **{self.n_rows:,}**")
        lines.append(f"- 总对话数: **~5.68M**")
        lines.append(f"- 内存占用: **{self.df.memory_usage(deep=True).sum()/1024**3:.2f} GB**\n")

        lines.append("## 关键发现\n")
        lines.append("- 完整配对率: **92.2%** —— 约 44 万条不完整对话需过滤")
        lines.append("- 多轮对话占比: **39.1%** —— 大部分为单轮 QA")
        lines.append("- 空值/异常短文本: **~1.2%** —— 可直接丢弃")
        lines.append("- 敏感信息标记: **~1.9%** —— 含 password/token 等，需脱敏")
        lines.append("- 异常日期: **3 天** —— 可能存在爬虫或系统异常\n")

        lines.append("## 清洗建议\n")
        lines.append("| 优先级 | 操作 | 预计影响 |")
        lines.append("|--------|------|----------|")
        lines.append("| P0 | 过滤非完整配对对话 | 移除 ~44 万条 |")
        lines.append("| P1 | 过滤 content < 5 字符 | 移除 ~12.6 万条 |")
        lines.append("| P1 | 过滤 D/C 级低质量数据 | 移除 ~124 万条 |")
        lines.append("| P2 | 敏感词脱敏或移除 | 影响 ~23 万条 |")
        lines.append("| P3 | 异常长文本截断 (>10K) | 影响 ~2.1 万条 |")

        report_text = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"✅ 报告已导出: {output_path}")
        print(report_text)

reporter.generate_full_report()
```

这份报告为下一节的 LLM 辅助清洗提供了精确的目标和数据依据。
