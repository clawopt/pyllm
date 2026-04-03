---
title: 数据质量报告生成模板
description: 一键生成完整的数据质量报告，包含形状、类型、缺失、重复、分布等全方位分析，可直接用于数据验收
---
# 数据质量报告生成


## 为什么需要自动化报告

每次拿到新数据时手动检查太慢且容易遗漏。一个**标准化的报告模板**可以：

1. **标准化流程**：团队内统一的数据审查标准
2. **可追溯**：每次处理都有记录存档
3. **快速决策**：一眼看出数据是否可用

## 完整报告模板

```python
import pandas as pd
import numpy as np
from datetime import datetime


class DataQualityReport:
    """LLM 数据集质量分析报告生成器"""
    
    def __init__(self, df, name="Dataset"):
        self.df = df.copy()
        self.name = name
        self.report_time = datetime.now()
        self.sections = []
    
    def _add_section(self, title, content):
        self.sections.append(f"\n{'█'*60}\n  {title}\n{'█'*60}")
        self.sections.append(content)
    
    def generate(self):
        """生成完整报告"""
        
        df = self.df
        
        self._basic_info()
        self._dtype_analysis()
        self._missing_analysis()
        self._duplicate_analysis()
        self._numeric_analysis()
        self._categorical_analysis()
        self._text_analysis()
        self._memory_analysis()
        self._recommendations()
        
        report = '\n'.join(self.sections)
        print(report)
        return report
    
    def _basic_info(self):
        df = self.df
        rows, cols = df.shape
        content = f"""
  📐 基本信息
  ────────────────────────────────
  数据集名称:     {self.name}
  报告时间:       {self.report_time.strftime('%Y-%m-%d %H:%M:%S')}
  行数:           {rows:,}
  列数:           {cols}
  内存占用:       {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
  空DataFrame:   {'是' if df.empty else '否'}
  
  列名列表:
  {', '.join(df.columns.tolist())}
"""
        self._add_section("基本信息", content)
    
    def _dtype_analysis(self):
        df = self.df
        dtype_counts = df.dtypes.value_counts()
        
        lines = []
        for dtype, count in dtype_counts.items():
            cols_of_type = [c for c in df.columns if str(df[c].dtype) == str(dtype)]
            lines.append(f"  {str(dtype):<25s} {count:>3} 列  → {', '.join(cols_of_type[:5])}")
        
        content = f"""
  🔧 类型分布
  ────────────────────────────────
{chr(10).join(lines)}
  
  ⚠️ 需要关注的类型:
"""
        
        object_cols = [c for c in df.columns if str(df[c].dtype) == 'object']
        if object_cols:
            content += f"    - object 类型列 ({len(object_cols)} 个): {', '.join(object_cols[:5])}"
            content += "\n      建议转换为 string[pyarrow] 或 category"
        else:
            content += "    - 无"
        
        self._add_section("类型分析", content)
    
    def _missing_analysis(self):
        df = self.df
        missing = df.isna().sum()
        missing_pct = (df.isna().mean() * 100).round(2)
        
        has_missing = missing[missing > 0]
        
        if len(has_missing) > 0:
            table_rows = []
            for col in has_missing.index:
                cnt = missing[col]
                pct = missing_pct[col]
                bar_len = int(pct / 5)
                bar = '█' * max(bar_len, 1) + '░' * (20 - max(bar_len, 1))
                table_rows.append(
                    f"  {col:<22s} {cnt:>8,}  {pct:>6.1f}%  {bar}"
                )
            
            content = f"""
  ❌ 缺失值详情
  ────────────────────────────────
  有缺失值的列: {len(has_missing)}/{len(df.columns)}
  
  {'列名':<22s} {'缺失数':>8s} {'占比':>8s}  可视化
  {'─'*55}
{chr(10).join(table_rows)}
  
  总缺失单元格: {has_missing.sum():,} / {df.shape[0]*df.shape[1]:,} 
                   ({has_missing.sum()/(df.shape[0]*df.shape[1])*100:.2f}%)
"""
        else:
            content = """
  ✅ 缺失值检查
  ────────────────────────────────
  所有列均无缺失值！
"""
        
        self._add_section("缺失值分析", content)
    
    def _duplicate_analysis(self):
        df = self.df
        
        full_dup = df.duplicated().sum()
        partial_dups = {}
        
        for col in df.columns[:min(5, len(df.columns))]:
            n_dup = df.duplicated(subset=[col]).sum()
            if n_dup > 0:
                partial_dups[col] = n_dup
        
        if full_dup > 0 or partial_dups:
            dup_lines = [f"  完全重复行数: {full_dup:,} ({full_dup/len(df)*100:.2f}%)"]
            for col, n in list(partial_dups.items())[:3]:
                dup_lines.append(f"  按 [{col}] 去重可去除: {n:,} 条")
            
            content = f"""
  🔁 重复值分析
  ────────────────────────────────
{chr(10).join(dup_lines)}
"""
        else:
            content = """
  ✅ 重复值检查
  ────────────────────────────────
  未发现重复数据！
"""
        
        self._add_section("重复值分析", content)
    
    def _numeric_analysis(self):
        df = self.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self._add_section("数值分析", "  无数值型列\n")
            return
        
        desc = df[numeric_cols].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]).T
        
        content = f"""
  📊 数值统计摘要
  ────────────────────────────────
  数值列数量: {len(numeric_cols)}
  
  {desc.round(2).to_string()}
"""
        
        anomaly_hints = []
        for col in numeric_cols:
            q1 = desc.loc[col, '25%']
            q3 = desc.loc[col, '75%']
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            
            outliers_lower = (df[col] < lower_fence).sum()
            outliers_upper = (df[col] > upper_fence).sum()
            total_outliers = outliers_lower + outliers_upper
            
            if total_outliers > len(df) * 0.01:
                anomaly_hints.append(
                    f"  ⚠️ {col}: {total_outliers:,} 个异常值 "
                    f"(下界={lower_fence:.2f}, 上界={upper_fence:.2f})"
                )
        
        if anomaly_hints:
            content += "\n  异常值警告:\n" + '\n'.join(anomaly_hints)
        
        self._add_section("数值分析", content)
    
    def _categorical_analysis(self):
        df = self.df
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) == 0:
            return
        
        content = "  📑 分类变量分布\n  ────────────────────────────────\n"
        
        for col in cat_cols[:5]:
            n_unique = df[col].nunique()
            top5 = df[col].value_counts(normalize=True).head(5)
            
            content += f"\n  【{col}】({n_unique:,} 种唯一值)\n"
            for val, pct in top5.items():
                bar = '█' * int(pct * 30)
                display_val = str(val)[:40] if len(str(val)) > 40 else val
                content += f"    {bar:<32s} {display_val:<42s} {pct*100:.1f}%\n"
            
            if n_unique > 5:
                content += f"    ... 其余 {n_unique-5} 种值\n"
        
        self._add_section("分类分析", content)
    
    def _text_analysis(self):
        df = self.df
        text_cols = [c for c in df.columns 
                    if str(df[c].dtype) in ('object', 'string') and df[c].dtype != 'category']
        
        if not text_cols:
            return
        
        content = "  📝 文本列特征\n  ────────────────────────────────\n"
        
        for col in text_cols[:3]:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            
            lengths = s.str.len()
            content += f"""
  【{col}】
    平均长度:   {lengths.mean():.0f} 字符
    中位数长度: {lengths.median():.0f} 字符
    最短:       {lengths.min()} 字符
    最长:       {lengths.max():,} 字符
    空字符串:    {(s.str.strip() == '').sum():,} 条 ({(s.str.strip()=='').mean()*100:.2f}%)
"""
        
        self._add_section("文本分析", content)
    
    def _memory_analysis(self):
        df = self.df
        mem = df.memory_usage(deep=True)
        total = mem.sum()
        
        mem_df = pd.DataFrame({
            '列名': mem.index,
            '内存(MB)': (mem / 1024**2).round(2),
            '占比(%)': ((mem / total) * 100).round(1),
        }).sort_values('内存(MB)', ascending=False)
        
        top_heavy = mem_df.head(8)
        
        content = f"""
  💾 内存占用分析
  ────────────────────────────────
  总内存: {total / 1024**2:.1f} MB
  
  Top 8 内存消耗列:
  {'列名':<22s} {'内存(MB)':>10s} {'占比':>8s}
  {'─'*44}
"""
        for _, row in top_heavy.iterrows():
            content += f"  {row['列名']:<22s} {row['内存(MB)']:>9.2f}MB  {row['占比(%)']:>7.1f}%\n"
        
        obj_mem = df.select_dtypes(include=['object']).memory_usage(deep=True).sum()
        if obj_mem > total * 0.3:
            savings_potential = (obj_mem * 0.6) / 1024**2
            content += f"\n  💡 提示: object 列占 {obj_mem/total*100:.0f}% 内存，转 category/string[pyarrow] 可能节省 ~{savings_potential:.0f}MB"
        
        self._add_section("内存分析", content)
    
    def _recommendations(self):
        df = self.df
        recs = []
        
        if df.isna().sum().sum() > len(df) * 0.01:
            recs.append("1. 存在较多缺失值，建议先执行 fillna/dropna 处理")
        
        if df.duplicated().sum() > 0:
            recs.append("2. 存在重复行，建议执行 drop_duplicates()")
        
        obj_cols = [c for c in df.columns if str(df[c].dtype) == 'object']
        if len(obj_cols) > 0:
            recs.append(f"3. {len(obj_cols)} 列为 object 类型，建议转为 string[pyarrow] 或 category")
        
        if df.memory_usage(deep=True).sum() > 500 * 1024**2:
            recs.append("4. 内存占用超过 500MB，考虑分块处理或 dtype 优化")
        
        if len(df) > 1_000_000:
            recs.append("5. 数据量较大，建议使用 Parquet 格式存储以加速后续 I/O")
        
        if recs:
            content = "\n  📋 改进建议\n  ────────────────────────────────\n" + '\n'.join(recs)
        else:
            content = "\n  ✅ 数据质量良好，无需特殊处理！"
        
        self._add_section("改进建议", content)


np.random.seed(42)
n = 200_000
demo_data = {
    'id': range(n),
    'prompt': [f'问题{i%500}' for i in range(n)],
    'response': [f'回答{i}' for i in range(n)],
    'quality': np.random.choice([None]+list(range(1,6)), n, p=[0.03,0.15,0.15,0.25,0.27,0.15]),
    'tokens': np.random.randint(10, 2000, n),
    'source': np.random.choice(['api','web','export'], n),
    'model': np.random.choice(['GPT-4o','Claude','Llama'], n),
}

df_demo = pd.DataFrame(demo_data)

reporter = DataQualityReport(df_demo, "SFT 训练语料 v2.1")
report_text = reporter.generate()

with open('data_quality_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
print("\n报告已保存到 data_quality_report.txt")
```
