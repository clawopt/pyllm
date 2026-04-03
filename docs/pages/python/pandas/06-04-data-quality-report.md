---
title: 数据质量报告生成模板
description: 一键生成完整的数据质量报告，包含形状、类型、缺失、重复、分布等全方位分析，可直接用于数据验收
---
# 自动化质量报告：把检查流程变成标准动作

前三节我们分别介绍了数据概览、缺失值检测和重复值检测。在实际工作中，你不会每次都手动敲这些代码——你需要一个**一键运行的标准化报告**，每次拿到新数据后跑一遍，几分钟内全面了解数据状况。

这一节我们把这些分散的检查逻辑整合成一个完整的报告生成器。

## 报告应该包含什么

一个好的数据质量报告至少要回答以下问题：

1. **数据的基本面貌**——多少行、多少列、占多少内存、列名是否合理
2. **每列的类型是否正确**——有没有被错误推断为 object 的数值列
3. **缺失值情况**——哪些列有空、空了多少、占比多大
4. **重复值情况**——有多少完全重复、按关键列去重能去掉多少
5. **数值列分布**——均值、方差、分位数、是否有异常值
6. **分类列分布**——每个类别各有多少、是否有极端不平衡
7. **总体健康评分**——一眼看出数据能否直接使用

## 完整报告生成器

下面是一个生产级的报告类，它把前面三节的所有检查整合在了一起：

```python
import pandas as pd
import numpy as np

class DataQualityReport:
    def __init__(self, df, name="Dataset"):
        self.df = df.copy()
        self.name = name
    
    def generate(self):
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  {self.name} — 数据质量报告")
        lines.append(f"{'='*60}")
        
        self._basic_info(lines)
        self._dtype_analysis(lines)
        self._missing_analysis(lines)
        self._duplicate_analysis(lines)
        self._numeric_stats(lines)
        self._categorical_dist(lines)
        self._summary(lines)
        
        report = '\n'.join(lines)
        print(report)
        return report
    
    def _basic_info(self, L):
        df = self.df
        mem = df.memory_usage(deep=True).sum() / 1024**2
        L.append(f"\n📐 形状: {df.shape[0]:,} 行 × {df.shape[1]} 列 | 内存: {mem:.1f} MB")
        L.append(f"   列名: {', '.join(df.columns.tolist())}")
    
    def _dtype_analysis(self, L):
        df = self.df
        obj_cols = [c for c in df.columns if str(df[c].dtype) == 'object']
        if obj_cols:
            L.append(f"\n⚠️ object 类型列 ({len(obj_cols)} 个): {', '.join(obj_cols[:5])}")
        else:
            L.append("\n✅ 无 object 类型列")
    
    def _missing_analysis(self, L):
        df = self.df
        na = df.isna().sum()
        has_na = na[na > 0]
        
        if len(has_na) > 0:
            L.append(f"\n❌ 缺失值 ({len(has_na)} 列有缺失):")
            for col in has_na.index:
                cnt = has_na[col]
                pct = cnt / len(df) * 100
                bar = '█' * int(pct / 2)
                L.append(f"  {col:<22s} {cnt:>8,} ({pct:>5.1f}%) {bar}")
        else:
            L.append("\n✅ 无缺失值")
    
    def _duplicate_analysis(self, L):
        df = self.df
        full_dup = df.duplicated().sum()
        L.append(f"\n🔁 完全重复行: {full_dup:,} ({full_dup/len(df)*100:.2f}%)")
        
        if len(df) > 1:
            partial = {}
            for col in df.columns[:5]:
                n = df.duplicated(subset=[col]).sum()
                if n > 0:
                    partial[col] = n
            if partial:
                top = sorted(partial.items(), key=lambda x: -x[1])[:3]
                L.append("  按单列去重可去除:")
                for col, n in top:
                    L.append(f"    [{col}]: {n:,} 条")
    
    def _numeric_stats(self, L):
        df = self.df
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) == 0:
            L.append("\n📊 无数值列")
            return
        
        desc = df[num_cols].describe().T[['count','mean','std','min','50%','max']]
        desc.columns = ['数量','均值','标准差','最小','中位数','最大']
        L.append(f"\n📊 数值统计 ({len(num_cols)} 列):")
        L.append(desc.round(2).to_string())
    
    def _categorical_dist(self, L):
        df = self.df
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols[:3]:
            dist = df[col].value_counts().head(6)
            L.append(f"\n📋 [{col}] 分布 ({df[col].nunique()} 种):")
            for val, cnt in dist.items():
                bar = '#' * min(int(cnt/len(df)*60), 50)
                L.append(f"  {bar} {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
    
    def _summary(self, L):
        df = self.df
        na_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        dup_pct = df.duplicated().sum() / len(df) * 100
        
        issues = []
        if na_pct > 1:
            issues.append(f"缺失值偏高 ({na_pct:.1f}%)")
        if dup_pct > 1:
            issues.append(f"重复率偏高 ({dup_pct:.1f}%)")
        
        obj_count = sum(1 for c in df.columns if str(df[c].dtype) == 'object')
        if obj_count > 0:
            issues.append(f"{obj_count} 列为 object 类型（建议优化）")
        
        if not issues:
            L.append(f"\n✅ 总体评价: 数据质量良好，可用于下一步处理")
        else:
            L.append(f"\n⚠️ 总体评价: 发现以下问题 → " + " | ".join(issues))


report = DataQualityReport(pd.DataFrame({
    'prompt': ['什么是AI' + str(i) for i in range(1000)],
    'response': [f'回答{i}' if i % 50 != 0 else None for i in range(1000)],
    'quality': np.random.choice([1,2,3,4,5], 1000),
    'tokens': np.random.randint(20, 2000, 1000),
    'source': np.random.choice(['api','web'], 1000),
}), "测试语料")

report.generate()
```

运行后会输出一份结构化报告，涵盖形状、内存、dtype 警告、缺失值详情（含可视化柱状图）、重复值分析、数值统计摘要、分类列分布，以及最终的总体健康评价。**把它作为项目的基础设施，每次加载新数据后跑一遍**，养成习惯后能避免绝大多数因数据质量问题导致的返工。

到这里，第六章就全部结束了。我们从"拿到数据先看什么"开始，逐步深入到缺失值的检测与分析机制判断，再到重复值的识别与智能去重，最后用一个自动化报告工具把所有检查串联起来。这三节构成了数据质量评估的完整闭环。接下来第七章，我们要从"发现问题"进入"解决问题"的阶段——真正动手清洗和修复数据。
