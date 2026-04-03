---
title: 知识体系总结
description: Pandas 核心知识图谱、LLM 场景下的技能树、各章节关联关系
---
# 知识体系总览

到这里，我们已经走完了从基础到高级、从单机到分布式的完整学习路径。这一节做一个系统性的回顾。

## 核心知识图谱

```
第一层：数据容器 (Ch4)
├── Series：一维带标签数组
├── DataFrame：二维表格
└── Index：行/列标签系统

第二层：数据 I/O (Ch5)
├── 文本格式：CSV / JSONL / Parquet
├── 数据库：SQL 读写
└── 高级格式：Excel / HTML

第三层：数据质量 (Ch6-7)
├── 探索：info / describe / value_counts
├── 缺失值：检测 → 判断机制 → 处理
├── 重复值：检测 → 智能去重
└── 类型修正：astype / to_numeric / category

第四层：数据处理 (Ch8-11)
├── 筛选：布尔索引 / query / isin
├── 变换：apply / map / 向量化
├── 排序：sort_values / rank / nlargest
└── 聚合：groupby / agg / pivot_table

第五层：数据整合 (Ch12-15)
├── 合并：merge / concat / join
├── 时间序列：resample / rolling / shift
└── 可视化：plot / style

第六层：进阶与 AI 集成 (Ch16-22)
├── 性能优化：内存 / 计算 / 并行
├── LLM 数据处理：清洗 / 评估 / RAG
├── Agent：PandasAI / LangChain / MCP Server
└── 多模态：图像 / 音频 / 统一管理

第七层：扩展 (Ch23-25)
├── 分布式：Dask / Polars / Ray
└── 进阶路线图
```
