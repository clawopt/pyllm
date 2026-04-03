---
title: 项目总结与进阶路线图
description: Pandas 学习路径总结、常见面试题、从入门到精通的推荐资源、LLM 工程师的 Pandas 技能树
---
# 总结：Pandas 在 LLM 工程中的知识体系

到这里，我们已经覆盖了 Pandas 的核心知识和 LLM 场景下的应用。这一节做一个系统性的回顾，并给出继续深入的方向。

## 知识体系总览

```
基础层 (Ch4-5)
├── Series / DataFrame 核心数据结构
├── 数据类型系统（dtype / Nullable / Category）
├── 数据探索（info / describe / value_counts）
└── 数据读写（CSV / JSONL / Parquet / SQL）

操作层 (Ch6-11)
├── 数据质量：缺失值、重复值、类型修正
├── 筛选：布尔索引 / query / isin / contains
├── 变换：apply / map / 向量化操作
├── 排序：sort_values / rank / nlargest
└── 聚合：groupby / agg / pivot_table

整合层 (Ch12-15)
├── 合并：merge / concat / join
├── 时间序列：resample / rolling / shift
└── 可视化：plot / style

高级层 (Ch16-20)
├── 性能优化：内存 / 计算 / 并行
├── LLM 专用：数据清洗 / 模型评估 / RAG 管理
├── AI 集成：PandasAI / LangChain Agent / MCP Server
└── 扩展方向：多模态 / 分布式处理
```

## 面试高频问题

1. **`loc` vs `iloc` 的区别？** → loc 用标签，iloc 用整数位置；loc 切片包含两端点，iloc 不包含
2. **`apply()` 和 `map()` 的区别？** → apply 对每个元素/行/列执行任意函数；map 只做值到值的映射（通常用字典）
3. **`merge()` 和 `concat()` 的区别？** → merge 是按键值匹配的智能合并（类似 SQL JOIN）；concat 是简单的堆叠拼接
4. **如何处理大数据集？** → usecols + dtype 优化内存 → chunksize 分块 → Parquet 列式存储 → 必要时升级 Dask/Polars

## 进阶推荐

- **Polars**：Rust 写的新一代 DataFrame 库，API 更现代，性能更快
- **DuckDB**：内嵌式 OLAP 数据库，支持直接查询 Parquet/CSV 文件
- **Modin**：Pandas 的 drop-in 替换，自动利用多核并行
- **PySpark**：超大规模数据的分布式处理标准方案
