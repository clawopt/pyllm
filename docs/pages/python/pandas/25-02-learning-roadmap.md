---
title: 学习路线图与资源推荐
description: 从入门到精通的进阶路径、官方文档/书籍/课程推荐、面试准备指南
---
# 学习路线图

## 初级（1-2 周）

目标：能独立完成日常数据处理任务

- [ ] 掌握 Series / DataFrame 的创建和基本操作
- [ ] 熟练使用 `read_csv()` / `to_csv()` / `read_parquet()`
- [ ] 能用布尔索引和 `query()` 做数据筛选
- [ ] 掌握 `groupby()` + `agg()` 的基本聚合
- [ ] 了解 `info()` / `describe()` / `value_counts()` 数据探索三件套

**推荐练习**：找一份 Kaggle 数据集（如 Titanic），完成数据加载 → 清洗 → 分析 → 可视化的完整流程。

## 中级（2-4 周）

目标：能处理生产级的复杂数据任务

- [ ] 深入理解 `merge()` / `concat()` 的各种连接方式
- [ ] 掌握时间序列的 `resample()` / `rolling()` / `shift()`
- [ ] 理解 dtype 系统，能做内存优化
- [ ] 能写链式调用的数据处理流水线
- [ ] 熟练处理缺失值和重复值的各种策略

**推荐项目**：构建一个 SFT 数据清洗流水线（从原始 JSONL 到训练就绪的 Parquet）。

## 高级（1-2 月）

目标：能解决性能瓶颈并集成 AI 工具

- [ ] 理解向量化 vs apply 的性能差异
- [ ] 能用 Dask / Polars 处理超大数据集
- [ ] 能构建 LangChain Pandas Agent
- [ ] 能开发 MCP Server 对接 Claude Desktop
- [ ] 理解多模态数据的元数据管理模式

## 推荐资源

| 资源 | 类型 | 适用阶段 |
|------|------|---------|
| Pandas 官方文档 | 文档 | 全阶段 |
| "Python for Data Analysis" (Wes McKinney) | 书籍 | 初中级 |
| Kaggle Learn Pandas course | 在线课程 | 初级 |
| Polars 官方文档 | 文档 | 高级 |
| LangChain 文档 - Pandas Agent | 文档 | 高级 |
