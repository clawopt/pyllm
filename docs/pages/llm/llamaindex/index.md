---
title: LlamaIndex 学习指南
description: LlamaIndex 数据框架教程，从数据连接到 RAG 进阶应用
---
# LlamaIndex 学习指南

> **一句话概括**：LlamaIndex 是一个专注于**数据连接与 RAG（检索增强生成）** 的 LLM 应用框架。如果说 LangChain 是"万能工具箱"，那 LlamaIndex 就是"数据工程师的瑞士军刀"——它在文档加载、索引构建、检索优化方面的能力远超 LangChain。

## 这份教程适合谁？

- **正在用 LangChain 做 RAG 但觉得检索质量不够好的开发者**
- 需要连接复杂数据源（数据库、API、非结构化文件）的工程师
- 想深入理解 RAG 内部机制并优化检索效果的技术人员
- 对比过 LlamaIndex 和 LangChain 后想系统学习 LlamaIndex 的开发者

## LlamaIndex vs LangChain：核心区别

这是最常被问到的问题。两者不是竞争关系，而是**互补关系**：

| 维度 | LangChain | LlamaIndex |
|------|----------|-----------|
| **核心定位** | 通用 LLM 编排框架 | 数据 + RAG 专用框架 |
| **最强项** | Agent / 工具链 / 多步编排 | 文档加载 / 索引 / 高级检索 |
| **RAG 能力** | 基础级（VectorStoreRetriever） | 企业级（Node 解析、自动路由、混合检索） |
| **数据连接** | 几十种 Loader | **数百种 Connector**（SQL/NoSQL/API/云服务） |
| **索引策略** | 手动分块 → 向量化 | **智能索引**（层级索引、摘要索引、知识图谱） |
| **适用场景** | 全栈 LLM 应用开发 | **以数据为中心的 RAG 系统** |
| **学习曲线** | 入门快，精通难 | 入门稍陡，RAG 方面更深 |

**简单来说**：如果你在构建一个需要深度处理数据的 RAG 系统——特别是涉及大量文档、复杂查询、需要高检索精度的场景——LlamaIndex 是更好的选择。

## 教程结构

本教程将涵盖以下内容：

### 基础篇
- **第1章：LlamaIndex 快速入门** — 安装、第一个 RAG 应用、与 LangChain 的对比
- **第2章：数据连接器 (Connectors)** — SQL/NoSQL/API/文件系统/云服务的统一接入
- **第3章：文档加载与解析** — PDF/Markdown/HTML/代码文件的智能解析
- **第4章：索引策略进阶** — Vector Store Index / Keyword Table Index / Summary Index / Knowledge Graph Index

### 进阶篇
- **第5章：高级检索技术** — Hybrid Search / Re-ranking / Metadata Filtering / Parent-Child Chunking
- **第6章：查询引擎 (Query Engine)** — 自定义查询转换 / Sub-Question Router / Multi-Step Query Engine
- **第7章：响应合成 (Response Synthesis)** — Refine / Compact-and-Refine / Tree Summarize / Multi-Response Modes
- **第8章：评估与调试** — RAGAS 评估 / Faithfulness / Response Relevance / Tracing & Debugging

### 实战篇
- **第9章：项目一：企业知识库问答系统** — 大规模文档的 RAG 最佳实践
- **第10章：项目二：多模态 RAG 应用** — 图像+文本联合检索与分析

## 学习建议

1. **如果你已经学完本站的 LangChain 教程**：直接从第2章开始，重点关注 LlamaIndex 在数据连接和索引方面与 LangChain 的差异
2. **如果你是 LLM 新手但主要需求是 RAG**：建议先学 LangChain 第1-4 章（建立基础），再转来学 LlamaIndex 第3-5 章（深化 RAG）
3. **如果你在做企业级知识库项目**：重点看第5章（高级检索）和第9章（实战项目）

## 技术栈要求

- **Python 3.10+**
- **OpenAI API Key** 或其他兼容 API
- 了解基本的向量搜索概念（如果学过 LangChain 第4章就足够了）
- 有 RAG 实际经验会更有体感

## 开始学习

> 📌 教程正在编写中，敬请期待！当前可以先访问 [LangChain 教程](/pages/llm/langchain/01-01-llm-limitations) 打好 RAG 基础。
