# LlamaIndex 教程大纲

---

## 总体设计思路

LlamaIndex 是专注于 RAG（检索增强生成）的数据框架——如果说 LangChain 是"全能编排框架"，那 LlamaIndex 就是"RAG 深度优化专家"。它提供了从数据接入、文档解析、索引构建、高级检索到查询引擎的完整工具链，在 RAG 场景下的深度和精细度远超 LangChain。

本教程的设计遵循以下原则：

1. **从"为什么需要"到"怎么用深"**：先讲清楚 LlamaIndex 与 LangChain 在 RAG 场景下的定位差异，再深入每个组件
2. **从数据接入到多模态 RAG**：数据连接器 → 文档解析 → 索引策略 → 高级检索 → 查询引擎 → 响应合成 → 评估调试
3. **两个完整实战项目**：企业知识库问答系统 + 多模态 RAG 应用
4. **与 LangChain 的对比贯穿始终**：同一问题两种解法，帮助读者做出正确选型

---

## 第01章：LlamaIndex 快速入门（5节）

### 定位
面向已经了解 RAG 概念但还没用过 LlamaIndex 的开发者。5 行代码跑通第一个 RAG 应用，建立整体认知。

### 01-01 为什么需要 LlamaIndex
- RAG 系统的真实痛点：数据接入复杂、检索质量差、响应不相关
- LlamaIndex 与 LangChain 的定位差异
- LlamaIndex 核心架构概览

### 01-02 安装与环境配置
- LlamaIndex 安装方式、环境变量配置
- 多 LLM 后端切换
- 常见安装问题排查

### 01-03 第一个 RAG 应用：5 行代码跑通问答系统
- 用最少代码实现完整的 RAG 流程
- 理解数据加载→索引构建→查询的全链路

### 01-04 与 LangChain RAG 的对比：同一问题的两种解法
- 同一个 RAG 任务用 LlamaIndex 和 LangChain 分别实现
- 深度对比两种框架的设计哲学差异

### 01-05 核心概念预览
- Document / Node / Index / Query Engine / Response Synthesizer
- 五大核心概念的通俗解释与相互关系

---

## 第02章：数据连接器 Connectors（7节）

### 定位
RAG 的第一步是把数据"搬进来"。LlamaIndex 提供了 160+ 数据连接器，这一章覆盖最常用的几类。

### 02-01 统一数据接入：LlamaIndex 的 Connector 体系
- Connector 架构设计、Reader/Loader 抽象
- 数据接入模式、与 LangChain Loader 的对比

### 02-02 文件连接器：PDF / Word / Markdown / HTML / Excel / 图片
- 各种文件格式的深度解析
- 多引擎选择、表格数据提取、图片 OCR 集成

### 02-03 数据库连接器：PostgreSQL / MySQL / MongoDB / SQLite / Redis
- 关系型与非关系型数据库的数据接入
- SQL 查询到 Document 的映射、增量同步策略

### 02-04 API 连接器：OpenAI / Notion / Slack / GitHub / Jira / 网页爬取
- 各类 SaaS 平台 API 的数据接入
- 认证机制、分页处理、速率限制应对

### 02-05 云服务连接器：AWS S3 / Google Drive / OneDrive / Notion DB
- 主流云存储平台的数据接入
- 认证配置、大文件处理、权限管理

### 02-06 自定义连接器：如何为私有数据源编写 Reader
- BaseReader 接口实现
- 自定义数据加载逻辑
- LlamaHub 发布流程

### 02-07 多源数据融合：同时从数据库 + 文件 + API 加载数据
- 异构数据源的统一加载策略
- 数据去重与冲突解决、元数据标准化

---

## 第03章：文档加载与解析（5节）

### 定位
数据搬进来了，但还不能直接用——需要把原始文档切分成可检索的 Node。这一章深入解析策略与质量评估。

### 03-01 文档解析的深度挑战
- 为什么简单分块不够用
- 语义完整性 vs 检索粒度的矛盾
- 文档结构丢失的代价

### 03-02 Node Parser 详解：SentenceSplitter 与其变体
- SentenceSplitter 内部机制、参数调优指南
- CodeSplitter / MetadataAwareSeparator
- 自定义 Parser 实现

### 03-03 层级化文档解析：标题感知与结构保留
- HTMLHierarchicalSplitter / MarkdownNodeParser
- 文档树构建、层级索引与检索

### 03-04 自定义解析策略：针对不同文档类型的优化
- 按文档类型选择解析器
- 复合解析管道
- JSON/日志/邮件等特殊格式的处理

### 03-05 解析质量评估与调优
- 解析质量的量化指标
- 基于检索效果的端到端评估方法
- A/B 测试框架、持续优化流程

---

## 第04章：索引策略进阶（5节）

### 定位
索引是 RAG 的核心——不同的索引策略决定了检索的质量和效率。这一章深入 VectorStoreIndex 的内部机制，并对比多种索引类型。

### 04-01 VectorStoreIndex 深入：向量索引的内部机制与优化
- 向量索引的工作原理
- 嵌入模型选择、向量存储内部结构
- 性能优化技巧

### 04-02 多种索引类型详解
- ListIndex / TreeIndex / KeywordTableIndex / SummaryIndex / GraphIndex
- 五种非向量索引的原理、适用场景、代码示例

### 04-03 索引组合策略：如何组合多种索引应对复杂查询
- Router Query Engine、多索引协作模式
- 查询路由策略、实际架构案例

### 04-04 向量存储后端选择：Chroma/Qdrant/Pinecone/pgvector 对比
- 主流向量数据库的特性对比
- 选型指南、性能基准测试
- 生产环境部署考量

### 04-05 索引持久化与增量更新
- StorageContext 机制、磁盘序列化
- 增量插入与删除、版本管理与回滚策略

---

## 第05章：高级检索技术（5节）

### 定位
检索质量直接决定 RAG 的效果。这一章从混合检索到重排序，全面提升检索的准确性和多样性。

### 05-01 混合检索：向量搜索 + 关键词搜索的协同
- Hybrid Search 原理
- BM25 与向量的互补性
- LlamaIndex 实现方式、权重调优

### 05-02 重排序（Reranking）：从粗排到精排
- Reranker 原理、Cross-Encoder vs Bi-Encoder
- Cohere Rerank / bge-reranker 集成
- 多阶段检索管道

### 05-03 HyDE：假设性文档嵌入
- HyDE 原理与实现
- 查询扩展技术、Multi-Query 检索
- Decompose Transform

### 05-04 检索后处理：过滤、去重与增强
- SimilarityPostprocessor / DeduplicateNodePostprocessor
- MetadataReplacementPostprocessor 使用详解

### 05-05 高级检索模式总结与最佳实践
- 完整检索管道架构设计
- 不同场景的推荐方案
- 常见反模式

---

## 第06章：查询引擎 Query Engine（5节）

### 定位
查询引擎是 LlamaIndex 的"大脑"——它把检索器和合成器组合在一起，提供从简单查询到复杂对话的完整能力。

### 06-01 Query Engine 架构与工作原理
- Query Engine 的内部组成
- Retriever + Synthesizer 协作模式
- 请求生命周期

### 06-02 RetrieverQueryEngine：检索型查询引擎详解
- 手动组装 Query Engine
- 自定义 Retriever/Synthesizer
- 高级配置选项与调试技巧

### 06-03 ChatEngine：多轮对话引擎
- ChatMode 与 QueryMode 的区别
- 记忆管理、上下文窗口优化
- 对话式 RAG 的最佳实践

### 06-04 自定义 Query Engine 与高级配置
- SubQuestionQueryEngine / RouterQueryEngine / MultiStepQueryEngine
- Tool-based Query Engine

### 06-05 流式输出与异步查询
- StreamingResponseGen / SSE 集成
- 异步 Query Engine
- 批量查询性能优化

---

## 第07章：响应合成 Response Synthesis（5节）

### 定位
检索到的 Node 怎么变成自然语言答案？响应合成是 RAG 管道的最后一环，也是最容易出幻觉的环节。

### 07-01 响应合成的核心挑战与设计理念
- 从检索节点到自然语言答案的转换难题
- Synthesizer 在 RAG 管道中的位置
- 为什么不能简单拼接文本

### 07-02 四种合成模式深度解析
- REFINE / COMPACT_ACCUMULATE / TREE_SUMMARIZE / SIMPLE_SUMMARIZE
- 内部机制、Prompt 结构、Token 消耗分析
- 选择指南

### 07-03 自定义 Synthesizer 与 Prompt 工程
- 完全控制合成流程
- Prompt 模板定制、多阶段合成管道
- 领域适配与风格控制

### 07-04 结构化输出：JSON / 表格 / 代码
- 强制 LLM 输出结构化格式
- Pydantic 输出解析器
- 表格自动生成、代码块格式化

### 07-05 合成质量优化与常见问题
- 合成质量的评估指标
- 幻觉检测与抑制
- 长答案的质量控制、Token 效率优化

---

## 第08章：评估与调试（5节）

### 定位
"检索到了吗？回答对了吗？"——没有评估，RAG 系统就是黑盒。这一章建立完整的评估体系。

### 08-01 RAG 评估体系概述
- 为什么 RAG 必须要评估
- 评估的三大阶段与五大维度
- 评估驱动开发的闭环流程

### 08-02 检索质量评估：Recall / Precision / MRR / Hits@K
- 检索评估指标详解
- 评估数据集构建方法
- LlamaIndex 内置评估工具

### 08-03 答案质量评估：忠实度 / 准确性 / 有用性
- RAGAS 框架集成
- LLM-as-Judge 模式
- Faithfulness/Accuracy/Relevance 指标详解

### 08-04 调试工具与可观测性
- Callbacks 事件系统
- LlamaIndex Trace 工具
- 日志最佳实践、性能瓶颈定位

### 08-05 完整评估工作流与生产实践
- 从实验室到生产线的评估体系
- CI 集成与自动化回归测试

---

## 第09章：项目一：企业知识库问答系统（5节）

### 定位
第一个完整实战项目。综合运用数据连接器 + 文档解析 + 索引策略 + 高级检索 + 查询引擎，构建一个企业级知识库问答系统。

### 09-01 项目概述与需求分析
- 从"Hello World"到生产系统的需求拆解

### 09-02 系统架构设计
- 架构设计的原则与权衡

### 09-03 核心模块实现
- 从设计到代码：核心模块的落地

### 09-04 API 服务与前后端集成
- 把 RAG 引擎变成可调用的服务

### 09-05 部署运维与性能优化
- 从"能跑"到"跑好"：生产化的最后一公里

---

## 第10章：项目二：多模态 RAG 应用（5节）

### 定位
第二个完整实战项目。从纯文本 RAG 升级到多模态 RAG——支持图片、表格、混合内容的检索与问答。

### 10-01 多模态 RAG 概述与场景分析
- 从纯文本到多模态：RAG 的下一个前沿

### 10-02 多模态数据接入与解析
- 当文档不只是文字：多模态数据接入的挑战

### 10-03 多模态检索与融合
- 让文本和图片在同一个空间里"对话"

### 10-04 多模态问答引擎实现
- 把所有组件组装成完整的问答系统

### 10-05 前端展示与部署
- 让多模态能力触手可及：前端体验设计
