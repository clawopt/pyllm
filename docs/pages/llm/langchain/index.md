---
title: "LangChain教程 - LLM应用开发编排框架实战 | PyLLM"
description: "LangChain LLM应用开发完整教程：模型I/O、RAG检索增强、记忆机制、LCEL链式语法、Agent智能体、流式异步、评估调试、生产部署"
head:
  - - meta
    - name: keywords
      content: LangChain,LLM,RAG,Agent,LCEL,提示模板,向量检索,智能体
  - - meta
    - property: og:title
      content: LangChain教程 - LLM应用开发编排框架实战 | PyLLM
  - - meta
    - property: og:description
      content: LangChain LLM应用开发完整教程：模型I/O、RAG检索增强、记忆机制、LCEL链式语法、Agent智能体、流式异步、评估调试、生产部署
  - - meta
    - name: twitter:title
      content: LangChain教程 - LLM应用开发编排框架实战 | PyLLM
  - - meta
    - name: twitter:description
      content: LangChain LLM应用开发完整教程：模型I/O、RAG检索增强、记忆机制、LCEL链式语法、Agent智能体、流式异步、评估调试、生产部署
---

# LangChain 教程大纲


## 总体设计思路

LangChain 是目前最流行的 LLM 应用开发编排框架——它把"调用大模型 API"这件简单的事，扩展成了一套完整的工程体系：从提示词管理、输出解析、RAG 检索增强、记忆机制，到 Agent 智能体自主决策，再到生产级部署与安全防护。如果你只会调 OpenAI API，那你能做的只是一个聊天机器人；但如果你掌握了 LangChain，你就能构建真正有业务价值的 AI 应用。

本教程的设计遵循以下原则：

1. **从"为什么需要"到"怎么用"到"怎么用好"**：每章先讲清楚动机和原理，再给代码，最后讲生产级最佳实践
2. **从简单到复杂**：模型调用 → 提示管理 → RAG → 记忆 → Agent → 多模态 → 部署，逐步递进
3. **三个实战项目贯穿**：智能客服、代码分析、数据分析，覆盖最常见的 LLM 应用场景
4. **面向 v1.0 新架构**：基于 LCEL + LangGraph 的新范式，不再停留在旧版 Chain 模式

---

## 第01章：为什么需要 LangChain（4节）

### 定位
面向已经用过 ChatGPT/OpenAI API 但还没接触过编排框架的开发者。讲清楚"直接调 API"和"用框架"的本质区别。

### 01-01 大模型的能力边界与痛点
- LLM 的三大核心局限：知识截止、幻觉、缺乏逻辑推理
- 为什么需要外部工具和编排框架来弥补这些不足
- 真实案例分析：一个"纯 API 调用"的项目是如何失控的

### 01-02 编排框架与设计哲学
- 编排（Orchestration）的核心概念
- LangChain 的模块化架构：Model I/O / Retrieval / Memory / Agents / LCEL
- 设计理念与核心抽象：为什么是 Runnable 而不是 Chain

### 01-03 同类框架对比与选型
- LlamaIndex / Semantic Kernel / CrewAI / Haystack 的定位差异
- 各框架的适用场景分析
- 选型决策指南：什么时候用 LangChain，什么时候用别的

### 01-04 v1.0 新特性概览
- LCEL 表达式语言、Agent 架构升级、LangGraph 统一架构
- 中间件系统
- 从 v0.x 迁移的注意事项

---

## 第02章：环境搭建与第一次运行（4节）

### 定位
从零开始搭建开发环境，跑通第一个 LangChain 应用。确保所有读者站在同一起跑线上。

### 02-01 开发环境准备
- Python 版本要求、虚拟环境配置、IDE 推荐
- 第一个 Python 程序的运行

### 02-02 安装 LangChain 与生态工具
- pip 安装 LangChain、OpenAI SDK
- 环境变量配置、依赖版本验证

### 02-03 模型接入实战
- 接入 OpenAI GPT 系列模型
- API Key 安全管理
- 本地模型（Ollama）的接入方式

### 02-04 第一个 LangChain 应用
- Hello World 程序
- 流式输出实现
- 完整的项目结构模板

---

## 第03章：模型 I/O — 与 LLM 对话的基础（4节）

### 定位
LangChain 的第一块拼图：如何把"调 API"这件事做得更优雅、更可复用、更可组合。

### 03-01 模型类型：LLM vs. 聊天模型
- 两种模型接口的区别与适用场景
- 迁移指南

### 03-02 输入管理：提示模板的多种玩法
- PromptTemplate 基础、Few-shot 模板
- 聊天消息模板、部分变量

### 03-03 输出处理：解析器的妙用
- StrOutputParser、PydanticOutputParser
- CommaSeparatedListOutputParser 及自定义解析器

### 03-04 实战：构建可配置参数的情感分析器
- 综合运用模型、模板和解析器
- 从零搭建一个生产级的情感分析工具

---

## 第04章：检索增强生成 (RAG) — 连接私有知识库（6节）

### 定位
RAG 是目前 LLM 应用最成熟、最广泛使用的架构模式。这一章从原理到实战，完整覆盖 RAG 的每个环节。

### 04-01 RAG 为什么成为刚需？基本原理与架构
- RAG 解决的三大痛点
- 完整数据流 6 步拆解
- RAG vs 微调对比、架构全景图

### 04-02 文档加载器：从 PDF、网页、Notion 等 100+ 数据源读取
- TextLoader / PyPDFLoader / WebBaseLoader / DirectoryLoader / NotionLoader 等主流加载器详解

### 04-03 文本分割：语义切分策略
- RecursiveCharacterTextSplitter / MarkdownHeaderTextSplitter
- 按语义切分 / 组合策略

### 04-04 向量存储与嵌入模型
- Embeddings 嵌入模型原理
- Chroma / FAISS / Milvus Lite 对比
- 嵌入模型选择指南

### 04-05 检索器：相似度搜索与高级检索策略
- 基础相似度搜索、MMR 多样性搜索
- 上下文压缩检索、metadata 过滤
- Retriever 统一接口

### 04-06 实战：从零实现一个"公司内部文档问答机器人"
- 完整 RAG 问答系统：索引构建、检索问答、来源展示、CLI 交互

---

## 第05章：记忆 — 让对话拥有上下文（4节）

### 定位
LLM 本身是无状态的，但真实应用需要多轮对话。这一章解决"怎么让模型记住之前说了什么"。

### 05-01 为什么需要记忆：无状态模型的局限
- LLM 的无状态特性、对话上下文丢失问题

### 05-02 LangChain 记忆组件全景
- Memory 类层次结构
- BufferMemory / WindowMemory / SummaryMemory / TokenBufferMemory

### 05-03 常用记忆类型实战
- RunnableWithMessageHistory 集成
- 各种 Memory 的代码示例与对比

### 05-04 实战：构建有记忆的智能对话助手
- 综合运用 Memory 组件
- 搭建带持久化、多会话管理的生产级对话系统

---

## 第06章：多模态交互 — 支持图像与语音（4节）

### 定位
从纯文本走向多模态。让 LLM 不仅能"读"，还能"看"和"听"。

### 06-01 多模态模型入门：从文本到视觉
- 多模态 AI 的概念演进
- GPT-4o 的多模态能力
- LangChain 中的多模态支持

### 06-02 图像理解：让 LLM "看"图片
- 多模态消息格式、图片 URL 与 Base64
- 视觉问答、OCR、图表分析实战

### 06-03 语音交互：语音转文字与文字转语音
- OpenAI Whisper STT、TTS 文字转语音
- LangChain 集成与完整对话流程

### 06-04 实战：构建多模态智能助手
- 整合视觉理解 + 语音交互
- 搭建能看图、听语音、语音回复的完整应用

---

## 第07章：LCEL — 新的链式语法（5节）

### 定位
LCEL 是 LangChain v1.0 的核心创新——用声明式语法组合组件，替代旧版命令式 Chain。掌握 LCEL 是理解现代 LangChain 的关键。

### 07-01 为什么需要 LCEL？声明式 vs 命令式
- LCEL 解决的问题、Runnable 统一接口
- 声明式链式组合的优势

### 07-02 LCEL 的核心原语：Runnable 接口
- Runnable 的 invoke/batch/stream 方法
- RunnableLambda、RunnablePassthrough、RunnableConfig

### 07-03 管道操作符与并行化（RunnableParallel）
- | 管道的深入理解
- RunnableParallel 并行执行、结果合并与引用

### 07-04 条件分支与路由（RunnableBranch）
- RunnableBranch 条件路由
- 动态选择执行路径
- 路由在 RAG/Agent 中的应用

### 07-05 实战：用 LCEL 重构复杂问答链
- 综合运用管道/并行/路由
- 从零构建一个多模式智能问答系统

---

## 第08章：智能体 — 让 AI 自主规划与执行任务（5节）

### 定位
Agent 是 LangChain 的"终极形态"——让 LLM 从被动回答变成主动执行。这一章从概念到实战，完整覆盖 Agent 的构建方法。

### 08-01 Agent 是什么：从 Chain 到自主决策
- Agent 与 Chain 的本质区别
- ReAct 思维模式、Agent 的核心组件与类型

### 08-02 Tool（工具）：给 AI 装上手和脚
- @tool 装饰器、内置工具、自定义工具
- Tool 的设计原则与最佳实践

### 08-03 ReAct 模式：构建你的第一个 Agent
- create_react_agent 用法
- 思考-行动-观察循环详解
- 调试与优化技巧

### 08-04 高级代理模式：多代理协作与长期记忆
- 多代理团队协作、Agent 的反思机制
- 长期记忆管理、工具使用优化策略

### 08-05 实战：构建自动研究助理 Agent
- 整合搜索/计算/文件操作/RAG 检索
- 搭建能自主完成研究任务的智能体

---

## 第09章：流式、异步与中间件（5节）

### 定位
从"能跑"到"能跑好"的关键一章。流式输出提升用户体验，异步提升并发能力，中间件处理横切关注点。

### 09-01 流式输出的价值与实现
- stream() vs invoke() 的区别
- Token 级流式输出、流式在 Chain 中的传递
- SSE/WebSocket 实时推送

### 09-02 异步编程：提升应用的并发能力
- ainvoke/astream/abatch 异步方法
- async/await 模式、FastAPI 集成
- 并发性能对比

### 09-03 中间件：横切关注点（日志、限流、重试、审核）
- RunnablePassthrough.assign / 自定义中间件
- 日志中间件、重试机制、内容审核

### 09-04 回调机制：深入 LangChain 内部运行流程
- CallbackHandler / StdOutCallbackHandler / ConsoleCallbackHandler
- 自定义回调、start/end/llm_start/chain_end 事件

### 09-05 实战：为应用添加"对话审核"中间件
- 综合流式+异步+中间件+回调
- 构建带审核、日志、限流、追踪的完整对话系统

---

## 第10章：项目一：智能客服系统（4节）

### 定位
第一个完整实战项目。综合运用 RAG + 路由 + 人工接管，构建一个真实可用的客服系统。

### 10-01 需求分析与技术选型
- 智能客服系统的真实业务场景
- 功能需求拆解、技术架构选型与模块划分

### 10-02 利用 RAG 加载产品知识库
- 构建客服专用的 RAG 管线
- 知识库文档设计、检索质量优化策略
- 端到端问答链实现

### 10-03 设计多轮对话的意图识别与分流
- 客服意图分类体系
- LLM 驱动的意图识别器、RunnableBranch 路由
- 多轮对话状态机

### 10-04 集成人工接管（Handoff）机制
- Handoff 触发条件设计
- 会话上下文无缝传递
- 完整客服系统组装、CLI 与 FastAPI 部署

---

## 第11章：项目二：代码分析助手（3节）

### 定位
第二个实战项目。聚焦代码 RAG + Agent 工具调用，构建一个能理解代码、定位 Bug、生成测试的助手。

### 11-01 加载代码仓库并构建代码知识库
- 代码 RAG 的特殊性、仓库加载策略
- AST 感知分块、代码嵌入模型选择
- 索引构建与检索优化

### 11-02 实现代码解释、Bug 定位与单元测试生成
- 三大核心能力：代码自然语言解释、智能 Bug 检测与定位、自动生成单元测试
- Prompt 工程与输出解析

### 11-03 利用代理调用代码执行工具
- ReAct Agent 驱动的代码分析
- PythonREPL 工具集成
- 代码修复-验证闭环、完整 CodeAssistant 系统组装

---

## 第12章：项目三：数据分析 Agent（4节）

### 定位
第三个实战项目。Text-to-SQL + 数据可视化 + 洞察报告，构建一个能自主分析数据的 Agent。

### 12-01 连接数据库（SQL 数据库）
- LangChain SQL 工具链
- 数据库连接管理、Schema 感知查询
- 安全沙箱与只读模式

### 12-02 设计 Text-to-SQL 代理
- SQL 生成 Chain 原理、ReAct Agent 组装
- Few-Shot 示例注入、查询结果自然语言解释
- 错误恢复机制

### 12-03 让代理生成数据图表（Pandas + Matplotlib）
- SQL 结果转 DataFrame
- LLM 驱动的图表类型选择
- Matplotlib 绑定代码生成、图表保存与展示

### 12-04 输出洞察报告
- 多维度数据分析报告生成
- Agent 自主规划分析路径
- 完整 DataAnalysisAgent 系统组装

---

## 第13章：评估与可观测性（4节）

### 定位
"看着不错"不等于"真的不错"。这一章建立系统化的评估体系，让 LLM 应用的质量可量化、可追踪、可回归。

### 13-01 为什么要评估 RAG 和代理？
- LLM 应用的评估困境
- RAG 与 Agent 的特殊挑战
- 评估体系设计原则

### 13-02 评估指标：准确率、忠实度、答案相关性
- RAG 三角评估框架
- 各指标的精确定义与计算方法
- LLM-as-Judge 实现方案、RAGAS 集成使用

### 13-03 LangSmith 实战：追踪、调试与性能评估
- LangSmith 平台入门、Trace 数据采集
- 运行时可视化调试、Dataset 与评估器集成
- 性能基线监控

### 13-04 离线评估与持续回归测试
- 离线评估工作流、pytest 集成
- CI/CD 自动化、回归测试套件设计
- 性能基准测试

---

## 第14章：部署与扩展（4节）

### 定位
从开发脚本到生产服务。这一章覆盖 API 封装、容器化、向量数据库生产化、成本优化。

### 14-01 将 LangChain 应用封装为 API 服务（FastAPI）
- 从脚本到服务的架构转变
- FastAPI 项目结构设计
- 同步/异步接口、流式 SSE 端点、错误处理与中间件

### 14-02 容器化部署：Docker + Kubernetes
- Dockerfile 最佳实践、多阶段构建
- docker-compose 本地编排
- K8s 生产部署清单、Helm Chart、滚动更新与回滚

### 14-03 向量数据库的生产环境考量
- 从 Chroma 到生产级向量库的迁移路径
- Milvus 集群部署、Pinecone 云服务对比
- 索引策略与性能调优

### 14-04 成本优化：缓存策略与模型路由
- LLM API 成本分析
- 语义缓存实现、模型路由策略
- Token 优化技巧、成本监控仪表盘

---

## 第15章：安全与限制（3节）

### 定位
安全不是可选项，而是必选项。这一章覆盖 LLM 应用最常见的安全威胁和防护策略。

### 15-01 提示词注入攻击与防御
- Prompt Injection 的原理与分类
- 真实攻击案例、多层防御体系
- 输入清洗与输出过滤

### 15-02 敏感数据泄露风险
- LLM 应用的数据泄露面分析
- PII 检测与脱敏、日志安全
- 知识库数据隔离、API Key 管理最佳实践

### 15-03 代理的权限控制与沙箱
- Agent 工具权限矩阵、沙箱执行环境构建
- 基于角色的访问控制（RBAC）
- 审计日志与操作追溯、安全 Agent 设计模式
