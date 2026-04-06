---
title: LangGraph 学习指南
description: LangGraph 有状态 Agent 编程教程，从状态机到复杂工作流编排
---
# LangGraph 学习指南

> **一句话概括**：LangGraph 是 LangChain 官方推出的**有状态 Agent（Stateful Agent）编程框架**。如果说 LangChain 的 `create_react_agent()` 能让 AI "思考→行动→观察"地走几步，那 LangGraph 就能让 AI "记住状态、规划路线、处理分支、循环迭代、容错恢复"——像真正的程序员一样构建复杂的工作流。

## 这份教程适合谁？

- **学完 LangChain 第8章 Agent 后觉得 ReAct 模式不够用的开发者**
- 需要构建**多步骤、有状态、可持久化、支持人机协作**的复杂 Agent 系统
- 想理解如何将 Agent 从"单次对话"升级为"长期运行的任务执行器"
- 对工作流引擎（如 Airflow/DAG）有经验，想了解 LLM 时代的替代方案

## LangGraph vs LangChain Agent：核心区别

| 维度 | LangChain Agent (ReAct) | LangGraph |
|------|------------------------|----------|
| **状态管理** | 无状态 / 手动维护 | **内置 State Schema**（类型化、可序列化） |
| **控制流** | 线性：Thought → Action → Observation 循环 | **图结构**：条件分支、并行、循环、跳转 |
| **持久化** | 不支持（进程结束即丢失） | **Checkpointing**（断点续跑、时间旅行调试） |
| **人机协作** | 只能在开始时输入 | **Interrupt**（任意节点暂停等人类确认） |
| **错误恢复** | 全部重试或放弃 | **条件边**（某步失败走备用路径） |
| **适用场景** | 单轮问答 / 简单工具调用 | 复杂任务 / 多轮审批 / 工作流自动化 |

### 一个直观对比

```
LangChain ReAct Agent:
  用户: "帮我调研 Python 和 Go 的并发模型差异并写报告"
  
  Step 1: Thought → Action: search("Python concurrency") → Observation
  Step 2: Thought → Action: search("Go concurrency") → Observation
  Step 3: Thought → Final Answer (一次性输出)
  ❌ 问题: 太长可能超时；中间出错全部重来；无法分步审核

LangGraph Agent:
  用户: "帮我调研 Python 和 Go 的并发模型差异并写报告"
  
  State = { topic: "...", python_info: null, go_info: null, report: null, status: "init" }
  
  [start] → [research_python] → {python_info: "..."} 
         → [research_go]     → {go_info: "..."}
         → [compare]        → {comparison: "..."}
         → [draft_report]   → {report_draft: "..."}
         → [human_review]    → INTERRUPT (等待用户确认)
         → [finalize]       → {report: "完整报告...", status: "done"}
  
  ✅ 优势: 可中断/续跑；每步都有状态快照；用户可在中间介入
```

## 教程结构

本教程将涵盖以下内容：

### 基础篇
- **第1章：LangGraph 核心概念** — Graph/Node/Edge/State/Checkpoint 的完整模型
- **第2章：第一个 Stateful Agent** — 用 State 定义类型化状态、用 Graph 编排节点
- **第3章：条件路由与分支** — 条件边（Conditional Edge）、动态决策
- **第4章：人机协作模式** — Interrupt 节点、Human-in-the-loop、审批流

### 进阶篇
- **第5章：循环与迭代** — 自省循环（Self-Reflection）、最大尝试次数
- **第6章：子图与模块化** — 将复杂流程拆分为可复用的子图
- **第7章：持久化与时间旅行** — CheckpointerSaver、断点续跑、历史回放
- **第8章：多 Agent 协作** — Supervisor 模式、Map-Reduce 并行、手-Off

### 实战篇
- **第9章：项目一：智能客服工单系统** — 自动分类 → 多级审批 → 执行 → 回访
- **第10章：项目二：自主研究助手** — 规划 → 分解任务 → 并行调研 → 综合报告

## 学习建议

1. **必须先掌握 LangChain 基础和 Agent 章节**（本站第1-8章），否则会缺少前置知识
2. **重点理解"图思维"**——LangGraph 不是写代码的线性逻辑，而是定义一张状态转换图。这对很多开发者来说是思维方式的转变
3. **从简单图开始**——不要一开始就画复杂的 DAG，先实现 2-3 个节点的最小图，逐步扩展
4. **善用 LangSmith**——LangGraph 与 LangSmith 的集成非常深，可视化图谱是调试利器

## 技术栈要求

- **Python 3.10+**
- **LangChain 已安装**（LangGraph 是 langchain 包的一部分）
- **熟悉 LangChain Agent 基础**（create_react_agent / Tool / ReAct）
- 了解基本的有状态编程概念（FSM/状态机）会有帮助但不是必须的

## 开始学习

> 📌 教程正在编写中，敬请期待！建议先完成 [LangChain 教程第8章](/pages/llm/langchain/08-01-agent-concepts) 打好 Agent 基础。
