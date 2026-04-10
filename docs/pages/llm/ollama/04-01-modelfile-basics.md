# 04-1 Modelfile 基础语法

## 什么是 Modelfile

如果你用过 Docker，那么理解 Modelfile 会非常自然——它就是 Ollama 版的 Dockerfile。Dockerfile 用声明式指令描述如何构建一个容器镜像（FROM、RUN、COPY、ENV），而 Modelfile 用声明式指令描述如何组装一个自定义模型（FROM、SYSTEM、PARAMETER、TEMPLATE）。这个类比不是巧合——Ollama 的设计哲学深受 Docker 启发：**一切皆模型，模型皆可编程定义**。

但 Modelfile 比 Dockerfile 更简洁。一个可用的 Modelfile 最短只需要两行：

```dockerfile
FROM llama3.1
SYSTEM """你是一个有帮助的助手"""
```

这两行做了什么？第一行 `FROM` 声明了基础模型——就像 Docker 镜像继承自基础镜像一样，你的自定义模型继承自 `llama3.1` 的全部能力（权重、词汇表、分词器配置等）。第二行 `SYSTEM` 注入了一个系统提示词——每次对话开始时，这段文字会自动作为系统消息发送给模型，告诉它应该扮演什么角色。就这么简单，你已经创建了自己的第一个定制模型。

## 核心指令完整参考

Modelfile 支持以下指令，按使用频率排序：

### FROM — 基础模型（必选）

```dockerfile
# 使用 Ollama 官方库中的模型
FROM llama3.1:8b
FROM qwen2.5:7b
FROM mistral:7b

# 使用本地已有的模型
FROM ./my-custom-model.gguf

# 使用第三方库的模型
FROM bartowski/llama3.1-8b-instruct-q5_K_M
```

`FROM` 必须是 Modelfile 的第一条指令，且只能出现一次。它指定了你的自定义模型所基于的"底座"。

### SYSTEM — 系统提示词

```dockerfile
# 单行形式
SYSTEM 你是一个Python专家。

# 多行形式（推荐用于复杂提示）
SYSTEM """你是一个资深的技术文档翻译器。
你的任务是将英文技术文档翻译成通顺、专业的中文。

规则：
1. 保持技术术语的准确性，不要过度意译
2. 专业术语首次出现时保留英文原文在括号中
3. 代码示例保持原样不做翻译
4. 输出格式为 Markdown
5. 不要添加任何解释性前言或后语，直接输出翻译结果
"""

# 从文件读取（适用于超长提示词）
SYSTEM """$(cat system_prompt.txt)"""
```

SYSTEM 的内容会在每次对话开始时注入到消息列表的最前面。用户在正常对话中看不到这个提示词，但它深刻地影响着模型的输出风格和行为模式。

### PARAMETER — 推理参数

```dockerfile
# 温度：控制随机性 (0=确定性, 0.7=平衡, 1.5=创造性)
PARAMETER temperature 0.7

# Top-K：采样时只考虑概率最高的 K 个 token
PARAMETER top_k 40

# Top-P：核采样阈值 (0.1=保守, 1.0=不限制)
PARAMETER top_p 0.9

# 上下文窗口大小 (token 数)
PARAMETER num_ctx 8192

# 最大生成长度 (token 数)
PARAMETER num_predict 2048

# 重复惩罚系数 (1.0=不惩罚, 1.18=默认值)
PARAMETER repeat_penalty 1.15

# 停止序列：遇到这些字符串时停止生成
PARAMETER stop "</s>"
PARAMETER stop "User:"
PARAMETER stop "Human:"

# 随机种子：固定种子可获得可复现的结果
PARAMETER seed 42

# Mirostat 参数：自动调节温度以维持目标熵
PARAMETER mirostat 2
PARAMETER mirostat_tau 5.0
PARAMETER mirostat_eta 0.1

# TFS (Tail Free Sampling)：截断低概率分布尾部
PARAMETER tfs 1.0

# Typical P：局部典型性采样
PARAMETER typical_p 1.0

# Penultimate last n：惩罚最后 N 个 token 的重复
PARAMETER penalty_last_n 64

# Penalty repeat / frequency / present
PARAMETER penalty_repeat 1.15
PARAMETER penalty_frequency 0.0
PARAMETER penalty_present 0.0
```

这些参数的含义和调优方法我们将在 04-3 节详细讨论。这里只需要知道它们都可以通过 PARAMETER 指令永久性地绑定到你的自定义模型上。

### TEMPLATE — 自定义提示模板

```dockerfile
# ChatML 格式（Qwen 系列默认）
TEMPLATE """{{- if .System }}{{ .System }}{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}
{{- end }}<|im_start|>assistant
"""

# Llama 格式
TEMPLATE """[INST] {{ .Prompt }} [/INST]
"""

# 自定义格式：带时间戳的对话日志
TEMPLATE """[{{ now | date "2006-01-02T15:04:05" }}] {{ if .System }}SYS: {{ .System }}{{ end }}
{{ range $i, $m := .Messages }}
[{{ $m.Role | upper }}] {{ $m.Content }}
{{ end }}
[ASSISTANT]
"""
```

TEMPLATE 是最强大的指令之一，它让你能完全控制消息如何被拼接成最终送入模型的 prompt 字符串。模板使用了 Go 的 `text/template` 语法，支持条件判断 (`if`)、循环 (`range`)、函数调用 (`date`, `json`, `trim_space` 等)。

### ADAPTER — LoRA 适配器加载

```dockerfile
# 加载单个 LoRA 适配器
ADAPTER ./my-lora-adapter.gguf

# 加载多个 LoRA 适配器（实验性）
ADAPTER ./domain-lora.gguf
ADAPTER ./style-lora.gguf
```

ADAPTER 指令用于加载 LoRA（Low-Rank Adaptation）格式的微调权重文件，可以在不修改基础模型的情况下注入特定领域的知识或能力。我们将在 04-4 节深入讲解。

### LICENSE — 许可证声明

```dockerfile
LICENSE Apache 2.0
LICENSE """Llama Community License
https://llama.meta.com/license/"""
```

### MESSAGE — Few-shot 示例消息

```dockerfile
# 添加示例对话来引导模型行为
MESSAGE user 请将 'hello' 翻译成中文。
MESSAGE assistant 你好！'hello' 在中文里是"你好"的意思。
MESSAGE user 请将 'goodbye' 翻译成中文。
MESSAGE assistant 再见！'goodbye' 在中文里通常翻译为"再见"或"拜拜"。
```

MESSAGE 指令提供了一种比 SYSTEM 更轻量的行为引导方式——通过具体的输入输出对（few-shot examples）让模型学会你期望的输出格式和风格。

## 完整 Modelfile 示例

下面是一个综合运用了所有主要指令的生产级 Modelfile：

```dockerfile
# ============================================================
#  Modelfile: tech-doc-translator
#  描述: 英文技术文档 → 中文专业翻译模型
#  基于: Qwen2.5 7B (中文能力最强的小型模型)
#  作者: AI Engineering Team
#  创建日期: 2024-12
# ============================================================

FROM qwen2.5:7b

# ---------- 系统提示词 ----------
SYSTEM """你是"TechDoc Pro"，一个专业的英中技术文档翻译引擎。

核心能力：
- 准确翻译计算机科学、软件工程、人工智能领域的技术文档
- 保持原文的技术精确性和逻辑结构
- 输出符合中文技术写作规范的专业译文

翻译规范：
1. 术语处理：
   - 标准技术术语首次出现时采用"中文译名（English Term）"格式
   - 后续出现直接使用中文译名
   - 无标准译名的术语保留英文原文
   - 示例：机器学习（Machine Learning）、容器化（Containerization）、Kubernetes
   
2. 代码与公式：
   - 所有代码块原样保留，不做任何翻译
   - 数学公式使用 LaTeX 格式
   - 命令行指令保持原样
   
3. 格式要求：
   - 使用 Markdown 格式输出
   - 保留原文的标题层级结构
   - 列表、表格结构保持一致
   - 超链接保留原始 URL

4. 禁止事项：
   - 不要添加译者注或解释性说明
   - 不要省略或概括任何内容
   - 不要改变原文的语气和立场
   - 不要输出除正文外的任何额外内容

工作流程：
接收英文文本 → 分析领域和术语 → 执行翻译 → 输出 Markdown 格式的中文译文
"""

# ---------- 推理参数 ----------
PARAMETER temperature 0.3          # 低温度确保翻译一致性
PARAMETER top_p 0.9                # 适度的核采样
PARAMETER top_k 40                 # 标准 Top-K
PARAMETER num_ctx 16384            # 支持长文档翻译
PARAMETER num_predict 8192         # 允许长输出
PARAMETER repeat_penalty 1.15      # 避免重复
PARAMETER stop "</think>"          # 防止思维链泄漏

# ---------- Few-shot 示例 ----------
MESSAGE user Translate the following technical documentation into Chinese:

## Overview
The Kubernetes control plane is the brain of your cluster, managing the state of your applications and the nodes that run them.

MESSAGE assistant ## 概述

Kubernetes 控制平面（Control Plane）是集群的大脑，负责管理应用程序的状态以及运行这些应用程序的节点（Node）。

MESSAGE user Translate:

## Key Components
- **API Server**: The front door to the control plane
- **etcd**: A consistent key-value store for cluster data
- **Scheduler**: Decides which nodes run which pods
- **Controller Manager**: Runs controller processes

MESSAGE assistant ## 关键组件

- **API Server（API 服务器）**：控制平面的前端入口
- **etcd**：用于存储集群数据的一致性键值存储（Key-Value Store）
- **调度器（Scheduler）**：决定哪些节点运行哪些 Pod
- **控制器管理器（Controller Manager）**：运行控制器进程

# ---------- 许可证 ----------
LICENSE """Apache 2.0
基础模型: Qwen2.5 (Apache 2.0)
本 Modelfile: MIT License
"""
```

这个 Modelfile 展示了生产级自定义模型的最佳实践：详细的系统提示词定义了角色边界和输出规范，精心调校的参数确保了翻译质量的一致性，Few-shot 示例提供了具体的格式参照，许可证声明保证了合规性。

## 创建和运行自定义模型

有了 Modelfile 之后，用两条命令就能完成从定义到运行的整个流程：

```bash
# 第一步：从 Modelfile 创建自定义模型
ollama create tech-doc-translator -f Modelfile

# 第二步：运行你的自定义模型
ollama run tech-doc-translator

# 测试效果
>>> 请翻译以下段落：
>>>
>>> Container orchestration automates the deployment, scaling,
>>> and management of containerized applications across clusters.
```

你会看到模型立即进入角色，输出符合规范的中文翻译：

```
容器编排（Container Orchestration）自动化了跨集群对容器化应用程序的部署、扩缩容和管理操作。
```

## Modelfile 与 Dockerfile 的对照表

为了帮助有 Docker 经验的开发者快速上手，这里是一份完整的对照表：

| Dockerfile 概念 | Modelfile 对应 | 说明 |
|----------------|---------------|------|
| `FROM ubuntu:22.04` | `FROM llama3.1:8b` | 继承基础镜像/模型 |
| `RUN apt-get install ...` | （无直接对应） | 模型不需要安装依赖 |
| `COPY file.txt /app/` | `ADAPTER ./lora.gguf` | 引入外部文件 |
| `ENV APP_ENV=prod` | `PARAMETER temperature 0.7` | 配置环境/参数 |
| `CMD ["nginx"]` | `SYSTEM "..."` + `PARAMETER` | 定义默认行为 |
| `EXPOSE 80` | （无直接对应） | Ollama 固定监听 11434 |
| `LABEL version="1.0"` | `LICENSE "Apache 2.0"` | 元数据标注 |
| `ENTRYPOINT` | `TEMPLATE "... "` | 控制入口逻辑 |

## 查看 Modelfile 内容

你可以随时查看已创建模型的 Modelfile：

```bash
# 查看完整的 Modelfile 定义
ollama show tech-doc-translator --modelfile

# 查看模型的基本信息
ollama show tech-doc-translator

# 只查看系统提示词部分
ollama show tech-doc-translator --modelfile | grep -A 50 "^SYSTEM"
```

输出类似：

```
Model:     tech-doc-translator
Modified:  2024-12-15T10:30:00+08:00
Size:      4.3 GB
Parameters:  7B
Template:
  [INST] {{ .Prompt }} [/INFO]
License:   Apache 2.0
Systems:
  你是"TechDoc Pro"，一个专业的英中技术文档翻译引擎...
Parameters:
  temperature    0.3    top_k    40    top_p    0.9
  num_ctx        16384  num_predict  8192  repeat_penalty  1.15
```

## 导出和分享 Modelfile

自定义模型可以导出为标准的 Modelfile 文件进行分享：

```bash
# 导出 Modelfile
ollama show my-model --modelfile > my-model.Modelfile

# 同事拿到 Modelfile 后可以直接复刻你的模型
# (前提是他们也有相同的基础模型)
ollama create my-model -f my-model.Modelfile
```

这使得团队内部的模型配置可以实现版本控制和协作管理——把 Modelfile 提交到 Git 仓库，团队成员拉取后即可获得完全一致的模型配置。

## 常见用法与误用

### ✅ 正确用法

1. **一个 Modelfile = 一个特定用途的模型**
   ```dockerfile
   # 好：每个模型有明确的单一职责
   FROM qwen2.5:7b
   SYSTEM "你是SQL查询生成器"
   
   FROM qwen2.5:7b
   SYSTEM "你是代码审查助手"
   ```

2. **参数针对场景优化**
   ```dockerfile
   # 翻译任务：低温度保证一致性
   PARAMETER temperature 0.3
   
   # 创意写作：高温度激发多样性
   PARAMETER temperature 1.2
   ```

3. **Few-shot 示例控制输出格式**
   ```dockerfile
   MESSAGE user 输入: 苹果
   MESSAGE assistant {"category": "水果", "color": "红色", "taste": "甜"}
   ```

### ❌ 常见误用

1. **SYSTEM 过长导致上下文浪费**
   ```dockerfile
   # 差：5000字的系统提示词占用了大量有效上下文
   SYSTEM """这里写了五千字的公司介绍、产品手册、FAQ..."""
   ```
   **正确做法**：SYSTEM 应该精炼地定义角色和行为约束，详细知识通过 RAG 注入。

2. **忽略基础模型的默认模板**
   ```dockerfile
   # 差：自定义 TEMPLATE 可能破坏模型的训练格式
   TEMPLATE "{{ .Prompt }}"  # 太简陋，丢失了特殊 token
   ```
   **正确做法**：除非你有充分的理由，否则不要覆盖默认 TEMPLATE。先用 `ollama show <model>` 查看原始模板再决定是否修改。

3. **FROM 指向不存在的模型**
   ```dockerfile
   # 差：如果本地没有这个模型，create 会失败
   FROM nonexistent-model:99b
   ```
   **正确做法**：先 `ollama pull` 基础模型，或者使用确切的名称。

## 本章小结

这一节我们全面介绍了 Modelfile 的基础语法：

1. **Modelfile 是 Ollama 版的 Dockerfile**，用声明式指令定义自定义模型
2. **七条核心指令**：FROM（必选）、SYSTEM、PARAMETER、TEMPLATE、ADAPTER、LICENSE、MESSAGE
3. **最小可用 Modelfile 只需两行**：FROM + SYSTEM
4. **生产级 Modelfile** 应包含详细的系统提示词、精心调校的参数、Few-shot 示例和许可证声明
5. **`ollama create <name> -f <Modelfile>`** 创建模型，**`ollama run <name>`** 运行模型
6. **Modelfile 可以导出分享**，支持团队协作和版本控制

下一节我们将深入探讨 SYSTEM 提示词的艺术——如何写出真正有效的系统提示词。
