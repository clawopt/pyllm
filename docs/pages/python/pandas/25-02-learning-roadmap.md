# 学习路线图

### 学习路线图：从入门到 AI 工程师

#### Pandas 学习的四个阶段

Pandas 的学习路径不是线性的——它更像一棵树。先扎根（基础语法），再长主干（核心操作），然后分叉出多个专业方向（AI/工程/可视化），最后在树冠上开花结果（综合项目）。下面是面向 LLM/AI 应用场景的分阶段路线图。

##### 阶段一：基础夯实（第1-5章）

**目标**：能独立完成数据读取、清洗、导出的完整流程。

**关键能力**：
- 熟练使用 Series / DataFrame 的创建和基本属性
- 掌握 CSV、JSON、Excel、Parquet 的读写及参数调优
- 理解缺失值处理（fillna / dropna / interpolate）的基本策略
- 能用 describe() / info() 快速了解数据全貌

**阶段验收标准**：

```python
import pandas as pd

df = pd.read_csv("sft_data.csv")
print(df.info())
print(df.describe())

df_clean = df.dropna(subset=["instruction", "output"])
df_clean["text_length"] = df_clean["instruction"].str.len()
df_clean.to_parquet("sft_cleaned.parquet", index=False)

print(f"原始: {len(df)} 行, 清洗后: {len(df_clean)} 行")
```

**常见陷阱**：
- 混淆 `loc` 和 `iloc` —— loc 用标签，iloc 用位置
- 链式赋值产生 SettingWithCopyWarning —— 用 `.copy()` 或一次性赋值解决
- 忘记 `inplace=True` 不返回新对象 —— 大多数情况推荐不用 inplace

**建议练习**：
- 找一个真实的 SFT 数据集（如 Alpaca / ShareGPT），完成读取→清洗→统计→导出的完整流程
- 尝试对比 CSV 和 Parquet 的文件大小和读取速度差异

---

##### 阶段二：核心进阶（第6-11章）

**目标**：能用 Pandas 完成复杂的数据变换和分析任务。

**关键能力**：
- 灵活运用布尔索引、query()、isin() / between() 进行数据筛选
- 掌握 apply() / map() / transform() 三者的区别和适用场景
- 理解 groupby + agg / transform / filter 的完整体系
- 能用 merge / join 处理多表关联

**阶段验收标准**：

```python
import pandas as pd

df = pd.read_parquet("eval_results.parquet")

top_models = (
    df.query("task in ['reasoning', 'coding', 'math']")
    .groupby("model")["score"].mean()
    .nlargest(10)
    .reset_index()
)

pivot = df.pivot_table(
    index="model", columns="task",
    values="score", aggfunc="mean"
)

print(top_models)
print(pivot.head())
```

**性能意识觉醒**：
这个阶段需要开始关注代码效率。比如：

| 操作 | 低效写法 | 高效写法 | 加速比 |
|------|----------|----------|--------|
| 条件筛选 | df[df.col > 0] | df.query("col > 0") | ~2x |
| 字符串长度 | df.col.apply(len) | df.col.str.len() | ~10x |
| 分组统计 | groupby + apply | groupby + agg | ~5-50x |
| 多条件合并 | 循环 merge | reduce(merge, dfs) | O(n)→O(log n) |

**建议练习**：
- 构建一个模型评估报告系统：多指标 → pivot_table → 排名 → 可视化
- 用 groupby 完成 API 成本按模型/日期/用户的分维度统计

---

##### 阶段三：AI 专业方向（第12-20章）

**目标**：将 Pandas 深度融入 LLM/AI 工作流。

**三条分支路径**：

**路径 A：数据处理工程师**
- 重点章节：12（合并）、13（拼接）、14（时间序列）、16（高级操作）
- 核心技能：大规模数据 ETL、增量更新、时序分析
- 典型输出：SFT 数据管道、评估数据管理器

**路径 B：AI 分析研究员**
- 重点章节：15（可视化）、17（性能优化）、18（PandasAI）
- 核心技能：快速探索性分析、自然语言查询、报告生成
- 典型输出：自动评估报告、交互式成本仪表板

**路径 C：RAG/Knowledge 工程师**
- 重点章节：19（RAG知识库）、20（综合项目）
- 核心技能：知识库 Schema 设计、检索增强生成流水线
- 典型输出：RAG 知识库管理系统、文档检索服务

**阶段验收标准（以路径 A 为例）**：

```python
class SFTDataPipeline:
    def __init__(self):
        self.raw = None
        self.clean = None
        self.scored = None
        self.final = None

    def run(self, input_path: str, output_path: str):
        self._load(input_path)
        self._clean()
        self._score_quality()
        self._engineer_features()
        self._stratified_sample()
        self._export(output_path)
        return self.final

pipeline = SFTDataPipeline()
result = pipeline.run("raw_sft.jsonl", "sft_train.jsonl")
print(f"最终数据集: {len(result)} 条")
```

---

##### 阶段四：架构与生态（第21-25章）

**目标**：能设计和实现生产级的数据处理架构。

**关键能力**：
- LangChain Agent 中集成 Pandas 作为工具后端
- MCP 协议构建标准化的数据服务接口
- Dask / Modin / Ray 进行分布式扩展
- 多模态数据（图像/音频）的统一管理
- 性能优化与架构选型的决策能力

**架构决策参考**：

```
数据量 < 100MB   →  Pandas 单机（首选）
数据量 100MB-10GB →  Pandas + 分块 + 类型优化
数据量 10GB-100GB  →  Dask 延迟执行 或 Modin
数据量 > 100GB     →  Ray Data 或 Spark
实时流式           →  Polars（流式查询）或 DuckDB
```

---

#### 按角色定制的学习路径

##### 角色 1：LLM 应用开发工程师

```
第1-5章  →  数据基础（3天）
第6-8章  →  清洗筛选（2天）
第9-11章 →  变换聚合（3天）← 重点
第12-13章 → 多表操作（2天）
第18章   → PandasAI 自然语言查询（1天）
第20章   → 综合项目实战（3天）
第21章   → LangChain Agent 集成（2天）← 重点
总计：~16天
```

**核心场景**：SFT 数据构建、模型评估报告、API 成本追踪

##### 角色 2：RAG / 知识库工程师

```
第1-5章  →  数据基础（2天）
第6-7章  →  文本清洗（2天）← 重点
第9章    →  特征工程（2天）
第12-13章 → 数据整合（2天）
第15章   → 可视化（1天）
第19章   → RAG 知识库（3天）← 核心
第22章   → MCP 服务化（2天）
第23章   → 多模态扩展（2天）
第25章   → 总结回顾（1天）
总计：~17天
```

**核心场景**：文档入库管理、元数据分析、检索质量监控

##### 角色 3：MLOps / 平台工程师

```
第1-5章  →  数据基础（2天）
第14章   → 时间序列（2天）← 重点
第16章   → 高级操作（2天）
第17章   → 性能优化（3天）← 核心
第20章   → 项目实战（2天）
第24章   → 分布式计算（3天）← 核心
第22章   → MCP 协议（2天）
第25章   → 总结回顾（1天）
总计：~17天
```

**核心场景**：大规模数据管道、训练日志监控、分布式推理数据预处理

---

#### 学习资源推荐

##### 官方文档（必读）

| 资源 | 地址 | 说明 |
|------|------|------|
| Pandas 官方文档 | https://pandas.pydata.org/docs/ | 最权威的 API 参考 |
| Pandas 3.0 迁移指南 | docs/whatsnew/v3.0.0.html | CoW、PyArrow 等新特性 |
| User Guide | docs/user_guide/ | 概念性的最佳实践 |

##### 社区资源

| 资源 | 适用阶段 |
|------|----------|
| "Python for Data Analysis" Wes McKinney 著 | 阶段一→二 |
| Kaggle Pandas Course（免费） | 阶段一 |
| Pandas Cookbook | 阶段二→三 |
| Real Python Pandas Tutorials | 全阶段 |
| Thomas Askew's Pandas Proficiency（YouTube） | 阶段二→三 |

##### AI 方向专项资源

| 资源 | 关联章节 |
|------|----------|
| PandasAI 官方文档 | 第18章 |
| LangChain 文档 - Tools | 第21章 |
| Model Context Protocol 规范 | 第22章 |
| Dask 文档 | 第24章 |
| Hugging Face Datasets 文档 | 全程关联 |

---

#### 常见学习误区

##### 误区 1：死记 API，不理解原理

很多初学者试图背诵所有函数签名。更好的方式是理解 **"数据形状思维"**——每一步操作都在改变数据的形状（行数、列数、索引结构），而不是记忆参数列表。

**正确思路**：
```
我想做什么？→ 数据会变成什么样？→ 哪个操作最直接？
```

##### 误区 2：忽略类型系统

Pandas 3.0 的 PyArrow 后端和 Nullable 类型是革命性变化。不理解 dtype 会导致：
- 内存浪费 5-10 倍
- 混合类型列的隐式转换 bug
- 合并时的类型不匹配错误

**建议**：从第一天就养成用 `df.dtypes` 和 `memory_usage(deep=True)` 检查数据的习惯。

##### 误区 3：用 Pandas 做所有事

Pandas 不是万能的：
- 纯数值计算 → NumPy 更快
- SQL 类查询 → DuckDB / Polars 更高效
- 图像/音频处理 → 专用库（PIL / librosa）
- 分布式计算 → Dask / Spark / Ray

**原则**：Pandas 是"胶水层"，负责数据的整理和编排，重计算交给专用引擎。

##### 误区 4：不关注 Pandas 版本变化

Pandas 3.0 引入了 Copy-on-Write 默认开启、PyArrow 字符串后端等重大变更。如果教程或代码基于旧版本，可能产生完全不同的行为。

**建议**：
```python
import pandas as pd
print(pd.__version__)   # 确认版本
pd.options.mode.copy_on_write = True  # 显式开启 CoW
```

---

#### 从这里出发：下一步行动

完成本教程全部 25 章后，你已经具备了以下能力：

✅ 用 Pandas 高效处理结构化数据
✅ 构建 LLM 训练/评估/RAG 的完整数据管道
✅ 通过 PandasAI 实现自然语言数据查询
✅ 将数据能力封装为 Agent 工具或 MCP 服务
✅ 用 Dask/Modin 扩展到分布式场景
✅ 管理多模态（文本+图像+音频）的训练数据

**推荐的下一步方向**：

1. **深入 Polars** — 新一代 DataFrame 库，查询性能领先 Pandas 5-50 倍
2. **DuckDB 集成** — 内嵌 OLAP 数据库，SQL 查询零拷贝
3. **Kubeflow / Airflow** — 将 Pandas 管道编排为生产级工作流
4. **Feature Store** — Feast / Tecton，在线/离线特征统一管理
5. **LLMOps 平台** — Weights & Biases / MLflow，实验跟踪与模型注册

Pandas 是起点，不是终点。它在 AI 工程师的工具箱中占据着不可替代的位置——就像锤子之于木匠，键盘之于程序员。掌握它，你就在数据驱动的 AI 时代站稳了脚跟。
