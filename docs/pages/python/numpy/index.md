# NumPy 教程大纲

---

## 总体设计思路

NumPy 是 Python 科学计算的基石——PyTorch 的 Tensor 底层就是 NumPy 的 ndarray，HuggingFace 的数据处理离不开 NumPy 数组，LangChain 的 Embedding 向量本质就是 NumPy 数组。不理解 NumPy，你永远只是在"调包"，遇到性能问题只能干瞪眼。

本教程不仅讲解 NumPy 的核心 API，更通过大量 LLM 相关实战（词嵌入、自注意力、LoRA、评估指标等），让你真正理解 NumPy 在大模型开发中的价值。教程的设计遵循以下原则：

1. **从第一天就面向大模型**：所有示例围绕 Transformer、注意力机制、语言建模展开
2. **从 API 到原理到实战**：先学会用，再理解为什么，最后用 NumPy 手写 LLM 核心组件
3. **22 章覆盖全链路**：基础概念 → 数组操作 → 数学运算 → 线性代数 → 内存优化 → 8 个 LLM 实战项目
4. **面试深度**：每节都讲清楚底层原理，面试官最爱问的广播机制、视图vs副本、内存布局都有覆盖

---

## 第1章：基础概念（5节）

### 定位
建立对 NumPy 的整体认知。理解 ndarray 为什么比 Python 列表快，以及 NumPy 在 LLM 开发中的位置。

### 01-00 NumPy是什么
- NumPy 的历史与定位
- ndarray vs Python list 的本质区别

### 01-01 NumPy优势对比
- 性能对比：为什么 NumPy 快 100 倍
- 向量化运算 vs 循环
- 底层 C/Fortran 实现原理

### 01-02 安装与配置
- pip / conda 安装
- 版本验证与常见问题

### 01-03 导入与约定
- `import numpy as np` 的约定
- 命名空间与常用别名

### 01-04 NumPy与LLM开发
- NumPy 在 PyTorch / Transformers / LangChain 中的角色
- 学习路线图：NumPy → PyTorch → Transformers → LangChain

---

## 第2章：ndarray属性（4节）

### 定位
ndarray 是 NumPy 的核心数据结构。理解它的属性，是正确使用 NumPy 的前提。

### 02-01 ndarray属性
- shape / ndim / size / dtype / itemsize
- 属性之间的数学关系

### 02-02 dtype深入理解
- NumPy 的数据类型体系
- 类型转换与溢出风险
- LLM 场景：float16 vs float32 vs bfloat16

### 02-03 从列表创建
- np.array() 的参数详解
- 嵌套列表到多维数组
- 常见陷阱：列表中混合类型

### 02-04 AI中的应用
- ndarray 在 AI 数据流中的位置
- 批量数据的表示方式

---

## 第3章：数组创建（6节）

### 定位
掌握各种创建数组的方法，为后续操作打基础。

### 03-01 zeros与ones
- np.zeros / np.ones / np.full / np.empty
- 形状参数与 dtype 指定

### 03-02 eye与identity
- 单位矩阵的创建
- 对角矩阵与带状矩阵

### 03-03 arange与linspace
- 等差数列 vs 等间距数列
- 浮点数序列的精度问题

### 03-04 tile与meshgrid
- 数组重复与网格生成
- 坐标矩阵在可视化中的应用

### 03-05 随机数组
- np.random 模块概览
- 均匀分布 / 正态分布 / 整数随机
- 随机种子与可复现性

### 03-06 从数据创建
- 从文件 / 字符串 / Pandas 创建数组
- 数据类型推断与显式指定

---

## 第4章：形状操作（6节）

### 定位
形状操作是 NumPy 最常用的功能之一——数据预处理、模型输入、批量处理都离不开 reshape 和转置。

### 04-01 shape与reshape
- reshape 的原理与用法
- -1 自动推断维度
- 常见陷阱：reshape 不改变数据

### 04-02 flatten与ravel
- 展平的两种方式
- copy vs view 的区别

### 04-03 转置操作
- .T / np.transpose / np.swapaxes
- 多维转置的轴顺序

### 04-04 expand与squeeze
- 增加维度与删除维度
- 广播前的维度对齐

### 04-05 concatenate与stack
- 水平/垂直拼接
- np.stack vs np.concatenate 的区别

### 04-06 split分割
- np.split / np.hsplit / np.vsplit
- 不等分分割

---

## 第5章：索引与切片（5节）

### 定位
索引是数据访问的基础。NumPy 的高级索引功能远超 Python 列表，是高效数据处理的关键。

### 05-01 基本索引
- 一维/多维索引与切片
- 步长与负索引

### 05-02 整数数组索引
- 花式索引的原理
- 多维整数索引

### 05-03 布尔掩码
- 条件筛选与布尔索引
- 多条件组合
- LLM 场景：过滤低质量数据

### 05-04 花式索引
- 高级索引模式
- 索引的组合使用

### 05-05 LLM嵌入查找
- 词嵌入矩阵的索引操作
- batch 查找的向量化实现

---

## 第6章：运算与性能（4节）

### 定位
NumPy 的核心优势——向量化运算。理解它，才能写出高效的数值计算代码。

### 06-01 元素级运算
- 算术运算 / 比较运算 / 逻辑运算
- 逐元素操作的向量化

### 06-02 向量化性能
- 向量化 vs 循环的性能对比
- 何时用向量化，何时用循环

### 06-03 矩阵乘法
- np.dot / np.matmul / @ 运算符
- 批量矩阵乘法

### 06-04 LLM自注意力
- 用纯 NumPy 实现自注意力计算
- Q·K^T → Softmax → ·V 的完整流程

---

## 第7章：广播机制（3节）

### 定位
广播是 NumPy 最强大也最容易出错的特性。理解广播规则，是避免 Bug 的关键。

### 07-01 广播规则
- 三条广播规则详解
- 形状兼容性判断

### 07-02 数组扩张
- 广播的内部机制
- 维度对齐的过程

### 07-03 LLM位置编码
- 用广播实现正弦位置编码
- 位置编码矩阵的向量化计算

---

## 第8章：数学函数（3节）

### 定位
NumPy 提供了丰富的数学函数，覆盖了深度学习中常用的激活函数和数值运算。

### 08-01 指数与对数
- np.exp / np.log / np.log10 / np.log2
- 数值稳定性：log-sum-exp 技巧

### 08-02 激活函数
- ReLU / Sigmoid / Tanh / GELU 的 NumPy 实现
- 梯度计算

### 08-03 clip截断
- np.clip 的用法
- 梯度裁剪与数值范围控制

---

## 第9章：统计与损失（4节）

### 定位
统计函数是评估模型性能的基础——损失函数、评估指标都离不开统计计算。

### 09-01 基础统计
- mean / std / var / min / max / median

### 09-02 axis统计
- axis 参数的深入理解
- 按行/按列/按通道统计

### 09-03 cumsum累加
- 累加/累乘操作
- 前缀和在 LLM 中的应用

### 09-04 LLM困惑度
- 用 NumPy 计算困惑度（PPL）
- 交叉熵损失的实现

---

## 第10章：线性代数（5节）

### 定位
线性代数是深度学习的数学基础。NumPy 的 linalg 模块提供了完整的线性代数工具。

### 10-01 点积
- 向量点积与矩阵乘法
- 内积的几何意义

### 10-02 特征值与特征向量
- np.linalg.eig / eigh
- PCA 降维的原理

### 10-03 奇异值分解
- SVD 的原理与 NumPy 实现
- SVD 在 LoRA 中的应用

### 10-04 矩阵求逆与伪逆
- np.linalg.inv / pinv
- 正规方程求解

### 10-05 范数计算
- L1 / L2 / Frobenius 范数
- 正则化与梯度裁剪

---

## 第11章：随机数生成（5节）

### 定位
随机数在深度学习中无处不在——权重初始化、数据增强、Dropout、采样。掌握 NumPy 的随机数生成是必备技能。

### 11-01 随机种子与可复现性
- np.random.seed 的原理
- 可复现实验的设置方法

### 11-02 常见分布
- 均匀分布 / 正态分布 / 二项分布 / 泊松分布
- 权重初始化中的分布选择

### 11-03 随机采样
- np.random.choice 的用法
- 词汇表采样与 Top-K/Top-P

### 11-04 随机排列
- shuffle vs permutation
- 数据集打乱与交叉验证

### 11-05 随机初始化权重
- Xavier / He 初始化的 NumPy 实现
- 不同初始化方法对训练的影响

---

## 第12章：内存布局（4节）

### 定位
理解内存布局是优化 NumPy 性能的关键——C 顺序 vs Fortran 顺序、视图 vs 副本，直接影响计算效率。

### 12-01 C顺序与Fortran顺序
- 行优先 vs 列优先
- 内存布局对遍历性能的影响

### 12-02 视图与副本
- view vs copy 的区别
- 什么时候会触发数据复制

### 12-03 ndarray.view
- view 的创建与使用
- 共享内存的注意事项

### 12-04 原地操作
- += / -= / *= 等原地运算
- out 参数的用法
- 内存优化技巧

---

## 第13章：结构化数组（3节）

### 定位
结构化数组让你在一个数组中存储异构数据——类似数据库的表结构，在处理 LLM 批次数据时非常有用。

### 13-01 创建结构化数组
- dtype 的结构化定义
- 字段名与类型

### 13-02 字段访问与操作
- 按字段名访问
- 字段的增删改

### 13-03 HuggingFace导出
- 将 HuggingFace Dataset 导出为结构化数组
- 与 PyTorch Tensor 的互转

---

## 第14章：文件IO（3节）

### 定位
数据的持久化存储——训练好的嵌入矩阵、预处理后的数据集，都需要高效的 IO 操作。

### 14-01 文本文件IO
- np.loadtxt / np.savetxt
- CSV 格式的读写

### 14-02 二进制文件IO
- np.save / np.savez / np.load
- .npy / .npz 格式的优势

### 14-03 内存映射
- np.memmap 的原理与用法
- 超大数组的处理方案

---

## 实战一：词嵌入矩阵（4节）

### 定位
第一个 LLM 实战项目。用纯 NumPy 实现词嵌入的完整流程：词汇表 → 嵌入矩阵 → 查找 → 相似度计算。

### 15-01 词汇表与索引映射
- 构建 token → index 的双向映射
- 特殊 token 的处理

### 15-02 嵌入矩阵创建
- 随机初始化与预训练加载
- 嵌入矩阵的形状与存储

### 15-03 嵌入查找实现
- 索引查找的向量化实现
- batch 查找

### 15-04 词向量相似度可视化
- 余弦相似度计算
- t-SNE 降维可视化

---

## 实战二：多头自注意力（4节）

### 定位
用纯 NumPy 手写 Transformer 的核心——多头自注意力机制。理解 Q/K/V 投影、缩放点积注意力、因果掩码。

### 16-01 QKV投影矩阵
- 线性变换的 NumPy 实现
- 多头并行的 QKV 计算

### 16-02 缩放点积注意力
- Scaled Dot-Product Attention 的完整实现
- Softmax 的数值稳定性

### 16-03 因果掩码
- Causal Mask 的原理与实现
- 自回归生成的关键约束

### 16-04 前向传播模拟
- 完整的简化 Transformer 块前向传播
- 各组件的协同工作

---

## 实战三：LayerNorm与残差（3节）

### 定位
实现 Transformer 的另外两个关键组件——LayerNorm 和残差连接。

### 17-01 LayerNorm实现
- LayerNorm 的原理与 NumPy 实现
- 与 BatchNorm 的对比

### 17-02 残差连接模拟
- 残差连接的原理
- 梯度流分析

### 17-03 NumPy vs PyTorch对比
- 同一功能两种实现的代码对比
- 性能与 API 差异分析

---

## 实战四：GPT文本生成（4节）

### 定位
综合前面的知识，构建一个简化但完整的 GPT-like 文本生成系统。

### 18-01 预训练嵌入概述
- GPT 架构的整体流程
- 从嵌入到输出的数据流

### 18-02 Transformer解码器
- 解码器的前向传播实现
- 因果掩码的应用

### 18-03 温度采样
- Temperature / Top-K / Top-P 采样策略
- NumPy 实现

### 18-04 简单文本序列生成
- 完整的自回归生成循环
- 生成质量的初步评估

---

## 实战五：LoRA低秩适配（3节）

### 定位
用 NumPy 理解 LoRA 的数学原理——低秩矩阵分解如何大幅减少微调参数量。

### 19-01 LoRA原理
- LoRA 的核心思想：W = W₀ + BA
- 低秩分解的数学基础

### 19-02 LoRA前向传播
- 冻结权重 + LoRA 适配器的 NumPy 实现
- 前向传播的计算流程

### 19-03 参数量节省
- 不同秩 r 下的参数量对比
- 压缩比计算

---

## 实战六：HuggingFace集成（3节）

### 定位
NumPy 与 HuggingFace 生态的互操作——将 Tokenizer 和 Datasets 的输出转换为 NumPy 数组。

### 20-01 tokenizer转NumPy
- HuggingFace Tokenizer 输出转 NumPy
- input_ids / attention_mask 的数组操作

### 20-02 数据集转NumPy
- HuggingFace Datasets 转 NumPy
- 批量数据处理

### 20-03 NumPy DataLoader
- 基于 NumPy 的自定义 DataLoader
- mini-batch 生成

---

## 实战七：框架互操作（3节）

### 定位
NumPy 与 PyTorch / TensorFlow 之间的数据互转——理解框架底层的共享内存机制。

### 21-01 PyTorch互操作
- torch.from_numpy() / tensor.numpy()
- 共享内存机制

### 21-02 TensorFlow互操作
- tf.convert_to_tensor / tensor.numpy()
- 与 TF 的数据交换

### 21-03 预处理与训练
- NumPy 预处理 + PyTorch 训练的混合工作流
- 最佳实践

---

## 实战八：LLM评估（3节）

### 定位
用 NumPy 实现 LLM 的常用评估指标——困惑度、BLEU、ROUGE。

### 22-01 困惑度计算
- 困惑度的数学定义与 NumPy 实现
- 不同模型的 PPL 对比

### 22-02 BLEU与ROUGE指标
- BLEU / ROUGE 的计算原理
- NumPy 实现中的统计操作

### 22-03 文本多样性评估
- Distinct / Entropy 等多样性指标
- 生成质量的综合评估
