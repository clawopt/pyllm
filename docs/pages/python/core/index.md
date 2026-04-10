# Python 核心教程大纲

---

## 总体设计思路

Python 是大模型开发的"第一语言"——无论是数据处理（Pandas/NumPy）、模型训练（PyTorch/Transformers）、应用开发（LangChain/LlamaIndex），还是推理部署（vLLM/FastAPI），所有工具链都以 Python 为基础。不会 Python，就进不了 LLM 开发的大门。

本教程面向零基础或初学者，从 Python 安装开始，系统讲解核心语法和编程范式。教程的设计遵循以下原则：

1. **面向大模型开发者**：所有示例和练习都围绕 LLM 开发场景，不教 MNIST
2. **从基础到进阶**：数据类型 → 函数 → 数据结构 → 面向对象 → 并发编程，逐步递进
3. **实战导向**：每章都有可运行的代码示例和常见误区提醒
4. **为后续教程铺路**：学完本教程后，可以无缝衔接 NumPy / Pandas / LangChain 等教程

---

## 第一章：介绍与环境搭建（2节）

### 定位
从零开始。安装 Python，配置开发环境，运行第一行代码。

### 00-intro 基本介绍
- Python 的历史与特点
- Python 在 LLM 开发中的核心地位
- Python 2 vs Python 3

### 01-install 安装与环境搭建
- Python 安装方式（官方 / Anaconda / pyenv）
- 虚拟环境配置（venv / conda）
- IDE 推荐（VS Code / PyCharm）
- 第一个 Python 程序

---

## 第二章：数据类型与分支控制（4节）

### 定位
Python 的基本数据类型和控制流。这是写任何程序的基础。

### 10-data-number 基本类型
- 整数、浮点数、布尔值
- 数值运算与类型转换
- 常见陷阱：浮点精度问题

### 11-data-str 字符串与编码
- 字符串的基本操作
- 格式化字符串（f-string / format / %）
- 编码问题：UTF-8 vs GBK

### 12-data-object 变量和对象
- Python 的"一切皆对象"
- 可变对象 vs 不可变对象
- 变量赋值与引用

### 13-data-ifwhile 条件、循环与异常
- if/elif/else 条件判断
- for/while 循环
- try/except 异常处理

---

## 第三章：函数与参数（5节）

### 定位
函数是代码复用的基本单元。Python 的函数参数非常灵活，掌握它是写出优雅代码的关键。

### 20-func-basics 函数基础
- 函数定义与调用
- 参数与返回值
- 作用域与命名空间

### 21-func-advanced 高级参数特性
- 默认参数、可变参数（*args/**kwargs）
- 关键字参数与位置参数
- 参数的常见陷阱：可变默认值

### 22-func-functional 函数式编程工具
- map / filter / reduce
- lambda 表达式
- 偏函数 functools.partial

### 23-func-closure 闭包与装饰器
- 闭包的原理与用途
- 装饰器的基本写法
- 常用装饰器：@property / @staticmethod / @classmethod

### 25-func-special 递归与特殊函数
- 递归的原理与适用场景
- 尾递归优化
- 生成器函数（yield）

---

## 第四章：数据结构（4节）

### 定位
Python 内置的四大核心数据结构。理解它们的特性，才能选择正确的工具。

### 30-struct-list 列表
- 列表的创建、索引、切片
- 列表方法与常见操作
- 列表推导式

### 31-struct-tuple 元组
- 元组与列表的区别
- 元组解包
- 命名元组

### 32-struct-dict 字典
- 字典的创建与操作
- 字典推导式
- 字典的遍历与排序

### 33-struct-set 集合
- 集合的创建与操作
- 集合运算（交集/并集/差集）
- 集合的去重应用

---

## 第五章：集合与映射（4节）

### 定位
collections 模块中的高级数据结构，解决常见场景的效率问题。

### 40-collection-dequeue 双端队列
- deque vs list 的性能对比
- 队列与栈的实现

### 41-collection-defaultdict defaultdict
- 默认值字典的用法
- 消除 KeyError 的优雅方式

### 42-collection-counter Counter
- 计数器的用法
- 词频统计实战

### 43-collection-namedtuple 命名元组
- 命名元组的创建与使用
- 与普通元组和 dataclass 的对比

---

## 第六章：高级特性（3节）

### 定位
Python 的"灵魂特性"——迭代器、推导式、生成器。理解它们，才能写出 Pythonic 的代码。

### 50-pro-iterator 迭代器
- 迭代器协议（__iter__ / __next__）
- 迭代器 vs 可迭代对象
- itertools 常用函数

### 51-pro-comprehensions 推导式
- 列表推导式 / 字典推导式 / 集合推导式
- 推导式 vs for 循环的性能对比
- 嵌套推导式

### 54-pro-generator 生成器
- 生成器函数（yield）
- 生成器表达式
- 生成器在数据处理管道中的应用

---

## 第七章：模块使用（5节）

### 定位
模块是 Python 代码组织的基本方式。掌握模块，才能管理大型项目。

### 60-module-basics 模块基础
- 模块的概念与创建
- __name__ == '__main__' 的用法

### 61-module-import 模块的导入
- import 语句的各种写法
- 相对导入与绝对导入
- 循环导入的解决方法

### 62-module-package 模块的组织
- 包（Package）的结构
- __init__.py 的作用
- 命名空间包

### 63-module-distribution 模块的安装与发布
- pip 与 requirements.txt
- setup.py / pyproject.toml
- 发布到 PyPI

### 64-module-advanced 模块的高级特性
- 动态导入（importlib）
- 插件架构模式
- 热重载

---

## 第八章：面向对象编程（7节）

### 定位
面向对象是构建大型 LLM 应用的基础——LangChain 的 Chain、Agent、Tool 都是类。理解 OOP，才能读懂框架源码。

### 70-oop-basics 类与对象基础
- 类的定义与实例化
- 属性与方法
- self 的含义

### 71-oop-attributes 属性与方法
- 实例属性 vs 类属性
- 实例方法 vs 类方法 vs 静态方法
- 属性描述符

### 72-oop-inheritance 继承与多态
- 单继承与多继承
- 方法重写与 super()
- 多态的实现

### 73-oop-magic 魔术方法
- __init__ / __repr__ / __str__
- __getitem__ / __setitem__ / __len__
- 运算符重载

### 74-oop-encapsulation 封装与访问控制
- Python 的"约定"封装（_ / __）
- @property 的用法
- 数据类 dataclass

### 75-oop-advanced 高级特性
- 元类（metaclass）
- 描述符协议
- 抽象基类（ABC）

### 76-oop-patterns 设计模式
- 常用设计模式在 Python 中的实现
- 单例 / 工厂 / 策略 / 观察者模式
- LLM 框架中的设计模式实例

---

## 第九章：输入输出（6节）

### 定位
文件操作是数据处理的起点——读取训练数据、保存模型权重、写入日志，都离不开 IO。

### 80-io-file 文件操作基础
- 文件的打开与关闭
- 读写模式（r/w/a/b）
- 编码问题

### 81-io-path 路径操作
- os.path vs pathlib
- 路径拼接与规范化
- 目录遍历

### 82-io-context 上下文管理器
- with 语句的原理
- 自定义上下文管理器
- contextlib 工具

### 83-io-text-binary 文本与二进制
- 文本模式 vs 二进制模式
- 编码与解码
- 二进制数据的处理

### 84-io-serialization 数据序列化
- JSON / pickle / msgpack
- 序列化的安全注意事项
- 与 LLM 数据格式的对接

### 85-io-stdio 标准输入输出
- stdin / stdout / stderr
- 命令行参数解析（argparse）
- 日志输出（logging）

---

## 第十章：并发编程（5节）

### 定位
LLM 应用天然需要并发——同时请求多个 API、并行处理数据、异步流式输出。掌握并发编程，是从"能跑"到"能跑好"的关键。

### 90-concurrency-basics 并发编程基础
- 并发 vs 并行
- CPU 密集型 vs IO 密集型
- Python 的 GIL 限制

### 91-concurrency-multiprocess 多进程编程
- multiprocessing 模块
- 进程间通信（Queue / Pipe）
- 进程池

### 92-concurrency-threading 多线程编程
- threading 模块
- 线程安全与锁
- 线程池

### 93-concurrency-async 异步编程
- async/await 语法
- asyncio 事件循环
- aiohttp 异步 HTTP 请求

### 94-concurrency-advanced 并发进阶与实践
- 并发模型选择指南
- 异步 + 多进程混合
- LLM API 的并发请求实战

---

## 附录：常用模块（4节）

### X0-datetime-random 日期时间与随机数
- datetime / timedelta / timezone
- random 模块与随机种子
- LLM 场景：时间戳处理与数据采样

### X1-regex 正则表达式
- re 模块的基本用法
- 常用正则模式
- LLM 场景：文本清洗与信息提取

### X2-system-process 系统与进程
- os / sys / subprocess
- 环境变量管理
- LLM 场景：调用外部命令

### X3-tools 工具模块
- typing 类型注解
- dataclass 数据类
- enum 枚举
- LLM 场景：类型安全的 API 设计
