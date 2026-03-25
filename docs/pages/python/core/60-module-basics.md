---
title: 模块基础
---

# 模块基础

当你写了一段代码，下次想用的时候发现得复制粘贴过去。有没有办法把代码保存起来，以后直接引用？模块就是来解决这个问题的。

在 Python 中，一个 `.py` 文件就是一个模块。你可以把自己写的函数、类、常量放到一个文件里，然后在其他文件中导入使用。这就是模块最朴素的理解——模块就是代码复用的一种方式。

但模块的意义远不止"保存代码"这么简单。模块是 Python 组织代码的基本单位，每一个 `.py` 文件在被导入时都会执行。模块和脚本的区别在于：脚本是直接运行的程序，模块是被其他程序导入的代码。这个区别看似简单，却影响了很多行为。

理解模块是理解 Python 项目结构、包管理、以及大型程序组织方式的基础。不管你是写小脚本还是大型项目，模块都是必须掌握的概念。

## 什么是模块

模块就是一个包含 Python 代码的 `.py` 文件。文件名叫 `math_utils.py`，它就是一个模块，模块名是 `math_utils`。

```python
# math_utils.py
def add(a, b):
    return a + b

PI = 3.14159
```

这个文件保存后，就可以在其他文件中使用了：

```python
import math_utils

result = math_utils.add(1, 2)
print(math_utils.PI)
```

模块可以包含：函数、类、变量、甚至直接放可执行代码。

当一个 `.py` 文件被导入时，Python 会执行这个文件里的所有代码，然后把这个模块对象缓存到 `sys.modules` 中。下次再导入时，直接从缓存里取，不会重新执行。

## 模块 vs 脚本

同一个文件，既可以作为模块被导入，也可以作为脚本直接运行。区别在于 `__name__` 变量的值。

```python
# example.py

def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
```

当直接运行 `python example.py` 时，`__name__` 的值是 `"__main__"`。当作为模块导入时，`__name__` 的值是模块名。

```python
# 导入时
import example
print(example.__name__)  # example
```

这个机制让你可以区分"直接运行"和"被导入"两种情况。很多库的代码里会有一个 `if __name__ == "__main__":` 块，用来放测试代码或者命令行入口。

## import 机制的工作原理

当你执行 `import math` 时，Python 内部经历了一系列步骤：

1. 首先在 `sys.modules` 中查找是否已经有这个模块。`sys.modules` 是一个字典，缓存了所有已经导入过的模块。如果找到，直接返回缓存的模块对象。

2. 如果没找到，在 `sys.path` 指定的路径中搜索 `math.py` 或 `math` 包。`sys.path` 是一个列表，包含搜索路径。

3. 找到文件后，加载并执行模块代码，创建模块对象。

4. 把模块对象存入 `sys.modules`，并绑定到当前命名空间。

```python
import sys

# 查看搜索路径
print(sys.path)
```

`sys.path` 默认包含：当前目录、Python 安装目录、标准库目录、以及 `PYTHONPATH` 环境变量指定的目录。

## 模块的搜索路径

`sys.path` 的顺序很重要。Python 会按照这个顺序依次查找模块，找到第一个匹配的就会停下来。

```python
import sys
print(sys.path)
```

输出类似：

```
['', '/usr/local/lib/python3.9/site-packages', '/usr/lib/python3.9', ...]
```

第一个空字符串 `''` 表示当前目录，优先搜索。如果当前目录有 `math.py`，会覆盖标准库的 `math` 模块。这有时候会导致意外行为，需要注意。

可以通过修改 `sys.path` 来添加自定义搜索路径：

```python
import sys
sys.path.insert(0, '/my/custom/path')
```

但这种做法不推荐，因为会影响全局。更好的做法是用虚拟环境或者设置 `PYTHONPATH` 环境变量。

## 常用标准库模块

Python 的一大优势是"batteries included"——内置了大量标准库模块。

`os` 模块提供操作系统相关功能：

```python
import os

os.getcwd()           # 获取当前工作目录
os.listdir('.')       # 列出当前目录文件
os.mkdir('new_dir')   # 创建目录
os.path.join(a, b)    # 路径拼接（跨平台）
```

`sys` 模块提供 Python 解释器相关功能：

```python
import sys

sys.version           # Python 版本
sys.path              # 模块搜索路径
sys.argv              # 命令行参数
sys.exit()            # 退出程序
```

`math` 模块提供数学函数：

```python
import math

math.sqrt(16)        # 平方根
math.sin(math.pi/2)   # 三角函数
math.log(math.e)      # 对数
```

`random` 模块提供随机数生成：

```python
import random

random.randint(1, 10)     # 随机整数
random.choice(['a','b'])  # 随机选择
random.shuffle([1,2,3])  # 随机洗牌
```

`datetime` 模块处理日期和时间：

```python
import datetime

datetime.datetime.now()           # 当前时间
datetime.date.today()            # 今天日期
datetime.timedelta(days=7)       # 时间差
```

## 模块的命名空间

每个模块都有自己独立的命名空间。当导入一个模块时，模块里的所有名字都绑定到这个模块对象上。

```python
import math

# math 是模块对象的引用
print(math.sqrt)     # <function sqrt>
print(math.pi)       # 3.141592653589793
```

这意味着你导入的模块名本身也是一个对象，可以赋值给其他变量：

```python
import math as m  # 给模块起别名

print(m.sqrt(16))   # 4.0
```

模块的命名空间是独立的，不同模块之间不会互相干扰。即使两个模块都定义了一个叫 `PI` 的变量，它们也是不同的。

```python
# module_a.py
PI = 3.14

# module_b.py
PI = 3.14159

# main.py
import module_a
import module_b

print(module_a.PI)  # 3.14
print(module_b.PI)   # 3.14159
```

## 模块的缓存机制

模块只会执行一次，之后的导入都从 `sys.modules` 缓存中获取。这带来一个重要后果：如果你修改了模块文件，需要重启 Python 或者重新加载模块才能看到变化。

```python
import math
import sys

print(sys.modules['math'])  # 显示模块对象
```

缓存机制也意味着循环导入时要小心：

```python
# a.py
import b

# b.py
import a
```

这会导致问题：当导入 `a` 时，开始导入 `b`，而 `b` 又导入 `a`，但此时 `a` 还没导入完成，会出现引用问题。

## 常见误区

第一个误区是忘记模块的搜索顺序。如果当前目录有个 `math.py`，它会覆盖标准库的 `math`。有时候代码在自己的机器上能跑，换个环境就不行了，很可能就是因为路径问题。

第二个误区是循环导入。当两个模块互相导入时，如果导入顺序不对，可能导致变量还没定义就用。解决方法包括：把导入移到函数内部、使用延迟导入、重新设计模块结构。

第三个误区是修改 `sys.path` 而不是使用虚拟环境。在开发库或者多项目时，应该用虚拟环境隔离不同的依赖，而不是修改 `sys.path`。

理解模块基础是理解 Python 项目结构的起点。模块让代码得以复用，让大型项目得以组织，让协作成为可能。
