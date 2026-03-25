---
title: 模块的导入
---

# 模块的导入

上一章讲了模块是什么，这一章来深入讲模块的导入方式。import 看似简单，只有 `import` 和 `from ... import` 两种基本形式，但背后有很多细节值得注意。

选择正确的导入方式、避免循环导入、理解导入的搜索路径，这些问题在实际项目中经常会遇到。尤其是做大模型开发时，经常要管理复杂的依赖和包结构，理解导入机制尤为重要。

## import vs from ... import

`import` 和 `from ... import` 都能导入模块，但效果不同。

`import math` 导入整个模块，使用时要加模块名前缀：

```python
import math

print(math.sqrt(16))  # 4.0
```

`from math import sqrt` 只导入特定的函数或变量，可以直接使用不需要前缀：

```python
from math import sqrt

print(sqrt(16))  # 4.0
```

两者的区别在于命名空间的影响范围。使用 `import math` 时，模块的所有名字都绑定在 `math` 这个命名空间下，不会和当前命名空间冲突。使用 `from math import sqrt` 时，`sqrt` 被直接导入当前命名空间，如果当前已经有 `sqrt` 这个名字，会覆盖它。

## import as 别名

可以用 `as` 给模块或函数起别名：

```python
import numpy as np
import tensorflow as tf

np.array([1, 2, 3])
tf.constant([1, 2, 3])
```

起别名有两个主要用途：一是解决命名冲突，比如两个模块都有 `utils` 函数；二是简化长名字，比如 `matplotlib.pyplot` 通常起别名 `plt`。

```python
from math import sqrt as math_sqrt
from my_math import sqrt as my_sqrt
```

## from ... import * 的使用

`from module import *` 导入模块的所有公开对象：

```python
# math_module.py
PI = 3.14
E = 2.718

def add(a, b):
    return a + b

__all__ = ['PI', 'add']  # 只导出这两个
```

```python
from math_module import *

print(PI)        # 3.14
print(add(1, 2))  # 3
print(E)          # 2.718，除非在 __all__ 中，否则不会导入
```

`__all__` 列表定义了 `import *` 时会导出哪些名字。如果模块没有定义 `__all__`，`import *` 会导入所有不以 `_` 开头的名字。

虽然 `import *` 写起来方便，但有几个问题：不清楚导入了哪些名字、可能覆盖当前命名空间的变量、在代码静态分析时无法追踪变量来源。所以大部分 Python 代码规范都建议不使用 `import *`。

## 循环导入问题

循环导入是 Python 项目中常见的问题。当两个模块互相导入时，可能会遇到问题：

```python
# a.py
import b

def func_a():
    return "from a"

result = b.func_b()
print(result)
```

```python
# b.py
import a

def func_b():
    return "from b"

result = a.func_a()
print(result)
```

运行 `python a.py` 会输出：

```
from b
from a
```

看起来正常，但如果调换顺序呢：

```python
# a.py
import b

def func_a():
    return "from a"

result = b.func_b()  # b.func_b 在这里调用
```

```python
# b.py
import a

result = a.func_a()  # 在这里调用

def func_b():
    return "from b"
```

运行 `python a.py` 会报错，因为 `b.py` 在导入时就会调用 `a.func_a()`，而 `a` 模块此时还没执行完，`func_a` 还不存在。

解决循环导入的方法有几种。

第一种是把导入移到函数内部：

```python
# b.py

def func_b():
    return "from b"

def use_a():
    import a
    return a.func_a()
```

第二种是重新设计模块结构，把共同依赖提取到第三个模块。

第三种是使用延迟导入，在需要时才真正导入。

## 导入搜索顺序

前面讲了 `sys.path` 控制搜索路径。但实际上搜索顺序更复杂一点：

1. 内置模块（如 `sys`、`os`、`math`）
2. `sys.path` 中的目录，按顺序搜索

`sys.path` 的默认值是：

```python
import sys
print(sys.path)
```

输出类似：

```
['', '/home/user/project', '/usr/local/lib/python3.9/site-packages', ...]
```

第一个空字符串 `''` 表示当前工作目录，优先搜索。如果当前目录有 `test.py` 文件，`import test` 会导入它而不是 Python 内置的 `test` 模块（如果有的话）。

可以通过修改 `PYTHONPATH` 环境变量来添加搜索路径：

```bash
export PYTHONPATH=/path/to/your/module:$PYTHONPATH
```

## 延迟导入

延迟导入是把 `import` 语句放到函数内部，而不是模块顶部。这有几个好处：加快启动速度（不立即加载所有依赖）、避免循环导入、在需要时才加载。

```python
def process_data():
    # 只有调用这个函数时才会导入 numpy
    import numpy as np
    return np.array([1, 2, 3])
```

但延迟导入也有代价：每次调用函数都要检查导入，增加一点开销。而且在调试时，导入错误可能不在模块加载时暴露，而在运行时才暴露。

Flask 等框架就大量使用了延迟导入，主模块只加载最基本的依赖，具体功能在请求处理时才导入。

## 导入与变量作用域

导入语句在模块级别执行，但导入的名字遵循 LEGB 作用域规则。

```python
# a.py
x = 1

def foo():
    print(x)  # 可以访问模块级别的 x
```

导入和作用域是两套独立的机制。`import` 把模块对象绑定到当前命名空间，`LEGB` 规则控制变量的查找顺序。

```python
import math

def foo():
    print(math)  # 可以访问模块级别导入的 math
```

如果在函数内部导入，模块对象只在函数内部可见：

```python
def foo():
    import math
    print(math)  # 只在函数内部可见

# print(math)  # 这里会报错，math 不在模块级别命名空间
```

## 常见误区

第一个误区是在循环导入时尝试"先定义再导入"。Python 的模块加载是整体执行的，在模块顶部写 `import` 语句时，整个模块代码都会执行。所以如果两个模块互相在顶部导入，一定会出问题。

第二个误区是 `import *` 导致的名字污染。如果当前模块已经有一个叫 `sqrt` 的函数，用 `from math import *` 会覆盖它。这在大型项目中尤其容易引发难以调试的 bug。

第三个误区是不理解导入的实际执行。`import` 语句不只是声明，它是可执行代码，会运行被导入模块的全部代码。理解这一点对于调试模块加载问题至关重要。

第四个误区是在循环导入时把问题归咎于 Python 而不是代码设计。循环导入通常是代码结构问题的信号，应该重新设计模块的依赖关系，而不是试图用技巧绕过。
