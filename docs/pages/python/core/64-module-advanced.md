---
title: 模块的高级特性
---

# 模块的高级特性

这一章讲一些模块的高级话题，包括 `__init__.py` 的高级用法、`importlib` 动态导入、模块的元属性、以及模块缓存与重新加载。

理解这些高级特性对于理解 Python 框架的内部机制很重要。很多框架和库都会用到这些高级特性，比如 Flask 的路由装饰器、Django 的信号机制，都和模块的元属性有关。

## __init__.py 的高级用法

`__init__.py` 不只是初始化文件，它还能控制包的导入行为。

可以在这里导入子模块，让包级别的导入更简洁：

```python
# my_package/__init__.py
from .module1 import Class1
from .module2 import function2

# 现在可以这样用：
# import my_package
# obj = my_package.Class1()
```

这叫做"显式子模块导入"，让包的使用者可以用更简洁的方式使用包。

也可以设置默认导出：

```python
# my_package/__init__.py
from .core import CoreClass

__all__ = ['CoreClass']
```

这样 `from my_package import *` 会导入 `CoreClass`。

还可以用 `__init__.py` 来做包的配置：

```python
# my_package/__init__.py
import os

DEBUG = os.environ.get('DEBUG', False)

if DEBUG:
    # 调试模式配置
    pass
```

## importlib 动态导入

有时候需要在运行时动态决定导入哪个模块，而不是写死 `import` 语句。`importlib` 提供了这个能力。

```python
import importlib

# 动态导入模块
module_name = 'math'
math = importlib.import_module(module_name)
print(math.sqrt(16))  # 4.0
```

还可以动态导入子模块：

```python
# 相当于 import os.path
path_module = importlib.import_module('os.path')
```

动态导入在插件系统、配置驱动的模块加载等场景下很有用。

比如一个插件系统，根据配置文件加载不同的插件：

```python
import importlib

plugins = ['plugin_a', 'plugin_b']

for plugin_name in plugins:
    plugin_module = importlib.import_module(f'plugins.{plugin_name}')
    plugin = plugin_module.Plugin()
    plugin.run()
```

`importlib.reload` 可以重新加载模块，这在开发时修改模块后不需要重启 Python：

```python
import importlib
import my_module

# 修改 my_module.py 后
importlib.reload(my_module)
```

## 模块的元属性

每个模块都有一组元属性，记录模块的信息。

`__name__` 是模块名：

```python
# 在 my_module.py 中
print(__name__)  # my_module

# 如果直接运行
# python my_module.py
# 输出 __main__
```

`__file__` 是模块文件路径：

```python
# my_module.py
print(__file__)  # /path/to/my_module.py
```

`__doc__` 是文档字符串：

```python
"""
这是 my_module 模块的文档
"""
print(__doc__)  # 这是 my_module 模块的文档
```

`__spec__` 是模块的详细规范：

```python
import importlib.machinery

print(__spec__)  # ModuleSpec(name='__main__', ...)
```

`__loader__` 是加载模块的加载器：

```python
print(__loader__)  # <class '_frozen_importlib.SourceFileLoader'>
```

`__package__` 是模块所属的包：

```python
# 在 my_package/sub_module.py 中
print(__package__)  # my_package
```

这些元属性在框架和工具开发中经常用到。比如 Django 的 `INSTALLED_APPS` 配置就是读取包的 `__file__` 属性来定位应用目录。

## 模块缓存与重新加载

模块被导入后会缓存在 `sys.modules` 中：

```python
import sys

import math
print('math' in sys.modules)  # True
print(sys.modules['math'])  # 模块对象
```

再次导入同一个模块会直接从缓存返回，不会重新执行模块代码：

```python
import math
import math  # 不会重新执行，直接从缓存返回
```

如果修改了模块源码，需要重新加载才能生效：

```python
import importlib
import my_module

# ... 修改 my_module.py ...

importlib.reload(my_module)  # 重新加载模块
```

注意：`reload` 只重新加载指定的模块，不会自动重新加载它依赖的模块。如果被修改的模块依赖其他模块，那些依赖模块不会被重新加载。

`reload` 在开发时有用，但生产环境应该用 `-e` 安装或者重启进程。

## 模块的线程安全性

Python 有 GIL（全局解释器锁），同一个时刻只有一个线程执行 Python 字节码。但模块的导入不是完全线程安全的。

在多线程环境中，如果多个线程同时导入同一个模块，可能会有竞态条件。不过 Python 的导入机制内部有锁，所以正常情况下不会有严重问题。

但如果模块的初始化代码有副作用（比如修改全局变量、打开文件、网络请求），在多线程环境中可能会有问题。

```python
# my_module.py
import threading

connection = None

def init():
    global connection
    connection = create_connection()

# 如果两个线程同时调用 init
# 可能会创建两个连接
```

正确的做法是在模块级别做初始化，或者使用线程同步机制。

## 模块与单例模式

模块本身就是一种单例。每个模块在进程中只会被导入一次，所有地方引用的是同一个模块对象。

```python
# singleton_module.py
class _Singleton:
    def __init__(self):
        self.value = None

_instance = None

def get_instance():
    global _instance
    if _instance is None:
        _instance = _Singleton()
    return _instance
```

```python
# 使用
from singleton_module import get_instance

obj1 = get_instance()
obj2 = get_instance()
print(obj1 is obj2)  # True，同一个对象
```

模块的全局变量在整个模块生命周期内存在，这让它天然适合做单例模式的实现方式。

## 常见误区

第一个误区是在 `__init__.py` 放太多代码。`__init__.py` 应该只做必要的初始化，比如导入子模块、设置包级别变量。复杂的初始化逻辑应该放在单独的模块中。

第二个误区是动态导入时不做错误处理。`importlib.import_module` 如果模块不存在会抛出 `ModuleNotFoundError`，应该捕获并处理。

第三个误区是在生产环境使用 `reload`。`reload` 主要用于开发调试，生产环境应该正确安装包并重启进程。

第四个误区是假设模块初始化是线程安全的。模块级别的代码在第一次导入时会执行，如果涉及共享资源的修改，需要加锁保护。

理解模块的高级特性，能让你更好地理解 Python 的内部机制，也能写出更健壮的代码。
