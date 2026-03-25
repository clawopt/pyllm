---
title: 模块的组织
---

# 模块的组织

当你写了十几个模块，每个模块有上百个函数时，怎么组织这些文件就成了问题。这就是包（package）要解决的问题。

包是一种组织模块的方式，它把多个模块打包成一个层次结构。你可以把它理解成"文件夹"，包就是"文件夹"，模块就是"文件"。包可以嵌套，形成树状的层次结构。

理解包的概念是理解大型 Python 项目结构的基础。几乎所有 Python 库都是用包来组织的。理解包的结构，才能理解第三方库是怎么组织的，才能正确安装和使用它们。

## 什么是包

包就是一个包含 `__init__.py` 文件的目录。这个目录本身就是一个包，目录里的 `.py` 文件就是模块。

```
my_package/
    __init__.py
    module1.py
    module2.py
```

这个结构表示：`my_package` 是一个包，里面有两个模块 `module1` 和 `module2`。

导入方式：

```python
import my_package.module1
from my_package import module2
```

包可以嵌套：

```
parent_package/
    __init__.py
    sub_package/
        __init__.py
        module.py
```

嵌套包的导入：

```python
import parent_package.sub_package.module
from parent_package.sub_package import module
```

## __init__.py 的作用

`__init__.py` 是包初始化文件。当包被导入时，这个文件会自动执行。

```python
# my_package/__init__.py
print("my_package is imported!")

# 其他初始化代码
```

运行 `import my_package` 会打印出那句话。

`__init__.py` 有什么用？它可以设置包级别的导入控制。可以在这里定义 `__all__` 列表，控制 `from package import *` 会导入哪些模块：

```python
# my_package/__init__.py
__all__ = ['module1', 'module2']
```

它也可以简化包级别的导入：

```python
# my_package/__init__.py
from .module1 import SomeClass
from .module2 import some_function

# 现在可以直接 import my_package，然后用 my_package.SomeClass
```

这个特性让包的对外接口更加清晰。包的内部结构可以很复杂，但对外只暴露必要的部分。

## 包 vs 文件夹

Python 的包和普通文件夹有什么区别？关键就是 `__init__.py` 文件。

Python 3.3 之前，必须有 `__init__.py` 文件才能被当作包。之后引入了 "namespace packages" 概念，可以没有 `__init__.py`，但这种包不支持相对导入，用处有限。

对于大多数场景，还是应该用传统的包结构，包含 `__init__.py`。

普通文件夹没有 `__init__.py`，不能被当作包导入：

```python
# 普通文件夹
my_folder/
    module.py

import my_folder.module  # 报错，my_folder 不是包
```

## __all__ 导出控制

`__all__` 列表控制 `from package import *` 会导出哪些名字。

```python
# my_package/__init__.py
__all__ = ['PublicClass', 'public_function']

class PublicClass:
    pass

class PrivateClass:
    pass

def public_function():
    pass

def _private_function():
    pass
```

```python
from my_package import *

PublicClass()    # OK
public_function()  # OK
PrivateClass()    # 报错，不在 __all__ 中
_private_function()  # 不在 __all__ 中，不会导入
```

如果模块没有定义 `__all__`，`from module import *` 会导入所有不以 `_` 开头的名字。所以私有函数（以 `_` 开头）天然不会被 `import *` 导入。

## 相对导入 vs 绝对导入

导入有两种方式：相对导入和绝对导入。

绝对导入使用完整的模块路径：

```python
from my_package.module1 import SomeClass
import my_package.module2
```

相对导入使用包内的相对路径，用 `.` 表示当前目录，`..` 表示上级目录：

```python
# 在 my_package/sub/module.py 中
from . import sibling_module  # 导入同级的 sibling_module
from .. import parent_module  # 导入上级的 parent_module
from .module1 import SomeClass  # 导入当前包中的 module1
```

相对导入只能用于包内部，不能在脚本中直接运行：

```python
# my_package/module.py
from . import utils  # OK，在包中

# 如果直接 python module.py 运行，会报错
```

这是因为相对导入依赖于包的结构，而直接运行脚本时，脚本不在包结构中。

## 子包与嵌套结构

包可以无限嵌套，形成树状结构：

```
project/
    __init__.py
    core/
        __init__.py
        engine.py
        utils.py
    api/
        __init__.py
        routes.py
        middleware.py
    models/
        __init__.py
        user.py
        post.py
```

导入方式：

```python
from project.core import engine
from project.api.routes import router
import project.models.user as UserModel
```

包的层次结构应该反映功能的层次结构。相关的模块放在一起，不相关的模块分开。

常见的项目结构：

```
myproject/
    __init__.py
    main.py
    config.py
    models/
    views/
    controllers/
    utils/
    tests/
```

## 包的结构最佳实践

一个好的包结构应该清晰、可扩展、易于维护。

首先，包的深度不要太深，一般不超过 3 层。 太深的层次会让导入语句变得很长：`from a.b.c.d.e import something`。

其次，每个包应该有明确的职责。一个包应该只包含相关的模块，而不是把所有东西都堆在一起。

第三，使用 `__init__.py` 来控制包的对外接口。包的内部实现可以很复杂，但对外应该简单明了。

第四，把 `__main__.py` 放在包的最外层，让包可以直接运行：

```
my_package/
    __init__.py
    __main__.py
    module.py
```

```bash
python -m my_package  # 会运行 __main__.py
```

## 常见误区

第一个误区是不理解相对导入的限制。相对导入只能在包内部使用，不能在直接运行的脚本中使用。如果需要在脚本中使用相对导入的模块，应该用 `python -m` 来运行：

```bash
python -m my_package.module  # 正确
python my_package/module.py  # 错误
```

第二个误区是混淆包名和模块名。`import json` 导入的是标准库的 json 模块，不是 json 包（如果存在的话）。

第三个误区是 `__init__.py` 放太多代码。`__init__.py` 应该只包含必要的初始化代码，不应该放太多业务逻辑。复杂的初始化应该放在单独的模块中。

第四个误区是创建太多嵌套层次。层次太深的包结构会让导入语句变得很长，也会让项目难以维护。

理解包的组织方式，是理解 Python 项目结构的基础。好的包结构让代码易于维护、扩展和共享。
