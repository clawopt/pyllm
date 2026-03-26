---
title: NumPy导入约定
---

# NumPy导入约定

这一章讲NumPy的标准导入方式。

## 标准导入

NumPy有且只有一个标准导入方式：

```python
import numpy as np
```

这行代码的含义是：导入numpy模块，并给它起个别名np。之后的代码中，用np来引用NumPy的所有函数。

这是整个Python科学计算社区的约定。遵守这个约定，让你的代码能被其他人轻松读懂——看到np.array你就知道这是NumPy的函数，看到torch.tensor你就能联想到PyTorch。

## 为什么要用np

不用别名可以吗？可以，但很麻烦：

```python
# 不用别名
import numpy
arr = numpy.array([1, 2, 3])
result = numpy.sum(arr)

# 用别名
import numpy as np
arr = np.array([1, 2, 3])
result = np.sum(arr)
```

用np写起来更简洁，而且一眼就能认出这是NumPy的函数。

## 常见错误

新手常犯的错误是导入方式不对：

```python
# 错误方式
from numpy import *
arr = array([1, 2, 3])  # 能工作，但不推荐

# 推荐方式
import numpy as np
arr = np.array([1, 2, 3])
```

`from numpy import *`会导入所有符号，可能覆盖内置函数，而且代码难以理解。你不知道arr是什么类型、来自哪里。

另一个常见错误是拼写错误：

```python
import numpy as np  # 正确

import numpy as mumpy  # 别名无所谓，但养成好习惯
```

## 子模块导入

NumPy有很多子模块，常见的有：

```python
import numpy as np

# 线性代数子模块
print(np.linalg.eigvals([[1, 2], [3, 4]]))

# 随机数子模块
print(np.random.randn(3))

# 也可以直接导入子模块
import numpy.linalg as npl
import numpy.random as npr
```

大多数情况下，直接用np.linalg、np.random就够了。

## 一行代码总结

学完这一章，你只需要记住一行代码：

```python
import numpy as np
```

这是你接下来所有NumPy代码的开始。
