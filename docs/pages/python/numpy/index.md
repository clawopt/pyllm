# NumPy 教程

NumPy 是 Python 科学计算的基础库，提供了高性能的多维数组对象和相关工具。

## 安装

```bash
pip install numpy
```

## 创建数组

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
identity = np.eye(3)
random_arr = np.random.rand(3, 3)
range_arr = np.arange(0, 10, 2)
```

## 数学运算

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)     # [5, 7, 9]
print(a - b)     # [-3, -3, -3]
print(a * b)     # [4, 10, 18]
print(np.sqrt(a))    # 平方根
print(np.mean(a))     # 平均值
```

## 下一步

继续学习：
- [Pandas教程](/pages/python/pandas/)
- [Matplotlib教程](/pages/python/matplotlib/)
