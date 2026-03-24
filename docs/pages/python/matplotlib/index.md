# Matplotlib 教程

Matplotlib 是 Python 最流行的数据可视化库。

## 安装

```bash
pip install matplotlib
```

## 基本示例

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('正弦函数')
plt.show()
```

## 下一步

继续学习：
- [LangChain教程](/pages/llm/langchain/)
