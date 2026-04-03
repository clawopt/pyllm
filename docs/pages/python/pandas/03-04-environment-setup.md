---
title: 开发环境推荐
description: VS Code + Jupyter 的基础配置、第一个 Pandas 脚本
---
# 开发环境配置

环境配好了，Pandas 也装好了。接下来你需要一个**写代码和运行代码的地方**。

## 推荐组合：VS Code

VS Code 是目前最流行的 Python 编辑器，免费、轻量、插件丰富。对于学习 Pandas 来说，它有几个关键优势：

- **Jupyter 扩展**：可以在 VS Code 里直接运行 `.ipynb` 文件
- **智能补全**：输入 `pd.` 会自动提示所有可用的方法
- **语法高亮**：让代码更易读
- **内置终端**：不用切换窗口就能运行脚本

### 基础配置步骤

1. 安装 VS Code（从 [code.visualstudio.com](https://code.visualstudio.com)）
2. 打开 VS Code，按 `Cmd+Shift+X` 打开扩展商店
3. 搜索并安装 **Python** 和 **Jupyter** 两个官方扩展

### 第一个 Pandas 脚本

创建一个新文件 `hello_pandas.py`：

```python
import pandas as pd

# 创建一个简单的表格
data = {
    '模型': ['GPT-4o', 'Claude', 'Llama'],
    '评分': [88.7, 89.2, 84.5],
    '价格': [15, 15, 0],
}

df = pd.DataFrame(data)

# 打印看看
print("数据预览:")
print(df)

print(f"\n平均分: {df['评分'].mean():.1f}")
```

运行它：

```bash
python hello_pandas.py
```

如果看到类似这样的输出：

```
数据预览:
      模型    评分  价格
0   GPT-4o   88.7    15
1    Claude   89.2    15
2     Llama   84.5     0

平均分: 87.5
```

恭喜你——**你已经成功运行了第一行 Pandas 代码**！接下来的章节我们将系统学习 Series、DataFrame 以及各种数据处理操作。
