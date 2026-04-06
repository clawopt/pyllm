---
title: 开发环境准备
description: Python 版本要求、虚拟环境配置、IDE 推荐、第一个 Python 程序的运行
---
# 开发环境准备

在安装 LangChain 之前，你需要先确保基础开发环境就绪。这一节我们用最少的步骤把环境搭好——不追求完美配置，只求"能跑起来"。

## 第一步：确认 Python 版本

LangChain 对 Python 版本有明确要求：**3.10 或更高**。这是因为 LangChain v1.0 使用了一些较新的 Python 语法特性（比如 `type X | Y` 这样的类型注写法），低版本不支持。

打开终端，输入：

```bash
python --version
```

你应该看到类似这样的输出：

```
Python 3.11.x 或更高
```

如果看到的是 `Python 3.8.x`、`3.9.x` 或者 `2.7.x`——那就需要升级了。最简单的方式是去 [python.org](https://www.python.org/downloads/) 下载最新版安装包。如果你用的是 macOS，系统其实已经自带了 Python 3.x（可以在终端输入 `python3 --version` 验证）。

> **为什么是 3.10？** 因为 Python 3.10 引入了 `match/case` 模式匹配和更完善的类型提示（Type Hints）语法，这些在 LangChain 的内部代码和现代 Python 最佳实践中被广泛使用。虽然你的代码不一定直接用到这些特性，但保持环境版本一致能避免各种隐晦的兼容性问题。

## 第二步：创建一个虚拟环境

这一步不是必须的，但**强烈推荐**。虚拟环境的作用是为每个项目创建独立的"包沙盒"——你在这个项目里装的库不会影响其他项目，也不会被其他项目的依赖版本冲突。

```bash
# 在项目目录下创建虚拟环境
python -m venv .venv

# 激活它（每次打开新终端都需要执行这行）
source .venv/bin/activate
```

激活成功后，你会注意到终端提示符前面多了一个 `(.venv)` 前缀——这说明当前所有 `pip install` 和 `python` 命令都会在这个隔离环境中执行。

如果你好奇虚拟环境到底做了什么：它本质上就是在 `.venv/` 目录下创建了一份 Python 解释器的副本和一个独立的 `site-packages/` 目录。后续所有通过 `pip` 安装的包都会放到这里，完全不影响系统全局的 Python 环境。

## 第三步：选一个编辑器

写 Python 代码需要一个编辑器。目前最主流的选择：

| 编辑器 | 特点 | 适合谁 |
|--------|------|---------|
| **VS Code** | 免费、插件丰富、内置终端 | 绝大多数开发者 |
| **PyCharm** | 专业 IDE、智能补全强 | 重度 Python 用户 |
| **Jupyter Notebook** | 交互式、单元格执行 | 数据探索和实验 |

对于学习 LangChain 来说，**VS Code + Jupyter 扩展** 是最省心的组合——既能写 `.py` 脚本文件，也能直接跑 `.ipynb` notebook 做交互式实验。

### VS Code 基础配置

1. 安装 VS Code（[code.visualstudio.com](https://code.visualstudio.com)）
2. 按 `Cmd + Shift + X` 打开扩展商店
3. 搜索并安装以下扩展：
   - **Python**（微软官方）
   - **Jupyter**（微软官方）

安装完成后，新建一个 `test_env.py` 文件验证一下：

```python
import sys

print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print("✅ 环境就绪！")
```

然后在终端里运行：

```bash
python test_env.py
```

如果看到类似下面的输出，说明一切正常：

```
Python 版本: 3.11.x (main, ...)
Python 路径: /Users/you/project/.venv/bin/python
✅ 环境就绪！
```

## 常见问题排查

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `command not found: python` | macOS 区分 python/python3 | 用 `python3` 替代，或加 alias |
| `No module named venv` | Python 太旧（<3.3） | 升级 Python |
| 激活后仍用全局 pip | PATH 未正确更新 | 关闭终端重新打开，或用绝对路径 `.venv/bin/pip` |
| VS Code 无法识别虚拟环境 | 未选择正确的解释器 | `Cmd+Shift+P` → 输入 `Python: Select Interpreter` → 选 `.venv` 下的那个 |

环境搭好了之后，下一节我们就要正式安装 LangChain 和它的核心依赖。
