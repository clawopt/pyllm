---
title: 安装 Pandas
description: pip 安装、版本检查、验证安装成功，以及常见问题排查
---
# 安装 Pandas

在开始写代码之前，你需要先把 Pandas 装到你的电脑上。好消息是——**安装非常简单**。

## 前置要求

Pandas 是一个 Python 库，所以你首先需要：

1. **已安装 Python**（推荐 3.10 或更高版本）
2. **有一个终端/命令行工具**

检查 Python 是否可用：

```bash
python --version
```

如果看到类似 `Python 3.x.x` 的输出，就说明 Python 已经装好了。如果没有，请先去 [python.org](https://python.org) 下载并安装。

## 安装 Pandas

打开终端，运行一条命令就够了：

```bash
pip install "pandas[all]"
```

这条命令做了三件事：
- **安装 pandas 本体**
- `[all]` 表示同时安装常用的依赖库（如读写 Excel、SQL 数据库的扩展包）

如果你想指定版本（比如最新稳定版）：

```bash
pip install "pandas>=2.0"
```

> **注意**：本教程基于 Pandas 2.x / 3.x 编写。如果你用的是更早的版本，部分 API 可能略有不同。

## 验证安装

安装完成后，用以下命令确认一切正常：

```bash
python -c "import pandas as pd; print(f'Pandas {pd.__version__} 安装成功!')"
```

如果看到 `Pandas x.x.x 安装成功!` 的输出，恭喜你，环境已经就绪了！

## 如果遇到问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `command not found: pip` | Python 没加到 PATH | 用 `python -m pip install ...` 代替 |
| `Permission denied` | 没有写入权限 | 加 `--user` 参数或用虚拟环境 |
| 下载速度慢 | 默认源在国外 | 换国内镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas` |
| 版本冲突 | 已有旧版 Pandas | 先 `pip uninstall pandas` 再重新安装 |

## 推荐做法：使用虚拟环境

虽然不是必须的，但强烈建议你在学习时使用**虚拟环境**（virtual environment）。它能为每个项目创建独立的 Python 包环境，避免不同项目之间的依赖冲突：

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Mac/Linux）
source .venv/bin/activate

# 激活后终端提示符前面会显示 (.venv)，说明已激活
# 现在可以正常 pip install 了
```

以后每次打开新终端做练习时，先执行 `source .venv/bin/activate` 即可。
