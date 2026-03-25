---
title: 模块的安装与发布
---

# 模块的安装与发布

写了一个好用的模块，想分享给同事或者发布到 PyPI？这一章讲怎么把代码变成可安装、可分发的包。

Python 的包分发系统经历了从 `setup.py` 到 `pyproject.toml` 的演进。不管用哪种方式，核心问题都一样：怎么让别人能 `pip install` 你的包，怎么声明依赖，怎么指定入口点。

理解包的分发机制，对于使用第三方库、管理项目依赖、甚至只是理解"为什么这个库要这样安装"都很有帮助。

## __main__.py 入口点

如果一个包可以直接运行，需要一个 `__main__.py` 文件：

```
my_package/
    __init__.py
    __main__.py
    module.py
```

```python
# my_package/__main__.py
from .module import main

if __name__ == "__main__":
    main()
```

现在可以这样运行：

```bash
python -m my_package
```

`python -m package_name` 会自动运行包中的 `__main__.py`。

`__main__.py` 让你的包既可以作为模块导入，又可以直接运行。很多 CLI 工具都是这样实现的。

## pip 与包管理

`pip` 是 Python 的包管理器，用来安装和卸载 Python 包。

```bash
pip install requests
pip uninstall requests
pip list  # 列出已安装的包
pip freeze > requirements.txt  # 导出依赖
pip install -r requirements.txt  # 从文件安装
```

`pip install` 的包安装到哪里去了？

```python
import site
print(site.getusersitepackages())  # 用户 site-packages 目录
print(site.getsitepackages())  # 全局 site-packages 目录
```

用户 site-packages 优先于全局 site-packages，这允许用户安装自己版本的包来覆盖系统包。

## requirements.txt 依赖管理

`requirements.txt` 是最简单也是最常用的依赖管理方式。

```txt
requests>=2.25.0
numpy~=1.20.0
pandas
```

常用标记：
- `>=` 最小版本
- `~=1.20.0` 兼容版本（>=1.20.0 且 <1.21.0）
- `==1.20.0` 精确版本
- `>=1.0,<2.0` 版本范围

```bash
pip install -r requirements.txt
```

`pip freeze` 可以导出当前环境的精确依赖：

```bash
pip freeze > requirements.txt
```

注意：`pip freeze` 会导出所有安装的包，包括传递依赖，通常需要手动清理。

## setup.py 的使用

`setup.py` 是传统的包分发配置方式：

```python
from setuptools import setup, find_packages

setup(
    name='mypackage',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'requests>=2.25.0',
    ],
    entry_points={
        'console_scripts': [
            'mytool=my_package.cli:main',
        ],
    },
)
```

现在可以用以下方式安装：

```bash
python setup.py install  # 安装
pip install .  # 推荐方式
pip install -e .  # 开发模式安装（可编辑）
```

`-e` 开发模式安装会创建一个链接到源码目录，而不是复制文件，这样修改代码立即生效。

## pyproject.toml

Python 3.11+ 和 pip 22.3+ 推荐使用 `pyproject.toml` 作为包配置：

```toml
[project]
name = "mypackage"
version = "0.1.0"
dependencies = [
    "requests>=2.25.0",
]

[project.scripts]
mytool = "my_package.cli:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

`pyproject.toml` 是 PEP 518 引入的标准，目标是统一 Python 项目的配置格式。

安装方式不变：

```bash
pip install .
pip install -e .
```

## 虚拟环境

虚拟环境是 Python 开发的标准实践。它创建一个独立的 Python 环境，有自己的 site-packages，不影响系统全局环境。

创建虚拟环境：

```bash
python -m venv myenv
```

激活虚拟环境：

```bash
# Linux/macOS
source myenv/bin/activate

# Windows
myenv\Scripts\activate
```

激活后，`python` 和 `pip` 都指向虚拟环境中的版本。安装的包只在这个环境中可见。

离开虚拟环境：

```bash
deactivate
```

使用虚拟环境的好处是：不同项目可以依赖不同版本的同一个包，互不干扰。比如项目 A 需要 Django 2，项目 B 需要 Django 4，用虚拟环境可以同时维护两个环境。

## 发布到 PyPI

把自己的包发布到 PyPI，让全世界的人都能用 `pip install` 安装。

首先，打包：

```bash
pip install build
python -m build
```

这会在 `dist/` 目录下生成 `.whl`（wheel）和 `.tar.gz`（source tarball）文件。

然后，发布到 Test PyPI（测试用）：

```bash
pip install twine
twine upload --repository testpypi dist/*
```

最后，正式发布到 PyPI：

```bash
twine upload dist/*
```

发布前需要到 pypi.org 注册账号，并创建 API token。

## 本地安装与编辑模式

本地安装用于开发和测试：

```bash
pip install .  # 安装当前目录的包
pip install -e .  # 编辑模式安装（开发用）
pip install ../other-package  # 安装其他目录的包
```

编辑模式 `-e` 创建一个指向源码的链接，修改代码后不需要重新安装。在开发自己的包时，应该总是用 `-e` 安装，这样改动立即生效。

## 常见误区

第一个误区是不使用虚拟环境。在全局环境中开发，多个项目依赖不同版本的包会互相冲突。正确做法是每个项目用独立的虚拟环境。

第二个误区是发布未测试的包。上传 PyPI 前应该先在 Test PyPI 测试，确认打包和安装都没问题。

第三个误区是 `pip install` 不知道安装到哪里去了。可以用 `pip show <package>` 查看包的信息和位置。

第四个误区是不指定依赖版本。发布包时应该明确声明依赖版本范围，否则在不同环境下安装可能得到不同结果，导致不可预期的行为。

理解包的分发和安装机制，是成为 Python 开发者的必备技能。不管是使用第三方库还是发布自己的代码，都离不开这套机制。
