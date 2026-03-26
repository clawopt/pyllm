---
title: NumPy安装与环境配置
---

# NumPy安装与环境配置

这一章讲如何安装NumPy和配置开发环境。

## 用pip安装

最简单的方式是用pip安装。打开终端，执行：

```bash
pip install numpy
```

如果想安装特定版本：

```bash
pip install numpy==1.24.3
```

升级到最新版本：

```bash
pip install --upgrade numpy
```

安装完成后，在Python里验证一下：

```python
import numpy as np
print(np.__version__)
```

如果输出了版本号比如`1.24.3`，说明安装成功了。

## 用Anaconda安装

如果你用Anaconda，NumPy通常已经预装了。可以检查一下：

```bash
conda list numpy
```

如果需要安装特定版本：

```bash
conda install numpy=1.24.3
```

Anaconda的好处是它自带了很多科学计算的库，而且会自动处理依赖关系。

## 虚拟环境

建议为不同项目创建独立的虚拟环境，这样不同项目用不同版本的NumPy也不会互相影响。

用venv创建虚拟环境：

```bash
# 创建
python -m venv myproject-env

# 激活
source myproject-env/bin/activate  # Linux或Mac
# myproject-env\Scripts\activate  # Windows

# 安装NumPy
pip install numpy

# 退出环境
deactivate
```

用conda创建：

```bash
conda create -n myproject python=3.10
conda activate myproject
conda install numpy
```

## Jupyter Notebook

Jupyter是科学计算的标准环境，非常适合NumPy开发。安装：

```bash
pip install jupyter notebook
```

启动：

```bash
jupyter notebook
```

浏览器会自动打开一个界面，你可以在里面写代码、运行、画图。做数据分析和算法原型时，Jupyter比命令行方便很多。

VS Code也支持Jupyter，方法是在VS Code里安装Python扩展，然后创建.ipynb文件即可。

## NumPy的依赖

NumPy底层依赖一些系统库，主要是BLAS（线性代数基础库）。大多数情况下，pip安装的wheel包已经包含了一切，不需要额外配置。

如果你是从源码编译安装，需要先安装编译工具链和BLAS开发库。这在树莓派等嵌入式设备上偶尔会遇到。

## 开发工具推荐

NumPy开发有几个常用工具：

IPython是增强的Python解释器，比普通解释器好用的多：

```bash
pip install ipython
```

运行：

```bash
ipython
```

IPython有自动补全、历史记录、魔法命令等功能。

pdb是Python的标准调试器：

```python
import numpy as np
import pdb

pdb.set_trace()
arr = np.array([1, 2, 3])
```

然后用n、s、c等命令单步调试。

## 快速验证

安装完成后，来一段简单的代码验证环境正常：

```python
import numpy as np

# 创建数组
arr = np.array([1, 2, 3, 4, 5])
print(f"数组: {arr}")
print(f"形状: {arr.shape}")
print(f"类型: {arr.dtype}")

# 基本运算
print(f"平方: {arr ** 2}")
print(f"求和: {arr.sum()}")
```

能正常运行就说明环境OK了。

## 多Python版本

如果你的系统装了多个Python版本，确保用正确的pip安装：

```bash
# 查看版本
python --version
pip --version

# 指定Python版本
python3.10 -m pip install numpy
```

有时候python和pip指向不同版本，会导致安装了却import不了。确认pip对应正确的python。
