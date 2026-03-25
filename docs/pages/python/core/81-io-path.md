---
title: 路径操作
---

# 路径操作

读写文件时，路径是必不可少的。Python 提供了两种处理路径的方式：`os.path` 模块和 `pathlib.Path` 对象。

`os.path` 是传统的基于字符串的路径操作方式，而 `pathlib` 是 Python 3.4 引入的面向对象的路径处理方式，更现代、更直观。

理解路径操作对于编写跨平台代码尤其重要。不同操作系统的路径格式不同（Windows 用 `\`，Unix 用 `/`），如果不正确处理路径，代码可能在不同系统上运行失败。

## os.path 模块

`os.path` 提供了一系列基于字符串的路径操作函数：

```python
import os

# 路径拼接
path = os.path.join('dir', 'subdir', 'file.txt')
# 在 Linux/macOS: dir/subdir/file.txt
# 在 Windows: dir\subdir\file.txt

# 获取绝对路径
abs_path = os.path.abspath('file.txt')

# 获取文件名
filename = os.path.basename('/path/to/file.txt')  # file.txt

# 获取目录
dirname = os.path.dirname('/path/to/file.txt')  # /path/to

# 判断是否存在
exists = os.path.exists('/path/to/file.txt')
is_file = os.path.isfile('/path/to/file.txt')
is_dir = os.path.isdir('/path/to')
```

`os.path.join` 是最重要的函数之一，它会自动根据操作系统选择正确的路径分隔符：

```python
import os

# 安全拼接路径
path = os.path.join('folder', 'file.txt')
# 在不同系统上都能正常工作
```

## pathlib.Path 面向对象路径

`pathlib` 提供了面向对象的路径处理方式，更直观易用：

```python
from pathlib import Path

# 创建路径对象
p = Path('folder/subfolder/file.txt')

# 路径拼接
p = Path('folder') / 'subfolder' / 'file.txt'

# 获取文件名
filename = p.name  # file.txt

# 获取后缀
suffix = p.suffix  # .txt

# 获取父目录
parent = p.parent  # folder/subfolder

# 获取绝对路径
abs_path = p.resolve()
```

`Path` 对象可以用 `/` 运算符拼接路径，比 `os.path.join` 更直观：

```python
from pathlib import Path

base = Path('/home/user')
result = base / 'documents' / 'file.txt'
print(result)  # /home/user/documents/file.txt
```

## 路径的创建、检测、遍历

`Path` 对象提供了丰富的方法来操作路径：

```python
from pathlib import Path

p = Path('/home/user/documents')

# 检测
p.exists()       # 路径是否存在
p.is_file()       # 是否是文件
p.is_dir()        # 是否是目录

# 创建
p.mkdir(parents=True, exist_ok=True)  # 创建目录
p.touch()                           # 创建空文件

# 删除
p.unlink()        # 删除文件
p.rmdir()         # 删除空目录

# 遍历目录
for item in p.iterdir():
    print(item.name)
```

`iterdir()` 遍历目录中的所有项，`glob()` 用通配符匹配文件：

```python
from pathlib import Path

p = Path('/home/user/documents')

# 遍历所有 .txt 文件
for txt_file in p.glob('*.txt'):
    print(txt_file)

# 递归遍历所有子目录
for py_file in p.rglob('*.py'):
    print(py_file)
```

## 跨平台路径处理

跨平台开发时，路径处理要特别注意。

不同操作系统的路径分隔符不同：

```python
import os

# Windows: dir\subdir\file.txt
# Unix/Linux/macOS: dir/subdir/file.txt

# 始终用 os.path.join 或 pathlib 的 / 运算符
path = os.path.join('dir', 'subdir', 'file.txt')

# 不要手动拼接路径
# 错误写法：
path = 'dir' + '/' + 'subdir' + '/' + 'file.txt'  # 可能在 Windows 上出错
```

`pathlib.Path` 默认使用当前操作系统的分隔符：

```python
from pathlib import Path

# Path 会自动处理跨平台问题
p = Path('folder') / 'subfolder' / 'file.txt'
print(p)  # 自动使用正确的分隔符
```

## 路径的读取与写入

`Path` 对象可以直接读写文件，不需要 `open()`：

```python
from pathlib import Path

p = Path('test.txt')

# 写入
p.write_text('Hello, World!')

# 读取
content = p.read_text()

# 二进制读写
p.write_bytes(b'\x00\x01\x02')
data = p.read_bytes()
```

`read_text()` 和 `write_text()` 是便捷方法，相当于用 `open()` 读写文本。`read_bytes()` 和 `write_bytes()` 用于二进制数据。

## 获取特殊路径

`pathlib` 提供了获取特殊目录的方法：

```python
from pathlib import Path

# 当前目录
current = Path.cwd()
print(current)

# 用户主目录
home = Path.home()
print(home)

# 临时目录
import tempfile
temp_dir = Path(tempfile.gettempdir())
```

`Path.cwd()` 获取当前工作目录，`Path.home()` 获取用户主目录。这些在处理配置文件、日志文件等时很有用。

## 常见误区

第一个误区是手动拼接路径。应该始终用 `os.path.join` 或 `pathlib.Path` 的 `/` 运算符，手动拼接会导致跨平台问题。

第二个误区是混淆绝对路径和相对路径。在不同工作目录下运行程序时，相对路径可能指向不同位置。如果需要确定路径，应该用 `resolve()` 或 `os.path.abspath()`。

第三个误区是路径中存在中文字符时编码问题。在 Windows 上，中文路径可能会有编码问题，应该显式指定编码或使用 `pathlib`。

第四个误区是删除文件前不检查存在性。删除不存在的文件会抛出异常，应该先检查或使用 `exist_ok=True` 参数。

理解路径操作，正确处理不同操作系统的路径差异，是编写可靠跨平台代码的基础。
