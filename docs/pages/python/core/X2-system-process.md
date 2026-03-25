---
title: 系统与进程
---

# 系统与进程

Python的标准库提供了丰富的系统编程接口，包括文件系统操作、进程管理、环境变量访问等。os、sys、subprocess和shutil是其中最常用的模块。

这一章介绍这些模块的核心用法。

## os模块

os模块提供操作系统相关的功能，是Python标准库中最常用的模块之一。

### 路径操作

os.path模块处理路径：

```python
import os

print(os.path.join("dir", "file.txt"))  # dir/file.txt (Unix) 或 dir\file.txt (Windows)
print(os.path.split("/home/user/file.txt"))  # ('/home/user', 'file.txt')
print(os.path.dirname("/home/user/file.txt"))  # /home/user
print(os.path.basename("/home/user/file.txt"))  # file.txt
```

检查路径：

```python
print(os.path.exists("file.txt"))  # 文件是否存在
print(os.path.isfile("file.txt"))  # 是否是文件
print(os.path.isdir("dir"))  # 是否是目录
print(os.path.islink("link"))  # 是否是符号链接
```

获取路径信息：

```python
print(os.path.getsize("file.txt"))  # 文件大小（字节）
print(os.path.getmtime("file.txt"))  # 修改时间（时间戳）
print(os.path.getatime("file.txt"))  # 访问时间
print(os.path.abspath("file.txt"))  # 绝对路径
print(os.path.realpath("link"))  # 解析符号链接后的真实路径
```

路径扩展：

```python
print(os.path.expanduser("~/file.txt"))  # 展开~为用户主目录
print(os.path.expandvars("$HOME/file.txt"))  # 展开环境变量
```

### 目录操作

创建和删除目录：

```python
os.mkdir("newdir")  # 创建单层目录
os.makedirs("a/b/c")  # 创建多层目录

os.rmdir("emptydir")  # 删除空目录
os.removedirs("a/b/c")  # 删除多层空目录
```

复制、移动、删除文件：

```python
import shutil

shutil.copy("src.txt", "dst.txt")  # 复制文件
shutil.copytree("srcdir", "dstdir")  # 复制目录
shutil.move("src.txt", "dst.txt")  # 移动文件
shutil.rmtree("dir")  # 删除目录树
os.remove("file.txt")  # 删除文件
```

### 工作目录

```python
print(os.getcwd())  # 获取当前工作目录

os.chdir("/tmp")  # 改变当前工作目录
```

### 环境变量

```python
print(os.environ.get("HOME"))  # 获取环境变量
print(os.environ.get("PATH"))

os.environ["MY_VAR"] = "value"  # 设置环境变量
```

环境变量常用于配置和密钥存储。

### 文件描述符

```python
fd = os.open("file.txt", os.O_RDONLY)  # 打开文件，返回文件描述符
print(os.read(fd, 100))  # 读取
os.close(fd)  # 关闭
```

低级文件操作，通常不需要直接使用。

## sys模块

sys模块提供Python解释器相关的功能。

### 命令行参数

```python
import sys

print(sys.argv)  # ['script.py', 'arg1', 'arg2']
print(len(sys.argv))
```

sys.argv[0]是脚本名，其余是参数。

### 模块路径

```python
print(sys.path)  # 模块搜索路径列表
sys.path.insert(0, "/my/modules")  # 添加搜索路径
```

### Python版本

```python
print(sys.version)  # Python版本字符串
print(sys.version_info)  # 版本信息元组
print(sys.platform)  # 操作系统平台
```

### 标准流

```python
sys.stdin  # 标准输入
sys.stdout  # 标准输出
sys.stderr  # 标准错误
```

可以重定向：

```python
old_stdout = sys.stdout
sys.stdout = open("output.txt", "w")
print("This goes to file")
sys.stdout = old_stdout
```

### 解释器信息

```python
print(sys.executable)  # Python解释器路径
print(sys.getrecursionlimit())  # 递归深度限制
sys.setrecursionlimit(2000)  # 设置递归深度限制
print(sys.getdefaultencoding())  # 默认编码
```

### 退出程序

```python
sys.exit(0)  # 正常退出
sys.exit(1)  # 异常退出
```

0表示成功，非0表示失败。

## subprocess模块

subprocess模块用于执行外部命令，是os.system的替代品。

### 基本用法

最简单的执行方式：

```python
import subprocess

result = subprocess.run(["ls", "-la"])
print(result.returncode)  # 退出码
```

run()等待命令完成并返回CompletedProcess对象。

### 获取输出

```python
result = subprocess.run(["echo", "hello"], capture_output=True, text=True)
print(result.stdout)  # hello
print(result.stderr)  # 空字符串
```

capture_output=True等价于stdout=PIPE, stderr=PIPE。text=True指定输出为字符串而非字节。

### 检查退出码

```python
result = subprocess.run(["ls", "/nonexistent"])
print(result.returncode)  # 非0

if result.returncode != 0:
    print("Command failed")
```

或者让run()在非0退出码时抛出异常：

```python
try:
    subprocess.run(["ls", "/nonexistent"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Command failed with {e.returncode}")
```

### 管道连接

用shell=True执行复杂命令：

```python
result = subprocess.run("ls -la | grep python", shell=True, capture_output=True, text=True)
print(result.stdout)
```

注意shell=True有安全风险，不要使用不可信的输入。

### 输入数据

向命令传递输入：

```python
result = subprocess.run(["grep", "pattern"], input="line1\npattern found\nline3\n", text=True, capture_output=True)
print(result.stdout)  # pattern found
```

### 超时控制

```python
try:
    subprocess.run(["sleep", "10"], timeout=3)
except subprocess.TimeoutExpired:
    print("Command timed out")
```

### 异步执行

不等待命令完成：

```python
process = subprocess.Popen(["tail", "-f", "log.txt"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 做其他事...

# 杀死进程
process.terminate()
process.wait()
```

Popen返回Popen对象，可以读写其stdin/stdout/stderr。

## shutil模块

shutil模块提供高级文件操作。

### 文件操作

```python
import shutil

shutil.copy("src.txt", "dst.txt")  # 复制文件
shutil.copy2("src.txt", "dst.txt")  # 复制文件，保留元数据
shutil.copytree("srcdir", "dstdir")  # 复制目录
```

copy2尝试保留文件的修改时间和权限。

### 移动和删除

```python
shutil.move("src.txt", "dst.txt")  # 移动文件或目录
shutil.rmtree("dir")  # 删除目录树
```

### 磁盘使用

```python
total, used, free = shutil.disk_usage("/")
print(f"Total: {total // (2**30)} GB")
print(f"Used: {used // (2**30)} GB")
print(f"Free: {free // (2**30)} GB")
```

### 文件名模式

```python
import glob

print(glob.glob("*.txt"))  # 当前目录下所有txt文件
print(glob.glob("**/*.py", recursive=True))  # 递归搜索所有py文件
print(glob.glob("dir/???.txt"))  # ?匹配单个字符
```

### make_archive

创建压缩包：

```python
shutil.make_archive("archive", "zip", "mydir")  # 创建zip压缩包
shutil.make_archive("archive", "tar", "mydir")  # 创建tar压缩包
```

## pathlib模块

pathlib是处理路径的现代方式，比os.path更直观。

### 基本用法

```python
from pathlib import Path

p = Path("dir/file.txt")

print(p.name)      # file.txt
print(p.stem)      # file
print(p.suffix)     # .txt
print(p.parent)     # dir
print(p.parts)      # ('dir', 'file.txt')
```

### 路径操作

```python
p = Path("dir/subdir/file.txt")

print(p.parent)        # dir/subdir
print(p.parent.parent)  # dir
print(p.anchor)        # 空字符串或盘符

new_p = p.with_name("other.txt")
print(new_p)  # dir/subdir/other.txt

new_p = p.with_suffix(".md")
print(new_p)  # dir/subdir/file.md
```

### 路径组合

```python
p = Path("dir")
print(p / "subdir" / "file.txt")  # dir/subdir/file.txt
```

除法运算符自动处理路径分隔符。

### 检查路径

```python
p = Path("file.txt")

print(p.exists())
print(p.is_file())
print(p.is_dir())
print(p.is_symlink())
```

### 创建路径

```python
p = Path("newdir")
p.mkdir()  # 创建目录
p.mkdir(parents=True)  # 创建多层目录
p.rmdir()  # 删除空目录
```

### 读取和写入

```python
p = Path("file.txt")
p.write_text("Hello")  # 写入文本
print(p.read_text())  # 读取文本

p.write_bytes(b"Hello")  # 写入字节
print(p.read_bytes())  # 读取字节
```

### 迭代目录

```python
p = Path("dir")
for child in p.iterdir():
    print(child.name)
```

glob：

```python
for py_file in p.glob("*.py"):
    print(py_file)

for py_file in p.rglob("*.py"):  # 递归
    print(py_file)
```

## 常见问题

### 跨平台路径

Unix和Windows路径分隔符不同：

```python
# 错误
path = "dir/file.txt"  # Windows上可能有问题

# 正确
path = os.path.join("dir", "file.txt")
path = Path("dir") / "file.txt"
```

### 文件权限

Unix系统上：

```python
import os

os.chmod("file.txt", 0o755)  # 设置权限
os.chown("file.txt", uid, gid)  # 设置所有者
```

### 临时文件

```python
import tempfile

with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(b"data")
    print(f.name)

with tempfile.TemporaryDirectory() as d:
    print(d)
```

### 进程环境

子进程继承父进程环境：

```python
import os
import subprocess

env = os.environ.copy()
env["MY_VAR"] = "value"

subprocess.run(["command"], env=env)
```

## 面试关注点

面试中关于系统与进程的常见问题包括：os.path和pathlib的区别？subprocess比os.system好在哪里？如何避免shell=True的安全风险？

理解进程和文件操作是基础。重点是跨平台处理和资源管理。
