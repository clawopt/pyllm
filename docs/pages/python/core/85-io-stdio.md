# 标准输入输出

标准输入、标准输出和标准错误输出是程序与外部环境交互的三种基本流。理解这三种流的差异和用法，是编写命令行工具和系统脚本的基础。

## 三种标准流

每个Python程序启动时，都会自动打开三个流：

- `sys.stdin`：标准输入流，默认连接到键盘输入
- `sys.stdout`：标准输出流，默认连接到终端屏幕
- `sys.stderr`：标准错误输出流，默认也连接到终端屏幕，但专门用于错误信息

```python
import sys

print("标准输出", file=sys.stdout)
print("标准错误", file=sys.stderr)
```

命令行运行时，两种输出都会显示在屏幕上，但它们本质上是分开的流。这种设计允许我们将正常输出和错误信息分别重定向到不同的目的地。

## input函数

`input()`是最常用的标准输入函数，它从`sys.stdin`读取一行：

```python
name = input("请输入你的名字：")
print(f"你好，{name}")
```

`input()`会阻塞程序执行，直到用户输入一行内容并按下回车键。括号内的字符串会作为提示信息显示。

读取多行输入时，可以结合循环：

```python
print("输入多行，结束后按Ctrl+D（Unix）或Ctrl+Z（Windows）：")
lines = []
while True:
    try:
        line = input()
        lines.append(line)
    except EOFError:
        break
```

`EOFError`在输入结束时被抛出，这发生于用户发送文件结束信号时。

## print函数

`print()`函数将内容写入`sys.stdout`：

```python
print("Hello")
print("Hello", "World")
print("Name:", "Python", sep="-", end="!\n")
```

- 多个参数会自动用空格连接
- `sep`参数指定分隔符
- `end`参数指定行尾字符，默认为换行符

### 格式化输出

Python提供多种格式化字符串的方式：

```python
name = "Python"
version = 3.10

print(f"{name} {version}")  # f-string
print("{} {}".format(name, version))  # format()
print("%s %d" % (name, version))  # %格式化
```

f-string是Python 3.6引入的，语法最简洁，是目前推荐的方式。

## 重定向与管道

Shell允许重定向标准流的方向：

```bash
python script.py > output.txt    # stdout重定向到文件
python script.py 2> errors.txt   # stderr重定向到文件
python script.py &> all.txt      # 所有输出重定向到文件
python script.py > output.txt 2>&1  # stdout和stderr都重定向到同一文件
```

管道将一个程序的stdout连接到另一个程序的stdin：

```bash
cat data.txt | grep "pattern" | sort | uniq
```

这个命令链中，数据从`data.txt`读取，经过grep过滤、sort排序、uniq去重，最后输出到屏幕。

## 在Python中重定向流

Python允许程序内部重定向标准流：

```python
import sys
from io import StringIO

old_stdout = sys.stdout
sys.stdout = StringIO()

print("这不会显示在屏幕上")

output = sys.stdout.getvalue()
sys.stdout = old_stdout

print("恢复输出:", output)
```

这种技术常用于捕获函数的输出，或在测试中验证打印内容。

### 重定向文件

将stdout重定向到文件：

```python
import sys

with open('output.log', 'w') as f:
    old_stdout = sys.stdout
    sys.stdout = f
    print("写入文件")
    sys.stdout = old_stdout

print("回到屏幕")
```

同时重定向stdout和stderr：

```python
import sys

with open('output.log', 'w') as out, open('errors.log', 'w') as err:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = out
    sys.stderr = err

    print("标准输出")
    print("标准错误", file=sys.stderr)

    sys.stdout = old_stdout
    sys.stderr = old_stderr
```

## 命令行参数

`sys.argv`提供命令行参数访问：

```bash
python script.py arg1 arg2 arg3
```

```python
import sys

print(sys.argv)  # ['script.py', 'arg1', 'arg2', 'arg3']
print(len(sys.argv))  # 4
print(sys.argv[1])  # arg1
```

`sys.argv[0]`是脚本名称，其余是传入的参数。

### argparse模块

对于复杂的命令行接口，应该使用`argparse`模块：

```python
import argparse

parser = argparse.ArgumentParser(description='处理文件的工具')
parser.add_argument('filename', help='输入文件名')
parser.add_argument('-o', '--output', help='输出文件名', default='result.txt')
parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
parser.add_argument('-n', '--number', type=int, default=10, help='处理数量')

args = parser.parse_args()

print(f"输入: {args.filename}")
print(f"输出: {args.output}")
print(f"详细模式: {args.verbose}")
print(f"数量: {args.number}")
```

`argparse`自动生成帮助信息、处理参数类型、验证必要参数。

```bash
python tool.py --help
usage: tool.py [-h] [-o OUTPUT] [-v] [-n NUMBER] filename

处理文件的工具

positional arguments:
  filename              输入文件名

optional arguments:
  -h, --help            显示帮助信息
  -o OUTPUT, --output OUTPUT
                        输出文件名
  -v, --verbose         详细输出
  -n NUMBER, --number NUMBER
                        处理数量
```

## 环境变量

程序可以通过环境变量获取配置信息：

```python
import os

lang = os.environ.get('LANG', 'en_US')
print(f"语言设置: {lang}")

home = os.environ.get('HOME')
print(f"主目录: {home}")
```

环境变量在程序启动时继承自shell，适合存储API密钥、调试开关等配置。

### 修改环境变量

修改环境变量只会影响当前进程及其子进程：

```python
import os

os.environ['MY_VAR'] = 'value'
print(os.environ.get('MY_VAR'))

os.environ['PATH'] = '/new/path:' + os.environ['PATH']
```

这种特性常用于在测试时临时修改环境配置，或在子进程中设置不同的环境。

## 文件描述符

文件描述符是操作系统分配给打开文件的整数标识。标准流对应的文件描述符是：

- 0：`sys.stdin`
- 1：`sys.stdout`
- 2：`sys.stderr`

```python
import sys

print(sys.stdin.fileno())  # 0
print(sys.stdout.fileno())  # 1
print(sys.stderr.fileno())  # 2
```

直接使用文件描述符可以进行低层次的IO操作：

```python
import os

os.write(1, b"Direct output\n")

data = os.read(0, 100)
```

这种方式在编写系统工具时很有用，但大多数应用层代码应该使用高级IO函数。

## 常见错误处理

### 输入错误

处理用户输入错误时，应该给出清晰的提示并允许重试：

```python
while True:
    try:
        age = int(input("请输入年龄："))
        if age < 0:
            print("年龄不能为负数")
            continue
        break
    except ValueError:
        print("请输入有效的数字")
```

### 编码问题

标准输入输出可能存在编码问题：

```python
import sys

print(sys.stdin.encoding)
print(sys.stdout.encoding)

print("你好", flush=True)
```

`flush=True`参数强制立即输出，不等待缓冲区满。

## 面试关注点

面试中关于标准IO的常见问题包括：stdin、stdout、stderr有什么区别？如何重定向标准输出？命令行参数如何传递和处理？

理解标准流的抽象很重要：它们是程序与外部环境交互的基本接口，无论是键盘、文件还是网络，都是通过这三个流进行数据交换。

高级面试题可能涉及：如何实现一个支持管道的命令行工具？为什么print要flush参数？如何处理stdin的编码问题？
