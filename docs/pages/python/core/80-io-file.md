---
title: 文件操作基础
---

# 文件操作基础

几乎所有程序都需要和文件打交道。读取配置文件、保存用户数据、写入日志——这些都涉及文件操作。

Python 的文件操作相对简单，核心就是 open 函数。但简单不代表可以随便用——文件操作有很多细节需要注意，比如正确关闭文件、处理编码、选择合适的打开模式。如果这些没处理好，可能会导致数据丢失、文件损坏，或者安全漏洞。

这一章讲文件操作的基础知识，让你能正确地读写文件。

## 打开文件

用 `open()` 函数打开文件，返回一个文件对象：

```python
f = open('test.txt', 'r')  # 打开文件用于读取
```

`open()` 的第一个参数是文件名，第二个参数是模式。

常用模式：

| 模式 | 含义 |
|-------|------|
| 'r' | 读取（默认） |
| 'w' | 写入（会清空文件） |
| 'a' | 追加 |
| 'x' | 创建并写入，文件存在则报错 |
| 'b' | 二进制模式（可组合） |
| '+' | 读写模式（可组合） |

组合使用：`'rb'` 是二进制读取，`'w+'` 是读写并清空文件。

## 读取文件

有多种读取方式：

`read()` 读取全部内容：

```python
f = open('test.txt', 'r')
content = f.read()
f.close()
```

`readline()` 读取一行：

```python
f = open('test.txt', 'r')
line = f.readline()
f.close()
```

`readlines()` 读取所有行并返回列表：

```python
f = open('test.txt', 'r')
lines = f.readlines()
f.close()
```

也可以直接遍历文件对象，它是可迭代的：

```python
with open('test.txt', 'r') as f:
    for line in f:
        print(line.strip())
```

## 写入文件

`write()` 写入字符串：

```python
f = open('test.txt', 'w')
f.write('Hello, World!\n')
f.close()
```

`writelines()` 写入多行：

```python
f = open('test.txt', 'w')
f.writelines(['Line 1\n', 'Line 2\n', 'Line 3\n'])
f.close()
```

注意：写入不会自动添加换行符，需要手动添加 `\n`。

## with 语句与资源管理

文件操作最重要的一点是：正确关闭文件。

如果文件打开后忘记关闭，会导致资源泄漏。在极端情况下，可能会耗尽系统能打开的文件数量。

```python
# 错误的写法
f = open('test.txt', 'w')
f.write('Hello')
# 如果这里发生异常，文件就不会被关闭
```

正确的方法是用 `with` 语句，它会自动关闭文件：

```python
with open('test.txt', 'w') as f:
    f.write('Hello')
# 文件在这里自动关闭，即使发生异常也会关闭
```

`with` 语句通过上下文管理器实现资源管理，是 Python 最推荐的 文件操作方式。任何需要关闭的资源（文件、网络连接、数据库连接）都应该用 `with` 语句管理。

## 文件指针与 seek()

文件指针标记当前读取或写入的位置：

```python
with open('test.txt', 'r+') as f:  # r+ 是读写模式
    content = f.read()  # 读取全部，指针到末尾
    f.seek(0)          # 把指针移回开头
    f.write('New content')  # 从开头写入
```

`seek(offset, whence)` 移动指针：
- `whence=0`（默认）：从文件开头计算偏移
- `whence=1`：从当前位置计算
- `whence=2`：从文件末尾计算

```python
with open('test.txt', 'rb') as f:
    f.seek(-10, 2)  # 从末尾往前10个字节
    data = f.read()
```

`tell()` 返回当前指针位置：

```python
with open('test.txt', 'r') as f:
    pos = f.tell()
    print(f"Current position: {pos}")
```

## 文本模式 vs 二进制模式

`'r'` 和 `'w'` 是文本模式，默认使用系统编码（通常是 UTF-8）。

二进制模式用 `'rb'` 和 `'wb'`，读写的是字节串 `bytes` 而不是字符串 `str`：

```python
# 文本模式
with open('text.txt', 'w') as f:
    f.write('你好')  # 字符串

# 二进制模式
with open('data.bin', 'wb') as f:
    f.write(b'\x00\x01\x02')  # 字节串
```

处理图片、音频、压缩文件等二进制数据时，必须用二进制模式。

## 读取大文件

读取大文件时，不要用 `read()` 一次性读取全部内容，否则会占用大量内存：

```python
# 不好：一次性读取整个文件
with open('big_file.txt', 'r') as f:
    content = f.read()  # 文件有 10GB，这里就会用 10GB 内存
```

应该用迭代方式逐行读取：

```python
# 好：逐行读取
with open('big_file.txt', 'r') as f:
    for line in f:
        process(line)  # 每次只占用一行内存
```

或者用生成器：

```python
def read_large_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line
```

## 常见误区

第一个误区是忘记关闭文件。忘记关闭文件会导致资源泄漏，正确做法是始终用 `with` 语句。

第二个误区是在文本模式下读写字节。如果用 `'r'` 模式读取二进制数据会报错，应该用 `'rb'`。

第三个误区是用 `'w'` 模式写入已存在的文件。`'w'` 会清空文件内容，如果想保留原内容应该用 `'a'` 追加模式。

第四个误区是不处理编码。跨平台或在非英文环境下，编码问题很常见。最好显式指定编码：`open('file.txt', 'r', encoding='utf-8')`。

理解文件操作的基础知识，是编写可靠程序的重要一步。
