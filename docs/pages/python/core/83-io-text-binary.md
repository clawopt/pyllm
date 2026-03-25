# 文本与二进制

Python处理文件时区分文本模式和二进制模式，这两种模式在数据存储方式和性能特性上有根本差异。理解它们的区别对于正确处理各类文件至关重要。

## 文本与二进制的本质差异

计算机内存中所有数据都以二进制形式存储。文本模式和二进制模式的区别在于如何解释这些二进制数据。

文本模式将文件内容视为字符序列。当写入字符串时，Python需要将字符转换为字节序列，这个转换过程称为编码。读取时，Python将字节序列转换回字符，这个过程称为解码。文本模式还会处理特定平台的换行符差异，在Windows上自动将`\r\n`转换为`\n`。

二进制模式则直接将字节传递给程序，不做任何转换。写入什么，文件就存储什么；读取什么，程序就得到什么。

```python
text_content = "你好"
encoded = text_content.encode('utf-8')
print(encoded)  # b'\xe4\xbd\xa0\xe5\xa5\xbd'
```

这段代码展示了字符串到字节的转换过程。文本模式下的"你好"是字符，二进制模式下的`b'\xe4\xbd\xa0\xe5\xa5\xbd'`是原始字节。

## 编码与解码

编码是将字符串转换为字节序列的过程。不同的编码方式将字符映射到不同的字节序列：

```python
s = "Python"

utf_8 = s.encode('utf-8')
gbk = s.encode('gbk')

print(len(utf_8))  # 6
print(len(gbk))    # 6
print(utf_8)       # b'Python'
print(gbk)         # b'Python'
```

对于纯ASCII字符，UTF-8和GBK的编码结果相同。但对于中文，差异就显现出来了：

```python
s = "中文"

utf_8 = s.encode('utf-8')
gbk = s.encode('gbk')

print(len(utf_8))  # 6
print(len(gbk))    # 4
print(utf_8)       # b'\xe4\xb8\xad\xe6\x96\x87'
print(gbk)         # b'\xd6\xd0\xce\xc4'
```

UTF-8使用1到4个字节表示一个字符，中文通常占3个字节。GBK使用2个字节表示一个中文字符。对于纯ASCII文本，UTF-8的优势在于与ASCII兼容——每个ASCII字符只占1个字节，与传统ASCII一致。

## 乱码的根源

乱码的本质是编码与解码使用了不匹配的字符集。当你用UTF-8编码写入文件，却用GBK打开读取时，就会出现乱码：

```python
content = "你好世界"

with open('example.txt', 'w', encoding='utf-8') as f:
    f.write(content)

with open('example.txt', 'r', encoding='gbk') as f:
    print(f.read())  # 乱码
```

反过来，用GBK编码写入，用UTF-8读取，同样会乱码。解决乱码的方法很简单：确保编码和解码使用相同的字符集。

```python
with open('example.txt', 'w', encoding='utf-8') as f:
    f.write("你好")

with open('example.txt', 'r', encoding='utf-8') as f:
    print(f.read())  # 正常：你好
```

## 换行符的差异

不同操作系统使用不同的换行符：Unix使用`\n`，Windows使用`\r\n`，旧版Mac使用`\r`。文本模式会自动处理这些差异：

```python
with open('unix.txt', 'w', newline='') as f:
    f.write("line1\nline2")

with open('windows.txt', 'w', newline='') as f:
    f.write("line1\r\nline2")

with open('unix.txt', 'rb') as f:
    print(repr(f.read()))

with open('windows.txt', 'rb') as f:
    print(repr(f.read()))
```

如果你不想让Python自动处理换行符，使用`newline=''`参数。这样换行符会原样保留。

在读取时，`newline=''`参数也会影响换行符的处理：

```python
with open('mixed.txt', 'r', newline='') as f:
    content = f.read()

with open('mixed.txt', 'r') as f:
    content = f.read()
```

前者会保留原始换行符，后者会将所有换行符统一为`\n`。

## 二进制文件的应用

图片、音频、视频、可执行文件等都应该以二进制模式处理：

```python
with open('image.png', 'rb') as f:
    data = f.read()

print(type(data))  # <class 'bytes'>
print(len(data))   # 文件大小（字节）
```

复制图片文件的正确方式：

```python
def copy_binary(src, dst):
    with open(src, 'rb') as f_src:
        with open(dst, 'wb') as f_dst:
            while chunk := f_src.read(8192):
                f_dst.write(chunk)
```

使用较大的缓冲区（如8192字节）可以避免频繁的系统调用，提高复制效率。对于大文件，这种逐块读取的方式比一次性读取全部内容更节省内存。

## 字节与字符串的转换

在处理二进制数据时，经常需要在字节和特定类型之间进行转换：

```python
import struct

packed = struct.pack('>I', 1024)
print(packed)  # b'\x00\x00\x04\x00'

unpacked = struct.unpack('>I', packed)
print(unpacked)  # (1024,)
```

`struct`模块用于处理二进制结构化数据。格式字符串`>I`表示大端序的无符号整数。

处理二进制协议时，经常需要逐字节解析：

```python
def parse_header(data):
    magic = data[0:4]
    version = data[4]
    length = int.from_bytes(data[5:9], 'big')
    return magic, version, length
```

`int.from_bytes()`将字节转换为整数，是处理二进制协议时的常用操作。

## Base64编码

Base64是一种将二进制数据转换为可打印ASCII字符的编码方式，常用于在文本协议中传输二进制数据：

```python
import base64

binary_data = b'\x00\x01\x02\x03'
encoded = base64.b64encode(binary_data)
print(encoded)  # b'AAECAw=='

decoded = base64.b64decode(encoded)
print(decoded)  # b'\x00\x01\x02\x03'
```

Base64编码后的数据比原始数据大约大33%，因为每3个字节被转换为4个字符。常见的应用场景包括电子邮件附件和JWT令牌。

## 内存中的字符串与字节

Python 3中字符串（`str`）和字节（`bytes`）是两种完全不同的类型：

```python
s = "你好"
b = b"hello"

print(type(s))  # <class 'str'>
print(type(b))  # <class 'bytes'>
print(isinstance(s, str))  # True
print(isinstance(b, bytes))  # True
```

字符串用于人类可读的文本内容，字节用于原始二进制数据。两者之间的转换必须通过编码和解码完成：

```python
str_to_bytes = "文本".encode('utf-8')
bytes_to_str = b'\xe6\x96\x87\xe6\x9c\xac'.decode('utf-8')
```

## 常见问题处理

### 指定默认编码

不同操作系统的默认编码可能不同，这会导致程序在不同环境下表现不一致：

```python
import sys

print(sys.getdefaultencoding())  # utf-8
```

为了代码的可移植性，始终显式指定编码是个好习惯：

```python
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()
```

### 处理BOM

UTF-8有BOM（Byte Order Mark）变体，Windows记事本保存UTF-8时可能添加BOM：

```python
with open('with_bom.txt', 'rb') as f:
    content = f.read()
    print(content[:3])  # b'\xef\xbb\xbf' BOM标记

with open('with_bom.txt', 'r', encoding='utf-8-sig') as f:
    content = f.read()  # 自动处理BOM
```

`encoding='utf-8-sig'`会在读取时自动跳过BOM，写入时添加BOM。如果确定文件不含BOM，使用`utf-8`即可。

### 大文件处理

处理超大文件时，不应一次性将内容读入内存：

```python
def process_large_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            process(line)
```

逐行读取可以保证内存使用量与文件大小无关。对于二进制文件，使用`read(chunk_size)`循环读取：

```python
def count_bytes(filepath):
    count = 0
    with open(filepath, 'rb') as f:
        while f.read(8192):
            count += 8192
    return count
```

## 面试关注点

面试中关于文本与二进进制的常见问题包括：什么是编码和解码？为什么会有乱码？如何处理大文件的读取？UTF-8和GBK有什么区别？

需要理解的核心概念是：文本模式是Unicode字符串的输入输出，二进制模式是字节序列的输入输出。编码是将字符串转换为字节，解码是将字节转换为字符串。乱码发生在编码和解码字符集不一致时。

高级面试题可能涉及：为什么UTF-8是Unicode的最佳实现之一？Python 3的str和bytes设计有什么考量？如何处理未知编码的文件？
