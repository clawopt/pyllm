---
title: 正则表达式
---

# 正则表达式

正则表达式是处理文本的强大工具。它使用特殊语法描述文本模式，可以实现复杂的文本匹配、提取、替换和验证操作。

正则表达式在数据验证、文本解析、日志分析、信息提取等场景中广泛应用。这一章介绍Python中正则表达式的用法。

## 正则基础

正则表达式由普通字符和元字符组成。普通字符匹配自身，元字符有特殊含义。

### 基本元字符

```python
import re

# 精确匹配
pattern = r"hello"
text = "hello world"
match = re.search(pattern, text)
print(match.group())  # hello
```

### 字符类

用方括号指定字符集：

```python
# 匹配a、b或c
pattern = r"[abc]"

# 匹配任意小写字母
pattern = r"[a-z]"

# 匹配任意字母
pattern = r"[a-zA-Z]"

# 匹配任意数字
pattern = r"[0-9]"
```

字符类内可以使用脱字符取反：

```python
# 匹配非数字
pattern = r"[^0-9]"
```

### 预定义字符类

| 符号 | 含义 |
|------|------|
| . | 任意字符（除换行） |
| \d | 数字 [0-9] |
| \D | 非数字 [^0-9] |
| \w | 单词字符 [a-zA-Z0-9_] |
| \W | 非单词字符 |
| \s | 空白字符 |
| \S | 非空白字符 |

```python
pattern = r"\d+"  # 匹配一个或多个数字
text = "abc123def"
print(re.search(pattern, text).group())  # 123
```

## 量词

量词指定匹配次数：

| 符号 | 含义 |
|------|------|
| * | 零次或多次 |
| + | 一次或多次 |
| ? | 零次或一次 |
| {n} | 恰好n次 |
| {n,} | 至少n次 |
| {n,m} | n到m次 |

```python
# 匹配手机号
pattern = r"1[3-9]\d{9}"
# 1开头，第二位3-9，后面9位数字

# 匹配邮箱
pattern = r"\w+@\w+\.\w+"
```

### 贪婪vs非贪婪

默认量词是贪婪的，会尽可能多地匹配：

```python
text = "<div>content</div>"
pattern = r"<div>.*</div>"
match = re.search(pattern, text)
print(match.group())  # <div>content</div>，贪婪匹配
```

非贪婪量词在量词后加`?`，尽可能少地匹配：

```python
pattern = r"<div>.*?</div>"
match = re.search(pattern, text)
print(match.group())  # <div>content</div>
```

对于`<div>a</div><div>b</div>`，贪婪匹配整个字符串，非贪婪分别匹配两个。

## re模块函数

### search与match

`search`在字符串中搜索匹配：

```python
import re

text = "The price is $99"

pattern = r"\$?\d+"
print(re.search(pattern, text).group())  # 99

pattern = r"\$"
print(re.search(pattern, text).group())  # $
```

`match`只在字符串开头匹配：

```python
text = "hello world"

print(re.match(r"hello", text).group())  # hello
print(re.match(r"world", text))  # None，不在开头
```

### findall

返回所有匹配的列表：

```python
text = "abc123def456"

pattern = r"\d+"
print(re.findall(pattern, text))  # ['123', '456']
```

### finditer

返回迭代器，比findall节省内存：

```python
text = "abc123def456"

pattern = r"\d+"
for match in re.finditer(pattern, text):
    print(match.group(), match.span())
```

### sub替换

替换匹配的文本：

```python
text = "abc123def456"

pattern = r"\d+"
result = re.sub(pattern, "NUM", text)
print(result)  # abcNUMdefNUM
```

可以用函数控制替换值：

```python
def add_one(match):
    num = int(match.group())
    return str(num + 1)

result = re.sub(r"\d+", add_one, text)
print(result)  # abc124def457
```

### split分割

按模式分割：

```python
text = "apple,banana;cherry|orange"

pattern = r"[,;|]"
result = re.split(pattern, text)
print(result)  # ['apple', 'banana', 'cherry', 'orange']
```

## 分组

用圆括号创建分组：

```python
text = "John has 5 apples"

pattern = r"(\w+) has (\d+) (\w+)"
match = re.search(pattern, text)

print(match.group())     # 全部匹配
print(match.group(1))   # John
print(match.group(2))   # 5
print(match.group(3))   # apples
```

命名分组：

```python
pattern = r"(?P<name>\w+) has (?P<count>\d+) (?P<fruit>\w+)"
match = re.search(pattern, text)

print(match.group("name"))   # John
print(match.group("count")) # 5
print(match.group("fruit")) # apples
```

### 提取数据

从复杂文本中提取结构化数据：

```python
log = '2024-01-15 10:30:45 ERROR connection failed'

pattern = r"(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) (?P<level>\w+) (?P<msg>.+)"
match = re.search(pattern, log)

if match:
    print(match.group("date"))   # 2024-01-15
    print(match.group("time"))   # 10:30:45
    print(match.group("level")) # ERROR
    print(match.group("msg"))   # connection failed
```

### 反向引用

在正则中引用分组：

```python
# 匹配引号内的内容
pattern = r'["\'](\w+)["\']'
text = '"hello" and \'world\''
print(re.findall(pattern, text))  # ['hello', 'world']
```

匹配成对的标签：

```python
text = "<div>content</div>"
pattern = r"<(\w+)>.*?</\1>"
print(re.search(pattern, text).group())  # <div>content</div>
```

`\1`引用第一个分组`(\w+)`。

## 编译与Flags

重复使用同一模式时，编译可以提高效率：

```python
import re

pattern = re.compile(r"\d+")

print(pattern.search("abc123"))  # 匹配123
print(pattern.findall("abc123def456"))  # ['123', '456']
```

编译后的pattern对象有相同的方法。

### Flags

flags修改匹配行为：

```python
text = "Hello\nWorld"

pattern = r"hello world"
print(re.search(pattern, text, re.IGNORECASE))  # 忽略大小写
print(re.search(pattern, text, re.DOTALL))  # .匹配换行
print(re.search(pattern, text, re.MULTILINE))  # ^$匹配每行
```

常用flags：

| Flag | 含义 |
|------|------|
| re.IGNORECASE 或 re.I | 忽略大小写 |
| re.DOTALL 或 re.S | .匹配换行 |
| re.MULTILINE 或 re.M | ^$匹配每行 |
| re.VERBOSE 或 re.X | 允许注释和空白 |

### VERBOSE模式

复杂正则可以用VERBOSE模式分行写：

```python
pattern = re.compile(r"""
    \d{4}  # 年份
    -
    \d{2}  # 月份
    -
    \d{2}  # 日期
""", re.VERBOSE)
```

## 常见应用

### 验证手机号

```python
def validate_phone(phone):
    pattern = r"^1[3-9]\d{9}$"
    return bool(re.match(pattern, phone))

print(validate_phone("13812345678"))  # True
print(validate_phone("12345678901"))  # False
```

### 提取URL

```python
text = "Visit https://example.com/path?q=1 or http://test.org"

pattern = r"https?://[\w.-]+(?:/[\w./?=&%-]*)?"
urls = re.findall(pattern, text)
print(urls)  # ['https://example.com/path?q=1', 'http://test.org']
```

### 处理日志

```python
log = '2024-01-15 10:30:45 ERROR [auth] Login failed for user admin'

pattern = r"(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) (?P<level>\w+) \[(?P<module>\w+)\] (?P<msg>.+)"
match = re.search(pattern, log)

if match:
    print(f"{match.group('date')} {match.group('time')} - {match.group('msg')}")
```

### 敏感信息脱敏

```python
def mask_phone(phone):
    return re.sub(r"(\d{3})\d{4}(\d{4})", r"\1****\2", phone)

def mask_id(id_card):
    return re.sub(r"(\d{6})\d{8}(\d{4})", r"\1********\2", id_card)

print(mask_phone("13812345678"))  # 138****5678
print(mask_id("110101199001011234"))  # 110101********1234
```

## 常见问题

### 转义字符

在正则中表示特殊字符需要转义：

```python
# 匹配IP地址
pattern = r"\d+\.\d+\.\d+\.\d+"

# 或者使用字符类
pattern = r"[\d.]+"
```

### 边界匹配

使用`^`和`$`匹配边界：

```python
pattern = r"^\d+$"
print(re.match(pattern, "123"))  # 匹配整个字符串是数字
print(re.match(pattern, "123abc"))  # None
```

### 匹配优先级

正则表达式有复杂的优先级规则：

```python
# "hello" "world" 或 "hi" "there"
pattern = r'"(?:hello|hi)" "(?:world|there)"'
```

括号既用于分组，也用于定义选择范围。

## 面试关注点

面试中关于正则的常见问题包括：贪婪和非贪婪的区别？如何提取分组？re.search和re.match有什么区别？

实际面试中，可能要求手写常见正则：手机号、邮箱、URL、IP地址等。或者给一个复杂场景，要求用正则解析。

正则的性能也很重要：预编译、使用恰当的量词、避免过度回溯。
