---
title: 高级参数特性
---

# 高级参数特性

函数基础那章解决了"怎么定义函数"和"怎么调用函数"的问题。但在实际编程中，有时候你需要更大的灵活性。比如你写一个打印日志的函数，可能需要支持任意数量的参数；你写一个数学运算函数，可能需要支持把一个列表或字典"展开"成参数传入。

这就是本章要解决的问题：高级参数特性。这些特性不是必须掌握的，但当你需要编写通用库函数或者处理各种输入时，它们会变得非常重要。

## 可变参数 *args

有时候你需要让函数接受任意数量的参数。比如你想写一个求和函数，但不知道会有多少个数相加。这时可以用 `*args` 来接收多余的位置参数：

```python
def sum_all(*args):
    total = 0
    for num in args:
        total += num
    return total

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15
print(sum_all())                 # 0
```

当调用 `sum_all(1, 2, 3)` 时，Python 把 1、2、3 打包成一个元组，赋值给 args。然后在函数体内遍历这个元组来求和。

`*args` 的星号表示"收集"（collecting）多余的参数。反过来，在调用函数时，星号可以表示"展开"（unpacking）一个可迭代对象：

```python
def add(a, b):
    return a + b

nums = [3, 5]
print(add(*nums))  # 8，等价于 add(3, 5)
```

`*` 把列表展开成位置参数传递给函数。

## 关键字参数 **kwargs

和 `*args` 类似，`**kwargs` 用于接收任意数量的关键字参数：

```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=18, city="Beijing")
```

输出：

```
name: Alice
age: 18
city: Beijing
```

在函数体内，kwargs 是一个字典，所以可以用字典的所有方法。

`**kwargs` 也可以在调用时展开字典：

```python
def connect(host, port, timeout):
    print(f"Connecting to {host}:{port} with timeout {timeout}s")

config = {"host": "localhost", "port": 8080, "timeout": 30}
connect(**config)
```

`*` 展开列表或元组，`**` 展开字典。这是对称的设计。

## 参数解包

参数解包是指在调用函数时，使用 `*` 或 `**` 把可迭代对象或字典展开成参数。

`*` 用于展开可迭代对象作为位置参数：

```python
def greet(greeting, name):
    print(f"{greeting}, {name}!")

words = ["Hello", "World"]
greet(*words)  # 等价于 greet("Hello", "World")
```

`**` 用于展开字典作为关键字参数：

```python
def create_user(name, age, email):
    print(f"User: {name}, {age}, {email}")

info = {"name": "Alice", "age": 18, "email": "alice@example.com"}
create_user(**info)
```

解包让函数调用更灵活。你可以把配置放在字典里，然后传给函数；可以把数据列表化，然后传给函数。这在编写通用工具函数时特别有用。

解包也可以组合使用：

```python
def func(a, b, c, d):
    print(a, b, c, d)

args = [1, 2]
kwargs = {"c": 3, "d": 4}
func(*args, **kwargs)  # 1 2 3 4
```

## 强制关键字参数

有时候你希望某些参数必须用关键字参数的形式传递，不能用位置参数。这在参数很多的时候可以提高可读性。

在 Python 3 中，可以在 `*` 后面定义参数，这些参数就变成了强制关键字参数：

```python
def greet(name, *, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Alice")           # 正确
greet("Bob", greeting="Hi")  # 正确，必须用关键字参数
greet("Charlie", "Hi")   # 错误！greeting 不能用位置参数
```

这里的 `*` 表示"结束位置参数"，之后的所有参数都必须用关键字参数传递。

如果函数同时有 `*args` 和强制关键字参数：

```python
def log(*messages, sep=" ", end="\n"):
    output = sep.join(str(m) for m in messages)
    print(output, end=end)

log("Error", "File not found", sep=" | ")  # 必须用关键字参数指定 sep
```

## 参数定义的完整顺序

Python 的参数定义有严格的顺序：

```python
def func(pos1, pos2, *args, kw1, kw2, **kwargs):
    pass
```

完整顺序是：位置参数 → 默认参数 → `*args` → 仅限关键字参数 → `**kwargs`。

如果混用默认参数和可变参数，默认参数必须在可变参数之前：

```python
# 正确的顺序
def func(a, b=2, *args, c, **kwargs):
    pass
```

不遵循这个顺序会导致语法错误。

面试中经常会让候选人写出参数顺序，或者问"为什么这样设计"。答案是：这是 Python 的语法规定，确保解析器能无歧义地识别每个参数的类型。

## 类型提示

Python 3.5 引入了类型提示（Type Hints），允许你标注参数和返回值的类型：

```python
def greet(name: str, times: int = 1) -> str:
    return (f"Hello, {name}! ") * times

print(greet("Alice"))  # Hello, Alice!
```

类型提示不影响程序运行，Python 仍然会执行程序而忽略这些提示。但类型提示能让 IDE 提供更好的代码补全和错误检查，也方便用 mypy 等工具做静态类型检查。

类型提示可以用在任何地方：

```python
from typing import List, Dict, Optional, Callable

def process_items(items: List[int], func: Callable[[int], int]) -> Dict[str, int]:
    return {"count": len(items), "sum": sum(items)}
```

常用的类型提示包括：`List`、`Dict`、`Set`、`Tuple`、`Optional`（可能为 None）、`Callable`（可调用对象）、`Union`（联合类型）。

Python 3.10+ 还支持更简洁的语法：

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

也可以直接用 list、dict 等（不需要 from typing 导入）：

```python
def func(items: list[int], mapping: dict[str, int]) -> list[int]:
    return [mapping.get(x, 0) for x in items]
```

类型提示是 Python 向大型项目发展的一个重要特性，虽然是可选的，但在团队协作和代码维护中非常有价值。

## None 作为默认值与 None 检查

前面讲默认参数时提到了可变默认参数的陷阱，以及用 None 作为默认值的正确做法。这里再深入一点。

用 None 作为默认值后，在函数体内检查是必要的：

```python
def append_item(item, lst=None):
    if lst is None:  # 检查默认值
        lst = []
    lst.append(item)
    return lst
```

但如果参数本身可能是 None 值，这种检查就不够了：

```python
def greet(name: str | None = None):
    if name is None:
        name = "World"
    return f"Hello, {name}!"
```

Python 3.10 引入了 `|` 联合类型语法，表示 name 可以是 str 或 None。

类型提示和默认值可以结合使用：

```python
def find_user(user_id: int, default_name: str = "Guest") -> str:
    # 查找用户的逻辑
    return default_name
```

理解这些高级参数特性，能让你编写出更灵活、更通用的函数。下一章我们会看到，这些特性如何与函数式编程工具结合使用。
