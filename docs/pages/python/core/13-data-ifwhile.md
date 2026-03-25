---
title: 条件、循环与异常
---

# 条件、循环与异常

在任何编程语言里，真正让程序"活起来"的，不是数据类型，而是控制流。

所谓控制流，简单来说，就是程序决定"接下来做什么"的规则。代码从头跑到尾只是理想情况，现实世界中的程序需要根据不同条件做不同的事，需要反复执行某些操作，需要在出错时优雅地处理而不是直接崩溃。Python 提供了三套核心机制来完成这件事：条件判断、循环结构，以及异常处理。

## if 条件判断与真值测试

if 是 Python 中最基本的分支结构，它解决的问题很直接：当某个条件成立时，执行一段代码，否则执行另一段代码。

```python
age = 20

if age >= 18:
    print("成年人")
```

这段代码背后的逻辑并不复杂，但真正值得理解的是：Python 是如何判断"条件成立"的？

在 Java 或 C 语言中，if 后面必须跟一个布尔表达式。但 Python 并不强制这一点。Python 的 if 后面可以是任何对象。解释器会对这个对象进行"真值测试"，判断它是"真"还是"假"。

真值测试的内部机制是这样的：当执行 `if x:` 时，Python 会先尝试调用 `x.__bool__()` 方法；如果这个方法不存在，就尝试调用 `x.__len__()` 方法；如果返回 0，就是假；如果返回非零，就是真。这个机制意味着我们可以让任何自定义对象在条件判断中表现出我们想要的行为。

最常见的例子是空容器。空列表、空字典、空字符串在条件判断中都是假的：

```python
data = []

if data:
    print("有数据")
else:
    print("没有数据")
```

空列表的长度为 0，`__len__()` 返回 0，所以被视为假。这是 Python 设计中一个非常优雅的地方：容器为空和布尔假天然等价，不需要额外的判断。

但这里存在一个容易踩的坑。比如你想判断一个数值是否被赋值：

```python
value = 0

if not value:
    print("值不存在")
```

这段代码会输出"值不存在"，因为 0 在真值测试中被视为假。但 value 实际上是 0，而不是 None。如果 value 真的是 None，这里会输出"值不存在"——但两者的语义完全不同。正确的判断方式是使用 `is None`：

```python
if value is None:
    print("值不存在")
```

这个例子告诉我们：真值测试和"是否存在"是两个不同的概念。0 是有效的值，None 表示没有值。混淆它们会导致难以发现的 bug。

## and 与 or 的短路机制

Python 的逻辑运算符 and 和 or 有两个经常被忽视的特性。第一，它们返回的不是布尔值，而是参与运算的对象本身。第二，它们遵循"短路求值"规则：只要能确定结果，就不再计算右边的表达式。

先看 or 的行为。or 的规则是：如果左操作数为真，就返回左操作数，否则返回右操作数。

```python
a = "" or "default"
print(a)  # default

b = "hello" or "world"
print(b)  # hello
```

这个行为看似奇怪，但实际上非常有用。在工程中，它常被用来设置默认值：

```python
username = input_name or "guest"
```

如果 input_name 是空字符串或者 None，or 会返回 "guest"。但这也带来一个陷阱：如果 input_name 的值是合法的空字符串，你希望保留它而不是替换为默认值，or 就会产生错误的默认值。在大模型开发中处理用户输入时尤其需要注意这一点。

and 的行为则相反：如果左操作数为假，就返回左操作数，否则返回右操作数。

```python
a = "" and "world"
print(a)  # ""

b = "hello" and "world"
print(b)  # world
```

and 常被用于需要确认前置条件的场景：

```python
user = {"active": True}

if user and user["active"]:
    print("允许访问")
```

如果 user 是 None，`user["active"]` 会抛出 TypeError。但因为 and 是短路求值，当 user 为假时，就直接返回 user 而不会继续计算 `"active"]`了，避免了异常。

## 三元表达式与值选择

在很多场景下，我们只是根据一个条件返回不同的值，而不是执行一段代码。这时候写完整的 if/else 语句就显得有些冗长。

```python
age = 20

if age >= 18:
    status = "成年"
else:
    status = "未成年"
```

这个逻辑只是"选择一个值"。Python 提供了条件表达式，也就是俗称的三元表达式：

```python
status = "成年" if age >= 18 else "未成年"
```

它的结构是：`结果A if 条件 else 结果B`。从执行逻辑上看，它等价于普通 if/else，只不过是一个表达式，可以直接赋值、作为函数参数、或者作为返回值。

三元表达式常见于列表推导式中：

```python
numbers = [1, 2, 3, 4]
labels = ["偶数" if n % 2 == 0 else "奇数" for n in numbers]
```

但如果逻辑开始变复杂，比如嵌套多个条件：

```python
result = "A" if score > 90 else "B" if score > 80 else "C"
```

虽然语法上合法，但可读性迅速下降。三元表达式适用于简单的值选择场景，不适用于复杂的流程控制。当判断逻辑变复杂时，应该使用完整的 if/elif/else 语句。

## 字典映射与多条件分支

有时候我们需要根据一个变量的多个取值执行不同的逻辑。最朴素的方式是写一长串 if-elif：

```python
status = 404

if status == 200:
    message = "OK"
elif status == 404:
    message = "Not Found"
elif status == 500:
    message = "Server Error"
else:
    message = "Unknown"
```

当分支数量变多时，这种写法会变得臃肿。而且每次判断都需要比较字符串或者整数，效率也不高。

如果每个分支只是返回一个值，更好的方式是使用字典：

```python
status_map = {
    200: "OK",
    404: "Not Found",
    500: "Server Error"
}

message = status_map.get(status, "Unknown")
```

字典的查找是 O(1) 操作，比顺序比较更高效，而且结构更加清晰。当需要修改逻辑时，只需要修改字典内容，而不需要改动代码结构。

字典映射的方式在大模型开发中非常常见。比如根据模型返回的错误码返回不同的提示信息，或者根据用户的选择执行不同的处理函数。

## match-case：结构化模式匹配

Python 3.10 引入了 match-case，这是结构化模式匹配的实现。它的作用类似其他语言中的 switch，但能力远比 switch 强大。

最简单的用法是匹配字面值：

```python
def check(status):
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case _:
            return "Unknown"
```

这里的 `_` 是通配符，表示匹配任何值，相当于 default。

但 match-case 真正强大的地方在于可以匹配结构。比如匹配元组：

```python
point = (0, 5)

match point:
    case (0, y):
        print(f"x为0，y={y}")
    case (x, 0):
        print(f"y为0，x={x}")
```

第一个 case 匹配 x 为 0 的情况，并把 y 的值绑定给变量 y。第二个 case 匹配 y 为 0 的情况。

还可以匹配字典：

```python
data = {"type": "text", "content": "hello"}

match data:
    case {"type": "text", "content": content}:
        print("文本内容:", content)
```

match-case 从上到下匹配，第一个匹配成功的 case 会被执行，后续 case 不会继续匹配。这和 if-elif 的行为是一致的。

在处理大模型 API 返回的多种格式数据时，match-case 可以让代码更加清晰。比如同一个接口可能返回成功结果、错误结果、或者需要重试的结果，每种结果的格式不同，可以用模式匹配来处理。

## for 循环与迭代器协议

for 循环在 Python 中不是传统 C 语言里的计数循环，而是用于遍历可迭代对象。理解 for 循环的关键在于理解迭代器协议。

当你写 `for item in [1, 2, 3]:` 时，Python 内部发生的事情是：首先调用 `iter([1, 2, 3])` 获取迭代器，然后不断调用 `next(iterator)` 获取下一个元素，直到遇到 `StopIteration` 异常时停止。

```python
for item in [1, 2, 3]:
    print(item)
```

Python 内部等价于：

```python
iterator = iter([1, 2, 3])
while True:
    try:
        item = next(iterator)
        print(item)
    except StopIteration:
        break
```

也就是说，for 循环本质上是在捕获 StopIteration 异常。这解释了为什么生成器可以用 for 循环遍历，因为生成器本身就是迭代器。

range 函数返回一个 range 对象，它也是可迭代的：

```python
for i in range(5):
    print(i)
```

这会输出 0 到 4。range 对象不会一次性生成所有数字，而是在迭代时按需产生，这使得它可以安全地用于非常大的范围而不会占用大量内存。

enumerate 函数在需要同时获取索引和值时非常有用：

```python
fruits = ["apple", "banana", "cherry"]

for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
```

zip 函数可以同时遍历多个序列：

```python
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 88]

for name, score in zip(names, scores):
    print(f"{name}: {score}")
```

## while 循环与无限循环

while 循环的作用是只要条件为真，就重复执行代码块：

```python
count = 0

while count < 3:
    print(count)
    count += 1
```

while 特别适合用于不知道具体迭代次数的场景，比如等待某个条件满足：

```python
import time

retry_count = 0
max_retries = 3

while retry_count < max_retries:
    response = call_api()
    if response.success:
        break
    retry_count += 1
    time.sleep(1)
```

无限循环可以用 `while True:` 实现，但要确保循环体内有 break 条件，否则程序会卡死。无限循环在事件驱动编程、服务器程序中非常常见：

```python
while True:
    event = get_next_event()
    if event is None:
        continue
    process(event)
```

这里用 continue 跳过没有事件的情况，用 break 可以退出循环。

## 异常处理与异常传播

程序运行中难免遇到错误：除零错误、访问不存在的键、打开不存在的文件。这些错误如果不能妥善处理，程序就会崩溃。Python 的异常处理机制让我们可以在错误发生时捕获并处理，而不是让程序直接崩溃。

最基本的结构是 try-except：

```python
try:
    result = int("abc")
except ValueError:
    print("转换失败")
```

当 try 块中的代码抛出 ValueError 时，执行会跳转到 except 块。处理完后，程序继续执行 except 块之后的代码。

异常有一个重要的特性：它们会沿着调用栈向上传播：

```python
def inner():
    raise ValueError("错误信息")

def middle():
    inner()

def outer():
    middle()

try:
    outer()
except ValueError as e:
    print(f"捕获到异常: {e}")
```

当 inner 函数抛出异常时，Python 会一级级往上找except 子句，直到找到匹配的异常处理器。这个传播机制让我们可以在合适的层级统一处理错误，而不是在每个函数中都写错误处理代码。

except 子句可以捕获特定类型的异常，也可以一次捕获多种类型：

```python
try:
    result = int("abc")
except ValueError:
    print("值错误")
except TypeError:
    print("类型错误")
except Exception as e:
    print(f"其他错误: {e}")
```

except 是从上到下匹配的，第一个匹配成功的会执行。所以更具体的异常要写在前面，更宽泛的异常要写在后面。如果把 Exception 写在最前面，它会捕获所有异常，后面的具体异常处理就永远不会被执行了。

## finally 与资源的释放

finally 块无论是否发生异常都会执行，常用于确保资源被正确释放：

```python
try:
    file = open("data.txt", "r", encoding="utf-8")
    content = file.read()
except FileNotFoundError:
    content = ""
finally:
    file.close()
```

无论读取是否成功，finally 都会确保文件被关闭。这在处理数据库连接、网络连接、锁等资源时特别重要。

但这种写法有一个问题：如果 try 块中的代码抛出异常，finally 中的 file.close() 也会执行，但如果 close() 本身也抛出异常，原始异常就会被掩盖。更好的方式是使用 with 语句。

## with 语句与上下文管理器

with 语句是 Python 中处理资源的标准方式，它确保资源在使用后被正确清理，即使发生异常也不例外：

```python
with open("data.txt", "r", encoding="utf-8") as file:
    content = file.read()
```

这个代码在功能上等价于前面的 try-finally 结构，但更加简洁和安全。with 语句的原理是上下文管理器协议：对象必须实现 `__enter__` 和 `__exit__` 方法。with 语句会在进入代码块前调用 `__enter__`，在退出代码块时调用 `__exit__`。

with 不仅能用于文件，几乎任何需要"获取-释放"配对的场景都可以用 with。比如线程锁：

```python
import threading

lock = threading.Lock()

with lock:
    # 临界区代码
    shared_resource += 1
# 锁在这里自动释放
```

这比手动调用 lock.acquire() 和 lock.release() 更安全，因为即使临界区代码抛出异常，锁也会被正确释放。

在异步编程中，async with 用于处理异步资源的获取和释放：

```python
async with asyncio.Lock():
    # 临界区代码
    await do_something()
```

## 自定义异常与异常层次

Python 定义了大量的内置异常，从最通用的 Exception 到具体的 ValueError、TypeError、KeyError 等等。在大型项目中，自定义异常可以让错误处理更加清晰：

```python
class ValidationError(Exception):
    pass

class ModelNotFoundError(Exception):
    pass

def validate_input(data):
    if "prompt" not in data:
        raise ValidationError("缺少必填字段 prompt")
    if len(data["prompt"]) > 1000:
        raise ValidationError("prompt 长度不能超过 1000")
```

自定义异常应该继承自 Exception 或者更具体的内置异常。自定义异常的层次结构应该和业务逻辑的层次结构一致。比如 ValidationError 和 ModelNotFoundError 都继承自 Exception，但它们代表不同类型的错误，可以在不同层级被捕获和处理。

理解条件判断的真值模型、循环背后的迭代器协议、异常的传播和捕获机制，是写出健壮 Python 代码的基础。这些机制看似底层，但它们决定了程序在各种边界情况下的行为。掌握它们，才能在面对复杂的业务逻辑和出错场景时写出正确且优雅的代码。
