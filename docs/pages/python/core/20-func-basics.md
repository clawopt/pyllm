---
title: 函数基础
---

# 函数基础

当你开始写代码时，最先接触的可能是变量、循环、条件判断这些基本语法。然后你会发现，随着代码量增加，同样的逻辑会被反复复制粘贴很多遍。比如你要计算一个数组的平均值，可能会在代码里写 `sum(nums) / len(nums)` 十几次。每次都这样写不仅繁琐，而且如果哪一天需要修改计算方式，就得改十几个地方。

函数就是为了解决这个问题而诞生的。函数就像是给你的代码提供一个"工具箱"，你可以把一段反复使用的逻辑打包起来，给它起个名字，以后想用的时候只需要调用这个名字就行了。这不只是为了省事，更是为了代码的可维护性、可读性，以及后期的扩展性。

在 Python 里，函数是"一等公民"，这意味着函数和其他数据类型（整数、字符串、列表）没有本质区别——函数可以赋值给变量，可以作为参数传给另一个函数，可以从函数里返回，可以放进数据结构里。这种灵活性让 Python 的函数远不止"封装代码"这么简单，它是我们理解装饰器、生成器、异步编程的基础。

## 定义函数

定义函数使用 `def` 关键字，后面跟着函数名、参数列表（如果有的话），最后是冒号。函数体需要缩进。

```python
def greet(name):
    print(f"Hello, {name}!")
```

函数名遵循变量命名规则：字母、数字、下划线，但不能以数字开头。函数名应该具有描述性，让人一眼看出这个函数是做什么的。比如 `calculate_average` 比 `calc` 更清晰，`find_max` 比 `fm` 更容易理解。

调用函数只需要使用函数名加括号：

```python
greet("Alice")  # 输出: Hello, Alice!
```

函数可以没有参数，也可以有多个参数。参数之间用逗号分隔：

```python
def add(a, b):
    return a + b

result = add(3, 5)  # result 是 8
```

如果函数没有 return 语句，或者 return 后面没有值，函数会返回 None：

```python
def say_hello():
    print("Hello!")

result = say_hello()  # 输出 Hello!
print(result)          # 输出 None
```

## 值传递还是引用传递

理解函数的关键在于理解函数调用时发生了什么。当执行 `add(2, 3)` 时，Python 内部经历了一个精确的流程：首先创建一个新的栈帧，这是函数独立运行的基础；然后把实参 2 和 3 绑定到这个栈帧中的形参变量 a 和 b；接着执行函数体的字节码；最后如果遇到 return 语句，就把返回值传递给调用方，然后销毁这个栈帧。

参数绑定这一步尤其重要。很多从其他语言转过来的人会问：Python 到底是值传递还是引用传递？这个问题本身在 Python 语境下就是有问题的，因为 Python 既不是值传递也不是引用传递，而是"对象引用传递"。这意味着当调用函数时，函数参数会获得一个指向原对象的新引用，而不是原对象本身，也不是原变量的副本。

看一个具体的例子来理解这个机制：

```python
def modify(x):
    x.append(100)

lst = [1, 2]
modify(lst)
print(lst)
```

输出的结果是 `[1, 2, 100]`。这个过程是这样的：调用 modify 时，参数 x 获得了指向列表对象 `[1, 2]` 的一个新引用。当执行 `x.append(100)` 时，修改的是列表对象本身，所有指向这个列表对象的引用都会看到变化，包括外面的 lst。如果把 `x.append(100)` 换成 `x = x + [100]`，结果就不一样了——这是因为 `x + [100]` 会创建一个新的列表对象，然后让 x 指向这个新列表，而外面的 lst 仍然指向原来的列表。

## 位置参数与关键字参数

调用函数时，实参可以按照位置一一对应传给形参，这叫做位置参数：

```python
def subtract(a, b):
    return a - b

result = subtract(10, 3)  # a=10, b=3，结果是 7
```

也可以使用关键字参数，直接指定形参的名字：

```python
result = subtract(b=3, a=10)  # 明确指定 a=10, b=3，结果是 7
```

关键字参数的好处是不用记住参数的顺序，尤其在参数很多的时候。而且如果混用位置参数和关键字参数，位置参数必须在前：

```python
def connect(host, port, timeout):
    pass

# 正确：位置参数在前，关键字参数在后
connect("localhost", 8080, timeout=30)

# 错误：关键字参数不能在位置参数之前
connect(host="localhost", 8080, 30)  # SyntaxError
```

## 默认参数

给参数指定默认值，这个参数就变成可选的了：

```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Alice")           # 使用默认 greeting: Hello, Alice!
greet("Bob", "Hi")       # 自定义 greeting: Hi, Bob!
```

默认参数让函数更灵活，调用者可以根据需要选择是否提供某些参数。这在很多场景下都很有用，比如连接数据库的函数，大部分时候使用默认配置就够了，只有特殊情况下才需要手动指定参数。

但默认参数有一个著名的陷阱，这个陷阱坑过无数 Python 开发者。先看一个看起来完全正常的例子：

```python
def append_item(item, lst=[]):
    lst.append(item)
    return lst

print(append_item(1))  # 输出什么？
print(append_item(2))  # 输出什么？
```

第一次调用输出 `[1]`，这是预期的。但第二次调用输出的不是 `[2]`，而是 `[1, 2]`。这看起来像是 Python 的 bug，但实际上是特性。

原因是：默认参数在函数**定义时**计算，而不是在每次**调用时**计算。对于 `lst=[]` 这个默认参数，Python 在定义函数时就创建了一个空列表对象，并让 lst 指向这个对象。以后每次调用函数时，如果没有传入 lst 参数，就会使用这个已经创建好的列表对象。所以第一次调用修改了这个共享对象，第二次调用继续修改它。

正确的做法是把默认参数设为 None：

```python
def append_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

print(append_item(1))  # [1]
print(append_item(2))  # [2]，这次是独立的新列表
```

这个模式是 Python 中处理默认参数的标准做法，在阅读他人代码时也会经常看到。

## 返回值

函数可以用 return 语句返回值。return 后面的值会作为函数的结果返回给调用者：

```python
def multiply(a, b):
    return a * b

result = multiply(4, 5)  # result 是 20
```

函数可以返回多个值，本质上返回的是一个元组：

```python
def divide(a, b):
    quotient = a // b
    remainder = a % b
    return quotient, remainder

q, r = divide(10, 3)  # q=3, r=1
result = divide(10, 3) # result 是 (3, 1)
```

返回多个值时，Python 实际上是创建了一个元组，然后返回这个元组。调用者可以用多个变量接收（自动解包），也可以只用一个变量接收（得到元组）。

## 作用域：LEGB 规则

当你在函数内部引用一个变量时，Python 会按照一套叫做 LEGB 的规则来查找这个变量。LEGB 代表四种作用域：Local（局部）、Enclosing（嵌套）、Global（全局）、Built-in（内置）。

```python
x = 10  # 全局作用域

def outer():
    x = 20  # 嵌套作用域
    def inner():
        print(x)  # 输出什么？
    inner()

outer()  # 输出 20
```

当 inner 函数执行 `print(x)` 时，Python 先在 inner 的局部作用域找 x，没找到；然后在外层函数 outer 的作用域找，找到了 x=20，所以使用了 20。

如果 inner 里面有自己的 x：

```python
x = 10

def outer():
    x = 20
    def inner():
        x = 30
        print(x)
    inner()

outer()  # 输出 30
```

这次 inner 自己的局部作用域有 x，所以使用了 30。

全局作用域的变量在函数内部可以直接读取，但不能直接修改：

```python
x = 10

def change():
    x = 20  # 这是在局部作用域创建新变量，不是修改全局的 x
    print(x)

change()  # 输出 20
print(x)  # 输出 10，全局的 x 没变
```

如果要修改全局变量，需要用 global 关键字：

```python
x = 10

def change():
    global x
    x = 20

change()
print(x)  # 输出 20
```

nonlocal 关键字用于修改嵌套作用域的变量：

```python
def outer():
    x = 10
    def inner():
        nonlocal x
        x = 20
    inner()
    print(x)

outer()  # 输出 20
```

理解作用域规则是理解闭包、装饰器等高级特性的基础。很多看似奇怪的问题，比如"为什么函数里看不到外面的变量"，本质上都是作用域规则在起作用。

## 函数是对象

在 Python 中，函数和其他对象没有本质区别。函数本身就是一个对象，函数名只是一个指向这个对象的变量。这句话的含义很深——当你用 def 定义一个函数时，你实际上是创建了一个函数对象，然后把这个对象绑定到函数名。

```python
def add(a, b):
    return a + b

print(type(add))  # <class 'function'>
print(add)         # <function add at 0x...>
```

既然函数是对象，就可以像其他对象一样被操作。可以赋值给变量：

```python
def add(a, b):
    return a + b

f = add  # f 和 add 指向同一个函数对象
print(f(3, 5))  # 输出 8
```

可以传给其他函数：

```python
def apply(func, x, y):
    return func(x, y)

result = apply(add, 3, 5)  # 输出 8
```

可以从函数里返回：

```python
def create_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = create_multiplier(2)
print(double(5))  # 输出 10
```

函数作为一等公民，是 Python 灵活性的重要来源，也是函数式编程、装饰器、生成器等高级特性的基础。
