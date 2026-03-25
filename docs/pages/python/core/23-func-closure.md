---
title: 闭包与装饰器
---

# 闭包与装饰器

闭包和装饰器是 Python 函数中最容易让人困惑的部分，但同时也是最强大的部分。很多初学者学到这里会觉得"这有什么用"，但当你真正理解它们之后，会发现它们是 Python 框架的基石——Flask 的路由装饰器、Django 的视图装饰器、日志装饰器、性能计时装饰器，到处都是装饰器的身影。

理解闭包是理解装饰器的前提。闭包让内部函数能够"记住"外部作用域的变量，即使外部函数已经返回。装饰器则是闭包最典型的应用——它接受一个函数作为参数，返回一个新函数来增强原函数的功能。

这一章解决的问题是：怎么让函数"记住"自己的运行环境，怎么用函数来增强另一个函数。

## 闭包：函数记住环境

先看一个例子：

```python
def outer():
    x = 10
    def inner():
        return x
    return inner

f = outer()
print(f())  # 输出什么？
```

执行 `outer()` 返回的是 inner 函数本身，不是调用 inner。当 `f()` 被执行时，它返回了 x 的值 10。

这里有一个看似奇怪的问题：outer 函数已经执行完了，它的局部变量 x 应该被销毁了，为什么 inner 函数还能访问到 x？

这就是闭包的作用。当 inner 函数被创建时，它"记住"了创建时所在的环境，即 outer 函数的作用域。inner 函数携带了一个指向 outer 作用域的引用，这个引用确保了 x 即使在 outer 返回后也不会被垃圾回收。

可以用 `__closure__` 属性来验证：

```python
def outer():
    x = 10
    def inner():
        return x
    return inner

f = outer()
print(f.__closure__)  # (<cell at ...>,)
print(f.__closure__[0].cell_contents)  # 10
```

`__closure__` 是一个元组，包含 inner 函数记住的所有外部变量。每个元素是一个 cell 对象，通过 `cell_contents` 可以访问变量的值。

## 闭包的应用：工厂函数

闭包最常见的应用是创建工厂函数。工厂函数返回的是记住特定参数的函数：

```python
def make_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

make_multiplier 返回的 multiplier 函数记住了创建时的 factor 值。所以 double 和 triple 虽然由同一个函数生成，但记住了不同的 factor。

这比直接传递参数有什么优势？优势在于：调用者不需要关心内部实现，只需要调用返回的函数就可以了。比如你想给 API 调用添加重试逻辑：

```python
def make_retry(func, max_attempts=3):
    def wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
    return wrapper
```

## 装饰器基础

装饰器本质上就是一个接受函数作为参数并返回新函数的闭包。装饰器的目的是在不修改原函数的情况下，给函数添加新的功能。

先看一个最简单的装饰器：

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before calling function")
        result = func(*args, **kwargs)
        print("After calling function")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

输出是：

```
Before calling function
Hello!
After calling function
```

`@my_decorator` 是装饰器的语法糖，它等价于 `say_hello = my_decorator(say_hello)`。

装饰器的工作流程是：接受原函数作为参数，在 wrapper 里调用原函数并在调用前后添加新功能，返回新函数替换原函数。

## 装饰器与函数签名

一个常见的陷阱是：装饰器会替换原函数的元数据：

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def add(a, b):
    """Adds two numbers"""
    return a + b

print(add.__name__)  # wrapper，不是 add
print(add.__doc__)   # None，不是 "Adds two numbers"
```

这是因为装饰器返回的是 wrapper 函数，wrapper 的元数据覆盖了原函数。Python 的 functools 模块提供了 `wraps` 装饰器来解决这个问题：

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # 保留原函数的元数据
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def add(a, b):
    """Adds two numbers"""
    return a + b

print(add.__name__)  # add
print(add.__doc__)   # Adds two numbers
```

`@wraps(func)` 会把原函数的 `__name__`、`__doc__`、`__module__` 等属性复制到 wrapper 函数上。这是一个标准做法，任何装饰器都应该使用 `@wraps`。

## 带参数的装饰器

装饰器工厂模式允许装饰器接受参数：

```python
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
```

`@repeat(times=3)` 等价于 `greet = repeat(times=3)(greet)`。这里 repeat 返回的是装饰器，装饰器再接受函数并返回 wrapper。

不带参数的装饰器和使用 `functools.wraps` 的写法是：

```python
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 装饰逻辑
        return func(*args, **kwargs)
    return wrapper
```

带参数的装饰器需要多一层包装：

```python
def my_decorator(arg1, arg2):  # 装饰器参数
    def decorator(func):         # 函数参数
        @wraps(func)
        def wrapper(*args, **kwargs):  # 实际包装
            # 使用 arg1, arg2 和 func
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

## 多层装饰器

多个装饰器可以叠加使用：

```python
@decorator1
@decorator2
def func():
    pass
```

这等价于 `func = decorator1(decorator2(func))`。

装饰器的执行顺序是从下到上的。decorator2 先作用在 func 上，然后 decorator1 作用在结果上。执行顺序是从外到内的，先执行 decorator1 的 wrapper 开头部分，然后 decorator2 的 wrapper 开头部分，然后原函数，然后 decorator2 的 wrapper 结尾部分，最后 decorator1 的 wrapper 结尾部分。

理解装饰器的执行顺序对于调试很重要。如果装饰器 A 在装饰器 B 上面（代码中先写 A），那么 A 会先执行。

## 内置装饰器

Python 提供了一些常用的内置装饰器。

`@staticmethod` 把方法变成静态方法，静态方法不需要实例就可以调用：

```python
class MyClass:
    @staticmethod
    def static_method():
        print("This is a static method")

MyClass.static_method()  # 不需要实例
```

`@classmethod` 把方法变成类方法，第一个参数是类本身而不是实例：

```python
class MyClass:
    @classmethod
    def class_method(cls):
        print(f"Called from {cls.__name__}")

MyClass.class_method()  # 输出 Called from MyClass
```

`@property` 把方法变成属性，可以像访问属性一样访问方法：

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def diameter(self):
        return self._radius * 2

circle = Circle(5)
print(circle.diameter)  # 10，像属性一样访问
```

`@functools.wraps` 前面已经讲过，用于保留被装饰函数的元数据。

## 装饰器的常见应用场景

装饰器最常见的应用场景包括：

日志记录：

```python
import logging

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {result}")
        return result
    return wrapper
```

性能计时：

```python
import time

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper
```

权限检查：

```python
def requires_admin(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_admin():
            raise PermissionError("Admin required")
        return func(*args, **kwargs)
    return wrapper
```

缓存（记忆化）：

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

装饰器是 Python 中最优雅的特性之一，理解它需要先理解闭包。闭包让函数记住外部环境，装饰器则是闭包的一种高级应用——用函数来增强函数。
