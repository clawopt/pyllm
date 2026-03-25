# 上下文管理器

上下文管理器是Python中处理资源获取与释放的优雅方式。无论是文件操作、数据库连接还是锁的获取，上下文管理器都能确保资源被正确清理，即使在发生异常的情况下也不例外。

## 传统方式的困境

在没有上下文管理器之前，我们通常这样处理文件的关闭：

```python
f = open('data.txt', 'w')
try:
    f.write('Hello')
finally:
    f.close()
```

这种模式虽然能保证文件被关闭，但代码冗长，每次都要写`try-finally`结构。当需要同时管理多个资源时，代码会变得嵌套过深，难以阅读。

```python
# 同时管理多个资源
f1 = None
f2 = None
try:
    f1 = open('input.txt', 'r')
    f2 = open('output.txt', 'w')
    # 处理逻辑...
finally:
    if f1:
        f1.close()
    if f2:
        f2.close()
```

`try-finally`的另一个问题在于，清理代码与业务逻辑混在一起，代码的可读性随资源数量的增加而急剧下降。

## with语句的魅力

`with`语句将资源的获取和释放封装为一个代码块，使我们得以用声明式的方式处理资源：

```python
with open('data.txt', 'w') as f:
    f.write('Hello')
# 文件自动关闭
```

这段代码的语义清晰明了：在`with`代码块内部，`f`是已经打开的文件对象；一旦离开这个代码块，文件就会自动关闭。无论代码块是正常结束还是因异常退出，清理工作都会执行。

`with`语句不仅适用于文件操作，还能用于任何实现了上下文管理协议的对象。

## 实现上下文管理器

实现一个上下文管理器有两种方式：定义`__enter__`和`__exit__`方法，或者使用`contextlib`模块提供的装饰器。

### 双下划线方法

最简单的自定义上下文管理器是一个定义了这三个特殊方法的类：

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False
```

`__enter__`方法在进入`with`代码块时被调用，它的返回值会绑定到`as`后面的变量上。`__exit__`方法在离开代码块时被调用，接受三个参数：异常类型、异常值和追踪对象。如果方法返回`True`，则异常被抑制；返回`False`或`None`则异常会继续传播。

```python
with FileManager('test.txt', 'w') as f:
    f.write('content')
# 自动关闭文件
```

### contextlib装饰器

`contextlib`模块提供了更简便的方式来创建上下文管理器。使用`@contextmanager`装饰器可以将一个生成器函数转换为上下文管理器：

```python
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()
```

在这个模式中，`yield`之前的代码相当于`__enter__`的职责，`yield`之后的代码相当于`__exit__`的职责。这种方式避免了定义一个完整的类，代码更简洁。

```python
with file_manager('data.txt', 'w') as f:
    f.write('hello')
```

生成器版本的实现特别适合简单的资源封装场景。但需要注意，如果`yield`之前发生异常，`finally`中的清理代码不会执行。

## 异常与退出处理

`__exit__`方法的三个参数让我们能够了解代码块内部是否发生了异常，以及异常的详细信息：

```python
class ErrorTracker:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print("正常退出")
        else:
            print(f"异常类型: {exc_type.__name__}")
            print(f"异常信息: {exc_val}")
        return False
```

默认情况下，`__exit__`返回`False`，这意味着任何未处理的异常都会继续向外传播。如果我们返回`True`，就可以抑制异常：

```python
class SilentError:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

with SilentError():
    raise ValueError("这会被沉默")
print("继续执行")
```

抑制异常要谨慎使用。常见的合理场景是，当清理工作已经完成、继续传播异常没有意义时。

## 嵌套上下文管理器

`with`语句可以嵌套使用，每一层都会正确调用其`__exit__`方法：

```python
with open('outer.txt', 'w') as outer:
    with open('inner.txt', 'w') as inner:
        outer.write('outer')
        inner.write('inner')
```

Python 3.10引入了并行上下文管理器，允许在单个`with`语句中打开多个资源：

```python
with open('input.txt', 'r') as source, \
     open('output.txt', 'w') as dest:
    dest.write(source.read())
```

这种写法比嵌套更加清晰，资源会按照与声明相反的顺序被关闭。

## 实用场景

### 数据库连接

数据库连接是上下文管理器的经典应用场景：

```python
import sqlite3

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()
        return False
```

这个实现不仅关闭连接，还在正常退出时提交事务，在异常退出时回滚事务。

### 线程锁

在多线程编程中，上下文管理器可以确保锁的正确释放：

```python
import threading

lock = threading.Lock()

with lock:
    # 临界区代码
    pass
# 锁自动释放
```

这比手动调用`lock.acquire()`和`lock.release()`更加安全，因为即使临界区抛出异常，锁也会被正确释放。

### 临时文件

使用上下文管理器处理临时文件：

```python
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    temp_file = os.path.join(tmpdir, 'data.txt')
    with open(temp_file, 'w') as f:
        f.write('temporary')
# 目录及其内容被自动删除
```

`tempfile.TemporaryDirectory`是一个内置的上下文管理器，它在退出时自动删除临时目录及其所有内容。

## 常见误区

第一个常见错误是在`__enter__`中打开资源，却在`__exit__`中试图重新打开：

```python
class Broken:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        self.file = None  # 不要在这里关闭后再打开
```

另一个错误是忘记处理`__enter__`可能失败的情况：

```python
@contextmanager
def risky():
    resource = acquire_expensive_resource()  # 可能失败
    try:
        yield resource
    finally:
        resource.release()
```

`acquire_expensive_resource`失败会导致`yield`永远无法执行，资源泄漏。使用`@contextmanager`时，将获取资源的代码放在`try`块内可以避免这个问题：

```python
@contextmanager
def safe():
    try:
        resource = acquire_expensive_resource()
        yield resource
    finally:
        if resource:
            resource.release()
```

## 面试关注点

面试中关于上下文管理器的常见问题包括：为什么`with`语句比`try-finally`更好？`__exit__`方法的返回值有什么意义？如何实现一个线程安全的上下文管理器？

理解上下文管理器的设计意图很重要：它将资源的获取、使用和释放封装为一个原子操作，确保资源不会泄漏。优秀的上下文管理器应该做到：获取资源时做好最坏准备，释放资源时确保无论何种情况都执行清理。

对于`contextlib`模块，高级面试题可能涉及`@contextmanager`的陷阱，比如生成器中发生异常时`finally`块是否能保证执行，以及如何正确处理嵌套的生成器上下文管理器。
