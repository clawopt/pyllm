---
title: 高级特性
---

# 高级特性

这一章讲 Python 面向对象的一些高级特性：描述器协议、元类基础、`__new__` 与 `__init__` 的区别、类装饰器、以及 ABC 抽象基类。

这些内容对于理解 Python 框架的内部机制很重要。比如 Django 的 ORM、Flask 的路由装饰器、SQLAlchemy 的映射器，都用到了这些高级特性。理解它们，才能更好地理解这些框架的工作原理。

## 描述器协议

描述器是 Python 面向对象中较高级的概念。它是一种协议，定义了对象属性的访问行为。

实现了 `__get__` 和 `__set__`（或 `__delete__`）的对象，称为数据描述器：

```python
class Descriptor:
    def __get__(self, obj, objtype=None):
        return f"Getting value: {obj._value}"

    def __set__(self, obj, value):
        obj._value = value

    def __delete__(self, obj):
        del obj._value
```

描述器在类属性中定义时，会拦截属性的访问：

```python
class MyClass:
    attr = Descriptor()

    def __init__(self):
        self._value = 0
```

```python
obj = MyClass()
print(obj.attr)   # "Getting value: 0"
obj.attr = 42
print(obj.attr)   # "Getting value: 42"
```

`@property` 就是一个描述器。实际上，`@property` 装饰的函数会被转换成描述器对象。

描述器在 Django 的 ORM 中广泛应用。比如 `CharField`、`IntegerField` 等都是描述器，它们负责把数据库值转换成 Python 对象，以及把 Python 对象转换回数据库值。

## 元类基础

元类是"类的类"。普通类定义对象怎么创建，元类定义类怎么创建。

```python
class MyClass:
    pass

print(type(MyClass))  # <class 'type'>
```

所有类都是 `type` 的实例。`type` 本身是元类。

可以自定义元类，继承自 `type`：

```python
class Meta(type):
    def __new__(cls, name, bases, attrs):
        print(f"Creating class: {name}")
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=Meta):
    pass
```

运行时会输出：

```
Creating class: MyClass
```

元类的 `__new__` 方法在类定义创建时被调用，可以修改类的属性。

元类的常见用途包括：自动注册类到某个注册表、为类添加属性或方法、实现ORM映射等。

Flask 的路由系统就用到了元类。当定义视图函数时，Flask 用元类来自动注册路由到应用中。

## __new__ 与 __init__ 的区别

`__new__` 是创建实例的方法，`__init__` 是初始化实例的方法。

`__new__` 的返回值是新创建的实例，`__init__` 的返回值被忽略。

```python
class MyClass:
    def __new__(cls, *args, **kwargs):
        print("__new__ called")
        return super().__new__(cls)

    def __init__(self, value):
        print("__init__ called")
        self.value = value
```

```python
obj = MyClass(42)
```

输出：

```
__new__ called
__init__ called
```

什么时候需要定义 `__new__`？主要场景包括：

1. 不可变对象的创建（如 `str`、`tuple` 的子类）
2. 单例模式
3. 实现工厂模式，返回不同类型的实例

```python
# 不可变对象的子类需要 __new__
class UpperStr(str):
    def __new__(cls, value):
        return super().__new__(cls, value.upper())
```

## 类装饰器

类装饰器类似函数装饰器，但作用在类上。类装饰器是一个函数，接收类作为参数，返回修改后的类。

```python
def add_greeting(cls):
    cls.greeting = "Hello!"
    return cls

@add_greeting
class Person:
    pass

print(Person.greeting)  # "Hello!"
```

类装饰器可以修改类的行为，比如添加属性、方法，或者包装方法。

```python
def log_creation(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        print(f"Creating {cls.__name__}")
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls

@log_creation
class Dog:
    def __init__(self, name):
        self.name = name

dog = Dog("旺财")  # 输出：Creating Dog
```

类装饰器的执行顺序是从下到上，和函数装饰器一致。

## ABC 抽象基类

抽象基类（Abstract Base Class）定义了一组方法子类必须实现。ABC 不能被实例化。

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass
```

```python
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)
```

```python
shape = Shape()  # TypeError: Can't instantiate abstract class Shape
rect = Rectangle(3, 4)  # OK
```

抽象方法用 `@abstractmethod` 装饰器标记。如果子类没有实现所有抽象方法，就不能被实例化。

ABC 的用途包括：定义接口契约、强制子类实现某些方法、让代码更清晰。

## mixin 模式

Mixin 是一种设计模式，通过多重继承来添加功能。

Mixin 类包含的方法是为了给其他类添加功能的，不单独使用。

```python
class LogMixin:
    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")

class Service(LogMixin):
    def run(self):
        self.log("Running service")
```

```python
service = Service()
service.run()  # [Service] Running service
```

Mixin 应该是可组合的，每个 Mixin 只做一件事。比如 Django 的 `UserCreationForm` 继承了 `UserCreationMixin` 和 `ModelForm`，`UserCreationMixin` 只负责用户创建相关的逻辑。

## 常见误区

第一个误区是滥用元类。元类是高级特性，大多数情况下不需要使用。如果你在考虑用元类解决问题，先想想有没有更简单的方案。

第二个误区是不理解 `__new__` 和 `__init__` 的分工。`__new__` 负责创建对象，`__init__` 负责初始化对象。大多数情况下只需要定义 `__init__`。

第三个误区是忘记 ABC 的抽象方法。如果子类没有实现所有抽象方法，会在实例化时报错。这有时候会让人困惑。

第四个误区是过度使用 Mixin。Mixin 虽然方便，但多重继承会让代码变得复杂。如果 Mixin 之间有依赖关系，问题会更复杂。

理解这些高级特性，能让你更好地理解 Python 框架的内部机制，写出更优雅的面向对象代码。
