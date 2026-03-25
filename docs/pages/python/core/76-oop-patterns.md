---
title: 设计模式
---

# 设计模式

设计模式是软件工程中总结出来的一套通用问题的解决方案。它们不是具体的代码，而是解决问题的思路和原则。

虽然 Python 的设计模式和其他语言类似，但因为 Python 本身的特性（如一等函数、装饰器、元类），有些模式的实现方式会有所不同。

这一章讲几个 Python 中常用的设计模式：单例模式、工厂模式、观察者模式、策略模式，以及 Python 特有的 OOP 设计原则。

理解设计模式不是死记硬背，而是理解它们解决的问题和解决思路。在实际编程中，应该根据问题选择合适的模式，而不是为了用模式而用模式。

## 单例模式

单例模式确保一个类只有一个实例，并提供一个全局访问点。

Python 中实现单例有多种方式：

方式一：使用模块

```python
# singleton.py
class Singleton:
    pass

singleton = Singleton()
```

```python
from singleton import singleton
```

因为模块在进程中只会被导入一次，模块级别的对象就是单例。

方式二：使用 `__new__`

```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

a = Singleton()
b = Singleton()
print(a is b)  # True
```

`__new__` 在对象创建前被调用，通过覆盖它可以控制对象的创建过程。

方式三：使用装饰器

```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class MyClass:
    pass
```

## 工厂模式

工厂模式用于创建对象，而不直接用 `__init__`。当你需要创建不同类型的对象，或者创建过程很复杂时，工厂模式很有用。

简单工厂：

```python
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown animal type: {animal_type}")

animal = AnimalFactory.create_animal("dog")
print(animal.speak())  # "Woof!"
```

工厂方法模式：

```python
class Animal(metaclass=ABCMeta):
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory(ABC):
    @abstractmethod
    def create_animal(self):
        pass

class DogFactory(AnimalFactory):
    def create_animal(self):
        return Dog()

factory = DogFactory()
animal = factory.create_animal()
print(animal.speak())  # "Woof!"
```

## 观察者模式

观察者模式定义了对象之间的一对多依赖关系，当一个对象改变时，所有依赖它的对象都会收到通知。

```python
class Observer:
    def update(self, message):
        pass

class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

class NewsAgency(Subject):
    def __init__(self):
        super().__init__()
        self._news = ""

    def set_news(self, news):
        self._news = news
        self.notify(news)

class NewsChannel(Observer):
    def __init__(self, name):
        self.name = name

    def update(self, message):
        print(f"{self.name} received: {message}")

agency = NewsAgency()
channel1 = NewsChannel("Channel 1")
channel2 = NewsChannel("Channel 2")

agency.attach(channel1)
agency.attach(channel2)
agency.set_news("Breaking news!")
```

输出：

```
Channel 1 received: Breaking news!
Channel 2 received: Breaking news!
```

观察者模式在事件处理系统、消息推送等场景中广泛使用。

## 策略模式

策略模式定义了一系列算法，把它们一个个封装起来，并让它们可以互换。

```python
from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class QuickSort(SortStrategy):
    def sort(self, data):
        print("Using QuickSort")
        return sorted(data)  # 简化实现

class MergeSort(SortStrategy):
    def sort(self, data):
        print("Using MergeSort")
        return sorted(data)  # 简化实现

class Sorter:
    def __init__(self, strategy):
        self._strategy = strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    def sort(self, data):
        return self._strategy.sort(data)

sorter = Sorter(QuickSort())
sorter.sort([3, 1, 2])

sorter.set_strategy(MergeSort())
sorter.sort([3, 1, 2])
```

策略模式让算法的选择可以独立于使用它的代码。

## Pythonic 的 OOP 设计原则

除了经典的设计模式，Python 还有一些特有的 OOP 设计原则。

"Mixins over inheritance"：优先使用组合而不是继承。Mixin 是一种通过多重继承添加功能的方式。

"Protocols over inheritance"：Python 推崇"鸭子类型"，关注对象有什么方法，而不是它是什么类型。

"Composition over inheritance"：组合优于继承。如果类之间的关系是"has-a"而不是"is-a"，用组合。

"Simple over complex"：简单优于复杂。不要过度设计，不要为了用模式而用模式。

```python
# 过度设计
class Dog(Animal, Runnable, Barkable, ...):
    pass

# 更 Pythonic 的方式
class Dog:
    def __init__(self):
        self.bark_behavior = BarkBehavior()
        self.run_behavior = RunBehavior()
```

## 常见设计模式的 Python 实现

| 模式 | Python 实现 |
|------|-----------|
| 单例 | 模块、`__new__`、装饰器 |
| 工厂 | 简单工厂、工厂方法、抽象工厂 |
| 观察者 | 手写、contextlib.contextmanager、asyncio |
| 策略 | 函数作为参数、策略类 |
| 装饰器 | @decorator、functools.wraps |
| 迭代器 | `__iter__`、`__next__` |

## 常见误区

第一个误区是过度使用设计模式。设计模式是工具，不是目的。不要为了用模式而用模式，简单直接的代码往往更好。

第二个误区是不考虑 Python 的特性就用模式。Python 有很多内置特性，比如一等函数、装饰器、生成器等。很多其他语言需要用模式解决的问题，Python 可以用更简单的方式解决。

第三个误区是死记硬背模式的实现。理解模式解决的问题和解决思路，比记住具体实现更重要。

第四个误区是忽略代码的可读性。设计模式可能会让代码变得更复杂。如果模式的实现比简单代码更难理解，就不应该使用。

理解设计模式的原则和思路，根据实际问题选择合适的解决方案，写出简洁可读的代码，是 Python 面向对象编程的更高境界。
