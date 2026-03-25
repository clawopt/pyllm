---
title: 类与对象基础
---

# 类与对象基础

当你第一次听说"面向对象编程"时，可能会被一堆抽象的概念搞得晕头转向。什么封装、继承、多态，什么类与对象，这些词听起来高大上，但到底是什么意思？

让我从最朴素的角度来理解。你现在看到的这个世界，是由各种各样的"物体"组成的。每个物体有自己的特征，比如形状、颜色、大小；也有自己的行为，比如移动、生长、变化。面向对象编程，就是用计算机程序来模拟这个真实世界，用"类"来定义物体的模板，用"对象"来表示具体的物体。

比如，"狗"是一个类，它定义了狗的特征（名字、年龄、品种）和行为（叫、跑、吃）。而"我的狗旺财"就是一个具体的对象，它有具体的名字、年龄和品种，会做出具体的行为。

理解类和对象，是理解面向对象编程的第一步。

## 什么是类

类是对一类事物的抽象描述。你可以把它理解成一份蓝图或者模板，它定义了这类事物共有的特征和行为。

在 Python 中，用 `class` 关键字来定义类：

```python
class Dog:
    """一个简单的狗类"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says Woof!"

    def get_older(self):
        self.age += 1
```

这个类定义了狗的特征（name、age）和行为（bark、get_older）。

类名通常使用 CapWords 命名约定（即每个单词首字母大写），比如 `Dog`、`Person`、`ConfigManager`。

## 什么是对象

对象是类的实例。类是模板，对象是按照模板创建出来的具体实体。

```python
my_dog = Dog("旺财", 3)
your_dog = Dog("小白", 5)
```

这里 `my_dog` 和 `your_dog` 都是 `Dog` 类的实例，它们是两个不同的对象，有各自的 name 和 age。

判断一个对象是不是某个类的实例，用 `isinstance`：

```python
print(isinstance(my_dog, Dog))  # True
print(type(my_dog) == Dog)     # True
```

## __init__ 构造器

`__init__` 是类的初始化方法，也叫构造器。当你创建对象时，Python 会自动调用这个方法。

```python
class Dog:
    def __init__(self, name, age):
        print("Creating a new dog!")
        self.name = name
        self.age = age

my_dog = Dog("旺财", 3)
```

输出：

```
Creating a new dog!
```

`__init__` 不是真正的构造函数，真正的构造是 `__new__`（后面会讲）。`__init__` 的作用是初始化对象的属性。

注意：`__init__` 的第一个参数必须是 `self`，这是对对象自身的引用。

## self 的本质

`self` 可能让初学者困惑。它到底是什么？

`self` 是对当前对象实例的引用。当你在类的方法里写 `self.name`，你是在访问这个对象的 name 属性。

```python
class Dog:
    def __init__(self, name):
        self.name = name  # 把 name 存储到这个对象里

    def greet(self):
        return f"Hello, I am {self.name}"  # 从这个对象里读取 name

my_dog = Dog("旺财")
print(my_dog.greet())  # "Hello, I am 旺财"
```

当调用 `my_dog.greet()` 时，Python 实际上把 `my_dog` 作为 `self` 参数传给了方法。所以 `self.name` 就是 `my_dog.name`。

这和函数参数传递的"对象引用传递"机制是一样的——Python 把对象的引用传给了方法，让方法能够访问和修改对象的属性。

## 实例属性 vs 类属性

属性分为两种：实例属性和类属性。

实例属性是每个对象独有的，存储在对象的 `__dict__` 中：

```python
class Dog:
    def __init__(self, name):
        self.name = name  # 实例属性，每个狗有自己的名字

dog1 = Dog("旺财")
dog2 = Dog("小白")

dog1.name = "大旺"  # 只修改 dog1 的 name
print(dog1.name)  # 大旺
print(dog2.name)  # 小白，不受影响
```

类属性是所有对象共享的，存储在类的 `__dict__` 中：

```python
class Dog:
    species = "犬科"  # 类属性，所有狗共享

    def __init__(self, name):
        self.name = name  # 实例属性

dog1 = Dog("旺财")
dog2 = Dog("小白")

print(dog1.species)  # 犬科
print(dog2.species)  # 犬科

Dog.species = "猫科"  # 修改类属性会影响所有对象
print(dog1.species)  # 猫科
print(dog2.species)  # 猫科
```

注意：如果给对象赋值类属性，不会修改类属性，而是会在对象的 `__dict__` 中创建一个新的实例属性，覆盖类属性的读取。

```python
dog1.species = "犬科2"  # 创建新的实例属性
print(dog1.species)  # 犬科2
print(dog2.species)  # 猫科（仍然是类属性）
print(Dog.species)   # 猫科
```

## 对象的创建过程

当你调用 `Dog("旺财", 3)` 创建对象时，Python 内部经历了以下步骤：

1. 首先调用 `__new__` 创建新对象（通常不需要自己定义）
2. 然后调用 `__init__` 初始化对象属性

```python
class Dog:
    def __new__(cls, *args, **kwargs):
        print("__new__ called")
        return super().__new__(cls)

    def __init__(self, name):
        print("__init__ called")
        self.name = name

dog = Dog("旺财")
```

输出：

```
__new__ called
__init__ called
```

大多数情况下，只需要定义 `__init__`，不需要定义 `__new__`。但 `__new__` 在某些场景下很有用，比如实现单例模式，或者让不可变对象（像 tuple 的子类）正确工作。

## 对象的 __dict__ 和 __class__

每个对象都有两个特殊属性：

`__dict__` 是对象存储自有属性的字典：

```python
dog = Dog("旺财", 3)
print(dog.__dict__)  # {'name': '旺财', 'age': 3}
```

`__class__` 是对象所属的类：

```python
print(dog.__class__)  # <class '__main__.Dog'>
print(dog.__class__.__name__)  # Dog
```

可以通过 `__dict__` 查看对象有哪些属性，但通常直接用 `dog.name` 更方便。

## 常见误区

第一个误区是混淆类属性和实例属性。类属性所有对象共享，实例属性每个对象独有。修改类属性会影响所有对象，修改实例属性只影响当前对象。

第二个误区是在方法中忘记使用 `self`。方法里的变量如果没有前缀，会被当成局部变量，而不是对象的属性。

```python
class Dog:
    def set_name(self, name):
        name = name  # 错误！这是局部变量
        self.name = name  # 正确

# 第三个误区是认为 __init__ 是构造函数。
```

实际上 `__init__` 是初始化方法，不是构造方法。真正的构造是 `__new__`，但大多数情况下不需要自己定义。

理解类和对象的基础概念，是理解面向对象编程的第一步。类是抽象的模板，对象是具体的存在；类定义了什么，对象就有什么。
