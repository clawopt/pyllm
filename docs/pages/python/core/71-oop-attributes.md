---
title: 属性与方法
---

# 属性与方法

上一章讲了类和对象的基础概念。这一章来深入讲属性和方法。属性是对象的数据，方法是对象的行为。但方法又分为实例方法、类方法和静态方法，它们的区别是什么？属性有实例属性和类属性，它们是怎么查找的？

理解属性和方法的机制，是理解 Python 面向对象的关键。Python 的属性查找比很多语言更灵活，但这种灵活性也带来了理解上的复杂性。

## 实例方法

实例方法是最常见的方法类型。它的第一个参数是 `self`，代表对象实例。

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):  # 实例方法
        return f"{self.name} says Woof!"

dog = Dog("旺财")
print(dog.bark())  # "旺财 says Woof!"
```

当调用 `dog.bark()` 时，Python 内部实际上是 `Dog.bark(dog)`。这就是为什么实例方法的第一个参数必须是 `self`——Python 会自动把对象实例传进去。

## 类方法

类方法的第一个参数是 `cls`，代表类本身而不是对象实例。用 `@classmethod` 装饰器来定义：

```python
class Dog:
    species = "犬科"

    def __init__(self, name):
        self.name = name

    @classmethod
    def create_puppy(cls, name):
        return cls(name + "崽")  # 用 cls 创建新实例

puppy = Dog.create_puppy("旺财")
print(puppy.name)  # 旺财崽
print(puppy.species)  # 犬科
```

类方法的典型用途是：作为工厂方法创建对象，或者访问类级别的属性。

类方法可以通过类本身调用，也可以通过实例调用：

```python
puppy = Dog.create_puppy("旺财")  # 通过类调用
puppy2 = puppy.create_puppy("小白")  # 通过实例调用，仍然传的是 Dog 类
```

## 静态方法

静态方法既不需要类参数也不需要实例参数。用 `@staticmethod` 装饰器来定义：

```python
class Math:
    @staticmethod
    def add(a, b):
        return a + b

result = Math.add(1, 2)
print(result)  # 3
```

静态方法可以看作是一种"放在类里的普通函数"。它和类、对象都没有绑定关系，只是逻辑上属于这个类。

什么时候用静态方法？当某个函数和这个类相关，但不需要访问类或实例的属性时。比如工具函数、验证函数等。

## 三种方法的对比

| 类型 | 装饰器 | 第一个参数 | 调用方式 | 访问权限 |
|------|--------|-----------|---------|---------|
| 实例方法 | 无 | self | 通过实例调用 | 可访问实例和类属性 |
| 类方法 | @classmethod | cls | 通过类或实例调用 | 可访问类属性 |
| 静态方法 | @staticmethod | 无 | 通过类或实例调用 | 不可访问类或实例属性 |

```python
class Example:
    class_attr = "类属性"

    def __init__(self):
        self.instance_attr = "实例属性"

    def instance_method(self):
        return "实例方法"

    @classmethod
    def class_method(cls):
        return f"类方法，{cls.class_attr}"

    @staticmethod
    def static_method():
        return "静态方法"
```

## property 装饰器

通常情况下，类的属性是直接暴露的。但有时候需要控制属性的读取和修改逻辑。比如检查赋值是否合法，或者计算衍生属性。

`@property` 装饰器让方法可以像属性一样被访问：

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("半径不能为负数")
        self._radius = value

    @property
    def area(self):
        return 3.14159 * self._radius ** 2
```

使用：

```python
circle = Circle(5)
print(circle.radius)  # 5，像属性一样访问
circle.radius = 10     # 自动调用 setter
print(circle.area)    # 314.159，不需要括号
```

注意：私有属性用 `_` 前缀命名（比如 `_radius`），这是 Python 的命名约定，表示"这是内部使用的属性"。

## getter 和 setter 的 Python 风格

在其他语言（如 Java）中，属性通常是私有的，通过 getter 和 setter 方法来访问。但 Python 推崇直接访问属性的风格。

Python 的哲学是："我们是成年人"——相信程序员会正确使用 API，而不是强制隐藏实现细节。所以 Python 的属性默认是公开的。

但如果需要在访问属性时做额外处理（比如验证），用 `@property` 是标准做法：

```python
class Person:
    def __init__(self, name):
        self.name = name  # 这里直接赋值，不会触发验证

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not value:
            raise ValueError("名字不能为空")
        self._name = value
```

## 属性的查找机制

当你访问 `obj.attr` 时，Python 会按顺序查找：

1. 对象的 `__dict__`（实例属性）
2. 类的 `__dict__`（类属性和方法）
3. 父类的 `__dict__`（按 MRO 顺序）

```python
class A:
    x = 1

class B(A):
    pass

b = B()
print(b.x)  # 1，从 A 继承来的
print(b.__dict__)  # {}，没有实例属性
```

当你给属性赋值时，如果对象没有这个属性，会在对象的 `__dict__` 中创建新的实例属性：

```python
b.x = 2  # 在 b.__dict__ 中创建 x = 2
print(b.x)  # 2，现在从实例获取
print(A.x)  # 1，类的 x 不受影响
```

这叫做"属性遮蔽"——实例属性遮蔽了同名的类属性。

## 方法的动态绑定

Python 的类方法在绑定时有动态性。方法本质上是一个函数，赋值给类属性时就变成了"未绑定方法"，赋值给实例属性时就变成了"绑定方法"。

```python
class Dog:
    def bark(self):
        return f"{self.name} says Woof!"

dog = Dog()
dog.name = "旺财"

# bark 被绑定到 dog 实例
print(dog.bark())  # "旺财 says Woof!"
```

这种动态绑定机制让 Python 的 OOP 比很多静态语言更灵活。

## 常见误区

第一个误区是不理解实例方法和类方法的区别。实例方法用于访问和修改对象状态，类方法用于访问类级别数据或创建工厂方法。

第二个误区是滥用 `@property`。如果属性不需要任何处理逻辑，直接暴露即可，不需要用 `@property`。

第三个误区是混淆实例属性和类属性。类属性在所有实例间共享，实例属性每个对象独有。

第四个误区是在实例方法中用硬编码的类名而不是 `self`。如果需要引用类，用 `self.__class__` 而不是直接写类名，这样子类继承时行为正确。

理解属性和方法的各种机制，是掌握 Python 面向对象的重要一步。
