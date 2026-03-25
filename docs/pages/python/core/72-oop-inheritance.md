---
title: 继承与多态
---

# 继承与多态

当你听到"继承"这个词，可能会想到财产继承——子女继承父母的财产。在面向对象编程中，继承也是类似的概念：子类继承父类的属性和方法。

继承解决了代码复用的问题。如果多个类有共同的特征和行为，可以把这些共同的部分提取到父类中，让子类继承。比如狗和猫都有"叫"的行为，但叫的声音不同。可以先定义一个"动物"类，然后让狗和猫继承它。

多态是面向对象编程的另一个核心概念。"多态"的意思是"多种形态"。同一个方法，在不同对象上有不同的行为。比如"叫"这个方法，在狗身上是"汪汪"，在猫身上是"喵喵"。

理解继承和多态，是理解面向对象编程的重要一步。

## 什么是继承

继承就是子类继承父类的属性和方法。子类可以直接使用父类定义的一切，不需要重新写。

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("子类必须实现 speak 方法")

class Dog(Animal):  # Dog 继承 Animal
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("旺财")
cat = Cat("小白")

print(dog.speak())  # "旺财 says Woof!"
print(cat.speak())  # "小白 says Meow!"
```

这里 `Dog` 和 `Cat` 都继承自 `Animal`。它们自动拥有了 `Animal` 的 `__init__` 方法，所以 `Dog("旺财")` 能够设置 `self.name`。

子类也可以覆盖（override）父类的方法，重新定义自己的行为。

## 单继承与多继承

Python 支持单继承（一个类只有一个父类）和多继承（一个类有多个父类）。

单继承是最常见的：

```python
class Dog(Animal):
    pass
```

多继承允许一个类继承多个父类：

```python
class Hybrid(Dog, Cat):  # 同时继承 Dog 和 Cat
    pass
```

多继承看起来很方便，但容易引发问题，比如"菱形继承"和 MRO（方法解析顺序）问题。除非特别需要，否则应该优先使用单继承。

## super() 与 MRO

`super()` 是调用父类方法的关键工具。它让你能够在子类中调用父类的方法。

最常见的用途是在子类的 `__init__` 中调用父类的 `__init__`：

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # 调用父类的 __init__
        self.breed = breed

dog = Dog("旺财", "金毛")
print(dog.name)   # 旺财
print(dog.breed)   # 金毛
```

`super()` 按照 MRO（方法解析顺序）来调用方法。MRO 定义了当查找方法时，Python 搜索父类的顺序。

可以用 `ClassName.__mro__` 查看类的 MRO：

```python
class A:
    pass

class B(A):
    pass

class C(A):
    pass

class D(B, C):
    pass

print(D.__mro__)
# (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```

Python 使用 C3 线性化算法来计算 MRO，确保每个类只出现一次，且保持继承顺序。

## 方法覆盖 (Override)

子类可以覆盖（override）父类的方法，重新定义自己的行为：

```python
class Animal:
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):  # 覆盖父类的 speak
        return "Woof!"

dog = Dog()
print(dog.speak())  # "Woof!"
```

覆盖后，调用 `dog.speak()` 会执行子类的方法，而不是父类的。

如果需要在子类中调用被覆盖的父类方法，用 `super().method()`：

```python
class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

    def speak_loud(self):
        parent_sound = super().speak()  # 调用父类的 speak
        return parent_sound.upper()
```

## 多态的实际应用

多态让不同对象对同一消息有不同的响应。这是面向对象编程的核心优势之一。

```python
def make_noise(animal):
    print(animal.speak())

dog = Dog("旺财")
cat = Cat("小白")

make_noise(dog)  # 旺财 says Woof!
make_noise(cat)  # 小白 says Meow!
```

`make_noise` 函数接受任何"会叫的东西"，不关心具体是什么类型，只要它有 `speak` 方法就行。这就是多态的威力——同样的代码，不同的对象，有不同的行为。

这种"鸭子类型"（duck typing）的思想是 Python 的核心：不需要显式声明"这是一个 Animal"，只需要对象有 `speak` 方法就可以了。

```python
# 甚至不需要继承任何类
class Car:
    def speak(self):
        return "Honk!"

car = Car()
make_noise(car)  # "Honk!"，Car 类没有继承 Animal，但一样能工作
```

这就是 Python 的哲学："如果走起来像鸭子，叫起来像鸭子，那它就是鸭子。"

## isinstance() 与类型检查

`isinstance()` 用来检查对象是否是某个类（或其子类）的实例：

```python
dog = Dog("旺财")

print(isinstance(dog, Dog))    # True
print(isinstance(dog, Animal))  # True，Dog 是 Animal 的子类
print(isinstance(dog, Cat))    # False
```

注意：`isinstance()` 会考虑继承关系，而 `type() ==` 不会：

```python
print(type(dog) == Dog)    # True
print(type(dog) == Animal)  # False
```

虽然可以用 `isinstance()` 做类型检查，但 Python 推崇"鸭子类型"，尽量避免显式类型检查。如果代码需要检查类型才能正常工作，通常是设计有问题。

## 继承与组合

虽然继承很好用，但不应该滥用。面向对象设计中有个原则："优先使用组合，而不是继承。"

继承的问题是：子类和父类之间形成了强耦合。父类的任何修改都可能影响子类。

组合是把其他类的实例作为自己的属性：

```python
class Engine:
    def start(self):
        return "Engine started"

class Car:
    def __init__(self):
        self.engine = Engine()  # 组合，而不是继承

    def start(self):
        return self.engine.start()
```

什么时候用继承，什么时候用组合？有一个简单的判断标准：如果"是"的关系（比如 Dog is Animal），用继承；如果"有"的关系（比如 Car has Engine），用组合。

## 常见误区

第一个误区是滥用继承。很多人觉得继承是代码复用的最佳方式，其实不然。过度使用继承会让类层次结构变得复杂难维护。如果类之间的关系不是"is-a"，应该考虑组合。

第二个误区是不理解 MRO。在多继承时，方法调用的顺序由 MRO 决定。搞不清楚 MRO 可能导致意外行为。可以用 `ClassName.__mro__` 查看。

第三个误区是覆盖方法时忘记调用 `super()`。有时候子类的方法需要在父类的基础上添加功能，而不是完全替代。

第四个误区是在 Python 中过度使用类型检查。Python 的多态依赖于"有某个方法"而不是"是某个类型"。应该让代码更灵活，而不是写很多 `isinstance()` 判断。

理解继承和多态，是理解面向对象编程的重要里程碑。它们让代码更加模块化、可复用、可扩展。
