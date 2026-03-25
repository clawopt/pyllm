---
title: 封装与访问控制
---

# 封装与访问控制

封装是面向对象编程的三大特性之一（另外两个是继承和多态）。封装的目的是隐藏实现细节，只暴露必要的接口。

比如，你开汽车的时候，不需要知道发动机是怎么工作的，只需要知道踩油门会加速、踩刹车会减速。汽车把你不需要知道的细节隐藏起来了，只暴露了"踏板"这个接口。这就是封装。

Python 的封装机制和其他语言（如 Java、C++）有些不同。Python 没有真正的 private 关键字，而是用命名约定来实现访问控制。

## 公有、私有、受保护属性

Python 的属性访问控制有三种：公有（public）、受保护（protected）、私有（private）。

公有属性没有前缀，可以自由访问：

```python
class Person:
    def __init__(self, name):
        self.name = name  # 公有属性

person = Person("Alice")
print(person.name)  # 可以访问
```

受保护属性用单下划线前缀 `_`，表示"这是内部使用的，不应该在外部直接访问"：

```python
class Person:
    def __init__(self, name):
        self._name = name  # 受保护属性

person = Person("Alice")
print(person._name)  # 可以访问，但不建议
```

私有属性用双下划线前缀 `__`，会触发 name mangling：

```python
class Person:
    def __init__(self, name):
        self.__name = name  # 私有属性

person = Person("Alice")
# print(person.__name)  # 会报错，访问不到
```

## name mangling 机制

Python 的私有属性采用 name mangling（名称重整）机制。双下划线前缀的属性，在编译时会被重整成 `_ClassName__attribute` 的形式。

```python
class Person:
    def __init__(self, name):
        self.__name = name  # 私有属性

    def get_name(self):
        return self.__name

person = Person("Alice")
print(person.get_name())  # "Alice"
print(person._Person__name)  # "Alice"，可以直接访问重整后的名字
```

```python
# 实际存储在：
print(person.__dict__)  # {'_Person__name': 'Alice'}
```

这种机制的目的不是真正阻止访问，而是提醒程序员"这是私有的，不应该从这里访问"。真正想访问的话，还是能通过 `_ClassName__attribute` 访问。

## getter 和 setter 的 Python 风格

在其他语言中，私有属性通过 getter 和 setter 方法来访问。但在 Python 中，通常直接访问属性，只有在需要验证或计算时才用 `@property`。

```python
class Person:
    def __init__(self, name):
        self.name = name  # 直接暴露属性

person = Person("Alice")
print(person.name)  # 直接访问
person.name = "Bob"  # 直接修改
```

如果需要验证或计算，才用 `@property`：

```python
class Person:
    def __init__(self, name):
        self._name = name  # 用受保护的属性存储

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not value:
            raise ValueError("名字不能为空")
        self._name = value

person = Person("Alice")
print(person.name)  # 通过 property 访问
person.name = "Bob"  # 通过 property 修改
```

## 属性的 getter、setter、deleter

`@property` 不仅可以定义 getter，还可以定义 setter 和 deleter。

```python
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not value:
            raise ValueError("名字不能为空")
        self._name = value

    @name.deleter
    def name(self):
        print("Deleting name...")
        del self._name
```

使用：

```python
person = Person("Alice")
print(person.name)  # Alice
person.name = "Bob"  # Bob
del person.name      # Deleting name...
```

## __slots__ 内存优化

默认情况下，每个对象都有自己的 `__dict__`，可以动态添加任意属性。但这会消耗额外的内存。

`__slots__` 可以限制对象能拥有的属性，减少内存消耗：

```python
class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y
```

```python
point = Point(1, 2)
print(point.x, point.y)  # 1 2
# point.z = 3  # 报错！AttributeError
# point.__dict__  # 报错！对象没有 __dict__
```

使用 `__slots__` 后，对象不会有 `__dict__`，这能显著减少内存消耗，尤其是在创建大量对象时。

```python
import sys

class WithoutSlots:
    def __init__(self):
        self.x = 1
        self.y = 2

class WithSlots:
    __slots__ = ['x', 'y']
    def __init__(self):
        self.x = 1
        self.y = 2

print(sys.getsizeof(WithoutSlots()))  # 更大
print(sys.getsizeof(WithSlots()))    # 更小
```

注意：如果类有父类，父类有 `__dict__`，子类也会有。

## 属性的查找顺序

当访问 `obj.attr` 时，Python 按以下顺序查找：

1. 数据描述器（data descriptor，如覆盖了 `__get__` 和 `__set__` 的类）
2. 实例对象的 `__dict__`
3. 非数据描述器（non-data descriptor）和类属性

```python
class Descriptor:
    def __get__(self, obj, objtype=None):
        return "from descriptor"

class MyClass:
    attr = Descriptor()  # 数据描述器
```

这种查找顺序允许描述器覆盖实例属性。

## 封装的原则

封装的核心原则是：只暴露必要的接口，隐藏实现细节。

对外暴露的接口应该是稳定的，不应该随着内部实现的变化而变化。比如，如果你的类内部用了列表存储数据，但后来改成用字典存储，对外暴露的接口不应该变。

Python 的封装是"约定优于强制"的哲学。没有真正的 private 属性，只有命名约定。这意味着程序员需要自律，遵守"不应该访问私有属性"的约定。

## 常见误区

第一个误区是认为双下划线属性是真正私有的。Python 的 `__private` 只是触发了 name mangling，仍然可以通过 `_ClassName__private` 访问。

第二个误区是过度封装。Python 的哲学是"我们是成年人"，不需要强制隐藏。如果一个属性没有下划线前缀，就是可以访问的，不需要用下划线前缀来"保护"它。

第三个误区是忘记 `__slots__` 的限制。使用 `__slots__` 后，不能动态添加新属性（除非父类有 `__dict__`），这有时候会限制灵活性。

第四个误区是在需要时才使用 `@property`。如果属性不需要任何处理，直接暴露即可，不需要用 `@property` 包装。

理解封装的原理和 Python 的访问控制机制，能帮助你写出更规范的代码。
