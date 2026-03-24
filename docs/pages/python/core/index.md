# Python 核心教程

本教程将带你从零开始学习 Python 编程语言的核心概念和语法。

## 变量与数据类型

Python 是动态类型语言，变量不需要声明类型。

```python
# 数字类型
age = 25
price = 99.99

# 字符串
name = "Python"
message = 'Hello, World!'

# 布尔值
is_valid = True
is_empty = False

# None
result = None
```

## 控制流

### 条件语句

```python
score = 85

if score >= 90:
    print("优秀")
elif score >= 60:
    print("及格")
else:
    print("不及格")
```

### 循环语句

```python
# for 循环
for i in range(5):
    print(i)

# while 循环
count = 0
while count < 5:
    print(count)
    count += 1
```

## 函数

```python
# 定义函数
def greet(name):
    return f"Hello, {name}!"

# 调用函数
message = greet("Python")
print(message)  # 输出: Hello, Python!

# 默认参数
def power(base, exp=2):
    return base ** exp

print(power(3))     # 9
print(power(3, 3))  # 27
```

## 列表

```python
# 创建列表
fruits = ["apple", "banana", "cherry"]

# 访问元素
print(fruits[0])  # apple

# 添加元素
fruits.append("orange")

# 列表推导式
squares = [x**2 for x in range(10)]
```

## 字典

```python
# 创建字典
person = {
    "name": "Alice",
    "age": 30,
    "city": "Beijing"
}

# 访问值
print(person["name"])  # Alice

# 添加键值对
person["email"] = "alice@example.com"
```

## 类与对象

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says Woof!"

# 创建对象
my_dog = Dog("Buddy", 3)
print(my_dog.bark())  # Buddy says Woof!
```

## 下一步

继续学习：
- [NumPy教程](/pages/python/numpy/)
- [Pandas教程](/pages/python/pandas/)
