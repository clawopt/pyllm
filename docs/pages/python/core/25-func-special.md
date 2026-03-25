---
title: 递归与特殊函数
---

# 递归与特殊函数

这一章是函数部分的最后一章，介绍两个独立的知识点：递归和可调用对象。

递归是一种编程技巧，让函数调用自身来解决问题。递归在处理树结构、分治算法等问题时非常自然。

可调用对象让类的实例可以像函数一样被调用。这在需要"带状态的函数"或者创建可配置的函数时很有用。

这一章解决的问题是：怎么用递归思想解决问题，怎么让对象变得可调用。

## 递归：函数调用自身

递归就是函数调用自身。听起来简单，但递归的核心在于：必须有一个终止条件来结束递归，否则会无限递归下去。

看一个经典例子——计算阶乘：

```python
def factorial(n):
    if n == 1:
        return 1  # 终止条件
    return n * factorial(n - 1)  # 递归调用

print(factorial(5))  # 120
```

执行流程是：factorial(5) 返回 5 * factorial(4)，factorial(4) 返回 4 * factorial(3)，直到 factorial(1) 返回 1，然后逐层返回 1*2*3*4*5 = 120。

递归需要两个要素：基线条件（终止条件）和递归步骤（调用自身）。没有基线条件，递归不会停止；没有递归调用，就不是递归了。

另一个经典例子是斐波那契数列：

```python
def fibonacci(n):
    if n <= 1:
        return n  # 基线条件
    return fibonacci(n - 1) + fibonacci(n - 2)  # 递归调用

print([fibonacci(i) for i in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

斐波那契数列的递归版本虽然代码简洁，但效率很低——会重复计算很多值。更高效的做法是用循环或者记忆化递归。

## 递归深度限制

Python 对递归深度有限制，默认大约是 1000。超过这个限制会抛出 RecursionError：

```python
def recursive_sum(n):
    if n == 0:
        return 0
    return n + recursive_sum(n - 1)

recursive_sum(2000)  # RecursionError: maximum recursion depth exceeded
```

可以用 `sys.setrecursionlimit` 调整限制：

```python
import sys
sys.setrecursionlimit(2000)

recursive_sum(1500)  # 不再报错
```

但这个做法有风险。如果递归太深，可能导致程序崩溃（栈空间耗尽）。Python 官方不支持尾递归优化（Tail Call Optimization），所以递归深度受限于栈大小。正确处理递归问题的方式是：要么确保递归深度可控，要么改用循环。

尾递归优化在其他语言（如 Scheme、Lisp）中存在，它把递归调用转换成循环，避免栈增长。但 Python 官方明确不支持尾递归优化。这意味着如果你的代码依赖深度递归，需要考虑改成循环实现，或者使用生成器来实现惰性递归。

## 递归与栈帧

每次函数调用都会创建一个栈帧（stack frame）。栈帧包含函数的局部变量、参数、返回地址等信息。递归调用就是不断创建新的栈帧。

当递归深入时，栈不断增长；当递归返回时，栈帧逐层销毁。如果递归太深，栈可能溢出。

可以用生成器来模拟"尾递归"，避免栈增长：

```python
def tail_recursive_sum(n, accumulator=0):
    if n == 0:
        return accumulator
    yield (n - 1, accumulator + n)  # 返回新参数，不增加栈深度

def sum_using_generator(n):
    result = 0
    gen = tail_recursive_sum(n)
    while True:
        try:
            n, result = next(gen)
        except StopIteration:
            return result
```

但这种写法不如直接用循环直观。递归的价值在于让某些问题的解决变得非常自然，不应该为了避免递归而牺牲代码可读性。

## 可调用对象：让类实例像函数一样被调用

通过实现 `__call__` 方法，可以让类的实例变得可调用，就像调用函数一样：

```python
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count

counter = Counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3
```

当执行 `counter()` 时，Python 实际上调用的是 `counter.__call__()`。这让我们可以用完全相同的方式使用 counter——无论 counter 是函数还是实例。

可调用对象和普通函数相比，优势在于可以维护状态。普通函数调用之间不共享状态（除非用全局变量），但可调用对象可以在实例属性中存储状态。

## 偏函数：固定部分参数

偏函数（Partial Function）来自 functools 模块，它允许你"冻结"一个函数的某些参数，创建一个新的函数：

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)  # 固定 exponent=2
cube = partial(power, exponent=3)     # 固定 exponent=3

print(square(5))   # 25，5 ** 2
print(cube(5))     # 125，5 ** 3
```

偏函数的实现原理很简单：创建一个可调用对象，存储原函数和被固定的参数，调用时把所有参数合并后传给原函数。

可以用 lambda 来实现简单的偏函数：

```python
square = lambda x: power(x, 2)
```

但偏函数的优势在于它显式地表达了"创建新函数"这个意图，而且在处理关键字参数时更清晰。

偏函数在回调函数场景下很有用。比如你要把一个多参数函数传给某个只接受单参数回调的 API：

```python
from functools import partial

def send_message(to, message):
    print(f"Sending {message} to {to}")

# 固定 to 参数
send_to_alice = partial(send_message, "alice@example.com")
send_to_alice("Hello!")  # 等价于 send_message("alice@example.com", "Hello!")
```

## 偏函数与回调模式

偏函数在事件处理和回调模式中特别有用。假设有一个按钮点击处理器：

```python
from functools import partial

def handle_click(button_name, event):
    print(f"Button {button_name} clicked at {event.x}, {event.y}")

# 创建特定按钮的处理器
handle_start = partial(handle_click, "start")
handle_stop = partial(handle_click, "stop")
handle_pause = partial(handle_click, "pause")

# 注册回调
button_start.on_click(handle_start)
button_stop.on_click(handle_stop)
```

这样每个按钮的处理器都绑定了自己的名称，不需要在回调里再做判断。

## 递归的适用场景

递归最适合处理天然具有递归结构的问题，比如：

树结构的遍历：

```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(node):
    if node is None:
        return []
    return inorder_traversal(node.left) + [node.value] + inorder_traversal(node.right)

# 树:     2
#        / \
#       1   3
tree = TreeNode(2, TreeNode(1), TreeNode(3))
print(inorder_traversal(tree))  # [1, 2, 3]
```

分治算法，如归并排序：

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```

递归的代码通常比循环版本更简洁、更易读。但要注意递归深度，必要时使用循环或者增大递归限制。

## 常见误区

第一个误区是忘记基线条件。没有基线条件的递归会无限调用，最终导致 RecursionError。

第二个误区是混淆递归和循环。虽然大多数递归可以改成循环，但某些问题用递归表达更自然。不要为了"避免递归"而把代码改成难读的循环。

第三个误区是忘记尾递归优化在 Python 中不存在。如果有人说"Python 支持尾递归优化"，这是错误的。

第四个误区是过度使用偏函数。偏函数虽然方便，但过度使用会导致代码难以理解——阅读代码的人需要追踪每个偏函数的来源。如果偏函数太多，考虑用类或者字典来组织。

第五个误区是不理解可调用对象的适用场景。可调用对象适合需要"带状态的函数"的场景，但如果只是简单的函数，直接用函数就行了，不需要创建类。

理解递归、可调用对象和偏函数，能让你在解决特定问题时选择最合适的工具。递归用于递归结构的问题，偏函数用于创建参数已固定的函数变体，可调用对象用于需要维护状态的函数。
