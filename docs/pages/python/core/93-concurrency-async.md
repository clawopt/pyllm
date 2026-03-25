---
title: 异步编程
---

# 异步编程

asyncio是Python 3.5引入的异步编程标准库，它基于事件循环，在单线程内实现并发。asyncio使用async/await语法，让异步代码看起来像同步代码，大大降低了异步编程的复杂度。

这一章介绍asyncio的核心概念和用法，包括协程、任务、事件循环等内容。

## async与await

asyncio的基石是协程函数。用`async def`定义的函数是协程函数，调用它返回一个协程对象：

```python
async def fetch_data():
    return "data"

coro = fetch_data()
print(coro)  # <coroutine object fetch_data at 0x...>
```

协程函数和普通函数不同：调用协程函数只是创建了协程对象，不会真正执行。要执行协程，必须用`asyncio.run()`或者在已有事件循环中用`await`：

```python
import asyncio

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

`asyncio.run()`创建新的事件循环，运行协程，然后关闭循环。这是最常用的启动协程的方式。

`await`关键字有两个作用：暂停当前协程，等待另一个协程完成。暂停期间，事件循环可以运行其他协程。

```python
async def task1():
    print("Task 1 start")
    await asyncio.sleep(1)
    print("Task 1 end")

async def task2():
    print("Task 2 start")
    await asyncio.sleep(0.5)
    print("Task 2 end")

async def main():
    await task1()
    await task2()

asyncio.run(main())
```

这个例子中，task1和task2是顺序执行的。如果让它们并发执行：

```python
async def main():
    await asyncio.gather(task1(), task2())

asyncio.run(main())
```

用`asyncio.gather()`可以并发运行多个协程。总耗时是1秒（最长任务的耗时），而不是1.5秒。

## 事件循环

事件循环是asyncio的核心。它是一个无限循环，监听IO事件，在等待期间执行可运行的任务。

`asyncio.run()`自动创建和管理事件循环。在高级用法中，可能需要手动控制事件循环：

```python
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

try:
    loop.run_until_complete(main())
finally:
    loop.close()
```

这在需要在同一事件循环中运行多个任务的场景下很有用。但大多数情况下，`asyncio.run()`就够了。

事件循环可以注册回调函数：

```python
def callback(name):
    print(f"Callback {name}")

loop = asyncio.new_event_loop()
loop.call_soon(callback, "first")
loop.call_soon(callback, "second")
loop.run_until_complete(asyncio.sleep(0))
```

`call_soon()`在下一个事件循环迭代时执行回调。`call_later()`延迟一段时间后执行。

## Task与Future

Task是Future的子类，代表一个可等待的对象，可以在事件循环中并发执行。

`asyncio.create_task()`创建Task并安排其执行：

```python
async def main():
    task = asyncio.create_task(fetch_data())
    print("Task created")
    result = await task
    print(f"Result: {result}")

asyncio.run(main())
```

task在创建后立即开始执行，不需要等待。创建task和等待task之间，协程可能在后台运行。

`asyncio.ensure_future()`也可以创建task，是旧API但仍广泛使用：

```python
task = asyncio.ensure_future(fetch_data())
```

Future代表一个异步操作的最终结果。可以手动创建Future并在其完成后设置结果：

```python
async def with_future():
    future = asyncio.Future()

    def set_result():
        future.set_result("Done")

    loop.call_soon(set_result)
    result = await future
    print(result)
```

通常不需要直接操作Future，asyncio的API会处理它。

## 并发执行

`asyncio.gather()`是最常用的并发执行工具：

```python
async def fetch(url):
    await asyncio.sleep(0.5)
    return url

async def main():
    urls = ['a', 'b', 'c']
    results = await asyncio.gather(*[fetch(url) for url in urls])
    print(results)

asyncio.run(main())
```

`asyncio.gather()`按顺序返回结果列表，总耗时是所有任务中最长任务的耗时。

`asyncio.wait()`等待多个task完成，返回(done, pending)元组：

```python
async def main():
    task1 = asyncio.create_task(task1())
    task2 = asyncio.create_task(task2())

    done, pending = await asyncio.wait([task1, task2])
    for t in done:
        print(t.result())
```

`asyncio.as_completed()`按完成顺序返回结果：

```python
async def main():
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    for future in asyncio.as_completed(tasks):
        result = await future
        print(result)
```

## 异步生成器与迭代器

异步生成器用`async def`定义，使用`yield`产出值：

```python
async def async_gen():
    for i in range(3):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for value in async_gen():
        print(value)

asyncio.run(main())
```

异步迭代器需要实现`__aiter__`和`__anext__`方法：

```python
class AsyncCounter:
    def __init__(self, n):
        self.n = n
        self.current = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current >= self.n:
            raise StopAsyncIteration
        value = self.current
        self.current += 1
        await asyncio.sleep(0.1)
        return value

async def main():
    async for i in AsyncCounter(3):
        print(i)

asyncio.run(main())
```

`async for`自动调用`__anext__`，并在遇到`StopAsyncIteration`时停止。

## 异步上下文管理器

异步上下文管理器用`async with`语句：

```python
async def main():
    async with async_resource() as resource:
        await resource.use()
```

异步上下文管理器需要实现`__aenter__`和`__aexit__`方法：

```python
class async_resource:
    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
```

`aiohttp`库的`ClientSession`就是异步上下文管理器的典型应用：

```python
import aiohttp

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()
```

`async with`确保资源在使用后被正确清理，即使在异步操作中发生异常也会执行清理逻辑。

## 异常处理

协程中的异常需要特别处理，否则可能被静默忽略：

```python
async def risky():
    raise ValueError("Oops!")

async def main():
    try:
        await risky()
    except ValueError as e:
        print(f"Caught: {e}")

asyncio.run(main())
```

用`try/except`包装`await`。

用`asyncio.gather()`时，可以收集所有结果或异常：

```python
async def main():
    results = await asyncio.gather(
        safe_task(),
        risky_task(),
        return_exceptions=True
    )
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} raised: {result}")
        else:
            print(f"Task {i} result: {result}")
```

`return_exceptions=True`让gather不抛出异常，而是将异常作为结果返回。

## 取消任务

可以取消正在运行的任务：

```python
async def long_task():
    try:
        while True:
            print("Working...")
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        print("Task was cancelled")
        raise

async def main():
    task = asyncio.create_task(long_task())
    await asyncio.sleep(1.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

asyncio.run(main())
```

`task.cancel()`请求取消任务。任务需要在`await`点检查取消状态，或者捕获`CancelledError`并重新抛出。

可以用`asyncio.shield()`保护任务不被取消：

```python
async def main():
    protected = asyncio.create_task(shield(some_task()))
    await asyncio.sleep(1)
    protected.cancel()
    try:
        await protected
    except asyncio.CancelledError:
        print("Protected task was cancelled despite shield")
```

`shield()`确保被保护的任务不会被外部取消请求取消。

## 常见误区

第一个误区是忘记启动协程。协程如果不用`await`调用或者不用`asyncio.run()`运行，什么都不会发生：

```python
async def hello():
    print("Hello!")

hello()  # 什么都不打印
asyncio.run(hello())  # 真正运行
```

第二个误区是在同步代码中使用`await`。`await`只能在协程函数中使用。如果需要在同步代码中运行协程，必须用`asyncio.run()`。

第三个误区是混淆并发和并行。asyncio是单线程并发，不是多线程并行。多个协程在同一个线程中交替执行。如果需要真正的并行计算，应该用`multiprocessing`。

第四个误区是在异步函数中使用阻塞调用。`time.sleep()`会阻塞整个事件循环，应该用`await asyncio.sleep()`：

```python
async def wrong():
    time.sleep(1)  # 阻塞事件循环

async def right():
    await asyncio.sleep(1)  # 让出控制权
```

如果必须使用阻塞调用，可以用`loop.run_in_executor()`在线程池中执行：

```python
async def with_blocking():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, blocking_function, arg)
```

## 异步与线程的结合

asyncio可以与线程结合使用，处理不支持异步的库：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def blocking_io():
    time.sleep(1)
    return "Done"

async def main():
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    result = await loop.run_in_executor(executor, blocking_io)
    print(result)

    executor.shutdown()
```

`run_in_executor()`在线程池中执行阻塞函数，不阻塞事件循环。

也可以从协程中启动线程：

```python
import asyncio
import threading

def thread_worker():
    time.sleep(1)
    print("Thread done")

async def main():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, thread_worker)

asyncio.run(main())
```

这种方式适合逐步将同步代码迁移到异步。
