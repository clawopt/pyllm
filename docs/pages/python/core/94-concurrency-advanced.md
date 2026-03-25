---
title: 并发进阶与实践
---

# 并发进阶与实践

这一章讨论并发编程中的高级主题，包括死锁预防、性能优化、实战场景，以及如何在具体项目中选择合适的并发方案。

## 死锁的预防与排查

死锁是并发编程中最棘手的问题之一。当两个或多个任务相互等待对方持有的资源时，就会发生死锁，所有任务都无法继续执行。

### 死锁的四个必要条件

1965年，计算机科学家Coffman提出了死锁发生的四个必要条件：

第一，互斥条件：资源一次只能被一个任务持有。第二，持有并等待条件：任务持有资源的同时请求其他资源。第三，不可抢占条件：资源不能被强制抢走，只能由任务主动释放。第四，循环等待条件：存在一个任务循环，每个任务都在等待下一个任务持有的资源。

破坏其中任何一个条件，就可以避免死锁。

### 固定加锁顺序

当多个任务需要获取多把锁时，以固定顺序获取可以避免循环等待：

```python
# 危险写法
def task1():
    with lock_a:
        with lock_b:
            pass

def task2():
    with lock_b:
        with lock_a:
            pass

# 安全写法：所有任务都按相同顺序获取锁
def task1():
    with lock_a:
        with lock_b:
            pass

def task2():
    with lock_a:  # 顺序与task1一致
        with lock_b:
            pass
```

当两个任务以不同顺序获取锁时，可能发生死锁。以相同顺序获取锁，消除了循环等待条件。

### 使用超时

给锁的获取添加超时，可以在发生死锁时及时退出：

```python
import threading

lock = threading.Lock()

def with_timeout(lock, timeout):
    return lock.acquire(timeout=timeout)

if __name__ == '__main__':
    result = with_timeout(lock, 1.0)
    if result:
        try:
            pass
        finally:
            lock.release()
    else:
        print("Failed to acquire lock within timeout")
```

超时不能完全防止死锁，但可以避免永久等待。在死锁发生时，程序可以检测到并恢复。

### 死锁检测

对于复杂的锁场景，可以使用资源分配图检测死锁。当图中存在环时，说明可能发生死锁。

Python的`threading`模块没有内置死锁检测。生产环境中，可以使用`faulthunter`库或编写自定义检测逻辑。

### 排查死锁

死锁发生时，程序会完全卡住。排查方法包括：

使用`faulthandler`模块获取线程堆栈：

```python
import faulthandler
import threading

faulthandler.enable()
faulthandler.dump_traceback()
```

或者在IDE的调试器中暂停程序，检查各个线程的堆栈和锁状态。

## 并发性能优化

并发编程的目标是提升性能。但不恰当的并发反而可能降低性能。

### 瓶颈分析

在优化之前，先用工具定位瓶颈：

```python
import cProfile
import pstats

cProfile.run('main()', 'profile.stats')

stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(20)
```

`cProfile`可以找出最耗时的函数。如果瓶颈在锁竞争，减少锁的粒度或改用无锁算法。

### 阿姆达尔定律

并发优化的收益受限于任务中不可并行的部分。设并行部分占比例为P，加速比为：

```
Speedup = 1 / (1 - P + P/N)
```

当P=0.9（即90%可并行），8核时加速比约为5.2倍。当P=1.0（全部可并行），8核时加速比接近8倍。

这意味着，如果任务中有大量无法并行的部分（比如GIL限制），增加更多线程或进程也无助益。

### 减少锁竞争

锁是并发性能的大敌。减少锁竞争的方法包括：

减小锁的粒度：用多把锁替代一把锁，让不同数据用不同锁保护：

```python
# 低效：一把锁保护所有数据
class Counter:
    def __init__(self):
        self.lock = threading.Lock()
        self.count_a = 0
        self.count_b = 0

# 高效：两把锁分别保护不同数据
class Counter:
    def __init__(self):
        self.lock_a = threading.Lock()
        self.lock_b = threading.Lock()
        self.count_a = 0
        self.count_b = 0
```

读写锁：读操作远多于写操作时，使用读写锁可以允许多个并发读：

```python
import threading

rwlock = threading.RLock()

def read():
    with rwlock:
        pass  # 读操作

def write():
    with rwlock:
        pass  # 写操作
```

无锁算法：在某些场景下，可以用原子操作替代锁：

```python
import threading

counter = 0

def increment():
    global counter
    for _ in range(100000):
        # 原子操作，无需锁
        atomic = getattr(threading, 'local', None)
        # 实际使用atomic_add等操作
```

Python中，`itertools`提供了一些无锁数据结构。对于简单计数，可以用`queue.Queue`或`collections.Counter`。

### 进程池vs线程池vs协程

选择合适的并发方案：

CPU密集型任务：用多进程。每个进程有独立GIL，可在多核真正并行。`multiprocessing.Pool`是标准选择。

IO密集型任务：用协程或线程池。协程开销最小，适合高并发；线程池适合中等并发。`asyncio`是现代Python的首选。

混合任务：可以用多进程+协程组合。每个进程运行事件循环，进程内用协程处理IO。

```python
import multiprocessing
import asyncio

def process_worker(pipe):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def task():
        await asyncio.sleep(0.1)
        return "done"

    while True:
        job = pipe.recv()
        result = loop.run_until_complete(task())
        pipe.send(result)

if __name__ == '__main__':
    multiprocessing.Pool(4)
```

## 实战：并发爬虫

用协程实现高效爬虫：

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def crawl(urls, concurrency=10):
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_fetch(session, url):
        async with semaphore:
            return await fetch(session, url)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    urls = [f"http://example.com/page{i}" for i in range(100)]
    results = await crawl(urls, concurrency=20)
    success = sum(1 for r in results if not isinstance(r, Exception))
    print(f"Success: {success}/{len(urls)}")

asyncio.run(main())
```

信号量限制并发数，避免对目标服务器造成过大压力。`aiohttp`是异步HTTP客户端，比`requests`快一个数量级。

## 实战：并发文件处理

处理大量文件时，用进程池并行：

```python
import multiprocessing
from pathlib import Path

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    return len(content)

def main():
    files = list(Path('data').glob('*.txt'))

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_file, files)

    print(f"Processed {len(results)} files")
    print(f"Total size: {sum(results)} bytes")

if __name__ == '__main__':
    main()
```

`Pool.map()`自动分配任务给工作进程，返回结果列表。对于需要控制并发数的场景，用`imap`或`apply_async`。

## 进程池调优

进程池的参数需要根据任务特性调整。

`processes`参数：通常设置为CPU核心数。对于IO密集型任务，可以设置更高：

```python
import multiprocessing
import os

cpu_count = os.cpu_count()
pool = multiprocessing.Pool(processes=cpu_count * 2)  # IO密集型
pool = multiprocessing.Pool(processes=cpu_count)       # CPU密集型
```

`maxtasksperchild`参数：限制每个进程执行的任务数，达到后重启进程。适用于内存泄漏或资源耗尽的情况：

```python
pool = multiprocessing.Pool(processes=4, maxtasksperchild=100)
```

## 常见问题

### 全局解释器锁

GIL限制了多线程的CPU并行。但某些场景下，多线程仍然有用：

等待IO时，线程会释放GIL，其他线程可以继续执行。C扩展可以释放GIL。`multiprocessing`可以绕过GIL。

### 调试并发程序

并发程序的bug往往是概率性的，难以复现。调试建议：

从简单开始：先用单线程验证逻辑，再逐步增加并发。添加日志：记录关键节点的线程/进程ID和时间。限制并发数：问题往往在并发量高时出现，控制并发便于定位。

### 资源限制

系统对线程数、进程数、文件描述符等有限制：

```bash
ulimit -a  # 查看限制
```

创建过多线程或进程会导致`ResourceWarning`。使用连接池、进程池管理资源。

## 面试关注点

面试中关于并发进阶的常见问题包括：如何排查和预防死锁？如何优化并发程序的性能？进程池、线程池、协程分别适合什么场景？

理解并发性能优化的本质很重要：阿姆达尔定律限定了优化的上限。如果任务大部分是串行的，增加并发可能无助益。优化应该聚焦在真正的瓶颈上。

高级面试题可能涉及：无锁数据结构的实现原理、CAS操作、C10K问题及其解决方案。
