---
title: 多线程编程
---

# 多线程编程

threading是Python中实现多线程编程的标准模块。与多进程不同，多线程在同一个进程内运行，共享进程的内存空间，这让线程间通信更便捷，但也带来了同步问题。

这一章介绍threading模块的基本用法，包括线程创建、同步原语、线程安全等内容。

## 线程与进程的区别

理解线程和进程的区别是正确选择并发方案的前提。

进程是资源分配的基本单位，有独立的地址空间、文件描述符、堆内存。进程之间相互隔离，一个进程崩溃不会影响其他进程。但进程创建和切换开销大，进程间通信需要额外机制。

线程是CPU调度的基本单位，运行在同一进程的地址空间内，共享进程的堆内存和文件描述符。线程创建和切换开销小，线程间可以直接读写共享数据。但正因为共享内存，一个线程的错误可能导致整个进程崩溃。

Python的GIL让多线程无法真正并行执行。对于CPU密集型任务，多线程甚至可能比单线程更慢。但对于IO密集型任务，多线程仍然有效——线程在等待IO时可以释放GIL，让其他线程继续执行。

选择多线程还是多进程，取决于任务特性和性能要求。IO密集型任务适合多线程，CPU密集型任务如果需要利用多核则用多进程。

## 创建线程

threading模块提供了多种创建线程的方式。

创建线程的最基本方式是通过Thread类：

```python
import threading

def worker(name):
    print(f"Hello from {name}")

t = threading.Thread(target=worker, args=("Worker",))
t.start()
t.join()
```

`target`参数指定线程执行的函数，`args`是传递给函数的参数。`start()`启动线程，`join()`等待线程结束。

给线程命名方便调试：

```python
t = threading.Thread(target=worker, name="MyWorker")
print(t.name)  # MyWorker
```

也可以继承Thread类，重写`run()`方法：

```python
class MyThread(threading.Thread):
    def __init__(self, name):
        super().__init__(name=name)

    def run(self):
        print(f"Running {self.name}")

t = MyThread(name="CustomThread")
t.start()
t.join()
```

这种方式适合需要在线程中封装更多逻辑的场景。

## 线程同步原语

多个线程访问共享资源时，需要同步机制防止数据不一致。

### Lock

Lock是最基本的锁，同一时刻只允许一个线程持有：

```python
counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1
```

`with lock:`语句确保同一时刻只有一个线程执行缩进内的代码。由于`counter += 1`不是原子操作，必须用锁保护。

### RLock

RLock允许同一线程多次获取：

```python
lock = threading.RLock()

with lock:
    with lock:  # 同一线程可以再次获取
        pass
```

RLock允许持有锁的线程重复获取锁，避免自己锁死自己。代价是比Lock稍复杂，性能也略低。

### Semaphore

Semaphore控制同时访问资源的线程数：

```python
semaphore = threading.Semaphore(2)

def worker(n):
    with semaphore:
        print(f"Worker {n} acquired semaphore")

for i in range(3):
    threading.Thread(target=worker, args=(i,)).start()
```

Semaphore内部维护一个计数器，获取时减1，释放时加1。计数器为0时，获取操作阻塞。适合限制并发连接数。

### Event

Event用于线程间信号通知：

```python
event = threading.Event()

def waiter():
    print("Waiting for event")
    event.wait()
    print("Event triggered!")

def setter():
    print("Setting event")
    event.set()

t1 = threading.Thread(target=waiter)
t2 = threading.Thread(target=setter)

t1.start()
t2.start()
t1.join()
t2.join()
```

`wait()`阻塞等待事件，`set()`触发事件。Event可以重复使用，`clear()`重置事件。

### Condition

Condition结合了锁和等待通知机制：

```python
condition = threading.Condition()

def consumer():
    with condition:
        while not data_ready:
            condition.wait()
        print("Consuming:", data)

def producer():
    global data_ready, data
    data = "produced"
    data_ready = True
    with condition:
        condition.notify()
```

`wait()`阻塞等待，`notify()`唤醒一个等待的线程，`notify_all()`唤醒所有等待的线程。Condition适合生产者-消费者模式。

## 线程安全的数据结构

Python的queue模块提供了线程安全的队列：

```python
import queue

q = queue.Queue()

def producer():
    for i in range(5):
        q.put(i)

def consumer():
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Got: {item}")

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)

t1.start()
t1.join()
q.put(None)  # 发送结束信号
t2.join()
```

Queue的`get()`和`put()`操作是线程安全的，不需要额外加锁。Queue还提供了`join()`方法等待队列清空。

## 线程池

concurrent.futures模块提供了线程池接口：

```python
from concurrent.futures import ThreadPoolExecutor

def task(n):
    return n * n

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task, i) for i in range(5)]
    for f in futures:
        print(f.result())
```

`ThreadPoolExecutor`管理线程池，自动复用线程。`submit()`提交任务返回Future对象，`result()`获取结果。

也可以用`map()`简化：

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(task, range(5))
    print(list(results))
```

`map()`按顺序返回结果，如果需要按完成顺序处理，用`as_completed()`：

```python
from concurrent.futures import as_completed

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(task, i) for i in range(5)]
    for f in as_completed(futures):
        print(f.result())
```

## GIL与线程性能

GIL是Python面试中的高频问题，理解它对正确使用多线程至关重要。

GIL的存在意味着同一时刻只有一个线程执行Python字节码。即使创建多个线程，在任意时刻只有一个线程真正运行。这意味着多线程无法加速CPU密集型任务。

```python
import threading
import time

def cpu_task():
    total = 0
    for i in range(10**7):
        total += i

start = time.time()
t1 = threading.Thread(target=cpu_task)
t2 = threading.Thread(target=cpu_task)
t1.start()
t2.start()
t1.join()
t2.join()
print(f"Time: {time.time() - start:.2f}")  # 可能比单线程更慢
```

对于IO密集型任务，多线程仍然有效：

```python
import requests

def fetch(url):
    return requests.get(url).status_code

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch, url) for url in urls]
```

当线程等待网络响应时，会释放GIL，让其他线程继续执行。因此多线程可以并发处理多个IO请求。

## 常见误区

第一个误区是以为多线程可以加速CPU密集型任务。由于GIL的存在，CPU密集型任务在多线程中反而可能更慢。如果需要加速CPU密集型任务，应该使用多进程。

第二个误区是在不需要的时候加锁。锁保护共享数据，但也会降低并发性。如果数据不被多线程共享，就不需要加锁。过度的锁反而降低性能。

第三个误区是忘记释放锁。使用`with lock:`自动释放，即使代码抛异常也会释放。避免手动`lock.acquire()`和`lock.release()`。

第四个误区是死锁。常见的死锁原因包括：持有多把锁时以不同顺序获取、忘记释放锁、线程相互等待。预防死锁的方法包括：固定加锁顺序、使用超时、使用RLock。

```python
# 潜在死锁：两把锁以不同顺序获取
def task1():
    with lock_a:
        with lock_b:
            pass

def task2():
    with lock_b:
        with lock_a:
            pass
```

## 线程与协程的选择

多线程和协程都可以实现并发，但适用场景不同。

多线程适合：需要真正并行执行（绕过GIL的场景，如IO等待期间）、需要利用多核CPU、任务相对独立、复杂计算配合C扩展释放GIL。

协程适合：高并发IO操作、需要极低并发开销、单线程内实现并发、避免锁和线程安全问题。

现代Python中，asyncio是处理高并发IO的首选方式。只有在asyncio无法满足需求时，才考虑多线程。
