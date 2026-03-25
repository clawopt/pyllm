---
title: 多进程编程
---

# 多进程编程

multiprocessing是Python中实现多进程编程的标准模块。它允许程序创建多个进程，每个进程有独立的Python解释器和内存空间，从而突破GIL的限制，真正实现并行执行。

这一章介绍multiprocessing的基本用法，包括进程创建、进程间通信、进程池等内容。

## 为什么需要多进程

Python的GIL让多线程无法真正并行执行。对于CPU密集型任务，比如图像处理、数据分析、机器学习模型训练，多线程反而可能比单线程更慢——因为线程切换有开销。

多进程解决了这个问题。每个进程有独立的Python解释器，也就独立的GIL。多个进程可以在多个CPU核心上真正同时执行。

但多进程也有代价：进程比线程更重量级，创建和切换开销更大；进程间通信比线程间通信更复杂，需要使用Queue、Pipe等机制而不是直接共享内存。

对于IO密集型任务，多线程通常是更好的选择，因为线程切换更轻量，而且IO等待时可以释放GIL。对于CPU密集型任务，如果需要利用多核，多进程是必经之路。

## 创建进程

multiprocessing提供了Process类来创建进程：

```python
import multiprocessing

def worker(name):
    print(f"Hello from {name}")

if __name__ == '__main__':
    p = multiprocessing.Process(target=worker, args=("Worker",))
    p.start()
    p.join()
```

`target`参数指定进程执行的函数，`args`是传递给函数的参数。`start()`启动进程，`join()`等待进程结束。

注意`if __name__ == '__main__':`这一行。在Windows上，这行是必需的，因为Windows没有fork系统调用，进程创建方式不同。如果没有这行保护，import模块时就会创建子进程，导致无限循环。

给进程命名可以方便调试：

```python
p = multiprocessing.Process(target=worker, name="MyWorker")
print(p.name)  # MyWorker
```

进程的常见属性：

```python
p = multiprocessing.Process(target=worker)
p.start()

print(p.pid)      # 进程ID
print(p.is_alive())  # 是否还在运行
p.join()
print(p.exitcode)  # 退出码
```

## 进程间通信

进程之间不能直接共享内存，需要通过通信机制交换数据。multiprocessing提供了几种通信方式。

### Queue队列

Queue是最常用的进程间通信方式，线程安全和进程安全：

```python
import multiprocessing

def producer(queue):
    queue.put("Hello")
    queue.put("World")

def consumer(queue):
    while not queue.empty():
        print(queue.get())

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=producer, args=(queue,))
    p2 = multiprocessing.Process(target=consumer, args=(queue,))

    p1.start()
    p1.join()
    p2.start()
    p2.join()
```

生产者和消费者通过Queue传递消息。`put()`方法放入数据，`get()`方法取出数据。Queue是阻塞的，当队列为空时，`get()`会等待。

### Pipe管道

Pipe提供双向通信能力：

```python
import multiprocessing

def worker(conn):
    conn.send("Hello from worker")
    response = conn.recv()
    print(f"Worker received: {response}")
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = multiprocessing.Pipe()

    p = multiprocessing.Process(target=worker, args=(child_conn,))
    p.start()

    print(f"Parent received: {parent_conn.recv()}")
    parent_conn.send("Hello from parent")

    p.join()
```

`Pipe()`返回一对连接对象，一端发送给子进程，一端留在父进程。`send()`发送对象，`recv()`接收对象。Pipe比Queue更快，但只适合两个进程之间的直接通信。

### 共享内存

对于需要高效共享少量数据的场景，可以使用共享内存：

```python
import multiprocessing

def worker(shared_value, shared_array):
    shared_value.value = 42
    shared_array[0] = 99

if __name__ == '__main__':
    shared_value = multiprocessing.Value('i', 0)
    shared_array = multiprocessing.Array('i', [0, 0, 0])

    p = multiprocessing.Process(target=worker, args=(shared_value, shared_array))
    p.start()
    p.join()

    print(shared_value.value)  # 42
    print(shared_array[:])    # [99, 0, 0]
```

`Value`和`Array`创建共享对象。第一个参数是数据类型代码，比如'i'表示有符号整数。共享内存的访问需要锁来保证原子性，但读写简单数据类型通常是原子的。

## 进程池

手动管理多个进程很繁琐。进程池自动管理进程的创建和销毁，简化了并行任务的提交：

```python
import multiprocessing

def square(x):
    return x * x

if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(square, [1, 2, 3, 4, 5])
        print(results)  # [1, 4, 9, 16, 25]
```

`Pool`创建指定数量进程的池。`map()`方法将函数应用到列表的每个元素，返回结果列表。`Pool`使用上下文管理器，自动关闭进程池。

`map()`是阻塞的，会等待所有任务完成。如果想异步执行，可以使用`apply_async()`：

```python
with multiprocessing.Pool(processes=4) as pool:
    result = pool.apply_async(square, (5,))
    print(result.get())  # 25

    results = pool.map_async(square, [1, 2, 3])
    print(results.get())  # [1, 4, 9]
```

`apply_async()`提交单个任务，`map_async()`提交多个任务。它们都返回AsyncResult对象，用`get()`获取结果。

进程池还支持回调函数：

```python
def callback(result):
    print(f"Task completed: {result}")

with multiprocessing.Pool(processes=4) as pool:
    pool.apply_async(square, (5,), callback=callback)
    pool.close()
    pool.join()
```

回调函数在任务完成后自动调用，适合处理耗时的后处理工作。

## 进程同步

当多个进程访问共享资源时，需要同步机制防止竞态条件。

`Lock`是最基本的同步原语：

```python
import multiprocessing

counter = multiprocessing.Value('i', 0)
lock = multiprocessing.Lock()

def increment():
    for _ in range(100000):
        with lock:
            counter.value += 1

if __name__ == '__main__':
    processes = [multiprocessing.Process(target=increment) for _ in range(4)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(counter.value)  # 400000
```

`with lock:`语句确保同一时间只有一个进程执行缩进内的代码。没有锁的情况下，由于`counter.value += 1`不是原子操作，最终结果会小于400000。

`RLock`是可重入锁，允许同一进程多次获取：

```python
lock = multiprocessing.RLock()

with lock:
    with lock:  # 同一进程可以再次获取
        pass
```

`Semaphore`控制同时访问资源的进程数量：

```python
semaphore = multiprocessing.Semaphore(2)

def worker(n):
    with semaphore:
        print(f"Worker {n} acquired semaphore")
        # 最多2个进程同时执行这里
        print(f"Worker {n} released semaphore")

multiprocessing.Process(target=worker, args=(1,)).start()
multiprocessing.Process(target=worker, args=(2,)).start()
multiprocessing.Process(target=worker, args=(3,)).start()
```

信号量常用于限制并发连接数，比如数据库连接池。

`Event`用于进程间信号通知：

```python
event = multiprocessing.Event()

def waiter():
    print("Waiting for event")
    event.wait()
    print("Event triggered!")

def setter():
    print("Setting event")
    event.set()

p1 = multiprocessing.Process(target=waiter)
p2 = multiprocessing.Process(target=setter)

p1.start()
p2.start()
p1.join()
p2.join()
```

`wait()`阻塞等待事件，`set()`触发事件。

## 进程间数据传递注意

进程间传递的数据会被序列化（pickle）后复制到子进程。返回的结果也会序列化后复制回父进程。这意味着：

1. 传递大对象会有复制开销
2. 传递不可pickle的对象会失败（比如lambda、锁对象）
3. 进程间不共享对象，只是传递副本

```python
import multiprocessing

def worker(obj):
    # obj是副本，修改不影响父进程
    obj['modified'] = True
    return obj

if __name__ == '__main__':
    data = {'original': True}
    with multiprocessing.Pool(1) as pool:
        result = pool.apply_async(worker, (data,))
        print(data)        # {'original': True}，未改变
        print(result.get()) # {'original': True, 'modified': True}
```

如果要真正共享数据，使用共享内存或管理器对象。

## Manager对象

`multiprocessing.Manager`提供更灵活的进程间共享方式：

```python
import multiprocessing

def worker(shared_dict, shared_list):
    shared_dict['key'] = 'value'
    shared_list.append(1)

if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict()
        shared_list = manager.list()

        p = multiprocessing.Process(target=worker, args=(shared_dict, shared_list))
        p.start()
        p.join()

        print(shared_dict)  # {'key': 'value'}
        print(shared_list)  # [1]
```

Manager通过代理对象实现进程间共享，比共享内存更灵活，但速度更慢。适用于需要共享复杂数据结构或需要跨网络共享的场景。

## 常见误区

第一个误区是不使用`if __name__ == '__main__':`。在Windows上，模块import时会执行顶级代码，导致子进程重复执行任务。

```python
# 错误
import multiprocessing

def worker():
    print("Working")

p = multiprocessing.Process(target=worker)
p.start()

# 正确
if __name__ == '__main__':
    p = multiprocessing.Process(target=worker)
    p.start()
```

第二个误区是在多进程中使用锁保护共享资源时，不清楚锁的传递方式。multiprocessing的锁可以pickle，但最好通过参数显式传递。

第三个误区是创建过多进程。进程创建和切换开销大，创建几百个进程可能耗尽系统资源。如果任务数量大，使用进程池限制并发数。

第四个误区是混淆进程池的`map`和`imap`。`map`返回完整列表，`imap`返回迭代器。对于大量任务，`imap`可以边处理边返回结果，减少内存占用。

```python
with multiprocessing.Pool() as pool:
    for result in pool.imap(heavy_task, large_list):
        process_result(result)
```
