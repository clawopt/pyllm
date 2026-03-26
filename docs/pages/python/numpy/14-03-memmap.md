# 内存映射

内存映射（Memory Mapping）是一种强大的技术，它允许你直接访问存储在磁盘上的大型数组，而不需要将整个数组加载到内存中。当处理超大型数据集（如包含百万级词汇的嵌入矩阵）时，内存映射可以让你在有限内存环境下处理远大于内存容量的数据。NumPy 的 `np.memmap` 提供了这一功能，它创建一个类似数组的对象，但数据存储在磁盘上，只有实际访问的部分才会加载到内存。

## np.memmap 的工作原理

内存映射文件是一种虚拟内存技术。操作系统将文件映射到进程的地址空间，程序可以像访问普通数组一样访问文件内容，但实际的数据加载是按需进行的（lazy loading）。这意味着即你需要处理一个 10GB 的嵌入矩阵，也只需要占用很少的内存。

```python
import numpy as np

# 创建内存映射文件
# mode: 'w+' 写入模式，'r' 只读模式，'c' 读写模式（写入不影响原文件）
fp = np.memmap('memmap_test.bin', dtype=np.float32, mode='w+', shape=(1000, 1000))

# 像普通数组一样操作
fp[:10, :10] = np.random.randn(10, 10)
print(f"前10x10元素:\n{fp[:10, :10]}")

# 写入后需要 flush 才能持久化
fp.flush()

# 删除 memmap 对象
del fp
```

## 加载内存映射

加载内存映射文件非常简单：

```python
# 以只读模式加载
fp = np.memmap('memmap_test.bin', dtype=np.float32, mode='r', shape=(1000, 1000))
print(f"前5x5元素:\n{fp[:5, :5]}")

del fp
```

## 在LLM场景中的应用

### 加载大型词嵌入矩阵

大型语言模型的词嵌入矩阵可能非常大（如包含数百万 token 的词汇表）。内存映射允许你在不占用大量内存的情况下访问这些嵌入：

```python
def create_embedding_memmap(filepath, vocab_size, embedding_dim, dtype=np.float32):
    """创建嵌入矩阵的内存映射

    参数:
        filepath: 内存映射文件路径
        vocab_size: 词汇表大小
        embedding_dim: 嵌入维度
        dtype: 数据类型
    """
    if not filepath.exists():
        fp = np.memmap(filepath, dtype=dtype, mode='w+', shape=(vocab_size, embedding_dim))
        print(f"创建内存映射: 形状={fp.shape}, 大小={fp.nbytes / 1024 / 1024:.1f} MB")
        return fp
    else:
        print(f"文件已存在: {filepath}")

def load_embedding_memmap(filepath, dtype=np.float32):
    """加载嵌入矩阵的内存映射

    参数:
        filepath: 内存映射文件路径
        dtype: 数据类型
    返回:
        内存映射数组（只读）
    """
    fp = np.memmap(filepath, dtype=dtype, mode='r')
    return fp.reshape(-1, 768) if fp.size % 768 == 0 else fp

# 示例：创建大型嵌入矩阵的内存映射
vocab_size = 100000
embedding_dim = 768

# 创建内存映射（模拟）
# fp = create_embedding_memmap('large_embeddings.bin', vocab_size, embedding_dim)

# 随机初始化部分数据
# np.random.seed(42)
# fp[:] = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
# fp.flush()
```

### 按需加载嵌入向量

内存映射的真正威力在于可以高效地只加载需要的嵌入向量：

```python
def get_embedding_vector(memmap_fp, token_id):
    """获取单个 token 的嵌入向量（按需加载）

    参数:
        memmap_fp: 内存映射文件指针
        token_id: token 的整数 ID
    返回:
        嵌入向量 (embedding_dim,)
    """
    return memmap_fp[token_id]

def get_embeddings_batch(memmap_fp, token_ids):
    """批量获取嵌入向量

    参数:
        memmap_fp: 内存映射文件指针
        token_ids: token ID 数组
    返回:
        嵌入向量数组 (batch_size, embedding_dim)
    """
    return memmap_fp[token_ids]

# 示例
# fp = load_embedding_memmap('large_embeddings.bin')
# 获取单个向量
# vec = get_embedding_vector(fp, 100)
# 获取批量向量
# batch = get_embeddings_batch(fp, np.array([0, 1, 2, 3]))
```

### 处理超大型数据集

当数据集远超内存容量时，内存映射可以让像处理普通数组一样处理数据：

```python
def process_large_dataset(filepath, transform_func, output_path):
    """对超大型数据集进行转换（逐块处理）

    参数:
        filepath: 输入内存映射文件路径
        transform_func: 转换函数
        output_path: 输出文件路径
    """
    # 打开输入内存映射（只读）
    input_fp = np.memmap(filepath, dtype=np.float32, mode='r')
    total_size = input_fp.size
    embedding_dim = 768
    vocab_size = total_size // embedding_dim
    input_fp = input_fp.reshape(vocab_size, embedding_dim)

    # 创建输出内存映射
    output_fp = np.memmap(output_path, dtype=np.float32, mode='w+',
                          shape=(vocab_size, embedding_dim))

    # 逐块处理
    chunk_size = 1000
    for i in range(0, vocab_size, chunk_size):
        end = min(i + chunk_size, vocab_size)
        chunk = input_fp[i:end]
        transformed = transform_func(chunk)
        output_fp[i:end] = transformed

        if (i // chunk_size) % 10 == 0:
            print(f"处理进度: {end}/{vocab_size}")

    # 刷新并关闭
    output_fp.flush()
    del input_fp, output_fp

    print(f"处理完成: {output_path}")

# 示例转换函数：L2 归一化
def l2_normalize(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-8)

# 处理大型嵌入矩阵
# process_large_dataset('large_embeddings.bin', l2_normalize, 'normalized_embeddings.bin')
```

## 内存映射 vs 普通加载

内存映射的优势在于处理超大型数据：

```python
import time

# 普通加载 vs 内存映射
large_shape = (1000000, 768)  # 100万词汇，768维

# 创建测试文件
print("创建测试文件...")
fp = np.memmap('temp_large.bin', dtype=np.float32, mode='w+', shape=large_shape)
fp[:] = np.random.randn(*large_shape).astype(np.float32)
fp.flush()
del fp

# 普通加载（占用大量内存）
print("\n普通加载:")
start = time.time()
arr = np.array(np.memmap('temp_large.bin', dtype=np.float32, mode='r', shape=large_shape))
load_time = time.time() - start
print(f"加载时间: {load_time:.2f}s")
print(f"内存占用: {arr.nbytes / 1024 / 1024:.1f} MB")

# 内存映射（按需加载）
print("\n内存映射:")
start = time.time()
mmap_arr = np.memmap('temp_large.bin', dtype=np.float32, mode='r', shape=large_shape)
mmap_time = time.time() - start
print(f"打开时间: {mmap_time:.4f}s")

# 只访问一小部分数据
start = time.time()
_ = mmap_arr[:100]  # 只访问 100 行
access_time = time.time() - start
print(f"访问100行时间: {access_time:.4f}s")

# 清理
del mmap_arr
import os
os.remove('temp_large.bin')
```

## 创建只读的共享嵌入

在生产环境中，可以创建只读的共享嵌入供多个进程使用：

```python
def create_shared_embedding(filepath, embeddings, dtype=np.float32):
    """创建只读共享嵌入文件

    参数:
        filepath: 文件路径
        embeddings: 嵌入矩阵
        dtype: 数据类型
    """
    fp = np.memmap(filepath, dtype=dtype, mode='w+', shape=embeddings.shape)
    fp[:] = embeddings[:]
    fp.flush()
    return fp

def open_shared_embedding(filepath, dtype=np.float32):
    """打开只读共享嵌入

    返回:
        只读内存映射数组
    """
    return np.memmap(filepath, dtype=dtype, mode='r')

# 示例
np.random.seed(42)
embeddings = np.random.randn(50257, 768).astype(np.float32)

# 创建
create_shared_embedding('shared_embeddings.bin', embeddings)

# 打开（只读，多个进程可以共享）
shared = open_shared_embedding('shared_embeddings.bin')
print(f"共享嵌入形状: {shared.shape}")
print(f"访问单个向量: {shared[0][:10]}...")
```

## 常见误区

**误区一：忘记 flush**

写入数据后需要调用 `flush()` 才能持久化到磁盘：

```python
fp = np.memmap('test.bin', dtype=np.float32, mode='w+', shape=(100, 100))
fp[:] = np.random.randn(100, 100)
# 忘记 flush！数据可能丢失
fp.flush()  # 必须调用
```

**误区二：模式选择错误**

`mode` 参数很重要，选错可能导致数据丢失或无法访问：
- `'w+'`: 写入模式，会创建新文件或覆盖已有文件
- `'r'`: 只读模式，修改会报错
- `'r+'`: 读写模式，不创建文件
- `'c'`: 读写模式，但不影响原文件（copy-on-write）

```python
# 正确选择模式
# 创建新文件
fp = np.memmap('new.bin', dtype=np.float32, mode='w+', shape=(100, 100))

# 只读访问
fp = np.memmap('existing.bin', dtype=np.float32, mode='r', shape=(100, 100))

# 读写已有文件
fp = np.memmap('existing.bin', dtype=np.float32, mode='r+', shape=(100, 100))
```

**误区三：删除前未关闭**

删除 memmap 对象前应该关闭文件：

```python
fp = np.memmap('test.bin', dtype=np.float32, mode='w+', shape=(100, 100))
fp[:10] = 1.0
fp.flush()
del fp  # 删除对象
# 或者显式关闭
fp.close()
del fp
```

**误区四：在内存映射上直接进行内存密集型操作**

某些操作（如 `arr.reshape(-1)`）可能触发整个数组的加载：

```python
fp = np.memmap('large.bin', dtype=np.float32, mode='r', shape=(1000000, 768))

# 这个操作会触发整个数组的加载！
# large_arr = fp.reshape(-1)  # 危险！

# 正确做法：先复制到内存（如果数据不大）
small_portion = fp[:1000].copy()
```

## API 总结

| 函数/方法 | 描述 |
|----------|------|
| `np.memmap(filename, dtype, mode, shape)` | 创建或打开内存映射 |
| `memmap.flush()` | 刷新到磁盘 |
| `memmap.close()` | 关闭内存映射 |

内存映射是处理超大型 LLM 数据的利器。掌握这一技术，可以让你在有限内存环境下处理 GB 级别的嵌入矩阵和模型参数。
