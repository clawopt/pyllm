# 二进制文件输入输出

NumPy 的二进制文件格式是存储和加载大型数组最高效的方式。相比文本格式，二进制格式的读写速度可以快几十倍甚至上百倍，同时文件体积也更小。在 LLM 应用中，模型的词嵌入矩阵、预训练参数等大型数据几乎总是以二进制格式存储。`np.save`、`np.savez` 和 `np.load` 是 NumPy 提供的三个核心二进制 IO 函数，它们使用轻量级的 .npy 或 .npz 格式，既保证了效率，又便于跨平台使用。

## np.save：保存单个数组

`np.save` 是保存单个数组到 .npy 文件的标准方法：

```python
import numpy as np

# 保存单个数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
np.save('array.npy', arr)

# 保存后可以用 np.load 加载
loaded = np.load('array.npy')
print(f"加载的数据:\n{loaded}")
```

### 自动添加扩展名

`np.save` 会自动添加 .npy 扩展名：

```python
# 两种写法效果相同
np.save('array', arr)  # 保存为 'array.npy'
np.save('array.npy', arr)  # 也是 'array.npy'
```

### 保存其他格式

默认情况下保存为 NumPy 的 .npy 格式。如果需要保存为其他二进制格式（如原始字节），可以使用 `tofile` 方法：

```python
# 使用 tofile 保存为原始二进制（无元数据）
arr.tofile('array.raw')  # 保存为原始字节，无形状/dtype 信息
# 加载时需要手动指定 dtype 和 shape
loaded_raw = np.fromfile('array.raw', dtype=arr.dtype).reshape(arr.shape)
```

## np.savez：保存多个数组

`np.savez` 可以在一个 .npz 文件中保存多个数组：

```python
# 保存多个数组到一个 .npz 文件
embedding = np.random.randn(50257, 768).astype(np.float32)
position_embedding = np.random.randn(1024, 768).astype(np.float32)
labels = np.arange(50257)

np.savez('embeddings.npz', 
         embedding=embedding, 
         position_embedding=position_embedding,
         labels=labels)

# 加载
data = np.load('embeddings.npz')
print(f"Keys: {data.files}")  # ['embedding', 'position_embedding', 'labels']
print(f"Embedding shape: {data['embedding'].shape}")
print(f"Position embedding shape: {data['position_embedding'].shape}")
```

### np.savez_compressed：压缩保存

如果磁盘空间有限，可以使用压缩格式：

```python
# 压缩保存（使用 gzip 压缩）
np.savez_compressed('embeddings_compressed.npz', 
                    embedding=embedding,
                    position_embedding=position_embedding)

# 压缩效果对比
import os
normal_size = os.path.getsize('embeddings.npz')
compressed_size = os.path.getsize('embeddings_compressed.npz')
print(f"普通: {normal_size / 1024 / 1024:.1f} MB")
print(f"压缩: {compressed_size / 1024 / 1024:.1f} MB")
```

## np.load：加载数组

`np.load` 用于加载 .npy 或 .npz 文件：

```python
# 加载 .npy 文件
arr = np.load('array.npy')

# 加载 .npz 文件（返回类似字典的对象）
data = np.load('embeddings.npz')
embedding = data['embedding']
position_embedding = data['position_embedding']
data.close()  # 关闭文件释放资源
```

### 使用上下文管理器自动关闭

```python
# 使用 with 语句自动关闭
with np.load('embeddings.npz') as data:
    embedding = data['embedding']
    position_embedding = data['position_embedding']
# 文件在这里自动关闭
```

## 在LLM场景中的应用

### 保存和加载预训练词嵌入

词嵌入矩阵通常很大，高效的二进制存储非常重要：

```python
def save_embeddings(embeddings, filepath):
    """保存词嵌入矩阵

    参数:
        embeddings: 嵌入矩阵 (vocab_size, embedding_dim)
        filepath: 保存路径
    """
    np.save(filepath, embeddings.astype(np.float32))
    print(f"嵌入矩阵已保存: {filepath}")
    print(f"形状: {embeddings.shape}, 大小: {embeddings.nbytes / 1024 / 1024:.1f} MB")

def load_embeddings(filepath):
    """加载词嵌入矩阵"""
    return np.load(filepath)

# 示例
np.random.seed(42)
embeddings = np.random.randn(50257, 768).astype(np.float32)
save_embeddings(embeddings, 'gpt2_embeddings.npy')

# 加载
loaded = load_embeddings('gpt2_embeddings.npy')
print(f"加载后形状: {loaded.shape}")
print(f"加载后 dtype: {loaded.dtype}")
```

### 保存和加载模型参数

模型的所有参数可以打包保存：

```python
def save_model_weights(weights_dict, filepath):
    """保存模型权重

    参数:
        weights_dict: 权重字典 {name: array}
        filepath: 保存路径
    """
    np.savez(filepath, **weights_dict)
    print(f"模型权重已保存: {filepath}")

def load_model_weights(filepath):
    """加载模型权重"""
    return np.load(filepath)

# 示例：保存 Transformer 权重
weights = {
    'token_embedding': np.random.randn(50257, 768).astype(np.float32),
    'position_embedding': np.random.randn(1024, 768).astype(np.float32),
    'layer.0.attn.W_q': np.random.randn(768, 768).astype(np.float32),
    'layer.0.attn.W_k': np.random.randn(768, 768).astype(np.float32),
    'layer.0.attn.W_v': np.random.randn(768, 768).astype(np.float32),
    'layer.0.attn.W_o': np.random.randn(768, 768).astype(np.float32),
}

save_model_weights(weights, 'transformer_weights.npz')

# 加载
loaded_weights = load_model_weights('transformer_weights.npz')
print(f"加载的权重键: {list(loaded_weights.keys())}")
```

### 增量保存大型检查点

对于超大型模型，可能需要分块保存和加载：

```python
def save_model_sharded(weights_dict, save_dir):
    """分片保存模型权重

    将大矩阵分成多个小文件保存
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    shard_files = []
    for name, weight in weights_dict.items():
        shard_path = os.path.join(save_dir, f'{name}.npy')
        np.save(shard_path, weight)
        shard_files.append(shard_path)
        print(f"保存 {name}: {weight.shape}")

    # 保存索引文件
    index = {name: path for name, path in zip(weights_dict.keys(), shard_files)}
    np.savez(os.path.join(save_dir, 'index.npz'), **index)

    return shard_files

def load_model_sharded(save_dir):
    """加载分片模型权重"""
    import os
    index_path = os.path.join(save_dir, 'index.npz')
    index = np.load(index_path, allow_pickle=True)

    weights = {}
    for name, path in index.items():
        weights[name] = np.load(path, allow_pickle=True)

    return weights

# 示例
save_model_sharded(weights, 'model_sharded')
loaded = load_model_sharded('model_sharded')
print(f"加载的权重: {list(loaded.keys())}")
```

## 内存映射大型数组

对于超大数组，可以结合内存映射使用：

```python
def save_large_array_chunked(arr, filepath, chunk_size=1000):
    """分块保存大型数组

    适用于无法一次性加载到内存的数组
    """
    num_chunks = (arr.shape[0] + chunk_size - 1) // chunk_size
    
    with open(filepath, 'wb') as f:
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, arr.shape[0])
            chunk = arr[start:end]
            np.save(f, chunk)
            print(f"保存 chunk {i+1}/{num_chunks}")
```

## 常见误区

**误区一：忘记 .npy 文件是二进制格式**

不能用文本编辑器直接查看 .npy 文件，需要用 Python 加载：

```python
# 正确加载
arr = np.load('array.npy')

# 错误：用文本方式打开
# with open('array.npy', 'r') as f:  # 会看到乱码
```

**误区二：忘记关闭 .npz 文件**

`.npz` 文件是多个数组的压缩包，需要显式关闭：

```python
# 忘记关闭可能导致资源泄露
data = np.load('weights.npz')
# ... 使用 data ...
# 忘记 data.close()

# 推荐：使用 with 语句
with np.load('weights.npz') as data:
    weights = data['embedding']
```

**误区三：加载时忽略 allow_pickle 安全警告**

`allow_pickle=True` 存在安全风险（可以执行任意代码），但在某些情况下是必要的：

```python
# 加载包含 Python 对象的数组时需要 allow_pickle=True
# 但要注意安全性
data = np.load('data_with_objects.npz', allow_pickle=True)
```

**误区四：保存前忘记转换 dtype**

为了节省空间，应该在保存前转换为合适的 dtype：

```python
# 默认 float64，占用空间大
arr_float64 = np.random.randn(50257, 768)

# 转换为 float32 节省一半空间
arr_float32 = arr_float64.astype(np.float32)
np.save('embeddings.npy', arr_float32)
print(f"float64: {arr_float64.nbytes / 1024 / 1024:.1f} MB")
print(f"float32: {arr_float32.nbytes / 1024 / 1024:.1f} MB")
```

## 性能对比

二进制格式比文本格式快很多：

```python
import time

arr = np.random.randn(10000, 768).astype(np.float32)

# 测量 save
start = time.time()
np.save('text_temp.csv', arr)  # 文本格式
text_time = time.time() - start

start = time.time()
np.save('binary_temp.npy', arr)  # 二进制格式
binary_time = time.time() - start

# 测量 load
start = time.time()
_ = np.loadtxt('text_temp.csv', delimiter=',')
text_load_time = time.time() - start

start = time.time()
_ = np.load('binary_temp.npy')
binary_load_time = time.time() - start

import os
text_size = os.path.getsize('text_temp.csv')
binary_size = os.path.getsize('binary_temp.npy')

print(f"文本: {text_size / 1024 / 1024:.1f} MB, 保存 {text_time:.2f}s, 加载 {text_load_time:.2f}s")
print(f"二进制: {binary_size / 1024 / 1024:.1f} MB, 保存 {binary_time:.4f}s, 加载 {binary_load_time:.4f}s")
```

## API 总结

| 函数 | 用途 | 说明 |
|------|------|------|
| `np.save(file, arr)` | 保存单个数组 | 自动添加 .npy 扩展名 |
| `np.savez(file, *args, **kwds)` | 保存多个数组 | 生成 .npz 文件 |
| `np.savez_compressed(file, *args)` | 压缩保存 | 使用 gzip 压缩 |
| `np.load(file)` | 加载数组 | 返回数组或类似字典对象 |

二进制 IO 是处理 LLM 大型数据的首选方式。掌握这些技巧可以让你高效地保存和加载模型权重、词嵌入等大型数据。
