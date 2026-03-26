# 将HuggingFace Dataset导出为结构化数组

HuggingFace datasets 库是处理 LLM 训练数据的主流工具，它提供了高效的数据加载和预处理功能。然而，在某些场景下，你可能希望将 HuggingFace dataset 导出为 NumPy 结构化数组，以便进行更底层的操作或与其他系统集成。本篇文章介绍如何将 HuggingFace dataset 转换为结构化数组，包括直接转换、批量处理、以及内存优化技巧。

## HuggingFace Dataset 基本结构

HuggingFace dataset 的核心是 `Dataset` 对象，它本质上是一个 Arrow 数据表，每列数据称为一个特征（feature）：

```python
# 伪代码：HuggingFace dataset 的典型结构
# from datasets import Dataset
# dataset = Dataset.from_dict({
#     'input_ids': [[464, 197, 35, ...], [...]],
#     'attention_mask': [[1, 1, 1, ...], [...]],
#     'labels': [[464, 197, 35, ...], [...]]
# })
# print(dataset)
# Dataset({
#     features: ['input_ids', 'attention_mask', 'labels'],
#     num_rows: 1000
# })
```

## 导出为NumPy数组

### 转换整个Dataset为字典

最简单的方法是将整个 dataset 转换为字典，然后创建结构化数组：

```python
def hf_dataset_to_structured_array(dataset, max_length=None):
    """将 HuggingFace Dataset 转换为结构化数组

    参数:
        dataset: HuggingFace Dataset 对象
        max_length: 序列最大长度（用于固定数组形状）
    返回:
        结构化数组
    """
    # 获取特征信息
    features = dataset.features
    num_rows = len(dataset)

    # 定义 dtype
    dtype_list = []

    # 处理 input_ids（整数列表）
    if 'input_ids' in dataset.column_names:
        input_dim = max_length or len(dataset[0]['input_ids'])
        dtype_list.append(('input_ids', 'i4', (input_dim,)))

    # 处理 attention_mask
    if 'attention_mask' in dataset.column_names:
        mask_dim = max_length or len(dataset[0]['attention_mask'])
        dtype_list.append(('attention_mask', 'i1', (mask_dim,)))

    # 处理 labels
    if 'labels' in dataset.column_names:
        label_dim = max_length or len(dataset[0]['labels'])
        dtype_list.append(('labels', 'i4', (label_dim,)))

    # 处理 token_type_ids（BERT 等模型）
    if 'token_type_ids' in dataset.column_names:
        seg_dim = max_length or len(dataset[0]['token_type_ids'])
        dtype_list.append(('token_type_ids', 'i1', (seg_dim,)))

    # 创建结构化数组
    arr = np.zeros(num_rows, dtype=dtype_list)

    # 填充数据
    for i in range(num_rows):
        sample = dataset[i]
        if 'input_ids' in sample:
            arr[i]['input_ids'][:len(sample['input_ids'])] = sample['input_ids']
        if 'attention_mask' in sample:
            arr[i]['attention_mask'][:len(sample['attention_mask'])] = sample['attention_mask']
        if 'labels' in sample:
            arr[i]['labels'][:len(sample['labels'])] = sample['labels']
        if 'token_type_ids' in sample:
            arr[i]['token_type_ids'][:len(sample['token_type_ids'])] = sample['token_type_ids']

    return arr

# 示例：模拟 HuggingFace dataset
def create_mock_dataset(num_samples=100, seq_len=128):
    """创建模拟的 HuggingFace dataset"""
    dataset = {
        'input_ids': np.random.randint(0, 50257, size=(num_samples, seq_len)),
        'attention_mask': np.ones((num_samples, seq_len), dtype=np.int32),
        'labels': np.random.randint(0, 50257, size=(num_samples, seq_len)),
    }
    return dataset

mock_data = create_mock_dataset(num_samples=1000, seq_len=128)
print(f"模拟数据 input_ids 形状: {mock_data['input_ids'].shape}")
```

### 使用批量转换提高效率

对于大型 dataset，批量处理比逐行转换更高效：

```python
def batch_convert_to_structured(data_dict, batch_size=1000):
    """批量将字典数据转换为结构化数组

    参数:
        data_dict: 包含 'input_ids', 'attention_mask' 等键的字典
        batch_size: 批处理大小
    返回:
        结构化数组
    """
    num_samples = len(data_dict['input_ids'])
    seq_len = data_dict['input_ids'].shape[1]

    # 定义 dtype
    dtype = [
        ('input_ids', 'i4', (seq_len,)),
        ('attention_mask', 'i1', (seq_len,)),
        ('labels', 'i4', (seq_len,)),
    ]

    # 创建结构化数组
    arr = np.zeros(num_samples, dtype=dtype)

    # 批量复制数据
    arr['input_ids'] = data_dict['input_ids']
    arr['attention_mask'] = data_dict['attention_mask'].astype('i1')
    arr['labels'] = data_dict['labels']

    return arr

# 批量转换
structured_data = batch_convert_to_structured(mock_data)
print(f"结构化数组形状: {structured_data.shape}")
print(f"结构化数组 dtype: {structured_data.dtype}")
print(f"第一个样本 input_ids 前5个: {structured_data[0]['input_ids'][:5]}")
```

## 处理变长序列

语言模型中的序列通常是变长的（不同样本的有效长度不同）。结构化数组要求固定形状，但可以通过填充来处理：

```python
def create_variable_length_structured(data_dict, pad_token_id=0):
    """处理变长序列，创建填充后的结构化数组

    参数:
        data_dict: 包含变长序列的字典
        pad_token_id: 填充 token 的 ID
    返回:
        填充后的结构化数组
    """
    # 找到最大长度
    max_len = max(len(seq) for seq in data_dict['input_ids'])
    num_samples = len(data_dict['input_ids'])

    # 定义 dtype
    dtype = [
        ('input_ids', 'i4', (max_len,)),
        ('attention_mask', 'i1', (max_len,)),
        ('valid_length', 'i4'),  # 添加有效长度字段
    ]

    arr = np.zeros(num_samples, dtype=dtype)

    # 填充数据
    for i in range(num_samples):
        seq_len = len(data_dict['input_ids'][i])
        arr[i]['input_ids'][:seq_len] = data_dict['input_ids'][i]
        arr[i]['input_ids'][seq_len:] = pad_token_id
        arr[i]['attention_mask'][:seq_len] = data_dict.get('attention_mask', [[1]*seq_len])[i]
        arr[i]['attention_mask'][seq_len:] = 0
        arr[i]['valid_length'] = seq_len

    return arr

# 变长序列示例
variable_data = {
    'input_ids': [np.array([464, 197, 35]), np.array([100, 200]), np.array([1, 2, 3, 4, 5])],
    'attention_mask': [np.array([1, 1, 1]), np.array([1, 1]), np.array([1, 1, 1, 1, 1])]
}

# 转换为填充的结构化数组
# 注意：上面的数据是列表，实际使用时需要先转换为数组
```

## 导出到文件

转换后的结构化数组可以保存到文件以便后续快速加载：

```python
def save_structured_array(arr, filepath):
    """保存结构化数组到 .npy 文件"""
    np.save(filepath, arr)
    print(f"已保存到 {filepath}")
    print(f"文件大小: {arr.nbytes / 1024 / 1024:.2f} MB")

def load_structured_array(filepath):
    """加载结构化数组"""
    return np.load(filepath, allow_pickle=True)

# 保存
save_structured_array(structured_data, 'training_data.npy')

# 加载
loaded_data = load_structured_array('training_data.npy')
print(f"加载后 dtype: {loaded_data.dtype}")
```

## 内存优化技巧

### 使用更小的数据类型

减少结构化数组的内存占用：

```python
def get_optimized_dtype():
    """获取优化的 dtype，减少内存占用"""
    return [
        ('input_ids', 'i2', (128,)),  # int16: 50257 < 32767，用 i2 足够（如果 vocab < 32767）
        # 注意：GPT-2 的 vocab_size 是 50257，需要 i4（int32）
        ('input_ids', 'i4', (128,)),  # 保持 int32
        ('attention_mask', 'i1', (128,)),  # int8: 0 或 1
        ('labels', 'i4', (128,)),
    ]

def estimate_memory_usage(num_samples, seq_len):
    """估算内存使用"""
    bytes_per_sample = (
        seq_len * 4 +  # input_ids: int32
        seq_len * 1 +  # attention_mask: int8
        seq_len * 4 +  # labels: int32
        4  # valid_length: int32
    )
    total_bytes = num_samples * bytes_per_sample
    return total_bytes / 1024 / 1024

num_samples = 100000
seq_len = 512
memory_mb = estimate_memory_usage(num_samples, seq_len)
print(f"估算内存占用: {memory_mb:.1f} MB")
```

### 使用内存映射处理大型数据集

对于超大型数据集，使用内存映射避免一次性加载：

```python
def create_memmap_dataset(filepath, shape, dtype):
    """创建内存映射的数据集文件

    参数:
        filepath: 文件路径
        shape: 数据形状
        dtype: 数据类型
    返回:
        memmap 数组
    """
    if not filepath.exists():
        fp = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
    else:
        fp = np.memmap(filepath, dtype=dtype, mode='r+', shape=shape)
    return fp

# 示例：创建超大数据集
# shape = (1000000, 512)  # 100万样本，512 序列长度
# memmap = create_memmap_dataset('large_dataset.dat', shape, 'i4')
```

## 与DataLoader集成

将结构化数组转换为 PyTorch DataLoader 兼容格式：

```python
def structured_to_torch_batch(arr, device='cpu'):
    """将结构化数组 batch 转换为 PyTorch 张量

    参数:
        arr: 结构化数组
        device: 目标设备
    返回:
        包含张量的字典
    """
    # 注意：这里只是示例，实际使用需要 torch
    batch = {
        'input_ids': arr['input_ids'],
        'attention_mask': arr['attention_mask'],
        'labels': arr['labels'],
    }
    return batch

# 示例 batch
batch = structured_to_torch_batch(structured_data[:32])
print(f"Batch keys: {batch.keys()}")
print(f"input_ids shape: {batch['input_ids'].shape}")
```

## 常见误区

**误区一：忽略 dtype 的字节序**

在不同平台间传输数据时，字节序可能不同：

```python
# 指定字节序
dtype = [
    ('input_ids', '>i4', (128,)),  # 大端序
    ('attention_mask', 'i1', (128,)),
]

# 或使用 native byte order
dtype = [
    ('input_ids', np.dtype('i4').newbyteorder('='), (128,)),
]
```

**误区二：直接修改 Arrow 数据**

HuggingFace dataset 基于 Arrow，修改需要特殊处理。转换为 NumPy 数组后可以自由修改：

```python
# HuggingFace dataset 通常是只读的
# 转换为 NumPy 数组后可以修改
arr = hf_dataset_to_structured_array(dataset)
arr['input_ids'][0] = 999  # 可行
```

**误区三：忘记处理 padding**

变长序列不填充会导致数组形状不一致：

```python
# 确保所有序列长度一致
max_len = max(len(seq) for seq in data_dict['input_ids'])
# 填充或截断到 max_len
```

## 完整工作流示例

```python
def full_pipeline(dataset_path, output_path, max_length=512):
    """完整的数据导出 pipeline

    1. 加载 HuggingFace dataset
    2. 转换为结构化数组
    3. 保存到文件
    """
    # Step 1: 加载 dataset（伪代码）
    # dataset = load_from_disk(dataset_path)

    # Step 2: 模拟加载
    num_samples = 10000
    dataset = {
        'input_ids': np.random.randint(0, 50257, size=(num_samples, max_length)),
        'attention_mask': np.ones((num_samples, max_length), dtype=np.int32),
        'labels': np.random.randint(0, 50257, size=(num_samples, max_length)),
    }

    # Step 3: 批量转换为结构化数组
    structured = batch_convert_to_structured(dataset)

    # Step 4: 保存
    save_structured_array(structured, output_path)

    return structured

# 运行 pipeline
data = full_pipeline('path/to/dataset', 'training_data.npy')
print(f"导出完成: {len(data)} 样本")
```

掌握将 HuggingFace dataset 导出为结构化数组的技巧，可以让你在需要时快速切换到纯 NumPy 环境进行数据处理，或者将数据与其他系统集成。
