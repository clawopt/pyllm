# HuggingFace Datasets 转 NumPy

HuggingFace `datasets` 库是处理大规模数据集的工具。本篇文章介绍如何将 HuggingFace `datasets` 格式的数据转换为 NumPy 数组，以便在纯 NumPy 环境中处理。

## HuggingFace Datasets 基本结构

```python
import numpy as np

class MockHFdataset:
    """模拟 HuggingFace Dataset 结构"""

    def __init__(self, data_dict):
        self.data = data_dict
        self.num_rows = len(next(iter(data_dict.values())))

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def to_numpy(self):
        """转换为 NumPy 数组字典"""
        return {k: np.array(v) for k, v in self.data.items()}

# 示例
mock_data = {
    'input_ids': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    'attention_mask': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
}

dataset = MockHFdataset(mock_data)
print(f"数据集大小: {len(dataset)}")
print(f"样本 0: {dataset[0]}")
```

## 数据集转 NumPy

```python
def dataset_to_numpy(dataset):
    """将 HuggingFace Dataset 转换为 NumPy 数组

    参数:
        dataset: HuggingFace Dataset 对象
    返回:
        numpy_data: 包含 NumPy 数组的字典
    """
    numpy_data = {}

    for key in dataset.features:
        numpy_data[key] = np.array(dataset[key])

    return numpy_data

# 示例
mock_data = {
    'input_ids': [[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]],
    'attention_mask': [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
    'labels': [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
}

dataset = MockHFdataset(mock_data)
numpy_data = dataset_to_numpy(dataset)

print("转换后的 NumPy 数组:")
for key, arr in numpy_data.items():
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
```

## 批量转换大数据集

```python
def dataset_to_numpy_batched(dataset, batch_size=1000):
    """分批将数据集转换为 NumPy 数组

    适用于大型数据集，避免内存溢出
    """
    numpy_data = None

    for i in range(0, len(dataset), batch_size):
        end = min(i + batch_size, len(dataset))

        # 获取 batch
        batch = dataset[i:end]
        batch_arrays = {k: np.array(v) for k, v in batch.items()}

        # 合并
        if numpy_data is None:
            numpy_data = {k: [arr] for k, arr in batch_arrays.items()}
        else:
            for k, arr in batch_arrays.items():
                numpy_data[k].append(arr)

    # 拼接
    for k in numpy_data:
        numpy_data[k] = np.concatenate(numpy_data[k])

    return numpy_data

# 示例
large_mock = {
    'input_ids': [list(range(i, i+10)) for i in range(100)],
    'labels': [[i % 2] * 10 for i in range(100)]
}

large_dataset = MockHFdataset(large_mock)
numpy_data = dataset_to_numpy_batched(large_dataset, batch_size=30)

print(f"大数据集转换: {numpy_data['input_ids'].shape}")
```

## 保存和加载

```python
def save_numpy_data(data, filepath_prefix):
    """将 NumPy 数据保存到文件

    参数:
        data: NumPy 数组字典
        filepath_prefix: 文件路径前缀
    """
    for key, arr in data.items():
        filepath = f"{filepath_prefix}_{key}.npy"
        np.save(filepath, arr)
        print(f"保存 {key} 到 {filepath}")

def load_numpy_data(filepath_prefix, keys):
    """加载 NumPy 数据"""
    data = {}
    for key in keys:
        filepath = f"{filepath_prefix}_{key}.npy"
        data[key] = np.load(filepath)
    return data

# 示例
# save_numpy_data(numpy_data, 'train_data')
# loaded = load_numpy_data('train_data', ['input_ids', 'attention_mask'])
```

掌握 HuggingFace datasets 与 NumPy 的转换对于高效处理训练数据非常重要。
