# NumPy DataLoader

本篇文章介绍如何创建一个高效的基于 NumPy 数组的 DataLoader，用于 LLM 训练时的批量数据加载。

## 简单的 NumPy DataLoader

```python
import numpy as np

class NumPyDataLoader:
    """基于 NumPy 的简单 DataLoader"""

    def __init__(self, data, batch_size=32, shuffle=True, seed=None):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.num_samples = len(next(iter(data.values())))
        self.indices = np.arange(self.num_samples)

        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            yield self._get_batch(batch_indices)

    def _get_batch(self, indices):
        batch = {}
        for key, value in self.data.items():
            batch[key] = value[indices]
        return batch

# 示例
data = {
    'input_ids': np.random.randint(0, 50000, size=(1000, 128)),
    'attention_mask': np.ones((1000, 128), dtype=np.int32),
    'labels': np.random.randint(0, 50000, size=(1000, 128))
}

loader = NumPyDataLoader(data, batch_size=32, shuffle=True)

for i, batch in enumerate(loader):
    print(f"Batch {i}: input_ids shape={batch['input_ids'].shape}")
    if i >= 2:
        break
```

## 带 collate_fn 的 DataLoader

```python
class DataLoaderWithCollate:
    """支持 collate_fn 的 DataLoader"""

    def __init__(self, data, batch_size=32, shuffle=True, collate_fn=None, seed=None):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or self.default_collate
        self.rng = np.random.default_rng(seed)
        self.num_samples = len(next(iter(data.values())))
        self.indices = np.arange(self.num_samples)

        if self.shuffle:
            self.rng.shuffle(self.indices)

    def default_collate(self, batch):
        """默认的 collate 函数"""
        return batch

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

        batch = []
        for idx in self.indices:
            batch.append({key: value[idx] for key, value in self.data.items()})

            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        if batch:
            yield self.collate_fn(batch)

# 示例 collate_fn
def simple_collate(samples):
    """简单的 collate 函数"""
    input_ids = np.array([s['input_ids'] for s in samples])
    attention_mask = np.array([s['attention_mask'] for s in samples])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

data_list = [
    {'input_ids': np.array([1, 2, 3]), 'attention_mask': np.array([1, 1, 1])},
    {'input_ids': np.array([4, 5, 6]), 'attention_mask': np.array([1, 1, 1])},
    {'input_ids': np.array([7, 8, 9]), 'attention_mask': np.array([1, 1, 1])},
]

data_dict = {
    'input_ids': np.array([d['input_ids'] for d in data_list]),
    'attention_mask': np.array([d['attention_mask'] for d in data_list])
}

loader = DataLoaderWithCollate(data_dict, batch_size=2, collate_fn=simple_collate)
for batch in loader:
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
```

掌握 NumPy DataLoader 的实现对于高效训练数据加载非常重要。
