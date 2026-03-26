# PyTorch 与 NumPy 互操作

PyTorch 是目前最流行的深度学习框架之一，而 PyTorch 与 NumPy 之间可以无缝互操作。本篇文章介绍 `torch.from_numpy()` 和 `tensor.numpy()` 的用法，以及如何在两个框架之间高效传递数据。

## torch.from_numpy()：NumPy 转 PyTorch

```python
import numpy as np

def numpy_to_torch():
    """NumPy 数组转 PyTorch 张量"""
    print("=== NumPy -> PyTorch ===\n")

    # 创建 NumPy 数组
    np_array = np.random.randn(3, 4).astype(np.float32)
    print(f"NumPy 数组 dtype: {np_array.dtype}")
    print(f"NumPy 数组形状: {np_array.shape}")

    # NumPy 转 PyTorch（共享内存）
    # import torch
    # torch_tensor = torch.from_numpy(np_array)
    # print(f"PyTorch 张量 dtype: {torch_tensor.dtype}")
    # print(f"PyTorch 张量形状: {torch_tensor.shape}")
    print("\n代码示例:")
    print("  torch_tensor = torch.from_numpy(np_array)")

numpy_to_torch()
```

## tensor.numpy()：PyTorch 转 NumPy

```python
def torch_to_numpy():
    """PyTorch 张量转 NumPy 数组"""
    print("\n=== PyTorch -> NumPy ===\n")

    # 创建 PyTorch 张量
    # torch_tensor = torch.randn(3, 4)
    # print(f"PyTorch 张量 device: {torch_tensor.device}")
    # print(f"PyTorch 张量 dtype: {torch_tensor.dtype}")

    # PyTorch 转 NumPy
    # np_array = torch_tensor.numpy()
    # print(f"NumPy 数组 dtype: {np_array.dtype}")

    print("代码示例:")
    print("  np_array = torch_tensor.numpy()")
    print("\n注意：GPU 张量需要先移到 CPU")

torch_to_numpy()
```

## 内存共享

```python
def memory_sharing():
    """NumPy 和 PyTorch 共享内存"""
    print("\n=== 内存共享 ===\n")

    # torch.from_numpy() 创建的张量与原始 NumPy 数组共享内存
    np_array = np.array([1, 2, 3, 4], dtype=np.float32)

    # import torch
    # torch_tensor = torch.from_numpy(np_array)

    # 修改 NumPy 数组会影响 PyTorch 张量
    # np_array[0] = 999
    # print(f"torch_tensor[0]: {torch_tensor[0].item()}")  # 999

    print("torch.from_numpy() 创建的张量与 NumPy 数组共享内存")
    print("修改一方的数据会影响另一方")
    print("\n如果需要独立副本，使用 torch.tensor() 或 .clone()")

memory_sharing()
```

掌握 PyTorch 与 NumPy 的互操作对于混合使用两个框架非常重要。
