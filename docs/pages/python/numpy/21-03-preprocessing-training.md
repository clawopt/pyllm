# 预处理与训练

本篇文章介绍如何将 NumPy 用于数据预处理，PyTorch/TensorFlow 用于模型训练的混合工作流程。

## 混合工作流程

```python
import numpy as np

def hybrid_workflow():
    """NumPy 预处理 + 框架训练"""
    print("=== 混合工作流程 ===\n")

    print("1. NumPy 数据预处理")
    print("   - 文本清洗和分词")
    print("   - Tokenization")
    print("   - 数据增强")
    print("   - 特征工程\n")

    print("2. 框架模型训练")
    print("   - PyTorch 或 TensorFlow")
    print("   - GPU 加速")
    print("   - 自动微分\n")

    print("数据流:")
    print("  NumPy 预处理 -> 转换为框架张量 -> 模型训练 -> 结果转回 NumPy")

hybrid_workflow()
```

## NumPy 数据预处理

```python
def numpy_preprocessing():
    """使用 NumPy 进行数据预处理"""
    print("\n=== NumPy 数据预处理示例 ===\n")

    # 模拟 tokenized 数据
    np.random.seed(42)
    batch_size = 32
    seq_len = 128

    # 生成随机 token IDs
    input_ids = np.random.randint(0, 50000, size=(1000, seq_len), dtype=np.int32)
    attention_mask = np.ones((1000, seq_len), dtype=np.int32)

    # NumPy 预处理示例
    # 1. 添加特殊 token
    input_ids[:, 0] = 2  # [BOS]
    input_ids[:, -1] = 3  # [EOS]

    # 2. 计算序列长度统计
    actual_lens = np.sum(attention_mask, axis=1)
    print(f"序列长度统计: mean={actual_lens.mean():.2f}, max={actual_lens.max()}, min={actual_lens.min()}")

    # 3. 准备 batch
    indices = np.random.permutation(1000)[:batch_size]
    batch = {
        'input_ids': input_ids[indices],
        'attention_mask': attention_mask[indices]
    }

    print(f"Batch 形状: input_ids={batch['input_ids'].shape}")

    return batch

batch = numpy_preprocessing()
```

## 转换为框架张量

```python
def convert_to_framework():
    """转换为框架张量"""
    print("\n=== 转换为框架张量 ===\n")

    batch = {
        'input_ids': np.random.randint(0, 50000, size=(32, 128), dtype=np.int32),
        'attention_mask': np.ones((32, 128), dtype=np.int32)
    }

    print("PyTorch 转换:")
    print("  import torch")
    print("  input_ids = torch.from_numpy(batch['input_ids'])")
    print("  attention_mask = torch.from_numpy(batch['attention_mask'])")

    print("\nTensorFlow 转换:")
    print("  import tensorflow as tf")
    print("  input_ids = tf.constant(batch['input_ids'])")
    print("  attention_mask = tf.constant(batch['attention_mask'])")

convert_to_framework()
```

掌握混合工作流程可以充分利用 NumPy 的灵活性和框架的计算能力。
