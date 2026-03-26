# TensorFlow 与 NumPy 互操作

TensorFlow 是另一个主流深度学习框架。本篇文章介绍 TensorFlow 与 NumPy 之间的互操作。

## tf.convert_to_tensor()：NumPy 转 TensorFlow

```python
import numpy as np

def numpy_to_tensorflow():
    """NumPy 数组转 TensorFlow 张量"""
    print("=== NumPy -> TensorFlow ===\n")

    # NumPy 转 TensorFlow
    np_array = np.random.randn(3, 4).astype(np.float32)
    print(f"NumPy 数组 dtype: {np_array.dtype}")

    # import tensorflow as tf
    # tf_tensor = tf.convert_to_tensor(np_array)
    # print(f"TensorFlow 张量 dtype: {tf_tensor.dtype}")

    print("代码示例:")
    print("  tf_tensor = tf.convert_to_tensor(np_array)")

numpy_to_tensorflow()
```

## tensor.numpy()：TensorFlow 转 NumPy

```python
def tensorflow_to_numpy():
    """TensorFlow 张量转 NumPy 数组"""
    print("\n=== TensorFlow -> NumPy ===\n")

    # import tensorflow as tf
    # tf_tensor = tf.constant([[1, 2], [3, 4]])
    # np_array = tf_tensor.numpy()

    print("代码示例:")
    print("  np_array = tf_tensor.numpy()")

tensorflow_to_numpy()
```

掌握 TensorFlow 与 NumPy 的互操作对于在不同框架间传递数据非常重要。
