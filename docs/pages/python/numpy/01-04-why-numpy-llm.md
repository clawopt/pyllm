---
title: 为什么AI/LLM离不开NumPy
---

# 为什么AI/LLM离不开NumPy

这一章回答一个根本问题：为什么学习大模型要先学NumPy？

## 一个真实的故事

让我从一个场景开始。假设你训练了一个神经网络，权重保存在一个shape为(4096, 4096)的矩阵里。你想看看这些权重的分布——比如计算均值、标准差、最大值、最小值。

用纯Python怎么做？你得写循环遍历每一个元素。但如果你用NumPy，一行就够了：

```python
import numpy as np

weights = np.random.randn(4096, 4096).astype(np.float32)
print(f"均值: {weights.mean():.4f}")
print(f"标准差: {weights.std():.4f}")
print(f"范围: [{weights.min():.4f}, {weights.max():.4f}]")
```

这就是NumPy的威力。它不仅代码简洁，速度还快几十倍。

## NumPy是什么

NumPy是Python科学计算的基础库。它的核心是一个叫ndarray的多维数组对象，以及一套用于操作这些数组的函数。

你可以把NumPy理解为Python的"向量计算引擎"。传统Python循环处理数据是一个一个来，NumPy是一批一批来。就像食堂打饭：阿姨可以一个一个盛（慢），也可以一大勺盛很多人（快）。

NumPy之所以快，主要有几个原因。首先，数据在内存中是连续存储的，不需要像列表那样每个元素单独管理。其次，大部分计算在底层C代码中完成，不经过Python解释器。最后，NumPy能利用CPU的SIMD指令，一个指令处理多个数据。

## 大模型和NumPy的关系

大模型的训练和推理，背后都是大规模矩阵运算。GPT-3有1750亿参数，这些参数以矩阵形式存储和计算。没有NumPy（或基于NumPy的框架如PyTorch、TensorFlow），这些运算根本无法高效完成。

具体来说，大模型中NumPy的应用场景包括：

处理文本时，你的输入会被转成token ID数组：
```python
import numpy as np

# 模拟token序列
token_ids = np.array([101, 2003, 3007, 1996, 102])
print(f"Token序列: {token_ids.shape}")
```

存储模型权重时，权重以多维数组形式存在：
```python
# 一个transformer层的权重
W_q = np.random.randn(768, 768).astype(np.float32)  # Query投影
W_k = np.random.randn(768, 768).astype(np.float32)  # Key投影
W_v = np.random.randn(768, 768).astype(np.float32)  # Value投影
print(f"每个投影矩阵: {W_q.nbytes / 1024:.1f} KB")
```

计算注意力分数时，核心运算是矩阵乘法：
```python
Q = np.random.randn(1, 8, 10, 64)  # (batch, heads, seq, dim)
K = np.random.randn(1, 8, 10, 64)

# QK^T / sqrt(d_k)
scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(64)
print(f"注意力分数: {scores.shape}")
```

## NumPy与面试

很多同学觉得学习大模型就是学PyTorch。但面试时，NumPy是基础考察内容。为什么？

因为PyTorch、TensorFlow的tensor和NumPy数组本质上是一样的。很多面试官会问：你知道NumPy的广播机制吗？你知道视图和副本的区别吗？你知道怎么优化大数组的内存使用吗？

这些问题都需要对NumPy有深入理解才能回答。而且，PyTorch的很多概念和NumPy是对应的——比如torch.from_numpy和np.array功能类似，torch.squeeze和np.squeeze功能类似。

## NumPy vs 其他工具

有同学问：我直接学PyTorch不行吗？

可以，但你会缺少很多基础概念。比如PyTorch的tensor也有shape、dtype、device等属性，和NumPy非常相似。先学NumPy再学PyTorch，就像是先学走路再学跑步——基础扎实了，后面更快。

而且，有些场景直接用NumPy更方便。比如数据预处理、简单的矩阵运算、结果可视化分析——这些不需要深度学习框架，NumPy就够了。

## 这一章小结

这一章我们认识了NumPy：它是Python科学计算的基础库，提供高效的多维数组对象和向量化的数学函数。在大模型中，从token处理到权重存储，从注意力计算到模型训练，处处都有NumPy的身影。理解NumPy，不仅是为了做数据处理，更是为了理解现代深度学习框架的底层原理。
