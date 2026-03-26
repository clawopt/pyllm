# AI 场景中的 ndarray

前面两节我们深入探讨了 ndarray 的数据类型和基本属性，这些知识在传统数值计算中非常重要，但在 AI 和大模型开发中，ndarray 有着更加直接和广泛的应用。理解 ndarray 在 AI 场景中的使用方式，能帮助我们更直观地理解深度学习框架的设计逻辑，也能让我们在处理模型数据时更加得心应手。

## 词嵌入矩阵

在大语言模型中，词嵌入（Word Embedding）是最基础的数据结构之一。假设我们有一个包含 50000 个词的词表，每个词被映射到一个 768 维的向量，那么整个词嵌入可以表示为一个 shape 为 (50000, 768) 的二维数组。

```python
import numpy as np

vocab_size = 50000
embed_dim = 768
embedding_matrix = np.random.randn(vocab_size, embed_dim).astype(np.float32)
print(f"词嵌入矩阵形状: {embedding_matrix.shape}")
print(f"内存占用: {embedding_matrix.nbytes / 1024 / 1024:.2f} MB")
```

这个矩阵在模型中扮演着"查找表"的角色。当模型需要获取某个词的向量表示时，实际上是在这个矩阵中按索引进行查找。假设我们有一个句子 [1024, 2053, 6789]，分别代表三个词的索引，获取这些词对应的向量就是：

```python
word_indices = np.array([1024, 2053, 6789])
word_vectors = embedding_matrix[word_indices]
print(f"提取的向量形状: {word_vectors.shape}")
```

这种按索引查找的操作在 NumPy 中非常高效，因为它直接利用了数组的随机访问特性，整个过程不需要任何循环。

## 注意力分数矩阵

Transformer 架构中的核心机制是自注意力（Self-Attention）。对于一个长度为 n 的序列，自注意力计算会产生一个 n×n 的注意力分数矩阵。这个矩阵的每一行代表一个词对所有词的注意力权重。

```python
seq_len = 512
d_k = 64

query = np.random.randn(seq_len, d_k).astype(np.float32)
key = np.random.randn(seq_len, d_k).astype(np.float32)

attention_scores = np.matmul(query, key.T) / np.sqrt(d_k)
print(f"注意力分数形状: {attention_scores.shape}")
```

在实际的大模型中，seq_len 可能达到 2048 甚至更长，这意味着注意力矩阵可能非常大。一个 2048×2048 的 float32 矩阵占用 16MB 内存，而实际模型中这样的矩阵可能有上百个。因此，理解 ndarray 的内存布局对于优化大模型推理至关重要。

注意力分数通常会经过 softmax 操作转换为概率分布：

```python
attention_weights = np.exp(attention_scores - attention_scores.max(axis=-1, keepdims=True))
attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
```

## 批量数据处理

在训练深度学习模型时，我们很少一次只处理一条样本，而是使用批量处理（Batch Processing）来提高计算效率。假设我们有一个批次包含 32 个样本，每个样本是一个长度为 512 的序列，每个词被映射到 768 维的向量，那么输入数据的形状就是 (32, 512, 768)。

```python
batch_size = 32
seq_len = 512
embed_dim = 768

batch_input = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
print(f"批次输入形状: {batch_input.shape}")
print(f"总元素数: {batch_input.size}")
```

批量处理的一个关键优势是向量化计算可以充分利用 GPU 的并行能力。当我们对整个批次执行某个操作时，比如计算所有样本的序列均值，NumPy 会一次性完成所有 32 个样本的计算，而不是逐个处理。

```python
seq_mean = batch_input.mean(axis=1)
print(f"序列均值形状: {seq_mean.shape}")
```

## 模型权重存储

大模型的权重通常以 ndarray 的形式存储和加载。一个拥有 70 亿参数的模型，如果使用 float32 格式存储，需要约 28GB 的内存。这就是为什么模型量化（Quantization）变得如此重要——通过将权重从 float32 转换为 int8 或 float16，可以大幅减少内存占用。

```python
weights_fp32 = np.random.randn(1000, 1000).astype(np.float32)
print(f"FP32 权重内存: {weights_fp32.nbytes / 1024 / 1024:.2f} MB")

weights_fp16 = weights_fp32.astype(np.float16)
print(f"FP16 权重内存: {weights_fp16.nbytes / 1024 / 1024:.2f} MB")

weights_int8 = (weights_fp32 / np.std(weights_fp32) * 127).astype(np.int8)
print(f"INT8 权重内存: {weights_int8.nbytes / 1024 / 1024:.2f} MB")
```

在实际的大模型推理中，权重通常被组织成多层结构，每一层都有对应的权重矩阵和偏置向量。理解这种组织方式，有助于我们进行模型剪枝、量化等优化操作。

## 中间激活值

在模型前向传播过程中，除了模型权重，还会产生大量的中间激活值（Activation）。这些激活值的大小取决于批次大小、序列长度和隐藏层维度。在反向传播时，这些激活值会被用来计算梯度。

```python
batch_size = 16
seq_len = 512
hidden_dim = 3072

hiddens = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
ffn_output = np.maximum(0, hiddens) * 1.1

intermediate_activations = [hiddens, ffn_output]
total_activation_memory = sum(arr.nbytes for arr in intermediate_activations)
print(f"中间激活值总内存: {total_activation_memory / 1024 / 1024:.2f} MB")
```

理解激活值的内存占用对于大模型训练至关重要。在反向传播时，我们不仅需要存储激活值，还需要存储它们的梯度。这意味着在训练过程中，激活值相关的内存占用可能会翻倍。

## 小结

ndarray 在 AI 场景中的应用远不止于数值计算，它本身就是深度学习数据表示的核心方式。从词嵌入到注意力分数，从批量输入到模型权重，从前向传播的激活值到反向传播的梯度，ndarray 贯穿了大模型的各个环节。掌握 ndarray 在这些场景中的使用方式，能帮助我们更好地理解 AI 系统的内部运作机制，也为后续学习 PyTorch 等深度学习框架打下坚实的基础。
