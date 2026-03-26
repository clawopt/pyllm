# 词向量相似度可视化

词向量相似度是理解词嵌入质量的重要指标。语义相近的词应该具有相似的向量表示，因此它们的余弦相似度应该较高。本篇文章介绍如何使用 NumPy 计算词向量之间的相似度，并可视化词向量空间的结构。通过相似度分析，我们可以验证嵌入矩阵是否学习到了有意义的语义关系。

## 余弦相似度

余弦相似度是最常用的词向量相似度度量：

```python
import numpy as np

def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度

    cos(θ) = (A · B) / (||A|| × ||B||)
    """
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2 + 1e-8)

# 示例
np.random.seed(42)
v1 = np.random.randn(768)
v2 = np.random.randn(768)

similarity = cosine_similarity(v1, v2)
print(f"余弦相似度: {similarity:.4f}")
```

## 批量余弦相似度计算

```python
def batch_cosine_similarity(vectors1, vectors2):
    """批量计算余弦相似度

    参数:
        vectors1: (n, d) 第一个向量组
        vectors2: (m, d) 第二个向量组
    返回:
        similarities: (n, m) 相似度矩阵
    """
    # 归一化向量
    norms1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(vectors2, axis=1, keepdims=True)

    normalized1 = vectors1 / (norms1 + 1e-8)
    normalized2 = vectors2 / (norms2 + 1e-8)

    # 矩阵乘法得到余弦相似度
    similarities = np.dot(normalized1, normalized2.T)

    return similarities

# 示例
np.random.seed(42)
vectors = np.random.randn(10, 768).astype(np.float32)

# 计算所有向量对之间的相似度
similarity_matrix = batch_cosine_similarity(vectors, vectors)
print(f"相似度矩阵形状: {similarity_matrix.shape}")
print(f"对角线元素（自身相似度）: {np.diag(similarity_matrix)[:3]}")
```

## 词向量空间可视化

### 创建示例词向量

为了演示，我们创建一些带有语义结构的词向量：

```python
def create_semantic_embeddings():
    """创建带有语义结构的词向量

    模拟：king - man + woman ≈ queen
    """
    np.random.seed(42)
    embed_dim = 64

    # 基础词汇
    words = ['king', 'man', 'woman', 'queen', 'prince', 'princess',
             'apple', 'fruit', 'orange', 'banana', 'car', 'bus', 'train', 'vehicle']

    # 创建嵌入向量
    embeddings = {}

    # 语义聚类：royalty
    embeddings['king'] = np.array([1.0, 0.5, 0.3] + [np.random.randn() * 0.1 for _ in range(embed_dim - 3)])
    embeddings['queen'] = embeddings['king'] + np.array([0.2, -0.1, 0.1] + [0] * (embed_dim - 3))
    embeddings['prince'] = embeddings['king'] + np.array([-0.1, 0.2, 0.1] + [0] * (embed_dim - 3))
    embeddings['princess'] = embeddings['queen'] + np.array([-0.1, 0.2, 0.1] + [0] * (embed_dim - 3))

    # 语义聚类：gender
    embeddings['man'] = np.array([0.5, 0.8, 0.2] + [np.random.randn() * 0.1 for _ in range(embed_dim - 3)])
    embeddings['woman'] = np.array([0.5, 0.8, 0.3] + [np.random.randn() * 0.1 for _ in range(embed_dim - 3)])

    # 语义聚类：fruit
    embeddings['apple'] = np.array([-0.5, -0.3, 0.8] + [np.random.randn() * 0.1 for _ in range(embed_dim - 3)])
    embeddings['fruit'] = embeddings['apple'] + np.array([0.1, 0.1, -0.2] + [0] * (embed_dim - 3))
    embeddings['orange'] = embeddings['apple'] + np.array([0.2, 0.1, 0.1] + [0] * (embed_dim - 3))
    embeddings['banana'] = embeddings['apple'] + np.array([-0.1, 0.2, 0.1] + [0] * (embed_dim - 3))

    # 语义聚类：vehicle
    embeddings['car'] = np.array([-0.3, 0.2, -0.6] + [np.random.randn() * 0.1 for _ in range(embed_dim - 3)])
    embeddings['bus'] = embeddings['car'] + np.array([0.1, -0.1, 0.2] + [0] * (embed_dim - 3))
    embeddings['train'] = embeddings['car'] + np.array([-0.1, 0.1, 0.3] + [0] * (embed_dim - 3))
    embeddings['vehicle'] = (embeddings['car'] + embeddings['bus'] + embeddings['train']) / 3

    return words, embeddings

words, embeddings = create_semantic_embeddings()
print(f"词汇列表: {words}")
print(f"嵌入维度: {len(embeddings['king'])}")
```

## 相似度分析

### 计算词对之间的相似度

```python
def analyze_word_similarities(embeddings, words):
    """分析词对之间的相似度"""
    word_list = list(words)
    n = len(word_list)

    # 构建嵌入矩阵
    emb_matrix = np.array([embeddings[w] for w in word_list])

    # 计算相似度矩阵
    similarity_matrix = batch_cosine_similarity(emb_matrix, emb_matrix)

    return word_list, similarity_matrix

word_list, sim_matrix = analyze_word_similarities(embeddings, words)

print("\n=== 相似度分析 ===")

# 找出最相似的词对
def find_most_similar(word, word_list, similarity_matrix, top_k=5):
    """找出与给定词最相似的词"""
    idx = word_list.index(word)
    similarities = similarity_matrix[idx]

    # 获取排序索引（降序）
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for i in sorted_indices[1:top_k+1]:  # 跳过自身
        results.append((word_list[i], similarities[i]))

    return results

# 示例
test_words = ['king', 'woman', 'apple', 'car']
for word in test_words:
    if word in word_list:
        similar = find_most_similar(word, word_list, sim_matrix)
        print(f"\n与 '{word}' 最相似的词:")
        for similar_word, sim in similar:
            print(f"  {similar_word}: {sim:.4f}")
```

### 验证类比关系

著名的词类比："king - man + woman ≈ queen"

```python
def verify_word_analogy(embeddings, word1, word2, word3, word4):
    """验证词类比：word1 - word2 + word3 ≈ word4

    king - man + woman ≈ queen
    """
    # 计算向量运算
    v1 = embeddings[word1]
    v2 = embeddings[word2]
    v3 = embeddings[word3]
    v4 = embeddings[word4]

    result = v1 - v2 + v3

    # 计算与预期结果的相似度
    similarity = cosine_similarity(result, v4)
    return similarity

# 示例
analogy_score = verify_word_analogy(embeddings, 'king', 'man', 'woman', 'queen')
print(f"\nking - man + woman ≈ queen 的相似度: {analogy_score:.4f}")

# 更多类比测试
analogies = [
    ('prince', 'man', 'woman', 'princess'),
    ('car', 'bus', 'train', 'vehicle'),
]

for w1, w2, w3, w4 in analogies:
    score = verify_word_analogy(embeddings, w1, w2, w3, w4)
    print(f"{w1} - {w2} + {w3} ≈ {w4}: {score:.4f}")
```

## 可视化（文本形式）

虽然无法直接绘制图形，但可以用文本形式展示相似度矩阵：

```python
def print_similarity_matrix(word_list, similarity_matrix, top_n=10):
    """打印相似度矩阵"""
    n = min(len(word_list), top_n)

    print("\n=== 相似度矩阵（前{}个词）===\n".format(n))

    # 打印表头
    header = "         " + "".join([f"{w:>8}" for w in word_list[:n]])
    print(header)
    print("-" * len(header))

    # 打印矩阵
    for i in range(n):
        row = f"{word_list[i]:>8}"
        for j in range(n):
            row += f"{similarity_matrix[i, j]:>8.2f}"
        print(row)

def print_heatmap(word_list, similarity_matrix):
    """用字符打印热力图"""
    n = len(word_list)
    print("\n=== 相似度热力图 ===\n")

    chars = ' .:-=+*#@'  # 从低到高的字符

    for i in range(n):
        row = f"{word_list[i]:>10} |"
        for j in range(n):
            sim = similarity_matrix[i, j]
            # 将相似度映射到字符
            char_idx = int((sim + 1) / 2 * (len(chars) - 1))
            char_idx = max(0, min(len(chars) - 1, char_idx))
            row += chars[char_idx]
        print(row)

print_similarity_matrix(word_list, sim_matrix)
print_heatmap(word_list, sim_matrix)
```

## 最近邻搜索

```python
def find_nearest_neighbors(target_word, word_list, embeddings, similarity_matrix, k=5):
    """找到最近邻词

    参数:
        target_word: 目标词
        word_list: 词列表
        embeddings: 嵌入字典
        similarity_matrix: 相似度矩阵
        k: 返回前 k 个最近邻
    """
    if target_word not in word_list:
        return []

    target_idx = word_list.index(target_word)
    similarities = similarity_matrix[target_idx]

    # 排序（降序）
    sorted_indices = np.argsort(similarities)[::-1]

    neighbors = []
    for idx in sorted_indices[1:k+1]:  # 跳过自身
        neighbors.append((word_list[idx], similarities[idx]))

    return neighbors

# 示例
print("\n=== 最近邻搜索 ===")
for word in ['king', 'woman', 'apple', 'car']:
    if word in word_list:
        neighbors = find_nearest_neighbors(word, word_list, embeddings, sim_matrix)
        print(f"\n{word} 的最近邻:")
        for neighbor, sim in neighbors:
            print(f"  {neighbor}: {sim:.4f}")
```

## 聚类分析

简单的聚类可以展示词向量的结构：

```python
def simple_clustering(word_list, embeddings, n_clusters=4):
    """简单的 K-means 聚类

    注意：这里使用简化的实现，实际应使用 sklearn
    """
    from collections import defaultdict

    # 获取嵌入矩阵
    emb_matrix = np.array([embeddings[w] for w in word_list])

    # 简化的聚类：基于相似度
    sim_matrix = batch_cosine_similarity(emb_matrix, emb_matrix)

    # 简单的基于距离的聚类
    clusters = defaultdict(list)
    assigned = set()

    for i, word in enumerate(word_list):
        if word in assigned:
            continue

        # 找到与这个词最相似的未分配词
        sims = sim_matrix[i]
        sorted_idx = np.argsort(sims)[::-1]

        cluster_id = len(clusters)
        for idx in sorted_idx:
            neighbor_word = word_list[idx]
            if neighbor_word not in assigned:
                clusters[cluster_id].append(neighbor_word)
                assigned.add(neighbor_word)

        if len(clusters) >= n_clusters:
            break

    # 处理剩余的词
    for word in word_list:
        if word not in assigned:
            clusters[len(clusters) % n_clusters].append(word)
            assigned.add(word)

    return dict(clusters)

# 示例
clusters = simple_clustering(word_list, embeddings, n_clusters=4)
print("\n=== 简单聚类结果 ===")
for cluster_id, words in clusters.items():
    print(f"Cluster {cluster_id}: {words}")
```

## 常见误区

**误区一：忘记归一化**

比较向量相似度前应该归一化：

```python
# 错误：直接比较
dot_product = np.dot(v1, v2)

# 正确：归一化后比较
norm1 = np.linalg.norm(v1)
norm2 = np.linalg.norm(v2)
cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
```

**误区二：使用欧氏距离代替余弦相似度**

欧氏距离和余弦相似度在某些情况下不等价：

```python
def euclidean_distance(v1, v2):
    """欧氏距离"""
    return np.linalg.norm(v1 - v2)

# 余弦相似度高不代表欧氏距离近
# 例如：两个方向相同但长度不同的向量，余弦相似度=1，欧氏距离很大
v1 = np.array([1, 0])
v2 = np.array([100, 0])
print(f"余弦相似度: {cosine_similarity(v1, v2):.4f}")
print(f"欧氏距离: {euclidean_distance(v1, v2):.4f}")
```

**误区三：在高维空间中使用欧氏距离**

在高维空间中，欧氏距离会趋于相近（维度诅咒）。余弦相似度更适合高维数据：

```python
# 高维空间中距离的稀释效应
high_dim = 1000
v1 = np.random.randn(high_dim)
v2 = np.random.randn(high_dim)

# 随机向量的余弦相似度通常接近 0
print(f"高维随机向量余弦相似度: {cosine_similarity(v1, v2):.4f}")
print(f"高维随机向量欧氏距离: {euclidean_distance(v1, v2):.4f}")
```

## API 总结

| 函数 | 描述 |
|------|------|
| `cosine_similarity(v1, v2)` | 计算两个向量的余弦相似度 |
| `batch_cosine_similarity(v1, v2)` | 批量计算余弦相似度 |
| `find_most_similar(word, ...)` | 找最相似的词 |
| `verify_word_analogy(...)` | 验证词类比关系 |

词向量相似度分析是理解词嵌入质量的重要工具。通过这些技术，我们可以验证嵌入是否学习到了有意义的语义关系。
