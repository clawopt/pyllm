# 特征值与特征向量

特征值和特征向量是线性代数中最重要的概念之一。想象一下，当你把一个矩阵当作一种"变换"时，特征向量就是那些经过这个变换后方向不变的向量（只是被缩放了），而特征值就是缩放的比例。特征值分解在数据分析中有广泛的应用：主成分分析（PCA）用特征值来确定主成分的方向，谱聚类用特征值来分析数据的结构。在深度学习中，理解特征值的概念有助于理解网络的稳定性和表达能力。NumPy 提供了 `np.linalg.eig` 和 `np.linalg.eigh` 来计算特征值和特征向量。

## 特征值与特征向量的定义

对于一个 n×n 的方阵 A，如果存在标量 λ 和非零向量 v 使得：
```
A v = λ v
```
则称 λ 是 A 的特征值，v 是对应的特征向量。

几何意义上，特征向量表示矩阵 A 作用后方向不变的向量，特征值表示这个向量被缩放的比例。

## np.linalg.eig：一般矩阵的特征值分解

```python
import numpy as np

A = np.array([[4, 2], [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"矩阵 A:\n{A}")
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")
```

注意：`np.linalg.eig` 返回的特征向量是按列排列的，即 `eigenvectors[:, i]` 是对应于 `eigenvalues[i]` 的特征向量。

## 验证特征值和特征向量

```python
# 验证 A @ v = λ * v
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_ = eigenvalues[i]

    left = A @ v
    right = lambda_ * v

    print(f"特征值 λ={lambda_:.4f}:")
    print(f"  A @ v = {left}")
    print(f"  λ * v = {right}")
    print(f"  相等: {np.allclose(left, right)}")
```

## np.linalg.eigh：对称矩阵的特征值分解

对于对称矩阵，使用 `eigh` 更高效更稳定：

```python
# 对称矩阵
A_sym = np.array([[4, 2], [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A_sym)
print(f"eig 特征值: {eigenvalues}")

eigenvalues_hermitian, eigenvectors_hermitian = np.linalg.eigh(A_sym)
print(f"eigh 特征值: {eigenvalues_hermitian}")
```

为什么对称矩阵要用 `eigh`？因为对称矩阵的特征值一定是实数，而非对称矩阵的特征值可能是复数。`eigh` 利用了对称性，可以使用更稳定、更快的算法。

## 特征值与特征向量的性质

### 特征值之和等于矩阵的迹

```python
A = np.array([[4, 2], [1, 3]])

eigenvalues, _ = np.linalg.eig(A)

trace = np.trace(A)
sum_eigenvalues = eigenvalues.sum()

print(f"矩阵的迹: {trace}")
print(f"特征值之和: {sum_eigenvalues}")
print(f"相等: {np.isclose(trace, sum_eigenvalues)}")
```

### 特征值之积等于矩阵的行列式

```python
det = np.linalg.det(A)
product_eigenvalues = eigenvalues.prod()

print(f"矩阵的行列式: {det}")
print(f"特征值之积: {product_eigenvalues}")
print(f"相等: {np.isclose(det, product_eigenvalues)}")
```

## 在LLM场景中的应用

### 分析位置编码的频谱

正弦/余弦位置编码可以看作是不同频率的信号叠加：

```python
def positional_encoding_matrix(seq_len, d_model):
    """生成位置编码矩阵"""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    div_term = np.exp(div_term)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe

seq_len = 50
d_model = 32
pe = positional_encoding_matrix(seq_len, d_model)

# 分析频谱
cov_matrix = pe.T @ pe  # 协方差矩阵
eigenvalues, _ = np.linalg.eigh(cov_matrix)

print(f"位置编码协方差矩阵的特征值（前10个）: {eigenvalues[-10:][::-1]}")
```

### PCA（主成分分析）

PCA 使用特征值分解来找到数据中方差最大的方向：

```python
def pca(X, n_components):
    """PCA 实现"""
    # 中心化数据
    X_centered = X - X.mean(axis=0)

    # 计算协方差矩阵
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 返回前 n_components 个主成分
    return eigenvalues[:n_components], eigenvectors[:, :n_components]

# 测试 PCA
np.random.seed(42)
X = np.random.randn(100, 5) @ np.diag([10, 5, 2, 1, 0.5])  # 不同方差的混合
X = X + np.array([1, 2, 3, 4, 5])  # 添加偏移

eigenvalues, components = pca(X, 3)
print(f"主成分的特征值: {eigenvalues}")
```

### 网络稳定性分析

在某些情况下，分析神经网络权重的特征值分布可以了解网络的特性：

```python
def analyze_weight_matrix(W, name="权重"):
    """分析权重矩阵的特性"""
    # 计算特征值
    eigenvalues, _ = np.linalg.eig(W)

    # 只看实部（对于对称矩阵，特征值是实数）
    eigenvalues_real = eigenvalues.real

    print(f"\n{name}:")
    print(f"  形状: {W.shape}")
    print(f"  特征值范围: [{eigenvalues_real.min():.4f}, {eigenvalues_real.max():.4f}]")
    print(f"  谱半径（最大特征值的绝对值）: {np.abs(eigenvalues_real).max():.4f}")

# 分析不同初始化的权重
np.random.seed(42)
W_xavier = np.random.randn(768, 768) * np.sqrt(2.0 / 768)
W_he = np.random.randn(768, 768) * np.sqrt(2.0 / 768)

analyze_weight_matrix(W_xavier, "Xavier 初始化")
analyze_weight_matrix(W_he, "He 初始化")
```

## 常见误区与注意事项

### 误区一：混淆特征向量的列/行顺序

```python
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

# eigenvectors 是按列排列的
print(f"第0个特征向量: {eigenvectors[:, 0]}")
print(f"第1个特征向量: {eigenvectors[:, 1]}")
```

### 误区二：忘记对称矩阵应该用 eigh

```python
# 对于对称矩阵，应该使用 eigh
A_sym = np.array([[4, 2], [2, 3]])

# eig 也能用，但 eigh 更稳定
eigenvalues_eig, eigenvectors_eig = np.linalg.eig(A_sym)
eigenvalues_hermitian, eigenvectors_hermitian = np.linalg.eigh(A_sym)

print(f"eig 特征值: {eigenvalues_eig}")  # 可能有微小的虚部（数值误差）
print(f"eigh 特征值: {eigenvalues_hermitian}")  # 一定是实数
```

### 误区三：非方阵没有特征值

```python
B = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 矩阵

try:
    np.linalg.eig(B)
except np.linalg.LinAlgError as e:
    print(f"非方阵没有特征值: {e}")

# 对于非方阵，应该使用 SVD
U, s, Vt = np.linalg.svd(B)
print(f"SVD 奇异值: {s}")
```

## 小结

特征值和特征向量是线性代数的核心概念。对于方阵 A，如果存在 λ 和 v 使得 Av = λv，则 λ 是特征值，v 是特征向量。对称矩阵应该使用 `np.linalg.eigh`，它更稳定且保证特征值是实数。在 LLM 场景中，特征值分解可以用于分析位置编码的频谱、实现 PCA、以及理解网络权重特性。

面试时需要能够解释特征值和特征向量的定义，理解 eig 和 eigh 的区别，以及能够描述特征值在数据分析中的应用。
