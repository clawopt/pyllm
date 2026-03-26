# LoRA前向传播

本篇文章详细介绍如何使用 NumPy 实现 LoRA 的前向传播，包括冻结权重、LoRA 参数的更新、以及与原始前向传播的对比。通过纯 NumPy 实现，你可以深入理解 LoRA 的工作机制。

## LoRA 前向传播公式

LoRA 的前向传播公式为：

```
Y = X @ W₀ + (α/r) * (X @ B @ A)
```

其中：
- X: 输入
- W₀: 冻结的原始权重
- B, A: 可训练的 LoRA 参数
- α: 缩放因子
- r: 秩

```python
import numpy as np

def lora_forward(X, W0, B, A, alpha):
    """LoRA 前向传播

    参数:
        X: 输入 (batch, d) 或 (d,)
        W0: 冻结的原始权重 (d, k)
        B: LoRA 矩阵 B (d, r)
        A: LoRA 矩阵 A (r, k)
        alpha: 缩放因子
    返回:
        Y: 输出
    """
    # 原始输出
    Y0 = X @ W0

    # LoRA 更新
    delta_Y = (alpha / B.shape[1]) * (X @ B @ A)

    return Y0 + delta_Y

# 示例
np.random.seed(42)

# 输入
X = np.random.randn(2, 1024).astype(np.float32)  # batch=2, d=1024

# 原始权重
W0 = np.random.randn(1024, 1024).astype(np.float32) * 0.02

# LoRA 参数
r = 8
A = np.random.randn(r, 1024).astype(np.float32) * 0.01
B = np.zeros((1024, r), dtype=np.float32)  # B 初始化为零

alpha = r  # 缩放因子

# 前向传播
Y = lora_forward(X, W0, B, A, alpha)

print(f"输入 X 形状: {X.shape}")
print(f"权重 W0 形状: {W0.shape}")
print(f"LoRA A 形状: {A.shape}")
print(f"LoRA B 形状: {B.shape}")
print(f"输出 Y 形状: {Y.shape}")
print(f"由于 B=0，输出等于 X @ W0: {np.allclose(Y, X @ W0)}")
```

## 完整 LoRA 层实现

```python
class LoRALinear:
    """LoRA 线性层

    将 LoRA 适配器添加到标准线性层
    """

    def __init__(self, in_features, out_features, r=8, alpha=None):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha or r

        # 原始权重（冻结）
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        self.weight.flags.writeable = False  # 冻结

        # 原始偏置（如果有）
        self.bias = None

        # LoRA 参数
        self.lora_A = np.random.randn(r, in_features).astype(np.float32) * 0.01
        self.lora_B = np.zeros((out_features, r), dtype=np.float32)

    def forward(self, x):
        """前向传播

        x: (batch, in_features)
        返回: (batch, out_features)
        """
        # 原始输出
        Y0 = x @ self.weight.T

        # LoRA 输出
        delta_Y = (self.alpha / self.r) * (x @ self.lora_A.T @ self.lora_B.T)

        return Y0 + delta_Y

    def merge_weights(self):
        """合并 LoRA 权重到原始权重

        返回合并后的权重（不修改原始权重）
        """
        delta_W = (self.alpha / self.r) * (self.lora_B @ self.lora_A)
        merged_weight = self.weight + delta_W
        return merged_weight

    def train(self):
        """设置为训练模式"""
        self.training = True

    def eval(self):
        """设置为评估模式"""
        self.training = False

# 示例
np.random.seed(42)

# 创建 LoRA 层
lora_layer = LoRALinear(in_features=1024, out_features=1024, r=8)

# 前向传播
x = np.random.randn(2, 1024).astype(np.float32)
y = lora_layer.forward(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {y.shape}")

# 验证 B=0 时的输出
y_expected = x @ lora_layer.weight.T
print(f"B=0 时输出等于原始输出: {np.allclose(y, y_expected)}")
```

## 训练过程中的前向传播

训练时，LoRA 层需要正确处理两个部分：冻结的原始权重和可训练的 LoRA 参数。

```python
class LoRATrainableLayer:
    """支持训练的 LoRA 层"""

    def __init__(self, in_features, out_features, r=8, alpha=None):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha or r

        # 原始权重（冻结）
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        self.weight.flags.writeable = False

        # LoRA 参数（可训练）
        self.lora_A = np.random.randn(r, in_features).astype(np.float32) * 0.01
        self.lora_B = np.zeros((out_features, r), dtype=np.float32)

        # 梯度
        self.grad_lora_A = np.zeros_like(self.lora_A)
        self.grad_lora_B = np.zeros_like(self.lora_B)

    def forward(self, x):
        """前向传播"""
        Y0 = x @ self.weight.T
        delta_Y = (self.alpha / self.r) * (x @ self.lora_A.T @ self.lora_B.T)
        return Y0 + delta_Y

    def backward(self, x, grad_output):
        """简化的反向传播（仅计算 LoRA 参数的梯度）

        实际实现需要更复杂的梯度计算
        """
        # d(L)/d(Y) = grad_output
        # 对于 LoRA 部分: d(L)/d(BA) = grad_output * x.T

        # 梯度缩放因子
        scale = self.alpha / self.r

        # d(L)/d(A) = (grad_output @ B.T) @ x
        self.grad_lora_A = scale * (grad_output @ self.lora_B.T).T @ x

        # d(L)/d(B) = (grad_output.T @ x) @ A.T
        self.grad_lora_B = scale * (grad_output.T @ x).T

        return grad_output  # 返回传给上一层的梯度

    def update_weights(self, learning_rate):
        """更新 LoRA 参数"""
        self.lora_A -= learning_rate * self.grad_lora_A
        self.lora_B -= learning_rate * self.grad_lora_B

# 模拟训练
np.random.seed(42)

layer = LoRATrainableLayer(in_features=512, out_features=512, r=8)

# 前向传播
x = np.random.randn(4, 512).astype(np.float32)
y = layer.forward(x)

# 模拟损失计算和反向传播
loss_grad = np.random.randn(4, 512).astype(np.float32) * 0.01
layer.backward(x, loss_grad)

print(f"LoRA A 梯度形状: {layer.grad_lora_A.shape}")
print(f"LoRA B 梯度形状: {layer.grad_lora_B.shape}")
print(f"LoRA A 梯度均值: {layer.grad_lora_A.mean():.8f}")
print(f"LoRA B 梯度均值: {layer.grad_lora_B.mean():.8f}")
```

## 在 Transformer 中的应用

```python
class LoRAAttention:
    """带 LoRA 的注意力层"""

    def __init__(self, hidden_size, num_heads, r=8, alpha=None):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.r = r
        self.alpha = alpha or r

        # QKV 投影（原始，冻结）
        self.W_q = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.W_k = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.W_v = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.W_o = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02

        # 冻结
        self.W_q.flags.writeable = False
        self.W_k.flags.writeable = False
        self.W_v.flags.writeable = False
        self.W_o.flags.writeable = False

        # LoRA 参数（仅应用于 QKV）
        self.lora_q = self._create_lora_params(hidden_size, hidden_size, r)
        self.lora_k = self._create_lora_params(hidden_size, hidden_size, r)
        self.lora_v = self._create_lora_params(hidden_size, hidden_size, r)

    def _create_lora_params(self, out_dim, in_dim, r):
        """创建 LoRA 参数"""
        A = np.random.randn(r, in_dim).astype(np.float32) * 0.01
        B = np.zeros((out_dim, r), dtype=np.float32)
        return {'A': A, 'B': B}

    def forward(self, x, mask=None):
        """前向传播

        x: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # QKV 投影（原始 + LoRA）
        Q = x @ self.W_q.T + (self.alpha / self.r) * (x @ self.lora_q['A'].T @ self.lora_q['B'].T)
        K = x @ self.W_k.T + (self.alpha / self.r) * (x @ self.lora_k['A'].T @ self.lora_k['B'].T)
        V = x @ self.W_v.T + (self.alpha / self.r) * (x @ self.lora_v['A'].T @ self.lora_v['B'].T)

        # 后续注意力计算...

        return Q, K, V

# 示例
np.random.seed(42)

attention = LoRAAttention(hidden_size=256, num_heads=4, r=8)
x = np.random.randn(2, 10, 256).astype(np.float32)

Q, K, V = attention.forward(x)
print(f"输入形状: {x.shape}")
print(f"Q 形状: {Q.shape}")
print(f"K 形状: {K.shape}")
print(f"V 形状: {V.shape}")
```

## 权重复制与加载

```python
def copy_lora_weights(source, target):
    """复制 LoRA 权重"""
    if hasattr(source, 'lora_A'):
        target.lora_A = source.lora_A.copy()
        target.lora_B = source.lora_B.copy()

def merge_and_export(layer):
    """合并 LoRA 权重并导出"""
    merged = layer.weight.copy()
    delta_W = (layer.alpha / layer.r) * (layer.lora_B @ layer.lora_A)
    merged = merged + delta_W
    return merged

# 示例
np.random.seed(42)
layer1 = LoRALinear(512, 512, r=8)
layer2 = LoRALinear(512, 512, r=8)

# 复制权重
copy_lora_weights(layer1, layer2)
print("权重复制完成")

# 合并并导出
merged = merge_and_export(layer1)
print(f"合并后权重形状: {merged.shape}")
```

## 常见问题与调试

### 检查权重是否被冻结

```python
def check_frozen_weights(layer):
    """检查权重是否正确冻结"""
    print("\n=== 权重冻结检查 ===")
    print(f"Writable (should be False): {layer.weight.flags.writeable}")

    # 尝试修改
    try:
        layer.weight[0, 0] = 999
        print("错误：权重可以被修改！")
    except ValueError as e:
        print(f"正确：权重被冻结，修改被拒绝")

check_frozen_weights(lora_layer)
```

### 检查 LoRA 输出

```python
def verify_lora_output():
    """验证 LoRA 输出"""
    print("\n=== LoRA 输出验证 ===")

    np.random.seed(42)
    layer = LoRALinear(512, 512, r=8)
    x = np.random.randn(2, 512).astype(np.float32)

    # B=0 时，输出应等于原始输出
    y_b0 = layer.forward(x)
    y_orig = x @ layer.weight.T
    print(f"B=0 时与原始输出一致: {np.allclose(y_b0, y_orig)}")

    # 修改 B 后，输出应变化
    layer.lora_B[:, 0] = 1.0
    y_b1 = layer.forward(x)
    print(f"修改 B 后输出变化: {not np.allclose(y_b0, y_b1)}")

verify_lora_output()
```

理解 LoRA 的前向传播是实现高效微调的基础。下一篇文章将详细介绍 LoRA 带来的参数量节省效果。
