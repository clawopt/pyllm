# 随机种子与可复现性

随机数在深度学习中无处不在：参数初始化、数据增强、Dropout、蒙特卡洛采样等都需要随机数生成。在训练大型语言模型时，能够精确复现实验结果至关重要——只有可复现的实验才能被验证和调试。NumPy 提供了 `np.random` 模块来生成各种随机数，而随机种子（seed）则是控制随机序列起始点的机制。设置种子后，即使使用随机数生成器，每次运行也会得到相同的随机序列。

## 伪随机数生成器的工作原理

计算机中的"随机数"实际上是伪随机数——它们由确定性算法生成，看起来像随机数但实际上是可预测的。伪随机数生成器（PRNG）从一个初始状态（称为种子）开始，通过确定性的更新规则生成序列。只要种子相同，生成的序列就完全相同。

NumPy 使用的默认随机数生成器是 Mersenne Twister（梅森旋转算法），它能产生高质量的伪随机数，周期长达 2¹⁹⁹³⁷-1。这意味着即使连续生成大量随机数，序列也要经过极长时间才会重复。

```python
import numpy as np

# 设置随机种子
np.random.seed(42)

# 生成的随机数序列
print("第一次随机序列:")
print(np.random.rand(5))

# 再次设置相同种子，生成相同序列
np.random.seed(42)
print("\n相同种子下的随机序列:")
print(np.random.rand(5))
```

## np.random.seed 的使用方式

NumPy 允许通过多种方式设置种子。最直接的方式是使用 `np.random.seed(n)`，其中 n 是一个整数。种子可以是任意非负整数。

```python
# 基本用法
np.random.seed(0)
a = np.random.rand(3)
print(f"种子0: {a}")

# 重新设置种子
np.random.seed(42)
b = np.random.rand(3)
print(f"种子42: {b}")

# 相同种子产生相同序列
np.random.seed(0)
c = np.random.rand(3)
print(f"再次种子0: {c}")
print(f"a == c: {np.allclose(a, c)}")
```

需要注意的是，`np.random.seed()` 会影响**全局随机状态**。在需要更精细控制时，可以使用 `np.random.Generator` 或 `np.random.default_rng()` 创建独立的随机数生成器实例。

```python
# 使用独立的随机数生成器
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)

arr1 = rng1.random(5)
arr2 = rng2.random(5)

print(f"独立生成器相同种子: {np.allclose(arr1, arr2)}")
print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
```

## 可复现性的重要性

在科研和工程实践中，可复现性是基本要求。想象一下，当你的模型在某次训练中取得了不错的效果，但你无法复现它——这意味着你无法确定是什么因素导致了性能提升，也意味着你无法与他人分享你的结果。

**调试阶段的可复现性**：在开发新模型或新算法时，能够复现结果意味着你可以精确控制变量，准确判断某个改动是改进了还是降低了性能。

**超参数调优的可复现性**：当你搜索最优超参数时，每次试验都应该可复现。否则，你无法确定观察到的性能差异是来自超参数还是随机波动。

**分布式训练中的可复现性**：在多 GPU 或多节点训练时，确保各进程使用不同的随机序列（但仍然可复现）很重要。这通常通过为每个进程设置不同的种子实现。

```python
def set_reproducible_seed(seed, gpu_id=None):
    """为分布式训练设置可复现种子

    参数:
        seed: 基础随机种子
        gpu_id: GPU 编号，用于为不同 GPU 设置不同种子
    """
    if gpu_id is not None:
        seed = seed + gpu_id

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 示例：为不同 GPU 设置种子
for gpu_id in range(4):
    set_reproducible_seed(42, gpu_id)
    print(f"GPU {gpu_id} 种子: {42 + gpu_id}")
```

## LLM训练中的可复现性实践

训练大型语言模型时，需要为多个组件设置随机种子，包括 NumPy、PyTorch（或其他深度学习框架）、Python 内置 random 等。仅仅为一个库设置种子是不够的。

```python
import random
import torch

def set_all_seeds(seed):
    """设置所有随机种子以确保完全可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 完整的可复现性设置
set_all_seeds(42)

# 验证设置
print(f"NumPy: {np.random.rand()}")
print(f"Python random: {random.random()}")
print(f"PyTorch: {torch.rand(1)}")
```

**注意**：`torch.backends.cudnn.deterministic = True` 会强制 CUDA 使用确定性算法，但这可能会牺牲一定性能。在追求极致性能时，可以关闭此选项。

## 常见误区

**误区一：只设置一个库的种子**

深度学习涉及多个库：NumPy、PyTorch（TensorFlow）、Python 内置 random 等。每个库都有自己的随机数生成器，只设置一个是不够的。确保为所有相关库设置种子。

**误区二：在训练过程中修改种子**

如果在训练过程中修改种子（例如在每个 epoch 后重新设置），会导致结果不可复现。种子应该在训练开始前设置一次，不要在训练中途更改。

**误区三：假设 GPU 计算是确定性的**

CUDA 操作默认不保证确定性。即使设置了所有种子，在 GPU 上运行的结果也可能因浮点数顺序等问题略有差异。启用 `torch.backends.cudnn.deterministic = True` 可以提高确定性，但会降低性能。

**误区四：忘记数据加载器的 worker 数**

在使用 DataLoader 时，如果使用多个 worker 加载数据，每个 worker 都有自己的随机数生成器。需要为每个 worker 设置种子，或者设置 `worker_init_fn` 来确保数据加载的可复现性。

```python
def worker_init_fn(worker_id):
    """为 DataLoader 的每个 worker 设置不同的种子"""
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 在 DataLoader 中使用
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, num_workers=4, worker_init_fn=worker_init_fn)
```

## 全局种子 vs 局部生成器

`np.random.seed()` 修改全局状态，这可能在某些情况下导致问题。例如，在库代码中使用种子可能意外影响用户的代码。更好的做法是使用局部生成器：

```python
# 全局方式（可能影响其他代码）
np.random.seed(42)
np.random.rand(3)

# 局部生成器（推荐，更安全）
rng = np.random.default_rng(42)
rng.random(3)
```

局部生成器不会影响全局随机状态，也不会被其他代码意外修改。这在编写库函数时特别重要。

```python
class ModelTrainer:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def train_step(self, data):
        # 使用实例的随机数生成器
        noise = self.rng.randn(*data.shape)
        return data + noise * 0.1

# 不同实例使用不同的生成器
trainer1 = ModelTrainer(42)
trainer2 = ModelTrainer(42)

# 相同种子产生相同结果
print(trainer1.rng.random(3))
print(trainer2.rng.random(3))
```

理解随机种子的机制和可复现性的重要性，是在深度学习项目中有效调试和验证结果的基础。在训练 LLMs 时养成正确设置种子的习惯，能为你节省大量调试时间。
