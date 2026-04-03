---
title: 安装与版本选择
description: PyPI / Conda 安装方式、版本选择策略、依赖管理、大模型开发环境推荐
---
# 安装与环境配置


## 安装前的准备

### Python 版本要求

Pandas 对 Python 版本有明确要求，选错版本会踩坑：

| Pandas 版本 | 最低 Python | 推荐Python | 备注 |
|-------------|------------|-----------|------|
| **1.x** | 3.9+ | 3.10-3.11 | 最后一个支持 3.8 的系列 |
| **2.x** | 3.10+ | 3.11-3.12 | 引入 PyArrow 后端 |
| **3.0** | 3.10+ | **3.12+** | CoW 默认开启 |

```python
import sys
print(f"Python {sys.version}")
print(f"版本元组: {sys.version_info}")

```

### 为什么推荐 3.12+

- **性能提升**：CPython 3.11 引入了自适应解释器（Specializing Adaptive Interpreter），3.12 进一步优化
- **更好的类型提示**：`type parameter syntax`（PEP 695）让泛型更简洁
- **错误信息改进**：Traceback 更易读
- **Pandas 3.0 最佳兼容**：官方测试覆盖最完整的版本

## 方式一：pip 安装（最常用）

### 基础安装

```bash
pip install "pandas>=3.0"

pip install pandas==3.0.5

python -c "import pandas as pd; print(pd.__version__)"
```

### 完整安装（含所有可选依赖）

```bash
pip install "pandas[all]"

```

### 大模型开发推荐组合

```bash
pip install \
    "pandas[all]>=3.0" \
    numpy>=2.0 \
    pyarrow>=15.0 \
    "polars[all]>=1.0" \
    datasets>=2.20 \
    transformers>=4.45 \
    scikit-learn>=1.5 \
    matplotlib>=3.9 \
    jupyter>=1.1
```

## 方式二：Conda 安装

Conda 在数据科学领域依然流行，特别是需要管理复杂依赖时：

```bash
conda install -c conda-forge pandas=3.0 pyarrow numpy

conda create -n llm-data python=3.12 -y
conda activate llm-data
conda install -c conda-forge \
    pandas=3.0 \
    pyarrow \
    ipython \
    jupyterlab \
    matplotlib \
    seaborn
```

### pip vs Conda 如何选择？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 虚拟环境隔离 | **pip + venv** | 轻量、标准、CI/CD 友好 |
| 科学计算全家桶 | **Conda** | 二进制预编译，避免编译痛苦 |
| 多 Python 版本共存 | **Conda** | 天然支持多版本切换 |
| Docker 容器内 | **pip** | 镜像构建更快 |
| 团队协作统一环境 | **Conda env export** 或 **pip freeze** | 两者都行，看团队习惯 |
| 大模型开发 | **pip**（为主） | HuggingFace 等库主要通过 pip 分发 |

## 版本选择策略

### 生产环境：锁定版本

```bash
pip freeze > requirements.txt

pip-tools compile requirements.in > requirements.txt

pandas==3.0.5
numpy==2.2.0
pyarrow==17.0.0
polars==1.7.0
```

### 开发环境：灵活升级

```bash
pandas>=3.0,<4.0
numpy>=2.0,<3.0
pyarrow>=15.0,<18.0

pip install --upgrade pandas numpy pyarrow
```

### 升级注意事项

从旧版 Pandas 升级到 3.0 时需要注意的**破坏性变更**：

```python
df = pd.DataFrame({'a': [1, 2, 3]})
subset = df[df['a'] > 1]
subset['a'] = 99

pd.options.dtype_backend  # 3.0 中可能是 'pyarrow' 或 None（取决于配置）

s = pd.Series([1, None, 3], dtype='Int64')
print(s[1])  # <NA>（不是 nan）

def check_migration_compatibility():
    import pandas as pd
    warnings = []
    
    if pd.__version__[0] < '3':
        warnings.append("建议升级到 Pandas 3.0+")
    
    if not pd.options.mode.copy_on_write:
        warnings.append("CoW 未启用，建议设置 pd.options.mode.copy_on_write = True")
    
    return warnings

for w in check_migration_compatibility():
    print(f"⚠️  {w}")
```

## 依赖关系图谱

理解 Pandas 的依赖有助于排查安装问题：

```
                    ┌─────────────┐
                    │   pandas    │
                    │   (核心)    │
                    └──┬────┬─────┘
                       │    │
           ┌───────────┘    └──────────┐
           ▼                              ▼
    ┌──────────────┐              ┌──────────────┐
    │    numpy     │              │ python-dateutil│
    │  (必须依赖)  │              │ tzdata        │
    └──────────────┘              └──────────────┘
           │
     ┌─────┴──────────┬──────────────┬──────────────┐
     ▼                ▼              ▼              ▼
┌──────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐
│ numexpr  │   │ bottleneck│  │  pyarrow │  │  scipy   │
│(可选)加速│   │(可选)加速 │  │(可选)后端│  │(可选)统计│
└──────────┘   └──────────┘  └──────────┘  └──────────┘
```

### 常见依赖冲突及解决

```bash
pip install --upgrade --force-reinstall numpy pandas

pip install --upgrade pyarrow>=15.0

python -m venv ~/envs/llm-data
source ~/envs/llm-data/bin/activate
pip install pandas==3.0
```

## 验证安装

```python
import sys
import numpy as np
import pandas as pd
import pyarrow

print("=" * 60)
print("Pandas 环境诊断报告")
print("=" * 60)
print(f"Python:      {sys.version.split()[0]}")
print(f"Pandas:      {pd.__version__}")
print(f"NumPy:       {np.__version__}")
print(f"PyArrow:     {pyarrow.__version__}")

features = {
    'Copy-on-Write': pd.options.mode.copy_on_write,
    'PyArrow 可导入': hasattr(pd, 'ArrowDtype'),
    'Nullable Int64': hasattr(pd, 'Int64Dtype'),
    'CSV 读取': hasattr(pd, 'read_csv'),
    'Parquet 读取': hasattr(pd, 'read_parquet'),
}

print("\n功能检测:")
for name, available in features.items():
    status = "✅" if available else "❌"
    print(f"  {status} {name}")

n = 1_000_000
df = pd.DataFrame({
    'a': np.random.randn(n),
    'b': np.random.choice(['x', 'y', 'z'], n),
    'c': np.random.randint(0, 100, n),
})

start = pd.Timestamp.now()
result = df.groupby('b')['a'].mean()
elapsed = (pd.Timestamp.now() - start).total_seconds()

print(f"\n性能基准 ({n:,} 行 GroupBy): {elapsed:.3f}s")

mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
print(f"内存占用: {mem_mb:.1f} MB")
print("=" * 60)
```

典型输出：

```
============================================================
Pandas 环境诊断报告
============================================================
Python:      3.12.3
Pandas:      3.0.5
NumPy:       2.2.0
PyArrow:     17.0.0

功能检测:
  ✅ Copy-on-Write
  ✅ PyArrow 可导入
  ✅ Nullable Int64
  ✅ CSV 读取
  ✅ Parquet 读取

性能基准 (1,000,000 行 GroupBy): 0.042s
内存占用: 76.3 MB
============================================================
```

下一节将深入 Pandas 3.0 的核心新特性——这些才是你升级到 3.0 的真正理由。
