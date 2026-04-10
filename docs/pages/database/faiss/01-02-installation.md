# 1.2 安装与环境配置

> **pip install faiss-cpu 就够了——GPU 版本是给有 CUDA 显卡的人准备的**

---

## 这一节在讲什么？

FAISS 的安装比 Milvus 简单得多——不需要 Docker、不需要 etcd、不需要 MinIO，一条 pip 命令就能装好。但 CPU 版本和 GPU 版本的选择、CUDA 版本的兼容性、conda 和 pip 的冲突，这些小问题可能会卡住你。这一节我们快速过一遍安装过程，帮你避开常见的坑。

---

## CPU 版本安装

CPU 版本是学习和中小规模使用的首选——不需要 GPU，任何机器都能跑：

```bash
pip install faiss-cpu
```

验证安装：

```python
import faiss
print(faiss.__version__)  # 输出版本号，如 1.8.0
```

如果你用 conda 管理 Python 环境，也可以用 conda 安装：

```bash
conda install -c conda-forge faiss-cpu
```

---

## GPU 版本安装

GPU 版本需要 NVIDIA GPU + CUDA 工具链，适合大规模向量搜索和高性能场景：

```bash
# 需要先安装 CUDA Toolkit（11.8 或 12.x）
pip install faiss-gpu
```

GPU 版本的安装比 CPU 版本复杂得多——你需要确保 CUDA 版本与 faiss-gpu 的版本匹配。faiss-gpu 的不同版本对应不同的 CUDA 版本：

| faiss-gpu 版本 | CUDA 版本 |
|---------------|----------|
| 1.7.x | CUDA 11.x |
| 1.8.x | CUDA 11.8 / 12.x |

### 常见安装问题

**问题1：macOS 不支持 GPU 版本**。Apple 的芯片（M1/M2/M3）使用的是 Metal 而不是 CUDA，faiss-gpu 无法在 macOS 上运行。macOS 用户只能用 faiss-cpu。

**问题2：conda 和 pip 混用导致冲突**。如果你用 conda 安装了 faiss-cpu，又用 pip 安装了 faiss-gpu，可能会出现版本冲突。建议在一个环境中只用一种包管理器。

**问题3：CUDA 版本不匹配**。如果你系统的 CUDA 是 12.1，但 faiss-gpu 编译时用的是 CUDA 11.8，运行时会报错 `CUDA driver version is insufficient`。解决方法是安装与系统 CUDA 版本匹配的 faiss-gpu。

```bash
# 检查 CUDA 版本
nvidia-smi  # 查看 CUDA Driver 版本
nvcc --version  # 查看 CUDA Toolkit 版本

# 安装匹配的 faiss-gpu
pip install faiss-gpu  # 默认安装最新版本
```

---

## CPU 版本 vs GPU 版本的功能差异

| 功能 | faiss-cpu | faiss-gpu |
|------|-----------|-----------|
| Flat 索引 | ✅ | ✅ |
| IVF 索引 | ✅ | ✅ |
| HNSW 索引 | ✅ | ❌（GPU 不支持 HNSW） |
| PQ/SQ 量化 | ✅ | ✅ |
| GPU 加速搜索 | ❌ | ✅ |
| 多 GPU 支持 | ❌ | ✅ |
| 安装难度 | 低 | 高 |

注意：**GPU 版本不支持 HNSW 索引**——HNSW 的图遍历模式不适合 GPU 的大规模并行计算。GPU 擅长的是 IVF + PQ 这种可以批量并行计算距离的索引。

---

## 小结

这一节我们快速过了 FAISS 的安装：CPU 版本 `pip install faiss-cpu` 一条命令搞定，GPU 版本需要 CUDA 环境且安装更复杂。macOS 用户只能用 CPU 版本，GPU 版本不支持 HNSW 索引。下一节我们直接上手写代码——5 分钟跑通第一个向量搜索。
