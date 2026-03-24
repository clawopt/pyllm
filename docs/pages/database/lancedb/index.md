# LanceDB 教程

LanceDB 是一个多模态向量数据库。

## 安装

```bash
pip install lancedb
```

## 快速开始

```python
import lancedb
import numpy as np

db = lancedb.connect("./lance_db")
table = db.create_table("vectors", data=[{"vector": np.random.rand(128), "id": i} for i in range(100)])
results = table.search(np.random.rand(128)).limit(5).to_df()
print(results)
```

## 下一步

继续学习：
- [OpenClaw教程](/pages/ai-coding/openclaw/)
