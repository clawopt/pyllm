# DuckDB 教程

DuckDB 是一个嵌入式分析数据库。

## 安装

```bash
pip install duckdb
```

## 快速开始

```python
import duckdb

con = duckdb.connect()
result = con.execute("SELECT 'Hello, DuckDB!' AS message").fetchdf()
print(result)
```

## 下一步

继续学习：
- [LanceDB教程](/pages/database/lancedb/)
