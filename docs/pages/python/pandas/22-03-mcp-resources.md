---
title: MCP 资源与安全
description: Resource 定义、权限控制、沙箱执行、生产部署注意事项
---
# MCP 安全：不能让 LLM 随意执行代码

上一节的 Server 有一个严重的安全隐患：`query` 工具使用了 `eval()` 来执行任意表达式。如果 LLM 生成了 `df.to_csv('/etc/passwd')` 或 `os.system('rm -rf /')` 这样的恶意代码，后果不堪设想。

## 安全加固方案

```python
import ast
import pandas as pd

SAFE_BUILTINS = {'pd': pd, 'max': max, 'min': min, 'len': len,
                 'sum': sum, 'abs': abs, 'round': round}

def safe_eval(expr, df):
    tree = ast.parse(expr, mode='eval')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("不允许 import")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ('to_csv', 'to_excel', 'system', 'exec'):
                    raise ValueError(f"禁止调用 {node.func.attr}")
    return eval(compile(tree, '<string>', 'eval'), {"__builtins__": {}}, {'df': df, **SAFE_BUILTINS})
```

这个 `safe_eval()` 做了三件事：
1. **AST 解析**：在执行前先解析代码的抽象语法树
2. **黑名单检查**：禁止 import 和危险函数调用（如 `system`、`exec`）
3. **受限命名空间**：只暴露安全的内置函数，不提供 `__builtins__`

**生产环境的铁律：永远不要对来自 LLM 的输入使用无限制的 `eval()`**。即使你信任 LLM 本身，也要防范 prompt injection 攻击——恶意用户可能通过精心构造的数据内容来诱导 LLM 执行危险操作。
