---
title: 环境配置与 API Key 管理
description: API Key 安全存储、环境变量管理、多模型配置、连接测试
---
# 案例 3：LangChain Pandas Agent — 环境配置

```python
import os
from dotenv import load_dotenv

load_dotenv()

config = {
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'model': os.getenv('LLM_MODEL', 'gpt-4o-mini'),
    'temperature': float(os.getenv('LLM_TEMPERATURE', '0')),
    'max_iterations': int(os.getenv('MAX_ITERATIONS', '5')),
}

assert config['openai_api_key'], "请设置 OPENAI_API_KEY"
print(f"✅ 配置加载完成 | 模型: {config['model']}")
```
