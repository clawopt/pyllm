# DeepSeek 教程

DeepSeek 是一个强大的 AI 编程模型。

## API 使用

```python
from openai import OpenAI

client = OpenAI(api_key="your-key", base_url="https://api.deepseek.com/v1")
response = client.chat.completions.create(model="deepseek-coder", messages=[{"role": "user", "content": "Hello!"}])
print(response.choices[0].message.content)
```

## 下一步

完成学习，返回首页继续探索。
