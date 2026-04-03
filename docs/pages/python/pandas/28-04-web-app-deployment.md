---
title: Web 应用部署
description: FastAPI 后端、Streamlit 前端、Agent 服务化、Docker 容器化
---
# Web 部署

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Pandas Agent API")

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'] * 30,
    'score': [88 + i*0.3 for i in range(90)],
})

agent = create_pandas_dataframe_agent(
    llm=ChatOpenAI(model='gpt-4o-mini', temperature=0),
    df=df,
    verbose=False,
)

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(query: Query):
    try:
        r = agent.invoke(query.question)
        return {"answer": r['output'], "status": "ok"}
    except Exception as e:
        return {"answer": str(e)[:500], "status": "error"}

@app.get("/health")
def health():
    return {"status": "ok", "rows": len(df)}
```

启动：`uvicorn app:app --host 0.0.0.0 --port 8000`
访问 `http://localhost:8000/docs` 查看 Swagger UI。
