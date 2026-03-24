# vLLM 教程

vLLM 是一个高性能的大语言模型推理引擎。

## 安装

```bash
pip install vllm
```

## 快速开始

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

## 下一步

继续学习：
- [LangChain教程](/pages/llm/langchain/)
