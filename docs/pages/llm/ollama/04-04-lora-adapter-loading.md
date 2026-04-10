# 04-4 ADAPTER：LoRA 适配器加载

## 为什么需要 LoRA 适配器

到目前为止，我们自定义模型的方式只有两种：一是通过 SYSTEM 提示词改变模型的行为模式，二是通过 PARAMETER 调整推理参数。这两种方式都是"软"修改——它们不改变模型的权重，只是影响模型如何使用已有的知识。但有时候你需要的是**"硬"修改**——让模型真正学会一些它原本不知道的知识或能力。

比如你想让 Qwen2.5 模型具备医学领域的专业知识，或者让 Llama 3.1 能理解你公司的内部术语和业务流程。这时候就需要 **LoRA（Low-Rank Adaptation）** 适配器了。

LoRA 的核心思想非常优雅：**不重新训练整个模型（那需要巨大的算力和数据），而是在原始权重的旁边添加少量可训练的参数**。这些参数以矩阵的形式存在，在推理时与原始权重合并，从而在不显著增加计算成本的前提下改变模型的行为。

```
┌─────────────────────────────────────────────────────────────┐
│                  LoRA 工作原理图解                            │
│                                                             │
│  原始权重矩阵 W (d x d):                                    │
│  ┌─────────────────────┐                                   │
│  │ 4096 x 4096 = 16M 参数 │ ← 基础模型，冻结不动            │
│  └─────────────────────┘                                   │
│                          +                                  │
│  LoRA 适配器 (低秩分解):                                     │
│  ┌───┐   ┌───┐                                             │
│  │ A │ x │ B │ = ΔW (秩 r << d)                           │
│  │dxr│   │rxd│    只有 ~0.1% 的参数量                       │
│  └───┘   ┘───┘                                             │
│                          =                                  │
│  有效权重 W' = W + ΔW                                       │
│  (模型行为被 LoRA "偏移"到新的方向)                          │
│                                                             │
│  资源对比:                                                   │
│  全量微调: 需要存储完整的新权重 (100%)                        │
│  LoRA:   只需存储 A 和 B 矩阵 (~0.1-1%)                     │
└─────────────────────────────────────────────────────────────┘
```

## Ollama 中使用 ADAPTER 指令

Ollama 通过 Modelfile 中的 `ADAPTER` 指令来加载 LoRA 适配器：

```dockerfile
# 基本用法
FROM qwen2.5:7b
ADAPTER ./medical-lora.gguf
SYSTEM """你是一个医学问答助手。"""
```

这三行代码做了什么？

1. `FROM qwen2.5:7b` — 加载基础模型（通用版 Qwen2.5）
2. `ADAPTER ./medical-lora.gguf` — 加载医学领域 LoRA 权重
3. `SYSTEM` — 设定角色提示词（配合 LoRA 使用效果更好）

当 Ollama 运行这个模型时，它会：
1. 将基础模型的 GGUF 权重加载到内存
2. 将 LoRA 适配器的 GGUF 权重也加载到内存
3. 在推理前将两者**按层合并**（W' = W + BA）
4. 使用合并后的权重进行正常的推理

### 多适配器组合（实验性）

```dockerfile
# 同时加载多个 LoRA（注意顺序可能影响结果）
FROM qwen2.5:7b
ADAPTER ./domain-knowledge.gguf     # 领域知识
ADAPTER ./writing-style.gguf        # 写作风格
SYSTEM """基于你的领域知识和特定写作风格回答问题。"""
```

> ⚠️ **注意**：多 LoRA 组合是 Ollama 的实验性功能，不同 LoRA 之间可能存在冲突。生产环境建议先充分测试。

## 获取 LoRA 适配器的渠道

### 渠道一：HuggingFace Hub

HuggingFace 是最大的 LoRA 适配器来源地：

```python
#!/usr/bin/env python3
"""搜索 HuggingFace 上适合 Ollama 的 GGUF 格式 LoRA"""

import requests

def search_gguf_lora(base_model, task=""):
    """搜索 GGUF 格式的 LoRA 适配器"""
    
    url = "https://huggingface/api/models"
    params = {
        "search": f"{base_model} lora gguf {task}".strip(),
        "author": "",
        "sort": "downloads",
        "direction": -1,
        "limit": 20
    }
    
    resp = requests.get(url, params=params, timeout=15)
    
    if resp.status_code != 200:
        print(f"❌ 搜索失败: HTTP {resp.status_code}")
        return
    
    models = resp.json()
    
    print(f"\n🔍 搜索: {base_model} + LoRA (GGUF格式)")
    print(f"{'='*70}\n")
    
    for m in models:
        model_id = m["id"]
        downloads = m.get("downloads", 0)
        likes = m.get("likes", 0)
        
        # 格式化下载量
        if downloads > 1e6:
            dl_str = f"{downloads/1e6:.1f}M"
        elif downloads > 1e3:
            dl_str = f"{downloads/1e3:.1f}K"
        else:
            dl_str = str(downloads)
        
        print(f"📦 {model_id}")
        print(f"   ⬇️ {dl_str}  ❤️ {likes}")
        print(f"   🔗 https://huggingface.co/{model_id}")
        print()

if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5"
    task = sys.argv[2] if len(sys.argv) > 2 else ""
    search_gguf_lora(base, task)
```

运行示例：

```bash
# 搜索 Qwen2.5 的代码相关 LoRA
python3 search_lora.py qwen2.5 code

# 搜索 Llama 3.1 的中文相关 LoRA  
python3 search_lora.py llama3.1 chinese
```

### 渠道二：Ollama 社区库

部分社区贡献者会将转换好的 LoRA 直接发布到 Ollama 库中：

```bash
# 查看是否有预打包的带 LoRA 模型
ollama library | grep -i lora
# 或通过 API
curl -s http://localhost:11434/api/tags | jq '.models[].name' | grep -i adapter
```

### 渠道三：自己转换（下一节详细讲）

如果你有 HuggingFace 格式的 safetensors LoRA 权重，可以转换为 GGUF 格式。

## 完整实战案例：构建医学问答模型

让我们走一遍从零开始构建一个带 LoRA 的自定义模型的完整流程：

### 第一步：准备基础模型和 LoRA 文件

```bash
# 1. 拉取基础模型
ollama pull qwen2.5:7b

# 2. 下载 LoRA 适配器（假设你已经找到了合适的）
# 从 HuggingFace 下载 GGUF 格式的 LoRA 文件
mkdir -p ~/ollama-adapters/medical
cd ~/ollama-adapters/medical

# 用 huggingface-cli 或 wget 下载 .gguf 文件
# 例如: wget https://huggingface.co/xxx/resolve/main/adapter.gguf
```

### 第二步：编写 Modelfile

```dockerfile
# ============================================================
#  Modelfile: medical-assistant
#  医学领域智能问答助手
#  基础模型: Qwen2.5 7B + 医学 LoRA 适配器
# ============================================================

FROM qwen2.5:7b

# ---------- LoRA 适配器 ----------
ADAPTER ./medical-qwen2.5-7b-lora.gguf

# ---------- 系统提示词 ----------
SYSTEM """你是"医问"，一个专业的医疗健康咨询 AI 助手。

专业背景：
- 你的知识库覆盖内科学、外科学、妇产科学、儿科学、
  中医学等主要临床学科
- 你了解常见疾病的症状、诊断方法、治疗方案和预防措施
- 你熟悉常用药物的适应症、用法用量和注意事项

核心原则：
1. 安全第一：任何涉及生命安全的建议必须明确标注风险
2. 不做诊断：你可以提供信息参考，但明确声明不能替代医生诊断
3. 建议就医：对于任何持续或严重的症状，建议尽快就医
4. 信息溯源：引用的信息应基于权威医学指南和研究

输出规范：
- 结构化回答：症状分析 → 可能原因 → 建议 → 就医指引
- 专业术语首次出现时附通俗解释
- 药物名称使用通用名，商品名放在括号中
- 所有医疗建议后附加标准免责声明

免责声明模板：
---
⚠️ 免责声明：以上信息仅供参考，不构成医疗诊断或治疗建议。
如有健康问题请及时就医，遵循专业医师指导。
---

交互规则：
- 如果用户描述的症状可能紧急，优先给出警示信息
- 对于非医疗问题（如保险、费用），礼貌地说明超出专业范围
- 保持同理心但不做过度情感化的表达
"""

# ---------- 推理参数 ----------
PARAMETER temperature 0.4          # 医学内容需要较高的准确性
PARAMETER top_p 0.9                # 标准核采样
PARAMETER top_k 40                 # 标准 Top-K
PARAMETER num_ctx 8192             # 支持较长的病情描述输入
PARAMETER num_predict 2048         # 控制输出长度避免冗长
PARAMETER repeat_penalty 1.15      # 避免重复
PARAMETER seed 42                  # 可复现（调试时有用）

# ---------- Few-shot 示例 ----------
MESSAGE user 我最近总是感到疲劳，睡眠也不好，是什么原因？
MESSAGE assistant ## 症状分析

您提到的"持续性疲劳伴睡眠障碍"是临床上常见的复合症状，可能涉及多个系统。

## 可能的原因（按常见程度排序）

| 可能原因 | 典型表现 | 相关因素 |
|---------|---------|---------|
| **睡眠呼吸暂停综合征** | 打鼾、白天嗜睡 | 肥胖、年龄>40 |
| **甲状腺功能减退** | 怕冷、便秘、体重增加 | 女性、中年 |
| **贫血** | 头晕、面色苍白 | 月经量大、饮食不均 |
| **焦虑/抑郁** | 心悸、兴趣减退 | 压力大、近期变故 |
| **慢性疲劳综合征** | 活动后加重、休息不缓解 | 病毒感染史 |

## 建议

1. 📋 **记录详情**：记录疲劳出现的时间、持续时间、加重/缓解因素
2. 🔬 **初步检查**：建议进行血常规（排除贫血）、甲状腺功能检查
3. 😴 **睡眠评估**：如伴有打鼾，建议做多导睡眠监测
4. 🏥 **就医指引**：如果症状持续超过 2 周，建议就诊于**全科医学科**或**内分泌科**

---
⚠️ 免责声明：以上信息仅供参考，不构成医疗诊断或治疗建议。
如有健康问题请及时就医，遵循专业医师指导。

MESSAGE user 头疼吃了布洛芬不管用怎么办？
MESSAGE assistant ## 分析

布洛芬（Ibuprofen）属于**非甾体抗炎药（NSAID）**，对大多数类型的头痛有效，但如果无效，需要考虑以下情况：

## 布洛芬无效的可能原因

1. **头痛类型不匹配**
   - 布洛芬对偏头痛的效果有限（尤其是不伴随炎症的典型偏头痛）
   - 紧张型头痛可能更需要肌肉松弛而非抗炎

2. **用药方式问题**
   - 剂量不足：成人单次推荐 400mg，最大日剂量不超过 1200mg（非处方）
   - 时机太晚：头痛发作早期用药效果更好

3. **药物过量性头痛**
   - ⚠️ 如果频繁服用止痛药（每月>15天），可能引发"反跳性头痛"

## 建议

🔴 **立即就医的红旗警示**：
- 突发的"一生中最剧烈的头痛"
- 伴发热、颈部僵硬、皮疹
- 伴视力改变、言语不清、肢体无力
- 头部外伤后的头痛

🟡 **建议就诊**：
- 内科 / 神经内科（根据头痛特征选择）

💊 **短期缓解建议**：
- 在医生指导下尝试其他类别药物（如对乙酰氨基酚、曲普坦类）
- 休息、安静环境、适量补水

---
⚠️ 免责声明：以上信息仅供参考，不构成医疗诊断或治疗建议。
如有健康问题请及时就医，遵循专业医师指导。

# ---------- 许可证 ----------
LICENSE """Apache 2.0
基础模型: Qwen2.5 7B (Apache 2.0)
LoRA 适配器: [具体来源和许可证]
本配置: MIT License
"""
```

### 第三步：创建并测试

```bash
# 创建自定义模型
ollama create medical-assistant -f Modelfile

# 测试基本功能
ollama run medical-assistant

>>> 我最近血糖有点高，空腹血糖 7.2 mmol/L，需要注意什么？
```

你会看到模型展现出明显的医学专业知识——它能准确识别 7.2 mmol/L 的空腹血糖处于糖尿病前期范围（正常 < 6.1，糖尿病 ≥ 7.0），并给出结构化且带有适当免责声明的回答。

## LoRA 与 SYSTEM 提示词的协同效应

一个重要的实践经验是：**LoRA 和 SYSTEM 提示词配合使用时效果远好于单独使用任何一个**。这是因为：

- **LoRA 改变了模型的"知识分布"**——让它知道更多关于某个领域的知识
- **SYSTEM 定义了模型的"行为框架"**——告诉它如何组织和使用这些知识

```python
#!/usr/bin/env python3
"""对比实验：纯 SYSTEM vs LoRA+SYSTEM vs 纯 LoRA"""

import requests
import time

def compare_approaches(questions):
    """三种方式的输出质量对比"""
    
    approaches = [
        {
            "name": "纯 SYSTEM (无 LoRA)",
            "model": "qwen2.5:7b",  # 基础模型
            "system": "你是一个医学专家，详细回答以下医学问题。"
        },
        {
            "name": "LoRA + SYSTEM",
            "model": "medical-assistant",  # 带 LoRA 的自定义模型
            "system": None  # 已内置在 Modelfile 中
        }
    ]
    
    for q in questions:
        print(f"\n{'='*70}")
        print(f"Q: {q[:60]}...")
        print(f"{'='*70}")
        
        for approach in approaches:
            start = time.time()
            
            messages = []
            if approach["system"]:
                messages.append({"role": "system", "content": approach["system"]})
            messages.append({"role": "user", "content": q})
            
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": approach["model"],
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.4}
                },
                timeout=120
            )
            
            elapsed = time.time() - start
            data = resp.json()
            content = data["message"]["content"]
            
            print(f"\n【{approach['name']}】({elapsed:.1f}s)")
            print(content[:500])
            if len(content) > 500:
                print("...")

compare_approaches([
    "2型糖尿病和1型糖尿病的主要区别是什么？",
    "长期服用阿司匹林有哪些副作用？",
    "儿童发热什么情况下需要立即送医？"
])
```

## 常见问题排查

### 问题一：LoRA 与基础模型不兼容

```bash
$ ollama create my-model -f Modelfile
Error: adapter architecture mismatch
```

**原因**：LoRA 适配器的架构必须与基础模型完全匹配。Qwen2.5 的 LoRA 不能用于 Llama 3.1。

**解决**：确认 LoRA 是为你的基础模型专门训练的。

### 问题二：LoRA 文件找不到

```bash
$ ollama create my-model -f Modelfile
Error: failed to open adapter: no such file or directory
```

**原因**：ADAPTER 中的路径是相对于当前工作目录的，不是绝对路径。

**解决**：
```bash
# 使用绝对路径
ADAPTER /home/user/adapters/my-lora.gguf

# 或者确保在正确的目录下执行 ollama create
cd /path/to/modelfile/directory
ollama create my-model -f Modelfile
```

### 问题三：加载 LoRA 后模型质量反而下降

**可能原因**：
1. LoRA 质量本身不好（过拟合、训练不充分）
2. LoRA 和 SYSTEM 提示词冲突（比如 SYSTEM 要求一种风格，LoRA 训练的是另一种风格）
3. 基础模型和 LoRA 的量化级别不匹配

**排查方法**：
```bash
# 先不带 LoRA 测试基础模型
ollama run qwen2.5:7b "测试问题"

# 再带 LoRA 测试
ollama run my-model "同样的测试问题"

# 对比两者的差异
```

## 本章小结

这一节我们学习了 Ollama 的 LoRA 适配器机制：

1. **LoRA 通过添加少量低秩参数**来改变模型行为，无需全量微调
2. **ADAPTER 指令**在 Modelfile 中加载 `.gguf` 格式的 LoRA 文件
3. **LoRA + SYSTEM 协同**能产生最佳效果——LoRA 提供领域知识，SYSTEM 定义行为框架
4. **HuggingFace 是获取 LoRA 的主要渠道**，也可以自己转换
5. **完整的实战案例**展示了如何构建一个医学问答模型
6. **兼容性是关键**——LoRA 必须与基础模型架构完全匹配

下一节我们将学习最硬核的内容：如何将任意 HuggingFace 模型转换为 Ollama 可用的 GGUF 格式。
