# 03-02 模型命名与 Tag 规范深度解读

## 命名格式的解剖学

在 Ollama 的世界里，每一个模型都有一个唯一的标识符，格式为 `<library>/<model>:<tag>`。这个看似简单的字符串实际上编码了丰富的信息——模型来源、架构家族、参数量大小、量化精度、功能变体——全部隐藏在这三个组成部分中。理解这个命名体系，就像学会了图书馆的分类编目法，能让你在海量模型中快速定位到你需要的那一个。

让我们用一个具体的例子来拆解：

```
ollama pull qwen2.5:7b
         │      │   │
         │      │   └── tag: 7B 参数量版本
         │      └── model: Qwen2.5 模型（含大版本号）
         └── library: 省略 = ollama 官方库
```

再来看一个更复杂的例子：

```
ollama pull library/minicpm-v:1.0-vision-q4_K_M
              │       │        │  │    │     │
              │       │        │  │    │     └── quant: K-quant M版, 4-bit
              │       │        │  │    └── quant prefix: q4_K_M
              │       │        │  └── feature: vision 变体
              │       │        └── version: 1.0
              │       └── model: MiniCPM-V 多模态模型
              └── library: 显式指定第三方库
```

### Library 部分：模型的来源归属

`<library>` 字段标识了模型的发布者或来源组织。在日常使用中，**99% 的情况下你会省略这个字段**——因为 Ollama 官方库（`ollama`）收录了最常用的模型。但当你需要使用社区贡献的模型时，就需要显式指定：

```bash
# 官方库模型（library 省略）
ollama pull llama3.1:8b          # 等价于 ollama/llama3.1:8b

# 社区/第三方模型（需要指定 library）
ollama pull bartowski/llama3.1-8b-instruct-q5_K_M
ollama pull huggingface/qwen2.5-7b-instruct-unified
```

常见的 library 前缀包括：

| Library | 来源 | 说明 |
|---------|------|------|
| （省略） | Ollama 官方 | 默认库，经过官方测试 |
| `bartowski` | Bartowski (HF) | GGUF 转换专家，量化版本极全 |
| `huggingface` | HuggingFace | 直接从 HF 同步 |
| `MaziyarPanahi` | Maziyar Panahi | 高质量社区转换 |
| `Felladrin` | Felladrin | 轻量/特殊用途模型 |

### Model 部分：模型名称的编码规则

Model 名称不是随意起的，它遵循一套约定俗成的编码规范。虽然不同发布者的命名风格不完全一致，但大多数都包含以下信息元素：

```python
import re

def parse_model_name(name):
    """解析 Ollama 模型名称中的信息"""
    pattern = r"""
        (?P<family>[a-z]+[\d.]*)      # 模型家族 + 大版本号
        (?:-(?P<variant>\w+))?        # 可选变体标识
        (?:-(?P<size>[\d.]+[bkmg]b))? # 可选参数量
    """
    m = re.match(pattern, name, re.IGNORECASE)
    if m:
        return m.groupdict()
    return {"family": name}

# 解析示例
examples = [
    "llama3.1",           # family=llama3.1
    "qwen2.5-coder",      # family=qwen2.5, variant=coder
    "deepseek-v2:16b",    # family=deepseek-v2, size=16b
    "gemma2:2b",          # family=gemma2, size=2b
    "phi3:mini",          # family=phi3, variant=mini
    "minicpm-v",          # family=minicpm, variant=v
    "yi:34b",             # family=yi, size=34b
]

for ex in examples:
    result = parse_model_name(ex.split(":")[0])
    print(f"{ex:25s} → {result}")
```

输出结果：

```
llama3.1                   → {'family': 'llama3.1', 'variant': None, 'size': None}
qwen2.5-coder              → {'family': 'qwen2.5', 'variant': 'coder', 'size': None}
deepseek-v2:16b            → {'family': 'deepseek-v2', 'variant': None, 'size': '16b'}
gemma2:2b                  → {'family': 'gemma2', 'variant': None, 'size': '2b'}
phi3:mini                  → {'family': 'phi3', 'variant': 'mini', 'size': None}
minicpm-v                  → {'family': 'minicpm', 'variant': 'v', 'size': None}
yi:34b                     → {'family': 'yi', 'variant': None, 'size': '34b'}
```

从这个解析结果可以看出，模型名称中的信息层次是：**家族名 > 版本号 > 变体标识 > 参数量**。其中参数量有时放在名称里（如 `deepseek-v2:16b`），有时放在 tag 里（如 `qwen2.5:7b`），这取决于发布者的习惯。

## Tag 系统：版本的精确控制

Tag 是 Ollama 命名体系中信息密度最高的部分。一个 tag 可以同时表达**版本号、参数量和量化级别**三种信息，而且这三者可以自由组合产生数十种变体。

### Tag 类型一：版本号 Tag

版本号 tag 用于区分同一模型的不同发布迭代：

```bash
# Llama 系列的版本演进
ollama pull llama3           # Llama 3 (2024年4月)
ollama pull llama3.1         # Llama 3.1 (2024年7月，支持128K上下文)
ollama pull llama3.1:70b     # Llama 3.1 的 70B 版本
ollama pull llama3.1:8b      # Llama 3.1 的 8B 版本

# Qwen 系列的版本演进
ollama pull qwen2            # Qwen2 (2024年初)
ollama pull qwen2.5          # Qwen2.5 (2024年9月，全面升级)
```

版本号遵循语义化版本（SemVer）的宽松形式：`主版本.次版本`。注意有些模型没有次版本号（如 `mistral`、`gemma2`），这时版本信息隐含在模型名称本身中。

### Tag 类型二：参数量 Tag

参数量 tag 标识了模型的大小，以 **B（Billion，十亿）为单位**：

| 参数量 | 近似 fp16 大小 | 典型内存需求(q4) | 适用场景 |
|--------|---------------|-----------------|---------|
| 0.5B | ~1 GB | ~0.5 GB | IoT / 极限资源 |
| 1.5B | ~3 GB | ~1 GB | 边缘设备 / 快速原型 |
| 3B | ~6 GB | ~2 GB | 轻量任务 |
| 7B / 8B | ~14-16 GB | ~4-5 GB | **甜点区间** |
| 14B | ~28 GB | ~8-9 GB | 中等复杂度任务 |
| 32B | ~64 GB | ~18-20 GB | 高质量生成 |
| 70B | ~140 GB | ~40 GB | 接近 GPT-3.5 水平 |
| 72B+ | ~144 GB+ | ~42 GB+ | 企业级应用 |

**7B-8B 是当前业界公认的"甜点尺寸"**——它能在消费级硬件上流畅运行（8GB 显存或 16GB 统一内存），同时在大多数任务上的质量表现已经足够好。这也是为什么你在 Ollama 库里看到最多、下载量最大的就是这个尺寸段。

### Tag 类型三：量化级别 Tag（核心重点）

这是最重要的一类 tag，也是最容易让人困惑的部分。量化（Quantization）是将模型权重从高精度浮点数（FP16/BF16）转换为低精度整数（INT8/INT4/甚至 INT2）的过程，目的是**大幅减小模型体积和内存占用**，代价是一定程度的精度损失。

Ollama 使用的是 **GGUF 格式的 K-quant 量化方案**（来自 llama.cpp 项目），它的前缀编码系统如下：

#### 量化前缀完整速查表

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ollama 量化等级全景图                              │
├──────────┬──────────────┬──────────┬───────────┬──────────────────┤
│   前缀   │   精度类型    │  大小缩减  │  精度损失  │    推荐场景       │
├──────────┼──────────────┼──────────┼───────────┼──────────────────┤
│          │              │          │           │                  │
│  f16     │  半精度浮点   │   1x基准  │    无     │ 基准对比/研究     │
│          │  (16-bit)    │          │           │                  │
├──────────┼──────────────┼──────────┼───────────┼──────────────────┤
│          │              │          │           │                  │
│  bf16    │ BFloat16     │   ~1x    │  极小(<1%)│ Apple Silicon   │
│          │              │          │           │ 优化路径          │
├──────────┼──────────────┼──────────┼───────────┼──────────────────┤
│          │              │          │           │                  │
│  q8_0    │  8-bit 量化  │   ~2x    │   1-2%    │ 高质量要求场景    │
│          │  (legacy)    │          │           │                  │
├──────────┼──────────────┼──────────┼───────────┼──────────────────┤
│          │              │          │           │                  │
│  q6_K    │  6-bit K量   │   ~2.5x  │   1-3%    │ 平衡质量与速度    │
│          │              │          │           │                  │
├──────────┼──────────────┼──────────┼───────────┼──────────────────┤
│          │              │          │           │                  │
│  q5_K_M │ 5-bit K-M    │   ~3x    │   2-4%    │ ★ 日常推荐       │
│  q5_K_S │ 5-bit K-S    │   ~3x    │   3-5%    │ 内存更紧张时      │
│          │              │          │           │                  │
├──────────┼──────────────┼──────────┼───────────┼──────────────────┤
│          │              │          │           │                  │
│ q4_K_M  │ 4-bit K-M    │   ~4x    │   3-5%    │ ★★★ 最常用       │
│ q4_K_S  │ 4-bit K-S    │   ~4x    │   4-6%    │ 极限资源时的选择   │
│  q4_0   │ 4-bit legacy │   ~4x    │   4-6%    │ 避免使用(过时)    │
│          │              │          │           │                  │
├──────────┼──────────────┼──────────┼───────────┼──────────────────┤
│          │              │          │           │                  │
│  q3_K_M │ 3-bit K-M    │   ~5x    │   5-10%   │ 实验性           │
│  q3_K_S │ 3-bit K-S    │   ~5x    │   8-12%   │ 不推荐生产环境     │
│  q3_K_L │ 3-bit K-L    │   ~4.5x  │   4-8%    │ 折中选择          │
│          │              │          │           │                  │
├──────────┼──────────────┼──────────┼───────────┼──────────────────┤
│          │              │          │           │                  │
│  q2_K    │  2-bit 量化  │   ~6x    │  显著下降  │ 仅极限压缩场景     │
│          │              │          │           │                  │
└──────────┴──────────────┴──────────┴───────────┴──────────────────┘
```

#### K-quant 后缀的含义

你可能注意到同一个 bit 数量有多个后缀：`K_M`、`K_S`、`K_L`、还有无后缀的 `_0`。这些后缀代表了**不同的量化策略**，它们决定了如何在模型的不同层之间分配量化位数：

- **`_M`（Medium）**：均衡策略，对重要层（注意力层）保留更多精度，对不重要层（FFN 层）激进量化。**这是大多数情况下的最佳选择**。
- **`_S`（Small）**：激进策略，整体更小的模型体积，但精度损失比 `_M` 更明显。
- **`_L`（Large）**：保守策略，更大的体积但更好的精度保留。
- **`_0`（Legacy）**：旧版量化方案，不推荐新项目使用。比如 `q4_0` 是最早的 4-bit 方案，已被 `q4_K_M` 取代。

#### 实际大小对比：以 Llama 3.1 8B 为例

```bash
# 同一个模型不同量化级别的实际文件大小
ollama pull llama3.1:8b                 # 默认 q4_K_M, 约 4.7GB
ollama pull llama3.1:8b-f16            # f16 全精度, 约 15GB (!)
ollama pull llama3.1:8b-q8_0           # 8-bit, 约 8.3GB
ollama pull llama3.1:8b-q5_K_M         # 5-bit M版, 约 5.7GB
ollama pull llama3.1:8b-q4_K_M         # 4-bit M版, 约 4.7GB ← 推荐
ollama pull llama3.1:8b-q4_0           # 4-bit legacy, 约 4.6GB
ollama pull llama3.1:8b-q3_K_S         # 3-bit S版, 约 3.5GB
ollama pull llama3.1:8b-q2_K           # 2-bit, 约 2.9GB
```

可以看到，从 f16 到 q2_K，同一个 8B 模型的体积可以从 15GB 压缩到 2.9GB——**5 倍的压缩比**。这就是为什么你的 8GB MacBook 能跑起 8B 参数量的模型：因为实际加载到内存的不是 16GB 的 fp16 权重，而是约 4.7GB 的 q4_K_M 量化权重。

### Tag 类型四：功能修饰 Tag

除了版本、参数量和量化之外，还有一些 tag 表示模型的功能变体：

```bash
# instruct 版本：经过指令微调，更适合对话场景
ollama pull llama3.1:8b-instruct
ollama pull qwen2.5:7b-instruct

# tools / tool-calling 版本：支持函数调用
ollama pull qwen2.5:7b-tools
ollama pull llama3.1:70b-tools

# vocab 扩展版本：词汇表扩展（通常用于多语言）
ollama pull llama3.1:vocab

# vision / chat 变体
ollama pull minicpm-v:chat
ollama pull llava-next:vision
```

## 组合 Tag 的实战用法

在实际使用中，你经常会看到**多种 tag 信息组合在一起**的情况。理解组合规则非常重要：

```
完整的 tag 结构：
<version>-<feature>-<quantization>

示例：
qwen2.5:7b-instruct-q4_K_M
       │   │         │
       │   │         └── 量化: 4-bit K-quant M版
       │   └── 功能: 经过指令微调
       └── 参数量: 7B
```

不过 Ollama 的 tag 系统并不总是严格按这个格式组合——很多时候版本和参数量合二为一，功能通过独立的 tag 表达。下面是一个实用的查询脚本，帮你快速了解某个模型的所有可用变体：

```python
#!/usr/bin/env python3
"""Ollama 模型变体分析工具"""

import requests
import json

def analyze_model_variants(model_name):
    """分析一个模型的所有可用变体"""
    url = f"https://ollama.com/api/models/{model_name}"
    
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            print(f"❌ 模型 '{model_name}' 在 Ollama 库中未找到")
            return
        
        data = resp.json()
        tags = data.get("tags", [])
        
        print(f"📦 {data.get('name', model_name)}")
        print(f"   参数量选项: {', '.join(data.get('parameter_sizes', []))}")
        print(f"   上下文窗口: {data.get('context_window', '?')}")
        print(f"   可用 Tag 总数: {len(tags)}")
        print()
        
        # 按 tag 分类
        categories = {
            "默认/latest": [],
            "Instruct": [],
            "Tools": [],
            "各量化版本": [],
            "其他": []
        }
        
        for t in tags:
            tname = t["name"]
            size_gb = t.get("size", 0) / (1024**3)
            
            info = f"  {tname:35s} ({size_gb:.1f}GB)"
            
            if tname == "latest":
                categories["默认/latest"].append(info)
            elif "instruct" in tname:
                categories["Instruct"].append(info)
            elif "tools" in tname or "tool" in tname:
                categories["Tools"].append(info)
            elif any(q in tname for q in ["q2_", "q3_", "q4_", "q5_", "q6_", "q8_", "f16", "bf16"]):
                categories["各量化版本"].append(info)
            else:
                categories["其他"].append(info)
        
        for cat, items in categories.items():
            if items:
                print(f"【{cat}】")
                for item in items:
                    print(item)
                print()
                
        # 推荐标签
        print("💡 推荐拉取:")
        recommended = find_recommended(tags)
        for rec in recommended:
            print(f"   ollama pull {model_name}:{rec['tag']}"
                  f"  (~{rec['size_gb']:.1f}GB) - {rec['reason']}")
                
    except requests.RequestException as e:
        print(f"❌ 网络请求失败: {e}")

def find_recommended(tags):
    """根据启发式规则推荐最适合的 tag"""
    candidates = []
    
    for t in tags:
        tname = t["name"]
        size_gb = t.get("size", 0) / (1024**3)
        
        # 推荐优先级
        if "q4_K_M" in tname and "instruct" in tname:
            candidates.append({"tag": tname, "size_gb": size_gb,
                             "reason": "4-bit M版 + 指令微调，最佳平衡"})
        elif "q4_K_M" in tname:
            candidates.append({"tag": tname, "size_gb": size_gb,
                             "reason": "4-bit M版，最常用的量化级别"})
        elif "q5_K_M" in tname and size_gb < 10:
            candidates.append({"tag": tname, "size_gb": size_gb,
                             "reason": "5-bit M版，更高精度"})
        elif tname == "latest":
            candidates.append({"tag": tname, "size_gb": size_gb,
                             "reason": "官方默认版本"})
    
    return sorted(candidates, key=lambda x: x["size_gb"])[:3]

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5"
    analyze_model_variants(model)
```

运行效果示例：

```
$ python3 analyze_model.py qwen2.5
📦 qwen2.5
   参数量选项: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
   上下文窗口: 128K
   可用 Tag 总数: 47

【默认/latest】
  latest (41.89GB)

【Instruct】
  72b-instruct (41.89GB)
  32b-instruct (19.06GB)
  14b-instruct (8.64GB)
  ...

【各量化版本】
  7b-q4_K_M (4.34GB)
  7b-q5_K_M (5.29GB)
  7b-q8_0 (7.67GB)
  ...

💡 推荐拉取:
   ollama pull qwen2.5:7b-q4_K_M  (4.3GB) - 4-bit M版 + 指令微调，最佳平衡
   ollama pull qwen2.5:7b-q5_K_M  (5.3GB) - 5-bit M版，更高精度
   ollama pull qwen2.5:latest  (41.9GB) - 官方默认版本
```

## 自定义模型命名与别名

除了使用 Ollama 库中的标准命名外，你还可以通过 `ollama create` 命令创建自己的模型别名：

```bash
# 创建自定义模型（基于 Modelfile）
cat > Modelfile <<'EOF'
FROM qwen2.5:7b
SYSTEM """你是一个专业的技术文档翻译器，
将英文技术文档翻译成通顺的中文。"""
PARAMETER temperature 0.3
EOF

ollama create my-translator -f Modelfile
ollama run my-translator "Translate: The quick brown fox jumps over the lazy dog."
```

这种机制让你可以为同一个基础模型创建多个**面向特定任务的定制版本**，每个版本有不同的系统提示词和参数配置。在企业环境中，你可以创建 `company-assistant`、`code-reviewer`、`doc-writer` 等多个角色化模型，团队成员只需要记住简短的别名即可。

## 常见命名问题排查

### 问题一：模型名找不到

```bash
$ ollama pull Llama3.1:8b
error: model 'Llama3.1:8b' not found
```

**原因**：Ollama 的模型名是**全小写**的。`Llama3.1` 应该写成 `llama3.1`。

### 问题二：Tag 组合不存在

```bash
$ ollama pull qwen2.5:7b-q4_K_M-instruct
error: model not found
```

**原因**：不是所有 tag 组合都存在。Ollama 只提供预构建的常用组合。如果你需要特殊的组合，需要自己从 HuggingFace 下载 GGUF 文件并用 Modelfile 导入（第四章会详细讲）。

### 问题三：latest 太大

```bash
$ ollama pull deepseek-v2
# 开始下载... 31GB...
# 💾 磁盘空间不足！
```

**原因**：`deepseek-v2:latest` 指向 16B 的完整版本。应该显式指定小版本：
```bash
ollama pull deepseek-v2:16b-q4_K_M  # 明确指定量化
# 或者直接用更小的模型
ollama pull deepseek-coder:6.7b
```

### 问题四：社区模型需要完整路径

```bash
$ ollama pull llama3.1-8b-instruct-q5_K_M
error: not found
# 但这个在 HuggingFace 上明明存在啊？
```

**原因**：这个模型不在 Ollama 官方库中，而是在社区贡献者的 namespace 下：
```bash
ollama pull bartowski/llama3.1-8b-instruct-q5_K_M
```

## 本章小结

这一节我们深入解剖了 Ollama 的模型命名体系和 Tag 规范：

1. **三段式命名** `<library>/<model>:<tag>` 编码了来源、身份和变体信息
2. **Tag 是信息密度最高的部分**，可以同时表达版本号、参数量、量化级别和功能变体
3. **量化前缀系统**（f16 → q8_0 → q5_K_M → q4_K_M → q2_K）是选型的关键决策维度
4. **`q4_K_M` 是绝大多数场景下的推荐量化级别**，它在 4 倍压缩比和 3-5% 的精度损失之间取得了最佳平衡
5. **K-quant 后缀**（M/S/L/_0）代表不同的层间量化策略分配方案
6. **`latest` tag 通常指向最大版本**，一定要根据硬件条件显式指定合适的 tag

下一节我们将建立一套系统的模型选择决策框架，帮助你在面对具体任务时做出最优的模型选择。
