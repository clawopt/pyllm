# 05-1 图像理解模型

## 多模态：从"读文字"到"看世界"

到目前为止，我们讨论的所有模型都只处理一种模态——文本。它们能读懂你写的代码、回答你的问题、翻译文档，但如果你给它们一张截图说"这个 UI 布局有什么问题"，它们会完全不知所措。这是因为纯文本模型的训练数据只包含文字，它们的"眼睛"从未被打开过。

**视觉语言模型（Vision-Language Model, VLM）**改变了这一切。这类模型在训练阶段同时接收图像和文本，学会了将视觉信息（像素、边缘、形状、颜色）与语义概念（"这是一只猫"、"这是登录按钮"、"图表显示趋势上升"）关联起来。Ollama 支持多种开源 VLM，让你可以在本地运行多模态 AI——无需调用 GPT-4V、Gemini Pro Vision 或 Claude 3 的云端 API。

```
┌─────────────────────────────────────────────────────────────┐
│              视觉语言模型的工作原理                           │
│                                                             │
│  输入图像                                                    │
│  ┌──────────┐                                               │
│  │ 🖼️ 图片   │ ──→ [视觉编码器 ViT] ──→ 图像特征向量          │
│  │ (像素)    │       (提取视觉特征)      (如: 768维 x N个patch) │
│  └──────────┘                                               │
│                         ↓                                    │
│                    [跨模态投影层]                             │
│                   (对齐到语言空间)                            │
│                         ↓                                    │
│  输入文本         ┌──────────────┐                          │
│  "描述这张图" ──→ │ 语言解码器 LLM │ ─→ 文本输出              │
│                  │ (生成回答)     │    "这是一张..."           │
│                  └──────────────┘                          │
│                                                             │
│  关键组件：                                                  │
│  - 视觉编码器：通常是 CLIP 或 SigLIP 的 ViT                 │
│  - 投影层：连接视觉和语言的桥梁 (MLP/Q-former)               │
│  - 语言解码器：就是普通的 LLM (Llama/Qwen/Mistral)          │
└─────────────────────────────────────────────────────────────┘
```

## Ollama 支持的视觉模型

### 模型列表与特点

| 模型 | 参数量 | 视觉编码器 | 大小(q4) | 特点 |
|------|--------|-----------|---------|------|
| **llava** | 7B/13B/34B | CLIP-ViT-L/336px | 4.5-20GB | 经典 VLM，生态最大 |
| **llava-next** | (同上) | CLIP-ViT-H/448px | 更大 | 更高分辨率支持 |
| **llava-phi3** | 3.8B | CLIP | ~2.5GB | 极小体积，适合低资源 |
| **bakllava** | 7B+13B | **SigLIP** | ~5GB | Llama + SigLIP 组合 |
| **moondream** | 1.6B | 自定义 | ~1.7GB | **最小**视觉模型 |
| **minicpm-v** | 2.5B/8B | 自定义 | 1.7-6GB | 中文优秀，高效 |
| **nanollava** | ~1B | 简化版 | <1GB | 极限压缩 |

### 如何选择

```python
#!/usr/bin/env python3
"""视觉模型选择指南"""

def recommend_vision_model(ram_gb, language="zh", quality="balanced"):
    """根据资源条件推荐视觉模型"""
    
    models = {
        "moondream": {
            "ram_min": 2, "size_gb": 1.7, "lang": "en>zh",
            "speed": "极快", "quality": "基础", "resolution": "336px"
        },
        "minicpm-v": {
            "ram_min": 4, "size_gb": 1.7, "lang": "zh>>en",
            "speed": "快", "quality": "良好", "resolution": "448px"
        },
        "llava-phi3": {
            "ram_min": 4, "size_gb": 2.5, "lang": "en~zh",
            "speed": "快", "quality": "良好", "resolution": "336px"
        },
        "bakllava": {
            "ram_min": 8, "size_gb": 5.0, "lang": "en>zh",
            "speed": "中等", "quality": "良好", "resolution": "336px"
        },
        "llava": {
            "ram_min": 8, "size_gb": 4.5, "lang": "en>zh",
            "speed": "中等", "质量": "标准", "resolution": "336px"
        },
        "llava-next": {
            "ram_min": 12, "size_gb": 8.0, "lang": "en>zh",
            "speed": "较慢", "quality": "优秀", "resolution": "448-672px"
        },
    }
    
    print(f"\n{'='*70}")
    print(f"  视觉模型推荐 (可用内存: {ram_gb}GB, 语言: {language})")
    print(f"{'='*70}\n")
    
    suitable = []
    for name, info in models.items():
        if ram_gb >= info["ram_min"]:
            info["name"] = name
            suitable.append(info)
    
    if not suitable:
        print("❌ 可用内存不足，建议至少 2GB")
        return
    
    suitable.sort(key=lambda x: x["ram_min"])
    
    for m in suitable:
        rec_mark = " ⭐推荐" if m["name"] == ("minicpm-v" if language == "zh" else "llava") else ""
        lang_match = "✅" if (language == "zh" and "zh>" in m["lang"]) or \
                     (language != "zh" and m["lang"].startswith("en")) else "⚠️"
        
        print(f"  {m['name']:<15s} | {m['size_gb']:>5.1f}GB | "
              f"{m['speed']:<6s} | {m['quality']:<6s} | "
              f"{m['resolution']:<10s} | {lang_match}{rec_mark}")
    
    best = suitable[0]
    print(f"\n💡 首选推荐: ollama pull {best['name']}")

if __name__ == "__main__":
    import sys
    ram = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
    recommend_vision_model(ram)
```

## 运行方式一：命令行交互

```bash
# 拉取视觉模型
ollama pull minicpm-v

# 方式一：直接传入图片路径
ollama run minicpm-v "描述这张图片: /path/to/image.png"

# 方式二：交互式对话中发送图片
ollama run minicpm-v
>>> 请分析这张图片
>>> (然后粘贴图片或输入路径)
```

## 运行方式二：API 调用（核心重点）

API 是多模态模型的主要使用场景。Ollama 通过 `/api/chat` 和 `/api/generate` 接口支持图片输入：

### Base64 编码方式

```python
#!/usr/bin/env python3
"""Ollama 视觉模型 API 调用 - Base64 方式"""

import requests
import base64
from pathlib import Path

class OllamaVisionClient:
    """Ollama 视觉模型客户端"""
    
    def __init__(self, base_url="http://localhost:11434", model="minicpm-v"):
        self.base_url = base_url
        self.model = model
    
    def _encode_image(self, image_path):
        """将图片编码为 base64"""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")
        
        # 根据文件扩展名确定 MIME 类型
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        ext = path.suffix.lower()
        mime_type = mime_map.get(ext, "image/png")
        
        with open(path, "rb") as f:
            data = f.read()
        
        b64 = base64.b64encode(data).decode("utf-8")
        return b64, mime_type
    
    def analyze_image(self, image_path, prompt, stream=False):
        """分析单张图片"""
        
        b64, mime_type = self._encode_image(image_path)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
                    ]
                }
            ],
            "stream": stream,
            "options": {"temperature": 0.3}
        }
        
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120
        )
        
        return resp.json()["message"]["content"]
    
    def analyze_images_batch(self, image_paths, prompt):
        """同时分析多张图片进行对比"""
        
        content = [{"type": "text", "text": prompt}]
        
        for img_path in image_paths:
            b64, mime_type = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"}
            })
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "options": {"temperature": 0.3}
        }
        
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=180
        )
        
        return resp.json()["message"]["content"]
    
    def stream_analyze(self, image_path, prompt):
        """流式输出分析结果"""
        
        b64, mime_type = self._encode_image(image_path)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
                    ]
                }
            ],
            "stream": True,
            "options": {"temperature": 0.3}
        }
        
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=180
        )
        
        for line in resp.iter_lines():
            if line:
                data = line.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk = json.loads(data[6:])
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        print(content, end="", flush=True)


if __name__ == "__main__":
    client = OllamaVisionClient(model="minicpm-v")
    
    # 示例 1：基本图像描述
    print("=== 图像描述 ===\n")
    result = client.analyze_image(
        "./test_screenshot.png",
        "请详细描述这张图片的内容，包括布局、文字、颜色等所有可见元素。"
    )
    print(result)
    
    # 示例 2：UI 截图分析
    print("\n=== UI 分析 ===\n")
    ui_result = client.analyze_image(
        "./dashboard_screenshot.png",
        """作为一个前端开发专家，请分析这个仪表盘界面：
1. 整体布局结构是什么？
2. 有哪些 UI 组件？
3. 从用户体验角度有什么改进建议？
4. 是否有明显的可用性问题？"""
    )
    print(ui_result)
    
    # 示例 3：多图对比
    print("\n=== 多图对比 ===\n")
    compare_result = client.analyze_images_batch(
        ["./design_v1.png", "./design_v2.png"],
        "对比这两张设计稿的差异，列出所有变化点并评价哪个版本更好。"
    )
    print(compare_result)
```

### URL / 本地路径方式

除了 Base64 编码，Ollama 还支持更简洁的路径引用方式：

```python
# 使用 file:// 协议引用本地图片
payload = {
    "model": "minicpm-v",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "分析这张图片"},
            {"type": "image_url",
             "image_url": {"url": "file:///Users/user/Desktop/screenshot.png"}}
        ]
    }],
    "stream": False
}

# 使用 http(s) URL 引用网络图片
payload = {
    "model": "minicpm-v",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "这张网络图片显示了什么？"},
            {"type": "image_url",
             "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    "stream": False
}

# 使用 /api/generate 接口（更底层的方式）
payload = {
    "model": "minicpm-v",
    "prompt": "描述这张图片",
    "images": ["file:///path/to/image.png"],
    "stream": False
}
resp = requests.post("http://localhost:11434/api/generate", json=payload)
```

## 实战案例集

### 案例一：UI 截图自动 Bug 报告

```python
#!/usr/bin/env python3
"""UI 截图 → 自动生成 Bug 报告"""

import requests
import base64
import json
from datetime import datetime

def generate_bug_report(image_path, context=""):
    """根据 UI 截图自动生成结构化 Bug 报告"""
    
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    system_prompt = """你是一个专业的 QA 工程师。当你收到 UI 截图时，
你需要像放大镜一样检查每一个细节，找出任何可能的缺陷。

检查维度：
🔴 功能性缺陷：按钮无响应、数据不显示、链接失效等
🟠 UI/UX 问题：错位、溢出、遮挡、不一致
🟡 设计规范违反：颜色/字体/间距不符合设计系统
🔵 性能暗示：加载状态缺失、骨架屏问题
⚪ 文案问题：错别字、翻译遗漏、语气不当

输出格式（严格 JSON）：
{
  "summary": "一句话概述发现的问题",
  "severity": "Critical|High|Medium|Low|Info",
  "category": "Functional|UIUX|Design|Performance|Content",
  "issues": [
    {
      "title": "问题标题",
      "description": "详细描述",
      "location": "在界面的什么位置",
      "expected": "期望行为",
      "actual": "实际行为",
      "suggestion": "修复建议"
    }
  ],
  "positive_notes": ["做得好的地方（至少一条）"]
}"""
    
    payload = {
        "model": "minicpm-v",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", 
                 "text": f"""请仔细审查这张 UI 截图，生成完整的 Bug 报告。
{context}
当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120
    )
    
    raw_output = resp.json()["message"]["content"]
    
    # 尝试解析 JSON
    try:
        # 提取 JSON 部分（模型可能包裹在 markdown 代码块中）
        import re
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            report = json.loads(json_match.group())
            return report
        return {"raw": raw_output}
    except json.JSONDecodeError:
        return {"raw": raw_output}


if __name__ == "__main__":
    report = generate_bug_report(
        "./bug_screenshot.png",
        context="这是一个电商网站的购物车页面，用户反馈无法使用优惠券。"
    )
    
    print(json.dumps(report, indent=2, ensure_ascii=False))
```

### 案例二：OCR 增强：发票信息提取

```python
#!/usr/bin/env python3
"""发票/收据 OCR + 结构化信息提取"""

import requests
import base64
import json

def extract_invoice_info(image_path):
    """从发票图片中提取结构化信息"""
    
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "model": "minicpm-v",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": """这是一张发票/收据的照片。
请精确提取以下信息并以 JSON 格式输出（如果某项找不到则填 null）：

{
  "invoice_type": "增值税专用发票/普通发票/电子发票/收据/其他",
  "invoice_number": "发票号码",
  "invoice_code": "发票代码",
  "issue_date": "开票日期 (YYYY-MM-DD)",
  "seller_name": "销售方名称",
  "seller_tax_id": "销售方税号",
  "buyer_name": "购买方名称",
  "buyer_tax_id": "购买方税号",
  "items": [
    {"name": "商品/服务名称", "specification": "规格型号",
     "unit": "单位", "quantity": 数量, "unit_price": 单价,
     "amount": 金额, "tax_rate": 税率, "tax_amount": 税额}
  ],
  "total_amount": "合计金额(不含税)",
  "total_tax": "合计税额",
  "total_with_tax": "价税合计(大写)",
  "currency": "CNY/USD/其他",
  "remarks": "备注"
}

注意：
- 金额数字请保留原始精度，不要四舍五入
- 日期统一为 YYYY-MM-DD 格式
- 只输出 JSON，不要其他内容"""},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120
    )
    
    output = resp.json()["message"]["content"]
    
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', output)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    return {"raw_text": output}


if __name__ == "__main__":
    result = extract_invoice_info("./invoice_photo.jpg")
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

### 案例三：图表数据提取

```python
#!/usr/bin/env python3
"""图表/图形理解 + 数据提取"""

import requests
import base64
import csv

def extract_chart_data(image_path, output_csv=None):
    """从图表图片中提取数据"""
    
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "model": "llava",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": """请分析这张图表并提取其中的数据。

步骤：
1. 识别图表类型（柱状图/折线图/饼图/散点图/面积图/其他）
2. 提取标题和坐标轴标签
3. 读取每个数据点的值（尽可能精确）
4. 输出 CSV 格式的原始数据

输出格式：
## 图表信息
类型: [图表类型]
标题: [图表标题]
X轴: [X轴标签]
Y轴: [Y轴标签]

## 数据 (CSV格式)
category,value
[数据行1]
[数据行2]
...

## 观察
[你对这个图表趋势的简要观察]"""},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120
    )
    
    output = resp.json()["message"]["content"]
    print(output)
    
    # 如果指定了输出文件，尝试提取 CSV 部分
    if output_csv:
        import re
        csv_match = re.search(r'## 数据.*?\n\n(.*?)(?:\n##|\Z)', output, re.DOTALL)
        if csv_match:
            with open(output_csv, "w") as f:
                f.write(csv_match.group(1).strip())
            print(f"\n✅ CSV 数据已保存到 {output_csv}")


if __name__ == "__main__":
    extract_chart_data("./sales_chart.png", "extracted_data.csv")
```

## 性能注意事项

视觉模型比纯文本模型消耗更多资源，原因如下：

```
┌──────────────────────────────────────────────────────────┐
│              视觉模型资源开销分析                           │
│                                                          │
│  一个 7B 的视觉模型 vs 7B 的纯文本模型：                    │
│                                                          │
│  纯文本 7B:                                                │
│  ├── 模型权重: ~4.7 GB (q4_K_M)                          │
│  ├── KV Cache: ~0.5 GB (4K ctx)                          │
│  └── 总计: ~5.2 GB                                       │
│                                                          │
│  视觉 7B (LLaVA):                                         │
│  ├── 语言模型权重: ~4.7 GB                                │
│  ├── 视觉编码器权重: ~0.5 GB (ViT-L/336)                 │
│  ├── 图像 token 化后: 一张图 ≈ 576-2304 个额外 token       │
│  │   └── 这些 token 占据大量 KV Cache                     │
│  ├── KV Cache: ~1.5-4 GB (因图像 token 而增大)            │
│  └── 总计: ~7-12 GB (!)                                  │
│                                                          │
│  结论：视觉模型的实际内存需求是同级文本模型的 1.5-2 倍       │
└──────────────────────────────────────────────────────────┘
```

优化建议：

```bash
# 1. 选择合适的分辨率（不要盲目追求高分辨率）
# minicpm-v 和 moondream 在较低分辨率下效果已经不错

# 2. 压缩输入图片
# 将图片缩放到 800x600 以下再送入模型
# 使用 PNG 而非无损压缩（避免解压开销）

# 3. 减少 num_ctx（如果不需要超长上下文）
PARAMETER num_ctx 4096  # 默认可能更大

# 4. 批量处理时串行执行（视觉模型通常不支持真正的并发）
```

## 本章小结

这一节全面介绍了 Ollama 的图像理解能力：

1. **视觉语言模型（VLM）**通过视觉编码器 + 跨模态投影层 + 语言解码器的架构实现"看图说话"
2. **Ollama 支持 7 种主流视觉模型**，从 1.7GB 的 moondream 到 20GB+ 的 llava-34b
3. **API 调用支持三种传图方式**：Base64 编码、file:// 本地路径、HTTP(S) URL
4. **三个实战案例**展示了 UI Bug 报告自动生成、发票 OCR 结构化提取、图表数据提取
5. **视觉模型的内存开销约为同级文本模型的 1.5-2 倍**，需要特别注意资源规划

下一节我们将探索视频理解的实现方法——虽然 Ollama 不原生支持视频，但可以通过帧抽取 + 批量分析的方案来实现。
