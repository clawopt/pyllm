---
title: 图像理解：让 LLM "看"图片
description: 多模态消息格式、图片 URL 与 Base64、视觉问答、OCR、图表分析实战
---
# 图像理解：让 LLM "看"图片

上一节我们了解了多模态的基本概念。现在让我们动手实现第一个多模态能力——**图像理解（Vision）**：把图片传给模型，让它"看到"图片内容并回答相关问题。这是多模态 AI 中最成熟、最有实用价值的能力之一。

## 基础用法：发送一张图片

在 LangChain 中，给模型发送图片只需要构造一个特殊格式的 `HumanMessage`：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

chat = ChatOpenAI(model="gpt-4o", temperature=0)

# 构造包含图片的消息
message = HumanMessage(content=[
    {
        "type": "text",
        "text": "这张图片里有什么？请详细描述一下"
    },
    {
        "type": "image_url",
        "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Unsharpened_Cat.png/220px-Unsharpened_Cat.png"}
    }
])

response = chat.invoke([message])
print(response.content)
```

输出类似这样：

```
这张图片展示了一只猫的轮廓剪影图像。这是一张黑白（灰度）图像，
猫呈现为坐姿或蹲姿，面向左侧。图像整体比较模糊，
看起来像是经过某种边缘检测或轮廓提取处理的效果。
```

关键点解析：

- `content` 不再是字符串，而是一个 **列表**
- 列表中每个元素是一个字典，用 `"type"` 标识内容类型
- `"type": "text"` 表示文本部分，`"type": "image_url"` 表示图片部分
- 图片通过 URL 引用——可以是网络 URL 或本地文件路径

## 两种传入图片的方式

### 方式一：URL 引用（推荐用于网络图片）

```python
message = HumanMessage(content=[
    {"type": "text", "text": "描述这张图"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])
```

这是最简单的方式——直接给一个 URL，LangChain/OpenAI SDK 会自动下载并处理。

### 方式二：Base64 编码（推荐用于本地文件）

如果你的图片在本地磁盘上（比如用户上传的文件），用 Base64 编码更可靠：

```python
import base64

def image_to_base64(image_path: str, encoding: str = "utf-8") -> str:
    """将本地图片文件转换为 Base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(encoding)

# 使用
image_base64 = image_to_base64("photos/cat.jpg")

message = HumanMessage(content=[
    {"type": "text", "text": "这是什么动物？"},
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}"
        }
    }
])

response = chat.invoke([message])
print(response.content)
```

注意 `data:image/jpeg;base64,{base64_string}` 这种格式——它叫做 **Data URI**，是一种把二进制数据内嵌在 URL 中的标准方式。前缀中的 `image/jpeg` 是 MIME 类型，需要根据实际图片格式调整：

| 图片格式 | MIME 类型 |
|---------|----------|
| JPEG / JPG | `image/jpeg` |
| PNG | `image/png` |
| GIF | `image/gif` |
| WebP | `image/webp` |

### 如何自动检测图片格式

```python
import mimetypes

def get_mime_type(file_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "image/jpeg"   # 默认 fallback

mime = get_mime_type("photo.png")     # image/png
mime = get_mime_type("screenshot.jpg") # image/jpeg
```

## 实战场景一：图片描述与问答

最基本的视觉能力是让模型描述图片内容并回答相关问题：

```python
chat = ChatOpenAI(model="gpt-4o", temperature=0)

def describe_image(image_source: str, question: str = "请详细描述这张图片"):
    """
    分析一张图片并回答问题
    
    Args:
        image_source: 图片 URL 或本地文件路径
        question: 向模型提问的问题
    """
    if os.path.exists(image_source):
        base64_data = image_to_base64(image_source)
        mime = get_mime_type(image_source)
        image_url = f"data:{mime};base64,{base64_data}"
    else:
        image_url = image_source

    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": image_url}}
    ])

    response = chat.invoke([message])
    return response.content
```

使用示例：

```python
# 描述一张风景照
result = describe_image(
    "photos/sunset.jpg",
    "这张照片是在什么时间拍摄的？天气怎么样？"
)
print(result)
# 这张照片拍摄于日落时分（黄金时段），天空呈现出橙色到紫色的渐变...
```

## 实战场景二：OCR — 从图片中提取文字

GPT-4o 的 OCR 能力非常强大，甚至能处理手写文字、倾斜文本和复杂排版：

```python
chat = ChatOpenAI(model="gpt-4o", temperature=0)

message = HumanMessage(content=[
    {
        "type": "text",
        "text": (
            "请完整提取图片中的所有文字内容。"
            "保持原有的结构和格式，不要遗漏任何信息。"
            "如果图片中没有文字，请明确说明。"
        )
    },
    {
        "type": "image_url",
        "image_url": {"url": "photos/receipt.jpg"}
    }
])

response = chat.invoke([message])
print(response.content)
```

假设 `receipt.jpg` 是一张超市小票的拍照图片，输出大概是这样的：

```
*** 超市收银小票 ***

店名：幸福超市
日期：2025-01-15 14:32
收银员：003-王芳

商品          数量   单价    小计
----------------------------------------
有机牛奶      2      ¥12.50  ¥25.00
全麦面包      1      ¥8.00   ¥8.00
鸡蛋(10枚)    1      ¥18.00  ¥18.00
西红柿        0.5kg  ¥6.80   ¥3.40

合计：¥54.40
支付方式：微信支付
订单号：2025011514320088
```

可以看到，模型不仅识别出了所有文字，还**保留了表格的结构和数字的准确性**。这比传统的 OCR 工具（如 Tesseract）强得多——传统工具经常把数字认错（比如 `8` 认成 `3`、`0` 认成 `O`），而且对复杂版面的结构还原能力有限。

## 实战场景三：图表数据分析

对于数据可视化图表（柱状图、折线图、饼图等），GPT-4o 能直接"读懂"趋势和数值：

```python
chat = ChatOpenAI(model="gpt-4o", temperature=0)

message = HumanMessage(content=[
    {
        "type": "text",
        "text": """请分析这张图表：
1. 图表类型是什么？
2. 横轴和纵轴分别代表什么？
3. 数据的整体趋势是什么？
4. 有没有异常值或值得关注的点？
5. 用一句话总结图表的核心结论"""
    },
    {
        "type": "image_url",
        "image_url": {"url": "photos/sales_chart.png"}
    }
])

response = chat.invoke([message])
print(response.content)
```

典型输出：

```
1. **图表类型**：柱状图（Bar Chart），采用分组柱状形式。

2. **坐标轴含义**：
   - 横轴（X轴）：季度（Q1、Q2、Q3、Q4）
   - 纵轴（Y轴）：销售额（单位：万元）

3. **整体趋势**：
   - A产品线：Q1→Q2 上升，Q2 达到峰值后逐步下降
   - B产品线：持续稳步增长，Q4 达到最高点
   - 整体来看，两条产品线呈现互补态势

4. **值得关注**：
   - Q2 的A产品线销售额（约85万）是全年最高点
   - B产品线在Q4反超A产品线（约78万 vs 约55万）
   - A产品线下半年下滑明显，可能需要分析原因

5. **核心结论**：公司总销售额保持增长，但产品结构正在发生变化，
   B产品线正在成为新的增长引擎。
```

这种能力在实际业务中非常有价值——你不需要手动从图表中读取数据再输入系统，直接把截图丢给模型就能得到结构化的分析结果。

## 实战场景四：代码截图诊断

程序员经常会截一张报错截图发给别人求助。有视觉能力的模型可以直接"看"报错信息并给出解决方案：

```python
chat = ChatOpenAI(model="gpt-4o", temperature=0)

message = HumanMessage(content=[
    {
        "type": "text",
        "text": """我运行这段代码时遇到了错误（见截图）。
请帮我：
1. 读出完整的错误信息
2. 分析错误原因
3. 给出修复代码"""
    },
    {
        "type": "image_url",
        "image_url": {"url": "photos/error_screenshot.png"}
    }
])

response = chat.invoke([message])
print(response.content)
```

输出：

```
**1. 错误信息**

```
TypeError: unsupported operand type(s) for +: 'int' and 'str'
    at line 12 in calculator.py
```

**2. 原因分析**

第 12 行尝试将整数和字符串直接相加。
Python 不支持 int + str 的操作（这与 JavaScript 不同）。
从代码上下文看，`result = age + suffix` 中 `age` 是整数（25），
而 `suffix` 是字符串（"岁"）。

**3. 修复方案**

```python
# 方案一：把整数转为字符串
result = str(age) + suffix

# 方案二：使用 f-string（更 Pythonic）
result = f"{age}{suffix}"

# 方案三：使用 format()
result = "{}{}".format(age, suffix)
```

推荐方案二（f-string），简洁且性能好。
```

这就是多模态编程助手的威力——**用户不需要手动敲出完整的报错堆栈，截图就够了**。

## 构建可复用的视觉 Chain

把上面的功能封装成一个标准的 LangChain Chain，方便后续组合使用：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

chat = ChatOpenAI(model="gpt-4o", temperature=0)

vision_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的图像分析师。仔细观察图片内容，准确回答用户的问题。如果不确定，就说不知道。"),
    MessagesPlaceholder(variable_name="user_message")
])

vision_chain = vision_prompt | chat

def analyze_image(image_source: str, question: str):
    """通用的图像分析函数"""
    import base64, mimetypes, os
    
    if os.path.exists(image_source):
        with open(image_source, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        mime = mimetypes.guess_type(image_source)[0] or "image/jpeg"
        url = f"data:{mime};base64,{b64}"
    else:
        url = image_source

    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": url}}
    ])

    return vision_chain.invoke({"user_message": [message]}).content


# 使用示例
result = analyze_image("photos/invoice.png", "这张发票的总金额是多少？税率是多少？")
print(result)
```

## 注意事项与限制

**图片大小限制**。GPT-4o 对单张图片的建议尺寸是不超过 2048×2048 像素，文件大小不超过 20MB。过大的图片不仅增加 API 成本和处理时间，还可能被拒绝处理。最佳实践是对大图先做缩放：

```python
from PIL import Image

def resize_image(input_path: str, max_size: int = 2048) -> str:
    """缩放图片到指定最大边长"""
    img = Image.open(input_path)
    img.thumbnail((max_size, max_size))
    
    output_path = f"resized_{os.path.basename(input_path)}"
    img.save(output_path)
    return output_path
```

**成本考量**。带图片的 API 调用比纯文本贵不少。GPT-4o 的图片输入按"token 等价量"计费——一张 1024×1024 的图片大约相当于 1000+ 个 token。如果应用中频繁处理图片，需要注意控制成本。

**隐私安全**。图片会发送到 OpenAI 的服务器进行处理。如果图片包含敏感信息（如身份证、银行卡、内部文档截图），需要评估合规风险。对于高敏感场景，可以考虑使用本地部署的多模态模型（如 Ollama + LLaVA）。

下一节我们将学习另一种重要的模态——**语音交互**，包括语音转文字（STT）和文字转语音（TTS），让你的助手不仅能"看见"，还能"听见"和"说话"。
