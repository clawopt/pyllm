# 05-3 多模态实战项目

## 项目一：智能截图助手 — 自动生成 Bug 报告

### 项目背景

在软件开发团队中，测试人员发现 UI 问题后的标准流程是：截图 → 手动描述问题 → 填写 Bug 报告模板 → 分配给开发。这个过程繁琐且容易遗漏信息。我们的目标是构建一个工具，让用户只需**截图并简单描述上下文**，就能自动生成一份结构完整、可以直接提交到 Jira/Linear 的 Bug 报告。

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│           智能截图助手 - 系统架构                             │
│                                                             │
│  用户操作                                                    │
│  截图 (Cmd+Shift+4) + 一句话描述                              │
│       │                                                     │
│       ▼                                                     │
│  [截图监听器]                                                │
│  │  监控剪贴板 / 指定目录 / macOS Screenshot API             │
│       │                                                     │
│       ▼                                                     │
│  [Ollama 视觉分析引擎]                                       │
│  │  模型: minicpm-v                                         │
│  │  输入: 截图 + 用户描述 + 项目上下文                       │
│  │  处理: 缺陷检测 / 严重度评估 / 分类 / 复现步骤推断        │
│       │                                                     │
│       ▼                                                     │
│  [报告生成器]                                                │
│  │  格式: Jira JSON / Markdown / HTML                      │
│  │  包含: 标题/描述/严重度/分类/复现步骤/预期vs实际/建议     │
│       │                                                     │
│       ▼                                                     │
│  输出: 结构化 Bug 报告 → 可直接提交或复制                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 完整代码实现

```python
#!/usr/bin/env python3
"""
智能截图 Bug 助手 (SmartScreenshot Bug Assistant)
功能: 截图 + 一句话描述 → 自动生成结构化 Bug 报告
依赖: Ollama (minicpm-v) + Pillow (可选)
"""

import requests
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

class SmartBugReporter:
    """智能 Bug 报告生成器"""
    
    def __init__(self, 
                 vision_model="minicpm-v",
                 text_model="qwen2.5:7b",
                 ollama_url="http://localhost:11434"):
        
        self.vision_model = vision_model
        self.text_model = text_model
        self.base_url = ollama_url
        
        # 项目上下文（可自定义）
        self.project_context = {
            "app_name": "Web Dashboard",
            "tech_stack": "React + TypeScript + Tailwind CSS",
            "design_system": "Ant Design 5.x",
            "target_devices": "Desktop (1920x1080), Tablet, Mobile"
        }
    
    def _encode_image(self, image_path):
        """编码图片为 Base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    def _call_vision(self, image_b64, prompt):
        """调用视觉模型"""
        payload = {
            "model": self.vision_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }],
            "stream": False,
            "options": {"temperature": 0.2}
        }
        resp = requests.post(f"{self.base_url}/api/chat", 
                            json=payload, timeout=120)
        return resp.json()["message"]["content"]
    
    def _call_text(self, prompt):
        """调用文本模型"""
        payload = {
            "model": self.text_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3}
        }
        resp = requests.post(f"{self.base_url}/api/chat", 
                            json=payload, timeout=120)
        return resp.json()["message"]["content"]
    
    def analyze_screenshot(self, image_path, user_description=""):
        """
        分析截图并生成 Bug 报告
        
        Args:
            image_path: 截图文件路径
            user_description: 用户的一句话描述（可选）
        
        Returns:
            dict: 结构化的 Bug 报告
        """
        
        print(f"\n🔍 正在分析截图: {image_path}")
        if user_description:
            print(f"📝 用户描述: {user_description}")
        
        start_time = time.time()
        image_b64 = self._encode_image(image_path)
        
        # === 阶段 1: 视觉分析 ===
        print("  [1/3] 视觉分析中...")
        
        vision_prompt = f"""你是一个专业的 QA 工程师和 UI/UX 审查专家。
请对这张应用截图进行全面的缺陷分析。

## 应用上下文
- 应用名称: {self.project_context['app_name']}
- 技术栈: {self.project_context['tech_stack']}
- 设计系统: {self.project_context['design_system']}

## 用户提供的额外信息
{user_description if user_description else '(无)'}

## 分析要求（请逐一检查以下每个维度）

### 🔴 功能性缺陷
- 按钮是否可点击？是否有无响应的情况？
- 数据是否正确显示？是否有空值、错误值、格式异常？
- 表单是否正常工作？验证、提交、重置？
- 导航链接是否有效？

### 🟠 UI/UX 问题  
- 元素是否对齐？间距是否一致？
- 文字是否溢出容器？是否有截断？
- 是否有元素重叠或遮挡？
- 颜色对比度是否符合可访问性标准？
- 加载状态/空状态/错误状态是否正确处理？

### 🟡 设计规范违反
- 字体大小、字重是否符合规范？
- 圆角、阴影、边距是否一致？
- 图标使用是否正确和统一？
- 暗色模式支持？（如果适用）

### ⚪ 其他问题
- 错别字或文案问题
- 性能问题暗示（如大量数据未分页）
- 安全隐患（如敏感信息暴露）

请以严格的 JSON 格式输出：
```json
{{
  "ui_summary": "一句话概括界面整体状态",
  "defects": [
    {{
      "id": "D01",
      "title": "简短的问题标题(10字以内)",
      "severity": "Critical|High|Medium|Low|Info",
      "category": "Functional|UIUX|Design|Performance|Content|Security",
      "location": "在界面的什么位置(用相对位置描述)",
      "description": "详细描述问题(2-3句话)",
      "expected_behavior": "期望的正确行为",
      "actual_behavior": "实际观察到的行为",
      "reproduction_steps": ["步骤1", "步骤2"],
      "suggestion": "修复建议"
    }}
  ],
  "positive_notes": ["做得好的地方(至少1条)"],
  "accessibility_issues": ["可访问性问题(如有)"]
}}
```"""
        
        vision_result = self._call_vision(image_b64, vision_prompt)
        print(f"  ✅ 视觉分析完成 ({time.time()-start_time:.1f}s)")
        
        # 解析视觉分析结果
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', vision_result)
            vision_data = json.loads(json_match.group()) if json_match else {}
        except:
            vision_data = {"raw": vision_result}
        
        # === 阶段 2: 报告生成 ===
        print("  [2/3] 生成结构化报告...")
        
        report_prompt = f"""基于以下视觉分析结果，生成一份完整的、可直接提交的 Bug 报告。

## 视觉分析结果
{json.dumps(vision_data, indent=2, ensure_ascii=False)}

## 用户原始描述
{user_description if user_description else '(无)'}

## 元信息
- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 应用: {self.project_context['app_name']}

请输出完整的 Bug 报告（Markdown 格式），包含：

---
# [自动生成的 Bug 报告]

## 概述
[一句话总结最关键的问题]

## 发现的缺陷

### 🔴 [缺陷标题] (严重程度: High)
**类别**: Functional/UIUX/Design
**位置**: [具体位置]

**问题描述**: [详细描述]

**复现步骤**:
1. [步骤]
2. [步骤]
3. [观察到的问题]

**预期行为**: [...]
**实际行为**: [...]

**建议修复方案**: [...]

---

（如果有多个缺陷，按严重程度从高到低排列）

## 整体评价
- **UI 健康评分**: X/10
- **主要风险**: [...]
- **改进优先级**: [...]"""
        
        report_md = self._call_text(report_prompt)
        print(f"  ✅ 报告生成完成 ({time.time()-start_time:.1f}s)")
        
        # === 阶段 3: Jira 格式转换 ===
        print("  [3/3] 生成 Jira 格式...")
        
        jira_prompt = f"""将以下 Bug 报告转换为 Jira API 可用的 JSON 格式。

## 原始报告
{report_md}

输出格式（纯 JSON）：
```json
{{
  "fields": {{
    "project": {{"key": "PROJ"}},
    "summary": "[Bug] 最关键的缺陷标题",
    "description": "{{完整描述，包含所有缺陷详情}}",
    "issuetype": {{"name": "Bug"}},
    "priority": {{"name": "High"}},
    "labels": ["ui", "automated-report", "screenshot"],
    "components": [{{"name": "Frontend"}}]
  }}
}}
```"""
        
        jira_json_str = self._call_text(jira_prompt)
        
        # 尝试提取 JSON
        try:
            jira_match = re.search(r'\{[\s\S]*\}', jira_json_str)
            jira_data = json.loads(jira_match.group()) if jira_match else {}
        except:
            jira_data = {"raw": jira_json_str}
        
        total_time = time.time() - start_time
        print(f"  ✅ 全部完成! 总耗时: {total_time:.1f}s\n")
        
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "image_path": str(image_path),
                "user_description": user_description,
                "processing_time": round(total_time, 1),
                "models_used": [self.vision_model, self.text_model]
            },
            "vision_analysis": vision_data,
            "report_markdown": report_md,
            "jira_format": jira_data
        }
    
    def save_report(self, report, output_dir="./bug_reports"):
        """保存报告到文件"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存 Markdown 报告
        md_path = output_dir / f"bug_report_{ts}.md"
        with open(md_path, "w") as f:
            f.write(report["report_markdown"])
        
        # 保存 Jira JSON
        jira_path = output_dir / f"bug_jira_{ts}.json"
        with open(jira_path, "w") as f:
            json.dump(report["jira_format"], f, indent=2, ensure_ascii=False)
        
        # 保存完整数据（包含视觉分析原始结果）
        full_path = output_dir / f"bug_full_{ts}.json"
        with open(full_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📁 报告已保存:")
        print(f"   Markdown: {md_path}")
        print(f"   Jira JSON: {jira_path}")
        print(f"   完整数据: {full_path}")
        
        return str(md_path)


def main():
    """主函数"""
    
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════╗
║         智能 Bug 报告助手 (SmartBug Reporter)              ║
║                                                          ║
║  用法: python3 smart_bug.py <截图路径> [用户描述]          ║
║                                                          ║
║  示例:                                                    ║
║    python3 smart_bug.py ./screenshot.png                  ║
║    python3 smart_bug.py ./screenshot.png "购物车无法结算"  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
        sys.exit(1)
    
    image_path = sys.argv[1]
    user_desc = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
    
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        sys.exit(1)
    
    reporter = SmartBugReporter()
    report = reporter.analyze_screenshot(image_path, user_desc)
    
    print("\n" + "="*70)
    print("📋 生成的 Bug 报告:")
    print("="*70 + "\n")
    print(report["report_markdown"][:2000])
    if len(report["report_markdown"]) > 2000:
        print("\n... (报告较长，已保存到文件)")
    
    reporter.save_report(report)


if __name__ == "__main__":
    main()
```

## 项目二：发票 OCR + 结构化数据提取

### 项目背景

财务部门每天需要处理大量发票和收据——手工录入金额、日期、商家名称等信息既耗时又容易出错。这个项目利用 Ollama 的视觉能力实现**拍照即录入**的自动化流程。

### 核心实现

```python
#!/usr/bin/env python3
"""
发票/收据智能识别与结构化提取系统
支持: 增值税发票 / 电子发票 / 普通收据 / 机打小票
"""

import requests
import base64
import json
import csv
import os
from pathlib import Path
from datetime import datetime


class InvoiceExtractor:
    """发票信息提取器"""
    
    INVOICE_TYPES = {
        "vat_special": "增值税专用发票",
        "vat_normal": "增值税普通发票",
        "vat_electronic": "增值税电子普通发票",
        "receipt": "通用收据",
        "pos_receipt": "POS机打小票",
        "taxi": "出租车票",
        "train": "火车票",
        "airline": "航空行程单",
        "other": "其他票据"
    }
    
    def __init__(self, model="minicpm-v", ollama_url="http://localhost:11434"):
        self.model = model
        self.base_url = ollama_url
    
    def extract(self, image_path, invoice_type_hint=""):
        """
        从发票图片中提取结构化信息
        
        Returns:
            dict: 提取的结构化数据
        """
        
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        
        type_hint = f"\n提示: 这可能是一张{invoice_type_hint}" if invoice_type_hint else ""
        
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f"""这是一张发票/收据/票据的照片。
请仔细阅读图片中的每一个文字，精确提取所有字段信息。

{type_hint}

## 提取规则
1. 数字类信息必须完全准确，不要四舍五入
2. 日期统一为 YYYY-MM-DD 格式
3. 金额保留原始精度（包括小数位数）
4. 如果某个字段在图片中找不到，填 null
5. 对于模糊不清的字段，标注 [不确定]

## 必须输出的 JSON 格式（严格遵守）：
```json
{{
  "document_type": "vat_normal|vat_electronic|receipt|pos_receipt|taxi|train|airline|other",
  "confidence": 0.95,
  
  "basic_info": {{
    "invoice_code": null,
    "invoice_number": null,
    "issue_date": null,
    "check_code": null,
    "machine_number": null
  }},
  
  "seller_info": {{
    "name": null,
    "tax_id": null,
    "address_phone": null,
    "bank_account": null
  }},
  
  "buyer_info": {{
    "name": null,
    "tax_id": null,
    "address_phone": null,
    "bank_account": null
  }},
  
  "items": [
    {{
      "name": "商品或服务名称",
      "specification": "规格型号",
      "unit": "单位",
      "quantity": null,
      "unit_price": null,
      "amount": null,
      "tax_rate": null,
      "tax_amount": null
    }}
  ],
  
  "amounts": {{
    "total_amount": null,
    "total_tax": null,
    "amount_with_tax": null,
    "amount_in_words": null,
    "currency": "CNY"
  }},
  
  "remarks": null,
  "payee": null,
  "reviewer": null,
  "drawer": null,
  
  "raw_ocr_text": "图片中识别出的全部原始文字（用于人工校验）"
}}
```

注意：只输出上面的 JSON 对象，不要添加任何额外文字。"""},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            "stream": False,
            "options": {"temperature": 0.05}  # 极低温度确保准确性
        }
        
        resp = requests.post(f"{self.base_url}/api/chat", 
                            json=payload, timeout=120)
        
        raw_output = resp.json()["message"]["content"]
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if json_match:
                data = json.loads(json_match.group())
                data["extraction_time"] = datetime.now().isoformat()
                data["source_image"] = str(image_path)
                return data
        except Exception as e:
            pass
        
        return {
            "error": "JSON 解析失败",
            "raw_output": raw_output,
            "source_image": str(image_path)
        }
    
    def export_to_csv(self, extracted_data_list, output_path):
        """将多张发票的提取结果导出为 CSV"""
        
        rows = []
        for data in extracted_data_list:
            if "error" in data:
                continue
            
            amounts = data.get("amounts", {})
            basic = data.get("basic_info", {})
            
            rows.append({
                "日期": basic.get("issue_date"),
                "发票号码": basic.get("invoice_number"),
                "销售方": data.get("seller_info", {}).get("name"),
                "购买方": data.get("buyer_info", {}).get("name"),
                "合计金额": amounts.get("total_amount"),
                "合计税额": amounts.get("total_tax"),
                "价税合计": amounts.get("amount_with_tax"),
                "置信度": data.get("confidence"),
                "来源图片": data.get("source_image")
            })
        
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✅ CSV 已导出: {output_path} ({len(rows)} 条记录)")
        return output_path
    
    def batch_extract(self, image_dir, output_csv=None):
        """批量处理一个目录下的所有发票图片"""
        
        image_dir = Path(image_dir)
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".heic"}
        
        images = sorted([
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ])
        
        if not images:
            print(f"❌ 目录中没有找到图片: {image_dir}")
            return []
        
        results = []
        print(f"\n📄 开始批量处理 {len(images)} 张发票...\n")
        
        for i, img in enumerate(images, 1):
            print(f"[{i}/{len(images)}] 处理: {img.name}", end=" ... ", flush=True)
            
            try:
                result = self.extract(str(img))
                
                if "error" not in result:
                    confidence = result.get("confidence", "?")
                    total = result.get("amounts", {}).get("amount_with_tax", "?")
                    print(f"✅ (置信度: {confidence}, 金额: ¥{total})")
                else:
                    print(f"⚠️ 解析失败")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ 错误: {e}")
                results.append({"error": str(e), "source_image": str(img)})
        
        if output_csv and results:
            self.export_to_csv(results, output_csv)
        
        return results


if __name__ == "__main__":
    import sys
    
    extractor = InvoiceExtractor()
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        if Path(path).is_dir():
            results = extractor.batch_extract(
                path, 
                output_csv=f"./invoices_{datetime.now().strftime('%Y%m%d')}.csv"
            )
        else:
            result = extractor.extract(path)
            print(json.dumps(result, indent=2, ensure_ascii=False))
```

## 项目三：论文图表数据提取与 CSV 导出

### 项目背景

研究人员经常需要从论文 PDF 的图表中提取数据来做对比实验或复现结果。手工读取柱状图/折线图的数值既不准确又极其枯燥。这个项目让 LLaVA 自动读取图表并输出结构化数据。

### 实现代码

```python
#!/usr/bin/env python3
"""
论文图表数据提取器
功能: 从学术论文的图表图片中提取数值数据并导出为 CSV
"""

import requests
import base64
import json
import csv
import re
from pathlib import Path


class ChartDataExtractor:
    """图表数据提取器"""
    
    def __init__(self, model="llava", ollama_url="http://localhost:11434"):
        self.model = model
        self.base_url = ollama_url
    
    def extract(self, image_path, chart_type_hint=""):
        """从图表中提取数据"""
        
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        
        hint = f"\n提示: 这可能是{chart_type_hint}" if chart_type_hint else ""
        
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f"""请仔细分析这张学术图表，提取其中所有的数值数据。

{hint}

## 分析步骤
1. 识别图表类型（柱状图/折线图/饼图/散点图/面积图/热力图/表格混合）
2. 读取图表标题和坐标轴标签
3. 逐个数据点读取精确数值（尽可能精确到小数点后一位）
4. 注意区分不同的数据系列（不同颜色/图案的线条或柱子）

## 输出格式

### 图表元信息
```json
{{
  "chart_type": "bar|line|pie|scatter|area|heatmap|table_mixed",
  "title": "图表标题",
  "x_axis_label": "X轴标签",
  "y_axis_label": "Y轴标签",
  "has_legend": true,
  "series_names": ["系列1名称", "系列2名称"]
}}
```

### 数据表格 (CSV 格式)
```
category,series_1,series_2,series_3
Method A,85.2,78.4,91.1
Method B,88.7,82.1,89.3
...
```

### 关键观察
[你对这个图表趋势和数据含义的专业解读]

注意：
- 数值尽量精确，如果无法确定请标注 ~ 符号
- 如果有误差棒(error bar)，也请一并记录
- 百分比数字保持原样，不要转换为小数"""},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }],
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        resp = requests.post(f"{self.base_url}/api/chat", 
                            json=payload, timeout=120)
        return resp.json()["message"]["content"]
    
    def extract_and_export(self, image_path, output_csv=None, 
                           chart_type_hint=""):
        """提取数据并可选地导出为 CSV"""
        
        result = self.extract(image_path, chart_type_hint)
        print(result)
        
        if output_csv:
            csv_match = re.search(
                r'```\s*\n(.*?)(?:\n```|\Z)', 
                result, 
                re.DOTALL
            )
            
            if csv_match:
                csv_content = csv_match.group(1).strip()
                if not csv_content.startswith("category"):
                    lines = csv_content.split("\n")
                    if lines and not lines[0].startswith("category"):
                        csv_content = "category,value\n" + csv_content
                
                with open(output_csv, "w", newline="") as f:
                    f.write(csv_content)
                print(f"\n✅ CSV 已保存: {output_csv}")
        
        return result
    
    def batch_extract_from_pdf_pages(self, page_images_dir, output_dir="./chart_data"):
        """批量处理 PDF 页面截图中的所有图表"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        page_images = sorted(Path(page_images_dir).glob("*.png"))
        
        all_results = []
        
        for i, img in enumerate(page_images, 1):
            print(f"\n{'='*60}")
            print(f"📊 处理页面 {i}: {img.name}")
            print(f"{'='*60}")
            
            result = self.extract_and_export(
                str(img),
                output_csv=str(output_dir / f"chart_page_{i:03d}.csv")
            )
            
            all_results.append({
                "page": i,
                "source": str(img.name),
                "result": result
            })
        
        return all_results


if __name__ == "__main__":
    import sys
    
    extractor = ChartDataExtractor()
    
    if len(sys.argv) < 2:
        print("用法: python3 chart_extractor.py <图片路径> [输出CSV] [图表类型提示]")
        sys.exit(1)
    
    img_path = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else None
    hint = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
    
    extractor.extract_and_export(img_path, out_csv, hint)
```

## 多模态项目的性能优化总结

| 优化维度 | 具体措施 | 效果 |
|---------|---------|------|
| **输入预处理** | 图片缩放到 800px 以下 | 减少 50%+ 的 token 数 |
| **模型选择** | minicpm-v 替代 llava (轻量场景) | 内存减少 60%，速度提升 2x |
| **温度设置** | OCR 类任务用 0.05-0.1 | 大幅提高准确性 |
| **批处理策略** | 串行处理 + 进度条 | 避免内存溢出 |
| **缓存机制** | 相同图片跳过重复分析 | 节省 30%+ 时间 |
| **Prompt 工程** | 结构化 JSON 输出格式 | 后续处理零成本 |

## 本章小结

这一节通过三个完整的多模态实战项目巩固了前两节的知识：

1. **智能截图 Bug 助手**：截图 + 视觉分析 + 文本综合 → 结构化 Bug 报告（支持 Markdown 和 Jira 格式）
2. **发票 OCR 系统**：拍照 → 发票类型识别 → 全字段结构化提取 → 批量 CSV 导出
3. **图表数据提取器**：论文图表 → 类型识别 + 数值读取 → CSV 导出 + 趋势分析
4. 三个项目都遵循了**"视觉模型做感知 + 文本模型做推理"的两阶段架构**
5. **极低 temperature（0.05-0.1）**是 OCR/数据提取类任务的关键参数设置

至此，第五章"多模态能力"全部完成。下一章我们将进入 RAG 领域——Embedding 模型与向量检索增强生成。
