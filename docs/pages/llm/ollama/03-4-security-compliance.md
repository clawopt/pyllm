# 03-4 模型安全性与合规

## 为什么安全和合规不容忽视

当你把一个开源大模型部署到公司内网、或者用它来处理客户数据、或者把它嵌入到面向公众的产品中时，"能跑起来"只是第一步。真正决定这个项目能不能长期稳定运行的，是**安全性**和**合规性**这两个经常被开发者忽视的维度。一个模型可能技术上完美无缺，但如果它的许可证禁止商业使用，或者它会在输出中泄露敏感信息，或者它的训练数据来源不明存在投毒风险——这些问题一旦爆发，后果远比一个 bug 严重得多。

这一节将从**许可证合规、内容安全、供应链安全和企业级考量**四个层面，系统地讨论如何在使用 Ollama 模型时保障安全和合规。

## 许可证全景图：你能用这个模型做什么

### 开源大模型的主要许可证类型

Ollama 库中的模型来自不同的发布组织，每个组织选择的许可证各不相同。理解这些许可证的含义，是合规使用的第一道防线：

```
┌───────────────────────────────────────────────────────────────┐
│                 大模型许可证严格程度谱系                         │
│                                                               │
│  最宽松 ──────────────────────────────────────── 最严格       │
│                                                               │
│  MIT / Apache 2.0                                            │
│  │  几乎无限制，商用/修改/分发均可                              │
│  │  适用: Gemma (Apache 2.0), Phi-3 (MIT)                    │
│  │                                                           │
│  │                                                            │
│  Llama Community License                                     │
│  │  商用有条件: 月活用户<7亿且年收入不限制(3.1版起)            │
│  │  适用: Llama 3.1 系列                                      │
│  │                                                           │
│  │                                                            │
│  Apache 2.0 (主流选择)                                        │
│  │  完全开放，商用友好                                       │
│  │  适用: Qwen2.5, Mistral, DeepSeek, Yi, InternLM          │
│  │                                                           │
│  │                                                            │
│  CC-BY-SA / 研究许可                                         │
│  │  仅限研究用途，禁止商用                                    │
│  │  适用: 部分学术模型、早期实验模型                          │
│  │                                                           │
│  ▼                                                            │
│  专有 / 自定义许可                                           │
│     需要单独协商授权                                          │
│     适用: 部分商业公司的"开源"模型                            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 各大模型的详细许可证信息

| 模型系列 | 许可证 | 商用 | 修改 | 再分发 | 特殊限制 |
|---------|--------|------|------|--------|---------|
| **Llama 3.1** | Llama Community License | ✅ 有条件 | ✅ | ✅ | 月活>7亿需申请 |
| **Qwen2.5** | Apache 2.0 | ✅ 无限制 | ✅ | ✅ | 无 |
| **Mistral** | Apache 2.0 | ✅ 无限制 | ✅ | ✅ | 无 |
| **DeepSeek-V2** | MIT | ✅ 无限制 | ✅ | ✅ | 无 |
| **Gemma 2** | Apache 2.0 | ✅ 无限制 | ✅ | ✅ | 无 |
| **Phi-3** | MIT | ✅ 无限制 | ✅ | ✅ | 无 |
| **Yi** | Apache 2.0 | ✅ 无限制 | ✅ | ✅ | 无 |
| **CodeLlama** | Llama Community License | ✅ 有条件 | ✅ | ✅ | 同 Llama |
| **StarCoder2** | BigCode Open Model | ✅ 有条件 | ✅ | ✅ | <500M MAU |

### 用代码自动检查许可证

```python
#!/usr/bin/env python3
"""Ollama 模型许可证检查器"""

import requests
import json

def check_model_license(model_name):
    """检查指定模型的许可证信息"""
    
    # 方法一：通过 Ollama 本地 API（如果已下载）
    try:
        resp = requests.post("http://localhost:11434/api/show", json={
            "name": model_name
        }, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            license_info = data.get("license", "未知")
            modelfile = data.get("modelfile", "")
            
            print(f"\n📋 模型: {model_name}")
            print(f"   许可证: {license_info}")
            
            # 从 modelfile 中提取更多信息
            for line in modelfile.split("\n"):
                if line.startswith("LICENSE"):
                    print(f"   详细: {line}")
                elif line.startswith("FROM"):
                    print(f"   基础: {line}")
            
            return license_info
    except requests.ConnectionError:
        print("[WARN] Ollama 服务未运行，尝试在线查询...")
    
    # 方法二：通过 Ollama 在线 API
    try:
        resp = requests.get(
            f"https://ollama.com/api/models/{model_name}",
            timeout=15
        )
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n📋 模型: {data['name']}")
            print(f"   许可证: {data.get('license', '未知')}")
            print(f"   参数量: {', '.join(data.get('parameter_sizes', []))}")
            print(f"   上下文窗口: {data.get('context_window', '?')}")
            return data.get("license", "未知")
        else:
            print(f"❌ 模型 '{model_name}' 未找到")
            return None
            
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return None

def compliance_check(model_name, use_case="commercial"):
    """根据使用场景进行合规检查"""
    
    license_map = {
        "apache 2.0": {
            "commercial": "✅ 完全允许",
            "internal": "✅ 完全允许",
            "saas": "✅ 允许",
            "notes": "无需特殊处理"
        },
        "mit": {
            "commercial": "✅ 完全允许",
            "internal": "✅ 完全允许",
            "saas": "✅ 允许",
            "notes": "最宽松的许可证之一"
        },
        "llama community license": {
            "commercial": "⚠️ 有条件 - 月活<7亿用户",
            "internal": "✅ 内部使用无限制",
            "saas": "⚠️ 需确认用户规模",
            "notes": "Llama 3.1 起取消了收入限制，但保留用户数门槛"
        },
        "bigcode open model agreement": {
            "commercial": "⚠️ <5亿月活",
            "internal": "✅ 内部使用无限制",
            "saas": "⚠️ 需确认",
            "notes": "StarCoder2 使用此许可"
        },
        "cc-by-nc": {
            "commercial": "❌ 禁止商用",
            "internal": "⚠️ 需法律确认",
            "saas": "❌ 禁止",
            "notes": "仅限非商业用途"
        }
    }
    
    license_name = check_model_license(model_name)
    
    if not license_name:
        return
    
    license_lower = license_name.lower()
    
    # 模糊匹配许可证类型
    matched_key = None
    for key in license_map:
        if key in license_lower:
            matched_key = key
            break
    
    if matched_key:
        info = license_map[matched_key]
        print(f"\n🔍 合规检查 ({use_case} 用途):")
        print(f"   结果: {info.get(use_case, '⚠️ 未知场景')}")
        print(f"   备注: {info['notes']}")
    else:
        print(f"\n⚠️ 未能自动识别许可证类型: {license_name}")
        print(f"   建议手动查阅原始许可证文本")

# 批量检查团队常用模型
TEAM_MODELS = [
    "qwen2.5:7b",      # 中文主力
    "llama3.1:8b",     # 英文主力
    "deepseek-coder:6.7b",  # 代码
    "nomic-embed-text",     # Embedding
    "phi3:mini",       # 轻量测试
]

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model = sys.argv[1]
        use_case = sys.argv[2] if len(sys.argv) > 2 else "commercial"
        compliance_check(model, use_case)
    else:
        print("=" * 60)
        print("  团队模型合规批量检查")
        print("=" * 60)
        for m in TEAM_MODELS:
            compliance_check(m, "commercial")
```

运行结果示例：

```
======================================================================
  团队模型合规批量检查
======================================================================

📋 模型: qwen2.5:7b
   许可证: Apache 2.0

🔍 合规检查 (commercial 用途):
   结果: ✅ 完全允许
   备注: 无需特殊处理

📋 模型: llama3.1:8b
   许可证: llama community license

🔍 合规检查 (commercial 用途):
   结果: ⚠️ 有条件 - 月活<7亿用户
   备注: Llama 3.1 起取消了收入限制，但保留用户数门槛
```

## 内容安全：模型输出的风险控制

### 模型的内置安全对齐程度

不同模型在训练阶段的安全对齐（Safety Alignment）程度差异很大：

| 安全维度 | 高对齐 | 中等对齐 | 低对齐 |
|---------|--------|---------|--------|
| 拒绝有害请求 | Llama 3.1, Qwen2.5 | Mistral, Gemma | 部分社区微调版 |
| 拒绝 PII 泄露 | Llama 3.1 + Guard | Qwen2.5 | 基础模型 |
| 拒绝生成恶意代码 | DeepSeek-Coder | 大部分代码模型 | 无过滤版本 |
| 偏见与公平性 | Llama 3.1 | Qwen2.5 | 早期模型 |

**关键认知**：没有绝对安全的模型。即使是最强安全对齐的 Llama 3.1，也可以通过精心设计的 prompt injection 绕过其安全防护。因此企业级应用需要**多层防御**。

### 内容安全防护架构

```python
#!/usr/bin/env python3
"""Ollama 输出内容安全过滤器"""

import re
import requests

class ContentSafetyFilter:
    """多层内容安全过滤器"""
    
    def __init__(self):
        # 敏感词模式库（实际项目中应从外部配置加载）
        self.patterns = {
            "pii": [
                r'\b\d{3}-\d{2}-\d{4}\b',           # SSN 格式
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b1[3-9]\d{9}\b',                  # 中国手机号
                r'\b\d{16,19}\b',                   # 银行卡号
            ],
            "sensitive_data": [
                r'(?:api[_-]?key|apikey)["\s]*[:=]["\']*[a-zA-Z0-9_-]{20,}',
                r'(?:password|passwd|pwd)["\s]*[:=]["\'][^"\']+',
                r'(?:secret|token)["\s]*[:=]["\'][a-zA-Z0-9._/-]{20,}',
            ],
            "harmful_content": [
                r'(?:如何制造|制作教程).*(?:炸弹|毒药|毒品)',
                r'(?:绕过|破解|hack).*(?:认证|密码|防火墙)',
            ]
        }
    
    def scan(self, text):
        """扫描文本中的安全问题"""
        issues = []
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    issues.append({
                        "category": category,
                        "pattern": pattern[:30] + "...",
                        "matches_count": len(matches),
                        "severity": self._get_severity(category)
                    })
        
        return issues
    
    def _get_severity(self, category):
        """根据类别返回严重级别"""
        severity_map = {
            "pii": "high",
            "sensitive_data": "critical",
            "harmful_content": "high"
        }
        return severity_map.get(category, "medium")
    
    def sanitize(self, text, mask_char="***"):
        """脱敏处理（替换敏感信息）"""
        result = text
        
        for category, patterns in self.patterns.items():
            if category in ["pii", "sensitive_data"]:
                for pattern in patterns:
                    result = re.sub(pattern, mask_char, result, flags=re.IGNORECASE)
        
        return result


class SafeOllamaClient:
    """带安全过滤的 Ollama 客户端"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.safety_filter = ContentSafetyFilter()
        self.audit_log = []
    
    def chat(self, model, messages, options=None, enable_filter=True):
        """带安全过滤的对话接口"""
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {}
        }
        
        # 记录审计日志
        log_entry = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "model": model,
            "input_length": sum(len(m.get("content", "")) for m in messages),
        }
        
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        data = resp.json()
        content = data["message"]["content"]
        
        # 安全扫描
        if enable_filter:
            issues = self.safety_filter.scan(content)
            
            if issues:
                critical_issues = [i for i in issues if i["severity"] == "critical"]
                
                if critical_issues:
                    # 严重问题：阻止输出并返回警告
                    sanitized = self.safety_filter.sanitize(content)
                    
                    log_entry.update({
                        "status": "blocked",
                        "issues": issues,
                        "output_sanitized": True
                    })
                    self.audit_log.append(log_entry)
                    
                    return {
                        "content": "[⚠️ 输出包含敏感信息已被过滤]",
                        "safety_issues": issues,
                        "sanitized_content": sanitized
                    })
                else:
                    # 警告级别：放行但记录
                    log_entry.update({
                        "status": "warning",
                        "issues": issues
                    })
                    content = self.safety_filter.sanitize(content)
            else:
                log_entry["status"] = "clean"
        
        log_entry["output_length"] = len(content)
        self.audit_log.append(log_entry)
        
        return {"content": content}
    
    def get_audit_report(self):
        """获取安全审计报告"""
        total = len(self.audit_log)
        blocked = sum(1 for l in self.audit_log if l.get("status") == "blocked")
        warnings = sum(1 for l in self.audit_log if l.get("status") == "warning")
        clean = sum(1 for l in self.audit_log if l.get("status") == "clean")
        
        report = {
            "total_requests": total,
            "clean": clean,
            "warnings": warnings,
            "blocked": blocked,
            "block_rate": round(blocked / total * 100, 1) if total > 0 else 0,
            "recent_issues": [l for l in self.audit_log[-10:] 
                             if l.get("status") in ("blocked", "warning")]
        }
        
        return report


# 使用示例
if __name__ == "__main__":
    client = SafeOllamaClient()
    
    # 正常请求
    result = client.chat(
        "qwen2.5:7b",
        [{"role": "user", "content": "解释什么是机器学习"}]
    )
    print("正常回答:", result["content"][:100])
    
    # 包含模拟敏感信息的请求
    sensitive_result = client.chat(
        "qwen2.5:7b",
        [{"role": "user", "content": 
          "用户的邮箱是 test@example.com，手机号是 13812345678，请帮我分析"}],
        enable_filter=True
    )
    print("\n安全检查:", sensitive_result.get("safety_issues", "无"))
    
    # 获取审计报告
    report = client.get_audit_report()
    print(f"\n📊 安全审计:")
    print(f"   总请求数: {report['total_requests']}")
    print(f"   清洁: {report['clean']} | 警告: {report['warnings']} | 拦截: {report['blocked']}")
```

这个安全过滤器实现了三层防护：

1. **输入层**：虽然上面的例子主要展示了输出过滤，但在生产环境中你也应该过滤输入（防止 prompt injection）
2. **输出层**：正则表达式匹配 PII（个人身份信息）、API 密钥、有害内容
3. **审计层**：所有请求和检测结果都记录在审计日志中，便于事后追溯

## 供应链安全：模型来源的可信度评估

### 投毒风险：为什么不能随便用 GGUF 文件

GGUF 格式的模型文件本质上是二进制的权重矩阵加上一些元数据。如果有人在权重中植入后门——比如让模型在遇到特定触发词时输出恶意内容——这种篡改是极难通过肉眼或自动化工具检测出来的。

```python
#!/usr/bin/env python3
"""Ollama 模型完整性校验工具"""

import hashlib
import os
import json
import requests
from pathlib import Path

OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"

def compute_file_hash(filepath, algorithm="sha256"):
    """计算文件的哈希值"""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_model_integrity(model_name):
    """验证本地模型的文件完整性"""
    
    manifests_dir = OLLAMA_MODELS_DIR / "manifests" / "registry.ollama.ai" / "library"
    blobs_dir = OLLAMA_MODELS_DIR / "blobs"
    
    # 找到模型的 manifest 文件
    model_path_part = model_name.replace("/", os.sep)
    manifest_files = list(manifests_dir.rglob(f"{model_path_part}*.json"))
    
    if not manifest_files:
        print(f"❌ 未找到模型 '{model_name}' 的 manifest")
        return False
    
    manifest_file = manifest_files[0]
    print(f"\n🔍 校验模型: {model_name}")
    print(f"   Manifest: {manifest_file}")
    
    try:
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Manifest JSON 解析失败: {e}")
        return False
    
    # 检查 blob 引用的完整性
    all_layers = manifest.get("layers", [])
    total_size = 0
    verified = 0
    corrupted = 0
    missing = 0
    
    print(f"   共 {len(all_layers)} 个层需要校验...")
    
    for i, layer in enumerate(all_layers, 1):
        digest = layer.get("digest", "")
        size = layer.get("size", 0)
        total_size += size
        
        if not digest.startswith("sha256-"):
            continue
        
        expected_hash = digest.replace("sha256-", "")
        blob_path = blobs_dir / digest.replace(":", "-")
        
        if not blob_path.exists():
            missing += 1
            print(f"   ⚠️ 层 {i}: 文件缺失 ({blob_path.name})")
            continue
        
        actual_hash = compute_file_hash(blob_path)
        
        if actual_hash == expected_hash:
            verified += 1
        else:
            corrupted += 1
            print(f"   ❌ 层 {i}: 哈希不匹配!")
            print(f"      期望: {expected_hash[:32]}...")
            print(f"      实际: {actual_hash[:32]}...")
    
    print(f"\n   📊 校验结果:")
    print(f"   总大小: {total_size / (1024**3):.2f} GB")
    print(f"   ✅ 通过: {verified} | ❌ 损坏: {corrupted} | ⚠️ 缺失: {missing}")
    
    if corrupted > 0 or missing > 0:
        print(f"\n   🚨 发现完整性问题！建议重新下载该模型:")
        print(f"      ollama rm {model_name}")
        print(f"      ollama pull {model_name}")
        return False
    
    print(f"   ✅ 模型完整性验证通过")
    return True

def check_source_trust(model_name):
    """评估模型来源的可信度"""
    
    trust_indicators = {
        "official_ollama": {
            "score": 10,
            "check": lambda n: "/" not in n,
            "desc": "Ollama 官方库收录"
        },
        "known_publisher": {
            "score": 8,
            "check": lambda n: any(p in n for p in ["bartowski", "MaziyarPanahi"]),
            "desc": "知名社区贡献者"
        },
        "huggingface_verified": {
            "score": 6,
            "check": lambda n: "huggingface" in n,
            "desc": "HuggingFace 来源"
        },
        "unknown": {
            "score": 2,
            "check": lambda n: True,
            "desc": "未知来源，请谨慎使用"
        }
    }
    
    score = 0
    reasons = []
    
    for indicator, info in trust_indicators.items():
        if info["check"](model_name):
            score = max(score, info["score"])
            reasons.append(info["desc"])
            break
    
    print(f"\n🔐 来源可信度评估:")
    print(f"   模型: {model_name}")
    print(f"   可信度评分: {score}/10")
    print(f"   依据: {', '.join(reasons)}")
    
    if score < 5:
        print(f"   ⚠️ 建议: 此模型来源可信度较低，建议优先使用官方版本")
    
    return score

if __name__ == "__main__":
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    
    # 双重检查：来源可信度 + 文件完整性
    check_source_trust(model)
    verify_model_integrity(model)
```

### 可信度最佳实践

1. **优先使用 Ollama 官方库**：`ollama/library` 下的模型经过官方审核和格式化
2. **知名社区贡献者可信度较高**：如 `bartowski`、`MaziyarPanahi` 等长期维护者
3. **自定义导入时必须校验哈希**：从 HuggingFace 或其他渠道获取 GGUF 时，务必对比 SHA256
4. **定期更新到最新版本**：安全补丁通常随模型更新一起发布
5. **隔离运行环境**：对于来源不确定的模型，先在容器或虚拟机中测试

## 企业级合规考量

### 金融行业

金融机构使用 AI 模型面临特殊的监管要求：

- **模型可解释性**：监管机构可能要求解释模型为何做出某个决策（这对黑盒 LLM 来说是个挑战）
- **数据留存**：所有输入输出可能需要保存一定年限以备审计
- **模型审批流程**：新模型上线前需要经过内部风控和合规部门审批
- **输出人工复核**：涉及资金操作的决策不能完全依赖模型输出

### 医疗健康行业

医疗场景下的额外要求：

- **HIPAA / GDPR 合规**：患者数据绝不能出现在训练数据或模型输出中
- **诊断辅助而非诊断替代**：模型的定位必须是辅助工具，最终决策权在医生
- **偏差检测**：需要定期检测模型在不同人群上的表现是否存在系统性偏差

### 政府与公共部门

政府机构的特殊约束：

- **数据主权**：数据不能出境，模型必须在本地运行（这正是 Ollama 的核心价值主张之一）
- **供应链审查**：模型的训练数据来源、开发团队背景都需要审查
- **国产化要求**：某些场景下可能要求使用国产模型（如 Qwen、InternLM 等）

## 本章小结

这一节我们从多个维度讨论了模型安全性和合规问题：

1. **许可证类型决定了你能否合法地商用**一个模型——Apache 2.0/MIT 最宽松，Llama Community License 有条件允许，CC-by-NC 禁止商用
2. **内容安全需要多层防护**：输入过滤 + 输出过滤 + 审计日志，单一层防护是不够的
3. **供应链安全常被忽视但至关重要**：只从可信源获取模型、校验 SHA256 哈希、隔离测试未知模型
4. **不同行业有不同的合规要求**：金融重可解释性、医疗重隐私保护、政府重数据主权
5. **工具链支持**：我们提供了许可证检查器、内容安全过滤器、完整性校验三套实用工具

至此，第三章"模型生态与选择指南"全部完成。下一章我们将进入 Ollama 最具特色的功能领域——Modelfile 与自定义模型。
