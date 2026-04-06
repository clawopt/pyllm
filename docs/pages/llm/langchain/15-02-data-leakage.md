---
title: 敏感数据泄露风险
description: LLM 应用的数据泄露面分析、PII 检测与脱敏、日志安全、知识库数据隔离、API Key 管理最佳实践
---
# 敏感数据泄露风险

Prompt Injection 关注的是"攻击者让 LLM 做坏事"，而**数据泄露**关注的是"LLM 不小心把不该说的信息说了出去"。两者经常相伴发生——注入攻击往往是数据泄露的入口——但即使没有恶意攻击，LLM 应用本身也可能因为设计不当而泄露敏感信息。

## LLM 应用中的数据泄露面

一个典型的 LangChain 应用在运行过程中会接触（或可能暴露）以下几类敏感数据：

| 数据类别 | 示例 | 泄露后果 | 风险等级 |
|---------|------|---------|---------|
| **凭证类** | API Key、数据库密码、JWT Secret | 账户被接管、资源被盗用 | 🔴 致命 |
| **个人身份信息 (PII)** | 姓名、手机号、身份证、邮箱 | 隐私侵犯、诈骗 | 🔴 高 |
| **业务机密** | 内部策略、未发布功能、客户名单 | 竞争劣势 | 🟠 中高 |
| **系统内部信息** | System Prompt、工具列表、架构细节 | 攻击面暴露 | 🟡 中 |
| **对话内容** | 用户的历史消息 | 隐私泄露 | 🟡 低-中 |

### 数据流向图：敏感信息在哪里可能泄露

```
用户输入 → [输入层] → LLM → [输出层] → 用户看到回复
                ↑              ↑           ↑
           PII 可能藏在这里   Prompt 含密钥  回复含内部信息
                
外部系统 ← [Agent 工具层]
     ↑
  API Key / DB 密码可能在工具调用中暴露
```

每一层都是潜在的泄露点。下面我们逐一分析。

## 泄露场景一：System Prompt 和上下文泄露

这是最常见也最容易被忽视的泄露方式。

### 场景 A：无意中输出内部指令

```
用户: "你是怎么工作的？"

AI (无防御): "我是基于 CloudDesk 产品文档构建的 RAG 系统。
我的工作流程是：
1. 接收你的问题
2. 从向量数据库检索相关文档
3. 使用 OpenAI GPT-4o-mini 生成回答
4. 我的 System Prompt 包含了完整的客服规则..."

→ 泄露了: 技术栈、模型名称、System Prompt 存在
```

### 场景 B：RAG 上下文中的隐藏信息

```python
# 知识库文档中不小心包含了敏感信息
doc_with_secrets = Document(
    page_content="""
# API 配置说明

生产环境 API 地址: https://api.internal.company.com/v2
管理员账号: admin@company.com
数据库连接串: postgresql://root:SuperSecret123@db.internal:5432/app
...
""",
    metadata={"source": "internal-config.md"},
)

# 当这条文档被检索到并作为上下文传给 LLM 时...
result = rag_chain.invoke({"question": "API 怎么配置？"})
# 回答中很可能包含上述连接字符串和密码
```

### 防御：上下文清洗

```python
class ContextSanitizer:
    """对 RAG 检索到的上下文进行脱敏处理"""

    PII_PATTERNS = [
        # 凭证/密钥
        (r'(?:https?://)?[\w.-]+\.[\w.-]+.*?(password|passwd|secret|token|key)\s*[:=]\s*\S+', '[REDACTED_CREDENTIAL]'),
        (r'postgres(?:ql)?://[^\s]+:[^\s]+@', '[REDACTED_DB_CONN]'),
        
        # 个人身份信息
        (r'\b1[3-9]\d{9}\b', '[REDACTED_PHONE]'),
        (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]'),
        (r'身份证\s*[号码]*[:：]\s*\d{17}[\dXx]', '[REDACTED_ID_CARD]'),
        (r'(?:bank|card)\s*(?:number|no)[:：]\s*\d{12,19}', '[REDACTED_BANK_CARD]'),
        
        # 内部地址/IP
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)\b', '[REDACTED_IP]'),
        (r'(?<=@)[\w.-]+\.(?:internal|corp|lan)', '[REDACTED_DOMAIN]'),
    ]

    def sanitize_context(self, context: str) -> str:
        cleaned = context
        redactions = []

        for pattern, replacement in self.PII_PATTERNS:
            matches = re.findall(pattern, cleaned)
            for match in matches:
                redactions.append(match)
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        if redactions:
            print(f"[ContextSanitizer] 已替换 {len(redactions)} 处敏感信息")

        return cleaned

    def sanitize_documents(self, documents: list) -> list:
        for doc in documents:
            doc.page_content = self.sanitize_context(doc.page_content)
            if "metadata" in doc.metadata:
                doc.metadata.pop("file_path", None)
        return documents


sanitizer = ContextSanitizer()
cleaned_docs = sanitizer.sanitize_documents(retrieved_docs)
```

## 泄露场景二：对话历史中的 PII

用户在对话中自然地提供了个人信息：

```
用户: "我叫张三，手机号 13800138000，邮箱 zhangsan@qq.com，
      我上个月买了你们的专业版，订单号 CS-20241101，
      我想申请退款，因为付款时扣了两次钱。"
```

这些信息会被存储在会话记忆中。如果记忆管理不当，可能导致：

1. **跨会话泄露**：A 用户的对话历史被展示给 B 用户
2. **日志泄露**：完整对话记录（含 PII）写入日志文件
3. **训练数据污染**：对话记录被用于模型微调，PII 进入模型权重

### 防御：PII 实时检测与匿名化

```python
import re
from typing import Optional
from dataclasses import dataclass

@dataclass
class PIIDetectionResult:
    contains_pii: bool
    pii_types: list[str]
    anonymized_text: str
    original_mask: list[tuple[int, int]]  # (start, end) of each PII

class PIIDetector:
    def __init__(self):
        self.patterns = {
            "phone_cn": r'1[3-9]\d{9}',
            "phone_intl": r'\+?\d[\d\s.-]{6,14}\d',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "id_card": r'\d{17}[\dXx]',
            "bank_card": r'\d{13,19}[^\d]',
            "name_zh": r'[\u4e00-\u9fff]{2,4}(?:先生|女士|同学|经理|总)',
            "order_id": r'[A-Z]{2}-\d{4,10}',
            "address": r'[\u4e00-\u9fff]{2,}(?:省|市|区|路|号|栋|室|单元)',
        }

    def detect(self, text: str) -> PIIDetectionResult:
        found_types = []
        masks = []

        for ptype, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                found_types.append(ptype)
                masks.append((match.start(), match.end()))

        if not found_types:
            return PIIDetectionResult(
                contains_pii=False,
                pii_types=[],
                anonymized_text=text,
                original_mask=[],
            )

        anonymized = self._anonymize(text, masks)

        return PIIDetectionResult(
            contains_pii=True,
            pii_types=found_types,
            anonymized_text=anonymized,
            original_mask=masks,
        )

    def _anonymize(self, text: str, masks: list) -> str:
        result = text
        offset = 0
        replacements = {
            "phone": ["张***", "李***", "王***", "138****0000"],
            "email": ["z****@qq.com", "l***@163.com"],
            "name": ["张**", "李**", "王**"],
            "id_card": ["*************************"],
            "order_id": ["CS-********"],
        }

        for start, end in sorted(masks):
            original = text[start:end]
            category = self._guess_category(original)
            placeholder = replacements.get(category, "***")
            result = result[:start + offset] + placeholder + result[end + offset:]
            offset += len(placeholder) - (end - start)

        return result

    def _guess_category(self, text: str) -> str:
        if '@' in text: return "email"
        if re.match(r'^1\d', text): return "phone"
        if re.match(r'^[A-Z]', text): return "order_id"
        if len(text) >= 15 and text.isdigit(): return "id_card"
        return "name"


detector = PIIDetector()
result = detector.detect("我叫张三，手机 13800138000，邮箱 zhangsan@qq.com")
print(f"包含 PII: {result.contains_pii}")
print(f"类型: {result.pii_types}")
print(f"匿名化后: {result.anonymized_text}")
# 输出: 我叫张**，手机 138****0000，邮箱 z****@qq.com
```

## 泄露场景三：日志中的敏感信息

这是最容易被忽略但影响最大的泄露渠道。很多开发者习惯在代码中写 `print()` 或 `logger.info()` 来调试问题，却忘了这些日志可能会包含：

```python
# ❌ 危险的日志记录
logger.info(f"User {user.email} logged in with token {user.jwt_token}")
logger.debug(f"DB connection: postgresql://admin:SuperSecret123@...")
logger.info(f"Full request: {request.json()}")  # 可能含 PII

# ✅ 安全的日志记录
logger.info(f"User {user.id} logged in successfully")
logger.debug(f"DB connection established (pool_size={pool.size()})")
logger.info(f"Request received: path={request.url.path}, method={request.method}, "
             f"user_id={user.id}, session={session_id[:8]}...")
```

### 日志安全最佳实践

```python
import logging
from functools import wraps

class SecureLogger:
    def __init__(self):
        self.logger = logging.getLogger("secure_api")
        self.sensitive_fields = [
            'password', 'secret', 'token', 'key', 'credential',
            'api_key', 'apikey', 'authorization',
            'ssn', 'credit_card', 'account_number',
        ]

    @wraps(logging.Logger.info)
    def safe_info(self, msg, *args, **kwargs):
        return self._log_safe('info', msg, args, kwargs)

    @wraps(logging.Logger.debug)
    def safe_debug(self, msg, *args, **kwargs):
        return self._log_safe('debug', msg, args, kwargs)

    def _log_safe(self, level, msg, args, kwargs):
        extra = kwargs.get('extra', {})
        if isinstance(msg, str):
            msg = self._scrub_message(msg)
        formatted = self.logger.makeRecord(
            self.logger.name, level, '', 0,
            ('',), (), extra, None
        )
        formatted.msg = msg
        return getattr(self.logger, level)(formatted)

    def _scrub_message(self, message: str) -> str:
        import json
        lower_msg = message.lower()

        for field in self.sensitive_fields:
            pattern = rf'{field}["^"]?\s*[:=]\s*\S+'
            if re.search(pattern, lower_msg):
                message = re.sub(pattern, f'{field}=[REDACTED]', message, flags=re.IGNORECASE)

        try:
            parsed = json.loads(message) if message.startswith('{') else None
            if parsed:
                message = json.dumps(self._scrub_dict(parsed))
        except (json.JSONDecodeError, TypeError):
            pass

        return message

    def _scrub_dict(self, d: dict) -> dict:
        for key in list(d.keys()):
            key_lower = key.lower()
            for field in self.sensitive_fields:
                if field in key_lower:
                    d[key] = '[REDACTED]'
                    break
        return d
```

## 泄露场景四：API Key 管理不当

API Key 是 LLM 应用中最常见也是最危险的泄露对象。

### 常见的 API Key 泄露途径

| 途径 | 示例 | 预防措施 |
|------|------|---------|
| **代码仓库** | `git push` 时提交了 `.env` 文件 | 使用 `.gitignore` + pre-commit hook + 扫描已提交历史 |
| **前端代码** | JS bundle 中硬编码了 API Key | 后端代理所有 API 调用，Key 不离开服务器 |
| **日志输出** | 错误堆栈中打印了环境变量 | 日志中不记录敏感字段 |
| **错误响应** | 异常时将 env 信息返回给客户端 | 统一错误处理器过滤敏感字段 |
| **浏览器 DevTools** | Network 标签页可见请求头 | 服务端不通过 header 传递 Key |

### API Key 安全管理方案

```python
import os
import hashlib
import hmac
import time
from typing import Optional

class APIKeyManager:
    def __init__(self):
        self.master_key = os.getenv("ENCRYPTION_KEY", os.urandom(32).hex())

    def encrypt_key(self, plaintext_key: str) -> str:
        timestamp = str(int(time.time()))
        sig = hmac.new(
            self.master_key.encode(),
            f"{timestamp}:{plaintext_key}".encode(),
            hashlib.sha256
        ).hexdigest()
        return f"v1.{timestamp}.{sig}.{plaintext_key}"

    def decrypt_key(self, encrypted_key: str) -> Optional[str]:
        parts = encrypted_key.split('.')
        if len(parts) != 3 or parts[0] != "v1":
            return None

        timestamp_str, signature, key_material = parts
        age = time.time() - float(timestamp_str)

        if age > 86400 * 30:  # 30天过期
            return None

        expected_sig = hmac.new(
            self.master_key.encode(),
            f"{timestamp_str}:{key_material}".encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            return None

        return key_material

    def create_limited_key(self, user_id: str, permissions: list[str]) -> str:
        raw_key = os.urandom(32).hex()
        payload = f"{user_id}:{','.join(permissions)}:{raw_key}"
        return self.encrypt_key(payload)


key_manager = APIKeyManager()

# 在配置中使用加密后的 key
encrypted_openai_key = key_manager.encrypt_key(os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = encrypted_openai_key

# 在使用前解密
def get_decrypted_llm():
    encrypted = os.environ.get("OPENAI_API_KEY", "")
    decrypted = key_manager.decrypt_key(encrypted)
    if decrypted is None:
        raise ValueError("API Key 无效或已过期")
    
    actual_key = decrypted.split(':')[-1]
    return ChatOpenAI(api_key=actual_key)
```

### Git 安全检查

```bash
# 安装 truffleHog — 自动扫描 git 历史中的密钥
pip install trufflehog

# 扫描当前仓库
trufflehog --regex --only-verified .

# 扫描特定目录
trufflehog . --exclude-pattern "*.md" --exclude-pattern "*.lock"

# 如果发现泄露的 key
# 1. 立即撤销该 key
# 2. 在对应平台重新生成新 key
# 3. 用 git filter-branch 从历史中移除
git filter-branch --force --index-filter '
  rm -f .env
' --prune-empty HEAD~1
```

## 知识库数据隔离

多租户场景下，不同租户的知识库必须严格隔离：

```python
class TenantAwareRetriever:
    def __init__(self, base_vectorstore, tenant_id: str):
        self.base_store = base_vectorstore
        self.tenant_id = tenant_id

    def invoke(self, query: str) -> list:
        results = self.base_store.similarity_search(query, k=5)

        filtered = []
        for doc, score in results:
            doc_tenant = doc.metadata.get("tenant_id")
            if doc_tenant == self.tenant_id or doc_tenant is None:
                filtered.append((doc, score))

        if not filtered:
            return []  # 不返回任何结果，也不提示"没有权限访问其他租户的数据"

        return filtered[:3]


retriever = TenantAwareRetriever(base_store, tenant_id="customer_A")
results = retriever.invoke("定价信息")
# 只能检索到 customer_A 的知识库文档
```

## 数据泄露应急响应清单

如果发现数据泄露事件，按以下步骤操作：

```
⚠️ 数据泄露应急响应流程

第 1 步: 止损控制 (0-30 分钟)
├── 立即轮换泄露的凭证（API Key / 密码 / Token）
├── 禁用受影响的账户或服务
└── 记录泄露时间范围和受影响数据

第 2 步: 影响评估 (30-60 分钟)
├── 确认泄露了什么类型的数据（PII / 凭证 / 业务数据）
├── 确定受影响的用户数量和范围
├── 评估潜在的业务和法律风险
└── 决定是否需要通知用户和监管机构

第 3 步: 根因修复 (1-24 小时)
├── 定位泄露的具体代码路径
├── 修复漏洞并加强同类防护
├── 全量扫描确认无其他泄露点
└── 更新依赖版本（如果有已知 CVE）

第 4 步: 长期改进 (持续)
├── 建立定期安全审计机制
├── 将安全检查集成到 CI/CD 流水线
├── 对团队进行安全意识培训
└── 制定数据分类和分级保护策略
```
