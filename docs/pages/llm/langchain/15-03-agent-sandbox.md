---
title: 代理的权限控制与沙箱
description: Agent 工具权限矩阵、沙箱执行环境构建、基于角色的访问控制（RBAC）、审计日志与操作追溯、安全 Agent 设计模式
---
# 代理的权限控制与沙箱

前两节我们讨论了 Prompt Injection 攻击和数据泄露风险。这两类威胁在所有 LLM 应用中都存在，但 **Agent 应用面临的额外风险更加严峻**——因为 Agent 有"手和脚"，它能执行工具、调用 API、读写文件、运行代码。一个被攻破的 Agent 不只是说错话的问题——它可能**删除数据库、发送钓鱼邮件、把公司源码打包发出去**。

本章聚焦于 Agent 特有的安全挑战：**如何让 Agent 在拥有执行能力的同时不失控？**

## Agent 的攻击面分析

让我们回顾一下第 8 章和第 11 章构建的 Agent 有哪些能力：

| 能力 | 潜在风险 | 如果被滥用 |
|------|---------|-----------|
| **搜索网络 (search_web)** | 访问恶意 URL | SSRF / 数据外传 |
| **执行 Python 代码 (python_repl)** | 执行任意代码 | 系统命令执行 / 文件读写 |
| **文件操作 (read/write)** | 读取/写入任意路径 | 敏感文件泄露 / 勒索 |
| **SQL 查询 (sql_db_query)** | 注入 SQL | 数据库破坏 / 数据窃取 |
| **HTTP 请求 (httpx)** | 调用内部 API | 内部服务未授权访问 |
| **Shell 命令** | 直接操作系统级 | 完全系统控制 |

一个没有权限控制的 Agent 就像一个**拿着万能钥匙的三岁小孩**——它有能力做任何事，但不知道哪些不该做。

## 第一层：工具级权限矩阵

最直接的控制方式是：**定义每个工具能做什么、不能做什么**。

### 权限定义框架

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Set

class Permission(Enum):
    READ_PUBLIC_DATA = "read_public"
    READ_PRIVATE_DATA = "read_private"
    WRITE_DATA = "write_data"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    ADMIN_OPERATION = "admin"

@dataclass
class ToolPermission:
    tool_name: str
    allowed_permissions: Set[Permission]
    denied_patterns: list[str] = field(default_factory=list)
    max_calls_per_session: int = -1  # -1 = 无限制
    require_confirmation: bool = False


TOOL_PERMISSIONS: dict[str, ToolPermission] = {
    "safe_search": ToolPermission(
        tool_name="web_search",
        allowed_permissions={Permission.READ_PUBLIC_DATA, Permission.NETWORK_ACCESS},
        denied_patterns=[
            r"file://", r"localhost", r"127\.0\.0\.1", r"\.internal",
            r"(?i)(password|secret|token|key)\s*[:=]",
        ],
        max_calls_per_session=20,
    ),
    
    "safe_sql": ToolPermission(
        tool_name="sql_query",
        allowed_permissions={Permission.READ_PRIVATE_DATA},
        denied_patterns=[
            r"(?i)(DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s*SET)",
            r";\s*(DROP|DELETE|INSERT|UPDATE)",
            r"\.\./|\.\.\\",
            r"(?i)(sqlite_master|pg_catalog|information_schema)",
            r"--\s*[\w]+=.*;.*--",
        ],
        max_calls_per_session=50,
        require_confirmation=True,
    ),

    "sandboxed_python": ToolPermission(
        tool_name="python_exec",
        allowed_permissions={Permission.EXECUTE_CODE},
        denied_patterns=[
            r"\bos\.", r"\bsubprocess", r"\beval\s*\(",
            r"\bimport\s+(os|subprocess|socket|shutil|pty|fcntl)",
            r"open\s*\(\s*['\"](/(?:etc|var|home|proc|sys)/",
            r"urllib|requests\.get\(.*verify=False",
        ],
        max_calls_per_session=10,
        require_confirmation=True,
    ),

    "readonly_file": ToolPermission(
        tool_name="read_file",
        allowed_permissions={Permission.READ_PUBLIC_DATA, Permission.READ_PRIVATE_DATA},
        denied_patterns=[
            r"/etc/(passwd|shadow|ssh/)",
            r"\.(pem|key|env|history)",
            r"/home/[a-z]+/(?:\.ssh|\.gnupg|\.bash_history)",
            r"~/(?:\.ssh|\.gnupg)",
        ],
        max_calls_per_session=30,
    ),
}
```

### 权限检查中间件

```python
class PermissionChecker:
    def __init__(self):
        self.permissions = TOOL_PERMISSIONS
        self.call_counts = {}

    def check(self, tool_name: str, tool_input: dict,
               session_id: str) -> tuple[bool, str]:
        perm = self.permissions.get(tool_name)
        if not perm:
            return False, f"未知工具: {tool_name}"

        # 检查调用次数限制
        session_key = f"{session_id}:{tool_name}"
        current_count = self.call_counts.get(session_key, 0)

        if perm.max_calls_per_session > 0:
            if current_count >= perm.max_calls_per_session:
                return False, f"工具 {tool_name} 已达到会话调用上限 ({perm.max_calls_per_session}次)"

        # 检查输入是否匹配拒绝模式
        input_str = json.dumps(tool_input, ensure_ascii=False)
        for pattern in perm.denied_patterns:
            if re.search(pattern, input_str, re.IGNORECASE | re.DOTALL):
                return False, f"输入包含不允许的操作模式"

        # 需要确认的工具
        if perm.require_confirmation:
            return "needs_confirmation", f"工具 {tool_name} 需要用户确认"

        # 通过检查，记录调用次数
        self.call_counts[session_key] = current_count + 1
        return True, ""


checker = PermissionChecker()

# 在 Agent 的工具调用流程中集成
def safe_tool_call(tool_name: str, tool_input: dict, session_id: str):
    result = checker.check(tool_name, tool_input, session_id)
    
    if result[0] == False:
        return {"error": f"[安全拦截] {result[1]}", "status": "blocked"}
    elif result[0] == "needs_confirmation":
        return {"action": "request_confirmation", 
                "tool": tool_name, "input": tool_input}
    
    # 正常执行
    actual_tool = get_tool_by_name(tool_name)
    return actual_tool.invoke(tool_input)
```

## 第二层：沙箱执行环境

即使通过了权限检查，代码仍然在一个可能不安全的环境中运行。我们需要**沙箱**来隔离执行过程。

### Python 代码执行沙箱

```python
import subprocess
import tempfile
import os
import resource
import signal
import sys
from typing import Tuple, Optional

class SandboxExecutor:
    def __init__(self,
                 max_memory_mb: int = 256,
                 max_time_seconds: int = 30,
                 allow_network: bool = False,
                 allowed_modules: set = None):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.max_time = max_time_seconds
        self.allow_network = allow_network
        self.allowed_modules = allowed_modules or {
            'math', 'json', 'datetime', 'collections',
            're', 'random', 'statistics', 'decimal',
            'itertools', 'functools', 'operator',
            'copy', 'string', 'typing', 'hashlib',
        }

    def execute(self, code: str) -> Tuple[int, Optional[str], Optional[str]]:
        """执行代码并返回 (exit_code, stdout, stderr)"""
        
        code = self._preprocess(code)
        if code is None:
            return -1, "", "代码预检未通过"

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py') as tmp:
            tmp.write(code)
            tmp.flush()
            
            try:
                process = subprocess.Popen(
                    [sys.executable, '-B', '-u', tmp.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=self._restrict_process,
                    cwd=tempfile.gettempdir(),
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=self.max_time)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    return -9, stdout, f"执行超时 (> {self.max_time}s)"

                return process.returncode, stdout, stderr
                
            finally:
                os.unlink(tmp.name)

    def _preprocess(self, code: str) -> Optional[str]:
        """预处理：静态分析代码是否安全"""
        dangerous_imports = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                module = stripped.split()[1].split('.')[0].strip(' ,')
                if module not in self.allowed_modules:
                    dangerous_imports.append(module)

        if dangerous_imports:
            blocked = ', '.join(dangerous_imports[:5])
            print(f"[Sandbox] 拒绝导入: {blocked}")
            return None

        if not self.allow_network:
            net_indicators = ['urllib', 'requests', 'httpx', 'socket', 'httplib']
            for indicator in net_indicators:
                if indicator in code:
                    print(f"[Sandbox] 拒绝网络操作: {indicator}")
                    return None

        return code

    def _restrict_process(self):
        """设置进程资源限制"""
        resource.setrlimit(resource.RLIMIT_AS, (self.max_memory, self.max_memory))
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))  # CPU 时间限制
        
        if not self.allow_network:
            import ctypes
            libc = ctypes.CDLL("libc.dylib") if sys.platform == "darwin" else ctypes.CDLL("libc.so.6")
            libc.prctl(38, 0, 0)  # PR_SET_NO_NEW_PRIVS — Linux only


sandbox = SandboxExecutor(max_memory_mb=128, max_time_seconds=15)

# 使用示例
code = """
import math
result = math.sqrt(2)
print(f"sqrt(2) = {result}")
# 尝试危险操作:
# import os; os.system("rm -rf /")
"""

exit_code, stdout, stderr = sandbox.execute(code)
print(f"退出码: {exit_code}")
print(f"输出: {stdout.strip()}")
```

### Docker 沙箱（更强隔离）

对于生产环境，推荐用 Docker 容器作为沙箱：

```dockerfile
FROM python:3.11-slim

RUN groupadd -r sandbox && useradd -r -g sandbox sandbox
USER sandbox

WORKdir /app

COPY executor.py .
CMD ["python", "executor.py"]
```

```python
class DockerSandboxExecutor:
    def __init__(self, image: str = "langchain-sandbox:latest"):
        self.image = image

    def execute(self, code: str) -> dict:
        import base64
        encoded_code = base64.b64encode(code.encode()).decode()

        result = subprocess.run([
            "docker", "run", "--rm",
            "--memory=256m",
            "--cpus=1",
            "--network=none",
            "--security-opt=no-new-privileges",
            "--security-opt=no-protection=false",
            "-v", "/dev/null:/dev:rw",
            self.image,
            "python3", "-c",
            f"exec(base64.b64decode('{encoded_code}').decode()); exec(compile(base64.b64decode('{encoded_code}').decode(), '<string>', 'exec'))"
        ], capture_output=True, text=True, timeout=30)

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
```

Docker 沙箱提供了**内核级别的隔离**：
- `--network=none`：完全禁止网络访问
- `--memory=256m`：硬性内存上限
- `--cpus=1`：CPU 使用限制
- `--no-new-privileges`：禁止获取新权限
- `--security-opt=no-protection` + `seccomp`：系统调用白名单

## 第三层：基于角色的访问控制（RBAC）

不同用户应该有不同的 Agent 能力：

```python
from enum import Enum
from dataclasses import dataclass

class UserRole(Enum):
    GUEST = "guest"
    USER = "user"
    VIP = "vip"
    ADMIN = "admin"
    SERVICE = "service"

@dataclass
class RolePolicy:
    role: UserRole
    can_use_agent: bool
    available_tools: list[str]
    max_agent_turns: int
    can_export_data: bool
    can_access_all_knowledge_bases: bool
    rate_limit_rpm: int


ROLE_POLICIES: dict[UserRole, RolePolicy] = {
    UserRole.GUEST: RolePolicy(
        role=UserRole.GUEST,
        can_use_agent=True,
        available_tools=["safe_search"],
        max_agent_turns=3,
        can_export_data=False,
        can_access_all_knowledge_bases=False,
        rate_limit_rpm=10,
    ),
    UserRole.USER: RolePolicy(
        role=UserRole.USER,
        can_use_agent=True,
        available_tools=["safe_search", "safe_sql", "readonly_file"],
        max_agent_turns=10,
        can_export_data=True,
        can_access_all_knowledge_bases=False,
        rate_limit_rpm=50,
    ),
    UserRole.VIP: RolePolicy(
        role=UserRole.VIP,
        can_use_agent=True,
        available_tools=["safe_search", "safe_sql", "sandboxed_python", "readonly_file"],
        max_agent_turns=30,
        can_export_data=True,
        can_access_all_knowledge_bases=False,
        rate_limit_rpm=200,
    ),
    UserRole.ADMIN: RolePolicy(
        role=UserRole.ADMIN,
        can_use_agent=True,
        available_tools=[],  # 全部可用
        max_agent_turns=-1,  # 无限制
        can_export_data=True,
        can_access_all_knowledge_bases=True,
        rate_limit_rpm=-1,  # 无限制
    ),
}


class RBACAgent:
    def __init__(self, user_role: UserRole):
        self.policy = ROLE_POLICIES[user_role]
        self.turn_count = 0

    def create_agent(self):
        tools = [
            t for name, t in ALL_TOOLS.items()
            if name in self.policy.available_tools or not self.policy.available_tools
        ]

        system_prompt = BASE_SYSTEM_PROMPT + f"""
        
        ## 你的权限等级: {self.policy.role.value}
        - 可用工具: {', '.join(self.policy.available_tools) or '全部'}
        - 最大对话轮数: {self.policy.max_agent_turns or '无限制'}
        - 每分钟请求限制: {self.policy.rate_limit_rpm or '无限制'}
        """

        return create_react_agent(llm=get_llm(), tools=tools, prompt=system_prompt)

    def invoke(self, question: str) -> dict:
        self.turn_count += 1
        if self.policy.max_agent_turns > 0:
            if self.turn_count > self.policy.max_agent_turns:
                return {
                    "response": (
                        f"您已达到本轮对话的最大轮数限制 "
                        f"({self.policy.max_agent_turns}轮)。如需继续，请开新会话。"
                    ),
                    "turn_count": self.turn_count,
                    "rate_limited": False,
                }

        agent = self.create_agent()
        return agent.invoke({"input": question})
```

## 第四层：审计日志与操作追溯

无论前面的防护多么严密，都必须有**完整的审计日志**——既能用于事后追责，也能帮助发现异常行为模式。

### 结构化审计事件

```python
@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    session_id: str
    user_id: str
    user_role: str
    event_type: str           # tool_call / prompt_injection_attempt / permission_denied / etc.
    tool_name: Optional[str]
    tool_input: Optional[str]
    tool_output: Optional[str]
    decision: str              # allowed / denied / needs_confirmation / executed
    risk_score: float          # 0.0 - 1.0
    ip_address: Optional[str]

class AuditLogger:
    def __init__(self, output_path: str = "./logs/audit.jsonl"):
        self.output_path = output_path
        self.events: list[AuditEvent] = []

    def log(self, event: AuditEvent):
        self.events.append(event)
        self._persist_if_needed()

        if event.risk_score >= 0.8:
            self._send_alert(event)

    def log_tool_call(self, tool_name: str, tool_input: dict,
                     decision: str, output: str, risk_score: float,
                     session_id: str, user_id: str, user_role: str):
        event = AuditEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            user_id=user_id,
            user_role=user_role,
            event_type="tool_call",
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, ensure_ascii=False)[:500],
            tool_output=output[:500] if output else None,
            decision=decision,
            risk_score=risk_score,
        )
        self.log(event)

    def _persist_if_needed(self):
        if len(self.events) >= 10:
            with open(self.output_path, "a") as f:
                for evt in self.events:
                    f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")
            self.events.clear()

    def _send_alert(self, event: AuditEvent):
        alert_msg = (
            f"🚨 高风险审计事件\n"
            f"  用户: {event.user_id} ({event.user_role})\n"
            f"  操作: {event.event_type}\n"
            f"  工具: {event.tool_name}\n"
            f"  决策: {event.decision}\n"
            f"  风险分: {event.risk_score}\n"
            f"  IP: {event.ip_address}"
        )
        print(alert_msg)
        # 实际项目中可接入 Slack/Email/PagerDuty


audit_logger = AuditLogger()

# 在 PermissionChecker 中记录审计日志
def checked_tool_call(tool_name, tool_input, session_id, user_id, user_role):
    result = checker.check(tool_name, tool_input, session_id)
    audit_logger.log_tool_call(
        tool_name=tool_name,
        tool_input=tool_input,
        decision="allowed" if result[0] else ("denied" if result[0] is False else "confirmed"),
        output="",
        risk_score=0.9 if result[0] is False else 0.1,
        session_id=session_id,
        user_id=user_id,
        user_role=user_role,
    )
```

## 安全 Agent 设计模式总结

把所有层次整合在一起，一个安全的 Agent 应该是这样的架构：

```
用户请求
    │
    ▼ [输入层: InputSanitizer] ← 第14章内容
    │   清洗输入 → 检测注入 → 中性化处理
    │
    ▼ [认证层: RBAC] ← 本节新增
    │   验证身份 → 加载角色策略 → 限制可用工具
    │
    ▼ [意图分类] ← 第10章内容
    │   判断问题类型 → 路由到对应处理器
    │
    ▼ [工具调度层: PermissionChecker] ← 本节核心
    │   工具是否存在？→ 是否有权限？→ 输入是否安全？
    │   → 超过调用次数？→ 需要确认？
    │
    ▼ [执行层: SandboxedExecutor] ← 本节核心
    │   代码预检（禁止导入）→ 进程隔离（资源限制）
    │   → 网络禁用（Docker --network=none）
    │
    ▼ [输出层: OutputFilter + PIIDetector] ← 第15章前两节
    │   输出是否含敏感信息？→ 是否含可执行代码？
    │   → 是否触发了注入响应？
    │
    ▼ [审计层: AuditLogger] ← 本节新增
    │   记录完整操作链路 → 高风险实时告警
    │
    ▼ 返回给用户（安全版本）
```

每一层都是独立的防线——即使某一层被突破，后续层仍有机会拦截或减轻损害。这就是纵深防御的核心思想。

## 安全不是一次性工作

最后需要强调的是：**安全是一个持续对抗的过程**。今天有效的防御措施可能在下周就被新的攻击技术绕过。建立一个安全意识文化比任何单一技术方案都重要：

1. **每次添加新工具时评估其安全影响**
2. **定期进行红队测试（模拟攻击）**
3. **关注 LLM 安全社区的动态更新**
4. **建立安全事件的标准响应流程**
5. **让安全审查成为 Code Review 的必选项**
