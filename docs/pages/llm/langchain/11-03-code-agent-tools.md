---
title: 利用代理调用代码执行工具
description: ReAct Agent 驱动的代码分析、PythonREPL 工具集成、代码修复-验证闭环、完整 CodeAssistant 系统组装
---
# 利用代理调用代码执行工具

前两节我们实现了代码解释、Bug 检测和测试生成——这些能力都是"只读"的：LLM 分析代码、输出文字，但**不真正运行代码**。这就像一个能看懂图纸但从不下工地的工程师——分析能力再强，也无法验证自己的判断是否正确。

这一节我们要把 Agent 引入进来，让 AI 不仅能**看**代码，还能**写**代码、**跑**代码、根据运行结果**修正**自己的判断。这是从"被动分析"到"主动实验"的质的飞跃。

## 为什么需要代码执行能力

用一个具体场景来说明。假设用户问：

> "这个 `calculate_discount` 函数，输入 999 元的时候输出对不对？"

纯分析模式的回答是：

```
根据代码逻辑，999 元落在 500-2000 区间，折扣率 10%，
所以折扣额 = 99.9 元，最终金额 = 899.1 元。
```

这个回答看起来没问题。但如果代码里有一个隐蔽的边界条件 bug——比如 `max_purchase` 的比较用了 `<=` 而不是 `<`，导致 2000 这个边界值被错误地归类到了上一档——纯分析模式可能发现不了（或者发现了但不敢确定）。

而有了代码执行能力之后，Agent 可以这样做：

```
Thought: 用户想知道 calculate_discount(999) 的输出是否正确。
       我应该直接运行这段代码来验证。
Action: python_repl → 执行 calculate_discount(999)
Observation: {'original': 999, 'discount_rate': 0.1,
            'discount_amount': 99.9, 'final_amount': 899.1}
Thought: 运行结果与分析一致。让我再测几个边界值来确认没有隐藏问题。
Action: python_repl → 执行 calculate_discount(100), calculate_discount(500),
                       calculate_discount(2000)
Observation: [结果列表...]
Thought: 所有边界值都符合预期。可以给用户确认了。
Final Answer: 经过实际运行验证...
```

**看到区别了吗？Agent 不只是在"猜"，它在用事实说话。**

## 设计代码执行工具集

LangChain 内置了 `PythonREPL` 工具，可以让 Agent 在沙箱环境中安全地执行 Python 代码。我们需要围绕它构建一套适合代码分析场景的工具集。

### 基础工具：Python 代码执行器

```python
from langchain_community.tools import PythonREPLTool
from langchain_core.tools import tool

python_repl = PythonREPLTool()

@tool
def execute_python_code(code: str) -> str:
    """在安全的沙箱环境中执行 Python 代码并返回输出结果。
    用于验证代码行为、计算表达式的值、或运行单元测试。

    Args:
        code: 要执行的 Python 代码字符串
    """
    try:
        result = python_repl.invoke(code)
        return f"✅ 执行成功\n\n输出:\n{result}"
    except Exception as e:
        return f"❌ 执行失败\n\n错误: {type(e).__name__}: {e}"
```

### 增强工具：带上下文的代码执行

原始的 `PythonREPLTool` 每次执行都是独立的——上一次定义的变量和函数在下一次执行中不存在。对于代码分析场景，我们经常需要先加载被分析的代码，然后多次调用其中的函数：

```python
@tool
def execute_with_context(code: str, context_code: str = "") -> str:
    """在给定上下文环境中执行代码。context_code 会先被执行，
    用于导入依赖和定义类/函数，然后再执行 code。

    Args:
        code: 要执行的主要代码
        context_code: 上下文代码（如 import、类定义等），会在 code 之前执行
    """
    full_code = ""
    if context_code:
        full_code += f"# === 上下文代码 ===\n{context_code}\n\n"
    full_code += f"# === 要执行的代码 ===\n{code}"

    try:
        result = python_repl.invoke(full_code)
        return f"✅ 执行成功\n\n输出:\n{result}"
    except Exception as e:
        return f"❌ 执行失败\n\n错误: {type(e).__name__}: {e}"


@tool
def run_unit_test(test_code: str, source_code: str) -> str:
    """运行 pytest 单元测试并返回测试报告。

    Args:
        test_code: 测试代码（包含 pytest 用例）
        source_code: 被测试的源代码
    """
    combined = f"{source_code}\n\n# --- 测试代码 ---\n{test_code}\n\n"
    run_code = f"""
import pytest
import sys
from io import StringIO

{combined}

if __name__ == "__main__":
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    exit_code = pytest.main(["-v", "--tb=short", __file__])
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    print(output)
    print(f"\\n退出码: {exit_code}")
"""
    try:
        result = python_repl.invoke(run_code)
        return f"🧪 测试完成\n\n{result}"
    except Exception as e:
        return f"❌ 测试执行出错: {type(e).__name__}: {e}"
```

### 检索工具：查询代码知识库

Agent 还需要能够主动从代码库中检索相关代码，而不是只能处理用户手动粘贴的代码片段：

```python
@tool
def search_codebase(query: str, top_k: int = 5) -> str:
    """在代码知识库中搜索与 query 相关的代码片段。

    Args:
        query: 搜索关键词或自然语言描述
        top_k: 返回的结果数量上限
    """
    results = retriever.invoke(query)[:top_k]
    if not results:
        return "未找到匹配的代码片段"

    output_parts = []
    for i, doc in enumerate(results):
        meta = doc.metadata
        output_parts.append(
            f"[{i+1}] {meta.get('node_type', '?')}: {meta.get('name', '?')} "
            f"({meta.get('source', '?')} L{meta.get('start_line', '?')}-{meta.get('end_line', '?')})\n"
            f"```python\n{doc.page_content}\n```"
        )

    return "\n\n".join(output_parts)


tools = [
    execute_python_code,
    execute_with_context,
    run_unit_test,
    search_codebase,
]
```

## 构建 ReAct 代码分析 Agent

有了工具集之后，按照第 8 章学过的 ReAct 模式创建 Agent：

```python
from langchain.agents import create_react_agent

CODE_AGENT_SYSTEM_PROMPT = """你是一个高级代码分析助手，具备以下能力：
1. 从代码仓库中搜索和检索代码
2. 解释代码的功能和设计意图
3. 发现代码中的 Bug 和安全问题
4. 执行代码来验证你的分析
5. 生成并运行单元测试
6. 提出修复建议并验证修复效果

工作原则：
- 当需要了解某个函数/类的实现时，使用 search_codebase 工具搜索
- 当需要验证代码的行为时，使用 execute_python_code 或 execute_with_context 执行
- 当需要验证修改后的代码是否正确时，先生成测试再运行测试
- 如果执行报错，仔细阅读错误信息并调整方案后重试
- 最终回答要基于实际执行结果，不要凭猜测下结论"""

agent = create_react_agent(
    llm=get_llm(),
    tools=tools,
    prompt=CODE_AGENT_SYSTEM_PROMPT,
)
```

### 场景一：交互式代码调试

让 Agent 来帮用户排查一个真实的 bug：

```python
result = agent.invoke({
    "input": (
        "我有一个 DiscountCalculator 类，但我怀疑它的阶梯折扣计算有 bug。\n"
        "请帮我：1) 先从代码库中找到这个类的完整实现 "
        "2) 用几个测试数据跑一下看看输出是否符合预期 "
        "3) 如果有 bug 请定位原因并给出修复方案"
    )
})

print(result["output"])
```

Agent 的内部推理过程大致如下（简化版）：

```
Thought: 我需要先找到 DiscountCalculator 的完整代码。
Action: search_codebase("DiscountCalculator class implementation")
Observation: [1] ClassDef: DiscountCalculator (discount.py L1-L30)
           完整代码...

Thought: 我看到了完整实现。现在用几个关键值来测试各档位的折扣计算。
       特别关注区间边界值：0, 100, 500, 2000。
Action: execute_with_context(code="...", context_code="完整的类定义")
Observation:
  amount=50   → rate=0.00, final=50.00     ✅ 正确
  amount=100  → rate=0.05, final=95.00      ✅ 正确
  amount=500  → rate=0.10, final=450.00     ✅ 正确
  amount=2000 → rate=0.15, final=1700.00    ✅ 正确

Thought: 边界值都正确。但让我检查一下浮点数精度问题和负数输入。
Action: execute_with_context(code="calculate(333.333); calculate(-1)", ...)
Observation:
  333.333 → discount=33.33, final=299.99    ✅ 四舍五入正确
  -1      → ValueError: 购买金额不能为负数   ✅ 异常处理正确

Thought: 所有测试都通过了。但等等——我注意到一个潜在的设计问题：
       第4档的 max_purchase 是 float("inf")，在 for 循环中的比较
       `purchase_amount < tier["max_purchase"]` 对 inf 总是 True，
       所以最后一个分支永远不会被执行到。虽然当前逻辑不会走到那里
       （因为前面的 if 已经覆盖了所有情况），但这属于冗余代码。
       
       更重要的问题：如果 tiers 列表的顺序被打乱，整个逻辑就错了。
       这说明代码依赖于隐式的列表顺序约束。

Final Answer: 经过完整测试，DiscountCalculator 在功能上是正确的...
[后续给出详细的分析报告和改进建议]
```

注意 Agent 的行为模式：**搜索 → 阅读 → 设计测试用例 → 执行 → 分析结果 → 发现更深层次的问题**。这不是一次性的回答，而是一个多步推理过程，每一步都有工具调用的实际结果作为依据。

### 场景二：Bug 修复 + 验证闭环

更强大的场景是：Agent 不仅发现问题，还**自动修复并验证修复效果**：

```python
result = agent.invoke({
    "input": (
        "下面这段 UserAPI 的登录函数有安全问题，请帮我：\n"
        "1) 找出所有问题\n"
        "2) 给出修复后的完整代码\n"
        "3) 写几个测试来验证修复后的代码确实解决了这些问题\n"
        "4) 运行测试确认通过\n\n"
        "[有问题的代码]\n" + buggy_code
    )
})
```

Agent 的工作流程：

```
Step 1: 分析代码，识别出 SQL 注入、JWT 密钥硬编码等问题
Step 2: 生成修复版代码（参数化查询、环境变量读取密钥等）
Step 3: 生成 pytest 测试用例
Step 4: 执行测试
Step 5: 测试全部通过 → 输出最终报告
```

这种 **"诊断 → 处方 → 验证"** 的闭环工作流，正是 Agent 区别于简单 Chain 的核心价值所在。

## 完整系统组装：CodeAssistant

现在我们把前三节的所有能力整合到一个完整的 `CodeAssistant` 类中：

```python
class CodeAssistant:
    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path
        self.retriever = None
        self.analyzer = None
        self.agent = None
        self._initialized = False

    def initialize(self):
        if self.repo_path:
            repo_loader = CodeRepositoryLoader(self.repo_path)
            docs = repo_loader.load_tracked_files()
            splitter = SmartCodeSplitter()
            chunks = []
            for doc in docs:
                chunks.extend(splitter.split_document(doc))
            enhanced = [enhance_code_chunk(c) for c in chunks]
            vectorstore = build_code_index(enhanced)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        self.analyzer = CodeAnalyzer(retriever=self.retriever)

        self.agent = create_react_agent(
            llm=get_llm(),
            tools=tools,
            prompt=CODE_AGENT_SYSTEM_PROMPT,
        )

        self._initialized = True
        return self

    def explain(self, question: str = None, code: str = None, file_path: str = None,
                target_name: str = None) -> str:
        if not self._initialized:
            raise RuntimeError("请先调用 initialize()")

        if question and self.retriever:
            return project_explain_chain.invoke({"question": question})
        elif code:
            return self.analyzer.analyze(AnalysisTask.EXPLAIN, code, "")
        elif file_path:
            return self.analyzer.analyze_from_repo(
                AnalysisTask.EXPLAIN, file_path, target_name
            )
        else:
            return "请提供 question、code 或 file_path 之一"

    def find_bugs(self, code: str, context: str = "") -> str:
        return self.analyzer.analyze(AnalysisTask.FIND_BUGS, code, context)

    def generate_tests(self, code: str, context: str = "") -> str:
        return self.analyzer.analyze(AnalysisTask.GENERATE_TESTS, code, context)

    def investigate(self, query: str) -> str:
        result = self.agent.invoke({"input": query})
        return result["output"]

    def chat(self, user_message: str, session_history: list = None) -> str:
        CHAT_SYSTEM = """你是一个代码分析助手。你可以帮助用户：
- 解释代码功能和设计
- 查找代码中的 Bug 和安全问题
- 生成单元测试
- 从代码库中搜索特定功能的实现
- 执行代码验证分析结果

如果用户的请求涉及代码执行或深度分析，使用你的工具来完成。
如果是简单的问答，直接回答即可。"""

        history_text = "\n".join([
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
            for m in (session_history or [])
        ])

        prompt = f"{CHAT_SYSTEM}\n\n{history_text}\n\nUser: {user_message}\nAssistant:"
        result = get_llm().invoke(prompt)
        return result.content
```

### CLI 交互界面

```python
def run_code_assistant_cli():
    print("=" * 60)
    print("   🧑‍💻 代码分析助手 (Code Assistant)")
    print("   支持命令:")
    print("   /explain <file> [<name>]  — 解释代码")
    print("   /bugs <paste code>        — 查找 Bug")
    print("   /test <paste code>        — 生成测试")
    print("   /search <keyword>         — 搜索代码库")
    print("   /investigate <question>   — Agent 深度调查")
    print("   /quit                     — 退出")
    print("=" * 60)

    assistant = CodeAssistant(repo_path="./sample_project")
    assistant.initialize()

    while True:
        try:
            user_input = input("\n🧑‍💻 > ").strip()

            if user_input.lower() in ("quit", "exit", "/quit"):
                break

            if user_input.startswith("/explain"):
                parts = user_input.split(maxsplit=2)
                target_name = parts[2] if len(parts) > 2 else None
                result = assistant.explain(file_path=parts[1], target_name=target_name)
                print(f"\n{result}")
                continue

            if user_input.startswith("/investigate"):
                query = user_input.replace("/investigate", "", 1).strip()
                print("\n🔍 Agent 调查中...\n")
                result = assistant.investigate(query)
                print(result)
                continue

            if user_input.startswith("/search"):
                query = user_input.replace("/search", "", 1).strip()
                result = search_codebase.invoke({"query": query})
                print(f"\n{result}")
                continue

            if user_input.startswith("/bugs") or user_input.startswith("/test"):
                task = AnalysisTask.FIND_BUGS if user_input.startswith("/bugs") else AnalysisTask.GENERATE_TESTS
                code = user_input.split(maxsplit=1)[1] if len(user_input.split()) > 1 else ""
                if not code:
                    print("请粘贴代码内容")
                    continue
                result = assistant.analyzer.analyze(task, code, "")
                print(f"\n{result}")
                continue

            result = assistant.chat(user_input)
            print(f"\n🤖 {result}")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    run_code_assistant_cli()
```

## 安全考量：代码执行的沙箱隔离

让 AI 执行任意代码是一个**高风险操作**。如果用户（或 Agent 本身）执行了 `os.system("rm -rf /")` 或 `subprocess.run(["format", "C:"])` 这样的恶意代码，后果不堪设想。

### LangChain PythonREPL 的内置安全机制

LangChain 的 `PythonREPLTool` 使用了以下安全措施：

| 措施 | 说明 |
|------|------|
| **全局变量隔离** | 每次执行在独立的全局命名空间中进行 |
| **超时控制** | 单次执行有时间限制（默认约 10 秒） |
| **禁止危险操作** | 默认禁用了 `import os`, `import subprocess` 等 |
| **stdout/stderr 捕获** | 不会影响宿主进程的标准输出 |

### 生产环境的额外加固

在生产环境中，建议进一步增加安全层：

```python
import re

DANGEROUS_PATTERNS = [
    r"\bos\.system\s*\(",
    r"\bos\.popen\s*\(",
    r"\bsubprocess",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"open\s*\([\"']\/",
    r"import\s+ctypes",
]

def sanitize_code(code: str) -> tuple[bool, str]:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return False, f"检测到危险操作: {pattern}，已拒绝执行"
    return True, ""

def safe_execute(code: str) -> str:
    safe, reason = sanitize_code(code)
    if not safe:
        return f"🚫 安全拦截: {reason}"
    return python_repl.invoke(code)
```

### 权限分级策略

不同场景应采用不同的执行权限级别：

| 场景 | 权限 | 允许的操作 |
|------|------|-----------|
| **本地开发 CLI** | 宽松 | 全部允许（开发者自己负责） |
| **内部 Web 服务** | 中等 | 禁止文件系统和网络操作 |
| **面向公众的服务** | 严格 | 只允许纯计算，禁止一切 I/O |

## 项目总结与扩展方向

至此，我们的代码分析助手已经具备了完整的能力矩阵：

| 能力 | 实现方式 | 章节 |
|------|---------|------|
| 代码库加载与索引 | Git Loader + AST Splitter + Chroma | 11.1 |
| 代码自然语言解释 | Explain Chain + RAG 增强 | 11.2 |
| Bug 检测与定位 | Bug Finder Chain + 严重度分类 | 11.2 |
| 单元测试自动生成 | Test Gen Chain + pytest 格式 | 11.2 |
| 代码执行与验证 | PythonREPL Tool + ReAct Agent | 11.3 |
| 多步调查与修复 | Agent 自主规划 + 工具调用 | 11.3 |

### 可继续探索的方向

**第一，支持更多语言**。目前的实现聚焦于 Python。如果要支持 JavaScript/TypeScript，可以用 `tree-sitter` 替代 `ast` 做语法解析，用 Node.js 的 `vm` 模块替代 Python REPL 做代码执行。

**第二，Git 集成增强**。不仅读取代码，还能理解 Git 历史——"这个函数是谁写的？上次改了什么？关联的 issue 是什么？"这需要在 metadata 中注入 `git blame` 和 `git log` 信息。

**第三，IDE 插件化**。把 CodeAssistant 封装成 VS Code 插件或 JetBrains 插件，用户选中一段代码就能右键调用"解释""查 bug""生成测试"。这是最自然的交互方式。

**第四，持续学习反馈闭环**。收集用户对 AI 分析结果的反馈（👍/👎），将"坏案例"用于微调 prompt 或 fine-tune 模型，形成持续改进的飞轮。这是第 12 章（评估与可观测性）的核心议题。
