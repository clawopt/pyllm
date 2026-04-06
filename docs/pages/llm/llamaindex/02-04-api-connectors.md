---
title: API 连接器：OpenAI / Notion / Slack / GitHub / Jira / 网页爬取
description: 各类 SaaS 平台 API 的数据接入、认证机制、分页处理、速率限制应对、网页爬取策略
---
# API 连接器：OpenAI / Notion / Slack / GitHub / Jira / 网页爬取

在现代企业的技术栈中，大量有价值的信息并不存在于本地文件或自建数据库中，而是散落在各种 **SaaS（软件即服务）平台**上：团队知识写在 Notion 里、代码讨论在 GitHub Issues 中、客户工单在 Jira 里、内部沟通在 Slack 中、设计稿在 Figma 上。这些平台都提供了 API 让程序化地访问数据，LlamaIndex 的 API 连接器就是用来把这些分散的知识汇聚到一个统一的 RAG 系统中的桥梁。

这一节我们会系统地介绍几种最常用的 API 连接器，以及通用的 API 数据接入模式和最佳实践。

## 通用 API 接入模式

在深入具体平台之前，我们先建立一个认知框架——几乎所有 API 连接器的使用都遵循相同的模式：

```
1. 认证（Authentication）
   ↓ 提供 API Key / OAuth Token / 其他凭证
2. 配置（Configuration）
   ↓ 设置查询参数、过滤条件、字段选择等
3. 加载（Loading）
   ↓ 调用 load_data() 获取 Document 列表
4. 分页处理（Pagination）
   ↓ 自动处理 API 的分页机制
5. 速率限制（Rate Limiting）
   ↓ 尊重 API 的调用频率限制
```

无论连接的是 Notion 还是 GitHub，这五个步骤都是一样的。区别只在于具体的参数名称和 API 行为差异。掌握了这个通用模式后，学习任何新的 API 连接器都会变得非常快。

## Notion 连接器

Notion 已经成为很多团队的"第二大脑"——产品需求文档、会议纪要、项目规划、团队 Wiki 全都在 Notion 里。如果能把这些知识接入 RAG 系统，价值巨大。

```bash
pip install llama-index-readers-notion
```

### 获取 Notion API 凭证

在使用 Notion API 之前，你需要：
1. 登录 [notion.so/my-integrations](https://www.notion.so/my-integrations)
2. 创建一个新的 Integration（集成）
3. 复制生成的 `Internal Integration Secret`（以 `ntn_` 或 `secret_` 开头）
4. 在你想访问的 Notion 页面上，点击右上角 `...` → `Connections` → 添加刚才创建的 Integration

```python
import os
from llama_index.readers.notion import NotionPageReader

reader = NotionPageReader(
    integration_token=os.getenv("NOTION_INTEGRATION_TOKEN"),
)

# 方式一：读取特定页面及其子页面
documents = reader.load_data(
    page_ids=["your-page-id-uuid"],
)

# 方式二：读取整个工作区（需要 Integration 有足够权限）
documents = reader.load_data(
    # 不传 page_ids 则读取所有可访问的页面
)

print(f"从 Notion 加载了 {len(documents)} 个页面/块")
for doc in documents[:5]:
    print(f"  [{doc.metadata.get('title', 'N/A')}] "
          f"{len(doc.text)} 字符")
```

NotionPageReader 的特点：
- **自动递归子页面**：如果你给的是一个父页面 ID，它会自动遍历所有子页面
- **保留块结构**：Notion 的内容是由"块"（blocks）组成的——段落、标题、列表、代码块等。Reader 会保留这些结构信息
- **元数据丰富**：自动包含页面标题、创建时间、最后编辑者、URL 等

### Notion 数据库（Database）读取

Notion 除了页面还有一种叫"数据库"的结构（类似于电子表格），这在企业中常被用作轻量的 CRM、项目管理工具等：

```python
from llama_index.readers.notion import NotionPageReader

reader = NotionPageReader(
    integration_token=os.getenv("NOTION_TOKEN"),
)

# 读取 Notion Database 中的记录
documents = reader.load_data(
    database_id="your-database-id-uuid",
    request_property_filter={  # Notion 的过滤语法
        "property": "Status",
        "select": {"equals": "Published"},
    },
)

for doc in documents:
    # Notion Database 的每个属性都会进入 metadata
    print(f"标题: {doc.metadata.get('Title', 'N/A')}")
    print(f"状态: {doc.metadata.get('Status', 'N/A')}")
    print(f"标签: {doc.metadata.get('Tags', 'N/A')}")
```

### Notion 连接的常见问题

**权限问题：** 最常见的错误是 `403 Unauthorized`。原因几乎总是 Integration 没有被添加到目标页面的 Connections 中。记住：**Notion 的权限模型是页面级别的**——即使你的 Integration Token 有效，如果没有被显式添加到某个页面，就无法访问它。

**速率限制：** Notion API 的速率限制是每秒 3 个请求（平均）。LlamaIndex 的 Reader 内置了基本的限速处理，但如果你的工作区页面数量很大（数千个），完整加载可能需要相当长的时间。建议在首次加载后缓存结果，之后只做增量更新。

## GitHub 连接器

对于技术团队来说，GitHub 不仅是代码托管平台，更是巨大的知识库——Issues 里记录了 Bug 讨论和功能需求、 Discussions 里沉淀了技术决策、Wiki 里写着架构文档、README 里说明了使用方法。

```bash
pip install llama-index-readers-github
```

```python
from llama_index.readers.github import GitHubReader

reader = GitHubReader(
    owner="your-org",          # 组织或用户名
    repo="your-repo",          # 仓库名
    github_token=os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
    verbose=False,
)

# 读取仓库的 Issues
documents = reader.load_data(
    filter_dir=(             # 过滤条件
        GitRepositoryReadFilterDir.ISSUES
    ),
)

print(f"加载了 {len(documents)} 个 Issues")
for doc in documents[:3]:
    print(f"  #{doc.metadata.get('issue_number')}: "
          f"{doc.metadata.get('title')}")
    print(f"  内容: {doc.text[:150]}...")
```

### GitHub Reader 支持的数据类型

GitHubReader 可以读取多种类型的 GitHub 数据：

```python
from llama_index.readers.github import GitHubRepositoryReader, GitRepositoryReadFilterDir

reader = GitHubRepositoryReader(
    owner="openai",
    repo="cookbook",
    github_token=os.getenv("GITHUB_TOKEN"),
)

# 不同 filter_dir 对应不同数据源
issues_docs = reader.load_data(
    filter_dir=GitRepositoryReadFilterDir.ISSUES,
)

discussions_docs = reader.load_data(
    filter_dir=GitRepositoryReadFilterDir.DISCUSSIONS,
)

code_docs = reader.load_data(
    filter_dir=GitRepositoryReadFilterDir.CODE,
    filter_file_extensions=[".py", ".md"],  # 只要 Python 和 Markdown
    branch="main",                           # 指定分支
)

wiki_docs = reader.load_data(
    filter_dir=GitRepositoryReadFilterDir.WIKI,
commits_docs = reader.load_data(
    filter_dir=GitRepositoryReadFilterDir.COMMITS,
)
```

### GitHub API 的速率限制

GitHub API 对认证用户的速率限制是每小时 5000 次请求，未认证用户只有 60 次。**强烈建议始终使用 Personal Access Token (PAT)**，否则很快就会触发限制。

如果仓库很大（几千个文件或 Issues），一次加载可能会触及限制。解决方案包括：
1. **分批加载**：先加载 Issues，下次加载 Code
2. **使用条件过滤**：只加载特定时间段内的数据
3. **利用 ETag 缓存**：GitHub API 支持 ETag，未变更的资源不计入限额

## Slack 连接器

Slack 里的对话记录蕴含着大量隐性知识——问题的解决方案、技术决策的理由、团队成员的经验分享。但这些知识通常"沉睡"在历史消息中，很难被主动检索到。

```bash
pip install llama-index-readers-slack
```

```python
from llama_index.readers.slack import SlackReader

reader = SlackReader(
    slack_token=os.getenv("SLACK_USER_TOKEN"),  # 用户级 OAuth Token
    channel_ids=["C01ABCDEF", "C02GHIJKL"],     # 频道 ID 列表
    earliest_timestamp="1706745600",             # Unix 时间戳（可选）
    latest_timestamp=int(time.time()),            # Unix 时间戳（可选）
)

documents = reader.load_data()
print(f"从 Slack 加载了 {len(documents)} 条消息")

for doc in documents[:5]:
    print(f"[{doc.metadata.get('channel')}] "
          f"{doc.metadata.get('user')}: {doc.text[:100]}")
```

### Slack 认证的坑

Slack API 的认证是所有连接器中最复杂的之一，因为 Slack 有**多种 Token 类型**，权限范围也不同：

| Token 类型 | 格式前缀 | 用途 | 权限范围 |
|-----------|---------|------|---------|
| Bot Token | `xoxb-` | 代表应用操作 | 由 OAuth Scope 决定 |
| User Token | `xoxp-` | 代表用户操作 | 由 OAuth Scope 决定 |
| Legacy Token | `xoxs-` | 旧版 token | 全部权限（不推荐） |

对于读取历史消息，你需要的是 **User Token**（代表用户身份去读取），并且需要有 `channels:history` 或 `im:history` 等 scope。

**重要提示：** Slack Reader 需要 User Token（不是 Bot Token），而且必须在 Slack App Settings 中开启 `token.rotation` 或者使用 legacy token。具体配置步骤比较繁琐，建议参考 Slack 官方的 API 文档。

## Jira 连接器

Jira 是项目管理的事实标准，其中的 Issue 描述、评论、工单历史包含了丰富的项目上下文信息。

```bash
pip install llama-index-readers-jira
```

```python
from llama_index.readers.jira import JiraReader

reader = JiraReader(
    server_url="https://your-company.atlassian.net",
    email="bot@company.com",
    api_token=os.getenv("JIRA_API_TOKEN"),  # Jira 的 PAT（非密码）
)

# 按 JQL（Jira Query Language）查询
documents = reader.load_data(
    jql_query=(
        'project = "PROD" '
        'AND status in (Done, Closed) '
        'AND updated >= -30d'  # 最近 30 天更新的
    ),
    max_results=100,
)

print(f"从 Jira 加载了 {len(documents)} 个 Issue")
for doc in documents[:3]:
    print(f"[{doc.metadata.get('key')}] {doc.metadata.get('summary')}")
    print(f"  类型: {doc.metadata.get('issue_type')} | "
          f"状态: {doc.metadata.get('status')}")
```

Jira Reader 的元数据非常丰富——Issue 的关键字段（key、summary、type、status、priority、assignee、reporter、created、updated 等）都会被自动提取到 metadata 中，这对于后续的过滤和溯源非常有用。

## 网页爬取

除了特定的 SaaS 平台，互联网本身就是一个巨大的知识源——技术博客、新闻网站、论坛帖子、文档站点等都包含有价值的信息。LlamaIndex 提供了多种网页数据获取方式。

### BeautifulSoupWebReader：基础爬取

适用于静态内容的网页（服务端渲染的 HTML）：

```bash
pip install llama-index-readers-web beautifulsoup4
```

```python
from llama_index.readers.web import BeautifulSoupWebReader

reader = BeautifulSoupWebReader()
documents = reader.load_data(
    urls=[
        "https://docs.example.com/guide/getting-started",
        "https://docs.example.com/api/reference",
        "https://blog.example.com/best-practices",
    ]
)

print(f"爬取了 {len(documents)} 个网页")
for doc in documents:
    print(f"URL: {doc.metadata.get('url')}")
    print(f"内容长度: {len(doc.text)} 字符")
```

### PlaywrightWebReader：JavaScript 渲染

现代网站大量使用 JavaScript 动态渲染内容（SPA、React/Vue 应用等）。普通的 HTTP 请求只能拿到空的 HTML shell，真正的内容需要浏览器执行 JS 后才会出现。这时候就需要 **Playwright** —— 一个自动化浏览器工具：

```bash
pip install llama-index-readers-web playwright
playwright install chromium  # 首次使用需要下载浏览器
```

```python
from llama_index.readers.web import PlaywrightWebReader

reader = PlaywrightWebReader(
    browser_type="chromium",
    headless=True,  # 无头模式，不显示浏览器窗口
)

documents = reader.load_data(
    urls=["https://example.com/dashboard"],  # SPA 页面
)
```

PlaywrightWebReader 会启动一个真实的 Chromium 浏览器，等待页面完全加载（包括 AJAX 请求、动态渲染），然后提取最终的 DOM 内容。代价是速度比 BeautifulSoupWebReader 慢很多（需要启动浏览器、渲染页面），并且消耗更多资源。

### 什么时候用哪种爬取器？

| 场景 | 推荐工具 | 原因 |
|------|---------|------|
| 静态 HTML 页面 | BeautifulSoupWebReader | 快速、轻量 |
| JavaScript 渲染的 SPA | PlaywrightWebReader | 能执行 JS |
| 需要登录的页面 | PlaywrightWebReader | 支持交互操作 |
| 大规模批量爬取 | 自定义（Scrapy 等） | LlamaIndex Reader 不适合大规模爬取 |
| RSS / 新闻订阅 | RssUrlReader | 专为订阅设计 |

### 爬取的伦理与法律边界

在爬取网页数据时，务必注意以下几点：

**1. 检查 robots.txt。** `https://example.com/robots.txt` 告诉你网站允许哪些爬虫访问哪些路径。尊重 robots.txt 既是法律风险规避，也是网络礼仪。

**2. 控制请求频率。** 高频请求会对目标服务器造成压力，也可能触发 IP 封禁。建议在请求之间加入随机延迟（1-3 秒）。

**3. 关注版权和使用条款。** 爬取来的数据用于内部 RAG 一般属于合理使用，但公开发布爬取到的内容可能涉及版权问题。

**4. 不要爬取个人隐私数据。** 用户的个人信息、私信等内容即使在技术上可以获取，也不应该在未经授权的情况下收集和使用。

## API 连接器的通用最佳实践

### 认证信息安全管理

API Key 和 Token 是敏感信息，**绝对不能硬编码在代码中**：

```python
# ❌ 错误做法
reader = NotionPageReader(integration_token="ntn_secret_xxxxxxxxxx")

# ✅ 正确做法
import os
reader = NotionPageReader(integration_token=os.getenv("NOTION_TOKEN"))

# ✅ 或者使用 .env 文件 + python-dotenv
from dotenv import load_dotenv
load_dotenv()
reader = NotionPageReader(integration_token=os.getenv("NOTION_TOKEN"))
```

### 错误处理与重试

API 调用不可避免会遇到网络抖动、服务暂时不可用等问题。好的实践是为 API 连接器添加重试机制：

```python
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def safe_load_data(reader):
    try:
        return reader.load_data()
    except Exception as e:
        print(f"加载失败 (将重试): {e}")
        raise

documents = safe_load_data(reader)
```

### 数据缓存

API 数据加载通常比较慢（受限于网络延迟和速率限制），而且很多 API 是按调用量计费的。**缓存加载结果可以大幅节省时间和成本**：

```python
import json
import hashlib
from pathlib import Path

CACHE_DIR = Path("./cache")

def cached_load(reader, cache_key):
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        print(f"从缓存加载: {cache_key}")
        return [Document(**d) for d in data]

    documents = reader.load_data()
    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump([{"text": d.text, "metadata": d.metadata} for d in documents], f)
    print(f"已缓存: {cache_key} ({len(documents)} 条)")
    return documents
```

## 常见误区

**误区一："API 连接器可以无限量地拉取数据"。** 几乎所有 SaaS API 都有速率限制（Rate Limit）。Notion 是 3 次/秒，GitHub 是 5000 次/小时（认证用户），Slack 大约是 1 次/秒（Tier 1），Jira 大约是 1000 次/分钟。超出限制会导致 429 错误甚至临时封禁。**在设计数据管道时要始终考虑速率限制。**

**误区二："Token 过期了重新生成就行"。** 很多 SaaS 平台的 Token 有过期时间（尤其是 OAuth Token），而且某些平台（如 Slack）的旧 Token 会在新 Token 生成后立即失效。**建立 Token 轮换机制或使用长期有效的 API Key 是必要的。**

**误区三:"网页爬取想爬什么就爬什么"。** 除了法律风险外，现代网站有越来越多的反爬措施：Cloudflare 验证、CAPTCHA、User-Agent 检测、IP 信誉评分等。如果目标网站有明确的 API（如大多数 SaaS 平台），优先使用 API 而非爬取。**爬取是最后的手段，不是首选方案。**

**误区四:"API 数据拿来就能用"。** API 返回的数据通常包含大量噪音——HTML 标签、冗余字段、内部编号、空值占位符等。在将 API 数据送入 RAG 索引之前，**数据清洗是不可省略的步骤**。至少要做：去除 HTML 标签、过滤空内容、标准化文本格式、剔除敏感字段。
