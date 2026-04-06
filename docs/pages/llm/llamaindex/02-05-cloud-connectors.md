---
title: 云服务连接器：AWS S3 / Google Drive / OneDrive / Notion DB
description: 主流云存储平台的数据接入、认证配置、大文件处理、权限管理
---
# 云服务连接器：AWS S3 / Google Drive / OneDrive / Notion DB

随着企业上云的普及，越来越多的知识资产不再存储在本地文件服务器或自建数据库中，而是托管在云服务上：设计稿在 AWS S3 的 bucket 里、技术文档在 Google Drive 的共享文件夹中、合同文件在 Microsoft OneDrive 上、项目数据在 Notion Database 中。这些云服务虽然各自提供了 Web 界面和官方客户端，但要让 RAG 系统自动从中获取数据，就需要通过它们的 API 进行程序化访问。

LlamaIndex 为各大主流云服务平台都提供了专门的连接器，让你用统一的模式接入这些分散在云端的数据源。

## AWS S3 连接器

AWS S3（Simple Storage Service）是最广泛使用的对象存储服务，企业中的备份文件、日志归档、文档库、模型权重等大量数据都可能存在 S3 中。

```bash
pip install llama-index-readers-s3 boto3
```

### 基本用法

```python
import os
from llama_index.readers.s3 import S3Reader

reader = S3Reader(
    bucket="company-knowledge-base",
    key="documents/product_manuals/",  # 可以是具体文件或前缀（目录）
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="ap-northeast-1",
)

documents = reader.load_data()
print(f"从 S3 加载了 {len(documents)} 个文档")
```

### 使用 IAM 角色（推荐用于 EC2 / Lambda / EKS）

如果你的 R 应用运行在 AWS 的计算资源（EC2、Lambda、EKS）上，**不需要硬编码 Access Key**——通过 IAM 角色自动获取凭证是更安全的做法：

```python
from llama_index.readers.s3 import S3Reader

# 不传密钥参数，boto3 会自动从 IAM 角色获取凭证
reader = S3Reader(
    bucket="knowledge-base",
    key="internal_docs/",
    # aws_access_key_id 和 aws_secret_access_key 都不填
)
documents = reader.load_data()
```

这种方式的好处是：
- **无需管理密钥**：没有泄露风险
- **自动轮换**：AWS 自动处理临时凭证
- **最小权限原则**：IAM 角色可以精确控制能访问哪些 bucket 和前缀

### S3 的高级操作

**列出并加载特定类型的文件：**

```python
import boto3

s3 = boto3.client("s3")
response = s3.list_objects_v2(
    Bucket="company-kb",
    Prefix="docs/2025/",
)

# 过滤出 PDF 文件
pdf_keys = [
    obj["Key"] for obj in response.get("Contents", [])
    if obj["Key"].endswith(".pdf")
]

from llama_index.readers.s3 import S3Reader
for key in pdf_keys:
    reader = S3Reader(bucket="company-kb", key=key)
    docs = reader.load_data()
    all_documents.extend(docs)
```

**使用 S3 Select 做服务端过滤：**

如果存储的是 CSV 或 JSON 文件，可以用 S3 Select 在服务端做过滤，只下载需要的行，大幅减少传输量：

```python
import boto3

response = s3.select_object_content(
    Bucket="data-bucket",
    Key="large_dataset.csv",
    ExpressionType='SQL',
    Expression="SELECT * FROM S3Object WHERE category = 'tech'",
    InputSerialization={'CSV': {"FileHeaderInfo": "Use"}},
    OutputSerialization={'CSV': {}},
)

for event in response["Payload"]:
    if "Records" in event:
        print(event["Records"]["Payload"].decode())
```

## Google Drive 连接器

Google Drive 是很多团队（尤其是使用 Google Workspace 的团队）的默认文档存储位置。Google Docs、Sheets、Slides 文件都存储在 Google Drive 中。

```bash
pip install llama-index-readers-google
```

### Google API 认证配置

Google API 的认证比其他平台复杂一些，需要以下步骤：

1. 在 [Google Cloud Console](https://console.cloud.google.com/) 创建项目
2. 启用 Google Drive API
3. 创建 OAuth 2.0 凭证（选择"桌面应用"类型）
4. 下载 `credentials.json` 文件
5. 首次运行时会打开浏览器要求授权，授权后生成 `token.json` 用于后续调用

```python
from llama_index.readers.google import GoogleDriveReader

reader = GoogleDriveReader()

# 方式一：按文件夹 ID 加载
documents = reader.load_data(
    folder_ids=["your-folder-id"],
    # 可选：递归子文件夹
    recursive=True,
)

# 方式二：按文件 ID 列表加载
documents = reader.load_data(
    file_ids=["file-id-1", "file-id-2", "file-id-3"],
)

print(f"从 Google Drive 加载了 {len(documents)} 个文件")
```

### Google Docs 的特殊处理

Google Drive 中的 Google Docs（`.gdoc`）、Google Sheets（`.gsheet`）、Google Slides（`.gslide`）文件与普通文件不同——它们不是原始的二进制文件，而是 Google 的专有格式。GoogleDriveReader 会自动将它们转换为可读的文本格式：

- **Google Docs** → 提取文本内容，保留基本格式（标题、列表等）
- **Google Sheets** → 将表格转为类似 CSV 的文本表示
- **Google Slides** → 提取每页的文本内容
- **PDF / DOCX 等** → 直接下载并解析

### 服务账号（Service Account）方式

对于服务器端应用（无浏览器交互的场景），应该使用 Service Account 而非 OAuth：

```python
from google.oauth2 import service_account
from llama_index.readers.google import GoogleDriveReader

credentials = service_account.Credentials.from_service_account_file(
    "./service-account-key.json",
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)

reader = GoogleDriveReader()
# 需要在底层设置 credentials
# 具体方式取决于 LlamaIndex 版本的实现细节
```

使用 Service Account 时，还需要确保 Service Account 对目标文件夹有访问权限——在 Google Drive 的共享设置中将文件夹共享给 Service Account 的邮箱地址。

## Microsoft OneDrive 连接器

对于使用 Microsoft 365（原 Office 365）的企业来说，OneDrive 和 SharePoint 是核心的文档协作平台。

```bash
pip install llama-index-readers-onedrive
```

```python
from llama_index.readers.onedrive import OneDriveReader

reader = OneDriveReader(
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET"),
    tenant_id=os.getenv("AZURE_TENANT_ID"),
)

documents = reader.load_data(
    folder_id="shared-folder-id",  # SharePoint 文件夹或 OneDrive 文件夹
    paths=["/Shared Documents/Contracts"],  # 或用路径表示
)

print(f"OneDrive/SharePoint: {len(documents)} 个文档")
```

OneDrive Reader 的特点：
- 同时支持个人 OneDrive 和 SharePoint Online
- 能解析 Office 文档格式（Word、Excel、PowerPoint）
- 支持 Microsoft Graph API 的完整权限控制

### Azure AD 认证详解

Microsoft 365 的认证基于 Azure Active Directory（现在叫 Microsoft Entra ID）。认证流程如下：

```
应用请求访问 OneDrive
       │
       ▼
Azure AD 验证 Client ID + Client Secret + Tenant ID
       │
       ▼
返回 Access Token（JWT 格式，有效期通常 1 小时）
       │
       ▼
用 Access Token 调用 Microsoft Graph API
       │
       ▼
获取文件列表和内容
```

Token 有过期时间（通常是 1 小时），长时间运行的程序需要实现 Token 自动刷新机制。LlamaIndex 的 OneDriveReader 内部通常会处理 Token 缓存和刷新，但你需要确保 `client_secret` 的安全性。

## Notion Database 作为数据源

虽然 Notion 在上一节已经作为 API 连接器介绍过，但 Notion Database 值得特别提及——因为它经常被用作轻量的结构化数据库来替代传统的数据库系统。

```python
from llama_index.readers.notion import NotionPageReader

reader = NotionPageReader(
    integration_token=os.getenv("NOTION_TOKEN"),
)

# Notion Database 本质上是"带属性的行集合"
documents = reader.load_data(
    database_id="database-uuid-here",
    # 使用 Notion API 的过滤语法
    request_property_filter={
        "and": [
            {"property": "Status", "status": {"equals": "Done"}},
            {"property": "Priority", "select": {"equals": "High"}},
            {
                "property": "Date",
                "date": {"on_or_after": "2025-01-01"},
            },
        ],
    },
    # 排序
    sort_query=[
        {"property": "Date", "direction": "descending"},
    ],
)
```

Notion Database 返回的 Document 结构很有意思——每个数据库行变成一个 Document，行的每个属性（Property）自动映射为 metadata：

```python
doc = documents[0]
print(doc.metadata)
# 输出类似：
# {
#   'Name': 'Q1 产品路线图',
#   'Status': 'Done',
#   'Priority': 'High',
#   'Owner': '张三',
#   'Date': '2025-03-15',
#   'Tags': ['产品', '规划'],
#   'id': 'row-uuid',
# }
print(doc.text)  # 行的内容/描述部分
```

这种丰富的 metadata 使得 Notion Database 特别适合做**元数据过滤查询**——比如"只搜索张三负责的高优先级已完成任务"。

## 云服务连接器的通用注意事项

### 大文件处理

云存储中的文件可能非常大（几百 MB 甚至几 GB），直接下载到内存中处理会导致 OOM。解决方案：

```python
# 流式下载 + 分块处理
import boto3

s3 = boto3.client("s3")

obj = s3.get_object(Bucket="big-files", Key="huge_document.pdf")
body = obj["Body"]

chunk_size = 10 * 1024 * 1024  # 10MB chunks
while chunk := body.read(chunk_size):
    process_chunk(chunk)  # 逐块处理
```

### 网络延迟与超时

云 API 调用涉及网络传输，延迟远高于本地文件读取。合理的超时配置很重要：

```python
from llama_index.readers.s3 import S3Reader

reader = S3Reader(
    bucket="my-bucket",
    key="docs/",
    # 如果底层支持超时配置
    # timeout=60,  # 60 秒超时
)
```

如果遇到频繁的超时错误，检查以下几点：
1. 你的网络到云服务区域的延迟是否过高（跨区域访问会很慢）
2. 文件是否过大导致下载超时
3. 是否触发了云服务的速率限制

### 成本意识

云服务的 API 调用和数据传输都是收费的：

| 操作 | AWS S3 | Google Drive | OneDrive |
|------|--------|-------------|----------|
| GET 请求 | $0.0004/千次 | 免费（有配额） | 包含在 M365 订阅 |
| 数据传出 | $0.09/GB | 免费（有配额） | 取决于订阅计划 |
| 存储 | $0.023/GB/月 | 免费 15GB | 取决于订阅 |

**成本优化建议：**
- 避免频繁地全量重新加载——做好缓存
- 选择离你的计算资源最近的区域
- 对于 S3，考虑使用 S3 Transfer Acceleration
- 监控 API 调用量，设置预算告警

## 多云环境的统一接入

有些企业同时使用多个云服务——文档在 Google Drive、备份在 AWS S3、合同在 OneDrive。LlamaIndex 的统一接口让同时从多个云源加载数据变得很简单：

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.s3 import S3Reader
from llama_index.readers.google import GoogleDriveReader
from llama_index.readers.onedrive import OneDriveReader

all_documents = []

# 1. 从 AWS S3 加载产品文档
s3_reader = S3Reader(bucket="kb-prod", key="manuals/")
all_documents.extend(s3_reader.load_data())

# 2. 从 Google Drive 加载团队 Wiki
gd_reader = GoogleDriveReader()
all_documents.extend(gd_reader.load_data(folder_ids=["wiki-folder-id"]))

# 3. 从 OneDrive 加载合同文件
od_reader = OneDriveReader(...)
all_documents.extend(od_reader.load_data(folder_id="contracts"))

# 4. 统一建索引
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(all_documents)

print(f"总共从 {3} 个云源加载了 {len(all_documents)} 个文档")
```

每个云源的 Document 都带有不同的 metadata 标识来源（如 `source: s3://...`、`source: googledrive://...`），这使得用户提问后不仅能得到答案，还能知道答案来自哪个云平台的哪个文件。

## 常见误区

**误区一："云服务和本地文件一样快"。** 不可能一样快。即使是在同一区域的 AWS EC2 访问 S3，也有毫秒级的网络延迟；跨区域或跨云服务商的延迟更高。在设计 RAG 系统时要考虑这个因素——数据加载阶段可能会比预期慢很多倍。

**误区二："把所有云文件一次性拉下来就行"。** 对于包含数千个文件的云存储，全量下载可能需要数小时甚至更久。而且每次重启都要重新下载？不可接受。**建立增量同步机制**（只下载新增或变更的文件）是生产环境的必备能力。

**误区三:"云 API 凭证可以随便分享给团队成员"。** 绝对不行。云服务凭证等同于 root 权限——拥有它的人可以做任何操作（包括删除所有数据、产生巨额账单）。使用最小权限原则、定期轮换凭证、审计凭证使用情况是基本的安全实践。
