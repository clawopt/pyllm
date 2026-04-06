---
title: 索引持久化与增量更新
description: StorageContext 机制、磁盘序列化、增量插入与删除、版本管理与回滚策略
---
# 索引持久化与增量更新

到目前为止，我们每次运行程序都要从头开始：加载文档 → 解析 → 计算嵌入 → 构建索引。对于只有几十个文档的原型来说这不是问题，但对于拥有数千甚至数万文档的生产系统来说，每次重建索引可能需要数十分钟甚至数小时——这是完全不可接受的。

这一节我们来学习 LlamaIndex 的**索引持久化和增量更新**机制，让你的 RAG 系统能够高效地管理不断增长和变化的知识库。

## StorageContext：存储抽象层

LlamaIndex 通过 `StorageContext` 来管理所有与索引相关的持久化数据。理解 StorageContext 是掌握索引持久化的关键。

```python
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 首次构建并保存
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

index.storage_context.persist(persist_dir="./my_index_storage")
print("索引已保存到 ./my_index_storage")
```

执行这段代码后，看看 `./my_index_storage` 目录里有什么：

```
my_index_storage/
├── docstore.json          # 文档存储（原始文本和元数据）
├── index_store.json       # 索引映射（node_id → 映射信息）
├── graph_store.json       # 图存储（节点间的关系）
├── vector_store.json      # 向量存储（embedding 向量）
└── image_store.json       # 图片存储（如果有图片的话）
```

StorageContext 将索引的不同组成部分分离存储，每个部分可以有不同的后端：

```python
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore

storage_context = StorageContext.from_defaults(
    vector_store=ChromaVectorStore(...),     # 向量存 ChromaDB
    docstore=SimpleDocumentStore(),           # 文档存本地 JSON
    index_store=SimpleIndexStore(),           # 索引映射存本地 JSON
    # graph_store 和 image_store 使用默认值
)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
```

这种可插拔的设计让你能够**灵活地组合不同的存储后端**——比如向量存到专业的向量数据库（Chroma/Qdrant/Pinecone），而文档内容和索引映射存到本地文件系统或对象存储（S3）中。

## 加载已保存的索引

保存之后，下次启动程序时可以直接从磁盘加载，无需重新构建：

```python
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(
    persist_dir="./my_index_storage"
)

index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

response = query_engine.query("公司的退款政策是什么？")
print(response.response)
```

加载过程几乎是瞬时的（取决于存储后端的类型），因为所有耗时的嵌入计算都已在之前完成了。

### 加载时的注意事项

**注意一：嵌入模型必须一致。** 加载索引时使用的嵌入模型必须与创建索引时使用的模型完全相同。如果你当时用的是 `text-embedding-3-small`（1536维），加载时也必须用同样的模型。否则向量维度不匹配会导致错误。

```python
# 加载前确保 Settings 中的嵌入模型与保存时一致
from llama_index.embeddings.openai import OpenAIEmbedding
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

storage_context = StorageContext.from_defaults(persist_dir="./my_index_storage")
index = load_index_from_storage(storage_context)
```

**注意二：版本兼容性。** LlamaIndex 的存储格式在不同版本之间可能发生变化。如果你升级了 LlamaIndex 版本，旧版本的索引文件可能无法直接加载。解决方法是先用旧版本加载，然后用新版本重新保存（或在代码中做格式转换）。

## 增量更新：插入新文档

最常见的变更操作是新文档加入知识库。与其重新构建整个索引，不如只对新文档做嵌入并插入现有索引：

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import StorageContext, load_index_from_storage

# 1. 加载已有索引
storage_context = StorageContext.from_defaults(
    persist_dir="./my_index_storage"
)
index = load_index_from_storage(storage_context)

# 2. 准备新文档
new_documents = SimpleDirectoryReader("./new_data").load_data()
print(f"新增 {len(new_documents)} 个文档")

# 3. 逐个插入（或批量插入）
for doc in new_documents:
    index.insert(doc)  # 自动完成: 解析 → 嵌入 → 存入向量存储

# 4. 持久化更新后的索引
index.storage_context.persist(persist_dir="./my_index_storage")
print("增量更新完成")
```

`insert()` 方法做的事情和 `from_documents()` 中的处理是一样的——解析文档、计算嵌入、存入向量存储——但它**只处理新增的文档**，不会动已有的数据。

### 批量插入 vs 逐个插入

对于大量新文档，批量插入效率更高：

```python
# 方式一：逐个插入（简单但较慢）
for doc in new_documents:
    index.insert(doc)

# 方式二：批量插入（更快，减少 API 调用开销）
for doc in new_documents:
    index.insert(doc, show_progress=True)  # 显示进度条

# 方式三：对于特别大的批量，考虑分批处理
BATCH_SIZE = 100
for i in range(0, len(new_documents), BATCH_SIZE):
    batch = new_documents[i:i + BATCH_SIZE]
    for doc in batch:
        index.insert(doc)
    print(f"已完成 {min(i+BATCH_SIZE, len(new_documents))}/{len(new_documents)}")
```

### 增量更新的性能考量

增量更新的耗时主要取决于两个因素：
1. **新文档的数量** —— 每个新文档都需要计算 embedding
2. **向量存储后端的写入性能** —— Chroma 本地写入很快，远程 Qdrant/Pinecone 取决于网络延迟

一般来说，新增 100 个文档的增量更新只需几秒钟；新增 10000 个文档可能需要几分钟。

## 删除过时文档

除了新增，删除同样重要——过时的文档应该及时从索引中移除，以免影响检索质量。

```python
# 删除指定文档（通过 document_id）
index.delete_ref_doc("document-id-uuid-here")

# 或者通过 Node 删除
index.delete_nodes(["node-id-1", "node-id-2"])

# 别忘了持久化！
index.storage_context.persist(persist_dir="./my_index_storage")
```

### 删除操作的注意事项

**注意：删除是基于引用的。** `delete_ref_doc()` 删除的是一个 Document 及其衍生的所有 Nodes。如果你只想删除某个特定的 Node 而保留其他来自同一 Document 的 Nodes，使用 `delete_nodes()`。

**注意：删除后向量空间可能出现"空洞"。** 大多数向量存储后端（包括 Chroma、Qdrant）不会立即回收被删除向量占用的空间。对于频繁增删的场景，定期执行压缩（compaction）或重建索引是有必要的。

## 更新已有文档的内容

如果一篇文档的内容发生了修改（而不是简单的增删），你需要先删除旧版本再插入新版本：

```python
def update_document(index, old_doc_id, new_content):
    """更新文档：先删后加"""
    # 1. 删除旧版本
    try:
        index.delete_ref_doc(old_doc_id)
        print(f"已删除旧文档: {old_doc_id}")
    except Exception as e:
        print(f"删除失败（可能不存在）: {e}")

    # 2. 插入新版本
    new_doc = Document(text=new_content, metadata={"id": old_doc_id})
    index.insert(new_doc)
    print(f"已插入新版本")

    # 3. 持久化
    index.storage_context.persist()


update_document(index, "doc-001", "更新后的文档内容...")
```

LlamaIndex 目前没有原生的"in-place update"操作（即直接修改已有 Node 的内容而不改变其 ID），所以"先删后加"是标准做法。

## 自动化同步管道

在生产环境中，手动调用 insert/delete 是不现实的。你需要建立自动化的同步管道：

```python
import os
import time
import hashlib
from pathlib import Path
from datetime import datetime
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader

SYNC_STATE_FILE = "./sync_state.json"
INDEX_DIR = "./my_index_storage"
DATA_DIR = "./data"


def compute_file_hash(filepath):
    """计算文件的 MD5 哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_sync_state():
    """加载上次同步的状态"""
    if os.path.exists(SYNC_STATE_FILE):
        with open(SYNC_STATE_FILE) as f:
            return json.load(f)
    return {"files": {}, "last_sync": None}


def save_sync_state(state):
    """保存同步状态"""
    state["last_sync"] = datetime.now().isoformat()
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def sync_index():
    """执行增量同步"""
    print("=" * 50)
    print("开始同步...")
    print("=" * 50)

    state = load_sync_state()
    known_files = state.get("files", {})

    current_files = {}
    for filepath in Path(DATA_DIR).rglob("*"):
        if filepath.is_file():
            key = str(filepath)
            current_files[key] = {
                "hash": compute_file_hash(filepath),
                "mtime": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
            }

    # 加载已有索引
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)

    added = 0
    updated = 0
    deleted = 0

    for filepath, info in current_files.items():
        if filepath not in known_files:
            # 新文件 → 插入
            reader = SimpleDirectoryReader(
                input_dir=os.path.dirname(filepath),
                required_exts=[os.path.splitext(filepath)[1]],
            )
            docs = reader.load_data()
            for doc in docs:
                index.insert(doc)
            added += 1
            print(f"  [+] 新增: {filepath}")

        elif known_files[filepath]["hash"] != info["hash"]:
            # 文件变化 → 先删后加
            file_name = os.path.basename(filepath)
            ref_id = known_files[filepath].get("ref_id", file_name)
            try:
                index.delete_ref_doc(ref_id)
            except Exception:
                pass
            reader = SimpleDirectoryReader(input_dir=os.path.dirname(filepath))
            docs = reader.load_data()
            for doc in docs:
                index.insert(doc)
            updated += 1
            print(f"  [~] 更新: {filepath}")

    # 检查已删除的文件
    for filepath in list(known_files.keys()):
        if filepath not in current_files:
            try:
                index.delete_ref_doc(known_files[filepath].get("ref_id"))
                deleted += 1
                print(f"  [-] 删除: {filepath}")
            except Exception:
                pass

    # 持久化
    index.storage_context.persist(persist_dir=INDEX_DIR)

    state["files"] = {
        fp: {**info, "ref_id": os.path.basename(fp)}
        for fp, info in current_files.items()
    }
    save_sync_state(state)

    print(f"\n同步完成: +{added} ~{updated} -{deleted}")
    return added + updated + deleted > 0


if __name__ == "__main__":
    sync_index()
```

这个同步脚本实现了完整的增量同步逻辑：
1. **检测新文件**（不在已知列表中的文件）
2. **检测变更文件**（哈希值变化的文件）
3. **检测删除文件**（在已知列表中但不再存在的文件）
4. **执行对应的 insert/update/delete 操作**
5. **更新同步状态**供下次使用

你可以通过 cron 定时任务（Linux）或 Task Scheduler（Windows）来定期执行这个脚本：

```bash
# 每小时执行一次同步
0 * * * * cd /path/to/project && python sync_index.py >> sync.log 2>&1
```

## 版本管理与回滚

对于关键的知识库，你可能还需要版本管理能力——在某次更新后发现质量下降时能够回滚到之前的版本。

```python
import shutil
from datetime import datetime


def backup_index(index_dir, backup_dir=None):
    """创建索引的备份"""
    if backup_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{index_dir}_backup_{timestamp}"

    shutil.copytree(index_dir, backup_dir)
    print(f"索引已备份到: {backup_dir}")
    return backup_dir


def restore_index(backup_dir, index_dir):
    """从备份恢复索引"""
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    shutil.copytree(backup_dir, index_dir)
    print(f"索引已从备份恢复: {backup_dir}")


# 在每次重大更新前创建备份
backup_path = backup_index("./my_index_storage")

# 执行更新...
sync_index()

# 如果发现问题，回滚
# restore_index(backup_path, "./my_index_storage")
```

对于更高要求的场景，可以考虑将索引备份到对象存储（如 AWS S3），这样即使本地数据丢失也能恢复。

## 常见误区

**误区一:"persist 之后就可以不管了"。** persist 只是做了序列化存储，不代表索引永远不会变。当底层的文档、嵌入模型、或 LlamaIndex 版本发生变化时，保存的索引可能会变得不一致或不兼容。**定期验证索引的可加载性和查询结果的正确性是必要的。**

**误区二:"insert 很快所以可以频繁调用"。** 每次 insert 都涉及嵌入计算和向量存储写入。虽然比全量重建快得多，但如果在高频更新的场景下（如每分钟都有新文档），累积的开销也不容忽视。考虑**批量收集变更后定时批量提交**，而非实时逐条插入。

**误区三:"删除了文档就万事大吉了"。** 如前面提到的，很多向量存储后端不会立即回收被删除向量占用的空间。长期频繁增删后，存储占用可能持续增长。**定期重建索引（如每周一次全量 rebuild）是保持索引健康的有效手段。**

**误区四:"增量更新和全量重建只能二选一"。** 最佳实践是结合两者：日常使用增量更新保持索引的时效性，定期（如每周或每月）做一次全量重建来清理碎片和修正可能的漂移。
