# 01-04 模型管理基础

## 你电脑上现在有哪些模型？

上一节我们用 `ollama run qwen2.5:1.5b` 运行了第一个模型。Ollama 在运行前会自动把模型文件下载到本地，所以即使你关掉了终端，下次再 `ollama run` 同一个模型时就不需要重新下载了。那么，这些模型文件到底存在哪里？占了多少空间？怎么查看和管理？这一节就来回答这些问题。

### `ollama list`：查看已安装的模型

```bash
$ ollama list
NAME                ID              SIZE      MODIFIED
qwen2.5:1.5b         abc123def456   919 MB    2026-04-08 10:30:00
```

输出包含四列信息：

| 列名 | 含义 | 示例解读 |
|------|------|---------|
| **NAME** | 模型名称（含 tag） | `qwen2.5:1.5b` 表示 Qwen2.5 系列的 1.5B 参数版本 |
| **ID** | 模型的唯一标识符（SHA256 哈希） | 用于精确引用某个特定版本的模型 |
| **SIZE** | 模型文件占用的磁盘空间 | `919 MB` 说明这是一个量化后的小模型 |
| **MODIFIED** | 最后修改时间 | 即最后一次 pull 或 create 的时间 |

如果你还没有运行过任何模型，这个列表会是空的——因为 Ollama 采用的是"按需拉取"策略，不会预装任何模型。

### `ollama pull`：预先下载模型

上一节我们直接用了 `ollama run`，它会在需要时自动拉取模型。但有时候你可能想先把模型下载好，之后再使用——比如在离线环境部署前提前准备、或者在带宽充裕的时段批量下载。

```bash
# 基本用法：下载但不运行
ollama pull qwen2.5:7b

# 下载过程中你会看到进度条：
pulling manifest...
pulling abcdef...
pulling 123456...
verifying sha256: abc...
success
```

`ollama pull` 和 `ollama run` 的区别在于：**pull 只做下载和校验，不会启动模型服务或进入交互模式。** 这意味着：
- pull 完成后不会占用额外的内存（模型文件只是存在磁盘上）
- 可以在任何时候后续通过 `ollama run` 快速启动（因为已经不需要再下载了）
- 适合批量预下载场景

```bash
# 批量下载多个常用模型（脚本方式）
models=(
    "qwen2.5:7b"
    "llama3.1:8b"
    "nomic-embed-text"
)

for model in "${models[@]}"; do
    echo ">>> Pulling $model..."
    ollama pull "$model"
    echo ""
done

echo "All models ready!"
```

### `ollama show`：深入了解一个模型

`list` 只告诉你"有什么"，`show` 则告诉你"它是什么"。这是理解模型能力的最重要命令。

```bash
$ ollama show qwen2.5:1.5b
Model
	architecture: arm64
	parameters: 1.5B
	context length: 32768
	embedding length: 0
	quantization: Q4_0

License
	MIT

Modelfile
	FROM qwen2.5:1.5b
	PARAMETER temperature 0.8
	PARAMETER top_k 40
	PARAMETER top_p 0.9
```

这里每一行信息都值得理解：

**Model 部分**是模型的硬性规格：
- **architecture**: 模型编译的目标架构（arm64 = Apple Silicon / amd64 = x86_64 Linux/Windows）
- **parameters**: 参数量（1.5B = 15 亿参数）。参数量越大通常能力越强但也越慢
- **context length**: 上下文窗口长度（32768 = 32K tokens）。这决定了模型一次能处理多长的输入。注意：这里的值是模型理论上限，实际可用还受你的内存限制
- **embedding length**: 如果非零说明这个模型可以生成向量嵌入（用于 RAG 场景）
- **quantization**: 量化级别。Q4_0 表示 4-bit 量化（legacy 格式），Q4_K_M 是更新的 K-quant 方法（效果更好）

**Modelfile 部分**展示了创建这个模型时使用的配置（如果是官方原版模型则显示默认配置）。

`show` 还有一个非常实用的子命令：

```bash
# 查看模型的完整 Modelfile（用于学习或复制）
ollama show qwen2.5:1.5b --modelfile

# 输出:
# FROM qwen2.5:1.5b
# PARAMETER temperature 0.8
# PARAMETER top_k 40
# ...

# 查看模型的系统提示词（如果有的话）
ollama show qwen2.5:1.5b --system

# 查看模型支持的功能标签
ollama show qwen2.5:1.5b --template
```

这些子命令在第四章深入 Modelfile 时会频繁用到。

## 模型存储结构：`~/.ollama/models/`

所有通过 Ollama 下载或创建的模型都存放在用户主目录下的 `.ollama/models/` 文件夹中。了解它的内部结构有助于你进行磁盘空间管理和问题排查。

```
~/.ollama/models/
├── blobs/
│   ├── sha256-abc123...   ← 模型权重文件（GGUF 格式的二进制数据）
│   ├── sha256-def456...   ← 另一个 blob（可能是 tokenizer 或 adapter）
│   └── ...
├── manifests/
│   └── registry.ollama.ai/library/qwen2.5:1.5b
│       └── latest        ← 模型清单文件（记录了依赖哪些 blobs + 配置）
└── ...
```

核心概念就两个：

**Blobs**：实际的二进制数据块。每个模型由一个或多个 blob 组成——通常是模型权重文件（最大的那个），可能还包括 tokenizer 文件、LoRA adapter 等。Blob 的文件名就是其内容的 SHA256 哈希值。

**Manifests**：描述文件。每个 `name:tag` 对应一个 manifest，里面记录了这个模型"由哪些 blob 组成 + 用什么配置运行"。你可以把 manifest 理解为 Docker 镜像中的 manifest.json——它是模型的"配方单"。

这种设计带来的一个重要特性是 **deduplication（去重）**：如果两个不同的模型共享同一个基础权重 blob（比如你基于同一个 7B 模型创建了两个不同 system prompt 的变体），磁盘上只存一份权重文件。

### 磁盘空间管理

```bash
# 查看 Ollama 总共占了多少空间
du -sh ~/.ollama/

# 典型输出示例:
# 12G     /Users/yourname/.ollama/
# （包含了几个大模型后的典型大小）

# 查看每个模型分别占多大
for model in $(ollama list --json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(m['name'])
"); done); do
    # ollama 本身没有 per-model size 命令
    # 但可以通过查看 manifest 关联的 blob 来估算
done

# 更实用的方式：直接看 blobs 目录
du -sh ~/.ollama/models/blobs/* | sort -rh | head -10
# 会列出最大的 10 个 blob 文件及其大小
```

### `ollama rm`：删除不需要的模型

磁盘空间不够时，删除不常用的模型是最直接的释放方式：

```bash
# 删除单个模型
ollama rm qwen2.5:1.5b

# 批量删除（保留正在使用的）
ollama rm llama3.1:8b nomic-embed-text deepseek-v2:16b

# 删除所有模型（慎用！）
ollama rm $(ollama list -q)  # 先列出来确认，再执行删除
```

**重要提醒**：`rm` 操作不可逆。删除后如果再次 `run` 同名模型，需要重新从网络下载。所以在批量清理前建议先确认。

### `ollama cp`：复制/重命名模型

如果你想基于已有模型创建一个新的别名（比如保存了一个经过特殊配置的对话状态），可以用 `cp`：

```bash
# 创建别名（不额外占用磁盘空间，共享底层 blob）
ollama cp qwen2.5:7b my-qwen

# 之后可以用新名字运行
ollama run my-qwen "你好"

# 新模型继承了原始模型的所有能力和配置
# 但可以有自己独立的 Modelfile 修改
```

`cp` 的本质是创建一个新的 manifest 指向相同的 blobs，所以几乎不消耗额外磁盘空间。这在第四章自定义模型时会经常用到。

## 多模型共存与切换

在实际开发中，你很可能需要在多个模型之间切换——比如代码生成用 DeepSeek-Coder、日常问答用 Qwen2.5、文档分析用 Llama 3.1。

### 同时加载多个模型

Ollama 默认允许同时将多个模型加载到内存中（受限于你的总内存大小）：

```bash
# 终端 A：启动模型 A
ollama run qwen2.5:7b

# 另开一个终端 B：启动模型 B
ollama run llama3.1:8b

# 两个模型同时运行，各自独立响应请求
```

API 层面也完全支持——客户端只需在请求中指定不同的 `model` 字段即可：

```python
import requests
import json

API_BASE = "http://localhost:11434"

def ask(model, question):
    resp = requests.post(f"{API_BASE}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    })
    return resp.json()["message"]["content"]

# 同一段代码，切换模型只需要改一个字符串
print(ask("qwen2.5:7b", "什么是 Python 装饰器？"))
print(ask("llama3.1:8b", "What is a decorator in Python?"))
print(ask("deepseek-coder:6.7b-instruct", "# Write a Python decorator for caching"))
```

### 内存竞争与自动卸载

当内存不足以容纳所有活跃模型时，Ollama 会自动把最久未使用的模型从内存中卸载（类似于操作系统的页面置换策略）。这个过程对用户基本透明——下次调用被卸载的模型时会自动重新加载（会有几秒延迟）。

你可以通过以下方式控制这个行为：

```bash
# 设置模型在内存中保持的最长时间（默认 5 分钟）
OLLAMA_KEEP_ALIVE=1h ollama serve
# 设置为 0 表示用完立即卸载（节省内存但每次都要重载）

# 手动触发卸载（不删除文件）
# Ollama 目前没有显式的 unload 命令
# 但可以通过发送空请求让系统回收资源
```

## 模型更新与版本管理

开源模型的迭代速度很快——Llama 3.1 发布几个月后 3.2 就出来了，Qwen2.5 也不断有微调版本。如何保持模型更新？

### Tag 版本管理

Ollama 的模型命名遵循 `<name>:<tag>` 规范，其中 tag 就相当于版本号：

```bash
# 拉取最新版本（latest 是默认 tag）
ollama pull qwen2.5:latest

# 拉取指定版本
ollama pull qwen2.5:7b          # 通常指向最新的 7b 量级版本
ollama pull qwen2.5:7b-instruct  # 带指令微调的版本

# 查看某个模型有哪些可用的 tags
# 访问 https://ollama.com/library/qwen2.5 页面的 Tags 区域
# 或者用 API 查询:
curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
names = set()
for m in data.get('models', []):
    names.add(m['name'].split(':')[0])
for name in sorted(names):
    print(name)
"
```

### 更新已安装的模型

当有新版本发布时：

```bash
# 方法一：先删旧版再拉新版（最干净）
ollama rm qwen2.5:7b
ollama pull qwen2.5:7b  # 自动获取最新版本

# 方法二：强制重新拉取（覆盖本地缓存）
ollama pull qwen2.5:7b  # 默认行为是检查本地已有则跳过
# 如果需要强制刷新，可以先 rm 再 pull
```

**注意**：Ollama 目前没有内置的 `ollama update` 或 `ollama upgrade` 命令。版本更新本质上就是"删除旧 manifest + 下载新 manifest + 复用不变的 blobs"。由于大多数情况下只有 manifest 变了而底层的模型权重 blob 不一定变，所以更新过程通常很快。

## 实用工具脚本汇总

下面是一个综合性的模型管理工具集，可以直接保存使用：

```python
#!/usr/bin/env python3
"""Ollama 模型管理工具箱"""
import subprocess
import json
import shutil
import sys
import os


def cmd(args):
    """执行 ollama 命令并返回结果"""
    result = subprocess.run(
        ["ollama"] + args,
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip(), result.returncode


def list_models():
    """列出所有已安装模型及详细信息"""
    output, code = cmd(["list"])
    if code != 0 or not output:
        print("没有找到已安装的模型")
        return []
    
    try:
        data = json.loads(output)
        models = data.get("models", [])
        
        if not models:
            print("模型列表为空")
            return []
        
        total_size = sum(m.get("size", 0) for m in models)
        
        print(f"\n{'名称':<35} {'大小':>10} {'修改时间':<20}")
        print("-" * 70)
        for m in models:
            name = m["name"]
            size_mb = m.get("size", 0) // (1024 * 1024)
            modified = m.get("modified_at", "N/A")[:19]
            print(f"{name:<35}{size_mb:>10} MB{modified:>20}")
        
        print("-" * 70)
        print(f"总计: {len(models)} 个模型, {total_size // (1024*1024)} MB")
        return models
    except json.JSONDecodeError:
        print(output)  # 非 JSON 格式，直接打印原始输出
        return []


def show_model_detail(model_name):
    """展示模型的完整详情"""
    output, _ = cmd(["show", model_name])
    print(output)


def disk_usage():
    """统计 Ollama 磁盘占用"""
    ollama_dir = os.path.expanduser("~/.ollama")
    if os.path.exists(ollama_dir):
        total_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(ollama_dir)
            for f in files
        )
        size_str = f"{total_size / (1024**3):.2f}" if total_size >= 1024**3 else f"{total_size / (1024**2):.0f}"
        print(f"\n📦 Ollama 目录 ({ollama_dir})")
        print(f"   总大小: {size_str}")
        
        # blobs 目录的大小
        blobs_dir = os.path.join(ollama_dir, "models", "blobs")
        if os.path.exists(blobs_dir):
            blob_size = sum(
                os.path.getsize(os.path.join(blobs_dir, f))
                for f in os.listdir(blobs_dir)
            )
            print(f"   Blobs (模型权重): {blob_size / (1024**2):.0f} MB")
    else:
        print("Ollama 目录不存在")


def cleanup_unused(days=30):
    """清理 N 天未使用的模型（仅列出，不自动删除）"""
    models = list_models()
    now = time.time()
    
    print(f"\n🧹 可能闲置的模型（{days} 天内未修改）：\n")
    
    found_any = False
    for m in models:
        modified = m.get("modified_at", "")
        if modified and modified != "N/A":
            # 简化的日期解析（ISO格式）
            from datetime import datetime
            mod_time = datetime.fromisoformat(modified.replace("Z", "+00:00"))
            days_since = (datetime.now() - mod_time).days
            
            if days_since > days:
                found_any = True
                size_mb = m.get("size", 0) // (1024 * 1024)
                print(f"  ⚠️  {m['name']:<35}{size_mb:>6} MB  ({days_since} 天前)")
    
    if not found_any:
        print("  ✅ 所有模型都在近期使用过")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama 模型管理工具箱")
    parser.add_argument("action", choices=["list", "detail", "disk", "cleanup"],
                       help="要执行的操作")
    parser.add_argument("--model", "-m", help="目标模型名称（detail 模式必需）")
    parser.add_argument("--days", "-d", type=int, default=30,
                       help="cleanup 模式的天数阈值（默认 30）")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_models()
    elif args.action == "detail":
        if not args.model:
            print("请用 -m 指定模型名称，或先用 list 查看可用模型")
            sys.exit(1)
        show_model_detail(args.model)
    elif args.action == "disk":
        disk_usage()
    elif args.action == "cleanup":
        cleanup_unused(args.days)
```

使用方式：

```bash
# 查看所有模型
python3 ollama_toolkit.py list

# 查看磁盘占用
python3 ollama_toolkit.py disk

# 查看某模型详情
python3 ollama_toolkit.py detail -m qwen2.5:7b

# 清理闲置模型（只列出不删除）
python3 ollama_toolkit.py cleanup -d 60
```

---

到这里，第一章的全部内容已经完成。你已经了解了为什么需要本地运行大模型、如何安装配置 Ollama、如何运行第一个模型并与之交互式对话、以及如何管理本地存储的模型文件。下一章我们将正式进入 API 编程的世界——学会用代码而不是终端来控制 Ollama。
