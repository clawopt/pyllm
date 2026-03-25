# 数据序列化

数据序列化是将程序中的数据结构转换为可以存储或传输的格式的过程。反序列化则是将存储或传输的数据恢复为原始数据结构。Python提供了多种序列化方案，每种都有其适用场景和权衡。

## 为什么需要序列化

程序运行时，数据存在于内存中的数据结构里。程序结束后，内存被释放，数据随之消失。如果希望持久化保存数据，或者在不同程序之间共享数据，就需要将数据转换为可以存储的格式。

常见的序列化应用场景包括：配置文件读写、缓存系统、数据持久化、API数据交换、服务间通信等。

## JSON：轻量级数据交换

JSON是当前最流行的数据交换格式，语法简洁、人类可读、与语言无关。Python的`json`模块提供了JSON的编解码功能。

### 基本使用

```python
import json

data = {
    "name": "Python",
    "version": 3.10,
    "features": ["dynamic", "interpreted", "object-oriented"],
    "popular": True
}

json_str = json.dumps(data)
print(json_str)

parsed = json.loads(json_str)
print(parsed)
```

`json.dumps()`将Python对象转换为JSON字符串，`json.loads()`执行反向操作。JSON支持的数据类型有限：字符串、数字、布尔值、null、数组和对象。

### 文件操作

处理JSON文件时，直接读写字符串容易出错：

```python
data = {"name": "project", "stars": 1000}

with open('config.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

with open('config.json', 'r', encoding='utf-8') as f:
    loaded = json.load(f)
```

`json.dump()`直接将对象写入文件，`json.load()`直接从文件读取并解析为Python对象。`indent=2`参数使输出的JSON格式化，便于阅读。

### 编码控制

`json.dump()`接受多个参数控制输出格式：

```python
data = {"name": "你好", "value": 3.14}

json.dumps(data, ensure_ascii=False)
json.dumps(data, indent=4, sort_keys=True)
```

`ensure_ascii=False`保留Unicode字符而不将其转义为`\uXXXX`形式。`sort_keys=True`对输出的键进行排序，便于版本控制时比较差异。

### 自定义类型

JSON不能直接序列化日期、正则表达式等Python特有类型。可以使用`default`参数自定义序列化逻辑：

```python
from datetime import datetime
import json

def json_serializer(obj):
    if isinstance(obj, datetime):
        return {"__type__": "datetime", "value": obj.isoformat()}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

data = {"created": datetime(2024, 1, 15, 10, 30)}

json_str = json.dumps(data, default=json_serializer)
```

反序列化时需要配合`object_hook`参数：

```python
def json_deserializer(obj):
    if obj.get("__type__") == "datetime":
        return datetime.fromisoformat(obj["value"])
    return obj

parsed = json.loads(json_str, object_hook=json_deserializer)
```

## pickle：Python对象序列化

`pickle`是Python特有的序列化模块，可以保存几乎所有Python对象，包括自定义类的实例：

```python
import pickle

data = {"name": "test", "values": [1, 2, 3]}

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

with open('data.pkl', 'rb') as f:
    loaded = pickle.load(f)
```

`pickle`生成的格式是Python特有的，只有Python程序能读取。这意味着pickle文件不能用于与其他编程语言的数据交换。

### 安全风险

使用`pickle`存在严重的安全风险。反序列化时，`pickle`可以执行任意代码：

```python
import pickle

class Dangerous:
    def __reduce__(self):
        return (os.system, ('whoami',))

malicious = pickle.dumps(Dangerous())
pickle.loads(malicious)  # 执行whoami命令
```

永远不要反序列化来源不明的pickle数据。在需要处理不可信数据的场景，应该使用JSON或其他安全的序列化格式。

如果必须存储复杂对象，可以考虑使用`pickle`配合签名或加密，或者使用`dill`这样的第三方库，它们提供了更细粒度的控制。

### 协议版本

`pickle`有多个协议版本，版本越高，序列化效率越好，但兼容性越差：

```python
pickle.dumps(data, protocol=0)  # 文本格式，兼容性好
pickle.dumps(data, protocol=1)  # 较旧的二进制格式
pickle.dumps(data, protocol=2)  # Python 2.3引入
pickle.dumps(data, protocol=3)  # Python 3.0引入，不兼容Python 2
pickle.dumps(data, protocol=4)  # Python 3.4引入，更多优化
pickle.dumps(data, protocol=5)  # Python 3.8引入，支持out-of-band数据
```

通常不需要指定协议，`pickle`会自动选择最高支持的协议。

## JSON与Pickle的选择

两种序列化方式各有优势，选择取决于具体需求：

| 特性 | JSON | Pickle |
|------|------|--------|
| 数据格式 | 文本 | 二进制 |
| 可读性 | 高 | 低 |
| Python对象 | 基础类型 | 几乎所有类型 |
| 跨语言 | 是 | 否 |
| 安全性 | 高 | 低 |
| 文件大小 | 大 | 小 |

JSON适合配置文件、数据交换、API通信等需要可读性或跨语言兼容的场景。Pickle适合Python程序之间的数据持久化、缓存等场景。

## 其他序列化方案

### MessagePack

MessagePack是一种高效的二进制格式，比JSON更紧凑，编解码速度更快：

```python
import msgpack

data = {"name": "python", "version": 3.10}
packed = msgpack.packb(data)
print(len(packed))

unpacked = msgpack.unpackb(packed, raw=False)
```

`msgpack`需要额外安装，但性能通常优于JSON和pickle。

### ProtoBuf与Thrift

Protocol Buffers和Thrift是Google和Facebook推出的序列化框架，采用IDL（接口定义语言）定义数据结构，生成强类型的序列化代码。它们主要用于高性能服务间通信，适合对性能和数据体积敏感的场景。

### YAML

YAML是比JSON更人性化的数据序列化格式，特别适合配置文件：

```python
import yaml

data = {"name": "project", "version": "1.0"}

with open('config.yaml', 'w') as f:
    yaml.dump(data, f, allow_unicode=True)

with open('config.yaml', 'r') as f:
    loaded = yaml.safe_load(f)
```

YAML的语法比JSON更宽松，但解析速度通常比JSON慢。`safe_load()`只加载基本的Python数据类型，比`load()`更安全。

## 常见问题

### 中文编码

处理包含中文的JSON时，需要注意编码问题：

```python
import json

data = {"message": "你好世界"}

result = json.dumps(data, ensure_ascii=False)
print(result)  # {"message": "你好世界"}
```

如果不设置`ensure_ascii=False`，中文会被转义为`\uXXXX`形式。

### 浮点数精度

JSON不区分整数和浮点数，也不保证精度：

```python
import json

data = {"value": 1.23456789012345678}
print(json.loads(json.dumps(data)))
```

对于需要精确小数的场景，应该使用`decimal.Decimal`类型，或者考虑其他序列化方案。

### 循环引用

JSON不支持循环引用：

```python
import json

a = {"name": "a"}
b = {"name": "b", "ref": a}
a["ref"] = b

json.dumps(a)  # 抛出异常：ValueError: Circular reference
```

`pickle`支持循环引用，但其他大多数序列化格式都不支持。

## 面试关注点

面试中关于序列化的常见问题包括：JSON和pickle有什么区别？pickle有什么安全风险？如何自定义JSON的序列化行为？

理解序列化的本质很重要：它是有损还是无损的转换过程？对于基础数据类型，序列化应该是完全无损的；对于对象实例，序列化可能丢失一些运行时信息（如类定义、函数闭包等）。

高级面试题可能涉及：如何实现一个支持循环引用的序列化器？为什么pickle不能跨语言使用？序列化的性能瓶颈在哪里？
