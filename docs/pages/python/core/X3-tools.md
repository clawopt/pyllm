---
title: 工具模块
---

# 工具模块

这一章介绍Python中常用的工具模块，包括日志、哈希、编码和类型提示。这些模块在日常开发中非常实用。

## logging模块

logging是Python的标准日志模块，比print更适合生产环境。

### 基本用法

```python
import logging

logging.debug("Debug message")  # 调试信息
logging.info("Info message")   # 一般信息
logging.warning("Warning message")  # 警告
logging.error("Error message")  # 错误
logging.critical("Critical message")  # 严重错误
```

默认级别是WARNING，所以debug和info不会显示。

### 配置级别

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug("Now you see it")
```

basicConfig应该在程序入口调用一次。

### 日志格式

```python
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
```

常用格式符：

| 符号 | 含义 |
|------|------|
| %(name)s | logger名称 |
| %(levelname)s | 日志级别 |
| %(message)s | 日志消息 |
| %(asctime)s | 时间 |
| %(filename)s | 文件名 |
| %(lineno)d | 行号 |

### 输出到文件

```python
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Logger对象

```python
logger = logging.getLogger(__name__)

logger.info("Info message")
logger.warning("Warning message")
```

使用`__name__`可以让日志显示模块名，便于追踪。

### 记录器层级

```python
logger = logging.getLogger("myapp")
child_logger = logging.getLogger("myapp.database")

child_logger.info("Using database")  # logger name: myapp.database
```

层级用点号分隔，child继承parent的配置。

### Handler

Handler决定日志输出到哪里：

```python
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

可以添加多个handler，分别处理不同级别的日志。

## hashlib模块

hashlib提供哈希函数，用于数据完整性校验和密码存储。

### 基本用法

```python
import hashlib

result = hashlib.md5(b"hello")
print(result.hexdigest())  # 5d41402abc4b2a76b9719d911017c592

result = hashlib.sha1(b"hello")
print(result.hexdigest())  # aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d

result = hashlib.sha256(b"hello")
print(result.hexdigest())  # 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
```

创建哈希对象，update添加数据，hexdigest获取十六进制字符串。

### 分段哈希

处理大文件：

```python
import hashlib

hasher = hashlib.sha256()
with open('largefile.bin', 'rb') as f:
    for chunk in iter(lambda: f.read(8192), b''):
        hasher.update(chunk)
print(hasher.hexdigest())
```

分块读取文件，避免一次性加载到内存。

### Unicode字符串

字符串需要编码：

```python
import hashlib

s = "你好"
print(hashlib.md5(s.encode('utf-8')).hexdigest())
```

### 口令哈希

bcrypt和argon2比md5和sha系列更适合密码存储：

```python
import bcrypt

password = b"secretpassword"
salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(password, salt)
print(hashed)

print(bcrypt.checkpw(password, hashed))  # True
```

MD5/SHA系列太快，不适合密码。bcrypt设计用于密码，慢哈希抗暴力破解。

## base64模块

base64将二进制数据编码为可打印ASCII字符。

### 编码解码

```python
import base64

data = b"Hello, World!"

encoded = base64.b64encode(data)
print(encoded)  # b'SGVsbG8sIFdvcmxkIQ=='

decoded = base64.b64decode(encoded)
print(decoded)  # b'Hello, World!'
```

### URL安全变体

标准base64可能包含+、/、=，URL中需要转义：

```python
import base64

data = b"Hello?"

encoded = base64.urlsafe_b64encode(data)
print(encoded)  # b'SGVsbG8_'
```

`urlsafe_b64encode`用`-`和`_`替代`+`和`/`。

### 常见应用

图片Base64：

```python
import base64

with open("image.png", "rb") as f:
    data = f.read()
    encoded = base64.b64encode(data)

print(f"data:image/png;base64,{encoded.decode()}")
```

JWT令牌：

```python
import base64
import json

header = {"alg": "HS256", "typ": "JWT"}
payload = {"user_id": 123}

header_encoded = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
payload_encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')

signature = "fake_signature"
token = f"{header_encoded}.{payload_encoded}.{signature}"
print(token)
```

## typing模块

typing提供类型提示支持。

### 基本类型提示

```python
def greet(name: str) -> str:
    return f"Hello, {name}"
```

参数后加冒号指定类型，箭头指定返回类型。

### 常用类型

```python
from typing import List, Dict, Set, Tuple, Optional

def process(items: List[int]) -> Dict[str, int]:
    return {"count": len(items)}

def find(key: str, data: Dict[str, int]) -> Optional[int]:
    return data.get(key)
```

### Union和Optional

```python
from typing import Union

def parse(value: Union[str, int]) -> str:
    return str(value)
```

Optional[str]等价于Union[str, None]。

### 泛型

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: List[T] = []

    def push(self, item: T) -> None:
        self.items.append(item)

    def pop(self) -> T:
        return self.items.pop()
```

### Callable

```python
from typing import Callable

def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

result = apply(lambda x, y: x + y, 1, 2)
print(result)  # 3
```

### 类型别名

```python
from typing import Dict, List

Matrix = List[List[float]]
Coordinates = Dict[str, float]

def transform(matrix: Matrix) -> Matrix:
    return [[x * 2 for x in row] for row in matrix]
```

### 运行时检查

类型提示不自动验证，需要配合mypy或运行时库：

```python
from typing import get_type_hints

def greet(name: str) -> str:
    return f"Hello, {name}"

hints = get_type_hints(greet)
print(hints)  # {'name': <class 'str'>, 'return': <class 'str'>}
```

类型提示主要作用：静态分析工具检查、IDE自动补全、文档。

## json模块回顾

json在IO章节已介绍，这里补充一些高级用法：

### 自定义编码

```python
import json
from datetime import datetime

class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        return super().default(obj)

data = {"created": datetime(2024, 1, 15)}
result = json.dumps(data, cls=DateEncoder)
print(result)
```

### 漂亮打印

```python
print(json.dumps(data, indent=2, sort_keys=True))
```

## collections模块回顾

collections提供特殊容器数据类型：

### deque双端队列

```python
from collections import deque

dq = deque()
dq.append(1)
dq.appendleft(0)
print(list(dq))  # [0, 1]
dq.pop()
dq.popleft()
```

### defaultdict

```python
from collections import defaultdict

dd = defaultdict(list)
dd["key"].append(1)
print(dd["key"])  # [1]
```

### Counter

```python
from collections import Counter

text = "hello world"
c = Counter(text)
print(c.most_common(3))  # [('l', 3), ('o', 2), (' ', 1)]
```

### namedtuple

```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)
```

## functools模块回顾

functools提供高阶函数工具：

### lru_cache

缓存装饰器：

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print(fib(100))  # 很快
```

### partial偏函数

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))   # 125
```

### reduce

```python
from functools import reduce

result = reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])
print(result)  # 15
```

## 常见问题

### 日志级别选择

- DEBUG：开发时详细调试信息
- INFO：正常运行的确认信息
- WARNING：潜在问题，不影响运行
- ERROR：错误，影响特定功能
- CRITICAL：严重错误，程序可能终止

### 密码存储

不要使用MD5或SHA存储密码：

```python
# 错误
hashlib.md5(password).hexdigest()

# 正确
bcrypt.hashpw(password, bcrypt.gensalt())
```

MD5太快，攻击者可以高速枚举常见密码。

### 类型提示与运行时

类型提示不强制执行：

```python
def greet(name: str) -> str:
    return name  # 忘记格式化字符串，类型检查器可能检测不到

result = greet(123)  # 运行时不会报错
```

使用mypy或pyright进行静态检查。

## 面试关注点

面试中关于工具模块的常见问题包括：logging的层级？hashlib的用途？base64的原理？

理解这些工具的设计意图很重要：logging用于生产环境追踪问题，hashlib用于数据完整性校验和密码存储，类型提示用于代码文档和静态检查。
