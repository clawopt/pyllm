---
title: 日期时间与随机数
---

# 日期时间与随机数

日期时间和随机数是编程中最常用的工具模块。处理时间、计算日期间隔、生成随机数——这些需求几乎在每个项目中都会出现。Python提供了datetime、time、calendar和random等模块来处理这些任务。

这一章介绍这些模块的核心用法。

## datetime模块

datetime模块是处理日期和时间的核心模块。它定义了四个主要类：date表示日期，time表示时间，datetime表示日期和时间，timedelta表示时间间隔。

### date日期

date对象代表一个日期：

```python
from datetime import date

today = date.today()
print(today)           # 2024-01-15
print(today.year)      # 2024
print(today.month)     # 1
print(today.day)       # 15
```

date对象可以比较：

```python
d1 = date(2024, 1, 1)
d2 = date(2024, 12, 31)
print(d1 < d2)  # True
```

可以用`replace()`创建修改后的副本：

```python
d = date(2024, 1, 15)
d2 = d.replace(month=2)
print(d2)  # 2024-02-15
```

### time时间

time对象代表一天中的时间：

```python
from datetime import time

t = time(14, 30, 45)
print(t.hour)      # 14
print(t.minute)    # 30
print(t.second)    # 45
print(t)           # 14:30:45
```

time对象不包含日期信息。

### datetime日期时间

datetime是date和time的组合：

```python
from datetime import datetime

now = datetime.now()
print(now)              # 2024-01-15 14:30:45.123456

dt = datetime(2024, 1, 15, 14, 30)
print(dt)               # 2024-01-15 14:30:00

print(dt.date())        # 2024-01-15
print(dt.time())        # 14:30:00
```

常用方法：

```python
now = datetime.now()
timestamp = now.timestamp()  # 时间戳
dt = datetime.fromtimestamp(timestamp)  # 从时间戳创建
utc = datetime.utcnow()  # UTC时间
```

### timedelta时间间隔

timedelta表示两个日期或时间之间的差值：

```python
from datetime import datetime, timedelta

dt1 = datetime(2024, 1, 1)
dt2 = datetime(2024, 1, 15)

diff = dt2 - dt1
print(diff)              # 14 days, 0:00:00
print(diff.days)         # 14
print(diff.total_seconds())  # 1209600.0
```

创建timedelta：

```python
d = timedelta(days=7)
d = timedelta(hours=12)
d = timedelta(minutes=30)
d = timedelta(seconds=30)
d = timedelta(weeks=1, days=2)
```

timedelta可以相加：

```python
from datetime import datetime, timedelta

dt = datetime(2024, 1, 15)
dt_after_week = dt + timedelta(weeks=1)
dt_before_day = dt - timedelta(days=1)
```

日期推算：

```python
dt = datetime(2024, 1, 15, 10, 30)
dt_after_2h = dt + timedelta(hours=2)  # 2024-01-15 12:30:00
```

### strftime格式化

datetime对象转换为字符串：

```python
dt = datetime(2024, 1, 15, 14, 30)

print(dt.strftime("%Y-%m-%d"))          # 2024-01-15
print(dt.strftime("%Y/%m/%d"))          # 2024/01/15
print(dt.strftime("%H:%M:%S"))          # 14:30:00
print(dt.strftime("%Y-%m-%d %H:%M:%S")) # 2024-01-15 14:30:00
print(dt.strftime("%B %d, %Y"))         # January 15, 2024
```

常用格式化符号：

| 符号 | 含义 | 示例 |
|------|------|------|
| %Y | 4位年份 | 2024 |
| %m | 2位月份 | 01-12 |
| %d | 2位日期 | 01-31 |
| %H | 24小时制 | 00-23 |
| %M | 分钟 | 00-59 |
| %S | 秒 | 00-59 |
| %B | 月份全名 | January |
| %b | 月份简称 | Jan |
| %A | 星期全名 | Monday |
| %a | 星期简称 | Mon |

### strptime解析

从字符串解析datetime：

```python
dt = datetime.strptime("2024-01-15", "%Y-%m-%d")
print(dt)  # 2024-01-15 00:00:00

dt = datetime.strptime("2024/01/15 14:30", "%Y/%m/%d %H:%M")
print(dt)  # 2024-01-15 14:30:00
```

strptime和strftime是可逆的操作。

## time模块

time模块提供时间相关的函数，与datetime侧重点不同。

### 时间戳

时间戳是从1970年1月1日到现在经过的秒数：

```python
import time

print(time.time())  # 1705312245.123456

time.sleep(1)  # 休眠1秒
```

### struct_time

struct_time将时间分解为各个组成部分：

```python
import time

now = time.time()
st = time.localtime(now)
print(st.tm_year)   # 2024
print(st.tm_mon)    # 1
print(st.tm_mday)   # 15
print(st.tm_hour)   # 14
```

时间戳和struct_time互转：

```python
now = time.time()
st = time.localtime(now)
timestamp = time.mktime(st)
```

### 格式化

```python
print(time.strftime("%Y-%m-%d %H:%M:%S"))
print(time.strftime("%B %d, %Y"))
```

与datetime.strftime用法相同。

### perf_counter

高精度计时：

```python
import time

start = time.perf_counter()
result = sum(range(10**7))
end = time.perf_counter()

print(f"Elapsed: {end - start:.6f} seconds")
```

`perf_counter`适合测量代码执行时间，比`time.time()`精度更高。

## calendar模块

calendar模块提供日历相关的功能：

```python
import calendar

print(calendar.month(2024, 1))  # 打印2024年1月日历

print(calendar.isleap(2024))  # True，闰年判断
print(calendar.leapdays(2000, 2025))  # 6，2000-2024间闰年数量
```

calendar使用weekheader显示星期：

```python
print(calendar.weekheader(3))  # Mon Tue Wed Thu Fri Sat Sun
```

获取某月所有日期：

```python
c = calendar.Calendar()
for day in c.itermonthdates(2024, 1):
    print(day)
```

## random模块

random模块用于生成随机数。

### 基本用法

```python
import random

print(random.random())  # 0.0-1.0之间的随机浮点数
print(random.randint(1, 100))  # 1-100之间的随机整数
print(random.randrange(0, 100, 2))  # 0-99之间的随机偶数
```

### choice选择

从序列中随机选择：

```python
colors = ['red', 'green', 'blue']
print(random.choice(colors))  # 随机选一个

print(random.sample(colors, 2))  # 随机选2个，不重复
```

### shuffle洗牌

打乱序列顺序：

```python
cards = list(range(52))
random.shuffle(cards)
print(cards)
```

shuffle就地修改列表，不返回新列表。

### 均匀分布

生成符合特定分布的随机数：

```python
print(random.uniform(0, 1))  # 均匀分布
print(random.gauss(0, 1))  # 高斯分布
```

### 随机种子

设置种子使随机数可重现：

```python
random.seed(42)
print(random.random())  # 每次seed(42)后结果相同
print(random.random())
```

同一种子产生相同的随机序列，用于测试和调试。

### 常见应用

随机密码：

```python
import string

def generate_password(length=12):
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(chars) for _ in range(length))

print(generate_password())
```

随机样本：

```python
# 从10000个ID中随机选100个
ids = list(range(10000))
selected = random.sample(ids, 100)
```

## 时区处理

datetime的aware对象包含时区信息：

```python
from datetime import datetime, timezone

dt = datetime.now(timezone.utc)
print(dt)  # 2024-01-15 14:30:00+00:00

dt_tokyo = dt.astimezone(timezone(timedelta(hours=9)))
print(dt_tokyo)  # 2024-01-15 23:30:00+09:00
```

使用pytz库处理更复杂的时区：

```python
import pytz

tokyo = pytz.timezone('Asia/Tokyo')
dt = tokyo.localize(datetime(2024, 1, 15, 14, 30))
print(dt.astimezone(pytz.utc))
```

## 常见问题

### 时区陷阱

 naive datetime不包含时区信息，可能导致错误：

```python
from datetime import datetime

# naive对象，不知道是哪个时区
dt = datetime(2024, 1, 15, 14, 30)

# 总是使用aware对象，明确时区
from datetime import timezone
dt_aware = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
```

存储和传输时使用UTC， 显示时转换为本地时区。

### 浮点数精度

timedelta的秒数可能是浮点数：

```python
from datetime import timedelta

td = timedelta(seconds=1.5)
print(td.total_seconds())  # 1.5
```

处理金融或科学计算时注意精度。

### 随机数安全

random模块不是密码学安全的。生成密码或token时使用secrets模块：

```python
import secrets

token = secrets.token_urlsafe(32)  # 安全的随机字符串
print(token)
```

## 面试关注点

面试中关于日期时间和随机数的常见问题包括：datetime和time模块的区别？timedelta有什么用？如何处理时区？random和secrets模块的区别？

理解时区处理是难点。推荐做法是：内部存储始终使用UTC时区，显示时转换为用户本地时区。

随机数方面，需要理解种子的作用，以及为什么random不适合安全场景。
