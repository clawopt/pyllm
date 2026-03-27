# 自动发微信公众号文章

这一章开始，我们通过实战案例来展示OpenClaw的强大能力。第一个案例是自动发布微信公众号文章——这是很多内容创作者的刚需。

## 场景描述

假设你写了一篇技术文章，想要发布到微信公众号。传统流程是：

1. 登录微信公众号后台
2. 点击"新建图文"
3. 复制粘贴标题和正文
4. 手动调整格式
5. 上传封面图片
6. 设置摘要和标签
7. 点击发布

整个过程至少需要10分钟，而且格式调整很繁琐。用OpenClaw，你可以一句话搞定。

## 完整提示词模板

**基础版：**

```
帮我发布这篇文章到微信公众号：
标题：[文章标题]
内容：[文章正文]
```

**进阶版：**

```markdown
帮我发布文章到微信公众号，具体信息如下：

## 文章信息
- 标题：深入理解Python异步编程
- 作者：张三
- 摘要：本文深入讲解Python异步编程的原理和实战技巧

## 正文内容
[粘贴Markdown格式的文章正文]

## 封面图
- 自动生成一张科技风格的封面图
- 尺寸：900×383（公众号推荐尺寸）

## 其他设置
- 标签：Python, 异步编程, 技术教程
- 原文链接：https://example.com/article/xxx
- 是否原创：是

## 注意事项
- 保留Markdown格式，转换为公众号支持的格式
- 代码块使用公众号支持的样式
- 图片自动上传到公众号图床
```

## 浏览器自动化配置

OpenClaw使用`agent-browser`技能实现浏览器自动化。

**安装技能：**

```bash
openclaw hub install agent-browser
```

**配置公众号登录：**

```bash
# 方式一：使用保存的登录状态
openclaw skill config agent-browser \
  --profile "wechat_mp" \
  --user-data-dir "~/.openclaw/browser_profiles/wechat_mp"

# 方式二：每次手动扫码
# 默认配置，发布时会提示扫码登录
```

**执行发布：**

```
用户：帮我发布这篇文章到微信公众号...

OpenClaw：[执行发布流程]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: 启动浏览器
  ✓ 浏览器已启动

Step 2: 登录公众号
  → 请扫描二维码登录
  ✓ 登录成功

Step 3: 创建图文
  ✓ 点击"新建图文"
  ✓ 填写标题
  ✓ 填写正文（已转换格式）
  ✓ 上传封面图

Step 4: 设置属性
  ✓ 填写摘要
  ✓ 添加标签
  ✓ 设置原创声明

Step 5: 发布
  ✓ 点击"保存并发布"
  ✓ 确认发布

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 文章已发布成功！

文章链接：https://mp.weixin.qq.com/s/xxxxxx
```

## 封面图自动生成

OpenClaw可以自动生成符合公众号规格的封面图。

**配置图像生成技能：**

```bash
openclaw hub install image_gen
```

**封面图生成提示词：**

```
为文章《深入理解Python异步编程》生成封面图：
- 尺寸：900×383像素
- 风格：科技、简约、现代
- 颜色：蓝色渐变为主
- 元素：代码符号、异步流程图示意
- 文字：不要包含文字（标题会在公众号中叠加）
```

**自动生成流程：**

```
OpenClaw：[生成封面图]
  → 分析文章主题：Python异步编程
  → 生成提示词：minimalist tech cover, async programming, blue gradient, code symbols
  → 调用DALL-E 3生成
  → 调整尺寸为900×383
  → 上传到公众号

✓ 封面图已生成并上传
```

## 完整工作流配置

你可以将整个发布流程配置为一个定时任务或快捷命令。

**创建发布技能：**

```markdown
# ~/.openclaw/skills/wechat_publish/SKILL.md

---
name: wechat_publish
version: 1.0.0
description: 自动发布文章到微信公众号
dependencies:
  - agent-browser
  - image_gen
permissions:
  - browser:control
  - network:access
---

# 微信公众号发布技能

## 工作流程

1. 解析文章信息（标题、正文、标签等）
2. 转换Markdown为公众号格式
3. 生成封面图（如未提供）
4. 启动浏览器，登录公众号后台
5. 创建图文，填写内容
6. 设置属性，发布文章
7. 返回文章链接

## 使用方式

```json
{
  "action": "wechat_publish.publish",
  "params": {
    "title": "文章标题",
    "content": "Markdown正文",
    "cover_style": "tech",
    "tags": ["标签1", "标签2"],
    "is_original": true
  }
}
```

## 注意事项

- 首次使用需要扫码登录
- 登录状态有效期约2小时
- 图片会自动上传到公众号图床
- 发布后无法修改，请确认内容正确
```

**创建快捷命令：**

```bash
# 添加别名
alias publish="openclaw skill run wechat_publish.publish"

# 使用
publish --title "文章标题" --file article.md
```

## 批量发布

如果你有多篇文章需要发布：

```bash
# 批量发布脚本
openclaw skill run wechat_publish.batch \
  --articles-dir ~/Drafts/articles/ \
  --interval 30  # 每篇文章间隔30分钟
```

**批量发布配置：**

```yaml
# ~/.openclaw/skills/wechat_publish/config.yaml

batch:
  # 发布间隔（分钟）
  interval: 30
  
  # 每天发布数量限制
  daily_limit: 8
  
  # 发布时间段
  allowed_hours:
    - 8-12   # 早上8点到12点
    - 14-22  # 下午2点到晚上10点
  
  # 失败重试
  retry:
    max_attempts: 3
    delay: 60
```

---

通过这个案例，你可以看到OpenClaw如何将繁琐的手动操作自动化。下一章，我们来看另一个实用案例：日报自动生成与发送。
