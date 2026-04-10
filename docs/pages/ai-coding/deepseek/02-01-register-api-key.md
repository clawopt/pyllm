# 2.1 注册与获取 API Key

> **3 分钟完成注册，1 行代码开始调用——DeepSeek 的上手门槛可能是所有主流大模型中最低的。**

---

## 这一节在讲什么？

理论讲完了，该动手了。这一节带你从零开始：注册 DeepSeek 平台、获取 API Key、了解定价和免费额度。DeepSeek 的注册流程非常简单——不需要信用卡，不需要科学上网，注册就送免费额度，支付宝微信都能充值。3 分钟后你就能开始调用 API。

---

## 注册 DeepSeek 平台

**第一步：访问平台**

打开浏览器，访问 [platform.deepseek.com](https://platform.deepseek.com)，点击"注册"。

**第二步：注册账号**

支持手机号注册（中国大陆）和邮箱注册。注册过程不需要信用卡。

**第三步：获取 API Key**

注册成功后，进入控制台，在"API Keys"页面点击"创建 API Key"：

```
你的 API Key 长这样：
sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

⚠️ 重要：API Key 只在创建时显示一次，请立即复制保存！
如果忘记了，只能删除重建。
```

**第四步：设置环境变量**

建议把 API Key 设置为环境变量，避免硬编码在代码中：

```bash
# ~/.zshrc 或 ~/.bashrc
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

然后在代码中通过环境变量引用：

```python
import os
api_key = os.environ.get("DEEPSEEK_API_KEY")
```

---

## API 定价

DeepSeek 的定价是它最大的优势之一。以下是当前的价格：

| 模型 | Input 价格 | Output 价格 | 缓存命中 Input |
|------|-----------|------------|--------------|
| deepseek-chat（V3.2） | $0.27/M | $1.10/M | $0.07/M |
| deepseek-reasoner（R1） | $0.55/M | $2.19/M | $0.14/M |

**缓存命中**是什么意思？DeepSeek API 支持前缀缓存——如果你的请求前缀跟之前的请求相同（比如相同的 system prompt），缓存命中的 token 价格降低 75%。在多轮对话场景下，缓存命中率通常 50% 以上，实际成本比标价低很多。

**与其他模型的定价对比**：

```
生成 100 万 token 的成本对比（input + output 各 50 万）：

  DeepSeek V3.2：$0.27×0.5 + $1.10×0.5 = $0.685
  DeepSeek R1：  $0.55×0.5 + $2.19×0.5 = $1.37
  GPT-4o：       $2.50×0.5 + $10×0.5   = $6.25
  Claude Sonnet：$3×0.5   + $15×0.5    = $9.00

  DeepSeek V3.2 的成本是 Claude 的 1/13
```

---

## 免费额度

DeepSeek 为新用户提供了慷慨的免费额度：

- 注册即送 **500 万 token** 免费额度
- 免费额度可用于 `deepseek-chat` 和 `deepseek-reasoner`
- 免费额度有有效期（通常 30 天），过期作废

500 万 token 能做什么？

```
500 万 token 大约等于：

  - 生成 5000 行代码（每行约 100 token）
  - 100 轮中等长度的对话
  - 审查 50 个文件的代码
  - 足够你完成整个教程的所有示例
```

---

## 充值方式

免费额度用完后，DeepSeek 支持多种充值方式：

- **支付宝**：最低充值 ¥10
- **微信支付**：最低充值 ¥10
- **按量计费**：用多少付多少，没有最低消费

DeepSeek 的充值门槛非常低——¥10 就能用很久，不像某些平台最低要充 $50。

---

## 常见误区

**误区一：以为 DeepSeek API 需要科学上网**

不需要。`api.deepseek.com` 是国内服务器，国内可以直接访问，延迟通常在 50-100ms。这是 DeepSeek 相比 OpenAI API 的一个重要优势——国内开发者不需要代理就能稳定调用。

**误区二：API Key 可以公开分享**

绝对不行。API Key 等同于你的账户密码——任何拥有你 Key 的人都可以用你的额度调用 API，产生的费用由你承担。如果 Key 泄露，立即到平台删除并创建新的 Key。

**误区三：免费额度用完就不能用了**

不是。免费额度用完后，你只需要充值就可以继续使用。充值金额没有最低限制，¥10 就能开始。

**误区四：缓存命中是自动的，不需要做任何事**

缓存命中需要你的请求前缀跟之前的请求一致。最简单的做法是保持 system prompt 不变——这样多轮对话中，system prompt 部分的 token 都能命中缓存，大幅降低成本。

---

## 小结

这一节我们完成了 DeepSeek 的注册和 API Key 获取：访问 platform.deepseek.com 注册，获取 API Key，设置环境变量。DeepSeek 的定价极低（V3.2 $0.27/M input），新用户赠送 500 万 token 免费额度，支持支付宝充值。下一节我们写第一行代码，调用 DeepSeek API。
