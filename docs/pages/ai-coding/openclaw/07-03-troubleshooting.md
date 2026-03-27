# 常见问题排查

使用OpenClaw过程中，你可能会遇到一些问题。这一章，我们汇总了最常见的问题及其解决方案。

## "openclaw not found"

**问题描述：**

```bash
$ openclaw --version
bash: openclaw: command not found
```

**原因分析：**

OpenClaw安装成功，但npm全局路径没有添加到PATH环境变量中。

**解决方案：**

**方式一：添加npm全局路径到PATH**

```bash
# 查看npm全局安装路径
npm config get prefix

# 输出示例
/usr/local

# 添加到PATH（bash用户）
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 添加到PATH（zsh用户）
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**方式二：重新安装OpenClaw**

```bash
# 使用sudo安装（可能需要密码）
sudo npm install -g @openclaw/cli

# 或使用nvm管理Node.js（推荐）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 22
nvm use 22
npm install -g @openclaw/cli
```

**方式三：使用完整路径**

```bash
# 查找openclaw位置
which openclaw
# 或
find /usr -name "openclaw" 2>/dev/null

# 使用完整路径
/usr/local/bin/openclaw --version
```

## 服务不稳定

**问题描述：**

OpenClaw服务经常自动停止，或响应不稳定。

**原因分析：**

可能没有安装daemon服务，OpenClaw以普通进程运行，容易被系统终止。

**解决方案：**

**重新安装daemon：**

```bash
# 停止当前服务
openclaw stop

# 重新运行onboarding，安装daemon
openclaw onboard --install-daemon

# 启动服务
openclaw start
```

**检查daemon状态：**

```bash
# Linux (systemd)
sudo systemctl status openclaw

# MacOS (launchd)
launchctl list | grep openclaw

# 查看服务日志
openclaw logs --service
```

**其他可能原因：**

```bash
# 检查内存是否充足
free -h

# 检查磁盘空间
df -h

# 检查Node.js版本
node --version  # 需要 >= 22.0.0
```

## 配置错误

**问题描述：**

OpenClaw无法启动，提示配置错误。

**解决方案：**

**自动修复：**

```bash
# 使用doctor自动修复
openclaw doctor --fix
```

**手动验证配置：**

```bash
# 验证配置文件格式
openclaw config validate

# 输出示例
Validating config.yaml...
✓ Valid JSON format
✓ All required fields present
✗ Invalid model provider: "unknown_provider"
  → Suggestion: Use one of: alibaba, openai, deepseek, anthropic
```

**重置配置：**

```bash
# 备份当前配置
cp ~/.openclaw/config.yaml ~/.openclaw/config.yaml.bak

# 重置为默认配置
openclaw config reset

# 重新配置
openclaw onboard
```

**常见配置错误：**

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `Invalid API Key` | API Key格式错误 | 检查Key是否完整复制 |
| `Model not found` | 模型名称错误 | 使用`openclaw models list`查看可用模型 |
| `Channel not configured` | 渠道缺少必要配置 | 检查渠道的必需字段 |
| `Permission denied` | 文件权限问题 | 检查`.openclaw`目录权限 |

## 端口冲突

**问题描述：**

```bash
$ openclaw start
Error: Port 18789 is already in use
```

**原因分析：**

18789端口已被其他进程占用。

**解决方案：**

**方式一：查找并终止占用进程**

```bash
# 查找占用端口的进程
lsof -i :18789

# 输出示例
COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
node     12345 user   22u  IPv6 123456      0t0  TCP *:18789 (LISTEN)

# 终止进程
kill 12345

# 或强制终止
kill -9 12345
```

**方式二：使用其他端口**

```bash
# 启动时指定端口
openclaw start --port 18790

# 或修改配置文件
# ~/.openclaw/config.yaml
channels:
  web:
    port: 18790
```

**方式三：检查是否有多个OpenClaw实例**

```bash
# 查找所有node进程
ps aux | grep openclaw

# 终止所有OpenClaw进程
pkill -f openclaw

# 重新启动
openclaw start
```

## API调用失败

**问题描述：**

OpenClaw无法调用大模型API，提示连接错误或认证失败。

**解决方案：**

**检查API Key：**

```bash
# 测试API连接
openclaw models test

# 输出示例
Testing model connection...
Provider: alibaba
Model: qwen-plus
✗ Connection failed: Invalid API Key

# 重新配置API Key
openclaw models set --api-key "sk-xxxx"
```

**检查网络：**

```bash
# 测试网络连接
curl -I https://dashscope.aliyuncs.com

# 如果使用代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

**检查API额度：**

```bash
# 查看API使用情况
openclaw models usage

# 输出示例
API Usage (alibaba):
  Today: 12,345 tokens
  This month: 234,567 tokens
  Remaining: 765,433 tokens
```

## 消息渠道问题

**飞书无法收到消息：**

```bash
# 检查飞书渠道状态
openclaw channels test feishu

# 常见原因：
# 1. 事件订阅URL配置错误
# 2. 权限未开启
# 3. 应用未发布

# 解决步骤：
# 1. 检查URL是否正确：http://IP:18789/webhook/feishu
# 2. 确认权限：im:message, im:message:send_as_bot
# 3. 发布应用到通讯录
```

**Telegram Bot无响应：**

```bash
# 检查Bot状态
openclaw channels test telegram

# 常见原因：
# 1. Webhook未设置
# 2. Token错误
# 3. 网络问题（需要代理）

# 设置Webhook
openclaw channel telegram set-webhook \
  --url "https://your-domain.com/webhook/telegram"
```

## 性能问题

**响应缓慢：**

```bash
# 检查系统资源
openclaw status

# 检查模型延迟
openclaw models test --latency

# 优化建议：
# 1. 使用更快的模型（如qwen-turbo）
# 2. 减少并发请求
# 3. 增加系统内存
```

**内存占用过高：**

```bash
# 查看内存使用
openclaw status --memory

# 清理缓存
openclaw workspace clean --cache

# 重启服务
openclaw restart
```

## 获取帮助

如果以上方法都无法解决问题：

**1. 查看日志**

```bash
openclaw logs --verbose
```

**2. 生成诊断报告**

```bash
openclaw doctor --report > diagnosis.txt
```

**3. 社区支持**

- GitHub Issues: https://github.com/openclaw/openclaw/issues
- Discord社区: https://discord.gg/openclaw
- 官方文档: https://docs.openclaw.ai

**4. 提交Issue时请包含：**

```
- OpenClaw版本：openclaw --version
- 系统信息：uname -a
- 诊断报告：openclaw doctor --report
- 错误日志：openclaw logs --lines 100
- 复现步骤
```

---

遇到问题时，首先运行`openclaw doctor`进行诊断，大多数问题都能得到解决。
