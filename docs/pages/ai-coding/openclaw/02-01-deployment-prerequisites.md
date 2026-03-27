# 部署前置准备

在正式安装OpenClaw之前，我们需要确保你的环境满足运行要求。这一章会帮你检查硬件、软件、账号和网络四个维度的准备工作。

## 硬件要求

OpenClaw的硬件需求取决于你选择哪种运行模式。

**基础模式（调用云端大模型）**——如果你使用OpenAI、阿里云百炼、DeepSeek等云端API，OpenClaw本身只负责任务编排和执行，不需要本地推理。这种情况下：

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| CPU | 2核 | 4核+ |
| 内存 | 4GB | 8GB+ |
| 存储 | 10GB可用空间 | 20GB+ SSD |
| 网络 | 稳定互联网连接 | 宽带/光纤 |

**本地模型模式**——如果你想运行本地大模型（如Ollama + Qwen），硬件需求会显著提高：

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| CPU | 4核 | 8核+ |
| 内存 | 8GB | 16GB+ |
| GPU | 无（CPU推理） | NVIDIA 8GB显存+ |
| 存储 | 20GB可用空间 | 50GB+ SSD |

**为什么本地模型需要更多资源？**

大模型推理是计算密集型任务。一个7B参数的模型（如Qwen-7B），仅模型权重就需要约14GB内存（FP16精度）或4GB（INT4量化）。如果内存不足，系统会频繁使用磁盘交换，导致响应速度从秒级降到分钟级。

**快速自检方法：**

```bash
# 查看CPU核心数
nproc

# 查看内存大小（GB）
free -h | awk '/Mem:/ {print $2}'

# 查看磁盘空间（GB）
df -h / | awk 'NR==2 {print $4}'
```

## 软件要求

OpenClaw基于Node.js开发，以下是必需和可选的软件依赖。

**必需依赖：**

| 软件 | 版本要求 | 用途 |
|------|---------|------|
| Node.js | ≥ 22.0.0 | 运行时环境 |
| npm | ≥ 10.0.0 | 包管理器（随Node.js安装） |
| Git | ≥ 2.30 | 版本控制、插件安装 |

**可选依赖：**

| 软件 | 用途 |
|------|------|
| Docker | 容器化部署，简化环境配置 |
| Python | 部分Skill需要Python环境 |
| Ollama | 本地运行大模型 |

**Node.js版本为什么要求22+？**

OpenClaw使用了Node.js 22引入的新特性，包括：
- 更好的ESM模块支持
- 改进的性能和内存管理
- 原生fetch API的增强

如果你使用旧版本Node.js，可能会遇到模块加载错误或性能问题。

**安装Node.js 22：**

```bash
# 方式一：使用nvm（推荐，可管理多版本）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 22
nvm use 22

# 方式二：使用官方安装包
# 访问 https://nodejs.org/ 下载LTS版本

# 验证安装
node --version  # 应输出 v22.x.x
npm --version   # 应输出 10.x.x
```

**安装Git：**

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git

# MacOS（通常已预装，或使用Homebrew）
brew install git

# Windows
# 访问 https://git-scm.com/download/win 下载安装
```

## 账号准备

OpenClaw本身不提供大模型能力，需要连接外部大模型API。你需要至少准备一个API Key。

**支持的模型提供商：**

| 提供商 | 模型 | 价格参考 | 特点 |
|--------|------|---------|------|
| OpenAI | GPT-4o, GPT-4-turbo | $2.5-10/百万token | 能力最强，价格较高 |
| 阿里云百炼 | Qwen-Max, Qwen-Plus | ¥0.02-0.12/千token | 国内访问稳定，性价比高 |
| DeepSeek | DeepSeek-V3, DeepSeek-Chat | ¥1/百万token | 极致性价比，推理能力强 |
| 智谱AI | GLM-4 | ¥0.1/千token | 中文能力强 |
| Anthropic | Claude-3.5-Sonnet | $3-15/百万token | 长文本处理优秀 |

**如何获取API Key：**

**阿里云百炼（推荐国内用户）：**

1. 访问 https://bailian.console.aliyun.com/
2. 使用阿里云账号登录
3. 进入"模型广场" → 选择Qwen系列模型
4. 点击"开通服务" → 创建API Key
5. 复制保存API Key（只显示一次）

**DeepSeek（推荐预算有限用户）：**

1. 访问 https://platform.deepseek.com/
2. 注册账号（支持微信扫码）
3. 进入"API Keys"页面
4. 点击"创建API Key"
5. 新用户赠送500万token额度

**OpenAI（推荐追求最佳效果用户）：**

1. 访问 https://platform.openai.com/
2. 注册账号（需要海外手机号或邮箱）
3. 进入"API Keys" → "Create new secret key"
4. 充值后即可使用（最低$5起充）

**API Key安全提示：**

API Key相当于你的账户密码，泄露可能导致：
- 额度被盗用
- 账户被封禁
- 敏感数据泄露

请务必：
- 不要将Key提交到公开代码仓库
- 使用环境变量存储，而非硬编码
- 定期轮换Key
- 设置使用额度上限

## 端口与安全

OpenClaw默认使用18789端口提供Web服务和API接口。你需要确保这个端口可用且可访问。

**端口检查：**

```bash
# 检查端口是否被占用
lsof -i :18789

# 如果被占用，查看占用进程
netstat -tunlp | grep 18789
```

**防火墙配置：**

**Ubuntu/Debian (ufw)：**

```bash
# 开放18789端口
sudo ufw allow 18789/tcp

# 查看防火墙状态
sudo ufw status
```

**CentOS/RHEL (firewalld)：**

```bash
# 开放端口
sudo firewall-cmd --permanent --add-port=18789/tcp
sudo firewall-cmd --reload

# 查看已开放端口
sudo firewall-cmd --list-ports
```

**云服务器安全组：**

如果你使用云服务器（阿里云、腾讯云等），还需要在控制台配置安全组规则：

1. 登录云服务器控制台
2. 找到"安全组"设置
3. 添加入站规则：
   - 协议：TCP
   - 端口：18789
   - 来源：0.0.0.0/0（公网访问）或指定IP（限制访问）

**安全建议：**

默认配置下，OpenClaw的Web界面没有任何认证机制。如果直接暴露在公网，任何人都可以使用你的服务。建议：

1. **仅本地访问**——绑定到127.0.0.1，通过SSH隧道访问
2. **添加认证**——配置OpenClaw的认证插件
3. **反向代理**——使用Nginx添加Basic Auth
4. **VPN/内网**——仅在内网或VPN环境下访问

---

完成以上准备工作后，你的环境已经就绪。下一章，我们将进入实际的安装部署环节，分别介绍本地部署和云端部署两种方式。
