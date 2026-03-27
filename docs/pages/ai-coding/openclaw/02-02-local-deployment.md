# 本地部署

环境准备就绪后，这一章我们开始实际的安装部署。OpenClaw支持三大主流操作系统，但安装方式略有差异。请根据你的系统选择对应的小节。

## MacOS部署

MacOS是OpenClaw开发团队的主要开发平台，安装体验最为顺畅。

**方式一：Homebrew安装（推荐）**

```bash
# 1. 安装Homebrew（如果尚未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 添加OpenClaw的tap源
brew tap openclaw/tap

# 3. 安装OpenClaw
brew install openclaw

# 4. 验证安装
openclaw --version
```

**方式二：npm全局安装**

```bash
# 1. 确保Node.js 22已安装
node --version  # 应输出v22.x.x

# 2. 全局安装OpenClaw
npm install -g @openclaw/cli

# 3. 验证安装
openclaw --version
```

**方式三：从源码安装（开发者推荐）**

```bash
# 1. 克隆仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 2. 安装依赖
npm install

# 3. 构建项目
npm run build

# 4. 全局链接
npm link

# 5. 验证
openclaw --version
```

**常见问题：**

**权限错误：**

```bash
# 如果遇到EACCES错误，修复npm权限
sudo chown -R $(whoami) $(npm config get prefix)/{lib/node_modules,bin,share}
```

**M1/M2芯片兼容性：**

OpenClaw原生支持Apple Silicon，无需额外配置。但如果使用本地模型，建议安装原生版本的Ollama：

```bash
brew install ollama
```

## Linux部署（Ubuntu推荐）

Ubuntu是OpenClaw在服务器端最常用的系统，以下以Ubuntu 22.04为例。

**方式一：npm安装（推荐）**

```bash
# 1. 更新系统包
sudo apt update && sudo apt upgrade -y

# 2. 安装Node.js 22
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# 3. 安装构建工具（部分依赖需要编译）
sudo apt install -y build-essential python3

# 4. 全局安装OpenClaw
sudo npm install -g @openclaw/cli

# 5. 验证安装
openclaw --version
```

**方式二：Docker部署（推荐服务器用户）**

Docker方式可以避免环境依赖问题，更适合生产环境。

```bash
# 1. 安装Docker
curl -fsSL https://get.docker.com | sh

# 2. 将当前用户加入docker组（避免每次sudo）
sudo usermod -aG docker $USER
newgrp docker

# 3. 拉取OpenClaw镜像
docker pull openclaw/openclaw:latest

# 4. 创建数据目录
mkdir -p ~/openclaw/workspace

# 5. 启动容器
docker run -d \
  --name openclaw \
  -p 18789:18789 \
  -v ~/openclaw/workspace:/app/workspace \
  -e OPENAI_API_KEY=your_api_key \
  openclaw/openclaw:latest

# 6. 查看运行状态
docker logs -f openclaw
```

**Docker Compose方式（推荐）：**

创建`docker-compose.yml`文件：

```yaml
version: '3.8'

services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    restart: unless-stopped
    ports:
      - "18789:18789"
    volumes:
      - ./workspace:/app/workspace
      - ./config:/app/config
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:18789/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

启动服务：

```bash
# 创建.env文件存储敏感信息
echo "OPENAI_API_KEY=sk-xxx" > .env

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f
```

**系统服务配置（systemd）：**

如果你使用npm方式安装，可以配置systemd让OpenClaw开机自启：

```bash
# 创建服务文件
sudo tee /etc/systemd/system/openclaw.service <<EOF
[Unit]
Description=OpenClaw AI Assistant
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER
ExecStart=/usr/bin/openclaw start
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 重载systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start openclaw

# 设置开机自启
sudo systemctl enable openclaw

# 查看状态
sudo systemctl status openclaw
```

## Windows部署

Windows用户推荐使用WSL2（Windows Subsystem for Linux）获得最佳体验。

**方式一：WSL2 + Ubuntu（推荐）**

```powershell
# 1. 在PowerShell（管理员）中启用WSL
wsl --install

# 2. 重启电脑后，WSL会自动安装Ubuntu
# 3. 设置Ubuntu用户名和密码

# 4. 进入Ubuntu
wsl

# 5. 后续步骤与Linux部署相同
# 安装Node.js 22
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# 安装OpenClaw
sudo npm install -g @openclaw/cli

# 验证
openclaw --version
```

**方式二：原生Windows安装**

如果你不想使用WSL，也可以直接在Windows上安装：

```powershell
# 1. 安装Node.js 22
# 访问 https://nodejs.org/ 下载Windows安装包

# 2. 打开PowerShell，验证安装
node --version
npm --version

# 3. 全局安装OpenClaw
npm install -g @openclaw/cli

# 4. 验证
openclaw --version
```

**Windows原生安装的注意事项：**

- 部分Skill依赖Unix命令，可能无法正常工作
- 文件路径使用反斜杠`\`，与文档中的正斜杠`/`不同
- 建议使用PowerShell 7+，避免CMD的兼容性问题

**Windows服务配置（nssm）：**

```powershell
# 1. 下载nssm
# https://nssm.cc/download

# 2. 解压后，在管理员PowerShell中执行
nssm install OpenClaw

# 3. 在弹出的GUI中配置：
#    Path: C:\Program Files\nodejs\node.exe
#    Arguments: C:\Users\YourName\AppData\Roaming\npm\node_modules\@openclaw\cli\bin\openclaw start
#    Startup directory: C:\Users\YourName

# 4. 启动服务
nssm start OpenClaw
```

## 验证安装

无论使用哪种方式安装，完成后都应该进行验证。

**基本验证：**

```bash
# 查看版本
openclaw --version
# 预期输出：OpenClaw v1.x.x

# 查看帮助
openclaw --help
# 预期输出：命令列表和使用说明

# 检查环境
openclaw doctor
```

**`openclaw doctor`命令会检查：**

```
Running diagnostics...

✓ Node.js version: v22.1.0 (OK, requires >= 22.0.0)
✓ npm version: 10.2.0 (OK)
✓ Git installed: /usr/bin/git
✓ OpenClaw CLI: v1.2.3
✓ Workspace directory: ~/openclaw/workspace (exists)
✓ Config file: ~/openclaw/config.yaml (exists)
⚠ API Key: Not configured (run 'openclaw onboard' to configure)
⚠ No channels configured

Diagnostics complete: 6 passed, 2 warnings
```

**启动服务：**

```bash
# 前台启动（调试用）
openclaw start

# 后台启动
openclaw start --daemon

# 查看状态
openclaw status
# 预期输出：OpenClaw is running on http://localhost:18789
```

**访问Web界面：**

打开浏览器，访问 `http://localhost:18789` ，你应该能看到OpenClaw的控制面板。此时还没有配置API Key，所以还无法正常对话，但界面应该能正常加载。

---

本地部署完成后，如果你是个人用户且有自己的服务器，可以继续阅读下一章的云端部署方案。如果你是新手，想快速体验OpenClaw，可以直接跳到云端部署章节，使用预装镜像会简单很多。
