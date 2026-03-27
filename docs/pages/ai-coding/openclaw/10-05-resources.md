# 相关资源链接

这一章整理了OpenClaw相关的资源链接，方便你深入学习。

## 官方资源

### OpenClaw官方

| 资源 | 地址 |
|------|------|
| 官方网站 | https://openclaw.ai |
| 官方文档 | https://docs.openclaw.ai |
| GitHub仓库 | https://github.com/openclaw/openclaw |
| 更新日志 | https://github.com/openclaw/openclaw/releases |
| 问题反馈 | https://github.com/openclaw/openclaw/issues |

### ClawHub技能市场

| 资源 | 地址 |
|------|------|
| ClawHub官网 | https://hub.openclaw.ai |
| 技能搜索 | https://hub.openclaw.ai/skills |
| 开发者文档 | https://hub.openclaw.ai/developers |
| 提交技能 | https://hub.openclaw.ai/submit |

### 社区资源

| 资源 | 地址 |
|------|------|
| Discord社区 | https://discord.gg/openclaw |
| Twitter | https://twitter.com/openclaw_ai |
| 微信公众号 | OpenClaw小龙虾 |
| 知乎专栏 | https://zhuanlan.zhihu.com/openclaw |

## 国内镜像与加速

### npm镜像

```bash
# 使用淘宝镜像
npm config set registry https://registry.npmmirror.com

# 安装OpenClaw
npm install -g @openclaw/cli
```

### GitHub镜像

```bash
# 使用Gitee镜像
git clone https://gitee.com/openclaw/openclaw.git
```

### 模型API国内加速

| 提供商 | 国内端点 |
|--------|---------|
| 阿里云百炼 | https://dashscope.aliyuncs.com |
| DeepSeek | https://api.deepseek.com |
| 智谱AI | https://open.bigmodel.cn |

## 学习资源

### 官方教程

| 资源 | 地址 |
|------|------|
| 快速开始 | https://docs.openclaw.ai/quickstart |
| 技能开发指南 | https://docs.openclaw.ai/skill-development |
| 最佳实践 | https://docs.openclaw.ai/best-practices |
| API参考 | https://docs.openclaw.ai/api-reference |

### 视频教程

| 平台 | 内容 |
|------|------|
| B站 | 搜索"OpenClaw教程" |
| YouTube | OpenClaw Official |
| 抖音 | OpenClaw小龙虾 |

### 示例项目

| 项目 | 地址 | 描述 |
|------|------|------|
| openclaw-examples | https://github.com/openclaw/examples | 官方示例集合 |
| openclaw-templates | https://github.com/openclaw/templates | 常用模板 |
| openclaw-cookbook | https://github.com/openclaw/cookbook | 实战案例 |

## 相关工具

### 开发工具

| 工具 | 地址 | 描述 |
|------|------|------|
| VS Code插件 | 搜索"OpenClaw" | IDE集成 |
| Postman集合 | https://docs.openclaw.ai/postman | API测试 |
| Docker镜像 | https://hub.docker.com/r/openclaw/openclaw | 容器部署 |

### 监控工具

| 工具 | 地址 | 描述 |
|------|------|------|
| SecureClaw | https://github.com/openclaw/secureclaw | 安全扫描 |
| OpenClaw Exporter | https://github.com/openclaw/exporter | Prometheus集成 |

### 集成工具

| 工具 | 描述 |
|------|------|
| n8n节点 | 工作流自动化集成 |
| Zapier集成 | 自动化流程连接 |
| Make集成 | 无代码自动化 |

## 常见问题

### 官方FAQ

| 资源 | 地址 |
|------|------|
| 常见问题 | https://docs.openclaw.ai/faq |
| 故障排除 | https://docs.openclaw.ai/troubleshooting |
| 已知问题 | https://github.com/openclaw/openclaw/labels/bug |

### 社区支持

| 渠道 | 适用场景 |
|------|---------|
| GitHub Discussions | 深度技术讨论 |
| Discord | 实时交流 |
| 微信群 | 国内用户交流 |

## 贡献指南

### 参与贡献

| 资源 | 地址 |
|------|------|
| 贡献指南 | https://github.com/openclaw/openclaw/blob/main/CONTRIBUTING.md |
| 行为准则 | https://github.com/openclaw/openclaw/blob/main/CODE_OF_CONDUCT.md |
| 开发指南 | https://docs.openclaw.ai/contributing |

### 技能贡献

| 资源 | 地址 |
|------|------|
| 技能模板 | https://github.com/openclaw/skill-template |
| 提交指南 | https://hub.openclaw.ai/submit/guide |
| 审核标准 | https://hub.openclaw.ai/review-criteria |

## 版本信息

### 当前版本

```bash
openclaw --version
# OpenClaw v1.2.3
```

### 版本历史

| 版本 | 发布日期 | 主要更新 |
|------|---------|---------|
| v1.2.0 | 2026-03-01 | RAG支持、多Agent |
| v1.1.0 | 2026-02-01 | ClawHub集成 |
| v1.0.0 | 2026-01-01 | 正式发布 |

### 升级指南

```bash
# 查看可用更新
openclaw update --check

# 升级到最新版本
npm update -g @openclaw/cli

# 或使用Homebrew
brew upgrade openclaw
```

---

如果你发现链接失效或有其他资源推荐，欢迎在GitHub提交Issue或PR。
