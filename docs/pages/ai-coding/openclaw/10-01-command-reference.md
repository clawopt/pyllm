# OpenClaw命令速查表

这一章整理了OpenClaw最常用的命令，方便你快速查阅。

## 核心命令

### 服务管理

| 命令 | 描述 |
|------|------|
| `openclaw start` | 启动服务 |
| `openclaw start --daemon` | 后台启动 |
| `openclaw stop` | 停止服务 |
| `openclaw restart` | 重启服务 |
| `openclaw status` | 查看状态 |
| `openclaw doctor` | 健康检查 |
| `openclaw doctor --fix` | 自动修复问题 |

### 配置管理

| 命令 | 描述 |
|------|------|
| `openclaw config path` | 查看配置路径 |
| `openclaw config edit` | 编辑配置文件 |
| `openclaw config validate` | 验证配置 |
| `openclaw config reset` | 重置配置 |

### 模型管理

| 命令 | 描述 |
|------|------|
| `openclaw models list` | 查看可用模型 |
| `openclaw models set <model>` | 设置主模型 |
| `openclaw models set-fallback <model>` | 设置备选模型 |
| `openclaw models test` | 测试模型连接 |
| `openclaw models show` | 查看模型配置 |
| `openclaw models usage` | 查看使用量 |

### 渠道管理

| 命令 | 描述 |
|------|------|
| `openclaw channels list` | 查看渠道列表 |
| `openclaw channels enable <channel>` | 启用渠道 |
| `openclaw channels disable <channel>` | 禁用渠道 |
| `openclaw channels test <channel>` | 测试渠道连接 |

### 技能管理

| 命令 | 描述 |
|------|------|
| `openclaw hub list` | 查看已安装技能 |
| `openclaw hub install <skill>` | 安装技能 |
| `openclaw hub uninstall <skill>` | 卸载技能 |
| `openclaw hub update <skill>` | 更新技能 |
| `openclaw hub search <keyword>` | 搜索技能 |
| `openclaw hub trending` | 查看热门技能 |

### 账户管理

| 命令 | 描述 |
|------|------|
| `openclaw account list` | 查看账户列表 |
| `openclaw account create` | 创建账户 |
| `openclaw account show <name>` | 查看账户详情 |
| `openclaw account update <name>` | 更新账户配置 |
| `openclaw account delete <name>` | 删除账户 |

### Agent管理

| 命令 | 描述 |
|------|------|
| `openclaw agent list` | 查看Agent列表 |
| `openclaw agent create` | 创建Agent |
| `openclaw agent use <name>` | 切换Agent |
| `openclaw agent current` | 查看当前Agent |

### 记忆管理

| 命令 | 描述 |
|------|------|
| `openclaw memory list` | 查看记忆列表 |
| `openclaw memory search <keyword>` | 搜索记忆 |
| `openclaw memory add <content>` | 添加记忆 |
| `openclaw memory delete <id>` | 删除记忆 |
| `openclaw memory clear` | 清除记忆 |

### RAG管理

| 命令 | 描述 |
|------|------|
| `openclaw rag status` | 查看知识库状态 |
| `openclaw rag upload <file>` | 上传文档 |
| `openclaw rag list` | 查看文档列表 |
| `openclaw rag search <query>` | 搜索知识库 |
| `openclaw rag delete <doc_id>` | 删除文档 |
| `openclaw rag reindex` | 重建索引 |

### 定时任务

| 命令 | 描述 |
|------|------|
| `openclaw heartbeat list` | 查看任务列表 |
| `openclaw heartbeat add` | 添加任务 |
| `openclaw heartbeat run <name>` | 手动执行任务 |
| `openclaw heartbeat pause <name>` | 暂停任务 |
| `openclaw heartbeat delete <name>` | 删除任务 |

### 日志与调试

| 命令 | 描述 |
|------|------|
| `openclaw logs` | 查看日志 |
| `openclaw logs -f` | 实时查看日志 |
| `openclaw logs --filter <keyword>` | 过滤日志 |
| `openclaw tui` | 启动终端界面 |

### 分析与优化

| 命令 | 描述 |
|------|------|
| `openclaw analytics report` | 生成分析报告 |
| `openclaw analytics suggestions` | 查看优化建议 |
| `openclaw experiment create` | 创建A/B测试 |
| `openclaw experiment results` | 查看测试结果 |

## 快捷命令

### 别名配置

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
alias oc='openclaw'
alias ocs='openclaw status'
alias ocd='openclaw doctor'
alias ocl='openclaw logs -f'
alias oct='openclaw tui'
```

### 常用组合

```bash
# 快速诊断并修复
openclaw doctor --fix && openclaw restart

# 查看完整状态
openclaw status && openclaw models show && openclaw channels list

# 备份配置
tar -czf openclaw_backup_$(date +%Y%m%d).tar.gz ~/.openclaw/

# 清理缓存
openclaw workspace clean --cache && openclaw restart
```

## 全局选项

| 选项 | 描述 |
|------|------|
| `-v, --version` | 显示版本 |
| `-h, --help` | 显示帮助 |
| `--config <path>` | 指定配置文件 |
| `--verbose` | 详细输出 |
| `--quiet` | 静默模式 |
| `--no-color` | 禁用颜色输出 |

---

更多命令详情，请使用 `openclaw <command> --help` 查看。
