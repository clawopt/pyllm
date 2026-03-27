# 安全配置检查清单

这一章提供一份完整的安全配置检查清单，帮助你确保OpenClaw的安全运行。

## 基础安全配置

### 认证与访问控制

- [ ] **启用Web界面认证**
  ```yaml
  channels:
    web:
      auth:
        enabled: true
        username: "admin"
        password_hash: "$2a$10$..."
  ```

- [ ] **设置强密码**
  - 长度至少12位
  - 包含大小写字母、数字、特殊字符
  - 不使用常见密码

- [ ] **配置访问白名单**
  ```yaml
  security:
    ip_whitelist:
      - "192.168.1.0/24"
      - "10.0.0.0/8"
  ```

- [ ] **禁用公网直接访问**
  - 使用反向代理（Nginx）
  - 配置HTTPS
  - 或仅绑定到127.0.0.1

### API Key安全

- [ ] **使用环境变量存储API Key**
  ```bash
  export ALIBABA_API_KEY="sk-xxx"
  # 不要硬编码在配置文件中
  ```

- [ ] **定期轮换API Key**
  - 建议每3个月更换一次
  - 发现泄露立即更换

- [ ] **设置API Key使用限额**
  ```yaml
  model:
    rate_limit:
      requests_per_minute: 60
      tokens_per_day: 1000000
  ```

- [ ] **监控API Key使用情况**
  ```bash
  openclaw models usage --alert
  ```

## 技能安全

### 安装前检查

- [ ] **检查技能来源**
  - 🟢 官方认证：安全
  - 🟡 社区验证：基本安全
  - ⚪ 新发布：需要审查
  - 🔴 安全警告：禁止安装

- [ ] **检查技能权限**
  ```bash
  openclaw hub show <skill_name> --permissions
  ```
  - 权限是否合理？
  - 是否请求过多权限？

- [ ] **审查技能源码**
  ```bash
  openclaw hub show <skill_name> --source
  ```
  - 是否有可疑网络请求？
  - 是否执行危险命令？
  - 是否读取敏感文件？

### 安装后管理

- [ ] **定期扫描已安装技能**
  ```bash
  openclaw secureclaw scan --all
  ```

- [ ] **及时更新技能**
  ```bash
  openclaw hub update --all
  ```

- [ ] **清理不用的技能**
  ```bash
  openclaw hub list --unused
  openclaw hub uninstall <skill_name>
  ```

## 网络安全

### 端口与防火墙

- [ ] **仅开放必要端口**
  ```bash
  # 开放OpenClaw端口
  sudo ufw allow 18789/tcp
  
  # 关闭不必要的端口
  sudo ufw deny <port>
  ```

- [ ] **配置防火墙规则**
  ```bash
  # 仅允许特定IP访问
  sudo ufw allow from 192.168.1.0/24 to any port 18789
  ```

- [ ] **检查端口监听**
  ```bash
  netstat -tunlp | grep LISTEN
  ```

### HTTPS配置

- [ ] **启用HTTPS**
  ```nginx
  server {
      listen 443 ssl;
      ssl_certificate /path/to/cert.pem;
      ssl_certificate_key /path/to/key.pem;
  }
  ```

- [ ] **使用有效的SSL证书**
  - Let's Encrypt（免费）
  - 或购买商业证书

- [ ] **配置HTTP到HTTPS重定向**
  ```nginx
  server {
      listen 80;
      return 301 https://$host$request_uri;
  }
  ```

## 数据安全

### 敏感信息保护

- [ ] **配置敏感信息过滤**
  ```yaml
  memory:
    privacy:
      sensitive_detection: true
      sensitive_types:
        - password
        - api_key
        - credit_card
  ```

- [ ] **加密存储凭证**
  ```bash
  openclaw credentials encrypt
  ```

- [ ] **配置日志脱敏**
  ```yaml
  logging:
    sensitive_filter: true
  ```

### 数据备份

- [ ] **配置自动备份**
  ```yaml
  backup:
    enabled: true
    interval: 86400
    max_backups: 7
    path: ~/.openclaw/backups/
  ```

- [ ] **验证备份可恢复**
  ```bash
  openclaw backup verify
  ```

- [ ] **异地备份重要数据**
  - 云存储
  - 或其他服务器

## 用户权限

### 多账户安全

- [ ] **为每个用户创建独立账户**
  ```bash
  openclaw account create --name <username>
  ```

- [ ] **配置最小权限原则**
  ```yaml
  account:
    permissions:
      - chat
      - file_ops
    # 不授予危险权限
  ```

- [ ] **设置使用限额**
  ```yaml
  limits:
    daily_messages: 100
    daily_tokens: 10000
  ```

### 渠道安全

- [ ] **配置渠道白名单**
  ```yaml
  channels:
    telegram:
      allowlist:
        - 123456789
  ```

- [ ] **禁用未使用的渠道**
  ```bash
  openclaw channels disable <channel>
  ```

## 监控与告警

### 日志监控

- [ ] **启用详细日志**
  ```yaml
  logging:
    level: info
    file: ~/.openclaw/logs/openclaw.log
  ```

- [ ] **配置日志轮转**
  ```yaml
  logging:
    max_size: 10MB
    max_files: 5
  ```

- [ ] **定期检查日志**
  ```bash
  openclaw logs --filter "error|warning"
  ```

### 异常告警

- [ ] **配置错误率告警**
  ```yaml
  monitoring:
    alerts:
      - name: high_error_rate
        condition: error_rate > 5%
        action: notify
  ```

- [ ] **配置成本告警**
  ```yaml
  monitoring:
    alerts:
      - name: high_cost
        condition: daily_cost > 10
        action: notify
  ```

## 定期安全检查

### 每日检查

```bash
# 运行健康检查
openclaw doctor

# 查看异常日志
openclaw logs --filter "error|warning" --today

# 检查API使用量
openclaw models usage --today
```

### 每周检查

```bash
# 扫描技能安全
openclaw secureclaw scan --all

# 检查账户活动
openclaw account list --activity

# 查看安全报告
openclaw security report
```

### 每月检查

```bash
# 更新所有技能
openclaw hub update --all

# 轮换API Key
openclaw credentials rotate

# 验证备份
openclaw backup verify

# 安全审计
openclaw security audit
```

## 安全检查清单总结

| 类别 | 检查项 | 频率 |
|------|--------|------|
| 认证 | 密码强度、白名单 | 首次配置 |
| API Key | 存储、轮换、限额 | 每月 |
| 技能 | 来源、权限、源码 | 安装前 |
| 网络 | 端口、防火墙、HTTPS | 首次配置 |
| 数据 | 脱敏、备份、加密 | 每周 |
| 用户 | 权限、限额 | 创建账户时 |
| 监控 | 日志、告警 | 每日 |

---

安全是一个持续的过程，请定期执行检查清单，确保OpenClaw的安全运行。
