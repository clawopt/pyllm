# PyLLM —— Python 与大模型教程网站

PyLLM 是一个基于 VitePress 构建的中文学习网站，提供系统化的 Python 基础、主流 LLM 框架、向量数据库以及 AI 编程实战内容，配合博客与精选推荐，帮助你快速迈向大模型应用开发。

## 网站定位
- Python 基础到数据科学常用库的系统教程
- 主流大模型与生态工具的实践指南
- 向量数据库选型与工程实践
- AI 编程工具与模型应用案例
- 博客与精选资源推荐

## 主要栏目
- **Python**：核心语法、NumPy、Pandas、Matplotlib
- **LLM**：PyTorch Lightning、Transformers、Ollama、vLLM、LangChain
- **数据库**：PG Vector、Milvus、Chroma、Faiss、DuckDB、LanceDB
- **AI 编程**：OpenClaw、OpenCode、DeepSeek
- **博客**：技术回顾、实践技巧、选型对比
- **关于**：站点介绍与联系

## 目录结构（部分）
```
docs/
  .vitepress/
    config.ts           # 站点配置
    theme/
      components/
        CustomHero.vue  # 首页 Hero + 精选推荐 + 最新博客
        HomeCategories.vue  # 首页课程导航与列表
  pages/
    blog/
      index.md
      llm-2024-review.md
      python-tips-2024.md
      vector-db-comparison.md
    python/ ...         # Python 教程
    llm/ ...            # LLM 教程
    database/ ...       # 数据库教程
    ai-coding/ ...      # AI 编程
```

## 快速开始
在 `docs` 目录下运行：
```bash
cd docs
npm install
npm run dev      # 本地开发
npm run build    # 生成静态文件到 .vitepress/dist
npm run preview  # 本地预览（默认 4173 端口）
```
也可以使用：
```bash
npm run restart  # 杀掉端口占用，构建并以 8085 端口预览
```

## 部署
- 已集成 GitHub Pages 工作流（`.github/workflows/pages.yml`）
- 推送到主分支后自动构建并发布到 Pages

## 自定义与扩展
- 修改站点标题与导航：`docs/.vitepress/config.ts`
- 首页样式与布局：`docs/.vitepress/theme/components/CustomHero.vue`、`HomeCategories.vue`
- 资源文件：`docs/public/`（例如 `pyllm-logo.svg`、`paperily-home.png`）
- 添加课程/文章：在 `docs/pages` 下对应目录新增 `index.md` 或文章文件

## 特色设计
- **Hero 两行标题**：第一行品牌色，第二行略小字号
- **精选推荐**：位于首页中部列，展示 5 条文本链接（可自定义）
- **最新博客**：右侧列表，提供“更多”链接（右下角）
- **课程导航**：左侧扁平图标 + 分类卡片，响应式布局

## 许可
本站内容默认采用 MIT License（部分外部资源除外），欢迎学习与复用。

---
如需新增课程、优化样式、接入更多推荐资源，请在 Issue 或 PR 中提出。
