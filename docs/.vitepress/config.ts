import { defineConfig, type DefaultTheme } from 'vitepress'

export default defineConfig({
  title: 'PyLLM',
  description: 'Python与大模型开发教程，从入门到实战',

  base: '/pyllm/',
  cleanUrls: true,
  head: [
    ['link', { rel: 'icon', href: '/pyllm/pyllm-logo.svg', type: 'image/svg+xml' }]
  ],

  themeConfig: {
    logo: '/pyllm-logo.svg',
    nav: nav(),

    search: {
      provider: 'local'
    },

    sidebar: {
      '/pages/python/': { base: '/pages/python/', items: sidebarPython() },
      '/pages/llm/': { base: '/pages/llm/', items: sidebarLLM() },
      '/pages/database/': { base: '/pages/database/', items: sidebarDatabase() },
      '/pages/ai-coding/': { base: '/pages/ai-coding/', items: sidebarAICoding() }
    },

    footer: {
      message: '基于 MIT 许可发布',
      copyright: '版权所有 © 2024-至今'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    outline: {
      label: '页面导航'
    },

    lastUpdated: {
      text: '最后更新于'
    },

    notFound: {
      title: '页面未找到',
      quote: '但如果你不改变方向，并且继续寻找，你可能最终会到达你所前往的地方。',
      linkLabel: '前往首页',
      linkText: '带我回首页'
    },

    langMenuLabel: '多语言',
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    skipToContentLabel: '跳转到内容'
  }
})

function nav(): DefaultTheme.NavItem[] {
  return [
    { text: '首页', link: '/' },
    {
      text: 'Python',
      activeMatch: '/pages/python/',
      items: [
        { text: 'Python核心教程', link: '/pages/python/core/00-intro' },
        { text: 'NumPy教程', link: '/pages/python/numpy/' },
        { text: 'Pandas教程', link: '/pages/python/pandas/' },
        { text: 'Matplotlib教程', link: '/pages/python/matplotlib/' }
      ]
    },
    {
      text: 'LLM',
      activeMatch: '/pages/llm/',
      items: [
        { text: 'PyTorch Lightning教程', link: '/pages/llm/pytorch-lightning/' },
        { text: 'Hugging Face Transformers教程', link: '/pages/llm/transformers/' },
        { text: 'Ollama教程', link: '/pages/llm/ollama/' },
        { text: 'vLLM教程', link: '/pages/llm/vllm/' },
        { text: 'LangChain教程', link: '/pages/llm/langchain/' }
      ]
    },
    {
      text: '数据库',
      activeMatch: '/pages/database/',
      items: [
        { text: 'PG Vector教程', link: '/pages/database/pgvector/' },
        { text: 'Milvus教程', link: '/pages/database/milvus/' },
        { text: 'Chroma教程', link: '/pages/database/chroma/' },
        { text: 'Faiss教程', link: '/pages/database/faiss/' },
        { text: 'DuckDB教程', link: '/pages/database/duckdb/' },
        { text: 'LanceDB教程', link: '/pages/database/lancedb/' }
      ]
    },
    {
      text: 'AI编程',
      activeMatch: '/pages/ai-coding/',
      items: [
        { text: 'OpenClaw教程', link: '/pages/ai-coding/openclaw/' },
        { text: 'OpenCode教程', link: '/pages/ai-coding/opencode/' },
        { text: 'DeepSeek教程', link: '/pages/ai-coding/deepseek/' }
      ]
    },
    { text: '博客', link: '/pages/blog/' },
    { text: '关于', link: '/pages/about/' }
  ]
}

function sidebarPython(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: 'Python核心教程',
      collapsed: false,
      items: [
        { text: '基本介绍', link: 'core/00-intro' },
        { text: '安装与环境搭建', link: 'core/01-install' }
      ]
    },
    {
      text: '基础',
      collapsed: false,
      items: [
        { text: '基本类型', link: 'core/10-data-number' },
        { text: '字符串与编码', link: 'core/11-data-str' },
        { text: '变量和对象', link: 'core/12-data-object' },
        { text: '条件、循环与异常', link: 'core/13-data-ifwhile' }
      ]
    },
    {      text: '函数',
      collapsed: false,
      items: [
        { text: '函数基础', link: 'core/20-func-basics' },
        { text: '高级参数特性', link: 'core/21-func-advanced' },
        { text: '函数式编程工具', link: 'core/22-func-functional' },
        { text: '闭包与装饰器', link: 'core/23-func-closure' },
        { text: '递归与特殊函数', link: 'core/25-func-special' }
      ]
    },
    {
      text: '数据结构',
      collapsed: false,
      items: [
        { text: '列表', link: 'core/30-struct-list' },
        { text: '元组', link: 'core/31-struct-tuple' },
        { text: '字典', link: 'core/32-struct-dict' },
        { text: '集合', link: 'core/33-struct-set' }
      ]
    },
    {
      text: '集合',
      collapsed: false,
      items: [
        { text: 'dequeue', link: 'core/40-collection-dequeue' },
        { text: 'defaultdict', link: 'core/41-collection-defaultdict' },
        { text: 'Counter', link: 'core/42-collection-counter' },
        { text: 'namedtuple', link: 'core/43-collection-namedtuple' }
      ]
    },
    {
      text: '高级特性',
      collapsed: false,
      items: [
        { text: '迭代器', link: 'core/50-pro-iterator' },
        { text: '推导式', link: 'core/51-pro-comprehensions' },
        { text: '生成器', link: 'core/54-pro-generator' }
      ]
    },
    {
      text: '模块',
      collapsed: false,
      items: [
        { text: '模块基础', link: 'core/60-module-basics' },
        { text: '导入', link: 'core/61-module-import' },
        { text: '组织', link: 'core/62-module-package' },
        { text: '安装与发布', link: 'core/63-module-distribution' },
        { text: '高级特性', link: 'core/64-module-advanced' }
      ]
    },
    {
      text: '面向对象',
      collapsed: false,
      items: [
        { text: '类与对象基础', link: 'core/70-oop-basics' },
        { text: '属性与方法', link: 'core/71-oop-attributes' },
        { text: '继承与多态', link: 'core/72-oop-inheritance' },
        { text: '魔术方法', link: 'core/73-oop-magic' },
        { text: '封装与访问控制', link: 'core/74-oop-encapsulation' },
        { text: '高级特性', link: 'core/75-oop-advanced' },
        { text: '设计模式', link: 'core/76-oop-patterns' }
      ]
    },
    {    
      text: 'IO',
      collapsed: false,
      items: [
        { text: '文件操作基础', link: 'core/80-io-file' },
        { text: '路径操作', link: 'core/81-io-path' },
        { text: '上下文管理器', link: 'core/82-io-context' },
        { text: '文本与二进制', link: 'core/83-io-text-binary' },
        { text: '数据序列化', link: 'core/84-io-serialization' },
        { text: '标准输入输出', link: 'core/85-io-stdio' }
      ]
    },
    {    
      text: '并发',
      collapsed: false,
      items: [
        { text: '并发编程基础', link: 'core/90-concurrency-basics' },
        { text: '多进程编程', link: 'core/91-concurrency-multiprocess' },
        { text: '多线程编程', link: 'core/92-concurrency-threading' },
        { text: '异步编程', link: 'core/93-concurrency-async' },
        { text: '并发进阶与实践', link: 'core/94-concurrency-advanced' }
      ]
    },
    {    
      text: '常用模块',
      collapsed: false,
      items: [
        { text: '日期时间与随机数', link: 'core/X0-datetime-random' },
        { text: '正则表达式', link: 'core/X1-regex' },
        { text: '系统与进程', link: 'core/X2-system-process' },
        { text: '工具模块', link: 'core/X3-tools' }
      ]
    }
  ]
}

function sidebarLLM(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: 'LLM',
      items: [
        { text: 'PyTorch Lightning教程', link: 'pytorch-lightning/' },
        { text: 'Hugging Face Transformers教程', link: 'transformers/' },
        { text: 'Ollama教程', link: 'ollama/' },
        { text: 'vLLM教程', link: 'vllm/' },
        { text: 'LangChain教程', link: 'langchain/' }
      ]
    }
  ]
}

function sidebarDatabase(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '数据库',
      items: [
        { text: 'PG Vector教程', link: 'pgvector/' },
        { text: 'Milvus教程', link: 'milvus/' },
        { text: 'Chroma教程', link: 'chroma/' },
        { text: 'Faiss教程', link: 'faiss/' },
        { text: 'DuckDB教程', link: 'duckdb/' },
        { text: 'LanceDB教程', link: 'lancedb/' }
      ]
    }
  ]
}

function sidebarAICoding(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: 'AI编程',
      items: [
        { text: 'OpenClaw教程', link: 'openclaw/' },
        { text: 'OpenCode教程', link: 'opencode/' },
        { text: 'DeepSeek教程', link: 'deepseek/' }
      ]
    }
  ]
}
