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
        { text: 'Python核心教程', link: '/pages/python/core/' },
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
      text: 'Python',
      items: [
        { text: 'Python核心教程', link: 'core/' },
        { text: 'NumPy教程', link: 'numpy/' },
        { text: 'Pandas教程', link: 'pandas/' },
        { text: 'Matplotlib教程', link: 'matplotlib/' }
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
