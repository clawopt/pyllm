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
      '/pages/python/core/': { base: '/pages/python/core/', items: sidebarPython() },
      '/pages/python/numpy/': { base: '/pages/python/numpy/', items: sidebarNumPy() },
      '/pages/llm/': { base: '/pages/llm/', items: sidebarLLM() },
      '/pages/database/': { base: '/pages/database/', items: sidebarDatabase() },
      '/pages/ai-coding/': { base: '/pages/ai-coding/openclaw/', items: sidebarAICoding() }
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
        { text: 'NumPy教程', link: '/pages/python/numpy/01-00-numpy-intro' },
        { text: 'Pandas教程', link: '/pages/python/pandas/' },
      ]
    },
    {
      text: 'LLM',
      activeMatch: '/pages/llm/',
      items: [
        { text: 'LangChain教程', link: '/pages/llm/langchain/' },
        { text: 'Hugging Face Transformers教程', link: '/pages/llm/transformers/' },
        { text: 'Ollama教程', link: '/pages/llm/ollama/' },
        { text: 'vLLM教程', link: '/pages/llm/vllm/' },
        { text: 'PyTorch Lightning教程', link: '/pages/llm/pytorch-lightning/' }
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
      text: 'Python编程：从入门到实战',
      items: [
        {
          text: '第一章 介绍与环境搭建',
          collapsed: false,
          items: [
            { text: '基本介绍', link: '00-intro' },
            { text: '安装与环境搭建', link: '01-install' }
          ]
        },
        {
          text: '第二章 数据类型与分支控制',
          collapsed: false,
          items: [
            { text: '基本类型', link: '10-data-number' },
            { text: '字符串与编码', link: '11-data-str' },
            { text: '变量和对象', link: '12-data-object' },
            { text: '条件、循环与异常', link: '13-data-ifwhile' }
          ]
        },
        {
          text: '第三章 函数与参数',
          collapsed: false,
          items: [
            { text: '函数基础', link: '20-func-basics' },
            { text: '高级参数特性', link: '21-func-advanced' },
            { text: '函数式编程工具', link: '22-func-functional' },
            { text: '闭包与装饰器', link: '23-func-closure' },
            { text: '递归与特殊函数', link: '25-func-special' }
          ]
        },
        {
          text: '第四章 数据结构',
          collapsed: false,
          items: [
            { text: '列表', link: '30-struct-list' },
            { text: '元组', link: '31-struct-tuple' },
            { text: '字典', link: '32-struct-dict' },
            { text: '集合', link: '33-struct-set' }
          ]
        },
        {
          text: '第五章 集合与映射',
          collapsed: false,
          items: [
            { text: 'dequeue', link: '40-collection-dequeue' },
            { text: 'defaultdict', link: '41-collection-defaultdict' },
            { text: 'Counter', link: '42-collection-counter' },
            { text: 'namedtuple', link: '43-collection-namedtuple' }
          ]
        },
        {
          text: '第六章 高级特性',
          collapsed: false,
          items: [
            { text: '迭代器', link: '50-pro-iterator' },
            { text: '推导式', link: '51-pro-comprehensions' },
            { text: '生成器', link: '54-pro-generator' }
          ]
        },
        {
          text: '第七章 模块使用',
          collapsed: false,
          items: [
            { text: '模块基础', link: '60-module-basics' },
            { text: '导入', link: '61-module-import' },
            { text: '组织', link: '62-module-package' },
            { text: '安装与发布', link: '63-module-distribution' },
            { text: '高级特性', link: '64-module-advanced' }
          ]
        },
        {
          text: '第八章 面向对象编程',
          collapsed: false,
          items: [
            { text: '类与对象基础', link: '70-oop-basics' },
            { text: '属性与方法', link: '71-oop-attributes' },
            { text: '继承与多态', link: '72-oop-inheritance' },
            { text: '魔术方法', link: '73-oop-magic' },
            { text: '封装与访问控制', link: '74-oop-encapsulation' },
            { text: '高级特性', link: '75-oop-advanced' },
            { text: '设计模式', link: '76-oop-patterns' }
          ]
        },
        {
          text: '第九章 输入输出',
          collapsed: false,
          items: [
            { text: '文件操作基础', link: '80-io-file' },
            { text: '路径操作', link: '81-io-path' },
            { text: '上下文管理器', link: '82-io-context' },
            { text: '文本与二进制', link: '83-io-text-binary' },
            { text: '数据序列化', link: '84-io-serialization' },
            { text: '标准输入输出', link: '85-io-stdio' }
          ]
        },
        {
          text: '第十章 并发编程',
          collapsed: false,
          items: [
            { text: '并发编程基础', link: '90-concurrency-basics' },
            { text: '多进程编程', link: '91-concurrency-multiprocess' },
            { text: '多线程编程', link: '92-concurrency-threading' },
            { text: '异步编程', link: '93-concurrency-async' },
            { text: '并发进阶与实践', link: '94-concurrency-advanced' }
          ]
        },
        {
          text: '第十章 常用模块',
          collapsed: false,
          items: [
            { text: '日期时间与随机数', link: 'X0-datetime-random' },
            { text: '正则表达式', link: 'X1-regex' },
            { text: '系统与进程', link: 'X2-system-process' },
            { text: '工具模块', link: 'X3-tools' }
          ]
        }
      ]
    }
  ]
}

function sidebarNumPy(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: 'Numpy学习指南',
      items: [
        {
          text: '第1章 基础概念',
          collapsed: false,
          items: [
            { text: 'NumPy是什么', link: '01-00-numpy-intro' },
            { text: 'NumPy优势对比', link: '01-01-numpy-vs-list' },
            { text: '安装与配置', link: '01-02-install-config' },
            { text: '导入与约定', link: '01-03-import-convention' },
            { text: 'NumPy与LLM开发', link: '01-04-why-numpy-llm' }
          ]
        },
        {
          text: '第2章 ndarray属性',
          collapsed: false,
          items: [
            { text: 'ndarray属性', link: '02-01-ndarray-attributes' },
            { text: 'dtype深入理解', link: '02-02-dtype-deep-dive' },
            { text: '从列表创建', link: '02-03-ndarray-from-list' },
            { text: 'AI中的应用', link: '02-04-ndarray-in-ai' }
          ]
        },
        {
          text: '第3章 数组创建',
          collapsed: false,
          items: [
            { text: 'zeros与ones', link: '03-01-zeros-ones' },
            { text: 'eye与identity', link: '03-02-eye-identity' },
            { text: 'arange与linspace', link: '03-03-arange-linspace' },
            { text: 'tile与meshgrid', link: '03-04-tile-meshgrid' },
            { text: '随机数组', link: '03-05-random-arrays' },
            { text: '从数据创建', link: '03-06-array-from-data' }
          ]
        },
        {
          text: '第4章 形状操作',
          collapsed: false,
          items: [
            { text: 'shape与reshape', link: '04-01-shape-reshape' },
            { text: 'flatten与ravel', link: '04-02-flatten-ravel' },
            { text: '转置操作', link: '04-03-transpose' },
            { text: 'expand与squeeze', link: '04-04-expand-squeeze' },
            { text: 'concatenate与stack', link: '04-05-concatenate-stack' },
            { text: 'split分割', link: '04-06-split-array' }
          ]
        },
        {
          text: '第5章 索引与切片',
          collapsed: false,
          items: [
            { text: '基本索引', link: '05-01-basic-indexing' },
            { text: '整数数组索引', link: '05-02-integer-array-indexing' },
            { text: '布尔掩码', link: '05-03-boolean-masking' },
            { text: '花式索引', link: '05-04-fancy-indexing' },
            { text: 'LLM嵌入查找', link: '05-05-llm-embedding-lookup' }
          ]
        },
        {
          text: '第6章 运算与性能',
          collapsed: false,
          items: [
            { text: '元素级运算', link: '06-01-element-wise-operations' },
            { text: '向量化性能', link: '06-02-vectorization-performance' },
            { text: '矩阵乘法', link: '06-03-matrix-multiplication' },
            { text: 'LLM自注意力', link: '06-04-llm-self-attention' }
          ]
        },
        {
          text: '第7章 广播机制',
          collapsed: false,
          items: [
            { text: '广播规则', link: '07-01-broadcasting-rules' },
            { text: '数组扩张', link: '07-02-array-expansion' },
            { text: 'LLM位置编码', link: '07-03-llm-positional-encoding' }
          ]
        },
        {
          text: '第8章 数学函数',
          collapsed: false,
          items: [
            { text: '指数与对数', link: '08-01-exp-log' },
            { text: '激活函数', link: '08-02-activation-functions' },
            { text: 'clip截断', link: '08-03-clip-functions' }
          ]
        },
        {
          text: '第9章 统计与损失',
          collapsed: false,
          items: [
            { text: '基础统计', link: '09-01-statistics-basic' },
            { text: 'axis统计', link: '09-02-axis-statistics' },
            { text: 'cumsum累加', link: '09-03-cumsum-operations' },
            { text: 'LLM困惑度', link: '09-04-llm-perplexity' }
          ]
        },
        {
          text: '第10章 线性代数',
          collapsed: false,
          items: [
            { text: '点积', link: '10-01-dot-product' },
            { text: '特征值与特征向量', link: '10-02-eigenvalues-eigenvectors' },
            { text: '奇异值分解', link: '10-03-svd' },
            { text: '矩阵求逆与伪逆', link: '10-04-pinv' },
            { text: '范数计算', link: '10-05-norm' }
          ]
        },
        {
          text: '第11章 随机数生成',
          collapsed: false,
          items: [
            { text: '随机种子与可复现性', link: '11-01-random-seed' },
            { text: '常见分布', link: '11-02-common-distributions' },
            { text: '随机采样', link: '11-03-random-sampling' },
            { text: '随机排列', link: '11-04-random-permutation' },
            { text: '随机初始化权重', link: '11-05-random-initialization' }
          ]
        },
        {
          text: '第12章 内存布局',
          collapsed: false,
          items: [
            { text: 'C顺序与Fortran顺序', link: '12-01-c-fortran-order' },
            { text: '视图与副本', link: '12-02-view-vs-copy' },
            { text: 'ndarray.view', link: '12-03-view-ndarray' },
            { text: '原地操作', link: '12-04-inplace-operations' }
          ]
        },
        {
          text: '第13章 结构化数组',
          collapsed: false,
          items: [
            { text: '创建结构化数组', link: '13-01-structured-arrays' },
            { text: '字段访问与操作', link: '13-02-field-access' },
            { text: 'HuggingFace导出', link: '13-03-huggingface-export' }
          ]
        },
        {
          text: '第14章 文件IO',
          collapsed: false,
          items: [
            { text: '文本文件IO', link: '14-01-text-io' },
            { text: '二进制文件IO', link: '14-02-binary-io' },
            { text: '内存映射', link: '14-03-memmap' }
          ]
        },
        {
          text: '实战一：词嵌入矩阵',
          collapsed: false,
          items: [
            { text: '词汇表与索引映射', link: '15-01-vocab-index' },
            { text: '嵌入矩阵创建', link: '15-02-embedding-matrix' },
            { text: '嵌入查找实现', link: '15-03-embedding-lookup' },
            { text: '词向量相似度可视化', link: '15-04-visualize-similarity' }
          ]
        },
        {
          text: '实战二：多头自注意力',
          collapsed: false,
          items: [
            { text: 'QKV投影矩阵', link: '16-01-qkv-projections' },
            { text: '缩放点积注意力', link: '16-02-scaled-attention' },
            { text: '因果掩码', link: '16-03-causal-mask' },
            { text: '前向传播模拟', link: '16-04-forward-propagation' }
          ]
        },
        {
          text: '实战三：LayerNorm与残差',
          collapsed: false,
          items: [
            { text: 'LayerNorm实现', link: '17-01-layer-norm' },
            { text: '残差连接模拟', link: '17-02-residual-connection' },
            { text: 'NumPy vs PyTorch对比', link: '17-03-numpy-vs-pytorch' }
          ]
        },
        {
          text: '实战四：GPT文本生成',
          collapsed: false,
          items: [
            { text: '预训练嵌入概述', link: '18-01-pretrained-embeddings' },
            { text: 'Transformer解码器', link: '18-02-transformer-decoder' },
            { text: '温度采样', link: '18-03-temperature-sampling' },
            { text: '简单文本序列生成', link: '18-04-text-generation' }
          ]
        },
        {
          text: '实战五：LoRA低秩适配',
          collapsed: false,
          items: [
            { text: 'LoRA原理', link: '19-01-lora-principle' },
            { text: 'LoRA前向传播', link: '19-02-lora-forward' },
            { text: '参数量节省', link: '19-03-parameter-savings' }
          ]
        },
        {
          text: '实战六：HuggingFace集成',
          collapsed: false,
          items: [
            { text: 'tokenizer转NumPy', link: '20-01-tokenizer-to-numpy' },
            { text: '数据集转NumPy', link: '20-02-datasets-to-numpy' },
            { text: 'NumPy DataLoader', link: '20-03-numpy-dataloader' }
          ]
        },
        {
          text: '实战七：框架互操作',
          collapsed: false,
          items: [
            { text: 'PyTorch互操作', link: '21-01-torch-numpy' },
            { text: 'TensorFlow互操作', link: '21-02-tensorflow-numpy' },
            { text: '预处理与训练', link: '21-03-preprocessing-training' }
          ]
        },
        {
          text: '实战八：LLM评估',
          collapsed: false,
          items: [
            { text: '困惑度计算', link: '22-01-perplexity' },
            { text: 'BLEU/ROUGE指标', link: '22-02-bleu-rouge' },
            { text: '文本多样性评估', link: '22-03-text-diversity' }
          ]
        }
      ]
    }
  ]
}

function sidebarLLM(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: 'LLM',
      items: [
        { text: 'LangChain教程', link: 'langchain/' },
        { text: 'Hugging Face Transformers教程', link: 'transformers/' },
        { text: 'Ollama教程', link: 'ollama/' },
        { text: 'vLLM教程', link: 'vllm/' },
        { text: 'PyTorch Lightning教程', link: 'pytorch-lightning/' }
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
      text: 'OpenClaw从入门到精通',
      items: [
        {
          text: '第1章 基础概念',
          collapsed: false,
          items: [
            { text: 'OpenClaw是什么', link: '01-01-what-is-openclaw' },
            { text: 'OpenClaw能做什么', link: '01-02-what-can-openclaw-do' },
            { text: '技术架构概览', link: '01-03-architecture-overview' }
          ]
        },
        {
          text: '第2章 部署指南',
          collapsed: false,
          items: [
            { text: '部署前置准备', link: '02-01-deployment-prerequisites' },
            { text: '本地部署', link: '02-02-local-deployment' },
            { text: '云端部署', link: '02-03-cloud-deployment' },
            { text: '初始化配置', link: '02-04-initialization' }
          ]
        },
        {
          text: '第3章 配置定制',
          collapsed: false,
          items: [
            { text: '配置文件详解', link: '03-01-config-files' },
            { text: '定义AI身份', link: '03-02-define-soul' },
            { text: '设置用户偏好', link: '03-03-user-preferences' },
            { text: '定时任务配置', link: '03-04-heartbeat-tasks' }
          ]
        },
        {
          text: '第4章 模型配置',
          collapsed: false,
          items: [
            { text: '支持的AI模型提供商', link: '04-01-supported-models' },
            { text: 'API Key配置', link: '04-02-api-key-config' },
            { text: '模型策略与成本优化', link: '04-03-model-strategy' },
            { text: '本地模型部署', link: '04-04-ollama-deployment' }
          ]
        },
        {
          text: '第5章 技能系统',
          collapsed: false,
          items: [
            { text: 'Skills技能系统概述', link: '05-01-skills-overview' },
            { text: '必备核心技能', link: '05-02-core-skills' },
            { text: '技能管理命令', link: '05-03-skill-management' },
            { text: '自定义Skill开发', link: '05-04-custom-skill' },
            { text: '技能安全', link: '05-05-skill-security' }
          ]
        },
        {
          text: '第6章 消息渠道',
          collapsed: false,
          items: [
            { text: '支持的消息平台', link: '06-01-message-platforms' },
            { text: '渠道配置方法', link: '06-02-channel-config' },
            { text: '多账户管理', link: '06-03-multi-account' }
          ]
        },
        {
          text: '第7章 日常使用',
          collapsed: false,
          items: [
            { text: '访问方式', link: '07-01-access-methods' },
            { text: '常用CLI命令', link: '07-02-cli-commands' },
            { text: '常见问题排查', link: '07-03-troubleshooting' }
          ]
        },
        {
          text: '第8章 实战案例',
          collapsed: false,
          items: [
            { text: '自动发微信公众号文章', link: '08-01-wechat-article' },
            { text: '日报自动生成与发送', link: '08-02-daily-report' },
            { text: '7×24小时信息中枢', link: '08-03-info-hub' },
            { text: '个人知识库管理', link: '08-04-knowledge-base' }
          ]
        },
        {
          text: '第9章 进阶玩法',
          collapsed: false,
          items: [
            { text: '多Agent配置', link: '09-01-multi-agent' },
            { text: 'RAG检索增强生成', link: '09-02-rag' },
            { text: '上下文记忆配置', link: '09-03-context-memory' },
            { text: '数据分析与优化', link: '09-04-analytics' }
          ]
        },
        {
          text: '附录',
          collapsed: false,
          items: [
            { text: 'OpenClaw命令速查表', link: '10-01-command-reference' },
            { text: '常用Skill推荐清单', link: '10-02-skill-recommendations' },
            { text: '各模型提供商控制台地址', link: '10-03-model-providers' },
            { text: '安全配置检查清单', link: '10-04-security-checklist' },
            { text: '相关资源链接', link: '10-05-resources' }
          ]
        }
      ]
    }
  ]
}
