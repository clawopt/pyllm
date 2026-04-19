import {defineConfig, type DefaultTheme} from 'vitepress'

const BASE = process.env.BASE_URL || '/pyllm/'

export default defineConfig({
  title: 'PyLLM',
  description: 'Python与大模型开发教程，从入门到实战',

  base: BASE,
  cleanUrls: true,
  sitemap: {
    hostname: 'https://pyllm.com',
    lastmodDateOnly: false
  },
  vite: {
    build: {
      chunkSizeWarningLimit: 50000,
    },
  },
  head: [
    ['link', {rel: 'icon', href: '/pyllm-logo.svg', type: 'image/svg+xml'}],
    ['meta', {name: 'keywords', content: 'Python,LLM,大模型,LangChain,RAG,PyTorch,Transformers,Ollama,vLLM,DeepSeek,教程'}],
    ['meta', {name: 'author', content: 'PyLLM'}],
    ['meta', {property: 'og:type', content: 'website'}],
    ['meta', {property: 'og:site_name', content: 'PyLLM'}],
    ['meta', {property: 'og:locale', content: 'zh_CN'}],
    ['meta', {name: 'twitter:card', content: 'summary_large_image'}],
    ['meta', {name: 'twitter:site', content: '@pyllm'}],
    ['meta', {name: 'baidu-site-verification', content: 'pyllm'}],
    ['meta', {name: 'google-site-verification', content: 'pyllm'}],
  ],

  themeConfig: {
    logo: '/pyllm-logo.svg',
    nav: nav(),

    search: {provider: 'local'},

    sidebar: {
      '/pages/python/core/':
          {base: '/pages/python/core/', items: sidebarPython()},
      '/pages/python/numpy/':
          {base: '/pages/python/numpy/', items: sidebarNumPy()},
      '/pages/python/pandas/':
          {base: '/pages/python/pandas/', items: sidebarPandas()},
      '/pages/llm/langchain/':
          {base: '/pages/llm/langchain/', items: sidebarLangChain()},
      '/pages/llm/llamaindex/':
          {base: '/pages/llm/llamaindex/', items: sidebarLlamaIndex()},
      '/pages/llm/langgraph/':
          {base: '/pages/llm/langgraph/', items: sidebarLangGraph()},
      '/pages/llm/transformers/':
          {base: '/pages/llm/transformers/', items: sidebarTransformers()},
      '/pages/llm/ollama/':
          {base: '/pages/llm/ollama/', items: sidebarOllama()},
      '/pages/llm/vllm/':
          {base: '/pages/llm/vllm/', items: sidebarVllm()},
      '/pages/llm/pytorch/':
          {base: '/pages/llm/pytorch/', items: sidebarPytorch()},
      '/pages/database/chroma/':
          {base: '/pages/database/chroma/', items: sidebarChroma()},
      '/pages/database/faiss/':
          {base: '/pages/database/faiss/', items: sidebarFaiss()},
      '/pages/database/milvus/':
          {base: '/pages/database/milvus/', items: sidebarMilvus()},
      '/pages/database/pgvector/':
          {base: '/pages/database/pgvector/', items: sidebarPgvector()},
      '/pages/ai-coding/openclaw/':
          {base: '/pages/ai-coding/openclaw/', items: sidebarAICoding()},
      '/pages/ai-coding/opencode/':
          {base: '/pages/ai-coding/opencode/', items: sidebarOpenCode()},
      '/pages/ai-coding/deepseek/':
          {base: '/pages/ai-coding/deepseek/', items: sidebarDeepSeek()}
    },

    footer: {message: '基于 MIT 许可发布', copyright: '版权所有 © 2024-至今'},

    docFooter: {prev: '上一页', next: '下一页'},

    outline: {label: '页面导航'},

    lastUpdated: {text: '最后更新于'},

    notFound: {
      title: '页面未找到',
      quote:
          '但如果你不改变方向，并且继续寻找，你可能最终会到达你所前往的地方。',
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

function nav(): DefaultTheme
    .NavItem[]{return [
      {text: '首页', link: '/'}, {
        text: 'Python',
        activeMatch: '/pages/python/',
        items: [
          {text: 'Python核心教程', link: '/pages/python/core/'},
          {text: 'NumPy教程', link: '/pages/python/numpy/'},
          {text: 'Pandas教程', link: '/pages/python/pandas/'},
        ]
      },
      {
        text: 'LLM',
        activeMatch: '/pages/llm/',
        items: [
          {text: 'LangChain教程', link: '/pages/llm/langchain/'}, 
          {text: 'LlamaIndex教程', link: '/pages/llm/llamaindex/'},
          {text: 'LangGraph教程', link: '/pages/llm/langgraph/'},
          {text: 'Hugging Face Transformers教程',link: '/pages/llm/transformers/'},
          {text: 'Ollama教程', link: '/pages/llm/ollama/'},
          {text: 'vLLM教程', link: '/pages/llm/vllm/'},
          {text: 'PyTorch教程', link: '/pages/llm/pytorch/'}
        ]
      },
      {
        text: '数据库',
        activeMatch: '/pages/database/',
        items: [
          {text: 'PG Vector教程', link: '/pages/database/pgvector/'},
          {text: 'Milvus教程', link: '/pages/database/milvus/'},
          {text: 'Chroma教程', link: '/pages/database/chroma/'},
          {text: 'Faiss教程', link: '/pages/database/faiss/'},
        ]
      },
      {
        text: 'AI编程',
        activeMatch: '/pages/ai-coding/',
        items: [
          {text: 'OpenClaw教程', link: '/pages/ai-coding/openclaw/'},
          {text: 'OpenCode教程', link: '/pages/ai-coding/opencode/'},
          {text: 'DeepSeek教程', link: '/pages/ai-coding/deepseek/'}
        ]
      },
      {text: '博客', link: '/pages/blog/'},
      {text: '关于', link: '/pages/about/'}
    ]}

function
sidebarPython():
    DefaultTheme
    .SidebarItem[] {
      return [{
        text: 'Python编程：从入门到实战',
        items: [
          {
            text: '第一章 介绍与环境搭建',
            collapsed: false,
            items: [
              {text: '基本介绍', link: '00-intro'},
              {text: '安装与环境搭建', link: '01-install'}
            ]
          },
          {
            text: '第二章 数据类型与分支控制',
            collapsed: false,
            items: [
              {text: '基本类型', link: '10-data-number'},
              {text: '字符串与编码', link: '11-data-str'},
              {text: '变量和对象', link: '12-data-object'},
              {text: '条件、循环与异常', link: '13-data-ifwhile'}
            ]
          },
          {
            text: '第三章 函数与参数',
            collapsed: false,
            items: [
              {text: '函数基础', link: '20-func-basics'},
              {text: '高级参数特性', link: '21-func-advanced'},
              {text: '函数式编程工具', link: '22-func-functional'},
              {text: '闭包与装饰器', link: '23-func-closure'},
              {text: '递归与特殊函数', link: '25-func-special'}
            ]
          },
          {
            text: '第四章 数据结构',
            collapsed: false,
            items: [
              {text: '列表', link: '30-struct-list'},
              {text: '元组', link: '31-struct-tuple'},
              {text: '字典', link: '32-struct-dict'},
              {text: '集合', link: '33-struct-set'}
            ]
          },
          {
            text: '第五章 集合与映射',
            collapsed: false,
            items: [
              {text: 'dequeue', link: '40-collection-dequeue'},
              {text: 'defaultdict', link: '41-collection-defaultdict'},
              {text: 'Counter', link: '42-collection-counter'},
              {text: 'namedtuple', link: '43-collection-namedtuple'}
            ]
          },
          {
            text: '第六章 高级特性',
            collapsed: false,
            items: [
              {text: '迭代器', link: '50-pro-iterator'},
              {text: '推导式', link: '51-pro-comprehensions'},
              {text: '生成器', link: '54-pro-generator'}
            ]
          },
          {
            text: '第七章 模块使用',
            collapsed: false,
            items: [
              {text: '模块基础', link: '60-module-basics'},
              {text: '导入', link: '61-module-import'},
              {text: '组织', link: '62-module-package'},
              {text: '安装与发布', link: '63-module-distribution'},
              {text: '高级特性', link: '64-module-advanced'}
            ]
          },
          {
            text: '第八章 面向对象编程',
            collapsed: false,
            items: [
              {text: '类与对象基础', link: '70-oop-basics'},
              {text: '属性与方法', link: '71-oop-attributes'},
              {text: '继承与多态', link: '72-oop-inheritance'},
              {text: '魔术方法', link: '73-oop-magic'},
              {text: '封装与访问控制', link: '74-oop-encapsulation'},
              {text: '高级特性', link: '75-oop-advanced'},
              {text: '设计模式', link: '76-oop-patterns'}
            ]
          },
          {
            text: '第九章 输入输出',
            collapsed: false,
            items: [
              {text: '文件操作基础', link: '80-io-file'},
              {text: '路径操作', link: '81-io-path'},
              {text: '上下文管理器', link: '82-io-context'},
              {text: '文本与二进制', link: '83-io-text-binary'},
              {text: '数据序列化', link: '84-io-serialization'},
              {text: '标准输入输出', link: '85-io-stdio'}
            ]
          },
          {
            text: '第十章 并发编程',
            collapsed: false,
            items: [
              {text: '并发编程基础', link: '90-concurrency-basics'},
              {text: '多进程编程', link: '91-concurrency-multiprocess'},
              {text: '多线程编程', link: '92-concurrency-threading'},
              {text: '异步编程', link: '93-concurrency-async'},
              {text: '并发进阶与实践', link: '94-concurrency-advanced'}
            ]
          },
          {
            text: '第十章 常用模块',
            collapsed: false,
            items: [
              {text: '日期时间与随机数', link: 'X0-datetime-random'},
              {text: '正则表达式', link: 'X1-regex'},
              {text: '系统与进程', link: 'X2-system-process'},
              {text: '工具模块', link: 'X3-tools'}
            ]
          }
        ]
      }]
    }

function sidebarOllama(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'Ollama 教程',
    items: [
      {
        text: '第1章 Ollama 入门与快速上手',
        collapsed: false,
        items: [
          {text: '为什么需要本地运行大模型', link: '01-01-why-local-llm'},
          {text: '安装与环境准备', link: '01-02-installation-and-setup'},
          {text: '第一个模型：运行与对话', link: '01-03-first-model-and-conversation'},
          {text: '模型管理基础', link: '01-04-model-management'}
        ]
      },
      {
        text: '第2章 API 服务与编程接入',
        collapsed: false,
        items: [
          {text: '启动 API 服务', link: '02-01-starting-api-service'},
          {text: 'RESTful API 完整指南', link: '02-02-restful-api-guide'},
          {text: 'Python 接入实战', link: '02-03-python-integration'},
          {text: 'JavaScript / TypeScript 接入', link: '02-04-javascript-typescript-integration'},
          {text: '其他语言接入速查', link: '02-05-other-languages-integration'}
        ]
      },
      {
        text: '第3章 模型生态与选择指南',
        collapsed: false,
        items: [
          {text: 'Ollama 模型库全景', link: '03-01-ollama-library-overview'},
          {text: '模型命名与 Tag 规范', link: '03-02-model-naming-tag-spec'},
          {text: '如何为任务选择最佳模型', link: '03-3-task-based-model-selection'},
          {text: '模型安全性与合规', link: '03-4-security-compliance'}
        ]
      },
      {
        text: '第4章 Modelfile 与自定义模型',
        collapsed: false,
        items: [
          {text: 'Modelfile 基础语法', link: '04-01-modelfile-basics'},
          {text: 'SYSTEM 提示词工程', link: '04-02-system-prompt-engineering'},
          {text: 'PARAMETER 调优艺术', link: '04-03-parameter-tuning'},
          {text: 'ADAPTER：LoRA 适配器加载', link: '04-04-lora-adapter-loading'},
          {text: '从零创建自定义 GGUF 模型', link: '04-5-custom-gguf-conversion'}
        ]
      },
      {
        text: '第5章 多模态能力',
        collapsed: false,
        items: [
          {text: '图像理解模型', link: '05-1-image-understanding'},
          {text: '视频理解', link: '05-2-video-understanding'},
          {text: '多模态实战项目', link: '05-3-multimodal-projects'}
        ]
      },
      {
        text: '第6章 Embedding 模型与 RAG',
        collapsed: false,
        items: [
          {text: 'Embedding 模型入门', link: '06-1-embedding-fundamentals'},
          {text: '向量数据库集成', link: '06-2-vector-database-integration'},
          {text: '完整 RAG 系统实战', link: '06-3-complete-rag-system'},
          {text: '高级 RAG 技术', link: '06-4-advanced-rag'}
        ]
      },
      {
        text: '第7章 LangChain / LlamaIndex 集成',
        collapsed: false,
        items: [
          {text: 'LangChain + Ollama', link: '07-1-langchain-integration'},
          {text: 'LlamaIndex + Ollama', link: '07-2-llamaindex-integration'},
          {text: '其他框架集成速查', link: '07-3-other-frameworks'},
          {text: '生产级集成架构', link: '07-4-production-architecture'}
        ]
      },
      {
        text: '第8章 性能优化与资源管理',
        collapsed: false,
        items: [
          {text: '硬件层面的优化', link: '08-1-hardware-optimization'},
          {text: '推理参数调优', link: '08-2-inference-parameter-tuning'},
          {text: '并发与吞吐优化', link: '08-3-concurrency-throughput'},
          {text: '模型预加载与热管理', link: '08-4-preload-thermal'},
          {text: '量化深度指南', link: '08-5-quantization-guide'}
        ]
      },
      {
        text: '第9章 Web UI 与工具生态',
        collapsed: false,
        items: [
          {text: 'Open WebUI', link: '09-1-open-webui'},
          {text: 'Ollama WebUI（官方）', link: '09-2-official-webui'},
          {text: '第三方工具生态', link: '09-3-third-party-tools'},
          {text: 'IDE 集成', link: '09-4-ide-integrations'}
        ]
      },
      {
        text: '第10章 企业部署与运维',
        collapsed: false,
        items: [
          {text: 'Docker 部署深度实践', link: '10-1-docker-deployment'},
          {text: '安全加固', link: '10-2-security-hardening'},
          {text: '监控与可观测性', link: '10-3-monitoring-observability'},
          {text: '高可用与扩展', link: '10-4-high-availability-scaling'},
          {text: '成本分析与优化', link: '10-5-cost-analysis-tco'}
        ]
      }
    ]
  }]
}

function sidebarVllm(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'vLLM 教程',
    items: [
      {
        text: '第1章 vLLM 入门',
        collapsed: false,
        items: [
          {text: '为什么需要 vLLM', link: '01-01-why-vllm'},
          {text: '安装与环境搭建', link: '01-02-installation-setup'},
          {text: '启动第一个模型服务', link: '01-03-first-model-service'},
          {text: '项目结构与配置', link: '01-04-project-structure'}
        ]
      },
      {
        text: '第2章 PagedAttention 核心原理',
        collapsed: false,
        items: [
          {text: 'KV Cache 的问题', link: '02-01-kv-cache-problem'},
          {text: 'PagedAttention 设计思想', link: '02-02-pagedattention-design'},
          {text: 'PagedAttention 实现细节', link: '02-03-pagedattention-implementation'},
          {text: '性能分析', link: '02-04-performance-analysis'}
        ]
      },
      {
        text: '第3章 连续批处理调度',
        collapsed: false,
        items: [
          {text: '从静态到连续批处理', link: '03-01-static-to-continuous'},
          {text: '调度器机制', link: '03-2-scheduler-mechanism'},
          {text: '抢占与资源管理', link: '03-3-preemption-resource'},
          {text: '调度器调优', link: '03-4-scheduler-tuning'}
        ]
      },
      {
        text: '第4章 API 服务',
        collapsed: false,
        items: [
          {text: 'API 服务器启动与配置', link: '04-01-api-server-startup'},
          {text: 'Chat Completions API', link: '04-02-chat-completions-api'},
          {text: 'Completions 与 Embeddings', link: '04-03-completions-embeddings'},
          {text: '其他 API 端点', link: '04-04-other-apis'},
          {text: '离线推理', link: '04-05-offline-inference'}
        ]
      },
      {
        text: '第5章 离线推理引擎',
        collapsed: false,
        items: [
          {text: 'LLM 直接推理', link: '05-01-llm-direct-inference'},
          {text: '高效批处理', link: '05-02-efficient-batch-processing'},
          {text: '特殊推理场景', link: '05-03-special-inference-scenarios'}
        ]
      },
      {
        text: '第6章 分布式推理',
        collapsed: false,
        items: [
          {text: '张量并行', link: '06-01-tensor-parallelism'},
          {text: '流水线并行', link: '06-02-pipeline-parallelism'},
          {text: '多节点部署', link: '06-03-multi-node-deployment'},
          {text: '性能基准测试', link: '06-04-performance-benchmark'},
          {text: '硬件选型指南', link: '06-05-hardware-selection'}
        ]
      },
      {
        text: '第7章 量化与压缩',
        collapsed: false,
        items: [
          {text: '量化基础', link: '07-01-quantization-basics'},
          {text: 'AWQ 量化', link: '07-02-awq'},
          {text: 'GPTQ 及其他方法', link: '07-03-gptq-other-methods'},
          {text: '量化方案对比', link: '07-04-quantization-comparison'}
        ]
      },
      {
        text: '第8章 LoRA 动态服务',
        collapsed: false,
        items: [
          {text: 'LoRA 基础', link: '08-01-lora-basics'},
          {text: 'LoRA 服务化', link: '08-02-lora-serving'},
          {text: 'LoRA 最佳实践', link: '08-03-lora-best-practices'}
        ]
      },
      {
        text: '第9章 框架集成',
        collapsed: false,
        items: [
          {text: 'LangChain + vLLM', link: '09-01-langchain-vllm'},
          {text: 'LlamaIndex + vLLM', link: '09-02-llamaindex-vllm'},
          {text: '其他框架集成', link: '09-03-other-frameworks'},
          {text: '生产级架构', link: '09-04-production-architecture'}
        ]
      },
      {
        text: '第10章 生产部署',
        collapsed: false,
        items: [
          {text: 'Docker 部署', link: '10-01-docker-deployment'},
          {text: 'Kubernetes 部署', link: '10-02-kubernetes'},
          {text: '监控与告警', link: '10-03-monitoring'},
          {text: '高可用方案', link: '10-04-high-availability'},
          {text: '性能调优', link: '10-05-performance-tuning'}
        ]
      }
    ]
  }]
}

function sidebarPytorch(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'PyTorch Lightning 教程',
    items: [
      {
        text: '第1章 PyTorch 核心基础',
        collapsed: false,
        items: [
          {text: '为什么选择 PyTorch', link: '01-01-why-pytorch'},
          {text: '张量操作', link: '01-02-tensor-operations'},
          {text: '自动微分 Autograd', link: '01-03-autograd'},
          {text: 'nn.Module 神经网络模块', link: '01-04-nn-module'},
          {text: 'MiniGPT 项目实战', link: '01-05-minigpt-project'}
        ]
      },
      {
        text: '第2章 数据加载与预处理',
        collapsed: false,
        items: [
          {text: 'Dataset 与 DataLoader', link: '02-01-dataset-dataloader'},
          {text: 'Collate Function', link: '02-02-collate-function'},
          {text: '高效数据处理', link: '02-03-efficient-data-processing'},
          {text: '完整数据管道', link: '02-04-complete-pipeline'}
        ]
      },
      {
        text: '第3章 Transformer 架构实现',
        collapsed: false,
        items: [
          {text: '单头注意力', link: '03-01-single-head-attention'},
          {text: '多头注意力', link: '03-02-multi-head-attention'},
          {text: '前馈网络块', link: '03-03-feed-forward-block'},
          {text: '位置编码', link: '03-04-positional-encoding'},
          {text: 'GPT 模型组装', link: '03-05-gpt-model'}
        ]
      },
      {
        text: '第4章 训练循环与优化',
        collapsed: false,
        items: [
          {text: '训练循环', link: '04-01-training-loop'},
          {text: '优化器', link: '04-02-optimizer'},
          {text: '学习率调度与梯度管理', link: '04-03-lr-scheduler-gradient'},
          {text: '训练监控与调试', link: '04-04-monitoring-debugging'}
        ]
      },
      {
        text: '第5章 PyTorch Lightning 入门',
        collapsed: false,
        items: [
          {text: 'Lightning 第一个模型', link: '05-01-lightning-first-model'},
          {text: '生命周期回调', link: '05-02-lifecycle-callbacks'},
          {text: 'Lightning 进阶', link: '05-03-advanced-lightning'},
          {text: 'Lightning GPT 实战', link: '05-04-lightning-gpt-practice'}
        ]
      },
      {
        text: '第6章 模型微调',
        collapsed: false,
        items: [
          {text: 'Trainer API', link: '06-01-trainer-api'},
          {text: 'PEFT 与 LoRA', link: '06-02-peft-lora'},
          {text: '量化与 QLoRA', link: '06-03-quantization-qlora'},
          {text: '三种方法对比', link: '06-04-three-methods-comparison'},
          {text: '评估与导出', link: '06-05-evaluation-export'}
        ]
      },
      {
        text: '第7章 分布式训练',
        collapsed: false,
        items: [
          {text: 'DDP 分布式数据并行', link: '07-01-ddp'},
          {text: 'FSDP 全分片数据并行', link: '07-02-fsdp'},
          {text: 'DeepSpeed 集成', link: '07-03-deepspeed'},
          {text: '分布式实战', link: '07-04-distributed-practice'}
        ]
      },
      {
        text: '第8章 模型优化与导出',
        collapsed: false,
        items: [
          {text: 'torch.compile', link: '08-01-torch-compile'},
          {text: '模型量化', link: '08-02-model-quantization'},
          {text: '模型导出', link: '08-03-model-export'},
          {text: '性能分析', link: '08-04-performance-analysis'}
        ]
      },
      {
        text: '第9章 模型服务与 MLOps',
        collapsed: false,
        items: [
          {text: '模型服务化', link: '09-01-model-serving'},
          {text: '监控', link: '09-2-monitoring'},
          {text: 'MLOps 生命周期', link: '09-3-mlops-lifecycle'}
        ]
      }
    ]
  }]
}

function sidebarOpenCode(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'OpenCode 教程',
    items: [
      {
        text: '第1章 OpenCode 简介',
        collapsed: false,
        items: [
          {text: '什么是 OpenCode', link: '01-01-what-is-opencode'},
          {text: '架构概览', link: '01-02-architecture'},
          {text: '快速上手', link: '01-03-quick-start'}
        ]
      },
      {
        text: '第2章 安装与配置',
        collapsed: false,
        items: [
          {text: '安装', link: '02-01-installation'},
          {text: '配置文件详解', link: '02-02-configuration'},
          {text: '模型提供商', link: '02-03-model-providers'}
        ]
      },
      {
        text: '第3章 TUI 界面与交互',
        collapsed: false,
        items: [
          {text: 'TUI 界面详解', link: '03-01-tui-interface'},
          {text: '文件引用与上下文', link: '03-02-file-references'},
          {text: 'Agent 工具', link: '03-03-agent-tools'}
        ]
      },
      {
        text: '第4章 模型选择与优化',
        collapsed: false,
        items: [
          {text: '模型选择策略', link: '04-01-model-selection'},
          {text: 'Ollama 本地模型', link: '04-02-ollama-local'},
          {text: 'Auto Compact 上下文管理', link: '04-03-auto-compact'}
        ]
      },
      {
        text: '第5章 MCP 协议集成',
        collapsed: false,
        items: [
          {text: '什么是 MCP', link: '05-01-what-is-mcp'},
          {text: 'MCP 配置', link: '05-02-mcp-configuration'},
          {text: 'MCP GitHub 集成', link: '05-03-mcp-github'},
          {text: 'MCP 数据库与浏览器', link: '05-04-mcp-database-browser'}
        ]
      },
      {
        text: '第6章 编码工作流',
        collapsed: false,
        items: [
          {text: '代码生成', link: '06-01-code-generation'},
          {text: '调试', link: '06-02-debugging'},
          {text: 'Git 工作流', link: '06-03-git-workflow'}
        ]
      },
      {
        text: '第7章 团队协作',
        collapsed: false,
        items: [
          {text: 'AGENTS.md', link: '07-01-agents-md'},
          {text: '自定义命令', link: '07-02-custom-commands'},
          {text: '多会话管理', link: '07-03-multi-session'}
        ]
      },
      {
        text: '第8章 安全与限制',
        collapsed: false,
        items: [
          {text: '安全考量', link: '08-01-security'},
          {text: '团队标准化', link: '08-02-team-standardization'},
          {text: '局限性与替代方案', link: '08-03-limitations-alternatives'}
        ]
      }
    ]
  }]
}

function sidebarDeepSeek(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'DeepSeek 教程',
    items: [
      {
        text: '第1章 DeepSeek 简介',
        collapsed: false,
        items: [
          {text: '什么是 DeepSeek', link: '01-01-what-is-deepseek'},
          {text: '模型家族', link: '01-02-model-family'},
          {text: '架构解析', link: '01-03-architecture'}
        ]
      },
      {
        text: '第2章 快速上手',
        collapsed: false,
        items: [
          {text: '注册与 API Key', link: '02-01-register-api-key'},
          {text: 'Hello World', link: '02-02-hello-world'},
          {text: 'Web 应用', link: '02-03-web-app'}
        ]
      },
      {
        text: '第3章 Chat Completions API',
        collapsed: false,
        items: [
          {text: 'Chat Completions', link: '03-01-chat-completions'},
          {text: '流式输出', link: '03-02-streaming-tokens'},
          {text: 'JSON Output', link: '03-03-json-output'},
          {text: 'FIM 前缀补全', link: '03-04-fim-prefix'}
        ]
      },
      {
        text: '第4章 R1 推理模式',
        collapsed: false,
        items: [
          {text: '思维模式', link: '04-01-thinking-mode'},
          {text: '推理 vs 非推理', link: '04-02-reasoning-vs-non-reasoning'},
          {text: 'Max Tokens 管理', link: '04-03-max-tokens'}
        ]
      },
      {
        text: '第5章 Agent 与 RAG',
        collapsed: false,
        items: [
          {text: 'Function Calling', link: '05-01-function-calling'},
          {text: '构建 Agent', link: '05-02-build-agent'},
          {text: 'RAG 实践', link: '05-03-rag-practice'}
        ]
      },
      {
        text: '第6章 代码智能',
        collapsed: false,
        items: [
          {text: '代码生成', link: '06-01-code-generation'},
          {text: '调试', link: '06-02-debugging'},
          {text: 'CI/CD', link: '06-03-cicd'}
        ]
      },
      {
        text: '第7章 本地部署',
        collapsed: false,
        items: [
          {text: '蒸馏模型', link: '07-01-distill-models'},
          {text: 'Ollama 部署', link: '07-02-ollama'},
          {text: 'vLLM 部署', link: '07-03-vllm'}
        ]
      },
      {
        text: '第8章 企业实践',
        collapsed: false,
        items: [
          {text: '成本优化', link: '08-01-cost-optimization'},
          {text: '企业部署', link: '08-02-enterprise-deployment'},
          {text: '局限性与替代方案', link: '08-03-limitations-alternatives'}
        ]
      }
    ]
  }]
}

function sidebarChroma(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'Chroma 教程',
    items: [
      {
        text: '第1章 为什么需要向量数据库',
        collapsed: false,
        items: [
          {text: '为什么需要向量数据库', link: '01-01-why-vector-db'},
          {text: '核心概念概览', link: '01-02-concepts-overview'},
          {text: '安装与 Hello World', link: '01-03-installation-hello-world'}
        ]
      },
      {
        text: '第2章 数据管理',
        collapsed: false,
        items: [
          {text: 'CRUD 操作', link: '02-01-crud-operations'},
          {text: 'Collection 管理', link: '02-02-collection-management'},
          {text: '元数据最佳实践', link: '02-03-metadata-best-practices'}
        ]
      },
      {
        text: '第3章 嵌入与分块',
        collapsed: false,
        items: [
          {text: 'Embedding Function', link: '03-01-embedding-function'},
          {text: '分块策略', link: '03-02-chunking-strategies'},
          {text: '归一化与距离度量', link: '03-03-normalization-distance-metrics'}
        ]
      },
      {
        text: '第4章 查询与过滤',
        collapsed: false,
        items: [
          {text: '查询深度解析', link: '04-01-query-deep-dive'},
          {text: 'Where 过滤语法', link: '04-02-where-filter-syntax'},
          {text: '重排序', link: '04-03-reranking'}
        ]
      },
      {
        text: '第5章 RAG 实战',
        collapsed: false,
        items: [
          {text: 'RAG 架构', link: '05-01-rag-architecture'},
          {text: 'PDF 问答 Demo', link: '05-02-pdf-qa-demo'},
          {text: '记忆层', link: '05-03-memory-layer'}
        ]
      },
      {
        text: '第6章 生产运维',
        collapsed: false,
        items: [
          {text: '持久化存储', link: '06-01-persistence-storage'},
          {text: '性能调优', link: '06-02-performance-tuning'},
          {text: '并发', link: '06-03-concurrency'},
          {text: '监控运维', link: '06-04-monitoring-ops'}
        ]
      },
      {
        text: '第7章 选型与生态',
        collapsed: false,
        items: [
          {text: '选型指南', link: '07-01-selection-guide'},
          {text: '框架集成', link: '07-02-framework-integration'},
          {text: '局限性与替代方案', link: '07-03-limitations-alternatives'}
        ]
      }
    ]
  }]
}

function sidebarFaiss(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'FAISS 教程',
    items: [
      {
        text: '第1章 FAISS 入门',
        collapsed: false,
        items: [
          {text: '什么是 FAISS', link: '01-01-what-is-faiss'},
          {text: '安装', link: '01-02-installation'},
          {text: 'Hello World', link: '01-03-hello-world'}
        ]
      },
      {
        text: '第2章 核心抽象',
        collapsed: false,
        items: [
          {text: 'Index 抽象', link: '02-01-index-abstraction'},
          {text: '距离度量', link: '02-02-distance-metrics'},
          {text: '向量 ID 映射', link: '02-03-vector-id-mapping'}
        ]
      },
      {
        text: '第3章 索引类型',
        collapsed: false,
        items: [
          {text: 'Flat 暴力索引', link: '03-01-flat-index'},
          {text: 'IVF 倒排索引', link: '03-02-ivf-index'},
          {text: 'HNSW 图索引', link: '03-03-hnsw-index'}
        ]
      },
      {
        text: '第4章 量化压缩',
        collapsed: false,
        items: [
          {text: '为什么需要量化', link: '04-01-why-quantization'},
          {text: 'PQ 乘积量化', link: '04-02-pq-quantization'},
          {text: 'OPQ 优化乘积量化', link: '04-03-opq-quantization'},
          {text: 'SQ 标量量化', link: '04-04-sq-quantization'}
        ]
      },
      {
        text: '第5章 GPU 加速与高级功能',
        collapsed: false,
        items: [
          {text: 'GPU 加速', link: '05-01-gpu-acceleration'},
          {text: '复合索引', link: '05-02-composite-index'},
          {text: '批量搜索', link: '05-03-batch-search'},
          {text: '序列化', link: '05-04-serialization'}
        ]
      },
      {
        text: '第6章 RAG 实战',
        collapsed: false,
        items: [
          {text: 'RAG 架构', link: '06-01-rag-architecture'},
          {text: 'RAG Demo', link: '06-02-rag-demo'},
          {text: 'FAISS + pgvector 混合', link: '06-03-faiss-pgvector-hybrid'}
        ]
      },
      {
        text: '第7章 选型与生态',
        collapsed: false,
        items: [
          {text: 'FAISS 作为引擎', link: '07-01-faiss-as-engine'},
          {text: 'FAISS vs 向量数据库', link: '07-02-faiss-vs-databases'},
          {text: '局限性与未来', link: '07-03-limitations-future'}
        ]
      }
    ]
  }]
}

function sidebarMilvus(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'Milvus 教程',
    items: [
      {
        text: '第1章 Milvus 入门',
        collapsed: false,
        items: [
          {text: '什么是 Milvus', link: '01-01-what-is-milvus'},
          {text: '架构概览', link: '01-02-architecture'},
          {text: '安装与连接', link: '01-03-installation-connection'}
        ]
      },
      {
        text: '第2章 Schema 与数据模型',
        collapsed: false,
        items: [
          {text: 'Collection 与 Schema', link: '02-01-collection-schema'},
          {text: '字段类型', link: '02-02-field-types'},
          {text: '距离度量', link: '02-03-distance-metrics'}
        ]
      },
      {
        text: '第3章 数据操作',
        collapsed: false,
        items: [
          {text: '数据插入', link: '03-01-data-insertion'},
          {text: '向量搜索', link: '03-02-vector-search'},
          {text: '标量过滤', link: '03-03-scalar-filtering'},
          {text: '查询/更新/删除', link: '03-04-query-update-delete'}
        ]
      },
      {
        text: '第4章 索引管理',
        collapsed: false,
        items: [
          {text: '索引概览', link: '04-01-index-overview'},
          {text: 'HNSW 索引', link: '04-02-hnsw-index'},
          {text: '量化索引', link: '04-03-quantization-index'},
          {text: '索引选择与调优', link: '04-04-index-selection-tuning'}
        ]
      },
      {
        text: '第5章 高级功能',
        collapsed: false,
        items: [
          {text: '分区管理', link: '05-01-partition-management'},
          {text: '多向量搜索', link: '05-02-multi-vector-search'},
          {text: '动态字段与 JSON', link: '05-03-dynamic-field-json'},
          {text: '一致性与持久化', link: '05-04-consistency-persistence'}
        ]
      },
      {
        text: '第6章 RAG 实战',
        collapsed: false,
        items: [
          {text: 'RAG 架构', link: '06-01-rag-architecture'},
          {text: 'PDF 问答 Demo', link: '06-02-pdf-qa-demo'},
          {text: '多租户', link: '06-03-multi-tenant'}
        ]
      },
      {
        text: '第7章 生产部署',
        collapsed: false,
        items: [
          {text: '集群部署', link: '07-01-cluster-deployment'},
          {text: '性能监控', link: '07-02-performance-monitoring'},
          {text: '局限性与替代方案', link: '07-03-limitations-alternatives'}
        ]
      }
    ]
  }]
}

function sidebarPgvector(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'PG Vector 教程',
    items: [
      {
        text: '第1章 PostgreSQL 基础',
        collapsed: false,
        items: [
          {text: '什么是 PostgreSQL', link: '01-01-what-is-postgresql'},
          {text: '安装与连接', link: '01-02-installation-connection'},
          {text: 'SQL CRUD 基础', link: '01-03-sql-crud-basics'}
        ]
      },
      {
        text: '第2章 PostgreSQL 进阶',
        collapsed: false,
        items: [
          {text: '索引与查询优化', link: '02-01-indexes-query-optimization'},
          {text: 'JSONB 半结构化数据', link: '02-02-jsonb-semistructured'},
          {text: '事务与并发', link: '02-03-transactions-concurrency'}
        ]
      },
      {
        text: '第3章 pgvector 入门',
        collapsed: false,
        items: [
          {text: '什么是 pgvector', link: '03-01-what-is-pgvector'},
          {text: '安装与 Hello World', link: '03-02-installation-hello-world'},
          {text: '向量数据类型', link: '03-03-vector-data-type'}
        ]
      },
      {
        text: '第4章 向量 CRUD',
        collapsed: false,
        items: [
          {text: '向量 CRUD', link: '04-01-vector-crud'},
          {text: '混合查询', link: '04-02-hybrid-query'},
          {text: 'Python CRUD 封装', link: '04-03-python-crud-wrapper'}
        ]
      },
      {
        text: '第5章 索引与性能',
        collapsed: false,
        items: [
          {text: '为什么需要向量索引', link: '05-01-why-vector-index'},
          {text: 'IVFFlat 索引', link: '05-02-ivfflat-index'},
          {text: 'HNSW 索引', link: '05-03-hnsw-index'},
          {text: '索引选择与调优', link: '05-04-index-selection-tuning'}
        ]
      },
      {
        text: '第6章 RAG 实战',
        collapsed: false,
        items: [
          {text: 'RAG 架构', link: '06-01-rag-architecture'},
          {text: 'PDF 问答 Demo', link: '06-02-pdf-qa-demo'},
          {text: '记忆与用户画像', link: '06-03-memory-user-profile'}
        ]
      },
      {
        text: '第7章 生产部署',
        collapsed: false,
        items: [
          {text: '生产部署', link: '07-01-production-deployment'},
          {text: '性能监控', link: '07-02-performance-monitoring'},
          {text: '局限性与替代方案', link: '07-03-limitations-alternatives'}
        ]
      }
    ]
  }]
}

function
sidebarNumPy():
    DefaultTheme
    .SidebarItem[] {
      return [{
        text: 'Numpy学习指南',
        items: [
          {
            text: '第1章 基础概念',
            collapsed: false,
            items: [
              {text: 'NumPy是什么', link: '01-00-numpy-intro'},
              {text: 'NumPy优势对比', link: '01-01-numpy-vs-list'},
              {text: '安装与配置', link: '01-02-install-config'},
              {text: '导入与约定', link: '01-03-import-convention'},
              {text: 'NumPy与LLM开发', link: '01-04-why-numpy-llm'}
            ]
          },
          {
            text: '第2章 ndarray属性',
            collapsed: false,
            items: [
              {text: 'ndarray属性', link: '02-01-ndarray-attributes'},
              {text: 'dtype深入理解', link: '02-02-dtype-deep-dive'},
              {text: '从列表创建', link: '02-03-ndarray-from-list'},
              {text: 'AI中的应用', link: '02-04-ndarray-in-ai'}
            ]
          },
          {
            text: '第3章 数组创建',
            collapsed: false,
            items: [
              {text: 'zeros与ones', link: '03-01-zeros-ones'},
              {text: 'eye与identity', link: '03-02-eye-identity'},
              {text: 'arange与linspace', link: '03-03-arange-linspace'},
              {text: 'tile与meshgrid', link: '03-04-tile-meshgrid'},
              {text: '随机数组', link: '03-05-random-arrays'},
              {text: '从数据创建', link: '03-06-array-from-data'}
            ]
          },
          {
            text: '第4章 形状操作',
            collapsed: false,
            items: [
              {text: 'shape与reshape', link: '04-01-shape-reshape'},
              {text: 'flatten与ravel', link: '04-02-flatten-ravel'},
              {text: '转置操作', link: '04-03-transpose'},
              {text: 'expand与squeeze', link: '04-04-expand-squeeze'},
              {text: 'concatenate与stack', link: '04-05-concatenate-stack'},
              {text: 'split分割', link: '04-06-split-array'}
            ]
          },
          {
            text: '第5章 索引与切片',
            collapsed: false,
            items: [
              {text: '基本索引', link: '05-01-basic-indexing'},
              {text: '整数数组索引', link: '05-02-integer-array-indexing'},
              {text: '布尔掩码', link: '05-03-boolean-masking'},
              {text: '花式索引', link: '05-04-fancy-indexing'},
              {text: 'LLM嵌入查找', link: '05-05-llm-embedding-lookup'}
            ]
          },
          {
            text: '第6章 运算与性能',
            collapsed: false,
            items: [
              {text: '元素级运算', link: '06-01-element-wise-operations'},
              {text: '向量化性能', link: '06-02-vectorization-performance'},
              {text: '矩阵乘法', link: '06-03-matrix-multiplication'},
              {text: 'LLM自注意力', link: '06-04-llm-self-attention'}
            ]
          },
          {
            text: '第7章 广播机制',
            collapsed: false,
            items: [
              {text: '广播规则', link: '07-01-broadcasting-rules'},
              {text: '数组扩张', link: '07-02-array-expansion'},
              {text: 'LLM位置编码', link: '07-03-llm-positional-encoding'}
            ]
          },
          {
            text: '第8章 数学函数',
            collapsed: false,
            items: [
              {text: '指数与对数', link: '08-01-exp-log'},
              {text: '激活函数', link: '08-02-activation-functions'},
              {text: 'clip截断', link: '08-03-clip-functions'}
            ]
          },
          {
            text: '第9章 统计与损失',
            collapsed: false,
            items: [
              {text: '基础统计', link: '09-01-statistics-basic'},
              {text: 'axis统计', link: '09-02-axis-statistics'},
              {text: 'cumsum累加', link: '09-03-cumsum-operations'},
              {text: 'LLM困惑度', link: '09-04-llm-perplexity'}
            ]
          },
          {
            text: '第10章 线性代数',
            collapsed: false,
            items: [
              {text: '点积', link: '10-01-dot-product'}, {
                text: '特征值与特征向量',
                link: '10-02-eigenvalues-eigenvectors'
              },
              {text: '奇异值分解', link: '10-03-svd'},
              {text: '矩阵求逆与伪逆', link: '10-04-pinv'},
              {text: '范数计算', link: '10-05-norm'}
            ]
          },
          {
            text: '第11章 随机数生成',
            collapsed: false,
            items: [
              {text: '随机种子与可复现性', link: '11-01-random-seed'},
              {text: '常见分布', link: '11-02-common-distributions'},
              {text: '随机采样', link: '11-03-random-sampling'},
              {text: '随机排列', link: '11-04-random-permutation'},
              {text: '随机初始化权重', link: '11-05-random-initialization'}
            ]
          },
          {
            text: '第12章 内存布局',
            collapsed: false,
            items: [
              {text: 'C顺序与Fortran顺序', link: '12-01-c-fortran-order'},
              {text: '视图与副本', link: '12-02-view-vs-copy'},
              {text: 'ndarray.view', link: '12-03-view-ndarray'},
              {text: '原地操作', link: '12-04-inplace-operations'}
            ]
          },
          {
            text: '第13章 结构化数组',
            collapsed: false,
            items: [
              {text: '创建结构化数组', link: '13-01-structured-arrays'},
              {text: '字段访问与操作', link: '13-02-field-access'},
              {text: 'HuggingFace导出', link: '13-03-huggingface-export'}
            ]
          },
          {
            text: '第14章 文件IO',
            collapsed: false,
            items: [
              {text: '文本文件IO', link: '14-01-text-io'},
              {text: '二进制文件IO', link: '14-02-binary-io'},
              {text: '内存映射', link: '14-03-memmap'}
            ]
          },
          {
            text: '实战一：词嵌入矩阵',
            collapsed: false,
            items: [
              {text: '词汇表与索引映射', link: '15-01-vocab-index'},
              {text: '嵌入矩阵创建', link: '15-02-embedding-matrix'},
              {text: '嵌入查找实现', link: '15-03-embedding-lookup'},
              {text: '词向量相似度可视化', link: '15-04-visualize-similarity'}
            ]
          },
          {
            text: '实战二：多头自注意力',
            collapsed: false,
            items: [
              {text: 'QKV投影矩阵', link: '16-01-qkv-projections'},
              {text: '缩放点积注意力', link: '16-02-scaled-attention'},
              {text: '因果掩码', link: '16-03-causal-mask'},
              {text: '前向传播模拟', link: '16-04-forward-propagation'}
            ]
          },
          {
            text: '实战三：LayerNorm与残差',
            collapsed: false,
            items: [
              {text: 'LayerNorm实现', link: '17-01-layer-norm'},
              {text: '残差连接模拟', link: '17-02-residual-connection'},
              {text: 'NumPy vs PyTorch对比', link: '17-03-numpy-vs-pytorch'}
            ]
          },
          {
            text: '实战四：GPT文本生成',
            collapsed: false,
            items: [
              {text: '预训练嵌入概述', link: '18-01-pretrained-embeddings'},
              {text: 'Transformer解码器', link: '18-02-transformer-decoder'},
              {text: '温度采样', link: '18-03-temperature-sampling'},
              {text: '简单文本序列生成', link: '18-04-text-generation'}
            ]
          },
          {
            text: '实战五：LoRA低秩适配',
            collapsed: false,
            items: [
              {text: 'LoRA原理', link: '19-01-lora-principle'},
              {text: 'LoRA前向传播', link: '19-02-lora-forward'},
              {text: '参数量节省', link: '19-03-parameter-savings'}
            ]
          },
          {
            text: '实战六：HuggingFace集成',
            collapsed: false,
            items: [
              {text: 'tokenizer转NumPy', link: '20-01-tokenizer-to-numpy'},
              {text: '数据集转NumPy', link: '20-02-datasets-to-numpy'},
              {text: 'NumPy DataLoader', link: '20-03-numpy-dataloader'}
            ]
          },
          {
            text: '实战七：框架互操作',
            collapsed: false,
            items: [
              {text: 'PyTorch互操作', link: '21-01-torch-numpy'},
              {text: 'TensorFlow互操作', link: '21-02-tensorflow-numpy'},
              {text: '预处理与训练', link: '21-03-preprocessing-training'}
            ]
          },
          {
            text: '实战八：LLM评估',
            collapsed: false,
            items: [
              {text: '困惑度计算', link: '22-01-perplexity'},
              {text: 'BLEU/ROUGE指标', link: '22-02-bleu-rouge'},
              {text: '文本多样性评估', link: '22-03-text-diversity'}
            ]
          }
        ]
      }]
    }

function
sidebarLangChain():
    DefaultTheme
    .SidebarItem[] {
      return [{
        text: 'LangChain学习指南',
        items: [
          {
            text: '第1章 为什么需要 LangChain', 
            collapsed: false, 
            items: [
            {text: '1.1 大模型的能力边界与痛点', link: '01-01-llm-limitations'},
            {text: '1.2 编排框架与设计哲学', link: '01-02-orchestration-and-design'},
            {text: '1.3 同类框架对比与选型', link: '01-03-framework-comparison'},
            {text: '1.4 v1.0 新特性概览', link: '01-04-v1-new-features'}
          ]},
          {
            text: '第2章 环境搭建与第一次运行', 
            collapsed: false, 
            items: [
            {text: '2.1 开发环境准备', link: '02-01-environment-setup'},
            {text: '2.2 安装 LangChain 与生态工具', link: '02-02-installation'},
            {text: '2.3 模型接入实战', link: '02-03-model-connection'},
            {text: '2.4 第一个 LangChain 应用', link: '02-04-first-app'}
          ]},
          {
            text: '第3章 模型 I/O — 与 LLM 对话的基础', 
            collapsed: false, 
            items: [
            {text: '3.1 模型类型：LLM vs. 聊天模型', link: '03-01-model-types'},
            {text: '3.2 输入管理：提示模板的多种玩法', link: '03-02-prompt-templates'},
            {text: '3.3 输出处理：解析器的妙用', link: '03-03-output-parsers'},
            {text: '3.4 实战：构建可配置参数的情感分析器', link: '03-04-sentiment-analyzer-practice'}
          ]},
          {
            text: '第4章 检索增强生成 (RAG) — 连接私有知识库', 
            collapsed: false, 
            items: [
            {text: '4.1 RAG 为什么成为刚需？基本原理与架构', link: '04-01-rag-principles'},
            {text: '4.2 文档加载器：从 PDF、网页、Notion 等读取', link: '04-02-document-loaders'},
            {text: '4.3 文本分割：语义切分策略', link: '04-03-text-splitting'},
            {text: '4.4 向量存储与嵌入模型', link: '04-04-vector-storage'},
            {text: '4.5 检索器：相似度搜索与高级策略', link: '04-05-retrieval'},
            {text: '4.6 实战：公司内部文档问答机器人', link: '04-06-rag-qa-bot'}
          ]},
          {
            text: '第5章 记忆 — 让对话拥有上下文', 
            collapsed: false, 
            items: [
            {text: '5.1 为什么需要记忆：无状态模型的局限', link: '05-01-why-memory'},
            {text: '5.2 LangChain 记忆组件全景', link: '05-02-memory-overview'},
            {text: '5.3 常用记忆类型实战', link: '05-03-memory-practice'},
            {text: '5.4 实战：构建有记忆的智能对话助手', link: '05-04-chat-assistant-practice'}
          ]},
          {
            text: '第6章 多模态交互 — 支持图像与语音', 
            collapsed: false, 
            items: [
            {text: '6.1 多模态模型入门：从文本到视觉', link: '06-01-multimodal-intro'},
            {text: '6.2 图像理解：让 LLM "看"图片', link: '06-02-image-understanding'},
            {text: '6.3 语音交互：语音转文字与文字转语音', link: '06-03-speech-interaction'},
            {text: '6.4 实战：构建多模态智能助手', link: '06-04-multimodal-assistant-practice'}
          ]},
          {
            text: '第7章 LCEL — 新的链式语法', 
            collapsed: false, 
            items: [
            {text: '7.1 为什么需要 LCEL？声明式 vs 命令式', link: '07-01-lcel-intro'},
            {text: '7.2 LCEL 的核心原语：Runnable 接口', link: '07-02-runnable-interface'},
            {text: '7.3 管道操作符与并行化（RunnableParallel）', link: '07-03-pipe-parallel'},
            {text: '7.4 条件分支与路由（RunnableBranch）', link: '07-04-branch-routing'},
            {text: '7.5 实战：用 LCEL 重构复杂问答链', link: '07-05-lcel-refactor'}
          ]},
          {
            text: '第8章 智能体 — 让 AI 自主规划与执行任务', 
            collapsed: false, 
            items: [
            {text: '8.1 Agent 是什么：从 Chain 到自主决策', link: '08-01-agent-concepts'},
            {text: '8.2 Tool（工具）：给 AI 装上手和脚', link: '08-02-tools'},
            {text: '8.3 ReAct 模式：构建你的第一个 Agent', link: '08-03-react-agent'},
            {text: '8.4 高级代理模式：多代理协作与长期记忆', link: '08-04-advanced-agent-patterns'},
            {text: '8.5 实战：构建自动研究助理 Agent', link: '08-05-auto-research-assistant'}
          ]},
          {
            text: '第9章 流式、异步与中间件', 
            collapsed: false, 
            items: [
            {text: '9.1 流式输出的价值与实现', link: '09-01-streaming'},
            {text: '9.2 异步编程：提升应用的并发能力', link: '09-02-async'},
            {text: '9.3 中间件：横切关注点（日志、限流、重试、审核）', link: '09-03-middleware'},
            {text: '9.4 回调机制：深入 LangChain 内部运行流程', link: '09-04-callbacks'},
            {text: '9.5 实战：为应用添加"对话审核"中间件', link: '09-05-moderated-chat'}
          ]},
          {
            text: '第10章 项目一：智能客服系统',
            collapsed: false,
            items: [
            {text: '10.1 需求分析与技术选型', link: '10-01-requirement-analysis'},
            {text: '10.2 利用 RAG 加载产品知识库', link: '10-02-rag-knowledge-base'},
            {text: '10.3 设计多轮对话的意图识别与分流', link: '10-03-intent-routing'},
            {text: '10.4 集成人工接管（Handoff）机制', link: '10-04-human-handoff'}
          ]},
          {
            text: '第11章 项目二：代码分析助手',
            collapsed: false,
            items: [
            {text: '11.1 加载代码仓库并构建代码知识库', link: '11-01-code-repo-loading'},
            {text: '11.2 实现代码解释、Bug 定位与单元测试生成', link: '11-02-code-analysis'},
            {text: '11.3 利用代理调用代码执行工具', link: '11-03-code-agent-tools'}
          ]},
          {
            text: '第12章 项目三：数据分析 Agent',
            collapsed: false,
            items: [
            {text: '12.1 连接数据库（SQL 数据库）', link: '12-01-sql-database'},
            {text: '12.2 设计 Text-to-SQL 代理', link: '12-02-text-to-sql-agent'},
            {text: '12.3 让代理生成数据图表（Pandas + Matplotlib）', link: '12-03-data-visualization'},
            {text: '12.4 输出洞察报告', link: '12-04-insight-report'}
          ]},
          {
            text: '第13章 评估与可观测性',
            collapsed: false,
            items: [
            {text: '13.1 为什么要评估 RAG 和代理？', link: '13-01-why-evaluate'},
            {text: '13.2 评估指标：准确率、忠实度、答案相关性', link: '13-02-evaluation-metrics'},
            {text: '13.3 LangSmith 实战：追踪、调试与性能评估', link: '13-03-langsmith-practice'},
            {text: '13.4 离线评估与持续回归测试', link: '13-04-offline-eval'}
          ]},
          {
            text: '第14章 部署与扩展',
            collapsed: false,
            items: [
            {text: '14.1 将 LangChain 应用封装为 API 服务（FastAPI）', link: '14-01-fastapi-service'},
            {text: '14.2 容器化部署：Docker + Kubernetes', link: '14-02-docker-k8s'},
            {text: '14.3 向量数据库的生产环境考量（Milvus、Pinecone）', link: '14-03-vector-db-prod'},
            {text: '14.4 成本优化：缓存策略与模型路由', link: '14-04-cost-optimization'}
          ]},
          {
            text: '第15章 安全与限制',
            collapsed: false,
            items: [
            {text: '15.1 提示词注入攻击与防御', link: '15-01-prompt-injection'},
            {text: '15.2 敏感数据泄露风险', link: '15-02-data-leakage'},
            {text: '15.3 代理的权限控制与沙箱', link: '15-03-agent-sandbox'}
          ]}
        ]
      }]
    }

function
sidebarLangGraph():
    DefaultTheme
    .SidebarItem[] {
      return [{
        text: 'LangGraph 教程',
        items: [
          {
            text: '第1章 LangGraph 核心概念入门',
            collapsed: false,
            items: [
              {text: '为什么需要 LangGraph', link: '01-01-why-langgraph'},
              {text: '第一个有状态的图', link: '01-02-first-stateful-graph'},
              {text: 'LangChain Agent vs LangGraph', link: '01-03-vs-langchain-agent'},
              {text: '核心概念全景', link: '01-4-core-concepts-overview'},
              {text: '工具链与调试', link: '01-5-tooling-and-debugging'}
            ]
          },
          {
            text: '第2章 状态机与图编程',
            collapsed: false,
            items: [
              {text: '状态设计模式', link: '02-01-state-design-patterns'},
              {text: '节点编程模式', link: '02-2-node-programming-patterns'},
              {text: '边路由与拓扑结构', link: '02-3-edge-routing-and-topology'},
              {text: '子图与组合模式', link: '02-04-subgraphs-and-composition'},
              {text: '实战模式与最佳实践', link: '02-5-practical-patterns-and-best-practices'}
            ]
          },
          {
            text: '第3章 条件路由深度解析',
            collapsed: false,
            items: [
              {text: '条件边原理与实现', link: '03-01-conditional-edges-deep-dive'},
              {text: '多层条件路由', link: '03-02-multi-layer-routing'},
              {text: 'LLM 驱动的动态路由', link: '03-03-dynamic-routing-with-llm'},
              {text: '路由测试与调试', link: '03-04-testing-and-routing'},
              {text: '路由模式总结', link: '03-05-routing-patterns-summary'}
            ]
          },
          {
            text: '第4章 人机协作 (Human-in-the-Loop)',
            collapsed: false,
            items: [
              {text: '中断机制基础', link: '04-01-interrupt-basics'},
              {text: '审批工作流', link: '04-2-approval-workflows'},
              {text: '高级中断模式', link: '04-3-advanced-interrupt-patterns'},
              {text: '构建交互式应用', link: '04-4-building-interactive-apps'},
              {text: 'HITL 模式总结', link: '04-5-hitl-summary'}
            ]
          },
          {
            text: '第5章 循环与迭代模式',
            collapsed: false,
            items: [
              {text: '自反思循环', link: '05-01-self-reflection-loops'},
              {text: '重试与容错', link: '05-2-retry-and-fault-tolerance'},
              {text: '循环状态与性能', link: '05-3-loop-state-and-performance'},
              {text: '高级循环模式', link: '05-4-advanced-loop-patterns'},
              {text: '循环模式总结', link: '05-5-loop-patterns-summary'}
            ]
          },
          {
            text: '第6章 子图与模块化架构',
            collapsed: false,
            items: [
              {text: '子图架构设计', link: '06-1-subgraph-architecture'},
              {text: '子图状态映射', link: '06-2-subgraph-state-mapping'},
              {text: '子图测试策略', link: '06-3-subgraph-testing'},
              {text: '动态子图选择', link: '06-4-dynamic-subgraph-selection'},
              {text: '子图工程指南', link: '06-5-subgraph-summary'}
            ]
          },
          {
            text: '第7章 持久化与时间旅行',
            collapsed: false,
            items: [
              {text: '检查点基础', link: '07-01-checkpointing-basics'},
              {text: '时间旅行调试', link: '07-2-time-travel-debugging'},
              {text: '持久化后端', link: '07-3-persistence-backends'},
              {text: '检查点与中断协同', link: '07-4-checkpoint-and-interrupt'},
              {text: '生产环境最佳实践', link: '07-5-persistence-summary'}
            ]
          },
          {
            text: '第8章 多智能体协作',
            collapsed: false,
            items: [
              {text: '多Agent 概览', link: '08-1-multi-agent-overview'},
              {text: '主管 (Supervisor) 模式', link: '08-2-supervisor-pattern'},
              {text: 'Map-Reduce 模式', link: '08-3-map-reduce-pattern'},
              {text: '交接 (Handoff) 模式', link: '08-4-handoff-pattern'},
              {text: '多Agent 总结', link: '08-5-multi-agent-summary'}
            ]
          },
          {
            text: '第9章 项目一：智能客服工单系统',
            collapsed: false,
            items: [
              {text: '项目概述与需求分析', link: '09-01-project-overview-and-requirements'},
              {text: '核心模块实现', link: '09-02-core-module-implementation'},
              {text: 'API 服务与前端', link: '09-03-api-and-frontend'},
              {text: '部署与优化', link: '09-04-deployment-and-optimization'},
              {text: '监控与分析', link: '09-05-monitoring-and-analytics'}
            ]
          },
          {
            text: '第10章 项目二：自主研究助手 Agent',
            collapsed: false,
            items: [
              {text: '项目概述与需求分析', link: '10-01-project-overview-and-requirements'},
              {text: '核心模块实现', link: '10-02-core-module-implementation'},
              {text: '多步推理与工具调用', link: '10-03-multi-step-reasoning-and-tools'},
              {text: '结果整合与展示', link: '10-04-result-integration-and-presentation'},
              {text: '部署与扩展', link: '10-05-deployment-and-extension'}
            ]
          }
        ]
      }]
    }

function
sidebarTransformers():
    DefaultTheme
    .SidebarItem[] {
      return [{
        text: 'Hugging Face Transformers 教程',
        items: [
          {
            text: '第1章 Transformer 原理与架构入门',
            collapsed: false,
            items: [
              {text: '1.1 从 RNN 到 Transformer', link: '01-01-from-rnn-to-transformer'},
              {text: '1.2 Self-Attention 机制深度拆解', link: '01-02-self-attention-deep-dive'},
              {text: '1.3 Encoder-Decoder 架构全景', link: '01-03-encoder-decoder-architecture'},
              {text: '1.4 Encoder/Decoder 变体', link: '01-04-encoder-decoder-variants'},
              {text: '1.5 HF 生态概览与快速上手', link: '01-05-hf-ecosystem-and-quickstart'}
            ]
          },
          {
            text: '第2章 Tokenizer 文本到模型的桥梁',
            collapsed: false,
            items: [
              {text: '2.1 分词算法演进史', link: '02-01-tokenization-algorithm-evolution'},
              {text: '2.2 HF Tokenizer 核心API', link: '02-02-hf-tokenizer-core-api'},
              {text: '2.3 多语言与多模态 Tokenizer', link: '02-03-multilingual-and-multimodal-tokenizer'},
              {text: '2.4 性能优化与自定义', link: '02-04-performance-optimization-and-custom-tokenizer'}
            ]
          },
          {
            text: '第3章 Model 核心组件与架构',
            collapsed: false,
            items: [
              {text: '3.1 Config 模型的DNA', link: '03-01-config-model-dna'},
              {text: '3.2 Model 的层级结构', link: '03-02-model-layer-hierarchy'},
              {text: '3.3 前向传播全过程追踪', link: '03-03-forward-pass-tracing'},
              {text: '3.4 核心模块源码级解读', link: '03-04-source-code-deep-dive'},
              {text: '3.5 GPT 架构细节与KV Cache', link: '03-05-gpt-architecture-and-kv-cache'},
              {text: '3.6 模型初始化与权重加载', link: '03-06-model-init-and-weight-loading'}
            ]
          },
          {
            text: '第4章 Pipeline 高阶 API 速通',
            collapsed: false,
            items: [
              {text: '4.1 Pipeline 基础与内置任务', link: '04-01-pipeline-basics-and-tasks'},
              {text: '4.2 Pipeline 高级配置', link: '04-02-pipeline-advanced-config'},
              {text: '4.3 自定义 Pipeline', link: '04-03-custom-pipeline'},
              {text: '4.4 Pipeline 性能优化', link: '04-04-pipeline-performance'}
            ]
          },
          {
            text: '第5章 数据集与数据处理',
            collapsed: false,
            items: [
              {text: '5.1 Datasets 库核心功能', link: '05-01-datasets-core'},
              {text: '5.2 数据预处理流水线', link: '05-02-data-preprocessing-pipeline'},
              {text: '5.3 数据增强与清洗', link: '05-03-data-augmentation-and-cleaning'},
              {text: '5.4 自定义数据集', link: '05-04-custom-dataset'}
            ]
          },
          {
            text: '第6章 模型微调 Fine-Tuning',
            collapsed: false,
            items: [
              {text: '6.1 微调的基本概念与策略', link: '06-01-finetuning-concepts-and-strategies'},
              {text: '6.2 Trainer API 入门', link: '06-02-trainer-api-getting-started'},
              {text: '6.3 训练过程监控与调试', link: '06-03-training-monitoring-and-debugging'},
              {text: '6.4 Sequence Classification 微调实战', link: '06-04-complete-finetuning-template'},
              {text: '6.5 高级训练技术', link: '06-05-advanced-training-techniques'},
              {text: '6.6 评估与部署', link: '06-06-evaluation-and-deployment'}
            ]
          },
          {
            text: '第7章 参数高效微调 PEFT',
            collapsed: false,
            items: [
              {text: '7.1 PEFT 概述与 LoRA 原理与实现', link: '07-01-peft-overview-and-lora'},
              {text: '7.2 其他 PEFT 方法 Adapter/Prefix/(IA)³', link: '07-02-adapter-and-other-peft-methods'},
              {text: '7.3 不同任务类型的 PEFT 配置策略', link: '07-03-different-task-peft-configs'},
              {text: '7.4 Adapter 合并与切换', link: '07-04-adapter-merging-and-switching'},
              {text: '7.5 PEFT 生产最佳实践', link: '07-05-peft-production-best-practices'}
            ]
          },
          {
            text: '第8章 文本生成与解码策略',
            collapsed: false,
            items: [
              {text: '8.1 文本生成基础 从 Logits 到 Token', link: '08-01-text-generation-basics'},
              {text: '8.2 高级解码策略', link: '08-02-advanced-decoding-strategies'},
              {text: '8.3 长度控制与重复控制', link: '08-03-length-repetition-control'},
              {text: '8.4 推测解码与 MoE', link: '08-04-speculative-decoding-and-moe'},
              {text: '8.5 自定义生成策略', link: '08-05-custom-generation-strategies'}
            ]
          },
          {
            text: '第9章 模型压缩与推理优化',
            collapsed: false,
            items: [
              {text: '9.1 模型量化基础', link: '09-01-model-quantization'},
              {text: '9.2 知识蒸馏 Knowledge Distillation', link: '09-02-knowledge-distillation'},
              {text: '9.3 模型剪枝 Pruning', link: '09-03-model-pruning'},
              {text: '9.4 推理加速引擎', link: '09-04-inference-engines'},
              {text: '9.5 系统级优化', link: '09-05-system-level-optimization'}
            ]
          },
          {
            text: '第10章 多模态与前沿方向',
            collapsed: false,
            items: [
              {text: '10.1 视觉 Transformer ViT', link: '10-01-vision-transformer'},
              {text: '10.2 音频语音模型', link: '10-02-audio-speech-models'},
              {text: '10.3 多模态融合', link: '10-03-multimodal-fusion'},
              {text: '10.4 视频理解', link: '10-04-video-understanding'},
              {text: '10.5 新兴架构 Mamba/RWKV/MoE/RoPE v2', link: '10-05-emerging-architectures'}
            ]
          }
        ]
      }]
    }

function
sidebarAICoding():
    DefaultTheme
    .SidebarItem[] {
      return [{
        text: 'OpenClaw从入门到精通',
        items: [
          {
            text: '第1章 基础概念',
            collapsed: false,
            items: [
              {text: 'OpenClaw是什么', link: '01-01-what-is-openclaw'},
              {text: 'OpenClaw能做什么', link: '01-02-what-can-openclaw-do'},
              {text: '技术架构概览', link: '01-03-architecture-overview'}
            ]
          },
          {
            text: '第2章 部署指南',
            collapsed: false,
            items: [
              {text: '部署前置准备', link: '02-01-deployment-prerequisites'},
              {text: '本地部署', link: '02-02-local-deployment'},
              {text: '云端部署', link: '02-03-cloud-deployment'},
              {text: '初始化配置', link: '02-04-initialization'}
            ]
          },
          {
            text: '第3章 配置定制',
            collapsed: false,
            items: [
              {text: '配置文件详解', link: '03-01-config-files'},
              {text: '定义AI身份', link: '03-02-define-soul'},
              {text: '设置用户偏好', link: '03-03-user-preferences'},
              {text: '定时任务配置', link: '03-04-heartbeat-tasks'}
            ]
          },
          {
            text: '第4章 模型配置',
            collapsed: false,
            items: [
              {text: '支持的AI模型提供商', link: '04-01-supported-models'},
              {text: 'API Key配置', link: '04-02-api-key-config'},
              {text: '模型策略与成本优化', link: '04-03-model-strategy'},
              {text: '本地模型部署', link: '04-04-ollama-deployment'}
            ]
          },
          {
            text: '第5章 技能系统',
            collapsed: false,
            items: [
              {text: 'Skills技能系统概述', link: '05-01-skills-overview'},
              {text: '必备核心技能', link: '05-02-core-skills'},
              {text: '技能管理命令', link: '05-03-skill-management'},
              {text: '自定义Skill开发', link: '05-04-custom-skill'},
              {text: '技能安全', link: '05-05-skill-security'}
            ]
          },
          {
            text: '第6章 消息渠道',
            collapsed: false,
            items: [
              {text: '支持的消息平台', link: '06-01-message-platforms'},
              {text: '渠道配置方法', link: '06-02-channel-config'},
              {text: '多账户管理', link: '06-03-multi-account'}
            ]
          },
          {
            text: '第7章 日常使用',
            collapsed: false,
            items: [
              {text: '访问方式', link: '07-01-access-methods'},
              {text: '常用CLI命令', link: '07-02-cli-commands'},
              {text: '常见问题排查', link: '07-03-troubleshooting'}
            ]
          },
          {
            text: '第8章 实战案例',
            collapsed: false,
            items: [
              {text: '自动发微信公众号文章', link: '08-01-wechat-article'},
              {text: '日报自动生成与发送', link: '08-02-daily-report'},
              {text: '7×24小时信息中枢', link: '08-03-info-hub'},
              {text: '个人知识库管理', link: '08-04-knowledge-base'}
            ]
          },
          {
            text: '第9章 进阶玩法',
            collapsed: false,
            items: [
              {text: '多Agent配置', link: '09-01-multi-agent'},
              {text: 'RAG检索增强生成', link: '09-02-rag'},
              {text: '上下文记忆配置', link: '09-03-context-memory'},
              {text: '数据分析与优化', link: '09-04-analytics'}
            ]
          },
          {
            text: '附录',
            collapsed: false,
            items: [
              {text: 'OpenClaw命令速查表', link: '10-01-command-reference'},
              {text: '常用Skill推荐清单', link: '10-02-skill-recommendations'},
              {text: '各模型提供商控制台地址', link: '10-03-model-providers'},
              {text: '安全配置检查清单', link: '10-04-security-checklist'},
              {text: '相关资源链接', link: '10-05-resources'}
            ]
          }
        ]
      }]
    }

function
sidebarPandas(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '深入浅出Pandas',
      items: [
        {
      text: '第1章 基本介绍',
      collapsed: false,
      items: [
        {text: '什么是 Pandas', link: '01-01-what-is-pandas'},
        {text: '为什么大模型需要 Pandas', link: '01-02-why-pandas-for-llm'},
        {text: '生态与学习路线', link: '01-03-ecosystem-and-roadmap'}
      ]
    },
    {
      text: '第2章 安装与环境配置',
      collapsed: false,
      items: [
        {text: '安装 Pandas', link: '03-01-installation'},
        {text: 'Pandas 3.0 要点速览', link: '03-02-pandas3-highlights'},
        {text: 'PyArrow 后端基础', link: '03-03-pyarrow-basics'},
        {text: '开发环境推荐', link: '03-04-environment-setup'}
      ]
    },
        {
          text: '第3章 核心数据结构',
          collapsed: false,
          items: [
            {text: 'Series：一维带标签数组', link: '04-01-series-in-depth'},
            {text: 'DataFrame：二维表格', link: '04-02-dataframe-in-depth'},
            {text: '数据查看与探索', link: '04-03-data-exploration'},
            {text: '数据类型（dtype）体系', link: '04-04-dtype-system'}
          ]
        },
        {
          text: '第4章 数据读写 I/O',
          collapsed: false,
          items: [
            {
              text: '文本格式：CSV 与 JSON',
              link: '05-01-text-formats-csv-json'
            },
            {text: '数据库读写 SQL', link: '05-02-database-sql'},
            {text: '大模型语料专用格式', link: '05-03-special-formats'},
            {text: '数据加载最佳实践', link: '05-04-loading-best-practices'}
          ]
        },
        {
          text: '第5章 数据审查与探索',
          collapsed: false,
          items: [
            {text: '基础信息查看', link: '06-01-basic-info-overview'},
            {text: '缺失值识别与统计', link: '06-02-missing-values'},
            {text: '重复值检测', link: '06-03-duplicates-detection'},
            {text: '数据质量报告生成模板', link: '06-04-data-quality-report'}
          ]
        },
        {
          text: '第6章 数据清洗实战',
          collapsed: false,
          items: [
            {text: '缺失值处理', link: '07-01-missing-values-handling'},
            {text: '重复值处理', link: '07-02-duplicates-handling'},
            {text: '数据类型转换', link: '07-03-data-type-conversion'},
            {text: '字符串处理', link: '07-04-string-processing'}, {
              text: '大模型驱动的高阶数据清洗',
              link: '07-05-llm-driven-cleaning'
            }
          ]
        },
        {
          text: '第7章 数据筛选',
          collapsed: false,
          items: [
            {text: '布尔索引', link: '08-01-boolean-indexing'},
            {text: 'query() 查询方法', link: '08-02-query-method'},
            {text: '便捷过滤函数', link: '08-03-convenient-filters'},
            {text: 'SFT 数据集筛选实战', link: '08-04-sft-filtering-scenario'}
          ]
        },
        {
          text: '第8章 数据转换与特征工程',
          collapsed: false,
          items: [
            {text: '列与行选择', link: '09-01-column-row-selection'},
            {text: '列的新增与修改', link: '09-02-column-modification'}, {
              text: 'apply() 深度解析与向量化',
              link: '09-03-apply-vectorization'
            },
            {text: '数据重塑：宽长格式转换', link: '09-04-data-reshaping'},
            {text: '特征工程实战', link: '09-05-feature-engineering'}
          ]
        },
        {
          text: '第9章 排序与排名',
          collapsed: false,
          items: [
            {text: '排序基础', link: '10-01-sorting'},
            {text: 'rank() 排名方法', link: '10-02-rank'}, {
              text: 'nlargest() / nsmallest() TopN 选择',
              link: '10-03-topn-selection'
            },
            {text: '排序实战场景', link: '10-04-sorting-scenarios'}
          ]
        },
        {
          text: '第10章 分组聚合与透视',
          collapsed: false,
          items: [
            {text: 'groupby() 分组基础', link: '11-01-groupby-basics'},
            {text: '聚合方法详解', link: '11-02-aggregation-methods'},
            {text: '分组转换与过滤', link: '11-03-transform-filter'},
            {text: '数据透视表与交叉表', link: '11-04-pivot-crosstab'},
            {text: '分组后拆分与导出', link: '11-05-groupby-split'},
            {text: '分组聚合实战场景', link: '11-06-groupby-scenarios'}
          ]
        },
        {
          text: '第11章 数据合并与连接',
          collapsed: false,
          items: [
            {text: 'merge() 基础', link: '12-01-merge-basics'},
            {text: 'merge() 高级技巧', link: '12-02-merge-advanced'},
            {text: 'join() 索引连接', link: '12-03-join-index'},
            {text: '合并性能优化', link: '12-04-merge-performance'},
            {text: '合并与连接实战', link: '12-05-merge-scenarios'}
          ]
        },
        {
          text: '第12章 数据拼接与组合',
          collapsed: false,
          items: [
            {text: 'concat() 数据拼接', link: '13-01-concat-basics'},
            {text: 'combine_first() 与 update()', link: '13-02-combine-update'},
            {text: '去重合并与对齐', link: '13-03-dedup-align'},
            {text: '拼接综合实战', link: '13-04-merge-scenarios'}
          ]
        },
        {
          text: '第13章 时间序列处理',
          collapsed: false,
          items: [
            {text: '时间序列基础与解析', link: '14-01-timeseries-basics'},
            {text: '时间索引与重采样', link: '14-02-resample-rolling'},
            {text: '偏移与差分', link: '14-03-shift-diff'},
            {text: 'API 监控与趋势分析', link: '14-04-timeseries-monitoring'},
            {text: '训练日志与成本趋势', link: '14-05-timeseries-scenarios'}
          ]
        },
        {
          text: '第14章 数据可视化',
          collapsed: false,
          items: [
            {text: 'Pandas 内置可视化基础', link: '15-01-plotting-basics'},
            {text: '高级可视化技巧', link: '15-02-advanced-plotting'},
            {text: 'LLM 评估可视化实战', link: '15-03-llm-visualization'}, {
              text: '可视化最佳实践与导出',
              link: '15-04-visualization-best-practices'
            }
          ]
        },
        {
          text: '第15章 高级操作技巧',
          collapsed: false,
          items: [
            {text: '迭代器与分块处理', link: '16-01-iteration-chunking'},
            {text: '样式与格式化输出', link: '16-02-style-formatting'},
            {text: 'MultiIndex 深度操作', link: '16-03-multiindex'},
            {text: 'Categorical 深度应用', link: '16-04-categorical-deep'},
            {text: '管道式链式操作', link: '16-05-pipeline-chaining'}
          ]
        },
        {
          text: '第16章 性能优化',
          collapsed: false,
          items: [
            {text: '内存优化实战', link: '17-01-memory-optimization'},
            {text: '计算性能优化', link: '17-02-compute-performance'},
            {text: 'I/O 性能优化', link: '17-03-io-performance'},
            {text: '性能分析工具集', link: '17-04-performance-toolkit'}
          ]
        },
        {
          text: '第17章 PandasAI 与自然语言查询',
          collapsed: false,
          items: [
            {text: 'PandasAI 简介与环境搭建', link: '18-01-pandasai-intro'},
            {text: 'PandasAI 核心用法', link: '18-02-pandasai-core'},
            {text: '高级功能与最佳实践', link: '18-03-pandasai-advanced'},
            {text: 'LLM 开发中的实战', link: '18-04-pandasai-scenarios'}
          ]
        },
        {
          text: '第19章 RAG 知识库管理',
          collapsed: false,
          items: [
            {text: '知识库数据结构设计', link: '19-01-rag-schema'},
            {text: '知识库检索与排序', link: '19-02-rag-retrieval'},
            {text: '知识库管理与维护', link: '19-03-rag-management'},
            {text: '完整工作流实战', link: '19-04-rag-workflow'}
          ]
        },
        {
          text: '第20章 综合项目实战',
          collapsed: false,
          items: [
            {text: 'SFT 数据集处理流水线', link: '20-01-sft-pipeline'},
            {text: 'SFT 分层抽样与导出', link: '20-02-sft-export'},
            {text: '模型评估报告系统', link: '20-03-eval-report'},
            {text: 'API 成本监控仪表盘', link: '20-04-cost-dashboard'}
          ]
        },
        {
          text: '第18章 LangChain + Pandas Agent',
          collapsed: false,
          items: [
            {text: 'Pandas × LangChain 概览', link: '21-01-langchain-overview'},
            {text: '数据分析 Agent 构建', link: '21-02-analysis-agent'},
            {text: 'Agent 记忆与状态管理', link: '21-03-agent-memory'},
            {text: 'Agent 综合应用场景', link: '21-04-langchain-scenarios'}
          ]
        },
        {
          text: '第19章 MCP Server 与多模态',
          collapsed: false,
          items: [
            {text: 'MCP 协议与 Pandas 服务', link: '22-01-mcp-overview'},
            {text: 'MCP 工具定义与路由', link: '22-02-mcp-tools'},
            {text: 'MCP 资源管理与版本', link: '22-03-mcp-resources'},
            {text: '生产级 MCP Server 实现', link: '22-04-mcp-complete'}
          ]
        },
        {
          text: '第23章 多模态数据处理',
          collapsed: false,
          items: [
            {text: '图像数据管理', link: '23-01-image-data'},
            {text: '音频数据管理', link: '23-02-audio-data'},
            {text: '多模态数据统一框架', link: '23-03-multimodal-unified'},
            {text: '多模态场景实战', link: '23-04-multimodal-scenarios'}
          ]
        },
        {
          text: '第20章 分布式处理与总结',
          collapsed: false,
          items: [
            {text: 'Dask 基础入门', link: '24-01-dask-basics'},
            {text: 'Modin 与 Ray 集成', link: '24-02-modin-ray'},
            {text: '分布式最佳实践', link: '24-03-distributed-best-practices'},
            {text: '分布式场景实战', link: '24-04-distributed-scenarios'}
          ]
        },
        {
          text: '第25章 总结与进阶路线',
          collapsed: false,
          items: [
            {text: '知识体系全景总结', link: '25-01-knowledge-summary'},
            {text: '学习路线图', link: '25-02-learning-roadmap'},
            {text: '进阶主题与生态扩展', link: '25-03-advanced-topics'}
          ]
        },
        {
          text: '第21章 案例1：千万级语料清洗',
          collapsed: false,
          items: [
            {text: '加载原始对话日志', link: '26-01-load-conversation-data'}, {
              text: '数据质量分析报告生成',
              link: '26-02-quality-analysis-report'
            },
            {
              text: '使用 LLM 辅助清洗异常数据',
              link: '26-03-llm-assisted-cleaning'
            },
            {
              text: '导出为 SFT / DPO 训练格式',
              link: '26-04-export-sft-dpo-format'
            }
          ]
        },
        {
          text: '第22章 案例2：RAG 知识库构建',
          collapsed: false,
          items: [
            {text: '文档解析与分块', link: '27-01-document-parsing-chunking'},
            {text: '元数据管理', link: '27-02-metadata-management'}, {
              text: '向量索引构建前的数据准备',
              link: '27-03-vector-prep-chunking'
            },
            {
              text: '构建端到端的自动化流水线',
              link: '27-04-end-to-end-pipeline'
            }
          ]
        },
        {
          text: '第23章 案例3：LangChain Pandas Agent',
          collapsed: false,
          items: [
            {text: '环境配置与 API Key 设置', link: '28-01-env-setup-apikey'}, {
              text: '创建 Pandas DataFrame Agent',
              link: '28-02-create-pandas-agent'
            },
            {text: '自然语言问答交互', link: '28-03-natural-language-qa'},
            {text: '部署为简单的 Web 应用', link: '28-04-web-app-deployment'}
          ]
        },
        {
          text: '第24章 案例4：模型评估对比分析',
          collapsed: false,
          items: [
            {
              text: '加载多个模型的推理结果',
              link: '29-01-load-multi-model-results'
            },
            {
              text: '自动化指标计算与对比表生成',
              link: '29-02-auto-metrics-comparison'
            },
            {text: '可视化对比图表', link: '29-03-visualization-charts'},
            {text: '生成评估报告', link: '29-04-generate-eval-report'}
          ]
        },
        {
          text: '第25章 案例5：迁移 Polars 获得加速',
          collapsed: false,
          items: [
            {
              text: '识别性能瓶颈（使用 %timeit 测量）',
              link: '30-01-identify-bottlenecks'
            },
            {
              text: '语法迁移：关键函数对照与改写',
              link: '30-02-syntax-migration-guide'
            },
            {
              text: '混合使用：Pandas ↔ Polars 数据互转',
              link: '30-03-hybrid-pandas-polars'
            },
            {
              text: '迁移前后的性能对比报告',
              link: '30-04-migration-performance-report'
            }
          ]
        }
      ]
    },
  ]
}

function
sidebarLlamaIndex():
    DefaultTheme
    .SidebarItem[] {
      return [{
        text: 'LlamaIndex 教程',
        items: [
          {
            text: '第1章 LlamaIndex 快速入门',
            collapsed: false,
            items: [
              {text: '为什么需要 LlamaIndex', link: '01-01-why-llamaindex'},
              {text: '安装与环境配置', link: '01-02-installation-and-setup'},
              {text: '第一个 RAG 应用', link: '01-03-first-rag-app'},
              {text: '与 LangChain RAG 对比', link: '01-04-vs-langchain-comparison'},
              {text: '核心概念预览', link: '01-05-core-concepts-preview'},
            ]
          },
          {
            text: '第2章 数据连接器 Connectors',
            collapsed: false,
            items: [
              {text: '统一数据接入体系', link: '02-01-connectors-overview'},
              {text: '文件连接器', link: '02-02-file-connectors'},
              {text: '数据库连接器', link: '02-03-database-connectors'},
              {text: 'API 连接器', link: '02-04-api-connectors'},
              {text: '云服务连接器', link: '02-05-cloud-connectors'},
              {text: '自定义连接器', link: '02-06-custom-connectors'},
              {text: '多源数据融合', link: '02-07-multi-source-fusion'},
            ]
          },
          {
            text: '第3章 文档加载与解析',
            collapsed: false,
            items: [
              {text: '解析挑战与策略', link: '03-01-parsing-challenges'},
              {text: 'Node 解析器详解', link: '03-02-node-parsers'},
              {text: '层次化解析', link: '03-03-hierarchical-parsing'},
              {text: '自定义解析策略', link: '03-04-custom-parsing-strategies'},
              {text: '解析质量评估', link: '03-05-parsing-evaluation'},
            ]
          },
          {
            text: '第4章 索引策略进阶',
            collapsed: false,
            items: [
              {text: 'VectorStoreIndex 深度解析', link: '04-01-vectorstore-index-deep-dive'},
              {text: '其他索引类型', link: '04-02-other-index-types'},
              {text: '索引组合', link: '04-03-index-composition'},
              {text: '向量存储后端', link: '04-04-vector-store-backends'},
              {text: '持久化与更新', link: '04-05-persistence-and-updates'},
            ]
          },
          {
            text: '第5章 高级检索技术',
            collapsed: false,
            items: [
              {text: '混合检索 Hybrid Search', link: '05-01-hybrid-retrieval'},
              {text: '重排序 Reranking', link: '05-02-reranking'},
              {text: 'HyDE 与查询变换', link: '05-03-hyde-and-query-transform'},
              {text: '后处理 Postprocessing', link: '05-04-postprocessing'},
              {text: '检索最佳实践', link: '05-05-retrieval-best-practices'},
            ]
          },
          {
            text: '第6章 查询引擎 Query Engine',
            collapsed: false,
            items: [
              {text: '查询引擎架构', link: '06-01-query-engine-architecture'},
              {text: 'RetrieverQueryEngine', link: '06-02-retriever-query-engine'},
              {text: 'ChatEngine 多轮对话', link: '06-03-chat-engine'},
              {text: '自定义查询引擎', link: '04-custom-query-engine'},
              {text: '流式输出与异步', link: '06-05-streaming-and-async'},
            ]
          },
          {
            text: '第7章 响应合成 Response Synthesis',
            collapsed: false,
            items: [
              {text: '合成面临的挑战', link: '07-1-synthesis-challenges'},
              {text: '合成模式深度剖析', link: '07-2-synthesis-modes-deep-dive'},
              {text: '自定义合成器', link: '07-3-custom-synthesizer'},
              {text: '结构化输出', link: '07-4-structured-output'},
              {text: '合成质量优化', link: '07-5-synthesis-quality-optimization'},
            ]
          },
          {
            text: '第8章 评估与调试',
            collapsed: false,
            items: [
              {text: '评估体系概览', link: '08-01-evaluation-overview'},
              {text: '检索质量评估', link: '08-02-retrieval-evaluation'},
              {text: '生成质量评估', link: '08-3-generation-evaluation'},
              {text: '调试与可观测性', link: '08-4-debugging-and-observability'},
              {text: '完整评估工作流与生产实践', link: '08-5-evaluation-pipeline-and-ci'},
            ]
          },
          {
            text: '第9章 项目一：企业知识库问答系统',
            collapsed: false,
            items: [
              {text: '项目概述与需求分析', link: '09-01-project-overview-and-requirements'},
              {text: '系统架构设计', link: '09-02-system-architecture-design'},
              {text: '核心模块实现', link: '09-03-core-module-implementation'},
              {text: 'API 服务与前后端集成', link: '09-04-api-and-frontend'},
              {text: '部署运维与性能优化', link: '09-05-deployment-and-optimization'},
            ]
          },
          {
            text: '第10章 项目二：多模态 RAG 应用',
            collapsed: false,
            items: [
              {text: '多模态 RAG 概述与场景分析', link: '10-01-multimodal-rag-overview'},
              {text: '多模态数据接入与解析', link: '10-02-multimodal-data-ingestion'},
              {text: '多模态检索与融合', link: '10-03-multimodal-retrieval'},
              {text: '多模态问答引擎实现', link: '10-04-multimodal-qa-engine'},
              {text: '前端展示与部署', link: '10-05-frontend-and-deployment'},
            ]
          },
        ]
      }]
    }
