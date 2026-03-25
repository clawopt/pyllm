<template>
  <div class="home-wrapper">
    <div class="main-container">
      <div class="sidebar desktop-nav">
        <button
          v-for="(category, index) in categories"
          :key="category.title"
          :class="['nav-item', { active: activeIndex === index }]"
          @click="activeIndex = index"
        >
          <span class="nav-icon">
            <svg v-if="category.title==='Python'" class="icon-svg" viewBox="0 0 24 24">
              <path d="M8 6 4 12 8 18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M16 6 20 12 16 18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M12 5 12 19" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/>
            </svg>
            <svg v-else-if="category.title==='LLM'" class="icon-svg" viewBox="0 0 24 24">
              <circle cx="7" cy="8" r="3" fill="currentColor"/>
              <circle cx="17" cy="8" r="3" fill="currentColor"/>
              <circle cx="12" cy="16" r="3" fill="currentColor"/>
              <path d="M7 11 L12 13 L17 11" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/>
            </svg>
            <svg v-else-if="category.title==='数据库'" class="icon-svg" viewBox="0 0 24 24">
              <ellipse cx="12" cy="6" rx="7" ry="3" stroke="currentColor" stroke-width="2" fill="none"/>
              <path d="M5 6v8" stroke="currentColor" stroke-width="2"/>
              <path d="M19 6v8" stroke="currentColor" stroke-width="2"/>
              <ellipse cx="12" cy="14" rx="7" ry="3" stroke="currentColor" stroke-width="2" fill="none"/>
            </svg>
            <svg v-else class="icon-svg" viewBox="0 0 24 24">
              <rect x="3" y="6" width="18" height="11" rx="2" stroke="currentColor" stroke-width="2" fill="none"/>
              <path d="M2 19h20" stroke="currentColor" stroke-width="2"/>
            </svg>
          </span>
          <span class="nav-text">{{ category.title }}</span>
        </button>
      </div>

      <div class="mobile-nav">
        <div class="tabs-container">
          <button
            v-for="(category, index) in categories"
            :key="category.title"
            :class="['tab-item', { active: activeIndex === index }]"
            @click="activeIndex = index"
          >
            <span class="tab-icon">
              <svg v-if="category.title==='Python'" class="icon-svg" viewBox="0 0 24 24">
                <path d="M8 6 4 12 8 18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M16 6 20 12 16 18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 5 12 19" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/>
              </svg>
              <svg v-else-if="category.title==='LLM'" class="icon-svg" viewBox="0 0 24 24">
                <circle cx="7" cy="8" r="3" fill="currentColor"/>
                <circle cx="17" cy="8" r="3" fill="currentColor"/>
                <circle cx="12" cy="16" r="3" fill="currentColor"/>
                <path d="M7 11 L12 13 L17 11" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/>
              </svg>
              <svg v-else-if="category.title==='数据库'" class="icon-svg" viewBox="0 0 24 24">
                <ellipse cx="12" cy="6" rx="7" ry="3" stroke="currentColor" stroke-width="2" fill="none"/>
                <path d="M5 6v8" stroke="currentColor" stroke-width="2"/>
                <path d="M19 6v8" stroke="currentColor" stroke-width="2"/>
                <ellipse cx="12" cy="14" rx="7" ry="3" stroke="currentColor" stroke-width="2" fill="none"/>
              </svg>
              <svg v-else class="icon-svg" viewBox="0 0 24 24">
                <rect x="3" y="6" width="18" height="11" rx="2" stroke="currentColor" stroke-width="2" fill="none"/>
                <path d="M2 19h20" stroke="currentColor" stroke-width="2"/>
              </svg>
            </span>
            <span class="tab-text">{{ category.title }}</span>
          </button>
        </div>
      </div>

      <div class="content-area">
        <div class="courses-grid">
          <a
            v-for="course in categories[activeIndex].courses"
            :key="course.link"
            :href="withBase(course.link)"
            class="course-card"
          >
            <h3 class="course-name">{{ course.name }}</h3>
            <p class="course-desc">{{ course.desc }}</p>
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { withBase } from 'vitepress'

const activeIndex = ref(0)

const categories = [
  {
    icon: '🐍',
    title: 'Python',
    courses: [
      { name: 'Python核心教程', desc: 'Python编程基础与核心概念', link: '/pages/python/core/00-intro' },
      { name: 'NumPy教程', desc: '科学计算与数组操作', link: '/pages/python/numpy/' },
      { name: 'Pandas教程', desc: '数据分析与处理', link: '/pages/python/pandas/' },
      { name: 'Matplotlib教程', desc: '数据可视化与绘图', link: '/pages/python/matplotlib/' }
    ]
  },
  {
    icon: '🤖',
    title: 'LLM',
    courses: [
      { name: 'PyTorch Lightning教程', desc: '快速训练深度学习模型', link: '/pages/llm/pytorch-lightning/' },
      { name: 'Hugging Face Transformers', desc: 'Transformer模型与应用', link: '/pages/llm/transformers/' },
      { name: 'Ollama教程', desc: '本地部署大语言模型', link: '/pages/llm/ollama/' },
      { name: 'vLLM教程', desc: '高效推理与服务部署', link: '/pages/llm/vllm/' },
      { name: 'LangChain教程', desc: '构建LLM应用链', link: '/pages/llm/langchain/' }
    ]
  },
  {
    icon: '🗄️',
    title: '数据库',
    courses: [
      { name: 'PG Vector教程', desc: 'PostgreSQL向量扩展', link: '/pages/database/pgvector/' },
      { name: 'Milvus教程', desc: '向量数据库与相似度搜索', link: '/pages/database/milvus/' },
      { name: 'Chroma教程', desc: '轻量级向量数据库', link: '/pages/database/chroma/' },
      { name: 'Faiss教程', desc: 'Facebook相似性搜索', link: '/pages/database/faiss/' },
      { name: 'DuckDB教程', desc: '分析型嵌入式数据库', link: '/pages/database/duckdb/' },
      { name: 'LanceDB教程', desc: '向量数据库的新选择', link: '/pages/database/lancedb/' }
    ]
  },
  {
    icon: '💻',
    title: 'AI编程',
    courses: [
      { name: 'OpenClaw教程', desc: 'AI编程助手', link: '/pages/ai-coding/openclaw/' },
      { name: 'OpenCode教程', desc: '智能代码生成', link: '/pages/ai-coding/opencode/' },
      { name: 'DeepSeek教程', desc: 'DeepSeek模型应用', link: '/pages/ai-coding/deepseek/' }
    ]
  }
]
</script>

<style scoped>
.home-wrapper {
  width: 100%;
  margin: 1rem 0 0;
  padding: 0 0 3rem;
}

.main-container {
  display: flex;
  flex-direction: column;
  gap: 0;
  background: var(--vp-c-bg);
  border-radius: 16px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
  overflow: hidden;
}

@media (min-width: 1024px) {
  .main-container {
    flex-direction: row;
  }
}

.mobile-nav {
  display: block;
  padding: 1rem;
  border-bottom: 1px solid var(--vp-c-divider);
}

@media (min-width: 1024px) {
  .mobile-nav {
    display: none;
  }
}

.tabs-container {
  display: flex;
  gap: 0.5rem;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
}

.tabs-container::-webkit-scrollbar {
  display: none;
}

.tab-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
  padding: 0.75rem 1.25rem;
  background: transparent;
  border: none;
  cursor: pointer;
  border-radius: 9999px;
  transition: all 0.2s ease;
  color: var(--vp-c-text-2);
}

.tab-item:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.tab-item.active {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

.tab-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  color: var(--vp-c-brand-1);
}

.tab-text {
  font-size: 0.9rem;
  font-weight: 500;
  white-space: nowrap;
}

.desktop-nav {
  display: none;
}

@media (min-width: 1024px) {
  .desktop-nav {
    display: block;
    width: 240px;
    flex-shrink: 0;
    padding: 1.5rem 0;
    border-right: 1px solid var(--vp-c-divider);
  }
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  width: 100%;
  padding: 0.875rem 1.5rem;
  background: transparent;
  border: none;
  cursor: pointer;
  text-align: left;
  transition: all 0.15s ease;
  color: var(--vp-c-text-2);
}

.nav-item:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.nav-item.active {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-left: 3px solid var(--vp-c-brand-1);
}

.nav-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  color: var(--vp-c-brand-1);
}

.nav-text {
  font-size: 0.95rem;
  font-weight: 500;
}
 
.icon-svg {
  width: 18px;
  height: 18px;
}
 
.tab-item.active .tab-icon {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

.content-area {
  flex: 1;
  padding: 1.5rem;
}

@media (min-width: 1024px) {
  .content-area {
    padding: 2rem;
  }
}

.courses-grid {
  display: grid;
  grid-template-columns: repeat(1, 1fr);
  gap: 1.25rem;
}

@media (min-width: 768px) {
  .courses-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 1024px) {
  .courses-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

.course-card {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1.25rem;
  text-decoration: none;
  border-radius: 12px;
  transition: all 0.2s ease;
}

.course-card:hover {
  background: var(--vp-c-bg-soft);
}

.course-card:active {
  background: var(--vp-c-gray-soft);
}

.course-name {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0;
  line-height: 1.3;
}

.course-desc {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  margin: 0;
  line-height: 1.5;
}
</style>
