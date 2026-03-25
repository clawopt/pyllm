<template>
  <div class="custom-hero">
    <div class="hero-left">
      <h1 class="hero-name">
        <span class="hero-line">PyLLM</span>
        <span class="hero-line sub">懂 Python，更懂大模型</span>
      </h1>
      <p class="hero-text">一站式大模型教程学习平台</p>
      <div class="hero-actions">
        <a :href="withBase('/pages/python/core/00-intro')" class="vp-button brand medium">立即开始</a>
        <a :href="withBase('/llm-learning-path')" class="vp-button alt medium">怎么学？看大模型学习路线！</a>
      </div>
    </div>
    <div class="hero-mid">
      <div class="recommend-section">
        <h3 class="recommend-title">精选推荐</h3>
        <div class="recommend-list">
          <a href="https://gpt.paperily.cn/" target="_blank" rel="noopener" class="recommend-item">
            极速高效方便的论文阅读神器
          </a>
          <a :href="withBase('/pages/llm/vllm/')" class="recommend-item">
            高吞吐量推理与部署 vLLM
          </a>
          <a :href="withBase('/pages/llm/transformers/')" class="recommend-item">
            Transformer 模型库入门
          </a>
          <a :href="withBase('/pages/database/milvus/')" class="recommend-item">
            向量数据库：Milvus 实战
          </a>
          <a :href="withBase('/pages/ai-coding/deepseek/')" class="recommend-item">
            DeepSeek 模型应用精选
          </a>
        </div>
      </div>
    </div>
    <div class="hero-right">
      <div class="blog-section">
        <h3 class="blog-title">最新博客</h3>
        <div class="blog-list">
          <a :href="withBase('/pages/blog/llm-2024-review')" class="blog-item">
            <span class="blog-date">2024-12-20</span>
            <span class="blog-name">2024年大模型技术回顾与展望</span>
          </a>
          <a :href="withBase('/pages/blog/python-tips-2024')" class="blog-item">
            <span class="blog-date">2024-12-15</span>
            <span class="blog-name">Python高级编程技巧总结</span>
          </a>
          <a :href="withBase('/pages/blog/vector-db-comparison')" class="blog-item">
            <span class="blog-date">2024-12-10</span>
            <span class="blog-name">向量数据库对比：Milvus vs Chroma</span>
          </a>
        </div>
        <a :href="withBase('/pages/blog/')" class="blog-more">更多 →</a>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { withBase } from 'vitepress'
</script>

<style scoped>
.custom-hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  max-width: 1280px;
  margin: 0 auto;
  padding: 1.5rem 1.5rem;
  gap: 1.25rem;
  min-height: 320px;
}

@media (max-width: 767px) {
  .custom-hero {
    flex-direction: column;
    gap: 1.5rem;
    padding: 1rem 1.5rem;
    min-height: auto;
  }

  .hero-mid {
    width: 100% !important;
    margin-left: 0 !important;
  }

  .recommend-section {
    width: 100% !important;
    margin: 0 !important;
  }

  .hero-right {
    width: 100% !important;
  }

  .recommend-title,
  .blog-title {
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
  }
}

@media (max-width: 1024px) {
  .hero-mid,
  .hero-right {
    width: 100% !important;
    margin-left: 0 !important;
  }

  .recommend-section {
    width: 100% !important;
    margin: 0 !important;
  }

  .recommend-title,
  .blog-title,
  .recommend-item,
  .blog-item {
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
    box-sizing: border-box;
  }
}

.hero-left {
  flex: 1;
  min-width: 0;
}

.hero-name {
  font-size: 2rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin: 0 0 0.5rem;
  line-height: 1.2;
  white-space: normal;
}

.hero-line {
  display: block;
  margin: 0;
}

.hero-line.sub {
  font-weight: 700;
  font-size: 0.8em;
}

.hero-line:first-child {
  color: var(--vp-c-brand-1);
  margin-bottom: 2rem;
}

.hero-text {
  font-size: 1rem;
  color: var(--vp-c-text-2);
  margin: 0 0 1.5rem;
}

.hero-actions {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.vp-button {
  display: inline-block;
  padding: 0.6rem 1.2rem;
  border-radius: 6px;
  text-decoration: none;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.vp-button.brand {
  background: var(--vp-c-brand-1);
  color: white;
}

.vp-button.brand:hover {
  background: var(--vp-c-brand-2);
}

.vp-button.alt {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  border: 1px solid var(--vp-c-divider);
}

.vp-button.alt:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.hero-right {
  flex-shrink: 0;
  width: 380px;
}

.hero-mid {
  flex-shrink: 0;
  width: 340px;
  margin-left: auto;
}

.recommend-section {
  width: 70%;
  margin: 0 auto;
}

.blog-section {
  padding-top: 0.5rem;
  display: flex;
  flex-direction: column;
}

.recommend-section {
  padding-top: 0.5rem;
  /* width/margin handled above and in responsive rules */
}

.recommend-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0 0 1rem;
}

.recommend-list {
  display: flex;
  flex-direction: column;
  gap: 0.625rem;
}

.recommend-item {
  display: block;
  padding: 0.625rem 0.75rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  text-decoration: none;
  color: var(--vp-c-text-1);
  transition: all 0.2s ease;
  font-size: 0.875rem;
  line-height: 1.3;
}

.recommend-item:hover {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

.blog-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0 0 1rem;
}

.blog-list {
  display: flex;
  flex-direction: column;
  gap: 0.625rem;
}

.blog-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  padding: 0.625rem 0.75rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  text-decoration: none;
  transition: all 0.2s ease;
}

.blog-item:hover {
  background: var(--vp-c-brand-soft);
}

.blog-date {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}

.blog-name {
  font-size: 0.875rem;
  color: var(--vp-c-text-1);
  font-weight: 500;
  line-height: 1.3;
}

.blog-more {
  display: inline-block;
  margin-top: 0.875rem;
  font-size: 0.875rem;
  color: var(--vp-c-brand-1);
  text-decoration: none;
  font-weight: 500;
  align-self: flex-end;
}

.blog-more:hover {
  text-decoration: underline;
}
</style>
