# 10.5 前端展示与部署

## 让多模态能力触手可及：前端体验设计

前面四节我们完成了多模态 RAG 系统的全部后端逻辑——数据解析、跨模态检索、问答引擎、API 服务。但一个系统如果只有后端没有好的前端，就像一台法拉利引擎装在了拖拉机底盘上——性能再好用户也感知不到。这一节要做的是给这台"法拉利"配上一套与之匹配的"驾驶舱"——一个精心设计的多模态交互界面，以及将整套系统部署到生产环境的完整方案。

## 多模态前端架构

与第9章的企业知识库前端相比，多模态前端需要额外处理几个维度的复杂性：

```
传统 RAG 前端 vs 多模态 RAG 前端的对比:

┌─────────────────────┐    ┌─────────────────────────────┐
│   传统 RAG 前端      │    │   多模态 RAG 前端              │
│                     │    │                             │
│ ┌───────────┐      │    │ ┌───────────┐  ┌──────────┐ │
│ │ 文本输入框  │      │    │ │ 文本输入框  │  │ 图片上传区│ │
│ └───────────┘      │    │ └───────────┘  └──────────┘ │
│                     │    │        ↓          ↓         │
│ ┌───────────┐      │    │ ┌──────────────────────┐   │
│ │ 回答文本区  │      │    │ │ 统一消息流            │   │
│ │ (Markdown) │      │    │ │ · 文本回答             │   │
│ └───────────┘      │    │ │ · 引用卡片(文本)       │   │
│                     │    │ │ · 图片引用卡片 (可预览)  │   │
│ ┌───────────┐      │    │ │ · 表格引用 (可展开)     │   │
│ │ 来源列表    │      │    │ │ · 图表引用 (可交互)     │   │
│ └───────────┘      │    │ └──────────────────────┘   │
│                     │    │                             │
│ 单一内容类型         │    │ 混合内容类型 + 富交互       │
└─────────────────────┘    └─────────────────────────────┘
```

核心差异在于**内容的异构性**。传统 RAG 的回答只包含文本和文本来源链接，渲染起来很简单——用 Markdown 渲染器转成 HTML 就行。但多模态 RAG 的回答可能同时包含文字段落、内嵌的图片引用（需要显示缩略图和点击放大）、表格数据（可能需要可排序的表格组件）、以及图表描述（如果能还原成可视化图表就更棒了）。

下面是一个完整的 Vue 3 组件实现，展示如何处理这些复杂的内容类型：

```vue
<template>
  <div class="multimodal-chat-container">
    <!-- 头部 -->
    <header class="mm-header">
      <div class="brand">
        <span class="icon">🖼️</span>
        <h1>多模态知识助手</h1>
      </div>
      <div class="header-actions">
        <button class="mode-toggle" :class="{ active: imageMode }"
                @click="imageMode = !imageMode">
          {{ imageMode ? '📷 图片模式' : '💬 纯文本模式' }}
        </button>
      </div>
    </header>

    <!-- 主聊天区域 -->
    <main class="chat-main" ref="chatContainer">
      <!-- 欢迎界面 -->
      <div v-if="messages.length === 0" class="welcome-screen">
        <div class="welcome-icon">🔍🖼️</div>
        <h2>支持图文混合问答</h2>
        <p>输入问题或上传图片，我会从知识库中检索相关的文档、图片、表格来回答你</p>

        <div class="feature-cards">
          <div class="feature-card">
            <div class="card-icon">📄</div>
            <h4>文档理解</h4>
            <p>PDF/Word/Markdown 全格式支持</p>
          </div>
          <div class="feature-card">
            <div class="card-icon">🖼️</div>
            <h4>图片识别</h4>
            <p>截图/照片/图表/图纸智能分析</p>
          </div>
          <div class="feature-card">
            <div class="card-icon">📊</div>
            <h4>表格提取</h4>
            <p>参数表/对比表/配置项结构化</p>
          </div>
          <div class="feature-card">
            <div class="card-icon">📈</div>
            <h4>图表解读</h4>
            <p>趋势图/柱状图/饼图语义理解</p>
          </div>
        </div>

        <div class="example-queries">
          <p class="examples-label">试试这些：</p>
          <div class="example-chips">
            <button v-for="q in exampleQueries" :key="q.text"
                    class="chip" @click="askExample(q)">
              <span class="chip-icon">{{ q.icon }}</span>{{ q.text }}
            </button>
          </div>
        </div>
      </div>

      <!-- 消息列表 -->
      <div v-for="(msg, idx) in messages" :key="idx"
           :class="['message-row', msg.role]">
        <div class="avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</div>
        <div class="message-body">
          <!-- 用户消息 -->
          <template v-if="msg.role === 'user'">
            <div class="text-content">{{ msg.content }}</div>
            <div v-if="msg.imagePreview" class="user-image-preview">
              <img :src="msg.imagePreview" alt="用户上传的图片" />
            </div>
          </template>

          <!-- AI 回复 -->
          <template v-else>
            <!-- Markdown 文本内容 -->
            <div class="ai-text-content" v-html="renderMarkdown(msg.content)"></div>

            <!-- 图片引用卡片 -->
            <div v-if="msg.imageCitations && msg.imageCitations.length > 0"
                 class="citation-section">
              <div class="section-label">📷 参考图片</div>
              <div v-for="cite in msg.imageCitations" :key="cite.index"
                   class="image-citation-card"
                   @click="openImageModal(cite)">
                <img v-if="cite.thumbnailUrl"
                     :src="cite.thumbnailUrl" class="citation-thumb"
                     loading="lazy" />
                <div class="citation-info">
                  <div class="cite-title">{{ cite.description || `图片 #${cite.index}` }}</div>
                  <div class="cite-meta">
                    <span class="source-badge">{{ extractFileName(cite.source) }}</span>
                    <span v-if="cite.page">第{{ cite.page }}页</span>
                  </div>
                </div>
                <div class="cite-action">🔍 查看大图</div>
              </div>
            </div>

            <!-- 表格引用 -->
            <div v-if="msg.tableReferences && msg.tableReferences.length > 0"
                 class="citation-section">
              <div class="section-label">📊 参考表格</div>
              <div v-for="(table, tIdx) in msg.tableReferences" :key="tIdx"
                   class="table-ref-card">
                <div class="table-header">
                  <span>来源: {{ extractFileName(table.source) }}</span>
                  <button class="expand-btn" @click="toggleTableExpand(tIdx)">
                    {{ expandedTables[tIdx] ? '收起' : '展开' }}
                  </button>
                </div>
                <div v-if="expandedTables[tIdx]" class="table-content"
                     v-html="renderMarkdown(table.content)"></div>
                <div v-else class="table-preview">
                  {{ table.content.substring(0, 150) }}...
                </div>
              </div>
            </div>

            <!-- 操作按钮 -->
            <div class="message-actions">
              <button @click="copyMessage(msg)" class="action-btn">📋 复制</button>
              <button @click="likeMessage(msg)" class="action-btn">👍</button>
              <button @click="dislikeMessage(msg)" class="action-btn">👎</button>
              <button v-if="hasImages(msg)" @click="regenerateWithoutImages(msg)"
                      class="action-btn secondary">🔄 仅文本重答</button>
            </div>

            <!-- 元信息 -->
            <div class="message-meta">
              <span class="source-type" :class="msg.sourceType">
                {{ sourceTypeLabel(msg.sourceType) }}
              </span>
              <span class="latency">⏱️ {{ msg.latency?.toFixed(1) }}s</span>
              <span class="confidence">置信度 {{ (msg.confidence * 100).toFixed(0) }}%</span>
            </div>
          </template>
        </div>
      </div>

      <!-- 加载状态 -->
      <div v-if="isLoading" class="typing-indicator">
        <div class="dots"><span></span><span></span><span></span></div>
        <span class="loading-text">
          {{ currentPhase === 'retrieving' ? '正在检索知识库...' :
             currentPhase === 'analyzing' ? '正在分析图片...' :
             '正在生成回答...' }}
        </span>
      </div>
    </main>

    <!-- 输入区域 -->
    <div class="input-area-multimodal">
      <!-- 图片上传预览区 -->
      <div v-if="selectedImage" class="image-preview-bar">
        <div class="preview-item">
          <img :src="selectedImagePreview" />
          <button class="remove-btn" @click="removeSelectedImage()">✕</button>
        </div>
      </div>

      <div class="input-row">
        <!-- 图片上传按钮 -->
        <label class="upload-btn" title="上传图片">
          <input type="file" ref="fileInput"
                 accept="image/*"
                 @change="handleFileSelect"
                 style="display: none" />
          📎
        </label>

        <!-- 或拖拽区域提示 -->
        <div class="drop-hint" v-if="!messageText"
             @dragover.prevent="isDragging = true"
             @dragleave="isDragging = false"
             @drop.prevent="handleDrop"
             :class="{ active: isDragging }">
          拖拽图片到此处，或输入问题...
        </div>

        <!-- 文本输入 -->
        <textarea v-model="messageText"
                  placeholder="输入您的问题，可附带图片..."
                  rows="1"
                  @keydown.enter.exact="handleSend"
                  @input="autoResize"
                  :disabled="isLoading"
                  ref="textInput"></textarea>

        <button class="send-btn"
                :disabled="!canSend"
                @click="handleSend">
          发送
        </button>
      </div>
    </div>

    <!-- 图片查看弹窗 -->
    <Teleport to="body">
      <div v-if="modalImage" class="image-modal-overlay" @click="closeImageModal">
        <div class="image-modal-content" @click.stop>
          <div class="modal-header">
            <span>{{ modalImage.description || '知识库图片' }}</span>
            <button @click="closeImageModal">✕</button>
          </div>
          <img :src="modalImageUrl" class="modal-image" />
          <div class="modal-footer">
            <span>来源: {{ modalImage.source }}</span>
            <span v-if="modalImage.page">第{{ modalImage.page }}页</span>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, computed, nextTick } from 'vue'
import { marked } from 'axios'
import DOMPurify from 'dompurify'

const messages = ref([])
const messageText = ref('')
const selectedImage = ref(null)
const selectedImagePreview = ref(null)
const isLoading = ref(false)
const isDragging = ref(false)
const imageMode = ref(true)
const currentPhase = ref('idle')
const expandedTables = ref({})
const modalImage = ref(null)

const exampleQueries = [
  { text: '系统的整体架构是怎样的？', icon: '🏗️' },
  { text: '这个报错截图是什么原因？', icon: '🖼️' },
  { text: 'API 参数对照表在哪里？', icon: '📊' },
  { text: '销售趋势图显示了什么？', icon: '📈' },
]

function renderMarkdown(text) {
  if (!text) return ''
  const rawHtml = marked(text)
  return DOMPurify.sanitize(rawHtml)
}

function extractFileName(path) {
  if (!path) return '未知'
  return path.split('/').pop().split('\\').pop()
}

function sourceTypeLabel(type) {
  return {
    text_only: '📝 纯文本',
    image_only: '🖼️ 纯图片',
    mixed: '🔀 图文混合',
  }[type] || '未知'
}

function hasImages(msg) {
  return msg.imageCitations?.length > 0
}

async function handleFileSelect(e) {
  const file = e.target.files[0]
  if (!file) return
  if (!file.type.startsWith('image/')) {
    alert('请选择图片文件')
    return
  }
  if (file.size > 20 * 1024 * 1024) {
    alert('图片大小不能超过 20MB')
    return
  }
  selectedImage.value = file
  selectedImagePreview.value = URL.createObjectURL(file)
}

function handleDrop(e) {
  isDragging.value = false
  const file = e.dataTransfer.files[0]
  if (file?.type.startsWith('image/')) {
    selectedImage.value = file
    selectedImagePreview.value = URL.createObjectURL(file)
  }
}

function removeSelectedImage() {
  selectedImage.value = null
  selectedImagePreview.value = null
}

const canSend = computed(() =>
  (messageText.value.trim() || selectedImage.value) && !isLoading.value
)

async function handleSend() {
  if (!canSend.value) return

  const userMsg = {
    role: 'user',
    content: messageText.value,
    imagePreview: selectedImagePreview.value,
  }
  messages.value.push(userMsg)

  const formData = new FormData()
  formData.append('message', messageText.value || '请分析这张图片')
  if (selectedImage.value) {
    formData.append('file', selectedImage.value)
  }

  messageText.value = ''
  removeSelectedImage()
  isLoading.value = true
  currentPhase.value = 'retrieving'
  scrollToBottom()

  try {
    currentPhase.value = 'analyzing'
    const response = await fetch('/api/v1/multimodal/chat', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${getToken()}` },
      body: formData,
    })

    const data = await response.json()
    currentPhase.value = 'generating'

    messages.value.push({
      role: 'assistant',
      content: data.answer,
      imageCitations: data.image_citations || [],
      tableReferences: data.table_references || [],
      confidence: data.confidence,
      latency: data.latency_ms / 1000,
      sourceType: data.source_type,
    })
  } catch (err) {
    messages.value.push({
      role: 'assistant',
      content: `❌ 出错了: ${err.message}`,
    })
  } finally {
    isLoading.value = false
    currentPhase.value = 'idle'
    scrollToBottom()
  }
}

function openImageModal(citation) {
  modalImage.value = citation
  modalImageUrl.value = `/api/images/${encodeURIComponent(citation.image_path)}`
}

function closeImageModal() {
  modalImage.value = null
}

function toggleTableExpand(idx) {
  expandedTables.value[idx] = !expandedTiles.value[idx]
}

function askExample(q) {
  messageText.value = q.text
}
</script>
```

这个前端组件虽然代码量较大，但它的结构是清晰的，核心的交互流程如下：

**图片上传的三种方式**：（1）点击 📎 按钮打开文件选择器；（2）直接拖拽图片到输入框区域；（3）从剪贴板粘贴（可以通过隐藏的 paste 事件监听实现）。三种方式最终都把选中的图片文件存入 `selectedImage`，并在输入框上方显示预览。

**分阶段的加载指示**：`currentPhase` 变量追踪当前的处理阶段——`retrieving`（正在搜索知识库）、`analyzing`（正在分析图片）、`generating`（正在生成回答）。每个阶段显示不同的提示文案，让用户知道系统在做什么而不是干等。

**图片引用卡片的交互**：每张被引用的知识库图片都以缩略图形式展示在回答下方，点击可以弹出全屏查看器（`image-modal-overlay`）。缩略图通过 `/api/images/` 端点从服务器安全地获取——这个端点需要在后端实现图片文件的静态服务功能，并加上认证检查防止未授权访问。

**来源类型的视觉标识**：回答底部的元信息栏用不同颜色和图标标注了本次回答的信息来源类型——纯文本（绿色📝）、纯图片（蓝色🖼️）、图文混合（紫色🔀）。这帮助用户直观地了解回答的可信度基础——图文混合的回答通常意味着系统找到了更全面的证据。

## Docker Compose 部署（多模态版）

```yaml
# docker-compose.multimodal.yml — 在第9章的基础上扩展

# 新增/修改的服务:

# 1. App 服务增加环境变量
  app:
    # ... 其他配置同第9章 ...
    environment:
      - MULTIMODAL_ENABLED=true
      - VISION_MODEL=gpt-4o
      - CLIP_MODEL=ViT-B/32
      - IMAGE_STORAGE_PATH=/app/data/images
      - MAX_UPLOAD_SIZE_MB=20
      - MAX_IMAGES_PER_QUERY=5
    volumes:
      # ... 已有卷挂载 ...
      - ./data/images:/app/data/images     # 图片存储目录
    deploy:
      resources:
        limits:
          memory: 6G  # 多模态需要更多内存（图片处理 + GPT-4V上下文）
        reservations:
          memory: 2G

# 2. MinIO 对象存储（新增）
  minio:
    image: minio/minio:latest
    container_name: kb-minio
    restart: unless-stopped
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"   # API
      - "9001:9001"   # Console (管理界面)
    networks:
      - kb-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s

# 3. MinIO 初始化桶（新增一次性任务）
  minio-init:
    image: minio/mc:latest
    container_name: kb-minio-init
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set myminio http://minio:9000 minioadmin ${MINIO_PASSWORD};
      mc mb myminio/kb-images;
      mc anonymous set download myminio/kb-images;
      echo 'MinIO initialized';
      "

volumes:
  # ... 已有 volumes ...
  minio_data:
```

相比第9章的部署配置，多模态版本的主要变化有：

**内存需求提升**：从 4GB 提升到 6GB 限制。原因是 GPT-4V 的上下文窗口中包含 base64 编码的图片数据时，单次请求的内存消耗显著增加——一张高清图片编码后可能有几 MB 的字符串。同时处理 5 张图片（`MAX_IMAGES_PER_QUERY=5`）的情况下，峰值内存可能达到 3-4GB。

**MinIO 对象存储**：生产环境中不应该把图片文件存在应用容器本地或直接挂在载的目录里——容器重启就会丢失，而且无法在多个实例间共享。MinIO 提供了 S3 兼容的对象存储服务，图片上传后通过 HTTP API 安全地访问。`kb-images` bucket 被设置为公开读取权限（因为经过应用层的鉴权后才能拿到访问 URL），这样前端可以直接通过 `<img src>` 标签展示图片。

## 性能优化专项

多模态 RAG 系统的性能优化有几个独特的关注点：

### 优化一：图片预处理管道

```python
class ImagePreprocessor:
    """图片预处理——在上传和分析前优化"""

    def __init__(self, max_dimension: int = 2048, target_size_kb: int = 500):
        self.max_dim = max_dimension
        self.target_size_kb = target_size_kb

    def preprocess_for_storage(self, image_bytes: bytes) -> tuple[bytes, dict]:
        """
        为存储优化的预处理：
        1. 尺寸调整（保持比例，长边不超过 max_dimension）
        2. 格式转换（统一转为 JPEG 以减小体积）
        3. 质量自适应压缩
        """
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))

        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        original_size = len(image_bytes)

        # 尺寸调整
        ratio = min(self.max_dim / max(img.size), 1.0)
        if ratio < 1.0:
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # 自适应质量压缩
        output = io.BytesIO()
        quality = 92
        while quality >= 60:
            output.seek(0)
            output.truncate(0)
            img.save(output, format='JPEG', quality=quality, optimize=True)
            if output.tell() <= self.target_size_kb * 1024:
                break
            quality -= 8

        processed_bytes = output.getvalue()
        metadata = {
            "original_size_bytes": original_size,
            "processed_size_bytes": len(processed_bytes),
            "compression_ratio": round(original_size / max(len(processed_bytes), 1), 2),
            "original_dimensions": f"{img.width}x{img.height}",
            "format": "JPEG",
            "quality_used": quality,
        }

        return processed_bytes, metadata
```

这个预处理器的目标是将任意大小的输入图片统一转换为适合存储和传输的格式。一张 4000×3000 的 PNG 截图（可能 8MB）会被压缩为 2048px 边长的 JPEG（通常 200-500KB），压缩比超过 16 倍。对于 GPT-4V 来说，2048px 已经足够看清所有细节（`detail: "high"` 模式下会自动裁取细节窗口），更大的图片只会浪费 token 和带宽。

### 优化二：智能缓存策略

```python
class MultiModalCacheManager:
    """多模态场景下的分层缓存"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.prefix = {
            "image_desc": "mm:image:desc:",
            "clip_embed": "mm:clip:embed:",
            "query_result": "mm:query:result:",
            "thumbnail": "mm:thumb:",
        }

    async def get_or_compute_image_description(
        self, image_hash: str, compute_fn, ttl_hours: int = 168
    ):
        """图片描述缓存（7天——图片描述不会频繁变化）"""
        key = self.prefix["image_desc"] + image_hash
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)

        result = await compute_fn()
        await self.redis.setex(key, ttl_hours * 3600, json.dumps(result))
        return result

    async def get_or_compute_clip_embedding(
        self, image_hash: str, compute_fn, ttl_days: int = 30
    ):
        """CLIP embedding 缓存（30天——embedding 很少变化）"""
        key = self.prefix["clip_embed"] + image_hash
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)

        result = await compute_fn()
        await self.redis.setex(key, ttl_days * 86400, json.dumps(result.tolist()))
        return result
```

注意不同类型数据的 TTL 差异化策略：图片描述缓存 7 天（因为同一张图片的语义描述基本不变），CLIP embedding 缓存 30 天（只要图片不变 embedding 就不变），但查询结果缓存时间最短（因为知识库内容可能随时更新）。这种差异化策略在保证新鲜度的同时最大化缓存命中率。

### 优化三：成本控制面板

```python
class CostTracker:
    """多模态 RAG 成本追踪"""

    PRICING = {
        "gpt-4o-input": 0.0025 / 1000,    # per 1K tokens (text)
        "gpt-4o-image-low": 0.01,         # per image (low res)
        "gpt-4o-image-high": 0.03,        # per image (high res)
        "gpt-4o-output": 0.01 / 1000,     # per 1K tokens
        "text-embedding-3-large": 0.00013 / 1000,
        "clip-vit-b32": 0.0,               # 本地运行，无API成本
    }

    def __init__(self):
        self.daily_stats = {"queries": 0, "images_processed": 0, "total_cost_usd": 0.0}

    def track_query(self, num_images: int, input_tokens: int, output_tokens: int):
        cost = (
            input_tokens * self.PRICING["gpt-4o-input"]
            + num_images * self.PRICING["gpt-4o-image-high"]
            + output_tokens * self.PRICING["gpt-4o-output"]
        )
        self.daily_stats["queries"] += 1
        self.daily_stats["images_processed"] += num_images
        self.daily_stats["total_cost_usd"] += cost

    def get_daily_report(self) -> dict:
        return {
            **self.daily_stats,
            "avg_cost_per_query": round(
                self.daily_stats["total_cost_usd"] / max(self.daily_stats["queries"], 1), 4
            ),
            "monthly_projection": round(self.daily_stats["total_cost_usd"] * 30, 2),
        }
```

在生产环境中建议每天检查一次成本报告。如果一个多模态 RAG 系统日均 500 次查询、平均每次涉及 2 张图片，月度成本大约在 1500-3000 美元之间（取决于图片大小和模型选择）。通过启用图片预处理压缩（减少 token 消耗）、提高缓存命中率、以及对简单查询跳过多模态处理，可以将成本降低 40-60%。

## 总结

