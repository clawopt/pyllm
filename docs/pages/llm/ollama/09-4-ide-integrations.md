# 09-4 IDE 集成

## 为什么 IDE 集成很重要

作为开发者，你每天有 **60-80% 的时间** 花在 IDE（编辑器）中。如果能在不离开编辑器的情况下直接与 Ollama 对话——问问题、解释代码、生成测试、审查 PR——开发效率会有显著提升。

这就是 IDE 集成的价值：**把 AI 能力嵌入到你最熟悉的工作流中**。

## VS Code + Continue 插件

Continue 是目前 VS Code 生态中最成熟的 AI 助手插件，对 Ollama 有原生支持。

### 安装与配置

```bash
# 方法一: VS Code 扩展商店搜索 "Continue"
# 1. 打开 VS Code
# 2. Ctrl+Shift+X → 搜索 "Continue"
# 3. 点击 Install

# 方法二: 命令行安装
code --install-extension Continue.continue
```

### 配置 Ollama 后端

安装完成后，打开 Continue 设置：

```
Settings (⚙️) → Models (🤖)

Model Provider: Ollama
Base URL: http://localhost:11434/v1
API Key: ollama (随意填，Ollama 不验证)

Models:
  - Chat Model: qwen2.5:7b (或你常用的模型)
  - Embedding Model: nomic-embed-text (可选)
  - Code Model: deepseek-coder:6.7b (可选)
```

### 核心功能使用

#### 功能一：内联对话（Inline Chat）

在代码中选中一段文字，然后：
- `Cmd + L` (macOS) / `Ctrl + L` (Windows/Linux)：在侧边栏打开对话
- 或输入 `/explain` 然后选中代码：直接在下方显示解释

```
代码:
def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    pivot = partition(arr, low, high)
    quicksort(arr, low, pivot - 1)
    quicksort(arr, pivot + 1, high)

/explain (选中后按 Enter)

→ Continue 回答:
这是一个快速排序的实现。它选择最后一个元素作为基准值(pivot)，
通过 partition 函数将数组分为三部分：小于 pivot 的元素、
等于 pivot 的元素和大于 pivot 的元素。然后递归地对左右两部分
继续排序...

时间复杂度: 平均 O(n log n), 最差 O(n²)
空间复杂度: O(log n) (递归栈)
```

#### 功能二：代码补全与生成

```
// 光标位置写注释或新行，然后:
// 按 Enter 触发 Continue 自动补全

function calculateTotal(items) {
    // ← Continue 在这里自动生成剩余代码
    let total = 0;
    for (let item of items) {
        total += item.price * item.quantity;
    }
    return total;
}
```

#### 功能三：自定义命令

在 Continue 的设置中添加自定义斜杠命令：

| 命令 | 触发方式 | 用途 |
|------|---------|------|
| `/explain` | 选中代码后 | 解释选中代码 |
| `/refactor` | 选中代码 | 重构/优化 |
| `/test` | 选中函数 | 生成单元测试 |
| `/fix` | 选中报错代码 | 修复 bug |
| `/review` | 选中 PR diff | 代码审查 |
| `/doc` | 选中函数 | 生成文档 |

#### 功能四：多文件上下文

Continue 可以理解整个项目的结构——当你问"这个项目的入口文件在哪里？"时，它可以搜索你的整个 workspace 并给出准确答案。

## JetBrains IDE + Continue

IntelliJ IDEA / PyCharm / WebStorm / GoLand 等 JetBrains 全家桶系列都支持 Continue 插件：

```
安装: Settings → Plugins → Marketplace → 搜索 "Continue" → Install
配置: 同 VS Code，指向 http://localhost:11434/v1
使用: Tools → Continue (或右侧边栏图标)
```

## Vim / Neovim: ollama.nvim

对于 Vim 党好者来说，这是最优雅的集成方式——不需要离开你熟悉的编辑环境：

```vim
" Plug 'oliverceden/ollama.nvim'

" 配置
let g:ollama_model = 'qwen2.5:7b'
let g:ollama_endpoint = 'http://localhost:11434/api'

" 使用
:Ollama 解释这段代码          "  ← 选中代码后执行
:Ollama 写一个快速排序         "  ← 直接生成代码
:Ollama 为这个函数写测试       "
:Ollama 重构这段代码使其更清晰   "

" 快捷键 (自定义)
nmap <Leader>o  :Ollama           "  # 正常模式对话
nmap <Leader>e  :Ollama<CR>      "  # 选区发送给 Ollama
nmap <Leader>r  :Ollama review<CR> "  # 代码审查
```

## Emacs: ellama-mode

```elisp
;; 在 init.el 中添加
(require 'ellama)

(setq ellama-provider "ollama")
(setq ellama-url "http://localhost:11434")

;; 使用 M-x ellama-chat 开始对话
;; 或 M-x ellama-summarize / M-x ellama-code-review
```

## 本章小结

这一节覆盖了主流 IDE 的 Ollama 集成方案：

1. **VS Code + Continue** 是最推荐的组合**——安装简单、功能全面、社区活跃
2. **核心工作流**：选中代码 → 解释/重构/测试/文档，全部不离开编辑器
3. **自定义命令** (`/explain`, `/refactor`, `/test`, `/fix`) 让操作标准化
4. **JetBrains 全家桶** 通过同一 Continue 插件获得一致体验
5. **Vim/Neovim 用户** 通过 `ollama.nvim` 实现"零上下文切换"的极致效率
6. **Emacs 用户** 通过 `ellama-mode` 获得类似的集成体验

至此，第九章"WebUI 与工具生态"全部完成。最后一章：企业部署与运维。
