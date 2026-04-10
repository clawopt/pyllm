# 04-5 从零创建自定义 GGUF 模型

## GGUF 格式：Ollama 的底层语言

到目前为止，我们使用的所有模型——无论是 `ollama pull qwen2.5:7b` 下载的，还是通过 Modelfile + ADAPTER 组装的——它们的底层存储格式都是 **GGUF**。理解 GGUF 是掌握 Ollama 高级用法的关键一步。

GGUF（GPT-Generated Unified Format）是 llama.cpp 项目定义的一种模型文件格式，它被设计来解决一个核心问题：**如何让大模型在各种硬件平台上高效加载和运行**。

```
┌─────────────────────────────────────────────────────────────┐
│                为什么选 GGUF 而不是 PyTorch？                 │
│                                                             │
│  PyTorch (.pt/.safetensors):                                │
│  ├── 需要完整的 Python/PyTorch 运行时                       │
│  ├── 加载慢：需要反序列化整个张量图                          │
│  ├── 内存占用高：每个权重都是 fp32/fp16 的完整张量           │
│  ├── 平台依赖强：需要匹配的 CUDA/cuDNN 版本                  │
│  └── 文件体积大：fp16 的 7B 模型 ≈ 15GB                     │
│                                                             │
│  GGUF (.gguf):                                              │
│  ├── 纯 C/C++ 加载，无需 Python                             │
│  ├── 内存映射（mmap）：秒级加载，按需读取                   │
│  ├── 支持原生量化：INT4/INT8 权重直接存储                    │
│  ├── 跨平台统一：同一文件在 Mac/Linux/Windows/RaspberryPi   │
│  └── 量化后极小：q4_K_M 的 7B 模型 ≈ 4.3GB                 │
│                                                             │
│  Ollama = Go 封装 + llama.cpp 推理引擎 + GGUF 模型格式       │
└─────────────────────────────────────────────────────────────┘
```

## 完整转换流程

将一个 HuggingFace 格式的模型转换为 Ollama 可用的 GGUF 格式，需要经过以下步骤：

```
HuggingFace Hub / 本地 safetensors
        │
        ▼
   [步骤1] 准备模型文件
   (config.json + tokenizer.json + *.safetensors)
        │
        ▼
   [步骤2] 安装 llama.cpp 工具链
   (convert_hf_to_gguf.py + llama-quantize)
        │
        ▼
   [步骤3] 执行格式转换
   convert_hf_to_gguf.py → model-f16.gguf
        │
        ▼
   [步骤4] 选择量化级别并执行量化
   llama-quantize model-f16.gguf model-q4_K_M.gguf Q4_K_M
        │
        ▼
   [步骤5] 编写 Modelfile 并导入 Ollama
   ollama create my-model -f Modelfile
```

### 步骤一：准备模型文件

你需要从 HuggingFace 下载模型的三个核心组件：

```bash
#!/bin/bash
# 准备 HuggingFace 模型文件

MODEL_DIR="./hf_model"
mkdir -p $MODEL_DIR

# 方法一：使用 huggingface-cli 下载（推荐）
pip install huggingface_hub

# 下载模型（以一个假设的自定义中文模型为例）
huggingface-cli download your-org/your-chinese-model \
    --local-dir $MODEL_DIR \
    --include "config.json" \
    --include "tokenizer.json" \
    --include "tokenizer_config.json" \
    --include "*.safetensors" \
    --include "special_tokens_map.json"

# 方法二：手动从网页下载
# 访问 https://huggingface.co/your-org/your-chinese-model
# 下载 Files and versions 标签页中的上述文件到 $MODEL_DIR

echo "✅ 模型文件准备完成"
ls -lh $MODEL_DIR/
```

验证文件完整性：

```python
#!/usr/bin/env python3
"""验证 HuggingFace 模型文件的完整性"""

import json
import os
from pathlib import Path

def validate_hf_model(model_dir):
    """检查模型目录是否包含所有必需文件"""
    
    required_files = {
        "config.json": "模型架构配置",
        "tokenizer.json": "分词器词汇表",
        "tokenizer_config.json": "分词器配置",
        "special_tokens_map.json": "特殊 token 映射",
    }
    
    model_path = Path(model_dir)
    
    print(f"\n📂 验证模型目录: {model_path}")
    print(f"{'='*55}\n")
    
    all_ok = True
    
    # 检查必需的配置文件
    for filename, desc in required_files.items():
        filepath = model_path / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✅ {filename:35s} ({size:,} bytes) — {desc}")
            
            # 对 config.json 做额外验证
            if filename == "config.json":
                with open(filepath) as f:
                    config = json.load(f)
                print(f"     模型类型: {config.get('model_type', '?')}")
                print(f"     隐藏层大小: {config.get('hidden_size', '?')}")
                print(f"     层数: {config.get('num_hidden_layers', '?')}")
                print(f"     词表大小: {config.get('vocab_size', '?')}")
                print(f"     头数: {config.get('num_attention_heads', '?')}")
        else:
            print(f"  ❌ {filename:35s} 缺失! — {desc}")
            all_ok = False
    
    # 检查权重文件
    safetensors_files = list(model_path.glob("*.safetensors"))
    if safetensors_files:
        total_size = sum(f.stat().st_size for f in safetensors_files)
        print(f"\n  ✅ 权重文件: {len(safetensors_files)} 个 safetensors 文件")
        for f in sorted(safetensors_files):
            size_gb = f.stat().st_size / (1024**3)
            print(f"     📦 {f.name:40s} ({size_gb:.2f} GB)")
        print(f"     总计: {total_size / (1024**3):.2f} GB")
    else:
        print(f"\n  ❌ 未找到任何 .safetensors 权重文件!")
        all_ok = False
    
    print()
    if all_ok:
        print("🎉 所有必需文件齐全，可以开始转换！")
    else:
        print("⚠️  缺少必要文件，请先补全再继续。")
    
    return all_ok

if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "./hf_model"
    validate_hf_model(model_dir)
```

### 步骤二：安装 llama.cpp 工具链

```bash
# 克隆 llama.cpp 仓库
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 编译转换工具（需要 CMake）
mkdir build && cd build
cmake ..
make -j$(nproc)

# 或者用 make（如果不需要 CUDA 支持）
cd ..
make

# 验证编译成功
./llama-quantize --help
./convert_hf_to_gguf.py --help
```

如果你需要 GPU 加速的量化：

```bash
# 带 CUDA 支持编译
mkdir build-cuda && cd build-cuda
cmake .. -DLLAMA_CUDA=on
make -j$(nproc)

# Apple Silicon 用户
cmake .. -DLLAMA_METAL=on
make -j$(sysctl -n hw.ncpu)
```

### 步骤三：执行格式转换

```bash
# 从 HuggingFace 格式转换为 GGUF (FP16 中间格式)
python convert_hf_to_gguf.py ../hf_model/ \
    --outfile ../output/model-f16.gguf \
    --outtype f16

# 转换参数说明：
# --outfile: 输出文件路径
# --outtype: 输出精度 (f16/f32/bf16)
# --vocab-type: 特殊词汇表处理 (spm/bpe/sentencepiece/hfft/piece)
# --dry-run: 只做验证不实际转换（推荐先跑这个！）

# 先做 dry-run 验证
python convert_hf_to_gguf.py ../hf_model/ --dry-run
```

Dry-run 的输出会告诉你模型的基本信息和潜在问题：

```
{{{
    "model_name": "my_chinese_model",
    "model_arch": "qwen2",
    "model_type": "7B",
    "layer_count": 28,
    "head_count": 32,
    "head_count_kv": 8,
    "embed_length": 4096,
    "block_count": 28,
    "context_length": 32768,
    "vocab_size": 152064,
    ...
}}}
✅ 模型结构验证通过，可以安全转换。
```

### 步骤四：选择量化级别并执行量化

这是最关键的一步——选择合适的量化级别决定了最终模型的大小和质量权衡：

```bash
# FP16 → Q4_K_M (最常用的量化级别)
./llama-quantize ../output/model-f16.gguf ../output/model-q4_K_M.gguf Q4_K_M

# 其他常用量化目标
./llama-quantize model-f16.gguf model-q8_0.gguf Q8_0      # 8-bit, 高质量
./llama-quantize model-f16.gguf model-q5_K_M.gguf Q5_K_M    # 5-bit, 平衡
./llama-quantize model-f16.gguf model-q4_K_S.gguf Q4_K_S    # 4-bit S版, 更小
./llama-quantize model-f16.gguf model-q3_K_M.gguf Q3_K_M    # 3-bit, 极限压缩
```

支持的量化目标完整列表：

```python
#!/usr/bin/env python3
"""量化级别选择指南"""

QUANTIZATION_LEVELS = [
    # (名称, 描述, 大小倍率, 质量损失, 推荐场景)
    ("F16", "半精度浮点", "1.0x", "无%", "基准对比"),
    ("BF16", "BFloat16", "~1.0x", "<1%", "Apple Silicon"),
    ("Q8_0", "8-bit legacy", "~2.0x", "1-2%", "高质量需求"),
    ("Q6_K", "6-bit K-quant", "~2.5x", "1-3%", "质量敏感场景"),
    ("Q5_K_M", "5-bit K-Medium", "~3.0x", "2-4%", "日常使用"),
    ("Q5_K_S", "5-bit K-Small", "~3.0x", "3-5%", "内存紧张"),
    ("Q4_K_M", "4-bit K-Medium", "~4.0x", "3-5%", "★★★ 推荐"),
    ("Q4_K_S", "4-bit K-Small", "~4.0x", "4-6%", "极限资源"),
    ("Q4_0", "4-bit legacy", "~4.0x", "4-6%", "兼容性"),
    ("Q3_K_M", "3-bit K-Medium", "~5.0x", "5-10%", "实验性"),
    ("Q3_K_S", "3-bit K-Small", "~5.0x", "8-12%", "不推荐"),
    ("Q3_K_L", "3-bit K-Large", "~4.5x", "4-8%", "折中选择"),
    ("Q2_K", "2-bit", "~6.0x", "显著", "仅特殊用途"),
]

def recommend_quantization(available_ram_gb, quality_priority="balanced"):
    """根据可用内存和质量优先级推荐量化级别"""
    
    print("\n📊 量化级别参考表\n")
    print(f"{'级别':<10s} {'描述':<20s} {'压缩比':>7s} {'质量':>7s} {'推荐场景'}")
    print(f"{'─'*10} {'─'*20} {'─'*7} {'─'*7} {'─'*30}")
    
    for name, desc, ratio, loss, scenario in QUANTIZATION_LEVELS:
        marker = " ★★★" if name == "Q4_K_M" else ""
        print(f"{name:<10s} {desc:<20s} {ratio:>7s} {loss:>7s} {scenario}{marker}")
    
    # 根据资源给出具体建议
    print(f"\n💡 基于 {available_ram_gb}GB 可用内存的建议:")
    
    if available_ram_gb >= 48:
        rec = "Q5_K_M 或 F16（追求极致质量）"
    elif available_ram_gb >= 24:
        rec = "Q4_K_M 或 Q5_K_M（最佳平衡）"
    elif available_ram_gb >= 12:
        rec = "Q4_K_M（推荐默认值）"
    elif available_ram_gb >= 6:
        rec = "Q4_K_S 或 Q3_K_L（需要精打细算）"
    else:
        rec = "Q3_K_S 或更小模型（考虑换用更小的基础模型）"
    
    print(f"   推荐: {rec}")

if __name__ == "__main__":
    import sys
    ram = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
    recommend_quantization(ram)
```

### 步骤五：编写 Modelfile 并导入 Ollama

```dockerfile
# Modelfile: my-custom-model
FROM ./model-q4_K_M.gguf

SYSTEM """你是基于自定义训练的中文语言模型。
你擅长[描述你的模型特长]。"""

PARAMETER temperature 0.7
PARAMETER num_ctx 8192

TEMPLATE """{{- if .System }}<<SYS>>
{{ .System }}
<</SYS>>

{{ end }}{{- range .Messages }}
{{- if eq .Role "user" }}[INST] {{ .Content }} [/INST]
{{- else if eq .Role "assistant" }} {{ .Content }}
{{ end }}
{{ end }}"""
```

```bash
# 导入 Ollama
ollama create my-custom-model -f Modelfile

# 测试
ollama run my-custom-model "你好，请介绍一下你自己"

# 验证模型信息
ollama show my-custom-model
```

## 常见转换问题与解决方案

### 问题一：Tokenizer 不兼容

```
Error: vocab type not recognized
Error: cannot find tokenizer
```

**原因**：你的模型使用了 llama.cpp 不直接支持的 tokenizer 类型。

**解决方案**：

```bash
# 方法一：指定 vocab 类型
python convert_hf_to_gguf.py ./hf_model/ \
    --outfile model.gguf \
    --vocab-type piece     # 或 spm / bpe / hfft

# 方法二：对于使用 sentencepiece 的模型
python convert_hf_to_gguf.py ./hf_model/ \
    --outfile model.gguf \
    --vocab-type spm
```

### 问题二：Vocab Size 不匹配

```
Error: vocab mismatch: expected 32000 got 152064
```

**原因**：模型配置中的词表大小和实际的 tokenizer 文件不一致。

**解决方案**：检查 `config.json` 和 `tokenizer.json` 是否来自同一个版本。确保下载的是同一 commit 的文件。

### 问题三：特殊 Token 缺失

```
Warning: special tokens not found in tokenizer
```

**影响**：可能导致模型输出格式异常、无法正确识别指令格式。

**解决方案**：

```python
#!/usr/bin/env python3
"""修复缺失的特殊 Token"""

import json

def fix_special_tokens(tokenizer_config_path):
    """在 tokenizer_config.json 中补充特殊 token 定义"""
    
    with open(tokenizer_config_path, "r") as f:
        config = json.load(f)
    
    # 常见的需要声明的特殊 token
    common_special_tokens = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>", 
        "unk_token": "<|end_of_text|>",
        "pad_token": "<|end_of_text|>",
        
        # ChatML 格式 (Qwen/Qwen2)
        "chat_template": "{% for item in messages %}{% if item['role'] == 'system' %}{{ '<|im_start|>\nsystem\n' + item['content'] + '<|im_end|>\n' }}{% elif item['role'] == 'user' %}{{ '<|im_start|>\nuser\n' + item['content'] + '<|im_end|>\n' }}{% elif item['role'] == 'assistant' %}{{ '<|im_start|>\nassistant\n' + item['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>\nassistant\n' }}{% endif %}",
        
        # Llama 格式
        # "chat_template": "{% for item in messages %}{% if item['role'] == 'system' %}{{ '[INST] <<SYS>>\n' + item['content'] + '\n<</SYS>> [/INST]\n' }}{% elif item['role'] == 'user' %}{{ '[INST] ' + item['content'] + ' [/INST]' }}{% elif item['role'] == 'assistant' %}{{ item['content'] }}{% endfor %}",
    }
    
    updated = False
    for key, value in common_special_tokens.items():
        if key not in config or not config[key]:
            config[key] = value
            updated = True
            print(f"  ✅ 添加: {key} = {value[:50]}...")
    
    if updated:
        with open(tokenizer_config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 已修复 {tokenizer_config_path}")
    else:
        print(f"\nℹ️  无需修复，所有字段已存在")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./hf_model/tokenizer_config.json"
    fix_special_tokens(path)
```

### 问题四：转换后的模型输出乱码

**可能原因**：
1. TEMPLATE 设置不正确，导致特殊 token 没有正确处理
2. 量化过度导致低概率 token（包括中文 token）丢失精度

**解决方案**：

```dockerfile
# 确保 TEMPLATE 与原始模型格式一致
# 对于 Qwen 系列，必须使用 ChatML 格式：
TEMPLATE """{{- if .System }}<<SYS>>
{{ .System }}
<</SYS>>

{{ end }}{{- range .Messages }}
{{- if eq .Role "user" }}[INST] {{ .Content }} [/INST]
{{- else if eq .Role "assistant" }} {{ .Content }}
{{ end }}
{{ end }}"""

# 如果乱码仍然存在，尝试更高的量化级别
# Q4_K_M → Q5_K_M → Q8_0 → F16
```

## 一键转换脚本

下面是一个完整的自动化脚本，把上述所有步骤整合在一起：

```bash
#!/bin/bash
# ============================================================
#  hf-to-ollama.sh
#  HuggingFace 模型 → GGUF → Ollama 一键转换脚本
#  用法: ./hf-to-ollama.sh <model_dir> <model_name> <quant>
#  示例: ./hf-to-ollama.sh ./my-model my-model Q4_K_M
# ============================================================

set -e

MODEL_DIR=${1:-"./hf_model"}
MODEL_NAME=${2:-"my-custom-model"}
QUANT=${3:-"Q4_K_M"}
OUTPUT_DIR="./ollama-output"
LLAMACPP_DIR="./llama.cpp"

echo "============================================="
echo "  HuggingFace → Ollama 一键转换工具"
echo "  模型目录: $MODEL_DIR"
echo "  模型名称: $MODEL_NAME"
echo "  量化级别: $QUANT"
echo "============================================="

mkdir -p "$OUTPUT_DIR"

# Step 1: 验证模型文件
echo ""
echo "[1/5] 验证模型文件..."
required_files=("config.json" "tokenizer.json" "tokenizer_config.json" "special_tokens_map.json")
weight_files=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l)

for f in "${required_files[@]}"; do
    if [ ! -f "$MODEL_DIR/$f" ]; then
        echo "❌ 缺少必需文件: $f"
        exit 1
    fi
done

if [ "$weight_files" -eq 0 ]; then
    echo "❌ 未找到 safetensors 权重文件"
    exit 1
fi

echo "  ✅ 配置文件: ${#required_files[@]} 个"
echo "  ✅ 权重文件: $weight_files 个"

# Step 2: Dry-run 验证
echo ""
echo "[2/5] Dry-run 验证..."
python "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR/" --dry-run
echo "  ✅ Dry-run 通过"

# Step 3: 转换为 F16 GGUF
echo ""
echo "[3/5] 转换为 F16 格式..."
F16_OUTPUT="$OUTPUT_DIR/${MODEL_NAME}-f16.gguf"
python "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR/" \
    --outfile "$F16_OUTPUT" \
    --outtype f16

F16_SIZE=$(du -h "$F16_OUTPUT" | cut -f1)
echo "  ✅ F16 文件: $F16_OUTPUT ($F16_SIZE)"

# Step 4: 量化
echo ""
echo "[4/5] 量化为 $QUANT..."
QUANT_OUTPUT="$OUTPUT_DIR/${MODEL_NAME}-${QUANT,,}.gguf"
"$LLAMACPP_DIR/llama-quantize" "$F16_OUTPUT" "$QUANT_OUTPUT" "$QUANT"

QUANT_SIZE=$(du -h "$QUANT_OUTPUT" | cut -f1)
echo "  ✅ 量化文件: $QUANT_OUTPUT ($QUANT_SIZE)"

# Step 5: 创建 Modelfile 并导入 Ollama
echo ""
echo "[5/5] 导入 Ollama..."

cat > "$OUTPUT_DIR/Modelfile" << EOF
FROM ./${MODEL_NAME}-${QUANT,,}.gguf

SYSTEM """你是一个有帮助的助手。"""

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
EOF

cd "$OUTPUT_DIR"
ollama create "$MODEL_NAME" -f Modelfile

echo ""
echo "============================================="
echo "  ✅ 转换完成！"
echo ""
echo "  使用方法:"
echo "    ollama run $MODEL_NAME"
echo ""
echo "  模型信息:"
ollama show "$MODEL_NAME"
echo "============================================="
```

## 本章小结

这一节我们完成了从 HuggingFace 到 Ollama 的完整转换链路：

1. **GGUF 是 Ollama 的原生格式**，相比 PyTorch 具有跨平台、快速加载、支持量化的优势
2. **五步转换流程**：准备文件 → 安装工具链 → 格式转换(F16) → 量化(Q4_K_M) → 导入 Ollama
3. **量化级别选择**是关键决策点——Q4_K_M 在绝大多数场景下是最优解
4. **常见问题**包括 tokenizer 不兼容、vocab 不匹配、特殊 token 缺失、输出乱码等
5. **一键转换脚本**将整个流程自动化，适合批量转换或团队共享

至此，第四章"Modelfile 与自定义模型"全部完成。下一章我们将进入多模态领域，探索 Ollama 如何理解和生成图像内容。
