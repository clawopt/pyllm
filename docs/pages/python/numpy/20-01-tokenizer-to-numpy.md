# HuggingFace Tokenizer 转 NumPy

HuggingFace Transformers 是目前最流行的 LLM 工具库，而 Tokenizer 是处理文本的核心组件。本篇文章介绍如何将 HuggingFace Tokenizer 的输出转换为 NumPy 数组，以便在纯 NumPy 环境中进行处理或与自定义训练循环集成。

## HuggingFace Tokenizer 基本概念

HuggingFace Tokenizer 将原始文本转换为模型可以处理的 token IDs。一个典型的 tokenizer 输出包含：

```python
# tokenizer 输出示例（伪代码）
{
    'input_ids': [101, 2003, 1037, 3235, 102],  # token IDs
    'attention_mask': [1, 1, 1, 1, 1],           # 注意力掩码
    'token_type_ids': [0, 0, 0, 0, 0],           # token 类型 ID
}
```

## 将 Tokenizer 输出转为 NumPy

```python
import numpy as np

def tokenizer_output_to_numpy(tokenizer_output):
    """将 tokenizer 输出转换为 NumPy 数组

    参数:
        tokenizer_output: tokenizer 的输出字典
    返回:
        numpy_output: 包含 NumPy 数组的字典
    """
    numpy_output = {}

    for key, value in tokenizer_output.items():
        if isinstance(value, list):
            numpy_output[key] = np.array(value, dtype=np.int32)

    return numpy_output

# 示例（模拟 tokenizer 输出）
tokenizer_output = {
    'input_ids': [101, 2003, 1037, 3235, 102],
    'attention_mask': [1, 1, 1, 1, 1],
    'token_type_ids': [0, 0, 0, 0, 0]
}

numpy_output = tokenizer_output_to_numpy(tokenizer_output)
print("转换后的 NumPy 数组:")
for key, arr in numpy_output.items():
    print(f"  {key}: {arr}, dtype={arr.dtype}")
```

## 批量 Tokenization 并转为 NumPy

```python
def batch_tokenize_and_convert(tokenizer, texts, max_length=128):
    """批量 tokenize 并转换为 NumPy 数组

    参数:
        tokenizer: HuggingFace tokenizer
        texts: 文本列表
        max_length: 最大长度
    返回:
        batch: NumPy 数组组成的字典
    """
    # Tokenize
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='np'  # 直接返回 NumPy
    )

    return batch

# 示例
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('gpt2')
#
# texts = [
#     "Hello, world!",
#     "This is a test sentence.",
#     "HuggingFace makes NLP easy."
# ]
#
# batch = batch_tokenize_and_convert(tokenizer, texts)
# print(f"input_ids shape: {batch['input_ids'].shape}")
```

## 创建兼容 NumPy 的 TokenDataset

```python
class TokenDataset:
    """兼容 NumPy 的 Token 数据集"""

    def __init__(self, tokenizer, texts, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

        # 预先 tokenize 并存储为 NumPy
        self.input_ids = []
        self.attention_mask = []

        for text in texts:
            tokens = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='np'
            )
            self.input_ids.append(tokens['input_ids'][0])
            self.attention_mask.append(tokens['attention_mask'][0])

        self.input_ids = np.array(self.input_ids, dtype=np.int32)
        self.attention_mask = np.array(self.attention_mask, dtype=np.int32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

# 示例
# dataset = TokenDataset(tokenizer, ["Hello world", "This is a test"])
# sample = dataset[0]
# print(f"Sample input_ids shape: {sample['input_ids'].shape}")
```

## 处理特殊 Token

```python
def handle_special_tokens(tokenizer):
    """处理特殊 token"""
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'unk_token': tokenizer.unk_token,
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token,
        'sep_token': tokenizer.sep_token,
        'cls_token': tokenizer.cls_token,
        'mask_token': tokenizer.mask_token,
    }

    print("特殊 Token 信息:")
    for name, token in special_tokens.items():
        token_id = getattr(tokenizer, name.replace('_token', '_token_id'), None)
        print(f"  {name}: '{token}' (ID: {token_id})")

    return special_tokens

# 示例
# handle_special_tokens(tokenizer)
```

## 完整的数据准备流程

```python
def prepare_training_data(tokenizer, texts, labels=None, max_length=128):
    """准备训练数据

    参数:
        tokenizer: HuggingFace tokenizer
        texts: 文本列表
        labels: 标签列表（可选）
        max_length: 最大长度
    返回:
        data: 包含 NumPy 数组的字典
    """
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='np'
    )

    data = {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }

    if 'token_type_ids' in encoded:
        data['token_type_ids'] = encoded['token_type_ids']

    if labels is not None:
        data['labels'] = np.array(labels, dtype=np.int32)

    return data

# 示例
# texts = ["Hello world", "This is a test"]
# labels = [1, 0]
# data = prepare_training_data(tokenizer, texts, labels)
# print(f"数据键: {data.keys()}")
# print(f"input_ids shape: {data['input_ids'].shape}")
```

## 常见问题与处理

### 处理不均匀序列长度

```python
def pad_sequences_numpy(sequences, pad_token_id=0, max_length=None):
    """填充序列到相同长度

    参数:
        sequences: 序列列表
        pad_token_id: 填充 token 的 ID
        max_length: 指定最大长度，None 则使用最长序列
    返回:
        padded: NumPy 数组
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    batch_size = len(sequences)
    padded = np.full((batch_size, max_length), pad_token_id, dtype=np.int32)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]

    return padded

# 示例
sequences = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9]
]

padded = pad_sequences_numpy(sequences)
print(f"填充后形状: {padded.shape}")
print(f"填充后数据:\n{padded}")
```

### 创建注意力掩码

```python
def create_attention_mask(input_ids, pad_token_id=0):
    """从 input_ids 创建注意力掩码

    参数:
        input_ids: token IDs 的 NumPy 数组
        pad_token_id: padding token 的 ID
    返回:
        attention_mask: 注意力掩码
    """
    return (input_ids != pad_token_id).astype(np.int32)

# 示例
attention_mask = create_attention_mask(padded)
print(f"注意力掩码:\n{attention_mask}")
```

掌握 HuggingFace Tokenizer 与 NumPy 的交互是高效处理 LLM 数据的基础。这些技巧允许你在纯 NumPy 环境中处理 tokenized 数据。
