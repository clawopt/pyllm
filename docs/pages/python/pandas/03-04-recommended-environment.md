---
title: 推荐开发环境
description: Jupyter Notebook / VS Code / Deepnote 等开发环境的选择与配置，以及 Pandas 开发最佳实践
---
# 推荐开发环境搭建


## 环境选择矩阵

| 环境 | 适合场景 | 优势 | 劣势 |
|------|---------|------|------|
| **Jupyter Notebook** | 探索性分析、教学、原型 | 即时可视化、分步执行、富文本 | 不适合大型项目、版本控制困难 |
| **JupyterLab** | 日常数据分析 | 多文件编辑器、终端集成、扩展丰富 | 启动较慢 |
| **VS Code + Jupyter** | 全栈开发 | Git 集成、调试器、补全强大 | 配置稍复杂 |
| **Deepnote / Colab** | 协作 / GPU 计算 | 云端运行、零配置、可分享 | 数据隐私顾虑、依赖网络 |
| **PyCharm** | 大型数据工程 | 重构能力、数据库工具 | 较重、启动慢 |
| **终端 + IPython** | 脚本化流水线 | 可复现、适合自动化 | 无 GUI |

### 大模型开发的推荐组合

```
日常探索:     VS Code + Jupyter Extension（本地）
协作/分享:    Deepnote 或 GitHub Codespaces
生产流水线:   Python 脚本 + 日志系统
GPU 训练相关:  Jupyter Lab in Docker（挂载 GPU）
```

## VS Code + Jupyter：最强本地组合

### 安装与配置

```bash
code --install-extension ms-toolsai.jupyter

python -m venv .venv
source .venv/bin/activate  # Linux/Mac

pip install "pandas[all]>=3.0" jupyterlab ipywidgets matplotlib seaborn
```

### VS Code Pandas 开发必备设置

`.vscode/settings.json`：

```json
{
  "jupyter.askForKernelRestart": false,
  "jupyter.interactiveWindow.textEditor.limit": 100,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "[python]": {
    "editor.wordBasedSuggestions": "includedDocuments",
    "editor.tabSize": 4,
    "editor.insertSpaces": true
  }
}
```

### 高效的 Notebook 结构模板

```python
"""
%% [markdown]
- 作者：
- 日期：
- 目标：
"""

## 1. 环境初始化

import sys
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"NumPy {np.__version__}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.options.mode.copy_on_write = True
pd.options.dtype_backend = 'pyarrow'

## 2. 数据加载

DATA_DIR = '../data/'
OUTPUT_DIR = '../output/'

df_raw = pd.read_parquet(f'{DATA_DIR}corpus.parquet')
print(f"原始数据: {len(df_raw):,} 行 × {len(df_raw.columns)} 列")

```

## Jupyter Lab 进阶技巧

### 常用魔法命令

```python
%who DataFrame Series

%timeit df.groupby('category').mean()

%prun -s cumulative df.merge(df2, on='id')

from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=True)
show(df.head(100))

from IPython.display import display, HTML
display(HTML("<h3>报告标题</h3>"))
```

### 大数据集的内存友好操作

```python
chunk_iter = pd.read_csv('huge_file.csv', chunksize=100_000)
results = []
for i, chunk in enumerate(chunk_iter):
    processed = process_chunk(chunk)  # 你的处理函数
    results.append(processed)
    if i % 10 == 0:
        print(f"已处理 {(i+1)*100_000:,} 行")
final = pd.concat(results)

df = pd.read_parquet('data.parquet', columns=['prompt', 'response', 'quality'])

dtypes = {
    'user_id': 'string',
    'turn_count': 'int16',
    'quality_score': 'float32',
    'category': 'category',
}
df = pd.read_csv('data.csv', dtype=dtypes)
```

## 生产级脚本开发规范

当数据处理流程稳定后，应该从 Notebook 迁移到 Python 脚本：

### 项目结构

```
llm-data-pipeline/
├── src/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── load.py            # 数据加载模块
│   ├── clean.py           # 清洗模块
│   ├── transform.py       # 转换模块
│   └── export.py          # 导出模块
├── data/                  # 原始数据（gitignore）
├── output/                # 输出结果
├── notebooks/             # 探索性分析
│   └── exploratory.ipynb
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── main.py                # 入口脚本
```

### main.py 示例

```python
"""LLM SFT 数据集构建主入口"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.load import load_all_sources
from src.clean import clean_data
from src.transform import build_sft_format
from src.export import export_datasets


def setup_logging(log_dir: Path, verbose: bool = False):
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pipeline.log'),
            logging.StreamHandler(),
        ]
    )


def run_pipeline(config: PipelineConfig):
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("SFT 数据集构建流水线")
    logger.info(f"配置: {config.model_dump()}")
    logger.info("=" * 60)
    
    raw = load_all_sources(config.data_dir)
    logger.info(f"原始数据: {sum(len(d) for d in raw.values()):,} 条")
    
    cleaned = clean_data(raw, config.cleaning_rules)
    logger.info(f"清洗后: {len(cleaned):,} 条 ({len(cleaned)/sum(len(d) for d in raw.values())*100:.1f}%)")
    
    sft_data = build_sft_format(cleaned, config.format_config)
    logger.info(f"SFT 格式转换完成: {len(sft_data):,} 条")
    
    export_datasets(sft_data, config.output_dir, config.split_config)
    logger.info(f"导出完成 → {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='LLM SFT 数据集构建')
    parser.add_argument('--data-dir', type=Path, default=Path('data/'),
                        help='原始数据目录')
    parser.add_argument('--output-dir', type=Path, default=Path('output/'),
                        help='输出目录')
    parser.add_argument('--target-size', type=int, default=500_000,
                        help='目标训练集大小')
    parser.add_argument('--verbose', action='store_true',
                        help='启用详细日志')
    args = parser.parse_args()
    
    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
    )
    
    setup_logging(args.output_dir / 'logs', args.verbose)
    run_pipeline(config)


if __name__ == '__main__':
    main()
```

### 运行方式

```bash
python main.py --data-dir ./raw_data --output-dir ./train_output --verbose

docker build -t llm-pipeline .
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output llm-pipeline

0 2 * * * cd /path/to/project && python main.py >> cron.log 2>&1
```

## Pandas 开发最佳实践清单

### 代码风格

```python
result = (
    df[df['quality'] >= 3]
    .drop_duplicates(subset=['prompt'])
    .assign(token_count=lambda x: x['prompt'].str.len() // 4)
    .sort_values('token_count', ascending=False)
)

df[df['quality'] >= 3]['new_col'] = value  # SettingWithCopyWarning!

subset = df[df['quality'] >= 3].copy()
subset['processed'] = subset['text'].str.lower()
```

### 性能意识

```python
df['length'] = df['text'].str.len()

df['length'] = df['text'].apply(len)  # 慢 5-20 倍

df['category'] = df['category'].astype('category')

pd.options.dtype_backend = 'pyarrow'
```

### 可复现性

```python
RANDOM_SEED = 42

import numpy as np
import pandas as pd

np.random.seed(RANDOM_SEED)
pd.options.mode.copy_on_write = True
pd.options.dtype_backend = 'pyarrow'

```

## 环境诊断速查表

遇到问题时，按此顺序排查：

```
1. 版本检查
   python -c "import pandas; print(pandas.__version__)"

2. 依赖冲突
   pip check pandas numpy pyarrow

3. 内存不足？
   df.memory_usage(deep=True).sum() / 1024**3  # 看 GB 数

4. CoW 是否开启？
   pd.options.mode.copy_on_write  # 应该是 True

5. PyArrow 是否可用？
   import pyarrow; print(pyarrow.__version__)

6. 文件编码问题？
   pd.read_csv(..., encoding='utf-8')  # 明确指定编码
```
