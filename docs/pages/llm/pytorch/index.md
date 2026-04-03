# PyTorch Lightning 教程

PyTorch Lightning 是一个轻量级的 PyTorch 封装，简化了深度学习模型的训练流程。

## 安装

```bash
pip install pytorch-lightning
```

## 基本使用

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.Linear(64, 1)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = SimpleModel()
trainer = pl.Trainer(max_epochs=10)
```

## 下一步

继续学习：
- [Hugging Face Transformers教程](/pages/llm/transformers/)
- [LangChain教程](/pages/llm/langchain/)
