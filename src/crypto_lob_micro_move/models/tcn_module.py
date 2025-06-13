from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score


class SimpleTCN(pl.LightningModule):
    def __init__(self, channels: int = 16, kernel_size: int = 3, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.conv = nn.Conv1d(30, channels, kernel_size, padding="same")
        self.fc = nn.Linear(channels, 3)
        self.criterion = nn.CrossEntropyLoss()
        self.f1 = MulticlassF1Score(num_classes=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.relu(x)
        x = x.mean(dim=2)
        return self.fc(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.squeeze())
        self.log("train_loss", loss)
        self.log("train_f1", self.f1(logits, y.squeeze()))
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.squeeze())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.f1(logits, y.squeeze()), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
