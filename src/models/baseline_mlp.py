from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.classification import MulticlassF1Score
from torchmetrics import ConfusionMatrix


class LOBMLP(pl.LightningModule):

    def __init__(
        self,
        input_dim: int = 43,   # 40 LOB + 3 extras
        hidden_dims=(128, 64),
        lr: float = 1e-3,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.lr = lr
        self.class_weights = class_weights

        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Dropout(0.1),
            ]
            d = h

        layers.append(nn.Linear(d, 3))
        self.net = nn.Sequential(*layers)

        # metrics
        self.val_f1 = MulticlassF1Score(num_classes=3, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=3, average="macro")
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=3)

    # -------------------------------------------------

    def forward(self, lob, extras):
        x = torch.cat([lob, extras], dim=1)
        return self.net(x)

    # -------------------------------------------------

    def training_step(self, batch, batch_idx):
        logits = self(batch["lob"], batch["extras"])
        weight = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )

        loss = F.cross_entropy(logits, batch["y_cls"], weight=weight)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------------------------------------------------

    def validation_step(self, batch, batch_idx):
        logits = self(batch["lob"], batch["extras"])
        preds = logits.argmax(dim=1)

        weight = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )

        loss = F.cross_entropy(logits, batch["y_cls"], weight=weight)
        acc = (preds == batch["y_cls"]).float().mean()

        self.val_f1.update(preds, batch["y_cls"])

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        self.log("val_f1_macro", f1, prog_bar=True)
        self.val_f1.reset()

    # -------------------------------------------------

    def test_step(self, batch, batch_idx):
        logits = self(batch["lob"], batch["extras"])
        preds = logits.argmax(dim=1)

        self.test_f1.update(preds, batch["y_cls"])
        self.test_cm.update(preds, batch["y_cls"])

    def on_test_epoch_end(self):
        print("\n===== BASELINE MLP TEST RESULTS =====")
        print(f"Macro F1: {self.test_f1.compute().item():.4f}")
        print("Confusion Matrix:")
        print(self.test_cm.compute().cpu().numpy())
        print("====================================")

        self.test_f1.reset()
        self.test_cm.reset()

    # -------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)