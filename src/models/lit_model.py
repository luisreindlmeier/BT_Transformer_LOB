from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.classification import MulticlassF1Score
from torchmetrics import ConfusionMatrix

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class LiT(pl.LightningModule):

    def __init__(
        self,
        event_dim: int = 6,
        lob_dim: int = 40,
        extra_dim: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-4,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.lr = lr
        self.class_weights = class_weights

        # -----------------
        # Event encoder
        # -----------------
        self.event_proj = nn.Linear(event_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # -----------------
        # Fusion head
        # -----------------
        fusion_dim = d_model + lob_dim + extra_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # -----------------
        # Metrics
        # -----------------
        self.val_f1_macro = MulticlassF1Score(
            num_classes=3, average="macro"
        )
        self.test_f1_macro = MulticlassF1Score(
            num_classes=3, average="macro"
        )

        self.test_confmat = ConfusionMatrix(
            task="multiclass", num_classes=3
        )

    # -------------------------------------------------

    def forward(self, events, lob, extras):
        """
        events: [B, T, 6]
        lob   : [B, 40]
        extras: [B, 3]
        """
        x = self.event_proj(events)
        x = self.pos_enc(x)
        x = self.transformer(x)

        # Mean pooling over time
        x = x.mean(dim=1)

        fused = torch.cat([x, lob, extras], dim=1)
        logits = self.mlp(fused)
        return logits

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    def training_step(self, batch, batch_idx):
        logits = self(batch["events"], batch["lob"], batch["extras"])
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None

        loss = F.cross_entropy(
            logits,
            batch["y_cls"],
            weight=weight,
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def validation_step(self, batch, batch_idx):
        logits = self(batch["events"], batch["lob"], batch["extras"])
        preds = logits.argmax(dim=1)

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None

        loss = F.cross_entropy(
            logits,
            batch["y_cls"],
            weight=weight,
        )

        # --- validation accuracy ---
        val_acc = (preds == batch["y_cls"]).float().mean()

        # --- metrics ---
        self.val_f1_macro.update(preds, batch["y_cls"])

        # --- logging ---
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)



    def on_validation_epoch_end(self):
        f1 = self.val_f1_macro.compute()
        self.log("val_f1_macro", f1, prog_bar=True)
        self.val_f1_macro.reset()

    # -------------------------------------------------
    # Test (FINAL EVALUATION)
    # -------------------------------------------------

    def test_step(self, batch, batch_idx):
        logits = self(batch["events"], batch["lob"], batch["extras"])
        preds = logits.argmax(dim=1)

        self.test_f1_macro.update(preds, batch["y_cls"])
        self.test_confmat.update(preds, batch["y_cls"])

    def on_test_epoch_end(self):
        f1 = self.test_f1_macro.compute()
        confmat = self.test_confmat.compute()

        print("\n================ TEST RESULTS ================")
        print(f"Macro F1: {f1.item():.4f}")
        print("Confusion Matrix:")
        print(confmat.cpu().numpy())
        print("=============================================")

        self.test_f1_macro.reset()
        self.test_confmat.reset()

    # -------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)