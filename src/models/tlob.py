"""T-LOB: Dual-attention transformer for LOB windows (Berti & Kasneci, 2025).

Temporal self-attention over snapshots + feature self-attention over channels,
followed by FFN. Predicts 3-class mid-price move from LOB windows.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class BilinearNorm(nn.Module):
    """Lightweight normalization over batch×time for each feature."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        mean = x.mean(dim=(0, 1), keepdim=True)
        std = x.std(dim=(0, 1), keepdim=True) + self.eps
        return (x - mean) / std


class TLOBBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.temporal_ln = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.feature_ln = nn.LayerNorm(d_model)
        self.feature_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal attention over time axis
        y = self.temporal_ln(x)
        y, _ = self.temporal_attn(y, y, y)
        x = x + y

        # Feature attention: treat features as sequence, keep same embedding
        z = x.transpose(1, 2)  # (B, F, D)
        z = self.feature_ln(z)
        z, _ = self.feature_attn(z, z, z)
        z = z.transpose(1, 2)
        x = x + z

        w = self.ffn_ln(x)
        w = self.ffn(w)
        x = x + w
        return x


class TLOB(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_classes: int = 3,
        d_model: int = 256,
        n_heads: int = 1,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_in = BilinearNorm()
        self.proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.blocks = nn.ModuleList([
            TLOBBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.norm_in(x)
        x = self.proj(x)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        rep = x[:, -1, :]
        return self.head(rep)
