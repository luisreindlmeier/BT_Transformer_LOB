import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class LOBTransformer(nn.Module):
    def __init__(
        self,
        input_dim=144,
        d_model=64,
        n_heads=4,
        n_layers=2,
        num_classes=3,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, 144)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        h = self.encoder(x)          # (B, T, d_model)
        h = h.mean(dim=1)            # temporal pooling

        return self.classifier(h)