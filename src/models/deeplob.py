"""DeepLOB CNN-GRU architecture for LOB prediction."""

from __future__ import annotations

import torch.nn as nn


class DeepLOB(nn.Module):
    """
    Simplified DeepLOB-style architecture combining CNN and GRU layers.
    
    Architecture:
    - Conv2D blocks for feature extraction
    - GRU for temporal modeling
    - FC layer for classification
    
    Args:
        n_features: Number of input features per timestep
        num_classes: Number of output classes (default: 3)
    """

    def __init__(self, n_features: int, num_classes: int = 3):
        super().__init__()

        # Input: (B, 1, T, F)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
        )

        # After convs: (B, C=64, T, F)
        self.gru = nn.GRU(
            input_size=64 * n_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, T, F) - Batch of sequences
            
        Returns:
            logits: (B, num_classes)
        """
        # x: (B, T, F)
        B, T, F = x.shape

        x = x.unsqueeze(1)          # (B, 1, T, F)
        x = self.conv_block_1(x)    # (B, 32, T, F)
        x = self.conv_block_2(x)    # (B, 64, T, F)

        x = x.permute(0, 2, 1, 3)   # (B, T, C, F)
        x = x.reshape(B, T, -1)     # (B, T, C*F)

        _, h = self.gru(x)          # h: (1, B, 128)
        h = h.squeeze(0)

        return self.fc(h)
