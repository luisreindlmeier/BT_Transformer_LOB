"""Simple MLP baseline for LOB prediction."""

from __future__ import annotations

import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Simple 3-layer MLP baseline for flattened LOB windows.
    
    Args:
        input_dim: Flattened input dimension (window_size * n_features)
        hidden_dims: Hidden layer sizes (default: [256, 128])
        num_classes: Number of output classes (default: 3)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        num_classes: int = 3,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) - Flattened input
            
        Returns:
            logits: (B, num_classes)
        """
        return self.net(x)
