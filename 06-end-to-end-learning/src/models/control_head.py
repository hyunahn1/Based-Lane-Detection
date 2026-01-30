"""
Control Head for End-to-End Learning
Features → [Steering, Throttle]
"""
import torch
import torch.nn as nn


class ControlHead(nn.Module):
    """
    Control output head
    Maps ViT features to steering and throttle
    """
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [steering, throttle]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, 768)
        return: (B, 2) - [steering, throttle]
        """
        output = self.head(features)  # (B, 2)
        
        # Steering: tanh [-1, 1]
        # Throttle: sigmoid [0, 1]
        steering = torch.tanh(output[:, 0:1])
        throttle = torch.sigmoid(output[:, 1:2])
        
        return torch.cat([steering, throttle], dim=1)
