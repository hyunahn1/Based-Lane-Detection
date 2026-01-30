"""
Complete End-to-End Model
Image → ViT → Control
"""
import torch
import torch.nn as nn
from .vit import VisionTransformer
from .control_head import ControlHead


class EndToEndModel(nn.Module):
    """
    Complete End-to-End model
    Image → Control (Steering, Throttle)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Vision encoder
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Control head
        self.control_head = ControlHead(
            embed_dim=embed_dim,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224)
        return: (B, 2) - [steering, throttle]
        """
        features = self.encoder(x)  # (B, 768)
        control = self.control_head(features)  # (B, 2)
        return control
