"""
Intrinsic Curiosity Module (ICM)

Paper: "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017)

핵심 아이디어:
- Forward Model: φ(s_t), a_t → φ̂(s_{t+1}) 예측
- Inverse Model: φ(s_t), φ(s_{t+1}) → â_t 예측
- Intrinsic Reward: 예측 오차 = 새로운 경험 = 높은 보상
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class FeatureNetwork(nn.Module):
    """
    Feature Encoder φ(s)
    이미지를 압축된 feature vector로 변환
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        # CNN for image (3, 84, 84) → features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # → 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 3, 84, 84)
        return: (batch, feature_dim)
        """
        return self.encoder(x)


class InverseModel(nn.Module):
    """
    Inverse Model: φ(s_t), φ(s_{t+1}) → â_t
    
    "상태 변화를 보고 어떤 행동을 했는지 예측"
    → 행동과 관련된 feature만 학습하도록 유도
    """
    def __init__(self, feature_dim: int = 256, action_dim: int = 2):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, phi_t: torch.Tensor, phi_t1: torch.Tensor) -> torch.Tensor:
        """
        phi_t: (batch, feature_dim)
        phi_t1: (batch, feature_dim)
        return: (batch, action_dim) predicted action
        """
        combined = torch.cat([phi_t, phi_t1], dim=1)
        return self.model(combined)


class ForwardModel(nn.Module):
    """
    Forward Model: φ(s_t), a_t → φ̂(s_{t+1})
    
    "현재 상태 + 행동으로 다음 상태 예측"
    → 예측 오차 = intrinsic reward
    """
    def __init__(self, feature_dim: int = 256, action_dim: int = 2):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, phi_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        phi_t: (batch, feature_dim)
        action: (batch, action_dim)
        return: (batch, feature_dim) predicted next state features
        """
        combined = torch.cat([phi_t, action], dim=1)
        return self.model(combined)


class IntrinsicCuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM)
    
    Components:
        1. Feature Network: φ(s)
        2. Inverse Model: φ(s_t), φ(s_{t+1}) → â_t
        3. Forward Model: φ(s_t), a_t → φ̂(s_{t+1})
    
    Intrinsic Reward:
        r_i = η * ||φ̂(s_{t+1}) - φ(s_{t+1})||²
        
        η: scaling factor
        예측 못함 = 새로운 경험 = 높은 보상
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        action_dim: int = 2,
        lr: float = 1e-3,
        beta: float = 0.2,  # inverse model loss weight
        eta: float = 0.5,   # intrinsic reward scaling
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.beta = beta
        self.eta = eta
        self.device = device
        
        # Networks
        self.feature_net = FeatureNetwork(feature_dim).to(device)
        self.inverse_model = InverseModel(feature_dim, action_dim).to(device)
        self.forward_model = ForwardModel(feature_dim, action_dim).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def compute_intrinsic_reward(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intrinsic reward (prediction error)
        
        Args:
            obs_t: (batch, 3, 84, 84) current observation
            obs_t1: (batch, 3, 84, 84) next observation
            action: (batch, action_dim) action taken
        
        Returns:
            intrinsic_reward: (batch,) scalar reward
        """
        with torch.no_grad():
            # Normalize images
            obs_t = obs_t.float() / 255.0
            obs_t1 = obs_t1.float() / 255.0
            
            # Encode features
            phi_t = self.feature_net(obs_t)
            phi_t1 = self.feature_net(obs_t1)
            
            # Predict next state
            phi_t1_pred = self.forward_model(phi_t, action)
            
            # Prediction error = curiosity
            error = (phi_t1_pred - phi_t1).pow(2).sum(dim=1)
            
            # Scale to reasonable range
            intrinsic_reward = self.eta * error
        
        return intrinsic_reward
    
    def update(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Update ICM networks
        
        Returns:
            inverse_loss, forward_loss
        """
        # Normalize images
        obs_t = obs_t.float() / 255.0
        obs_t1 = obs_t1.float() / 255.0
        
        # Encode features
        phi_t = self.feature_net(obs_t)
        phi_t1 = self.feature_net(obs_t1)
        
        # Inverse model loss
        action_pred = self.inverse_model(phi_t, phi_t1)
        inverse_loss = nn.MSELoss()(action_pred, action)
        
        # Forward model loss
        phi_t1_pred = self.forward_model(phi_t, action)
        forward_loss = nn.MSELoss()(phi_t1_pred, phi_t1.detach())
        
        # Combined loss
        total_loss = self.beta * inverse_loss + (1 - self.beta) * forward_loss
        
        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        return inverse_loss.item(), forward_loss.item()
    
    def save(self, path: str):
        """Save ICM"""
        torch.save({
            'feature_net': self.feature_net.state_dict(),
            'inverse_model': self.inverse_model.state_dict(),
            'forward_model': self.forward_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load ICM"""
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_net.load_state_dict(checkpoint['feature_net'])
        self.inverse_model.load_state_dict(checkpoint['inverse_model'])
        self.forward_model.load_state_dict(checkpoint['forward_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
