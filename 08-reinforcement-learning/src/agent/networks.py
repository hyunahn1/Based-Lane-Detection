"""
Actor-Critic Networks for PPO
"""
import torch
import torch.nn as nn
from typing import Tuple


class ActorCritic(nn.Module):
    """
    Actor-Critic Network
    
    Simple architecture for quick testing:
    - CNN for image
    - MLP for scalars
    - Actor head (Gaussian policy)
    - Critic head (value function)
    """
    
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        self.action_dim = action_space.shape[0]
        
        # CNN for image (3, 84, 84) → features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # → 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        
        # MLP for scalars (velocity, steering, etc.)
        # velocity(1) + steering(1) + lateral_offset(1) + heading_error(1) + 
        # distance_to_obstacle(1) + prev_actions(2) = 7 dims
        self.mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU()
        )
        
        # Actor head (mean, log_std)
        self.actor_mean = nn.Linear(128, self.action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        # Critic head (value)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            action_mean, action_std, value
        """
        # Image features
        image = obs['image'].float() / 255.0  # Normalize
        cnn_features = self.cnn(image)
        
        # Scalar features
        scalars = torch.cat([
            obs['velocity'],
            obs['steering'] / 45.0,  # Normalize
            obs['lateral_offset'] * 2.0,  # Scale
            obs['heading_error'],
            obs['distance_to_obstacle'] / 10.0,
            obs['prev_actions'].flatten(start_dim=1)
        ], dim=1)
        mlp_features = self.mlp(scalars)
        
        # Combine
        combined = torch.cat([cnn_features, mlp_features], dim=1)
        shared_features = self.shared(combined)
        
        # Actor (policy)
        action_mean = torch.tanh(self.actor_mean(shared_features))
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        # Critic (value)
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
