"""
PPO Agent (Simplified for quick testing)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

from .networks import ActorCritic


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    Simplified version for testing:
    - Basic PPO loss
    - GAE for advantages
    - Clipped surrogate objective
    """
    
    def __init__(
        self,
        obs_space,
        action_space,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        device: str = 'cpu'
    ):
        self.obs_space = obs_space
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.device = device
        
        # Network
        self.policy = ActorCritic(obs_space, action_space).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_action(
        self,
        obs: Dict,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Select action"""
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs)
            
            action_mean, action_std, value = self.policy(obs_tensor)
            
            if deterministic:
                action = action_mean
                log_prob = 0.0
            else:
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().numpy().item() if not deterministic else 0.0
            value = value.cpu().numpy()[0, 0]
        
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action, log_prob, value
    
    def update(
        self,
        trajectories: List[Dict],
        num_epochs: int = 4,
        batch_size: int = 64
    ) -> Dict:
        """
        PPO update (simplified)
        """
        if len(trajectories) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Extract data
        obs_list = [t['obs'] for t in trajectories]
        actions = torch.FloatTensor([t['action'] for t in trajectories]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in trajectories]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in trajectories]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in trajectories]).to(self.device)
        values = torch.FloatTensor([t['value'] for t in trajectories]).to(self.device)
        
        # Compute returns with GAE
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        
        for epoch in range(num_epochs):
            # Mini-batch update
            obs_batch = self._stack_obs(obs_list)
            
            # Forward
            action_mean, action_std, values_pred = self.policy(obs_batch)
            
            dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * (returns - values_pred.squeeze()).pow(2).mean()
            
            # Total loss
            loss = policy_loss + value_loss
            
            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        return {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs
        }
    
    def _obs_to_tensor(self, obs: Dict) -> Dict:
        """Convert observation to tensor"""
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                if key == 'image':
                    # Image: (H, W, C) -> (C, H, W)
                    obs_tensor[key] = torch.FloatTensor(value).permute(2, 0, 1).unsqueeze(0).to(self.device)
                else:
                    obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            else:
                obs_tensor[key] = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
        return obs_tensor
    
    def _stack_obs(self, obs_list: List[Dict]) -> Dict:
        """Stack observations into batch"""
        obs_batch = {}
        for key in obs_list[0].keys():
            if key == 'image':
                # Image: (B, H, W, C) -> (B, C, H, W)
                stacked = np.array([obs[key] for obs in obs_list])
                obs_batch[key] = torch.FloatTensor(stacked).permute(0, 3, 1, 2).to(self.device)
            else:
                obs_batch[key] = torch.FloatTensor(
                    np.array([obs[key] for obs in obs_list])
                ).to(self.device)
        return obs_batch
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
