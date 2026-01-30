"""
RL Agent Node (Module 08 Integration)
"""
import sys
from pathlib import Path

# Add Module 08 to path
module08_path = Path(__file__).parent.parent.parent / '08-reinforcement-learning'
sys.path.insert(0, str(module08_path))

import torch
import numpy as np
from typing import Dict, Tuple
import time


class RLAgentNode:
    """
    Module 08 wrapper for CARLA integration
    PPO Agent with Curiosity
    """
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda'
    ):
        self.device = device
        
        # Load Module 08 agent
        from src.agent.ppo_agent import PPOAgent
        from src.environment.rc_track_env import RCTrackEnv
        
        # Create dummy env for obs/action spaces
        dummy_env = RCTrackEnv()
        
        self.agent = PPOAgent(
            obs_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            device=device
        )
        
        if Path(checkpoint_path).exists():
            self.agent.policy.load_state_dict(
                torch.load(checkpoint_path, map_location=device)
            )
            print(f"✅ RL Agent loaded from {checkpoint_path}")
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            print(f"   Using untrained agent (for interface testing)")
        
        print(f"✅ RL Agent initialized ({device})")
    
    def select_action(
        self,
        obs: Dict,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action using PPO policy
        
        Args:
            obs: Observation dict
            deterministic: Use mean action (no exploration)
        
        Returns:
            action, value, processing_time
        """
        start = time.time()
        
        action, log_prob, value = self.agent.select_action(
            obs, deterministic=deterministic
        )
        
        processing_time = (time.time() - start) * 1000
        
        return action, value, processing_time
