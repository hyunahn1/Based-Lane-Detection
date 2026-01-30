"""
CARLA-Gymnasium Environment Wrapper
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from typing import Dict, Tuple


class CARLAGymEnv(gym.Env):
    """
    CARLA를 Gymnasium 환경으로 래핑
    """
    def __init__(self, carla_interface):
        super().__init__()
        
        self.carla = carla_interface
        
        # Observation space (Module 08 호환)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, (84, 84), dtype=np.uint8),
            'velocity': spaces.Box(0, 10, (1,), dtype=np.float32),
            'steering': spaces.Box(-1, 1, (1,), dtype=np.float32),
            'prev_action': spaces.Box(-1, 1, (2,), dtype=np.float32)
        })
        
        # Action space: [steering, throttle]
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        
        # State
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        print("✅ CARLA-Gym Environment initialized")
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Reset state
        self.prev_action = np.zeros(2)
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action in CARLA
        
        Args:
            action: [steering, throttle] in [-1, 1]
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Apply action to CARLA
        steering = float(action[0]) * 45.0  # Scale to degrees
        throttle = float(np.clip(action[1], 0, 1))  # Only positive
        
        self.carla.apply_control(steering, throttle)
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(next_obs)
        
        # Check termination
        terminated = False  # Continuous task
        truncated = self.episode_steps >= 1000  # Max episode length
        
        # Update state
        self.prev_action = action
        self.episode_reward += reward
        self.episode_steps += 1
        
        info = {
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps
        }
        
        return next_obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict:
        """Get current observation"""
        # Get image
        image = self.carla.get_latest_image()
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Preprocess: Grayscale + Resize
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        
        # Get vehicle state
        vehicle_state = self.carla.get_vehicle_state()
        velocity = vehicle_state.get('velocity', 0.0)
        
        # Get current steering (from prev_action)
        current_steering = self.prev_action[0] if len(self.prev_action) > 0 else 0.0
        
        obs = {
            'image': resized,
            'velocity': np.array([velocity], dtype=np.float32),
            'steering': np.array([current_steering], dtype=np.float32),
            'prev_action': self.prev_action.copy()
        }
        
        return obs
    
    def _calculate_reward(self, obs: Dict) -> float:
        """Calculate reward"""
        # Speed reward
        velocity = obs['velocity'][0]
        speed_reward = velocity / 3.0  # Target: 3 m/s
        
        # Smoothness penalty
        action_diff = np.abs(self.prev_action).sum()
        smoothness_penalty = -0.1 * action_diff
        
        # Total reward
        reward = speed_reward + smoothness_penalty
        
        return float(reward)
