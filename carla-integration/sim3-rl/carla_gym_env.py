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
            'image': spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8),  # RGB
            'velocity': spaces.Box(0, 10, (1,), dtype=np.float32),
            'steering': spaces.Box(-1, 1, (1,), dtype=np.float32),
            'lateral_offset': spaces.Box(-5, 5, (1,), dtype=np.float32),
            'heading_error': spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32),
            'distance_to_obstacle': spaces.Box(0, 10, (1,), dtype=np.float32),
            'prev_actions': spaces.Box(-1, 1, (2,), dtype=np.float32)  # 's' 추가
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
        
        # Preprocess: Resize (keep RGB)
        resized = cv2.resize(image, (84, 84))
        
        # Get vehicle state
        vehicle_state = self.carla.get_vehicle_state()
        velocity = vehicle_state.get('velocity', 0.0)
        
        # Get lane info
        lane_info = self.carla.get_lane_info()
        lateral_offset = lane_info.get('lateral_offset', 0.0)
        heading_error = lane_info.get('heading_error', 0.0)
        
        # Get obstacle distance
        distance_to_obstacle = self.carla.get_obstacle_distance()
        
        # Get current steering (from prev_action)
        current_steering = self.prev_action[0] if len(self.prev_action) > 0 else 0.0
        
        obs = {
            'image': resized,
            'velocity': np.array([velocity], dtype=np.float32),
            'steering': np.array([current_steering], dtype=np.float32),
            'lateral_offset': np.array([lateral_offset], dtype=np.float32),
            'heading_error': np.array([heading_error], dtype=np.float32),
            'distance_to_obstacle': np.array([distance_to_obstacle], dtype=np.float32),
            'prev_actions': self.prev_action.copy()  # 's' 추가
        }
        
        return obs
    
    def _calculate_reward(self, obs: Dict) -> float:
        """
        Calculate reward (정석적인 자율주행 RL 방법)
        
        Reward components:
        1. Speed reward: 목표 속도 유지
        2. Lane keeping: 차선 중앙 유지
        3. Heading alignment: 차선 방향 정렬
        4. Safety: 장애물 회피
        5. Smoothness: 부드러운 제어
        """
        # 1. Speed reward (target: 5 m/s = 18 km/h)
        velocity = obs['velocity'][0]
        target_speed = 5.0
        speed_reward = 1.0 - abs(velocity - target_speed) / target_speed
        speed_reward = max(0.0, speed_reward)
        
        # 2. Lane keeping penalty
        lateral_offset = obs['lateral_offset'][0]
        lane_penalty = -abs(lateral_offset) * 0.5
        
        # 3. Heading alignment penalty
        heading_error = obs['heading_error'][0]
        heading_penalty = -abs(heading_error) * 0.3
        
        # 4. Safety reward (obstacle avoidance)
        distance = obs['distance_to_obstacle'][0]
        if distance < 2.0:
            safety_penalty = -2.0 * (2.0 - distance)  # 큰 페널티
        else:
            safety_penalty = 0.0
        
        # 5. Smoothness penalty (부드러운 제어)
        action_diff = np.abs(self.prev_action).sum()
        smoothness_penalty = -0.05 * action_diff
        
        # Total reward
        reward = (
            1.0 * speed_reward +
            1.0 * lane_penalty +
            0.5 * heading_penalty +
            1.5 * safety_penalty +
            0.3 * smoothness_penalty
        )
        
        return float(reward)
