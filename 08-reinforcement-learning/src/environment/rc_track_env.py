"""
RC Track Gymnasium Environment
Simple 2D simulation for testing RL algorithms
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional
import pygame


class RCTrackEnv(gym.Env):
    """
    RC Track Environment
    
    간단한 2D 시뮬레이션:
    - 차량이 직선 트랙을 따라 주행
    - 목표: 중앙 유지하며 빠르게 주행
    - 장애물 회피 (optional)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        track_type: str = 'easy',
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        dt: float = 0.1
    ):
        super().__init__()
        
        self.track_type = track_type
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.dt = dt
        
        # Track parameters (simplified)
        self.track_width = 1.0  # meters
        self.track_length = 100.0  # meters
        
        # Car parameters
        self.car_x = 0.0  # along track
        self.car_y = 0.0  # lateral position
        self.car_velocity = 0.0
        self.car_steering = 0.0
        self.car_heading = 0.0  # relative to track
        
        # Observation space
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, shape=(3, 84, 84), dtype=np.uint8),
            'velocity': gym.spaces.Box(0, 3.0, shape=(1,), dtype=np.float32),
            'steering': gym.spaces.Box(-45, 45, shape=(1,), dtype=np.float32),
            'lateral_offset': gym.spaces.Box(-0.5, 0.5, shape=(1,), dtype=np.float32),
            'heading_error': gym.spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
            'distance_to_obstacle': gym.spaces.Box(0, 10.0, shape=(1,), dtype=np.float32),
            'prev_actions': gym.spaces.Box(-1, 1, shape=(5, 2), dtype=np.float32)
        })
        
        # Action space: [steering, throttle]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # State
        self.step_count = 0
        self.prev_actions = np.zeros((5, 2))
        
        # Pygame for rendering
        self.screen = None
        self.clock = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # Reset car to start
        self.car_x = 0.0
        self.car_y = 0.0  # center of track
        self.car_velocity = 0.0
        self.car_steering = 0.0
        self.car_heading = 0.0
        
        self.step_count = 0
        self.prev_actions = np.zeros((5, 2))
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one timestep
        """
        # Denormalize action
        steering = action[0] * 45.0  # [-1, 1] → [-45, 45] degrees
        throttle = action[1]  # [0, 1]
        
        # Simple kinematic model
        self.car_steering = steering
        
        # Update velocity
        self.car_velocity += throttle * 2.0 * self.dt  # acceleration
        self.car_velocity -= 0.5 * self.car_velocity * self.dt  # friction
        self.car_velocity = np.clip(self.car_velocity, 0, 3.0)
        
        # Update heading
        self.car_heading += np.deg2rad(steering) * self.car_velocity * self.dt * 0.1
        
        # Update position
        self.car_x += self.car_velocity * np.cos(self.car_heading) * self.dt
        self.car_y += self.car_velocity * np.sin(self.car_heading) * self.dt
        
        # Update prev actions
        self.prev_actions = np.roll(self.prev_actions, -1, axis=0)
        self.prev_actions[-1] = action
        
        # Check termination
        off_track = abs(self.car_y) > self.track_width / 2
        goal_reached = self.car_x >= self.track_length
        
        terminated = off_track or goal_reached
        truncated = self.step_count >= self.max_steps
        
        # Compute reward
        reward = self._compute_reward(off_track, goal_reached)
        
        # Get observation
        obs = self._get_obs()
        info = self._get_info()
        info['off_track'] = off_track
        info['goal_reached'] = goal_reached
        
        self.step_count += 1
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> Dict:
        """Get current observation"""
        # Generate simple image (gradient based on position)
        image = self._render_observation()
        
        obs = {
            'image': image,
            'velocity': np.array([self.car_velocity], dtype=np.float32),
            'steering': np.array([self.car_steering], dtype=np.float32),
            'lateral_offset': np.array([self.car_y], dtype=np.float32),
            'heading_error': np.array([self.car_heading], dtype=np.float32),
            'distance_to_obstacle': np.array([10.0], dtype=np.float32),
            'prev_actions': self.prev_actions.copy()
        }
        
        return obs
    
    def _render_observation(self) -> np.ndarray:
        """
        Render observation image (84x84x3)
        Simple visualization: track lanes, car position
        """
        image = np.ones((84, 84, 3), dtype=np.uint8) * 128  # gray background
        
        # Draw track boundaries
        left_lane = int(42 - 20 + self.car_y * 40)
        right_lane = int(42 + 20 + self.car_y * 40)
        
        image[:, max(0, min(83, left_lane)), :] = [255, 255, 255]
        image[:, max(0, min(83, right_lane)), :] = [255, 255, 255]
        
        # Draw center line
        center = int(42 + self.car_y * 40)
        if 0 <= center < 84:
            image[::10, center, :] = [255, 255, 0]
        
        # Transpose to (3, 84, 84) for PyTorch
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def _compute_reward(self, off_track: bool, goal_reached: bool) -> float:
        """
        Reward function
        """
        reward = 0.0
        
        # Speed reward
        reward += self.car_velocity * 0.5
        
        # Centering reward
        centering = np.exp(-5 * abs(self.car_y))
        reward += centering * 1.0
        
        # Smoothness (penalize large steering changes)
        if len(self.prev_actions) > 1:
            action_diff = np.linalg.norm(self.prev_actions[-1] - self.prev_actions[-2])
            smoothness = np.exp(-action_diff)
            reward += smoothness * 0.2
        
        # Off-track penalty
        if off_track:
            reward -= 50.0
        
        # Goal bonus
        if goal_reached:
            reward += 200.0
        
        return reward
    
    def _get_info(self) -> Dict:
        """Get info dict"""
        return {
            'car_x': self.car_x,
            'car_y': self.car_y,
            'velocity': self.car_velocity,
            'step': self.step_count
        }
    
    def render(self):
        """Render environment (pygame)"""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 400))
            self.clock = pygame.time.Clock()
        
        # Draw
        self.screen.fill((50, 50, 50))
        
        # Track
        track_y = 200
        pygame.draw.line(self.screen, (255, 255, 255),
                        (0, track_y - 100), (800, track_y - 100), 2)
        pygame.draw.line(self.screen, (255, 255, 255),
                        (0, track_y + 100), (800, track_y + 100), 2)
        
        # Car position (scaled)
        car_screen_x = int((self.car_x / self.track_length) * 700 + 50)
        car_screen_y = int(track_y + self.car_y * 100)
        
        pygame.draw.circle(self.screen, (0, 255, 0), (car_screen_x, car_screen_y), 10)
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def close(self):
        """Clean up"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
