#!/usr/bin/env python3
"""
Simulation 3: Reinforcement Learning with PPO + Curiosity
Module 08 Integration with CARLA

Usage:
    python main.py
"""
import sys
from pathlib import Path

# Add Sim1 to reuse CarlaInterface
sys.path.insert(0, str(Path(__file__).parent.parent / 'sim1-traditional'))

import time
import numpy as np

from carla_interface import CarlaInterface  # Reuse!
from carla_gym_env import CARLAGymEnv
from rl_agent_node import RLAgentNode


def main():
    """Main execution"""
    print("="*80)
    print("Simulation 3: Reinforcement Learning with PPO + Curiosity")
    print("Module 08 (RL + ICM)")
    print("="*80)
    
    # Configuration
    CHECKPOINT_PATH = Path(__file__).parent.parent.parent / '08-reinforcement-learning' / 'checkpoints' / 'best_ppo.pth'
    DEVICE = 'cuda'  # or 'cpu'
    
    # Initialize CARLA
    carla = CarlaInterface()
    
    try:
        # 1. Connect to CARLA
        print("\n[Step 1] Connecting to CARLA...")
        carla.connect()
        
        # 2. Spawn vehicle
        print("\n[Step 2] Spawning vehicle...")
        vehicle = carla.spawn_vehicle()
        
        # 3. Spawn camera
        print("\n[Step 3] Spawning camera...")
        camera = carla.spawn_camera()
        
        # Wait for camera
        print("\n[Step 4] Waiting for camera stream...")
        time.sleep(3.0)
        
        # 4. Initialize RL components
        print("\n[Step 5] Initializing RL agent...")
        
        # CARLA-Gym wrapper
        carla_gym = CARLAGymEnv(carla)
        
        # RL agent
        rl_agent = RLAgentNode(
            checkpoint_path=str(CHECKPOINT_PATH),
            device=DEVICE
        )
        
        print("\n✅ All modules initialized!")
        print("\n" + "="*80)
        print("Starting RL control (30Hz)")
        print("PPO Agent with Curiosity")
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        # Reset environment
        obs, info = carla_gym.reset()
        
        # Main loop
        frame_count = 0
        episode_reward = 0.0
        total_latency = []
        
        while True:
            loop_start = time.time()
            
            # RL agent action selection
            action, value, agent_time = rl_agent.select_action(
                obs, deterministic=True
            )
            
            # Step environment
            next_obs, reward, terminated, truncated, info = carla_gym.step(action)
            
            episode_reward += reward
            
            # Update obs
            obs = next_obs
            
            # Reset if done
            if terminated or truncated:
                print(f"\n📊 Episode finished!")
                print(f"  Total reward: {episode_reward:.2f}")
                print(f"  Steps: {frame_count}\n")
                
                obs, info = carla_gym.reset()
                episode_reward = 0.0
                frame_count = 0
            
            # Logging
            loop_time = (time.time() - loop_start) * 1000
            total_latency.append(loop_time)
            
            if frame_count % 30 == 0:  # Every second
                avg_latency = np.mean(total_latency[-30:]) if total_latency else 0
                fps = 1000 / avg_latency if avg_latency > 0 else 0
                
                steering = action[0] * 45.0
                throttle = action[1]
                
                print(f"[Frame {frame_count:04d}] FPS: {fps:.1f}")
                print(f"  Action (RL): Steer={steering:+.2f}°, Throttle={throttle:.2f}")
                print(f"  Value (V): {value:.3f}")
                print(f"  Reward: {reward:+.3f}")
                print(f"  Episode Reward: {episode_reward:+.2f}")
                print(f"  RL Latency: {agent_time:.1f}ms")
                print(f"  Total Latency: {avg_latency:.1f}ms")
                print()
            
            frame_count += 1
            
            # Target 30Hz
            sleep_time = max(0, 0.033 - (time.time() - loop_start))
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("⏹️ Stopped by user")
        print("="*80)
        
        if total_latency:
            print(f"\nStatistics:")
            print(f"  Total frames: {frame_count}")
            print(f"  Avg latency: {np.mean(total_latency):.1f}ms")
            print(f"  Avg FPS: {1000/np.mean(total_latency):.1f}")
            print(f"  Episode reward: {episode_reward:.2f}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up...")
        carla.cleanup()
        print("✅ Done!")


if __name__ == '__main__':
    main()
