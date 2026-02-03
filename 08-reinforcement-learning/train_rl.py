#!/usr/bin/env python3
"""
Module 08: Reinforcement Learning Training Script
PPO + Curiosity (ICM) in CARLA Environment
"""

import argparse
import sys
from pathlib import Path
import time

# Add CARLA integration to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'carla-integration' / 'sim1-traditional'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'carla-integration' / 'sim3-rl'))
from carla_interface import CarlaInterface
from carla_gym_env import CARLAGymEnv

# Add Module 08 to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from agent.ppo_agent import PPOAgent
from curiosity.icm import IntrinsicCuriosityModule


def main():
    parser = argparse.ArgumentParser(description='Train RL Agent in CARLA')
    parser.add_argument('--carla-host', default='localhost', help='CARLA host')
    parser.add_argument('--carla-port', type=int, default=2000, help='CARLA port')
    parser.add_argument('--total-steps', type=int, default=3000000, help='Total training steps')
    parser.add_argument('--save-interval', type=int, default=100000, help='Save interval')
    parser.add_argument('--use-curiosity', action='store_true', default=True, help='Use ICM')
    parser.add_argument('--save-dir', default='runs/rl_training')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Module 08: Reinforcement Learning Training")
    print("="*80)
    print(f"CARLA: {args.carla_host}:{args.carla_port}")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Curiosity: {args.use_curiosity}")
    print("="*80)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment
    print("\n🔌 Connecting to CARLA...")
    carla_interface = CarlaInterface()
    carla_interface.connect()
    carla_interface.spawn_vehicle()
    carla_interface.spawn_camera()
    print("⏳ Waiting for camera stream...")
    time.sleep(3)
    env = CARLAGymEnv(carla_interface)
    print("✅ Connected to CARLA")
    
    # Agent
    print("\n🤖 Creating PPO Agent...")
    agent = PPOAgent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        device='cuda'
    )
    print(f"✅ Agent created")
    
    # Curiosity
    curiosity = None
    if args.use_curiosity:
        print("\n🔍 Creating Curiosity Module...")
        curiosity = IntrinsicCuriosityModule(
            feature_dim=256,
            action_dim=2,
            device='cuda'
        )
        print(f"✅ Curiosity module created")
    
    # Training loop
    print("\n" + "="*80)
    print("🚀 Starting Training")
    print("="*80)
    
    episode = 0
    total_steps = 0
    best_reward = float('-inf')
    
    try:
        while total_steps < args.total_steps:
            episode += 1
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            trajectory = []
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Select action
                action, log_prob, value = agent.select_action(obs)
                
                # Step environment
                next_obs, reward_ext, done, truncated, info = env.step(action)
                
                # Curiosity reward
                reward_int = 0.0
                if curiosity is not None:
                    import torch
                    # Image: (H, W, C) -> (C, H, W)
                    obs_tensor = torch.tensor(obs['image'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to('cuda')
                    next_obs_tensor = torch.tensor(next_obs['image'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to('cuda')
                    action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to('cuda')
                    
                    reward_int = curiosity.compute_intrinsic_reward(
                        obs_tensor, next_obs_tensor, action_tensor
                    ).item()
                
                # Combined reward
                reward = reward_ext + 0.2 * reward_int
                
                # Store transition
                trajectory.append({
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'value': value,
                    'log_prob': log_prob,
                    'done': done
                })
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                obs = next_obs
            
            # Update agent
            agent.update(trajectory)
            
            # Update curiosity
            if curiosity is not None and len(trajectory) > 1:
                import torch
                # Get consecutive obs-action pairs
                obs_imgs = torch.stack([
                    torch.tensor(t['obs']['image'], dtype=torch.float32).permute(2, 0, 1)
                    for t in trajectory[:-1]
                ]).to('cuda')
                next_obs_imgs = torch.stack([
                    torch.tensor(trajectory[i+1]['obs']['image'], dtype=torch.float32).permute(2, 0, 1)
                    for i in range(len(trajectory)-1)
                ]).to('cuda')
                actions = torch.tensor([t['action'] for t in trajectory[:-1]], dtype=torch.float32).to('cuda')
                
                inv_loss, fwd_loss = curiosity.update(obs_imgs, next_obs_imgs, actions)
            
            # Logging
            print(f"[Episode {episode:4d}] "
                  f"Steps: {total_steps:7d}/{args.total_steps:7d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Length: {episode_steps:4d}")
            
            # Save best
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(save_dir / 'best_ppo_agent.pth')
                if curiosity is not None:
                    curiosity.save(save_dir / 'best_curiosity.pth')
                print(f"  ✅ Saved best model (reward: {best_reward:.2f})")
            
            # Save checkpoint
            if total_steps % args.save_interval == 0:
                agent.save(save_dir / f'checkpoint_agent_{total_steps}.pth')
                if curiosity is not None:
                    curiosity.save(save_dir / f'checkpoint_curiosity_{total_steps}.pth')
                print(f"  💾 Checkpoint saved at {total_steps} steps")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted")
    
    finally:
        # Final save
        agent.save(save_dir / 'final_ppo_agent.pth')
        if curiosity is not None:
            curiosity.save(save_dir / 'final_curiosity.pth')
        
        # Cleanup
        carla_interface.cleanup()
        
        print("\n" + "="*80)
        print("✅ Training Complete!")
        print("="*80)
        print(f"Total episodes: {episode}")
        print(f"Total steps: {total_steps:,}")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Models saved to: {save_dir}/")
        print("="*80)


if __name__ == '__main__':
    main()
