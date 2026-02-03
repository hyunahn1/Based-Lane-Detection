"""
Performance Benchmarking for All Modules
Measure inference speed and model efficiency
"""
import sys
import time
import numpy as np
import torch
from pathlib import Path

print("="*80)
print("Performance Benchmarking - Autonomous Driving Modules")
print("="*80)

results = {}

# ============================================================================
# Module 03: Object Detection (YOLOv8)
# ============================================================================
print("\n" + "="*80)
print("Module 03: Object Detection (YOLOv8)")
print("="*80)

try:
    sys.path.insert(0, str(Path('03-object-detection')))
    from src.detector import ObjectDetector
    
    detector = ObjectDetector(weights='yolov8n.pt', device='cpu', conf_thres=0.25)
    
    # Warmup
    fake_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        _ = detector.detect(fake_image)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        result = detector.detect(fake_image)
        times.append((time.time() - start) * 1000)
    
    avg_latency = np.mean(times)
    std_latency = np.std(times)
    fps = 1000 / avg_latency
    
    results['Module 03 (YOLOv8)'] = {
        'latency_ms': avg_latency,
        'std_ms': std_latency,
        'fps': fps,
        'device': 'CPU'
    }
    
    print(f"✅ Module 03 Benchmark Complete")
    print(f"   Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"   FPS: {fps:.1f}")
    
except Exception as e:
    print(f"❌ Module 03 Failed: {e}")
    results['Module 03 (YOLOv8)'] = {'error': str(e)}

# ============================================================================
# Module 06: End-to-End Learning (ViT)
# ============================================================================
print("\n" + "="*80)
print("Module 06: End-to-End Learning (ViT)")
print("="*80)

try:
    sys.path.insert(0, str(Path('06-end-to-end-learning/src')))
    from models.e2e_model import EndToEndModel
    
    model = EndToEndModel()
    model.eval()
    
    # Warmup
    fake_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(10):
            _ = model(fake_input)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            control = model(fake_input)
        times.append((time.time() - start) * 1000)
    
    avg_latency = np.mean(times)
    std_latency = np.std(times)
    fps = 1000 / avg_latency
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    results['Module 06 (ViT E2E)'] = {
        'latency_ms': avg_latency,
        'std_ms': std_latency,
        'fps': fps,
        'params': total_params,
        'device': 'CPU'
    }
    
    print(f"✅ Module 06 Benchmark Complete")
    print(f"   Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"   FPS: {fps:.1f}")
    print(f"   Parameters: {total_params:,}")
    
except Exception as e:
    print(f"❌ Module 06 Failed: {e}")
    results['Module 06 (ViT E2E)'] = {'error': str(e)}

# ============================================================================
# Module 08: Reinforcement Learning (PPO)
# ============================================================================
print("\n" + "="*80)
print("Module 08: Reinforcement Learning (PPO)")
print("="*80)

try:
    sys.path.insert(0, str(Path('08-reinforcement-learning/src')))
    from agent.ppo_agent import PPOAgent
    from environment.rc_track_env import RCTrackEnv
    
    env = RCTrackEnv()
    agent = PPOAgent(env.observation_space, env.action_space, device='cpu')
    
    # Warmup
    obs, _ = env.reset()
    for _ in range(10):
        action, _, _ = agent.select_action(obs)
    
    # Benchmark
    times = []
    for _ in range(100):
        obs, _ = env.reset()
        start = time.time()
        action, log_prob, value = agent.select_action(obs)
        times.append((time.time() - start) * 1000)
    
    avg_latency = np.mean(times)
    std_latency = np.std(times)
    fps = 1000 / avg_latency
    
    # Count parameters
    total_params = sum(p.numel() for p in agent.policy.parameters())
    
    results['Module 08 (PPO RL)'] = {
        'latency_ms': avg_latency,
        'std_ms': std_latency,
        'fps': fps,
        'params': total_params,
        'device': 'CPU'
    }
    
    print(f"✅ Module 08 Benchmark Complete")
    print(f"   Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"   FPS: {fps:.1f}")
    print(f"   Parameters: {total_params:,}")
    
except Exception as e:
    print(f"❌ Module 08 Failed: {e}")
    results['Module 08 (PPO RL)'] = {'error': str(e)}

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("📊 Performance Summary")
print("="*80)

for module, metrics in results.items():
    print(f"\n{module}:")
    if 'error' in metrics:
        print(f"  ❌ Error: {metrics['error']}")
    else:
        print(f"  Latency: {metrics['latency_ms']:.2f} ± {metrics.get('std_ms', 0):.2f} ms")
        print(f"  FPS: {metrics['fps']:.1f}")
        if 'params' in metrics:
            print(f"  Parameters: {metrics['params']:,}")
        print(f"  Device: {metrics['device']}")

print("\n" + "="*80)
print("✅ Benchmarking Complete!")
print("="*80)

# Save results
import json
with open('performance_results.json', 'w') as f:
    # Convert numpy types to native Python types
    results_serializable = {}
    for module, metrics in results.items():
        results_serializable[module] = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                results_serializable[module][key] = float(value)
            else:
                results_serializable[module][key] = value
    
    json.dump(results_serializable, f, indent=2)

print("\n📄 Results saved to: performance_results.json")
