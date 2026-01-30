#!/usr/bin/env python3
"""
Simulation 2: End-to-End Learning with Vision Transformer
Module 06 Integration with CARLA

Usage:
    python main.py
"""
import sys
from pathlib import Path

# Add Sim1 to path to reuse CarlaInterface
sys.path.insert(0, str(Path(__file__).parent.parent / 'sim1-traditional'))

import time
import numpy as np

from carla_interface import CarlaInterface  # Reuse from Sim1!
from e2e_model_node import E2EModelNode


def main():
    """Main execution"""
    print("="*80)
    print("Simulation 2: End-to-End Learning with Vision Transformer")
    print("Module 06 (E2E + ViT)")
    print("="*80)
    
    # Configuration
    MODEL_PATH = Path(__file__).parent.parent.parent / '06-end-to-end-learning' / 'checkpoints' / 'best_e2e.pth'
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
        
        # 4. Initialize E2E model
        print("\n[Step 5] Initializing E2E model...")
        
        e2e_model = E2EModelNode(
            model_path=str(MODEL_PATH),
            device=DEVICE,
            img_size=224
        )
        
        print("\n✅ All modules initialized!")
        print("\n" + "="*80)
        print("Starting E2E control (30Hz)")
        print("Image → ViT → Control (Direct!)")
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        # Main loop
        frame_count = 0
        total_latency = []
        
        while True:
            loop_start = time.time()
            
            # Get image
            image = carla.get_latest_image()
            if image is None:
                time.sleep(0.01)
                continue
            
            vehicle_state = carla.get_vehicle_state()
            
            # E2E prediction (Image → Control directly!)
            prediction = e2e_model.predict(image)
            
            # Extract control
            steering = prediction['steering'] * 45.0  # Scale to degrees [-45, 45]
            throttle = prediction['throttle']
            
            # Safety override
            if vehicle_state['velocity'] > 5.0:
                throttle = 0.3  # Slow down
            
            # Apply control
            carla.apply_control(steering=steering, throttle=throttle)
            
            # Logging
            loop_time = (time.time() - loop_start) * 1000
            total_latency.append(loop_time)
            
            if frame_count % 30 == 0:  # Every second
                avg_latency = np.mean(total_latency[-30:]) if total_latency else 0
                fps = 1000 / avg_latency if avg_latency > 0 else 0
                
                print(f"[Frame {frame_count:04d}] FPS: {fps:.1f}")
                print(f"  Steering (ViT): {steering:+.2f}°")
                print(f"  Throttle (ViT): {throttle:.2f}")
                print(f"  Velocity: {vehicle_state['velocity']:.2f} m/s")
                print(f"  E2E Latency: {prediction['processing_time']:.1f}ms")
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
