#!/usr/bin/env python3
"""
Simulation 1: Traditional LKAS
Module 01 + 02 Integration with CARLA

Usage:
    python main.py
"""
import sys
from pathlib import Path
import time
import numpy as np

from carla_interface import CarlaInterface
from lane_detector_node import LaneDetectorNode
from lane_keeper_node import LaneKeeperNode


def main():
    """Main execution"""
    print("="*80)
    print("Simulation 1: Traditional LKAS (Simplified)")
    print("Module 01 (Lane Detection) + Module 02 (PID Control)")
    print("="*80)
    
    # Configuration
    MODEL_PATH = Path(__file__).parent.parent.parent / '01-lane-detection' / 'checkpoints' / 'best_model.pth'
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
        
        # Wait for camera to start streaming
        print("\n[Step 4] Waiting for camera stream...")
        time.sleep(3.0)
        
        # 4. Initialize modules
        print("\n[Step 5] Initializing modules...")
        
        lane_detector = LaneDetectorNode(
            model_path=str(MODEL_PATH),
            device=DEVICE
        )
        
        lane_keeper = LaneKeeperNode(
            kp=1.5,
            ki=0.1,
            kd=0.8,
            track_width=1.5
        )
        
        print("\n✅ All modules initialized!")
        print("\n" + "="*80)
        print("Starting main loop (30Hz)")
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        # Main loop
        frame_count = 0
        prev_time = time.time()
        
        # Stats
        total_latency = []
        
        while True:
            loop_start = time.time()
            
            # Get sensor data
            image = carla.get_latest_image()
            if image is None:
                time.sleep(0.01)
                continue
            
            vehicle_state = carla.get_vehicle_state()
            
            # Lane detection (Module 01)
            lane_info = lane_detector.detect(image)
            
            # Lane keeping control (Module 02)
            dt = time.time() - prev_time
            prev_time = time.time()
            
            control = lane_keeper.compute_control(
                lateral_offset=lane_info['lateral_offset'],
                heading_error=lane_info['heading_error'],
                velocity=vehicle_state['velocity'],
                dt=dt
            )
            
            # Apply control
            carla.apply_control(
                steering=control['steering'],
                throttle=control['throttle']
            )
            
            # Logging
            loop_time = (time.time() - loop_start) * 1000
            total_latency.append(loop_time)
            
            if frame_count % 30 == 0:  # Every second
                avg_latency = np.mean(total_latency[-30:]) if total_latency else 0
                fps = 1000 / avg_latency if avg_latency > 0 else 0
                
                print(f"[Frame {frame_count:04d}] FPS: {fps:.1f}")
                print(f"  Lateral offset: {lane_info['lateral_offset']:+.3f}m")
                print(f"  Heading error: {lane_info['heading_error']:+.3f}rad")
                print(f"  Steering: {control['steering']:+.2f}°")
                print(f"  Throttle: {control['throttle']:.2f}")
                print(f"  Risk: {control['warning']}")
                print(f"  Latency: {avg_latency:.1f}ms")
                print()
            
            frame_count += 1
            
            # Target 30Hz (33ms per frame)
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
