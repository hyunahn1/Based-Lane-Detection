#!/usr/bin/env python3
"""
🚗 CARLA 자동 데이터 수집 스크립트
=================================

사용법:
    python auto_collect.py --duration 10

설명:
    - CARLA Autopilot으로 자동 주행
    - 이미지 + steering/throttle 동시 저장
    - Object detection용 bbox도 자동 생성
    - 10분 돌리면 ~10,000장 수집

작성: 2026-01-30
"""

import carla
import numpy as np
import cv2
import pandas as pd
import argparse
import time
import os
import json
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from collections import deque


class CARLADataCollector:
    """CARLA 자동 데이터 수집기"""
    
    def __init__(self, output_dir='collected_data', fps=10):
        """
        Args:
            output_dir: 저장 폴더
            fps: 초당 저장 프레임 수 (10 = 1초에 10장)
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # 폴더 생성
        self.image_dir = self.output_dir / 'images'
        self.label_dir = self.output_dir / 'labels'
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        
        # CARLA 연결
        print("🔌 Connecting to CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Traffic Manager 초기화 (autopilot 필수!)
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        
        # 안전 주행 설정
        self.traffic_manager.set_global_distance_to_leading_vehicle(3.0)  # 앞차 거리 증가
        self.traffic_manager.global_percentage_speed_difference(30.0)  # 속도 30% 감소 (안전 주행)
        
        # 동기 모드 설정 (중요!)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS simulation
        self.world.apply_settings(settings)
        
        print("✅ Connected to CARLA")
        
        # 데이터 저장용
        self.image_queue = Queue()
        self.frame_count = 0
        self.data_records = []
        
        # 차량, 카메라
        self.vehicle = None
        self.camera = None
    
    def spawn_vehicle(self):
        """차량 생성"""
        print("\n🚗 Spawning vehicle...")
        
        # 차량 blueprint
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        # 스폰 포인트
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        
        # 차량 생성
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        print(f"✅ Vehicle spawned at {spawn_point.location}")
        return self.vehicle
    
    def spawn_camera(self):
        """카메라 생성 (차량에 부착)"""
        print("📷 Spawning camera...")
        
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        
        # 카메라 설정
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        
        # 차량 앞쪽 위에 부착
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp, 
            camera_transform, 
            attach_to=self.vehicle
        )
        
        # 이미지 콜백 등록
        self.camera.listen(self.image_queue.put)
        
        print("✅ Camera attached")
        return self.camera
    
    def get_bounding_boxes(self):
        """주변 차량의 bounding box 추출"""
        bboxes = []
        
        # 카메라 위치
        camera_transform = self.camera.get_transform()
        
        # 주변 차량 찾기
        vehicles = self.world.get_actors().filter('vehicle.*')
        
        for vehicle in vehicles:
            if vehicle.id == self.vehicle.id:
                continue  # 자기 자신 제외
            
            # 거리 체크 (너무 먼 차량 제외)
            distance = vehicle.get_location().distance(
                self.vehicle.get_location()
            )
            if distance > 50.0:  # 50m 이내만
                continue
            
            # Bounding box 좌표 계산
            bbox = self.get_image_bbox(vehicle, camera_transform)
            
            if bbox is not None:
                bboxes.append({
                    'class': 0,  # vehicle
                    'bbox': bbox
                })
        
        return bboxes
    
    def get_image_bbox(self, actor, camera_transform):
        """3D bbox를 2D 이미지 좌표로 투영"""
        # Bounding box 꼭짓점
        bbox = actor.bounding_box
        vertices = bbox.get_world_vertices(actor.get_transform())
        
        # 카메라 좌표계로 변환
        K = self.build_projection_matrix(640, 480, 90)
        
        # 2D 투영
        points_2d = []
        for vertex in vertices:
            # World to camera
            point_camera = self.world_to_camera(vertex, camera_transform)
            
            # 카메라 뒤에 있으면 제외
            if point_camera[2] < 0:
                return None
            
            # Camera to image
            point_2d = self.camera_to_image(point_camera, K)
            points_2d.append(point_2d)
        
        # Bounding box (min/max)
        points_2d = np.array(points_2d)
        x_min, y_min = points_2d.min(axis=0)
        x_max, y_max = points_2d.max(axis=0)
        
        # 이미지 밖이면 제외
        if x_max < 0 or x_min > 640 or y_max < 0 or y_min > 480:
            return None
        
        # Clip to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(640, x_max)
        y_max = min(480, y_max)
        
        # YOLO format: x_center, y_center, width, height (normalized)
        x_center = (x_min + x_max) / 2.0 / 640
        y_center = (y_min + y_max) / 2.0 / 480
        width = (x_max - x_min) / 640
        height = (y_max - y_min) / 480
        
        return [x_center, y_center, width, height]
    
    def world_to_camera(self, point, camera_transform):
        """World 좌표를 Camera 좌표로 변환"""
        # Camera matrix (world to camera)
        world_2_camera = np.array(camera_transform.get_inverse_matrix())
        
        # Point to homogeneous
        point_homo = [point.x, point.y, point.z, 1]
        
        # Transform
        point_camera = world_2_camera.dot(point_homo)
        
        # Change from UE4's coordinate system to camera
        # (x, y, z) -> (y, -z, x)
        return [point_camera[1], -point_camera[2], point_camera[0]]
    
    def camera_to_image(self, point, K):
        """Camera 좌표를 Image 좌표로 투영"""
        # Perspective projection
        x = K[0, 0] * point[0] / point[2] + K[0, 2]
        y = K[1, 1] * point[1] / point[2] + K[1, 2]
        return [x, y]
    
    def build_projection_matrix(self, w, h, fov):
        """Intrinsic matrix"""
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K
    
    def save_frame(self, image_data, control, velocity):
        """프레임 저장"""
        # 이미지 변환
        array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
        array = array.reshape((480, 640, 4))[:, :, :3]  # BGRA -> BGR
        
        # 파일명
        filename = f'{self.frame_count:06d}'
        
        # 이미지 저장
        image_path = self.image_dir / f'{filename}.jpg'
        cv2.imwrite(str(image_path), array)
        
        # Bounding boxes 저장 (YOLO format)
        bboxes = self.get_bounding_boxes()
        if bboxes:
            label_path = self.label_dir / f'{filename}.txt'
            with open(label_path, 'w') as f:
                for bbox_data in bboxes:
                    cls = bbox_data['class']
                    bbox = bbox_data['bbox']
                    f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        # CSV 데이터 저장
        self.data_records.append({
            'frame': self.frame_count,
            'image': f'{filename}.jpg',
            'steering': control.steer,
            'throttle': control.throttle,
            'brake': control.brake,
            'velocity': velocity,
            'num_objects': len(bboxes),
            'timestamp': time.time()
        })
        
        self.frame_count += 1
    
    def collect(self, duration_minutes=10):
        """
        데이터 수집 메인 루프
        
        Args:
            duration_minutes: 수집 시간 (분)
        """
        print(f"\n📊 Starting data collection for {duration_minutes} minutes...")
        print(f"   Target FPS: {self.fps}")
        print(f"   Expected frames: ~{int(duration_minutes * 60 * self.fps)}")
        print(f"   Output: {self.output_dir}/")
        print("\n⏱️  Press Ctrl+C to stop early\n")
        
        # 차량 생성
        self.spawn_vehicle()
        self.spawn_camera()
        
        # Autopilot 활성화 (Traffic Manager 사용)
        self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
        print("🤖 Autopilot enabled with Traffic Manager\n")
        
        # 수집 시작
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_save_time = 0
        
        try:
            while time.time() < end_time:
                # CARLA tick
                self.world.tick()
                
                # 이미지 가져오기
                try:
                    image_data = self.image_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # FPS 제어 (지정된 간격마다만 저장)
                current_time = time.time()
                if current_time - last_save_time < self.frame_interval:
                    continue
                
                last_save_time = current_time
                
                # 차량 정보 (actor 유효성 체크)
                try:
                    control = self.vehicle.get_control()
                    velocity_vec = self.vehicle.get_velocity()
                    velocity = np.linalg.norm([velocity_vec.x, velocity_vec.y, velocity_vec.z])
                except RuntimeError:
                    # Actor가 파괴됨 (충돌 등) - 재생성
                    print("\n⚠️  Vehicle destroyed, respawning...")
                    self.spawn_vehicle()
                    self.spawn_camera()
                    self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
                    print("✅ Vehicle respawned, continuing collection\n")
                    continue
                
                # 프레임 저장
                self.save_frame(image_data, control, velocity)
                
                # 진행상황 출력
                elapsed = current_time - start_time
                remaining = end_time - current_time
                fps_actual = self.frame_count / elapsed if elapsed > 0 else 0
                
                print(f"\r[{self.frame_count:5d} frames] "
                      f"Elapsed: {elapsed/60:.1f}m | "
                      f"Remaining: {remaining/60:.1f}m | "
                      f"FPS: {fps_actual:.1f} | "
                      f"Steering: {control.steer:+.3f} | "
                      f"Speed: {velocity*3.6:.1f} km/h", 
                      end='', flush=True)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
        
        finally:
            print("\n\n💾 Saving metadata...")
            self.cleanup()
    
    def cleanup(self):
        """정리 및 저장"""
        # CSV 저장
        if self.data_records:
            df = pd.DataFrame(self.data_records)
            csv_path = self.output_dir / 'labels.csv'
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved {len(df)} records to {csv_path}")
        
        # 통계 저장
        stats = {
            'total_frames': self.frame_count,
            'duration_seconds': time.time() - self.data_records[0]['timestamp'] if self.data_records else 0,
            'fps_average': len(self.data_records) / (time.time() - self.data_records[0]['timestamp']) if self.data_records else 0,
            'output_dir': str(self.output_dir),
            'collection_date': datetime.now().isoformat()
        }
        
        stats_path = self.output_dir / 'stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Saved statistics to {stats_path}")
        
        # 카메라 listening 중단 (중요!)
        try:
            if self.camera is not None and self.camera.is_listening:
                self.camera.stop()
        except Exception:
            pass
        
        # 동기 모드 해제
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass
        
        # 액터 삭제는 CARLA가 자동으로 처리
        # (명시적 destroy() 호출 시 C++ 에러 발생)
        
        print("\n" + "="*80)
        print("✅ Data collection complete!")
        print("="*80)
        print(f"\n📁 Output directory: {self.output_dir}/")
        print(f"   - images/: {self.frame_count} images")
        print(f"   - labels/: {len(list(self.label_dir.glob('*.txt')))} YOLO labels")
        print(f"   - labels.csv: E2E training data")
        print(f"   - stats.json: Collection statistics")
        print("\n💡 Next steps:")
        print(f"   1. Check data quality:")
        print(f"      python check_data.py --data {self.output_dir}")
        print(f"   2. Split for training:")
        print(f"      python split_data.py --data {self.output_dir}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='CARLA 자동 데이터 수집',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 10분 동안 수집 (기본)
  python auto_collect.py --duration 10
  
  # 30분 동안, 초당 20프레임으로 수집
  python auto_collect.py --duration 30 --fps 20
  
  # 커스텀 출력 폴더
  python auto_collect.py --duration 5 --output my_data
        """
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='수집 시간 (분) [기본: 10]'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='초당 저장 프레임 수 [기본: 10]'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='collected_data',
        help='출력 폴더 [기본: collected_data]'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚗 CARLA Auto Data Collector")
    print("="*80)
    print(f"Duration: {args.duration} minutes")
    print(f"FPS: {args.fps}")
    print(f"Output: {args.output}/")
    print("="*80)
    
    # 수집 시작
    collector = CARLADataCollector(
        output_dir=args.output,
        fps=args.fps
    )
    
    collector.collect(duration_minutes=args.duration)


if __name__ == '__main__':
    main()
