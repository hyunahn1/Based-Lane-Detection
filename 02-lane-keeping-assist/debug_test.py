#!/usr/bin/env python3
"""
디버깅 스크립트 - polyline 추출 및 offset 계산 확인
"""
import sys
import numpy as np
import cv2

sys.path.insert(0, 'src')
from tracking.lane_tracker import LaneTracker


def create_straight_mask():
    """직선 차선 마스크"""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:480, 315:325] = 1  # X=315~325, 중앙 320
    return mask


def debug_tracking():
    """단계별 디버깅"""
    print("="*60)
    print("DEBUGGING: LaneTracker")
    print("="*60)
    
    tracker = LaneTracker()
    mask = create_straight_mask()
    
    # 1. Mask 확인
    print(f"\n1. Mask shape: {mask.shape}")
    print(f"   Non-zero pixels: {mask.sum()}")
    print(f"   X range: {np.where(mask)[1].min()} ~ {np.where(mask)[1].max()}")
    print(f"   Y range: {np.where(mask)[0].min()} ~ {np.where(mask)[0].max()}")
    
    # 2. Polyline 추출
    polyline = tracker._extract_polyline_from_mask(mask)
    print(f"\n2. Polyline extraction:")
    print(f"   Success: {polyline is not None}")
    if polyline:
        print(f"   Points: {len(polyline)}")
        print(f"   First 3: {polyline[:3]}")
        print(f"   Last 3: {polyline[-3:]}")
        
        # X 좌표 범위
        x_coords = [p[0] for p in polyline]
        print(f"   X range: {min(x_coords):.1f} ~ {max(x_coords):.1f}")
        print(f"   X mean: {np.mean(x_coords):.1f}")
    
    # 3. 차량 위치
    print(f"\n3. Vehicle position: {tracker.vehicle_position}")
    
    # 4. 가장 가까운 점
    if polyline:
        nearest_point, nearest_idx = tracker._find_nearest_point(
            polyline, tracker.vehicle_position
        )
        print(f"\n4. Nearest point: {nearest_point} (idx={nearest_idx})")
        
        # 5. Lateral offset (픽셀)
        smoothed = tracker._smooth_polyline(polyline)
        lateral_offset_px = tracker._calculate_lateral_offset(
            tracker.vehicle_position,
            nearest_point,
            smoothed,
            nearest_idx
        )
        print(f"\n5. Lateral offset (pixels): {lateral_offset_px:.2f}")
        
        # 6. Pixel to meter
        lateral_offset_m = tracker._pixel_to_meter(
            lateral_offset_px,
            y_position=nearest_point[1]
        )
        print(f"   Lateral offset (meters): {lateral_offset_m:.4f} m = {lateral_offset_m*100:.2f} cm")
        
        # 7. 픽셀-미터 비율
        ratio = tracker._get_pixel_to_meter_ratio(y_position=nearest_point[1])
        print(f"   Pixel-to-meter ratio: {ratio:.6f} m/px")
        
        # 8. 예상값 계산
        print(f"\n8. Expected values:")
        print(f"   Vehicle X: {tracker.vehicle_position[1]}")
        print(f"   Lane center X (mean): {np.mean(x_coords):.1f}")
        print(f"   Expected offset (pixels): ~{tracker.vehicle_position[1] - np.mean(x_coords):.1f}")
        print(f"   Expected offset (meters): ~{(tracker.vehicle_position[1] - np.mean(x_coords)) * ratio:.4f} m")


if __name__ == "__main__":
    debug_tracking()
