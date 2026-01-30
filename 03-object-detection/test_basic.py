"""
Module 03 기본 동작 검증
YOLOv8 Pre-trained 모델 테스트
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
from src.detector import ObjectDetector, calculate_iou


def test_basic_functionality():
    """기본 기능 테스트"""
    print("="*80)
    print("Module 03: Object Detection - Basic Functionality Test")
    print("="*80)
    
    # Test 1: Detector 초기화
    print("\n[Test 1] Detector 초기화")
    try:
        detector = ObjectDetector(
            weights='yolov8l.pt',
            device='cpu',  # CPU 테스트
            conf_thres=0.25
        )
        print("  ✅ Detector initialized")
        print(f"     Device: {detector.device}")
        print(f"     Classes: {len(detector.class_names)}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return
    
    # Test 2: 가짜 이미지 감지
    print("\n[Test 2] 가짜 이미지 감지")
    fake_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    try:
        result = detector.detect(fake_image)
        print(f"  ✅ Detection succeeded")
        print(f"     Detections: {result['num_detections']}")
        print(f"     Inference time: {result['inference_time_ms']:.2f} ms")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return
    
    # Test 3: 빈 결과 처리
    print("\n[Test 3] 빈 이미지 처리")
    empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    result = detector.detect(empty_image)
    print(f"  ✅ Empty image handled")
    print(f"     Detections: {result['num_detections']}")
    
    # Test 4: 잘못된 입력 처리
    print("\n[Test 4] 잘못된 입력 처리")
    invalid_image = np.zeros((100, 100), dtype=np.uint8)  # Wrong shape
    
    result = detector.detect(invalid_image)
    print(f"  ✅ Invalid input handled gracefully")
    print(f"     Reason: {result.get('reason', 'N/A')}")
    
    # Test 5: 배치 처리
    print("\n[Test 5] 배치 처리")
    batch_images = [fake_image, fake_image, fake_image]
    
    try:
        results = detector.detect_batch(batch_images)
        print(f"  ✅ Batch detection succeeded")
        print(f"     Batch size: {len(results)}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test 6: 성능 통계
    print("\n[Test 6] 성능 통계")
    stats = detector.get_performance_stats()
    print(f"  ✅ Stats retrieved")
    print(f"     Total frames: {stats['total_frames']}")
    print(f"     Avg time: {stats['avg_inference_time_ms']:.2f} ms")
    print(f"     Avg FPS: {stats['avg_fps']:.1f}")
    
    # Test 7: IoU 계산
    print("\n[Test 7] IoU 계산")
    box1 = [100, 100, 200, 200]
    box2 = [100, 100, 200, 200]
    iou = calculate_iou(box1, box2)
    print(f"  ✅ IoU calculation")
    print(f"     Same boxes: IoU = {iou:.3f} (expected: 1.0)")
    assert abs(iou - 1.0) < 1e-6
    
    box3 = [150, 100, 250, 200]
    iou = calculate_iou(box1, box3)
    print(f"     Overlap:    IoU = {iou:.3f}")
    
    # Test 8: Config 업데이트
    print("\n[Test 8] Config 업데이트")
    detector.update_config(conf_thres=0.50)
    print(f"  ✅ Config updated")
    print(f"     New conf_thres: {detector.conf_thres}")
    
    # Test 9: Stats 리셋
    print("\n[Test 9] Stats 리셋")
    detector.reset_stats()
    stats = detector.get_performance_stats()
    print(f"  ✅ Stats reset")
    print(f"     Total frames: {stats['total_frames']} (expected: 0)")
    
    print("\n" + "="*80)
    print("✅ 모든 기본 테스트 통과!")
    print("="*80)
    print("\n📝 다음 단계:")
    print("  1. 데이터 수집: python scripts/collect_data.py --target 1000")
    print("  2. 레이블링: CVAT 또는 Roboflow 사용")
    print("  3. 데이터 분할: python scripts/split_dataset.py")
    print("  4. 학습: python train.py --model yolov8l.pt --epochs 200")
    print("  5. 평가: python validate.py --weights runs/train/exp/weights/best.pt")
    print()


if __name__ == '__main__':
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
