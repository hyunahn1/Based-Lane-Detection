"""
ObjectDetector 단위 테스트
"""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import ObjectDetector, calculate_iou


class TestObjectDetector:
    """ObjectDetector 테스트 suite"""
    
    @pytest.fixture
    def detector(self):
        """기본 detector 인스턴스"""
        return ObjectDetector(
            weights='yolov8l.pt',
            device='cpu',  # CPU for testing
            conf_thres=0.25
        )
    
    @pytest.fixture
    def mock_image(self):
        """테스트용 가짜 이미지"""
        # 640×640×3 RGB 이미지
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    def test_initialization(self, detector):
        """초기화 테스트"""
        assert detector.model is not None
        assert detector.conf_thres == 0.25
        assert detector.iou_thres == 0.45
        assert len(detector.class_names) == 5
        
        print("\n✅ Initialization test passed")
    
    def test_empty_image(self, detector):
        """빈 이미지 처리"""
        empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect(empty_image)
        
        # 검증
        assert 'num_detections' in result
        assert result['num_detections'] >= 0
        
        print(f"\n✅ Empty image test: {result['num_detections']} detections")
    
    def test_invalid_input(self, detector):
        """잘못된 입력 처리"""
        # Wrong shape
        invalid_image = np.zeros((100, 100), dtype=np.uint8)
        result = detector.detect(invalid_image)
        
        # Should handle gracefully
        assert 'num_detections' in result or 'reason' in result
        
        print("\n✅ Invalid input handled gracefully")
    
    def test_detect_with_return_image(self, detector, mock_image):
        """이미지 반환 옵션 테스트"""
        result = detector.detect(mock_image, return_image=True)
        
        # 검증
        assert 'image_annotated' in result
        if result['image_annotated'] is not None:
            assert result['image_annotated'].shape == mock_image.shape
        
        print("\n✅ Return image test passed")
    
    def test_batch_detection(self, detector, mock_image):
        """배치 감지 테스트"""
        images = [mock_image, mock_image, mock_image]
        
        results = detector.detect_batch(images)
        
        # 검증
        assert len(results) == 3
        assert all('num_detections' in r for r in results)
        
        print(f"\n✅ Batch detection: {len(results)} images processed")
    
    def test_performance_stats(self, detector, mock_image):
        """성능 통계 테스트"""
        # 여러 프레임 처리
        for _ in range(5):
            detector.detect(mock_image)
        
        stats = detector.get_performance_stats()
        
        # 검증
        assert stats['total_frames'] == 5
        assert stats['avg_inference_time_ms'] > 0
        assert stats['avg_fps'] > 0
        
        print(f"\n✅ Performance stats:")
        print(f"   Avg time: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"   Avg FPS:  {stats['avg_fps']:.1f}")
    
    def test_config_update(self, detector):
        """설정 업데이트 테스트"""
        original_conf = detector.conf_thres
        
        detector.update_config(conf_thres=0.50)
        
        # 검증
        assert detector.conf_thres == 0.50
        assert detector.conf_thres != original_conf
        
        print(f"\n✅ Config update: {original_conf} → {detector.conf_thres}")
    
    def test_reset_stats(self, detector, mock_image):
        """통계 리셋 테스트"""
        # 프레임 처리
        detector.detect(mock_image)
        detector.detect(mock_image)
        
        # Reset
        detector.reset_stats()
        
        stats = detector.get_performance_stats()
        assert stats['total_frames'] == 0
        
        print("\n✅ Stats reset successful")


class TestUtilities:
    """유틸리티 함수 테스트"""
    
    def test_calculate_iou(self):
        """IoU 계산 테스트"""
        # 동일한 박스
        box1 = [100, 100, 200, 200]
        iou = calculate_iou(box1, box1)
        assert abs(iou - 1.0) < 1e-6
        
        # 겹치지 않는 박스
        box2 = [300, 300, 400, 400]
        iou = calculate_iou(box1, box2)
        assert abs(iou - 0.0) < 1e-6
        
        # 50% 겹침
        box3 = [150, 100, 250, 200]
        iou = calculate_iou(box1, box3)
        assert 0.3 < iou < 0.4  # ~33% IoU
        
        print("\n✅ IoU calculation tests passed")
        print(f"   Same boxes:      IoU = 1.0")
        print(f"   No overlap:      IoU = 0.0")
        print(f"   Partial overlap: IoU = {iou:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
