"""
LaneTracker 단위 테스트
"""
import pytest
import numpy as np
import sys
import os

# src 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tracking.lane_tracker import LaneTracker


class TestLaneTracker:
    """LaneTracker 테스트 suite"""
    
    @pytest.fixture
    def tracker(self):
        """기본 tracker 인스턴스"""
        return LaneTracker(image_shape=(480, 640))
    
    @pytest.fixture
    def straight_mask(self):
        """직선 차선 마스크 생성"""
        mask = np.zeros((480, 640), dtype=np.uint8)
        # 중앙 수직선
        mask[100:480, 315:325] = 1
        return mask
    
    @pytest.fixture
    def curved_mask(self):
        """곡선 차선 마스크 생성"""
        mask = np.zeros((480, 640), dtype=np.uint8)
        # 원형 곡선
        center = (320, 240)
        radius = 150
        for theta in np.linspace(-np.pi/2, np.pi/2, 100):
            x = int(center[0] + radius * np.cos(theta))
            y = int(center[1] + radius * np.sin(theta))
            if 0 <= y < 480 and 0 <= x < 640:
                mask[y-5:y+5, x-5:x+5] = 1
        return mask
    
    def test_initialization(self, tracker):
        """초기화 테스트"""
        assert tracker.smoothing_window == 5
        assert tracker.min_confidence == 0.6
        assert tracker.image_shape == (480, 640)
        assert len(tracker._history) == 0
    
    def test_track_low_confidence(self, tracker, straight_mask):
        """낮은 신뢰도 입력 처리"""
        result = tracker.track(straight_mask, confidence=0.3)
        
        # 검증
        assert result["is_valid"] == False
        assert "reason" in result
        assert result["reason"] == "Low confidence"
    
    def test_track_empty_mask(self, tracker):
        """빈 마스크 처리"""
        empty_mask = np.zeros((480, 640), dtype=np.uint8)
        result = tracker.track(empty_mask, confidence=0.9)
        
        # 검증
        assert result["is_valid"] == False
        assert "reason" in result
    
    def test_track_straight_lane(self, tracker, straight_mask):
        """직선 차선 추적"""
        result = tracker.track(straight_mask, confidence=0.95)
        
        # 검증
        assert result["is_valid"] == True
        assert abs(result["lateral_offset"]) < 0.03  # 3cm 이내
        assert abs(result["curvature"]) < 0.5  # 거의 직선
        
        print(f"\n✅ Straight Lane Test:")
        print(f"   Lateral Offset: {result['lateral_offset']*100:.2f} cm")
        print(f"   Heading Error:  {result['heading_error']:.2f}°")
        print(f"   Curvature:      {result['curvature']:.4f} m^-1")
    
    def test_track_curved_lane(self, tracker, curved_mask):
        """곡선 차선 추적"""
        result = tracker.track(curved_mask, confidence=0.9)
        
        # 검증
        if result["is_valid"]:
            assert result["curvature"] != 0.0  # 곡률 존재
            assert abs(result["curvature"]) < 10.0  # 합리적 범위
            
            print(f"\n✅ Curved Lane Test:")
            print(f"   Lateral Offset: {result['lateral_offset']*100:.2f} cm")
            print(f"   Heading Error:  {result['heading_error']:.2f}°")
            print(f"   Curvature:      {result['curvature']:.4f} m^-1")
    
    def test_offset_left_lane(self, tracker):
        """왼쪽 치우친 차선 (차량은 오른쪽 이탈)"""
        mask = np.zeros((480, 640), dtype=np.uint8)
        # 왼쪽으로 이동한 차선
        mask[100:480, 250:260] = 1
        
        result = tracker.track(mask, confidence=0.95)
        
        if result["is_valid"]:
            # 차선이 왼쪽 → 차량은 상대적으로 오른쪽
            assert result["lateral_offset"] > 0
            
            print(f"\n✅ Left Offset Test:")
            print(f"   Lateral Offset: {result['lateral_offset']*100:.2f} cm (positive = right)")
    
    def test_offset_right_lane(self, tracker):
        """오른쪽 치우친 차선 (차량은 왼쪽 이탈)"""
        mask = np.zeros((480, 640), dtype=np.uint8)
        # 오른쪽으로 이동한 차선
        mask[100:480, 380:390] = 1
        
        result = tracker.track(mask, confidence=0.95)
        
        if result["is_valid"]:
            # 차선이 오른쪽 → 차량은 상대적으로 왼쪽
            assert result["lateral_offset"] < 0
            
            print(f"\n✅ Right Offset Test:")
            print(f"   Lateral Offset: {result['lateral_offset']*100:.2f} cm (negative = left)")
    
    def test_history_management(self, tracker, straight_mask):
        """히스토리 관리 테스트"""
        # 여러 프레임 처리
        for _ in range(5):
            tracker.track(straight_mask, confidence=0.9)
        
        history = tracker.get_history()
        assert len(history) == 5
        
        # Reset 테스트
        tracker.reset()
        assert len(tracker.get_history()) == 0
    
    def test_pixel_to_meter_conversion(self, tracker):
        """픽셀-미터 변환 테스트"""
        # 하단 (가까움): 작은 scale
        ratio_bottom = tracker._get_pixel_to_meter_ratio(y_position=432)
        
        # 상단 (멀리): 큰 scale
        ratio_top = tracker._get_pixel_to_meter_ratio(y_position=100)
        
        # 원근 효과: 상단 > 하단
        assert ratio_top > ratio_bottom
        
        print(f"\n✅ Perspective Correction:")
        print(f"   Ratio (bottom): {ratio_bottom:.6f} m/px")
        print(f"   Ratio (top):    {ratio_top:.6f} m/px")
        print(f"   Scale factor:   {ratio_top/ratio_bottom:.2f}x")


def test_integration_with_mock_module01():
    """Module 01 Mock 출력과 통합 테스트"""
    # Module 01 출력 시뮬레이션
    lane_mask = np.zeros((480, 640), dtype=np.uint8)
    lane_mask[100:480, 315:325] = 1
    
    # LaneTracker 처리
    tracker = LaneTracker()
    result = tracker.track(lane_mask, confidence=0.95)
    
    # 검증
    assert result["is_valid"] == True
    assert "lateral_offset" in result
    assert "heading_error" in result
    assert "curvature" in result
    
    print(f"\n✅ Integration Test:")
    print(f"   Status:         {'Valid' if result['is_valid'] else 'Invalid'}")
    print(f"   Lateral Offset: {result['lateral_offset']*100:.2f} cm")
    print(f"   Confidence:     {result['confidence']:.2f}")


if __name__ == "__main__":
    # 직접 실행 시
    pytest.main([__file__, "-v", "-s"])
