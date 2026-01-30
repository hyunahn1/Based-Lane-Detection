"""
DepartureDetector 단위 테스트
"""
import pytest
import time
import sys
import os

# src 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detection.departure_detector import DepartureDetector, DepartureThresholds


class TestDepartureDetector:
    """DepartureDetector 테스트 suite"""
    
    @pytest.fixture
    def detector(self):
        """기본 detector 인스턴스"""
        return DepartureDetector()
    
    @pytest.fixture
    def custom_detector(self):
        """커스텀 임계값 detector"""
        thresholds = DepartureThresholds(
            level_2_offset=0.10,
            level_3_offset=0.15,
            level_4_offset=0.20,
            level_5_offset=0.25
        )
        return DepartureDetector(thresholds=thresholds)
    
    def test_initialization(self, detector):
        """초기화 테스트"""
        assert detector.track_width == 0.35
        assert detector._prev_risk_level == 0
        assert detector._risk_start_time is None
    
    def test_safe_driving(self, detector):
        """Test Case 5: 안전 주행"""
        # 작은 오프셋 (3cm)
        result = detector.detect(
            lateral_offset=0.03,  # 3cm
            heading_error=2.0,     # 2도
            vehicle_speed=20.0,    # 20 km/h
            timestamp=time.time()
        )
        
        # 검증
        assert result["is_departing"] == False
        assert result["risk_level"] == 0
        assert result["departure_side"] == "right"
        
        print(f"\n✅ Test Case 5: Safe Driving")
        print(f"   Offset:     {0.03*100:.1f} cm")
        print(f"   Heading:    {2.0:.1f}°")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Departing:  {result['is_departing']}")
    
    def test_caution_level(self, detector):
        """Test Case 6: 경고 레벨 (13cm offset)"""
        result = detector.detect(
            lateral_offset=0.13,  # 13cm
            heading_error=8.0,    # 8도
            vehicle_speed=25.0,   # 25 km/h
            timestamp=time.time()
        )
        
        # 검증
        assert result["is_departing"] == True
        assert result["risk_level"] == 3  # Level 3 (Warning)
        assert result["departure_side"] == "right"
        assert result["time_to_crossing"] < float('inf')
        
        print(f"\n✅ Test Case 6: Caution Level")
        print(f"   Offset:     {0.13*100:.1f} cm")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   TTC:        {result['time_to_crossing']:.2f} s")
    
    def test_emergency_situation(self, detector):
        """Test Case 7: 긴급 상황 (19cm offset)"""
        result = detector.detect(
            lateral_offset=0.19,  # 19cm (트랙 경계 초과)
            heading_error=35.0,   # 35도
            vehicle_speed=30.0,   # 30 km/h
            timestamp=time.time()
        )
        
        # 검증
        assert result["is_departing"] == True
        assert result["risk_level"] == 5  # Level 5 (Emergency)
        assert result["departure_side"] == "right"
        
        print(f"\n✅ Test Case 7: Emergency")
        print(f"   Offset:     {0.19*100:.1f} cm")
        print(f"   Heading:    {35.0:.1f}°")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   TTC:        {result['time_to_crossing']:.2f} s")
    
    def test_left_departure(self, detector):
        """왼쪽 이탈"""
        result = detector.detect(
            lateral_offset=-0.10,  # -10cm (왼쪽)
            heading_error=-15.0,   # -15도
            vehicle_speed=20.0,
            timestamp=time.time()
        )
        
        # 검증
        assert result["departure_side"] == "left"
        assert result["risk_level"] >= 2
        
        print(f"\n✅ Left Departure Test")
        print(f"   Offset:     {-0.10*100:.1f} cm (left)")
        print(f"   Risk Level: {result['risk_level']}")
    
    def test_heading_dominant_risk(self, detector):
        """헤딩이 주도하는 위험도"""
        # 작은 오프셋이지만 큰 헤딩 에러
        result = detector.detect(
            lateral_offset=0.05,   # 5cm (Level 1)
            heading_error=35.0,    # 35도 (Level 5)
            vehicle_speed=25.0,
            timestamp=time.time()
        )
        
        # 검증: 더 높은 레벨 선택
        assert result["risk_level"] == 5
        
        print(f"\n✅ Heading Dominant Risk")
        print(f"   Offset:     {0.05*100:.1f} cm → Level 1")
        print(f"   Heading:    {35.0:.1f}° → Level 5")
        print(f"   Final:      Level {result['risk_level']} (max)")
    
    def test_ttc_low_speed(self, detector):
        """낮은 속도에서 TTC"""
        # 매우 낮은 속도
        result = detector.detect(
            lateral_offset=0.10,
            heading_error=10.0,
            vehicle_speed=3.0,  # 3 km/h (매우 느림)
            timestamp=time.time()
        )
        
        # 검증: TTC가 무한대
        assert result["time_to_crossing"] == float('inf')
        
        print(f"\n✅ TTC at Low Speed")
        print(f"   Speed:      {3.0:.1f} km/h")
        print(f"   TTC:        inf (too slow)")
    
    def test_ttc_boundary_crossing(self, detector):
        """경계 도달 시 TTC"""
        # 이미 경계를 넘어섬
        result = detector.detect(
            lateral_offset=0.20,  # 20cm (트랙 폭 초과)
            heading_error=20.0,
            vehicle_speed=25.0,
            timestamp=time.time()
        )
        
        # 검증: TTC = 0
        assert result["time_to_crossing"] == 0.0
        
        print(f"\n✅ TTC at Boundary")
        print(f"   Offset:     {0.20*100:.1f} cm (beyond track)")
        print(f"   TTC:        {result['time_to_crossing']:.2f} s")
    
    def test_state_management(self, detector):
        """상태 관리 테스트"""
        timestamp = time.time()
        
        # 첫 번째 감지 (Level 2)
        result1 = detector.detect(0.10, 5.0, 20.0, timestamp)
        assert detector._prev_risk_level == result1["risk_level"]
        
        # 두 번째 감지 (Level 3)
        result2 = detector.detect(0.14, 10.0, 20.0, timestamp + 1.0)
        assert detector._prev_risk_level == result2["risk_level"]
        
        # Reset
        detector.reset()
        assert detector._prev_risk_level == 0
        assert detector._risk_start_time is None
        
        print(f"\n✅ State Management")
        print(f"   Level 1:    {result1['risk_level']}")
        print(f"   Level 2:    {result2['risk_level']}")
        print(f"   After Reset: 0")
    
    def test_custom_thresholds(self, custom_detector):
        """커스텀 임계값 테스트"""
        result = custom_detector.detect(
            lateral_offset=0.12,  # 12cm
            heading_error=8.0,
            vehicle_speed=20.0,
            timestamp=time.time()
        )
        
        # 기본 임계값이면 Level 3이지만,
        # 커스텀 임계값(level_2=10cm, level_3=15cm)이면 Level 2
        assert result["risk_level"] == 2
        
        print(f"\n✅ Custom Thresholds")
        print(f"   Offset:     {0.12*100:.1f} cm")
        print(f"   Threshold:  Level 2 (10-15cm)")
        print(f"   Risk Level: {result['risk_level']}")


def test_integration_ttc_calculation():
    """TTC 계산 통합 테스트"""
    detector = DepartureDetector()
    
    # 시나리오: 10cm 오프셋, 15도 헤딩, 36 km/h (10 m/s)
    result = detector.detect(
        lateral_offset=0.10,  # 10cm
        heading_error=15.0,   # 15도
        vehicle_speed=36.0,   # 36 km/h = 10 m/s
        timestamp=time.time()
    )
    
    # 검증
    # 남은 거리 = (0.35/2) - 0.10 = 0.075 m
    # 횡방향 속도 = 10 * sin(15°) ≈ 2.59 m/s
    # TTC = 0.075 / 2.59 ≈ 0.029 s
    assert result["time_to_crossing"] > 0
    assert result["time_to_crossing"] < 1.0  # 1초 이내
    
    print(f"\n✅ TTC Integration Test")
    print(f"   Offset:     10 cm")
    print(f"   Heading:    15°")
    print(f"   Speed:      36 km/h")
    print(f"   TTC:        {result['time_to_crossing']:.3f} s")
    print(f"   Expected:   ~0.029 s")


if __name__ == "__main__":
    # 직접 실행 시
    pytest.main([__file__, "-v", "-s"])
