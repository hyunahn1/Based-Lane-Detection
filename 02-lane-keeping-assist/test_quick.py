#!/usr/bin/env python3
"""
LaneTracker 빠른 검증 스크립트 (pytest 불필요)
"""
import sys
import numpy as np

# src 경로 추가
sys.path.insert(0, 'src')

from tracking.lane_tracker import LaneTracker


def create_straight_mask():
    """직선 차선 마스크 생성"""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:480, 315:325] = 1
    return mask


def create_left_offset_mask():
    """왼쪽으로 치우친 차선"""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:480, 250:260] = 1
    return mask


def create_right_offset_mask():
    """오른쪽으로 치우친 차선"""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:480, 380:390] = 1
    return mask


def test_straight_lane():
    """Test 1: 직선 차선 추적"""
    print("\n" + "="*60)
    print("TEST 1: 직선 차선 추적")
    print("="*60)
    
    tracker = LaneTracker()
    mask = create_straight_mask()
    result = tracker.track(mask, confidence=0.95)
    
    print(f"✅ Valid:          {result['is_valid']}")
    print(f"📏 Lateral Offset: {result['lateral_offset']*100:+.2f} cm")
    print(f"🧭 Heading Error:  {result['heading_error']:+.2f}°")
    print(f"↪️  Curvature:      {result['curvature']:.4f} m^-1")
    print(f"🎯 Confidence:     {result['confidence']:.2f}")
    
    # 검증
    assert result['is_valid'], "❌ Tracking failed!"
    assert abs(result['lateral_offset']) < 0.05, f"❌ Offset too large: {result['lateral_offset']*100:.2f} cm"
    print("\n✅ PASSED - Straight lane tracking works!")
    
    return result


def test_left_offset():
    """Test 2: 왼쪽 치우친 차선"""
    print("\n" + "="*60)
    print("TEST 2: 왼쪽 치우친 차선 (차량은 오른쪽 이탈)")
    print("="*60)
    
    tracker = LaneTracker()
    mask = create_left_offset_mask()
    result = tracker.track(mask, confidence=0.95)
    
    print(f"✅ Valid:          {result['is_valid']}")
    print(f"📏 Lateral Offset: {result['lateral_offset']*100:+.2f} cm")
    print(f"🧭 Heading Error:  {result['heading_error']:+.2f}°")
    
    # 검증
    assert result['is_valid'], "❌ Tracking failed!"
    assert result['lateral_offset'] > 0, f"❌ Should be positive (right): {result['lateral_offset']*100:.2f} cm"
    print(f"\n✅ PASSED - Correctly detected RIGHT offset: {result['lateral_offset']*100:.2f} cm")
    
    return result


def test_right_offset():
    """Test 3: 오른쪽 치우친 차선"""
    print("\n" + "="*60)
    print("TEST 3: 오른쪽 치우친 차선 (차량은 왼쪽 이탈)")
    print("="*60)
    
    tracker = LaneTracker()
    mask = create_right_offset_mask()
    result = tracker.track(mask, confidence=0.95)
    
    print(f"✅ Valid:          {result['is_valid']}")
    print(f"📏 Lateral Offset: {result['lateral_offset']*100:+.2f} cm")
    print(f"🧭 Heading Error:  {result['heading_error']:+.2f}°")
    
    # 검证
    assert result['is_valid'], "❌ Tracking failed!"
    assert result['lateral_offset'] < 0, f"❌ Should be negative (left): {result['lateral_offset']*100:.2f} cm"
    print(f"\n✅ PASSED - Correctly detected LEFT offset: {result['lateral_offset']*100:.2f} cm")
    
    return result


def test_low_confidence():
    """Test 4: 낮은 신뢰도"""
    print("\n" + "="*60)
    print("TEST 4: 낮은 신뢰도 입력 (Fail-safe)")
    print("="*60)
    
    tracker = LaneTracker()
    mask = create_straight_mask()
    result = tracker.track(mask, confidence=0.3)
    
    print(f"❌ Valid:   {result['is_valid']}")
    print(f"📝 Reason:  {result.get('reason', 'N/A')}")
    
    # 검증
    assert not result['is_valid'], "❌ Should reject low confidence!"
    assert result['reason'] == "Low confidence", "❌ Wrong reason!"
    print("\n✅ PASSED - Correctly rejected low confidence input!")
    
    return result


def test_perspective_correction():
    """Test 5: 원근 보정"""
    print("\n" + "="*60)
    print("TEST 5: 원근 보정 (Perspective Correction)")
    print("="*60)
    
    tracker = LaneTracker()
    
    # 하단 (가까움)
    ratio_bottom = tracker._get_pixel_to_meter_ratio(y_position=432)
    
    # 상단 (멀리)
    ratio_top = tracker._get_pixel_to_meter_ratio(y_position=100)
    
    scale_factor = ratio_top / ratio_bottom
    
    print(f"📏 Ratio (bottom): {ratio_bottom:.6f} m/px")
    print(f"📏 Ratio (top):    {ratio_top:.6f} m/px")
    print(f"📊 Scale factor:   {scale_factor:.2f}x")
    
    # 검증
    assert ratio_top > ratio_bottom, "❌ Perspective correction not working!"
    assert 2.0 < scale_factor < 4.0, f"❌ Scale factor unrealistic: {scale_factor:.2f}x"
    print("\n✅ PASSED - Perspective correction working correctly!")


def main():
    """모든 테스트 실행"""
    print("\n" + "🚗"*30)
    print("   LANE TRACKER - QUICK VALIDATION")
    print("🚗"*30)
    
    try:
        # Test 1: 직선
        test_straight_lane()
        
        # Test 2: 왼쪽 오프셋
        test_left_offset()
        
        # Test 3: 오른쪽 오프셋
        test_right_offset()
        
        # Test 4: 낮은 신뢰도
        test_low_confidence()
        
        # Test 5: 원근 보정
        test_perspective_correction()
        
        # 최종 결과
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\n✅ LaneTracker 구현이 성공적으로 검증되었습니다!")
        print("✅ Module 01 출력(mask)을 받아서 차량 위치를 정확히 추적합니다!")
        print("✅ 원근 보정이 적용되어 위치에 따라 다른 변환 비율을 사용합니다!")
        print("\n📝 다음 단계: DepartureDetector, PIDController 구현")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
