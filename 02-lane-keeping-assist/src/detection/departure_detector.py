"""
Departure Detector: 차선 이탈 감지 및 위험도 평가
RC Car 환경에 최적화
"""
from typing import Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class DepartureThresholds:
    """이탈 판정 임계값 (RC 트랙 기준)"""
    level_2_offset: float = 0.08  # meters (8cm)
    level_3_offset: float = 0.12  # 12cm
    level_4_offset: float = 0.15  # 15cm
    level_5_offset: float = 0.18  # 18cm (트랙 경계)
    
    level_2_heading: float = 10.0  # degrees
    level_3_heading: float = 20.0
    level_4_heading: float = 30.0
    level_5_heading: float = 40.0


class DepartureDetector:
    """
    차선 이탈 감지 및 위험도 평가 (RC Car)
    
    Risk Levels:
        0: Safe - 정상 주행 (< 5cm offset)
        1: Normal - 약간 벗어남 (5-8cm), 모니터링
        2: Caution - 주의 필요 (8-12cm), 시각 경고
        3: Warning - 경고 필요 (12-15cm), 청각 경고
        4: Critical - 위험 (15-18cm), 개입 준비
        5: Emergency - 긴급 (> 18cm), 즉시 개입
    
    Note:
        RC 트랙 폭 = 35cm이므로 18cm 이탈 = 트랙 경계 도달
    """
    
    def __init__(
        self,
        thresholds: DepartureThresholds = None,
        track_width: float = 0.35  # RC track width in meters
    ):
        """
        Parameters:
            thresholds: 이탈 판정 임계값
            track_width: 트랙 폭 (meters)
        """
        self.thresholds = thresholds or DepartureThresholds()
        self.track_width = track_width
        
        # 상태
        self._prev_risk_level = 0
        self._risk_start_time = None
    
    def detect(
        self,
        lateral_offset: float,
        heading_error: float,
        vehicle_speed: float,
        timestamp: float
    ) -> Dict:
        """
        이탈 감지 수행
        
        Parameters:
            lateral_offset: 횡방향 오프셋 (meters)
            heading_error: 헤딩 오차 (degrees)
            vehicle_speed: 차량 속도 (km/h)
            timestamp: 현재 시각 (Unix timestamp)
        
        Returns:
            {
                "is_departing": bool,
                "risk_level": int,  # 0-5
                "time_to_crossing": float,  # seconds
                "departure_side": str,  # "left", "right", "none"
                "confidence": float
            }
        """
        # 1. 위험도 레벨 결정
        risk_level = self._calculate_risk_level(
            abs(lateral_offset),
            abs(heading_error)
        )
        
        # 2. 이탈 방향
        departure_side = self._determine_side(lateral_offset)
        
        # 3. Time To Crossing 계산
        ttc = self._calculate_ttc(
            lateral_offset,
            heading_error,
            vehicle_speed
        )
        
        # 4. 이탈 여부
        is_departing = risk_level >= 2
        
        # 5. 신뢰도 (간단한 휴리스틱)
        confidence = min(1.0, risk_level / 5.0)
        
        result = {
            "is_departing": is_departing,
            "risk_level": risk_level,
            "time_to_crossing": ttc,
            "departure_side": departure_side,
            "confidence": confidence,
            "timestamp": timestamp
        }
        
        # 상태 업데이트
        self._update_state(risk_level, timestamp)
        
        return result
    
    def _calculate_risk_level(
        self,
        abs_offset: float,
        abs_heading: float
    ) -> int:
        """
        위험도 레벨 계산
        
        offset과 heading 중 더 높은 레벨 선택
        """
        # Offset 기반 레벨
        if abs_offset >= self.thresholds.level_5_offset:
            offset_level = 5
        elif abs_offset >= self.thresholds.level_4_offset:
            offset_level = 4
        elif abs_offset >= self.thresholds.level_3_offset:
            offset_level = 3
        elif abs_offset >= self.thresholds.level_2_offset:
            offset_level = 2
        elif abs_offset >= 0.05:  # 5cm
            offset_level = 1
        else:
            offset_level = 0
        
        # Heading 기반 레벨
        if abs_heading >= self.thresholds.level_5_heading:
            heading_level = 5
        elif abs_heading >= self.thresholds.level_4_heading:
            heading_level = 4
        elif abs_heading >= self.thresholds.level_3_heading:
            heading_level = 3
        elif abs_heading >= self.thresholds.level_2_heading:
            heading_level = 2
        elif abs_heading >= 5.0:  # 5도
            heading_level = 1
        else:
            heading_level = 0
        
        # 최대값 선택
        return max(offset_level, heading_level)
    
    def _determine_side(self, lateral_offset: float) -> str:
        """이탈 방향 결정"""
        if lateral_offset > 0.01:  # 1cm 임계값
            return "right"
        elif lateral_offset < -0.01:
            return "left"
        else:
            return "none"
    
    def _calculate_ttc(
        self,
        lateral_offset: float,
        heading_error: float,
        vehicle_speed: float
    ) -> float:
        """
        Time To Crossing 계산
        
        차선 경계까지 도달하는 시간 추정
        """
        # 속도가 너무 낮으면 TTC 무한대
        if vehicle_speed < 5.0:  # 5 km/h
            return float('inf')
        
        # 트랙 경계까지 남은 거리
        remaining_distance = (self.track_width / 2) - abs(lateral_offset)
        
        if remaining_distance <= 0:
            return 0.0
        
        # 횡방향 속도 추정 (간단한 근사)
        # lateral_velocity ≈ vehicle_speed * sin(heading_error)
        vehicle_speed_ms = vehicle_speed / 3.6  # km/h → m/s
        heading_rad = np.radians(heading_error)
        lateral_velocity = abs(vehicle_speed_ms * np.sin(heading_rad))
        
        if lateral_velocity < 0.01:
            return float('inf')
        
        ttc = remaining_distance / lateral_velocity
        
        return min(ttc, 10.0)  # 최대 10초로 제한
    
    def _update_state(self, risk_level: int, timestamp: float):
        """내부 상태 업데이트"""
        if risk_level != self._prev_risk_level:
            self._risk_start_time = timestamp
        
        self._prev_risk_level = risk_level
    
    def reset(self):
        """상태 초기화"""
        self._prev_risk_level = 0
        self._risk_start_time = None
