"""
Lane Keeping Assist System (LKAS) - Main Orchestrator
모든 컴포넌트를 통합하는 메인 클래스
"""
from typing import Dict, Optional
import time
import numpy as np

from .tracking.lane_tracker import LaneTracker
from .detection.departure_detector import DepartureDetector, DepartureThresholds
from .control.pid_controller import PIDController, PIDParams
from .alert.warning_system import WarningSystem


class LaneKeepingAssist:
    """
    LKAS 메인 오케스트레이터
    
    컴포넌트 통합:
        1. LaneTracker: 차선 추적
        2. DepartureDetector: 이탈 감지
        3. WarningSystem: 경고 시스템
        4. PIDController: 조향 제어
    """
    
    def __init__(
        self,
        image_shape: tuple = (480, 640),
        track_width: float = 0.35,
        enable_control: bool = True,
        enable_warning: bool = True,
        intervention_threshold: int = 4,  # Level 4 이상에서 개입
        pid_params: Optional[PIDParams] = None,
        departure_thresholds: Optional[DepartureThresholds] = None
    ):
        """
        Parameters:
            image_shape: 이미지 크기 (H, W)
            track_width: RC 트랙 폭 (meters)
            enable_control: PID 제어 활성화
            enable_warning: 경고 시스템 활성화
            intervention_threshold: 개입 시작 레벨
            pid_params: PID 파라미터
            departure_thresholds: 이탈 임계값
        """
        # 컴포넌트 초기화
        self.tracker = LaneTracker(
            image_shape=image_shape,
            track_width_m=track_width
        )
        
        self.detector = DepartureDetector(
            thresholds=departure_thresholds,
            track_width=track_width
        )
        
        self.controller = PIDController(
            params=pid_params
        )
        
        self.warning_system = WarningSystem(
            enable_visual=enable_warning,
            enable_audio=False  # 기본 false
        )
        
        # 설정
        self.enable_control = enable_control
        self.enable_warning = enable_warning
        self.intervention_threshold = intervention_threshold
        
        # 상태
        self._is_intervening = False
        self._intervention_start_time = None
        self._frame_count = 0
    
    def process_frame(
        self,
        lane_detection: Dict,  # Module 01 출력
        vehicle_state: Dict
    ) -> Dict:
        """
        전체 파이프라인 실행
        
        Parameters:
            lane_detection:
                {
                    "lane_mask": np.ndarray,  # (H, W) binary mask
                    "confidence": float       # 0.0 ~ 1.0
                }
            vehicle_state:
                {
                    "speed": float,     # km/h (RC car speed)
                    "timestamp": float  # optional, Unix timestamp
                }
        
        Returns:
            {
                "steering_angle": float,        # 조향각 (degree)
                "throttle_adjustment": float,   # 스로틀 조정 (-1.0 ~ 1.0)
                "warning_level": int,           # 경고 레벨 (0-5)
                "is_intervening": bool,         # 개입 여부
                "lateral_offset": float,        # 횡방향 오프셋 (m)
                "heading_error": float,         # 헤딩 오차 (degree)
                "curvature": float,             # 곡률 (1/m)
                "is_valid": bool,               # 추적 유효성
                "timestamp": float
            }
        """
        self._frame_count += 1
        timestamp = vehicle_state.get("timestamp", time.time())
        
        # 1. Lane Tracking
        track_result = self.tracker.track(
            lane_mask=lane_detection["lane_mask"],
            confidence=lane_detection["confidence"]
        )
        
        # 추적 실패 시
        if not track_result["is_valid"]:
            return self._safe_fallback(timestamp)
        
        # 2. Departure Detection
        departure_result = self.detector.detect(
            lateral_offset=track_result["lateral_offset"],
            heading_error=track_result["heading_error"],
            vehicle_speed=vehicle_state["speed"],
            timestamp=timestamp
        )
        
        risk_level = departure_result["risk_level"]
        
        # 3. Warning System Update
        if self.enable_warning:
            self.warning_system.update(
                risk_level=risk_level,
                departure_side=departure_result["departure_side"],
                timestamp=timestamp
            )
        
        # 4. Control Decision
        steering_angle = 0.0
        throttle_adjustment = 0.0
        is_intervening = False
        
        if self.enable_control and risk_level >= self.intervention_threshold:
            # PID 제어 개입
            steering_angle = self.controller.compute(
                lateral_offset=track_result["lateral_offset"],
                heading_estimate=track_result["heading_error"],
                curvature=track_result["curvature"]
            )
            
            is_intervening = True
            
            # 긴급 상황 (Level 5)에서는 감속
            if risk_level == 5:
                throttle_adjustment = -0.3  # 30% 감속
            
            # 개입 시작 시각 기록
            if not self._is_intervening:
                self._intervention_start_time = timestamp
        else:
            # 개입하지 않음
            self.controller.reset()  # PID 상태 초기화
            self._intervention_start_time = None
        
        self._is_intervening = is_intervening
        
        # 5. 결과 반환
        result = {
            "steering_angle": steering_angle,
            "throttle_adjustment": throttle_adjustment,
            "warning_level": risk_level,
            "is_intervening": is_intervening,
            "lateral_offset": track_result["lateral_offset"],
            "heading_error": track_result["heading_error"],
            "curvature": track_result["curvature"],
            "time_to_crossing": departure_result["time_to_crossing"],
            "departure_side": departure_result["departure_side"],
            "is_valid": True,
            "timestamp": timestamp,
            "frame_count": self._frame_count
        }
        
        return result
    
    def render_warning(
        self,
        frame: np.ndarray,
        lkas_output: Dict
    ) -> np.ndarray:
        """
        경고 오버레이 렌더링
        
        Parameters:
            frame: 입력 이미지
            lkas_output: process_frame() 출력
        
        Returns:
            output_frame: 경고가 오버레이된 이미지
        """
        if not self.enable_warning:
            return frame
        
        return self.warning_system.render_visual_warning(
            frame=frame,
            lateral_offset=lkas_output.get("lateral_offset"),
            ttc=lkas_output.get("time_to_crossing")
        )
    
    def _safe_fallback(self, timestamp: float) -> Dict:
        """
        안전 모드 (추적 실패 시)
        
        개입 중단, 모든 컴포넌트 리셋
        """
        self.controller.reset()
        self.warning_system.reset()
        
        return {
            "steering_angle": 0.0,
            "throttle_adjustment": 0.0,
            "warning_level": 0,
            "is_intervening": False,
            "lateral_offset": 0.0,
            "heading_error": 0.0,
            "curvature": 0.0,
            "time_to_crossing": float('inf'),
            "departure_side": "none",
            "is_valid": False,
            "timestamp": timestamp,
            "frame_count": self._frame_count,
            "reason": "Tracking failed - safe fallback"
        }
    
    def reset(self):
        """전체 시스템 리셋"""
        self.tracker.reset()
        self.detector.reset()
        self.controller.reset()
        self.warning_system.reset()
        
        self._is_intervening = False
        self._intervention_start_time = None
        self._frame_count = 0
    
    def update_pid_params(self, **kwargs):
        """PID 파라미터 동적 업데이트"""
        self.controller.update_params(**kwargs)
    
    def get_telemetry(self) -> Dict:
        """실시간 텔레메트리 데이터"""
        return {
            "is_intervening": self._is_intervening,
            "intervention_start_time": self._intervention_start_time,
            "frame_count": self._frame_count,
            "tracker_history_length": len(self.tracker.get_history()),
            "controller_integral": self.controller._integral,
            "warning_level": self.warning_system.get_warning_level()
        }
