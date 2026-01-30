"""
PID Controller: PID 기반 조향 제어기
RC Car 환경에 최적화
"""
from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PIDParams:
    """PID 파라미터 (RC Car 초기 추정값)"""
    kp: float = 2.0  # 강한 비례 제어 (작은 스케일)
    ki: float = 0.2
    kd: float = 0.5  # 강한 미분 (빠른 변화 대응)
    k_heading: float = 0.2  # heading 근사값이므로 낮은 가중치
    
    max_steering_angle: float = 45.0  # degrees (RC 서보 범위)
    max_steering_rate: float = 100.0  # deg/s (RC 서보는 빠름)
    
    windup_limit: float = 5.0  # 작은 스케일에 맞춤
    
    wheelbase: float = 0.25  # meters (PiRacer)


class PIDController:
    """
    PID 기반 조향 제어기
    
    Control Law:
        u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt + FF
        
        where:
            e(t) = lateral_offset + K_heading * heading_error
            FF = arctan(wheelbase * curvature)
    
    Note:
        heading_estimate는 IMU 없이 polyline 변화율로 추정하므로
        정확도가 낮음 (±10도 오차). k_heading을 낮게 설정.
    """
    
    def __init__(self, params: PIDParams = None, dt: float = 0.05):
        """
        Parameters:
            params: PID 파라미터
            dt: 제어 주기 (seconds)
        """
        self.params = params or PIDParams()
        self.dt = dt
        
        # 내부 상태
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_steering = 0.0
    
    def compute(
        self,
        lateral_offset: float,
        heading_estimate: float,
        curvature: float = 0.0
    ) -> float:
        """
        제어 신호 계산
        
        Parameters:
            lateral_offset: 횡방향 오프셋 (meters)
            heading_estimate: 헤딩 추정값 (degrees, IMU 없이 근사)
            curvature: 차선 곡률 (1/m)
        
        Returns:
            steering_angle: 조향각 (degrees)
        """
        # 1. 통합 에러 계산
        # heading_estimate는 부정확하므로 낮은 가중치 적용
        error = lateral_offset + self.params.k_heading * heading_estimate
        
        # 2. PID 항 계산
        # P term
        p_term = self.params.kp * error
        
        # I term (with anti-windup)
        self._integral += error * self.dt
        self._integral = np.clip(
            self._integral,
            -self.params.windup_limit,
            self.params.windup_limit
        )
        i_term = self.params.ki * self._integral
        
        # D term
        derivative = (error - self._prev_error) / self.dt
        d_term = self.params.kd * derivative
        
        # 3. Feedforward term (곡률 보상)
        # Bicycle model: tan(δ) = L * κ (δ = steering, L = wheelbase, κ = curvature)
        if abs(curvature) < 1e-6:
            ff_term = 0.0
        else:
            ff_term = np.degrees(np.arctan(self.params.wheelbase * curvature))
            # RC카 스케일에서 너무 극단적인 값 방지
            ff_term = np.clip(ff_term, -15.0, 15.0)
        
        # 4. 총 제어 신호
        steering_raw = p_term + i_term + d_term + ff_term
        
        # 5. 제약 조건 적용
        # 최대 조향각 제한
        steering_clamped = np.clip(
            steering_raw,
            -self.params.max_steering_angle,
            self.params.max_steering_angle
        )
        
        # 최대 조향 속도 제한 (rate limiting)
        steering_rate = (steering_clamped - self._prev_steering) / self.dt
        if abs(steering_rate) > self.params.max_steering_rate:
            steering_rate = np.sign(steering_rate) * self.params.max_steering_rate
            steering_clamped = self._prev_steering + steering_rate * self.dt
        
        # 6. 상태 업데이트
        self._prev_error = error
        self._prev_steering = steering_clamped
        
        return steering_clamped
    
    def reset(self):
        """제어기 상태 초기화"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_steering = 0.0
    
    def update_params(self, **kwargs):
        """파라미터 동적 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
