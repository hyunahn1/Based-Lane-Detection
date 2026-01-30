"""
Warning System: 위험도 기반 다단계 경고 시스템
RC Car 환경에 최적화
"""
from typing import Optional, Tuple
import numpy as np
import cv2


class WarningSystem:
    """
    위험도 기반 다단계 경고 시스템
    
    경고 타입:
        - Visual: OpenCV로 화면에 경고 표시
        - Audio: 비프음 (선택적, 구현 간단히)
        - Haptic: 미구현 (PiRacer 하드웨어 없음)
    
    위험도별 경고:
        Level 0-1: 경고 없음
        Level 2:   시각 경고 (노란색)
        Level 3:   시각 + 청각 (주황색 + 비프음 1회)
        Level 4:   시각 + 청각 반복 (빨간색 + 비프음 2회)
        Level 5:   전체 화면 경고 (깜빡임 + 연속 경보음)
    """
    
    def __init__(
        self,
        enable_visual: bool = True,
        enable_audio: bool = False,  # 기본 false (하드웨어 없음)
        blink_interval: float = 0.5  # seconds
    ):
        """
        Parameters:
            enable_visual: 시각 경고 활성화
            enable_audio: 청각 경고 활성화
            blink_interval: 깜빡임 간격
        """
        self.enable_visual = enable_visual
        self.enable_audio = enable_audio
        self.blink_interval = blink_interval
        
        # 내부 상태
        self._current_risk_level = 0
        self._departure_side = "none"
        self._last_blink_time = 0.0
        self._blink_state = False
        
        # 색상 정의 (BGR)
        self._colors = {
            0: (0, 255, 0),       # Green (Safe)
            1: (0, 255, 255),     # Yellow-Green (Normal)
            2: (0, 255, 255),     # Yellow (Caution)
            3: (0, 165, 255),     # Orange (Warning)
            4: (0, 0, 255),       # Red (Critical)
            5: (0, 0, 255)        # Red (Emergency)
        }
    
    def update(
        self,
        risk_level: int,
        departure_side: str,
        timestamp: Optional[float] = None
    ):
        """
        위험도 업데이트
        
        Parameters:
            risk_level: 위험도 레벨 (0-5)
            departure_side: 이탈 방향 ("left", "right", "none")
            timestamp: 현재 시각 (선택적)
        """
        self._current_risk_level = risk_level
        self._departure_side = departure_side
        
        # 깜빡임 상태 업데이트
        if timestamp is not None:
            if timestamp - self._last_blink_time > self.blink_interval:
                self._blink_state = not self._blink_state
                self._last_blink_time = timestamp
        
        # 청각 경고 트리거
        if self.enable_audio:
            self._trigger_audio_warning(risk_level)
    
    def render_visual_warning(
        self,
        frame: np.ndarray,
        lateral_offset: Optional[float] = None,
        ttc: Optional[float] = None
    ) -> np.ndarray:
        """
        프레임에 경고 오버레이
        
        Parameters:
            frame: 입력 이미지 (H, W, 3)
            lateral_offset: 횡방향 오프셋 (meters, 선택적)
            ttc: Time To Crossing (seconds, 선택적)
        
        Returns:
            output_frame: 경고가 오버레이된 이미지
        """
        if not self.enable_visual:
            return frame
        
        output = frame.copy()
        h, w = output.shape[:2]
        
        risk_level = self._current_risk_level
        
        # Level 0-1: 경고 없음
        if risk_level <= 1:
            return output
        
        # Level 2: 노란색 테두리
        if risk_level == 2:
            cv2.rectangle(output, (10, 10), (w-10, h-10), self._colors[2], 5)
            self._draw_text(output, "CAUTION", (w//2, 50), self._colors[2])
        
        # Level 3: 주황색 테두리 + 방향 표시
        elif risk_level == 3:
            cv2.rectangle(output, (10, 10), (w-10, h-10), self._colors[3], 8)
            self._draw_text(output, "WARNING", (w//2, 50), self._colors[3], scale=1.2)
            self._draw_direction_arrow(output, self._departure_side)
        
        # Level 4: 빨간색 + 깜빡임
        elif risk_level == 4:
            if self._blink_state:
                cv2.rectangle(output, (5, 5), (w-5, h-5), self._colors[4], 12)
                self._draw_text(output, "CRITICAL!", (w//2, 50), self._colors[4], scale=1.5)
            self._draw_direction_arrow(output, self._departure_side)
        
        # Level 5: 전체 화면 경고
        elif risk_level == 5:
            if self._blink_state:
                # 반투명 빨간색 오버레이
                overlay = output.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), self._colors[5], -1)
                output = cv2.addWeighted(output, 0.5, overlay, 0.5, 0)
            
            self._draw_text(output, "LANE DEPARTURE!", (w//2, h//2), 
                          (255, 255, 255), scale=2.0, thickness=3)
            self._draw_direction_arrow(output, self._departure_side)
        
        # 추가 정보 표시
        if lateral_offset is not None:
            info_text = f"Offset: {lateral_offset*100:.1f}cm"
            self._draw_text(output, info_text, (20, h-60), (255, 255, 255), scale=0.6)
        
        if ttc is not None and ttc < 10.0:
            ttc_text = f"TTC: {ttc:.2f}s"
            self._draw_text(output, ttc_text, (20, h-30), (255, 255, 255), scale=0.6)
        
        # Risk level indicator
        self._draw_risk_indicator(output)
        
        return output
    
    def _draw_text(
        self,
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        scale: float = 1.0,
        thickness: int = 2
    ):
        """텍스트 그리기 (중앙 정렬)"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        
        x = position[0] - text_w // 2
        y = position[1] + text_h // 2
        
        # 외곽선 (검은색)
        cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness+2)
        # 텍스트
        cv2.putText(img, text, (x, y), font, scale, color, thickness)
    
    def _draw_direction_arrow(
        self,
        img: np.ndarray,
        direction: str
    ):
        """이탈 방향 화살표 그리기"""
        h, w = img.shape[:2]
        
        if direction == "left":
            # 왼쪽 화살표
            cv2.arrowedLine(img, (w//2, h-100), (w//2-100, h-100), 
                          (0, 0, 255), 8, tipLength=0.3)
        elif direction == "right":
            # 오른쪽 화살표
            cv2.arrowedLine(img, (w//2, h-100), (w//2+100, h-100), 
                          (0, 0, 255), 8, tipLength=0.3)
    
    def _draw_risk_indicator(self, img: np.ndarray):
        """위험도 인디케이터 (우측 상단)"""
        h, w = img.shape[:2]
        
        # 5단계 바
        bar_width = 40
        bar_height = 20
        start_x = w - 60
        start_y = 20
        
        for i in range(5):
            level = i + 1
            y = start_y + i * (bar_height + 5)
            
            if level <= self._current_risk_level:
                color = self._colors[level]
            else:
                color = (100, 100, 100)  # Gray
            
            cv2.rectangle(img, (start_x, y), 
                        (start_x + bar_width, y + bar_height), 
                        color, -1)
            cv2.rectangle(img, (start_x, y), 
                        (start_x + bar_width, y + bar_height), 
                        (255, 255, 255), 2)
    
    def _trigger_audio_warning(self, risk_level: int):
        """
        청각 경고 트리거
        
        Note:
            실제 하드웨어가 없으므로 print로 시뮬레이션
            실제 구현 시 buzzer/speaker 제어 코드로 대체
        """
        if risk_level >= 3 and risk_level <= 5:
            # 비프음 횟수
            beeps = risk_level - 2
            print(f"🔊 BEEP! " * beeps)
    
    def get_warning_level(self) -> int:
        """현재 경고 레벨 반환"""
        return self._current_risk_level
    
    def reset(self):
        """상태 초기화"""
        self._current_risk_level = 0
        self._departure_side = "none"
        self._blink_state = False
