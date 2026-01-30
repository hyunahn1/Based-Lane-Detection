"""
Lane Tracker: 차선 추적 및 차량 위치 계산
RC Car 환경에 최적화
"""
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2


class LaneTracker:
    """
    차선 중심선 추적 및 차량 위치 계산
    
    Workflow:
        1. Mask → Polyline 추출 (Skeleton + Contour)
        2. Polyline 스무딩 (Moving Average)
        3. 차량 위치에서 가장 가까운 점 찾기
        4. 횡방향 오프셋 계산 (픽셀 → 미터)
        5. 헤딩 추정 (Polyline 곡률 기반)
        6. 곡률 계산
    
    Attributes:
        smoothing_window (int): 스무딩 윈도우 크기
        min_confidence (float): 최소 신뢰도 임계값
        vehicle_position (Tuple[float, float]): 차량 중심 위치 (이미지 좌표)
        image_shape (Tuple[int, int]): 이미지 크기 (H, W)
    """
    
    def __init__(
        self,
        smoothing_window: int = 5,
        min_confidence: float = 0.6,
        vehicle_position: Optional[Tuple[float, float]] = None,
        image_shape: Tuple[int, int] = (480, 640),
        track_width_m: float = 0.35
    ):
        """
        Parameters:
            smoothing_window: Moving average window size
            min_confidence: Minimum confidence threshold
            vehicle_position: Vehicle center position in image coordinates
                             Default: (image_height * 0.9, image_width / 2)
            image_shape: Image dimensions (height, width)
            track_width_m: RC track width in meters
        """
        self.smoothing_window = smoothing_window
        self.min_confidence = min_confidence
        self.image_shape = image_shape
        self.track_width_m = track_width_m
        
        if vehicle_position is None:
            # 기본: 이미지 하단 중앙 (RC 카메라는 차량 전방 하단)
            self.vehicle_position = (
                image_shape[0] * 0.9,
                image_shape[1] / 2
            )
        else:
            self.vehicle_position = vehicle_position
        
        # 내부 상태
        self._history: List[Dict] = []
        self._max_history = 30
    
    def track(
        self,
        lane_mask: np.ndarray,
        confidence: float
    ) -> Dict:
        """
        차선 추적 수행 (Mask 기반)
        
        Parameters:
            lane_mask: Binary lane mask from Module 01 (H, W) {0, 1}
            confidence: Detection confidence (0.0 ~ 1.0)
        
        Returns:
            {
                "lane_center": List[Tuple[float, float]],  # Polyline
                "lateral_offset": float,  # meters
                "heading_error": float,   # degrees
                "curvature": float,       # 1/m
                "is_valid": bool,
                "confidence": float,
                "nearest_point": Tuple[float, float],
                "nearest_idx": int
            }
        """
        # 1. 유효성 검사
        if confidence < self.min_confidence:
            return self._invalid_result("Low confidence")
        
        # 2. Mask → Polyline 변환
        lane_polyline = self._extract_polyline_from_mask(lane_mask)
        
        if lane_polyline is None or len(lane_polyline) < 3:
            return self._invalid_result("Polyline extraction failed")
        
        # 3. 스무딩
        smoothed_polyline = self._smooth_polyline(lane_polyline)
        
        # 4. 차량 위치에서 가장 가까운 점 찾기
        nearest_point, nearest_idx = self._find_nearest_point(
            smoothed_polyline,
            self.vehicle_position
        )
        
        # 5. 횡방향 오프셋 계산 (픽셀 → 미터)
        lateral_offset_px = self._calculate_lateral_offset(
            self.vehicle_position,
            nearest_point,
            smoothed_polyline,
            nearest_idx
        )
        # 원근 보정을 위해 Y 좌표 전달
        lateral_offset_m = self._pixel_to_meter(
            lateral_offset_px,
            y_position=nearest_point[1]
        )
        
        # 6. 헤딩 추정값 계산 (IMU 없이 근사)
        heading_error = self._calculate_heading_error(
            smoothed_polyline,
            nearest_idx
        )
        
        # 7. 곡률 계산
        curvature = self._calculate_curvature(
            smoothed_polyline,
            nearest_idx
        )
        
        result = {
            "lane_center": smoothed_polyline,
            "lateral_offset": lateral_offset_m,
            "heading_error": heading_error,
            "curvature": curvature,
            "is_valid": True,
            "confidence": confidence,
            "nearest_point": nearest_point,
            "nearest_idx": nearest_idx
        }
        
        # 히스토리 업데이트
        self._update_history(result)
        
        return result
    
    def _extract_polyline_from_mask(
        self,
        mask: np.ndarray
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Binary mask에서 차선 중심선 polyline 추출
        
        Algorithm:
            1. Skeleton 추출 (Thinning)
            2. Contour 찾기
            3. 가장 긴 contour 선택
            4. 하단부터 상단으로 정렬
            5. Douglas-Peucker 근사로 점 개수 줄이기
        
        Parameters:
            mask: Binary mask (H, W) {0, 1}
        
        Returns:
            polyline: [(x, y), ...] or None
        """
        if mask.sum() == 0:
            return None
        
        # 1. Skeleton 추출 (Zhang-Suen thinning)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # ximgproc.thinning 사용 (opencv-contrib 필요)
        try:
            skeleton = cv2.ximgproc.thinning(mask_uint8)
        except AttributeError:
            # ximgproc 없으면 morphology로 대체
            kernel = np.ones((3, 3), np.uint8)
            skeleton = cv2.morphologyEx(mask_uint8, cv2.MORPH_THIN, kernel)
        
        # 2. Contours 찾기
        contours, _ = cv2.findContours(
            skeleton,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # 3. 가장 긴 contour 선택
        longest_contour = max(contours, key=cv2.contourArea)
        
        # 4. Contour → polyline
        polyline = longest_contour.squeeze().tolist()
        
        # 단일 점이면 실패
        if not isinstance(polyline, list) or len(polyline) == 0:
            return None
        
        if not isinstance(polyline[0], (list, tuple)):
            return None
        
        # 5. Y 좌표 기준 정렬 (하단 → 상단)
        polyline = sorted(polyline, key=lambda p: p[1], reverse=True)
        
        # 6. Douglas-Peucker 근사 (점 개수 줄이기)
        polyline_np = np.array(polyline, dtype=np.float32).reshape(-1, 1, 2)
        epsilon = 2.0  # 근사 정확도
        approx = cv2.approxPolyDP(polyline_np, epsilon, closed=False)
        polyline_approx = approx.squeeze().tolist()
        
        # 리스트 형식 보장
        if not isinstance(polyline_approx, list):
            polyline_approx = [polyline_approx]
        
        if not isinstance(polyline_approx[0], (list, tuple)):
            polyline_approx = [tuple(polyline_approx)]
        
        return [tuple(p) for p in polyline_approx]
    
    def _smooth_polyline(
        self,
        polyline: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Moving average로 polyline 스무딩
        """
        if len(polyline) < self.smoothing_window:
            return polyline
        
        smoothed = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(polyline)):
            start = max(0, i - half_window)
            end = min(len(polyline), i + half_window + 1)
            
            window = polyline[start:end]
            avg_x = np.mean([p[0] for p in window])
            avg_y = np.mean([p[1] for p in window])
            smoothed.append((avg_x, avg_y))
        
        return smoothed
    
    def _find_nearest_point(
        self,
        polyline: List[Tuple[float, float]],
        reference: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], int]:
        """
        Polyline에서 reference에 가장 가까운 점 찾기
        
        Returns:
            (nearest_point, index)
        """
        min_dist = float('inf')
        nearest_point = polyline[0]
        nearest_idx = 0
        
        for i, point in enumerate(polyline):
            dist = np.hypot(point[0] - reference[0], point[1] - reference[1])
            if dist < min_dist:
                min_dist = dist
                nearest_point = point
                nearest_idx = i
        
        return nearest_point, nearest_idx
    
    def _calculate_lateral_offset(
        self,
        vehicle_pos: Tuple[float, float],
        nearest_point: Tuple[float, float],
        polyline: List[Tuple[float, float]],
        nearest_idx: int
    ) -> float:
        """
        횡방향 오프셋 계산 (픽셀 단위)
        
        양수: 차선 중심 오른쪽
        음수: 차선 중심 왼쪽
        """
        # 벡터: nearest_point → vehicle_pos
        dx = vehicle_pos[0] - nearest_point[0]
        dy = vehicle_pos[1] - nearest_point[1]
        
        # 차선 방향 벡터 (tangent)
        if nearest_idx < len(polyline) - 1:
            next_point = polyline[nearest_idx + 1]
        else:
            if nearest_idx > 0:
                next_point = nearest_point
                nearest_point = polyline[nearest_idx - 1]
            else:
                return 0.0
        
        tx = next_point[0] - nearest_point[0]
        ty = next_point[1] - nearest_point[1]
        t_norm = np.hypot(tx, ty)
        
        if t_norm < 1e-6:
            return 0.0
        
        # 정규화
        tx /= t_norm
        ty /= t_norm
        
        # Cross product (2D)로 부호 결정
        # lateral_offset = dx * (-ty) + dy * tx
        lateral_offset = dx * (-ty) + dy * tx
        
        return lateral_offset
    
    def _calculate_heading_error(
        self,
        polyline: List[Tuple[float, float]],
        nearest_idx: int
    ) -> float:
        """
        헤딩 추정값 계산 (IMU 없이 근사)
        
        Note:
            - IMU/Gyro 없이는 정확한 차량 heading을 알 수 없음
            - Polyline의 곡률 변화율로 heading 변화 추정
            - 정확도: ±10도 오차 가능
        
        Method:
            연속된 polyline 세그먼트의 각도 변화로 추정
            - 직선: 각도 변화 작음 → heading_error ≈ 0
            - 좌커브: 각도 증가 → heading_error > 0
            - 우커브: 각도 감소 → heading_error < 0
        
        Returns:
            heading_estimate: 추정 헤딩 변화 (degrees)
        """
        # 경계 체크
        if nearest_idx < 2 or nearest_idx >= len(polyline) - 2:
            return 0.0
        
        # 이전 세그먼트 각도
        p_prev = polyline[nearest_idx - 2]
        p_curr = polyline[nearest_idx]
        angle_prev = np.arctan2(
            p_curr[1] - p_prev[1],
            p_curr[0] - p_prev[0]
        )
        
        # 다음 세그먼트 각도
        p_next = polyline[nearest_idx + 2]
        angle_next = np.arctan2(
            p_next[1] - p_curr[1],
            p_next[0] - p_curr[0]
        )
        
        # 각도 변화 (polyline의 곡률 방향)
        heading_change = angle_next - angle_prev
        
        # -π ~ π 범위로 정규화
        heading_change = np.arctan2(np.sin(heading_change), np.cos(heading_change))
        
        # degree로 변환
        heading_deg = np.degrees(heading_change)
        
        return heading_deg
    
    def _calculate_curvature(
        self,
        polyline: List[Tuple[float, float]],
        center_idx: int,
        window: int = 5
    ) -> float:
        """
        곡률 계산 (1/m)
        
        3점을 이용한 곡률 근사 (Menger curvature)
        """
        if center_idx < window or center_idx >= len(polyline) - window:
            return 0.0
        
        # 3개 점 선택
        p1 = polyline[center_idx - window]
        p2 = polyline[center_idx]
        p3 = polyline[center_idx + window]
        
        # 삼각형 변의 길이
        a = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        b = np.hypot(p3[0] - p2[0], p3[1] - p2[1])
        c = np.hypot(p3[0] - p1[0], p3[1] - p1[1])
        
        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            return 0.0
        
        # Heron's formula로 넓이 계산
        s = (a + b + c) / 2
        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
        
        # 곡률 = 4 * Area / (a * b * c)
        curvature_px = 4 * area / (a * b * c + 1e-10)
        
        # 픽셀 → 미터 변환 (center_idx 위치 기준)
        y_pos = p2[1]
        curvature_m = curvature_px * self._get_pixel_to_meter_ratio(y_pos)
        
        return curvature_m
    
    def _pixel_to_meter(self, pixel_value: float, y_position: float) -> float:
        """
        픽셀 값을 미터로 변환 (원근 보정 포함)
        
        Parameters:
            pixel_value: 픽셀 거리
            y_position: 이미지 내 Y 좌표 (원근 보정용)
        
        Returns:
            meter_value: 미터 거리 (근사값)
        
        Warning:
            - 카메라 캘리브레이션 없이 경험적 근사 사용
            - 정확도: ±30% 오차 가능
            - 권장: Camera calibration + IPM 사용
        """
        # RC 트랙 환경 기준
        image_height = self.image_shape[0]
        
        # 원근 보정 계수 (경험적)
        # 이미지 하단(가까움): scale = 1.0
        # 이미지 상단(멀리): scale = 3.0
        y_normalized = y_position / image_height
        perspective_scale = 1.0 + 2.0 * (1.0 - y_normalized)
        
        # 기준 픽셀-미터 비율 (이미지 하단 기준)
        # 가정: 이미지 하단에서 트랙이 이미지 폭의 60% 차지
        track_width_px_bottom = self.image_shape[1] * 0.6  # 384 pixels
        base_ratio = self.track_width_m / track_width_px_bottom  # 0.35 / 384
        
        # 원근 보정 적용
        ratio = base_ratio * perspective_scale
        
        return pixel_value * ratio
    
    def _get_pixel_to_meter_ratio(self, y_position: float = None) -> float:
        """
        픽셀-미터 변환 비율 (위치 의존적)
        
        Parameters:
            y_position: Y 좌표 (None이면 이미지 하단 기준)
        """
        if y_position is None:
            y_position = self.image_shape[0] * 0.9  # 기본: 하단
        
        y_normalized = y_position / self.image_shape[0]
        perspective_scale = 1.0 + 2.0 * (1.0 - y_normalized)
        
        track_width_px_bottom = self.image_shape[1] * 0.6
        base_ratio = self.track_width_m / track_width_px_bottom
        
        return base_ratio * perspective_scale
    
    def _invalid_result(self, reason: str) -> Dict:
        """Invalid 결과 반환"""
        return {
            "lane_center": [],
            "lateral_offset": 0.0,
            "heading_error": 0.0,
            "curvature": 0.0,
            "is_valid": False,
            "confidence": 0.0,
            "reason": reason
        }
    
    def _update_history(self, result: Dict):
        """히스토리 업데이트"""
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)
    
    def get_history(self) -> List[Dict]:
        """히스토리 반환"""
        return self._history.copy()
    
    def reset(self):
        """상태 초기화"""
        self._history.clear()
