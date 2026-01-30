"""
Object Detector: YOLOv8 기반 고정확도 객체 감지
정확도 우선 설정
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import torch
from pathlib import Path


class ObjectDetector:
    """
    YOLOv8 기반 객체 감지기
    
    정확도 우선 설정:
        - Model: YOLOv8l (Large) - 최고 정확도
        - Confidence: 0.25 (낮게 유지 → Recall 향상)
        - IoU NMS: 0.45 (중복 제거)
        - Image Size: 640×640 (고해상도)
    
    Classes:
        0: traffic_cone (트래픽 콘)
        1: obstacle (장애물)
        2: robot_car (RC 카)
        3: traffic_sign (교통 표지판)
        4: pedestrian (보행자 피규어)
    
    Attributes:
        model: YOLOv8 model instance
        device (str): 'cuda' or 'cpu'
        conf_thres (float): Confidence threshold
        iou_thres (float): NMS IoU threshold
        class_names (Dict[int, str]): Class ID to name mapping
    """
    
    def __init__(
        self,
        weights: str = 'yolov8l.pt',
        device: Optional[str] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        imgsz: int = 640
    ):
        """
        Parameters:
            weights: Model weights path
                    - 'yolov8l.pt': Pre-trained COCO weights
                    - Custom path: Fine-tuned weights
            device: Device for inference ('cuda' or 'cpu')
                   None = auto-detect
            conf_thres: Confidence threshold (0.0 ~ 1.0)
                       낮을수록 Recall 높아짐
            iou_thres: NMS IoU threshold (0.0 ~ 1.0)
                      높을수록 더 많은 박스 유지
            imgsz: Input image size (default: 640)
        
        Note:
            - 정확도 우선: conf_thres를 낮게 (0.25) 설정
            - False Positive 제어: post-processing에서 추가 필터링
        """
        # Device 설정
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        
        # YOLOv8 모델 로드
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            print(f"✅ Model loaded: {weights}")
            print(f"   Device: {self.device}")
            print(f"   Confidence threshold: {conf_thres}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # 클래스 이름
        self.class_names = {
            0: 'traffic_cone',
            1: 'obstacle',
            2: 'robot_car',
            3: 'traffic_sign',
            4: 'pedestrian'
        }
        
        # 클래스별 색상 (BGR for OpenCV)
        self.class_colors = {
            'traffic_cone': (0, 165, 255),   # Orange
            'obstacle': (0, 0, 255),         # Red
            'robot_car': (0, 255, 0),        # Green
            'traffic_sign': (255, 0, 0),     # Blue
            'pedestrian': (0, 255, 255)      # Yellow
        }
        
        # 통계
        self._inference_times = []
        self._detection_counts = []
    
    def detect(
        self,
        image: np.ndarray,
        return_image: bool = False,
        filter_small: bool = True,
        min_area: int = 100
    ) -> Dict:
        """
        단일 이미지 객체 감지
        
        Parameters:
            image: Input image (H×W×3), RGB format
            return_image: Return annotated image
            filter_small: Filter small boxes (< min_area px²)
            min_area: Minimum box area (pixels²)
        
        Returns:
            {
                "boxes": List[List[float]],       # [[x1, y1, x2, y2], ...]
                "classes": List[int],             # [0, 1, 2, ...]
                "confidences": List[float],       # [0.95, 0.87, ...]
                "class_names": List[str],         # ["cone", "obstacle", ...]
                "num_detections": int,            # Total detections
                "inference_time_ms": float,       # Inference time
                "image_annotated": Optional[np.ndarray]  # Annotated image
            }
        
        Example:
            >>> detector = ObjectDetector()
            >>> image = cv2.imread('test.jpg')
            >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            >>> result = detector.detect(image, return_image=True)
            >>> print(f"Detected {result['num_detections']} objects")
        """
        import time
        
        # 입력 검증
        if image is None or image.size == 0:
            return self._empty_result("Invalid input image")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            return self._empty_result(f"Invalid image shape: {image.shape}")
        
        start_time = time.time()
        
        # 추론
        try:
            results = self.model.predict(
                source=image,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                stream=False
            )
        except Exception as e:
            return self._empty_result(f"Inference error: {e}")
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # 결과 파싱
        if len(results) == 0:
            return self._empty_result("No results")
        
        result = results[0]  # Single image
        
        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.array([]).reshape(0, 4)
            classes = np.array([]).reshape(0)
            confidences = np.array([]).reshape(0)
        else:
            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
            confidences = result.boxes.conf.cpu().numpy()  # (N,)
        
        # Post-processing: 작은 박스 필터링
        if filter_small and len(boxes) > 0:
            boxes, classes, confidences = self._filter_small_boxes(
                boxes, classes, confidences, min_area=min_area
            )
        
        # 클래스 이름 매핑
        class_names = [self.class_names.get(cls, 'unknown') for cls in classes]
        
        # 통계 업데이트
        self._inference_times.append(inference_time)
        self._detection_counts.append(len(boxes))
        
        output = {
            "boxes": boxes.tolist(),
            "classes": classes.tolist(),
            "confidences": confidences.tolist(),
            "class_names": class_names,
            "num_detections": len(boxes),
            "inference_time_ms": inference_time
        }
        
        # Annotated image (선택적)
        if return_image:
            output["image_annotated"] = self._draw_boxes(
                image, boxes, classes, confidences, class_names
            )
        
        return output
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        filter_small: bool = True
    ) -> List[Dict]:
        """
        배치 이미지 처리
        
        Parameters:
            images: List of images (H×W×3)
            filter_small: Filter small boxes
        
        Returns:
            List of detection results
        """
        if len(images) == 0:
            return []
        
        try:
            results_list = self.model.predict(
                source=images,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                stream=False
            )
        except Exception as e:
            return [self._empty_result(f"Batch inference error: {e}") 
                    for _ in images]
        
        outputs = []
        for result in results_list:
            if result.boxes is None or len(result.boxes) == 0:
                boxes = np.array([]).reshape(0, 4)
                classes = np.array([]).reshape(0)
                confidences = np.array([]).reshape(0)
            else:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
            
            # Post-processing
            if filter_small and len(boxes) > 0:
                boxes, classes, confidences = self._filter_small_boxes(
                    boxes, classes, confidences
                )
            
            class_names = [self.class_names.get(cls, 'unknown') for cls in classes]
            
            outputs.append({
                "boxes": boxes.tolist(),
                "classes": classes.tolist(),
                "confidences": confidences.tolist(),
                "class_names": class_names,
                "num_detections": len(boxes)
            })
        
        return outputs
    
    def _filter_small_boxes(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        confidences: np.ndarray,
        min_area: float = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        작은 박스 필터링 (노이즈 제거)
        
        Parameters:
            boxes: (N, 4) [x1, y1, x2, y2]
            classes: (N,)
            confidences: (N,)
            min_area: Minimum box area (pixels²)
        
        Returns:
            Filtered boxes, classes, confidences
        """
        if len(boxes) == 0:
            return boxes, classes, confidences
        
        # 박스 면적 계산
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        # 필터링
        keep = areas >= min_area
        
        return boxes[keep], classes[keep], confidences[keep]
    
    def _draw_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        classes: np.ndarray,
        confidences: np.ndarray,
        class_names: List[str]
    ) -> np.ndarray:
        """
        이미지에 박스 그리기
        
        Parameters:
            image: Input image (RGB)
            boxes: (N, 4) bounding boxes
            classes: (N,) class IDs
            confidences: (N,) confidence scores
            class_names: List of class names
        
        Returns:
            Annotated image (RGB)
        """
        output = image.copy()
        
        for box, cls_name, conf in zip(boxes, class_names, confidences):
            x1, y1, x2, y2 = map(int, box)
            color = self.class_colors.get(cls_name, (128, 128, 128))
            
            # 박스 그리기 (두꺼운 선)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # 레이블 그리기
            label = f"{cls_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # 레이블 배경
            cv2.rectangle(output, (x1, y1 - label_h - baseline - 5), 
                         (x1 + label_w + 5, y1), color, -1)
            
            # 레이블 텍스트
            cv2.putText(output, label, (x1 + 2, y1 - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 감지 개수 표시
        info_text = f"Detections: {len(boxes)}"
        cv2.putText(output, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return output
    
    def _empty_result(self, reason: str = "") -> Dict:
        """빈 결과 반환"""
        return {
            "boxes": [],
            "classes": [],
            "confidences": [],
            "class_names": [],
            "num_detections": 0,
            "inference_time_ms": 0.0,
            "reason": reason
        }
    
    def get_performance_stats(self) -> Dict:
        """
        성능 통계 반환
        
        Returns:
            {
                "avg_inference_time_ms": float,
                "avg_fps": float,
                "avg_detections": float,
                "total_frames": int,
                "min_time_ms": float,
                "max_time_ms": float
            }
        """
        if len(self._inference_times) == 0:
            return {
                "avg_inference_time_ms": 0.0,
                "avg_fps": 0.0,
                "avg_detections": 0.0,
                "total_frames": 0
            }
        
        avg_time = np.mean(self._inference_times)
        avg_fps = 1000 / avg_time if avg_time > 0 else 0
        avg_det = np.mean(self._detection_counts)
        
        return {
            "avg_inference_time_ms": float(avg_time),
            "avg_fps": float(avg_fps),
            "avg_detections": float(avg_det),
            "total_frames": len(self._inference_times),
            "min_time_ms": float(min(self._inference_times)),
            "max_time_ms": float(max(self._inference_times))
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self._inference_times.clear()
        self._detection_counts.clear()
    
    def update_config(
        self,
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None
    ):
        """
        설정 업데이트
        
        Parameters:
            conf_thres: New confidence threshold
            iou_thres: New NMS IoU threshold
        """
        if conf_thres is not None:
            self.conf_thres = conf_thres
            print(f"✓ Confidence threshold updated: {conf_thres}")
        
        if iou_thres is not None:
            self.iou_thres = iou_thres
            print(f"✓ NMS IoU threshold updated: {iou_thres}")
    
    def export_onnx(self, output_path: str = 'model.onnx'):
        """
        ONNX 포맷으로 export
        
        Parameters:
            output_path: Output file path
        """
        try:
            self.model.export(format='onnx', imgsz=self.imgsz)
            print(f"✅ Model exported to ONNX: {output_path}")
        except Exception as e:
            print(f"❌ Export failed: {e}")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    두 박스의 IoU 계산
    
    Parameters:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        IoU: 0.0 ~ 1.0
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area
