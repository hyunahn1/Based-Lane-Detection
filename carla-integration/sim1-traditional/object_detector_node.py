"""
Object Detector Node (Module 03 Integration)
"""
import sys
from pathlib import Path

# Add Module 03 to path
module03_path = Path(__file__).parent.parent.parent / '03-object-detection'
sys.path.insert(0, str(module03_path))

import cv2
import numpy as np
from typing import Dict, List, Tuple
import time


class ObjectDetectorNode:
    """
    Module 03 wrapper for CARLA integration
    """
    def __init__(
        self,
        model_name: str = 'yolov8l',
        conf_threshold: float = 0.5,
        device: str = 'cuda'
    ):
        try:
            from ultralytics import YOLO
            
            # Load pre-trained YOLO (COCO weights for demo)
            self.model = YOLO(f'{model_name}.pt')
            self.conf_threshold = conf_threshold
            self.device = device
            
            print(f"✅ Object Detection model loaded ({model_name})")
        except Exception as e:
            print(f"⚠️ YOLO load failed: {e}")
            print(f"   Object detection will be skipped")
            self.model = None
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect objects in image
        
        Args:
            image: (H, W, 3) RGB image
        
        Returns:
            {
                'objects': List[Dict],
                'num_objects': int,
                'collision_risk': bool,
                'closest_distance': float,
                'processing_time': float
            }
        """
        start = time.time()
        
        if self.model is None:
            return {
                'objects': [],
                'num_objects': 0,
                'collision_risk': False,
                'closest_distance': float('inf'),
                'processing_time': 0.0
            }
        
        # Inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        objects = []
        for r in results:
            for box in r.boxes:
                objects.append({
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1,y1,x2,y2]
                    'center': self._get_bbox_center(box.xyxy[0].cpu().numpy())
                })
        
        # Collision risk assessment
        collision_risk, closest_distance = self._assess_collision_risk(
            objects, image.shape
        )
        
        processing_time = (time.time() - start) * 1000
        
        return {
            'objects': objects,
            'num_objects': len(objects),
            'collision_risk': collision_risk,
            'closest_distance': closest_distance,
            'processing_time': processing_time
        }
    
    def _get_bbox_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Get bounding box center"""
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) / 2), float((y1 + y2) / 2))
    
    def _assess_collision_risk(
        self,
        objects: List[Dict],
        image_shape: Tuple[int, int, int]
    ) -> Tuple[bool, float]:
        """
        Assess collision risk based on object positions
        
        Logic:
            - Objects in bottom 30% of image = close = risky
            - Objects in center = on path = risky
        """
        if not objects:
            return False, float('inf')
        
        h, w = image_shape[:2]
        
        # Risk zone: bottom 30%, center 50% (width)
        risk_y_threshold = h * 0.7
        center_x = w / 2
        risk_x_range = w * 0.25  # ±25% from center
        
        closest_distance = float('inf')
        collision_risk = False
        
        for obj in objects:
            bbox = obj['bbox']
            bottom_y = bbox[3]  # y2
            center_x_obj = (bbox[0] + bbox[2]) / 2
            
            # Distance to bottom (proxy for actual distance)
            distance = h - bottom_y
            closest_distance = min(closest_distance, distance)
            
            # Check if in risk zone
            in_bottom = bottom_y > risk_y_threshold
            in_center = abs(center_x_obj - center_x) < risk_x_range
            
            if in_bottom and in_center:
                collision_risk = True
        
        return collision_risk, float(closest_distance)
