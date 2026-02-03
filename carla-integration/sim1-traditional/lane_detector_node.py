"""
Lane Detector Node (Module 01 Integration)
"""
import sys
from pathlib import Path

# Add Module 01 to path
module01_path = Path(__file__).parent.parent.parent / '01-lane-detection'
sys.path.insert(0, str(module01_path))

import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple
import time

# Import Module 01 model (at module level, not inside __init__)
from src.models.deeplabv3plus import get_model


class LaneDetectorNode:
    """
    Module 01 wrapper for CARLA integration
    """
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        input_size: int = 320
    ):
        self.device = device
        self.input_size = input_size
        
        # Load Module 01 model (get_model already imported above)
        
        self.model = get_model(num_classes=2)
        
        if Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            print(f"✅ Lane Detection model loaded from {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
            print(f"   Using untrained model (for testing)")
        
        self.model.to(device)
        self.model.eval()
        
        print(f"✅ Lane Detector initialized ({device})")
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect lanes from RGB image
        
        Args:
            image: (H, W, 3) RGB image
        
        Returns:
            {
                'lane_mask': np.ndarray,
                'lane_polyline': List[Tuple[int, int]],
                'confidence': float,
                'lateral_offset': float,
                'heading_error': float,
                'processing_time': float
            }
        """
        start = time.time()
        
        # Preprocess
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (self.input_size, self.input_size))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Resize back to original
        mask = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Post-processing
        mask = self._postprocess(mask)
        
        # Extract polyline
        polyline = self._extract_polyline(mask)
        
        # Calculate metrics
        lateral_offset = self._calculate_lateral_offset(polyline, (h, w))
        heading_error = self._calculate_heading_error(polyline)
        
        processing_time = (time.time() - start) * 1000
        
        return {
            'lane_mask': mask,
            'lane_polyline': polyline,
            'confidence': 0.95,
            'lateral_offset': lateral_offset,
            'heading_error': heading_error,
            'processing_time': processing_time
        }
    
    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Morphological post-processing"""
        if mask.max() == 0:
            return mask
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        return mask
    
    def _extract_polyline(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Extract lane centerline"""
        if mask.sum() == 0:
            return []
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Approximate polyline
        epsilon = 0.01 * cv2.arcLength(largest, closed=False)
        polyline = cv2.approxPolyDP(largest, epsilon, closed=False)
        
        return [(int(p[0][0]), int(p[0][1])) for p in polyline]
    
    def _calculate_lateral_offset(
        self,
        polyline: List[Tuple[int, int]],
        image_shape: Tuple[int, int]
    ) -> float:
        """Calculate lateral offset (meters)"""
        if not polyline:
            return 0.0
        
        h, w = image_shape
        image_center_x = w / 2
        
        # Lane center (average x coordinate)
        lane_center_x = np.mean([p[0] for p in polyline])
        
        # Pixel offset
        pixel_offset = lane_center_x - image_center_x
        
        # Convert to meters (calibration factor)
        # Assume: 640 pixels ≈ 2 meters at bottom of image
        meters_per_pixel = 2.0 / w
        lateral_offset = pixel_offset * meters_per_pixel
        
        return float(lateral_offset)
    
    def _calculate_heading_error(
        self,
        polyline: List[Tuple[int, int]]
    ) -> float:
        """Calculate heading error (radians)"""
        if len(polyline) < 2:
            return 0.0
        
        # Use points at different heights
        # Bottom 20% and middle 50%
        sorted_poly = sorted(polyline, key=lambda p: p[1], reverse=True)
        
        if len(sorted_poly) < 2:
            return 0.0
        
        bottom_idx = 0
        middle_idx = min(len(sorted_poly) // 2, len(sorted_poly) - 1)
        
        bottom = sorted_poly[bottom_idx]
        middle = sorted_poly[middle_idx]
        
        # Calculate angle
        dx = middle[0] - bottom[0]
        dy = middle[1] - bottom[1]
        
        if dy == 0:
            return 0.0
        
        # Angle relative to vertical (image y-axis)
        angle = np.arctan2(dx, -dy)  # Negative dy because y increases downward
        
        return float(angle)
