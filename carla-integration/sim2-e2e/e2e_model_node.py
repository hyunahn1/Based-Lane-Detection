"""
E2E Model Node (Module 06 Integration)
End-to-End: Image → Control
"""
import sys
from pathlib import Path

# Add Module 06 to path
module06_path = Path(__file__).parent.parent.parent / '06-end-to-end-learning'
sys.path.insert(0, str(module06_path))

import torch
import cv2
import numpy as np
from typing import Dict
import time
from torchvision import transforms


class E2EModelNode:
    """
    Module 06 wrapper for CARLA integration
    End-to-End: Image → Control
    """
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        img_size: int = 224
    ):
        self.device = device
        self.img_size = img_size
        
        # Load Module 06 model
        from src.models.e2e_model import EndToEndModel
        
        self.model = EndToEndModel(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12
        )
        
        if Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            print(f"✅ E2E model loaded from {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
            print(f"   Using untrained model (for interface testing)")
        
        self.model.to(device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"✅ E2E Model initialized ({device})")
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        End-to-End prediction: Image → Control
        
        Args:
            image: (H, W, 3) RGB image
        
        Returns:
            {
                'steering': float,  # -1 to +1
                'throttle': float,  # 0 to 1
                'processing_time': float
            }
        """
        start = time.time()
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            control = self.model(image_tensor)
            steering, throttle = control[0]
        
        processing_time = (time.time() - start) * 1000
        
        return {
            'steering': float(steering.cpu().numpy()),
            'throttle': float(throttle.cpu().numpy()),
            'processing_time': processing_time
        }
