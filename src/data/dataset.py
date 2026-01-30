"""
Lane Segmentation Dataset for PyTorch
"""
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class LaneDataset(Dataset):
    """
    차선 Segmentation Dataset
    
    COCO 포맷의 어노테이션을 읽어서 binary mask 생성
    """
    
    def __init__(self, coco_json_path, image_dir, transform=None, target_size=(480, 640)):
        """
        Parameters:
        -----------
        coco_json_path : str
            COCO 포맷 JSON 경로
        image_dir : str
            이미지 디렉토리
        transform : callable, optional
            Albumentations transform
        target_size : tuple
            출력 이미지 크기 (H, W)
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_size = target_size
        
        # COCO JSON 로드
        with open(coco_json_path, 'r') as f:
            self.coco = json.load(f)
        
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        
        # image_id → annotations 매핑
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        print(f"✅ Loaded {len(self.images)} images, {len(self.annotations)} annotations")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
        --------
        image : torch.Tensor, (3, H, W), float32, [0, 1]
        mask : torch.Tensor, (H, W), long, {0, 1}
        """
        # 이미지 정보
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        
        # 이미지 로드
        img_path = self.image_dir / img_filename
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 마스크 생성
        height, width = img_info['height'], img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 어노테이션이 있으면 폴리곤 채우기
        if img_id in self.img_to_anns:
            for ann in self.img_to_anns[img_id]:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)  # 1 = lane
        
        # 리사이즈
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 정규화 및 텐서 변환
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, 3) → (3, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)
        
        return image, mask


def get_train_transform():
    """학습용 데이터 증강"""
    import albumentations as A
    
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.8
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.0,  # 차선 색상 유지
            p=0.8
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
    ])


def get_val_transform():
    """검증용 (증강 없음)"""
    return None
