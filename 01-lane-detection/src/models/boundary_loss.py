"""
Boundary-Aware Loss Functions
경계 강조 손실 함수

기존 문제:
    - Dice/CE Loss는 전체 영역에 균등한 가중치
    - 차선 경계 (중요!)와 중심 (덜 중요)을 동등하게 취급
    
해결:
    - 경계 픽셀에 더 높은 가중치 부여
    - 경계가 정확할수록 navigation 성능 향상
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class BoundaryLoss(nn.Module):
    """
    Boundary-Aware Loss
    
    논문 기반: 
        - "Boundary Loss" (Kervadec et al., 2019)
        - "Boundary-Aware Segmentation" (Cheng et al., 2020)
    
    핵심 아이디어:
        1. Ground truth에서 경계 추출 (morphological gradient)
        2. 경계 영역에 높은 가중치 (10x)
        3. CE Loss와 결합
    
    예상 효과:
        - Boundary IoU +5% 향상
        - 더 정확한 polyline 추출
    """
    def __init__(self, boundary_weight: float = 10.0, kernel_size: int = 5):
        """
        Parameters:
            boundary_weight: Weight for boundary pixels (default: 10x)
            kernel_size: Kernel size for boundary extraction
        """
        super(BoundaryLoss, self).__init__()
        
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        
        # CE loss for base
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        경계 추출 (Morphological Gradient)
        
        Parameters:
            mask: (B, H, W) binary mask
        
        Returns:
            boundary: (B, H, W) boundary mask
        """
        B, H, W = mask.shape
        device = mask.device
        
        # Convert to numpy for cv2 operations
        boundaries = []
        
        for i in range(B):
            mask_np = mask[i].cpu().numpy().astype(np.uint8)
            
            # Morphological gradient = dilation - erosion
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.kernel_size, self.kernel_size)
            )
            
            dilated = cv2.dilate(mask_np, kernel, iterations=1)
            eroded = cv2.erode(mask_np, kernel, iterations=1)
            
            boundary = dilated - eroded
            boundaries.append(torch.from_numpy(boundary).to(device))
        
        return torch.stack(boundaries, dim=0).float()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary-aware loss
        
        Parameters:
            pred: (B, C, H, W) predictions (logits)
            target: (B, H, W) ground truth (class indices)
        
        Returns:
            Weighted loss (scalar)
        """
        # Extract boundaries from target
        boundary_mask = self.extract_boundary(target)  # (B, H, W)
        
        # Compute per-pixel CE loss
        ce = self.ce_loss(pred, target)  # (B, H, W)
        
        # Weight map: boundary pixels get higher weight
        weight_map = torch.ones_like(boundary_mask)
        weight_map[boundary_mask > 0] = self.boundary_weight
        
        # Weighted loss
        weighted_loss = ce * weight_map
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Loss: CE + Dice + Boundary
    
    최종 손실 함수:
        Loss = λ1 * CE + λ2 * Dice + λ3 * Boundary
             = 1.0 * CE + 3.0 * Dice + 2.0 * Boundary
    
    개선 포인트:
        - Boundary Loss 추가 (연구 기여!)
        - 경계 정확도 향상
        - 기존 손실 함수들과 시너지
    """
    def __init__(
        self, 
        ce_weight: float = 1.0,
        dice_weight: float = 3.0,
        boundary_weight: float = 2.0,
        boundary_pixel_weight: float = 10.0
    ):
        """
        Parameters:
            ce_weight: Weight for CrossEntropy loss
            dice_weight: Weight for Dice loss
            boundary_weight: Weight for Boundary loss
            boundary_pixel_weight: Weight for boundary pixels
        """
        super(CombinedLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        # Individual losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss(boundary_weight=boundary_pixel_weight)
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> dict:
        """
        Compute combined loss
        
        Parameters:
            pred: (B, C, H, W) predictions
            target: (B, H, W) ground truth
        
        Returns:
            Dictionary with total loss and components
        """
        # Individual losses
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        # Combined
        total = (
            self.ce_weight * ce +
            self.dice_weight * dice +
            self.boundary_weight * boundary
        )
        
        return {
            'total': total,
            'ce': ce.item(),
            'dice': dice.item(),
            'boundary': boundary.item()
        }


class DiceLoss(nn.Module):
    """Dice Loss (from original implementation)"""
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        
        Returns:
            Dice loss (scalar)
        """
        # Softmax
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=pred.size(1))  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Flatten
        pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)  # (B, C, H*W)
        target_one_hot = target_one_hot.contiguous().view(
            target_one_hot.size(0), target_one_hot.size(1), -1
        )  # (B, C, H*W)
        
        # Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=2)  # (B, C)
        cardinality = pred.sum(dim=2) + target_one_hot.sum(dim=2)  # (B, C)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Loss = 1 - Dice
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss
