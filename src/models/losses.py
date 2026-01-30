"""
Segmentation 손실 함수
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    CrossEntropy + Dice Loss 조합
    
    단순하지만 효과적인 조합
    """
    
    def __init__(self, ce_weight=1.0, dice_weight=3.0):
        """
        Parameters:
        -----------
        ce_weight : float
            CrossEntropy 가중치
        dice_weight : float
            Dice Loss 가중치 (더 높게)
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        """
        Parameters:
        -----------
        pred : torch.Tensor, (B, C, H, W), logits
        target : torch.Tensor, (B, H, W), class indices
        
        Returns:
        --------
        loss : torch.Tensor, scalar
        """
        # CrossEntropy Loss
        ce = self.ce_loss(pred, target)
        
        # Dice Loss
        dice = self.dice_loss(pred, target)
        
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        
        return total_loss, {'ce': ce.item(), 'dice': dice.item()}
    
    def dice_loss(self, pred, target, smooth=1.0):
        """
        Dice Loss for multi-class segmentation
        """
        # Softmax
        pred = F.softmax(pred, dim=1)
        
        # One-hot encoding
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, H, W, C) → (B, C, H, W)
        
        # Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_coeff.mean()
        
        return dice_loss


def get_loss_fn(ce_weight=1.0, dice_weight=3.0):
    """손실 함수 생성 헬퍼"""
    return CombinedLoss(ce_weight=ce_weight, dice_weight=dice_weight)
