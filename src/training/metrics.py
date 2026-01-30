"""
Evaluation Metrics
"""
import torch
import numpy as np


def calculate_iou(pred, target, num_classes=2):
    """
    IoU (Intersection over Union) 계산
    
    Parameters:
    -----------
    pred : torch.Tensor, (B, C, H, W), logits
    target : torch.Tensor, (B, H, W), class indices
    num_classes : int
    
    Returns:
    --------
    iou : float
    """
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    # Mean IoU
    return np.mean(ious)


def calculate_pixel_accuracy(pred, target):
    """Pixel-wise accuracy"""
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    correct = (pred == target).sum().float()
    total = target.numel()
    return (correct / total).item()


def calculate_precision_recall(pred, target, pos_class=1):
    """
    Precision & Recall for positive class (lane)
    """
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    
    tp = ((pred == pos_class) & (target == pos_class)).sum().float()
    fp = ((pred == pos_class) & (target != pos_class)).sum().float()
    fn = ((pred != pos_class) & (target == pos_class)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return precision.item(), recall.item()
