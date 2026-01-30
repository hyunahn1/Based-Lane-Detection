"""
후처리 파이프라인
Precision 향상을 위한 노이즈 제거 및 정제
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class PostProcessor:
    """차선 마스크 후처리기"""
    
    def __init__(self, 
                 threshold: float = 0.6,
                 min_area: int = 100,
                 morph_kernel_size: int = 5,
                 apply_morph: bool = True,
                 apply_cca: bool = True):
        """
        Parameters:
        -----------
        threshold : float
            이진화 임계값 (0.5 → 0.6으로 증가하면 Precision 향상)
        min_area : int
            최소 영역 크기 (픽셀)
        morph_kernel_size : int
            Morphological operation 커널 크기
        apply_morph : bool
            Morphological operations 적용 여부
        apply_cca : bool
            Connected Component Analysis 적용 여부
        """
        self.threshold = threshold
        self.min_area = min_area
        self.morph_kernel_size = morph_kernel_size
        self.apply_morph = apply_morph
        self.apply_cca = apply_cca
        
        # Morphological 커널
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morph_kernel_size, morph_kernel_size)
        )
    
    def __call__(self, pred_prob: np.ndarray) -> np.ndarray:
        """
        후처리 적용
        
        Parameters:
        -----------
        pred_prob : np.ndarray, (H, W) or (H, W, C)
            예측 확률 맵 (softmax 출력)
        
        Returns:
        --------
        mask : np.ndarray, (H, W)
            후처리된 이진 마스크 {0, 1}
        """
        # 확률 → 이진 마스크 (임계값 적용)
        if len(pred_prob.shape) == 3:
            # (H, W, C) → (H, W) - lane class만
            pred_prob = pred_prob[:, :, 1] if pred_prob.shape[2] == 2 else pred_prob[:, :, 0]
        
        mask = (pred_prob > self.threshold).astype(np.uint8)
        
        # 1. Morphological Operations
        if self.apply_morph:
            mask = self.morphological_operations(mask)
        
        # 2. Connected Component Analysis
        if self.apply_cca:
            mask = self.connected_component_filtering(mask)
        
        return mask
    
    def morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        Morphological 연산으로 노이즈 제거 및 구멍 메우기
        
        - Opening: 작은 노이즈 제거 (False Positive 감소)
        - Closing: 작은 구멍 메우기 (연속성 향상)
        """
        # Opening (침식 → 팽창): 작은 노이즈 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        
        # Closing (팽창 → 침식): 작은 구멍 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        
        return mask
    
    def connected_component_filtering(self, mask: np.ndarray) -> np.ndarray:
        """
        Connected Component Analysis로 작은 영역 제거
        
        - 작은 노이즈 영역 제거 (False Positive 대폭 감소)
        - 가장 큰 N개 영역만 유지
        """
        # Connected Components 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # 새 마스크 생성
        filtered_mask = np.zeros_like(mask)
        
        # 각 컴포넌트 평가
        valid_components = []
        for i in range(1, num_labels):  # 0은 배경
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= self.min_area:
                valid_components.append((i, area))
        
        # 면적 기준 상위 컴포넌트만 유지 (최대 3개 - 차선 개수 제한)
        valid_components.sort(key=lambda x: x[1], reverse=True)
        max_components = min(3, len(valid_components))
        
        for i, _ in valid_components[:max_components]:
            filtered_mask[labels == i] = 1
        
        return filtered_mask
    
    def optimize_threshold(self, pred_probs, gt_masks, 
                          threshold_range=(0.3, 0.8, 0.05)):
        """
        최적 임계값 탐색 (Validation 셋으로)
        
        Returns:
        --------
        best_threshold : float
        best_precision : float
        """
        from src.training.metrics import calculate_precision_recall
        import torch
        
        thresholds = np.arange(*threshold_range)
        best_threshold = 0.5
        best_f1 = 0.0
        
        results = []
        
        for thresh in thresholds:
            self.threshold = thresh
            
            precisions = []
            recalls = []
            
            for pred_prob, gt_mask in zip(pred_probs, gt_masks):
                # 후처리 적용
                pred_mask = self(pred_prob)
                
                # Precision, Recall 계산
                pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)
                gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)
                
                # One-hot으로 변환
                pred_onehot = torch.zeros(1, 2, *pred_mask.shape)
                pred_onehot[0, 0] = (pred_tensor == 0)
                pred_onehot[0, 1] = (pred_tensor == 1)
                
                precision, recall = calculate_precision_recall(pred_onehot, gt_tensor)
                precisions.append(precision)
                recalls.append(recall)
            
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-8)
            
            results.append({
                'threshold': thresh,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        self.threshold = best_threshold
        
        print(f"\n🎯 Optimal Threshold: {best_threshold:.2f}")
        print(f"   Precision: {results[-1]['precision']:.4f}")
        print(f"   Recall: {results[-1]['recall']:.4f}")
        print(f"   F1: {best_f1:.4f}")
        
        return best_threshold, results


def apply_tta(model, image, device='cuda', tta_transforms=None):
    """
    Test-Time Augmentation
    
    여러 변형으로 예측 후 앙상블
    """
    import torch
    import torch.nn.functional as F
    
    if tta_transforms is None:
        # 기본 TTA 변형
        tta_transforms = [
            {'flip': None, 'brightness': 0},      # 원본
            {'flip': None, 'brightness': 0.1},    # 밝게
            {'flip': None, 'brightness': -0.1},   # 어둡게
            {'flip': 'horizontal', 'brightness': 0},  # 수평 반전 (선택적)
        ]
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            # 이미지 변형
            img = image.clone()
            
            # 밝기 조정
            if transform['brightness'] != 0:
                img = img + transform['brightness']
                img = torch.clamp(img, 0, 1)
            
            # 수평 반전
            if transform['flip'] == 'horizontal':
                img = torch.flip(img, dims=[3])
            
            # 예측
            pred = model(img.to(device))
            pred = F.softmax(pred, dim=1)
            
            # 원래대로 복원
            if transform['flip'] == 'horizontal':
                pred = torch.flip(pred, dims=[3])
            
            predictions.append(pred)
    
    # 앙상블 (평균)
    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    
    return ensemble_pred
