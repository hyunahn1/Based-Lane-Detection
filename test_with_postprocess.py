"""
후처리 적용 후 테스트
"""
import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from src.data.dataset import LaneDataset, get_val_transform
from src.models.deeplabv3plus import get_model
from src.training.metrics import calculate_iou, calculate_pixel_accuracy, calculate_precision_recall
from src.inference.postprocess import PostProcessor


def evaluate_with_postprocess(checkpoint_path, test_json, image_dir, 
                              threshold=0.6, device='cuda'):
    """후처리 적용 후 테스트 셋 평가"""
    
    print(f"\n{'='*60}")
    print(f"🧪 Test with Post-Processing")
    print(f"{'='*60}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}\n")
    
    # 모델 로드
    print("📦 Loading model...")
    model = get_model(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print(f"✅ Model loaded (Epoch {checkpoint['epoch']})")
    
    # 데이터셋 로드
    print("\n📁 Loading test dataset...")
    test_dataset = LaneDataset(
        coco_json_path=test_json,
        image_dir=image_dir,
        transform=get_val_transform(),
        target_size=(320, 320)
    )
    print(f"✅ Test samples: {len(test_dataset)}")
    
    # 후처리기 초기화
    postprocessor = PostProcessor(
        threshold=threshold,
        min_area=100,
        morph_kernel_size=5,
        apply_morph=True,
        apply_cca=True
    )
    
    # 평가
    print(f"\n🔍 Evaluating with post-processing...\n")
    
    all_ious = []
    all_precisions = []
    all_recalls = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc='Testing'):
            image, mask = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0)
            
            # 추론
            output = model(image)
            
            # Softmax
            pred_prob = torch.softmax(output, dim=1)
            pred_prob_np = pred_prob[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            
            # 후처리 적용
            pred_mask_np = postprocessor(pred_prob_np)
            pred_mask = torch.from_numpy(pred_mask_np).unsqueeze(0).long()
            
            # 메트릭 계산 (후처리된 마스크로)
            pred_onehot = torch.zeros(1, 2, *pred_mask.shape[1:])
            pred_onehot[0, 0] = (pred_mask[0] == 0)
            pred_onehot[0, 1] = (pred_mask[0] == 1)
            
            iou = calculate_iou(pred_onehot, mask)
            precision, recall = calculate_precision_recall(pred_onehot, mask)
            
            all_ious.append(iou)
            all_precisions.append(precision)
            all_recalls.append(recall)
    
    # 결과
    results = {
        'iou': {
            'mean': np.mean(all_ious),
            'std': np.std(all_ious)
        },
        'precision': {
            'mean': np.mean(all_precisions),
            'std': np.std(all_precisions)
        },
        'recall': {
            'mean': np.mean(all_recalls),
            'std': np.std(all_recalls)
        }
    }
    
    f1 = 2 * (results['precision']['mean'] * results['recall']['mean']) / \
         (results['precision']['mean'] + results['recall']['mean'] + 1e-8)
    
    print(f"\n{'='*60}")
    print(f"📊 Results with Post-Processing")
    print(f"{'='*60}\n")
    print(f"🎯 IoU:       {results['iou']['mean']:.4f} ± {results['iou']['std']:.4f}")
    print(f"🎯 Precision: {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}")
    print(f"🎯 Recall:    {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}")
    print(f"🎯 F1-Score:  {f1:.4f}")
    print(f"\n{'='*60}\n")
    
    return results


def main():
    """여러 임계값으로 테스트"""
    
    checkpoint = 'checkpoints/baseline/best_iou0.6583_epoch45.pth'
    
    print("\n🔬 Testing different thresholds...\n")
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    best_threshold = 0.5
    best_iou = 0.0
    
    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing threshold: {thresh}")
        print(f"{'='*60}")
        
        results = evaluate_with_postprocess(
            checkpoint_path=checkpoint,
            test_json='training_data/splits/test.json',
            image_dir='training_data/images',
            threshold=thresh,
            device='cuda'
        )
        
        if results['iou']['mean'] > best_iou:
            best_iou = results['iou']['mean']
            best_threshold = thresh
    
    print(f"\n{'='*60}")
    print(f"🏆 Best Configuration")
    print(f"{'='*60}")
    print(f"Threshold: {best_threshold}")
    print(f"IoU: {best_iou:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
