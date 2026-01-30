"""
Optimized 모델 테스트 평가
"""
import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import LaneDataset, get_val_transform
from src.models.deeplabv3plus import get_model
from src.training.metrics import calculate_iou, calculate_pixel_accuracy, calculate_precision_recall


def test_model(checkpoint_path, test_json, image_dir, output_dir='test_results_optimized'):
    """Optimized 모델 테스트"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"🧪 Optimized Model Test Evaluation")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test set: {test_json}")
    print(f"{'='*80}\n")
    
    # 모델 로드
    print("📦 Loading model...")
    model = get_model(num_classes=2, pretrained=False).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # aux_classifier 키 제거
        new_state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}
        model.load_state_dict(new_state_dict, strict=False)
        
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✅ Model loaded from epoch {epoch}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    model.eval()
    
    # 데이터셋 로드
    print("\n📁 Loading test dataset...")
    test_dataset = LaneDataset(
        coco_json_path=test_json,
        image_dir=image_dir,
        transform=get_val_transform(),
        target_size=(384, 384)  # Optimized 해상도
    )
    print(f"✅ Test samples: {len(test_dataset)}")
    
    # 평가
    print(f"\n🔍 Evaluating on test set...\n")
    
    results = {
        'iou': [],
        'pixel_acc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    per_sample_results = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc='Testing'):
            image, mask = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)  # GPU로 이동
            
            # 추론
            output = model(image)
            
            # 메트릭 계산
            iou = calculate_iou(output, mask)
            pixel_acc = calculate_pixel_accuracy(output, mask)
            precision, recall = calculate_precision_recall(output, mask)
            
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            results['iou'].append(iou)
            results['pixel_acc'].append(pixel_acc)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            
            per_sample_results.append({
                'index': idx,
                'iou': float(iou),
                'pixel_acc': float(pixel_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            })
    
    # 통계 계산
    summary = {}
    for metric_name, values in results.items():
        summary[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"📊 Test Results Summary (Optimized Model)")
    print(f"{'='*80}\n")
    
    for metric_name, stats in summary.items():
        print(f"{metric_name.upper():12s}: "
              f"Mean {stats['mean']:.4f} ± {stats['std']:.4f} | "
              f"Min {stats['min']:.4f} | Max {stats['max']:.4f} | "
              f"Median {stats['median']:.4f}")
    
    print(f"\n{'='*80}\n")
    
    # 결과 저장
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    result_json = {
        'summary': summary,
        'per_sample': per_sample_results,
        'checkpoint': str(checkpoint_path),
        'test_samples': len(test_dataset),
        'resolution': '384x384'
    }
    
    json_path = output_path / 'test_results.json'
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"💾 Results saved to {json_path}")
    
    # 시각화
    print(f"\n📊 Generating visualizations...")
    
    # 1. Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Optimized Model - Metric Distributions', fontsize=16, fontweight='bold')
    
    metrics = ['iou', 'pixel_acc', 'precision', 'recall', 'f1']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        ax.hist(results[metric], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(summary[metric]['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {summary[metric]['mean']:.4f}")
        ax.axvline(summary[metric]['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {summary[metric]['median']:.4f}")
        ax.set_xlabel(metric.upper())
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    dist_path = output_path / 'distribution.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {dist_path}")
    plt.close()
    
    # 2. Box plots
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = [results[m] for m in metrics]
    bp = ax.boxplot(data, labels=[m.upper() for m in metrics], patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_ylabel('Score')
    ax.set_title('Optimized Model - Metric Distributions (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    box_path = output_path / 'boxplot.png'
    plt.savefig(box_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {box_path}")
    plt.close()
    
    # 3. Per-sample IoU
    fig, ax = plt.subplots(figsize=(14, 6))
    
    indices = list(range(len(results['iou'])))
    ax.plot(indices, results['iou'], marker='o', linestyle='-', linewidth=2, markersize=6, color='steelblue')
    ax.axhline(summary['iou']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {summary['iou']['mean']:.4f}")
    ax.axhline(0.70, color='orange', linestyle='--', linewidth=2, label='Target: 0.70')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('IoU')
    ax.set_title('Optimized Model - Per-Sample IoU Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    sample_path = output_path / 'per_sample.png'
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ {sample_path}")
    plt.close()
    
    print(f"\n✅ All results saved to {output_dir}/")
    
    return summary


def main():
    """메인"""
    
    checkpoint = 'checkpoints/optimized/best_iou0.6656_epoch96.pth'
    
    if not Path(checkpoint).exists():
        print(f"❌ Checkpoint not found: {checkpoint}")
        # 최신 모델 찾기
        checkpoint_dir = Path('checkpoints/optimized')
        best_models = sorted(checkpoint_dir.glob('best*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
        if best_models:
            checkpoint = str(best_models[0])
            print(f"✅ Using latest checkpoint: {checkpoint}")
        else:
            print("❌ No checkpoint found!")
            return
    
    results = test_model(
        checkpoint_path=checkpoint,
        test_json='training_data/splits/test.json',
        image_dir='training_data/images',
        output_dir='test_results_optimized'
    )
    
    if results:
        print(f"\n{'='*80}")
        print(f"🎯 Comparison with Baseline")
        print(f"{'='*80}")
        print(f"Baseline  (320×320): IoU 0.6576")
        print(f"Optimized (384×384): IoU {results['iou']['mean']:.4f}")
        print(f"Improvement: {(results['iou']['mean'] - 0.6576) * 100:+.2f}%")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
