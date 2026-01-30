"""
Test 셋 평가 및 성능 분석
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


def evaluate_test_set(checkpoint_path, test_json, image_dir, device='cuda'):
    """Test 셋 전체 평가"""
    
    print(f"\n{'='*60}")
    print(f"🧪 Test Set Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 모델 로드
    print("📦 Loading model...")
    model = get_model(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # State dict 로드 (strict=False로 aux_classifier 무시)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except:
        # 만약 실패하면 키 이름 변경 시도
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # aux_classifier는 제외
            if 'aux_classifier' not in k:
                new_state_dict[k] = v
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
    
    # 평가
    print(f"\n🔍 Evaluating {len(test_dataset)} samples...\n")
    
    all_ious = []
    all_pixel_accs = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    per_sample_results = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc='Testing'):
            image, mask = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            # 추론
            output = model(image)
            
            # 메트릭 계산
            iou = calculate_iou(output, mask)
            pixel_acc = calculate_pixel_accuracy(output, mask)
            precision, recall = calculate_precision_recall(output, mask)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            all_ious.append(iou)
            all_pixel_accs.append(pixel_acc)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            
            per_sample_results.append({
                'index': idx,
                'iou': iou,
                'pixel_acc': pixel_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
    
    # 통계 계산
    results = {
        'iou': {
            'mean': np.mean(all_ious),
            'std': np.std(all_ious),
            'min': np.min(all_ious),
            'max': np.max(all_ious),
            'median': np.median(all_ious)
        },
        'pixel_acc': {
            'mean': np.mean(all_pixel_accs),
            'std': np.std(all_pixel_accs),
            'min': np.min(all_pixel_accs),
            'max': np.max(all_pixel_accs),
            'median': np.median(all_pixel_accs)
        },
        'precision': {
            'mean': np.mean(all_precisions),
            'std': np.std(all_precisions),
            'min': np.min(all_precisions),
            'max': np.max(all_precisions),
            'median': np.median(all_precisions)
        },
        'recall': {
            'mean': np.mean(all_recalls),
            'std': np.std(all_recalls),
            'min': np.min(all_recalls),
            'max': np.max(all_recalls),
            'median': np.median(all_recalls)
        },
        'f1': {
            'mean': np.mean(all_f1s),
            'std': np.std(all_f1s),
            'min': np.min(all_f1s),
            'max': np.max(all_f1s),
            'median': np.median(all_f1s)
        }
    }
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"📊 Test Results")
    print(f"{'='*60}\n")
    
    print(f"🎯 IoU (Intersection over Union)")
    print(f"  Mean:   {results['iou']['mean']:.4f} ± {results['iou']['std']:.4f}")
    print(f"  Median: {results['iou']['median']:.4f}")
    print(f"  Range:  [{results['iou']['min']:.4f}, {results['iou']['max']:.4f}]")
    
    print(f"\n📍 Pixel Accuracy")
    print(f"  Mean:   {results['pixel_acc']['mean']:.4f} ± {results['pixel_acc']['std']:.4f}")
    print(f"  Median: {results['pixel_acc']['median']:.4f}")
    print(f"  Range:  [{results['pixel_acc']['min']:.4f}, {results['pixel_acc']['max']:.4f}]")
    
    print(f"\n🎯 Precision (Lane Pixels)")
    print(f"  Mean:   {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}")
    print(f"  Median: {results['precision']['median']:.4f}")
    print(f"  Range:  [{results['precision']['min']:.4f}, {results['precision']['max']:.4f}]")
    
    print(f"\n🎯 Recall (Lane Pixels)")
    print(f"  Mean:   {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}")
    print(f"  Median: {results['recall']['median']:.4f}")
    print(f"  Range:  [{results['recall']['min']:.4f}, {results['recall']['max']:.4f}]")
    
    print(f"\n🎯 F1-Score")
    print(f"  Mean:   {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")
    print(f"  Median: {results['f1']['median']:.4f}")
    print(f"  Range:  [{results['f1']['min']:.4f}, {results['f1']['max']:.4f}]")
    
    print(f"\n{'='*60}")
    
    # 목표 대비 평가
    print(f"\n📋 목표 대비 평가")
    print(f"{'='*60}")
    
    target_iou_min = 0.70
    target_iou_realistic = 0.75
    target_iou_optimistic = 0.80
    
    achieved_iou = results['iou']['mean']
    
    if achieved_iou >= target_iou_optimistic:
        status = "🎉 EXCELLENT (낙관적 목표 달성!)"
    elif achieved_iou >= target_iou_realistic:
        status = "✅ GOOD (현실적 목표 달성!)"
    elif achieved_iou >= target_iou_min:
        status = "⚠️ ACCEPTABLE (최소 기준 달성)"
    else:
        status = f"❌ BELOW TARGET (최소 기준 미달, {target_iou_min - achieved_iou:.3f} 부족)"
    
    print(f"\n목표:")
    print(f"  낙관적: IoU ≥ {target_iou_optimistic:.2f}")
    print(f"  현실적: IoU ≥ {target_iou_realistic:.2f}")
    print(f"  최소:   IoU ≥ {target_iou_min:.2f}")
    print(f"\n달성:")
    print(f"  Test IoU: {achieved_iou:.4f}")
    print(f"  상태: {status}")
    
    # Val vs Test 비교
    if 'metrics' in checkpoint:
        val_iou = checkpoint['metrics'].get('iou', None)
        if val_iou is not None:
            gap = achieved_iou - val_iou
            print(f"\n📊 Val vs Test 비교:")
            print(f"  Val IoU:  {val_iou:.4f}")
            print(f"  Test IoU: {achieved_iou:.4f}")
            print(f"  Gap:      {gap:+.4f} {'(일반화 양호)' if abs(gap) < 0.05 else '(일반화 주의)'}")
    
    print(f"\n{'='*60}\n")
    
    # 결과 저장
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    
    # JSON 저장
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump({
            'summary': results,
            'per_sample': per_sample_results,
            'checkpoint': str(checkpoint_path),
            'test_samples': len(test_dataset)
        }, f, indent=2)
    
    print(f"💾 Results saved to: {output_dir / 'test_results.json'}")
    
    # 시각화
    create_visualizations(results, per_sample_results, output_dir)
    
    return results, per_sample_results


def create_visualizations(results, per_sample_results, output_dir):
    """결과 시각화"""
    print(f"\n📊 Creating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Figure 1: Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Test Set Performance Distribution', fontsize=16, fontweight='bold')
    
    metrics = ['iou', 'pixel_acc', 'precision', 'recall', 'f1']
    titles = ['IoU', 'Pixel Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        values = [s[metric] for s in per_sample_results]
        
        # Histogram
        ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(results[metric]['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results[metric]['mean']:.3f}")
        ax.axvline(results[metric]['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {results[metric]['median']:.3f}")
        
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{title} Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide last subplot if odd number
    if len(metrics) % 3 != 0:
        axes[-1, -1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir / 'distribution.png'}")
    plt.close()
    
    # Figure 2: Box plots
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    data = [
        [s['iou'] for s in per_sample_results],
        [s['pixel_acc'] for s in per_sample_results],
        [s['precision'] for s in per_sample_results],
        [s['recall'] for s in per_sample_results],
        [s['f1'] for s in per_sample_results]
    ]
    
    bp = ax.boxplot(data, labels=titles, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Performance Metrics - Box Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal lines for targets
    ax.axhline(0.70, color='orange', linestyle=':', linewidth=1.5, label='Min Target (0.70)')
    ax.axhline(0.75, color='green', linestyle=':', linewidth=1.5, label='Realistic Target (0.75)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir / 'boxplot.png'}")
    plt.close()
    
    # Figure 3: Per-sample performance
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    indices = [s['index'] for s in per_sample_results]
    ious = [s['iou'] for s in per_sample_results]
    
    ax.plot(indices, ious, marker='o', linestyle='-', color='blue', alpha=0.6, label='IoU')
    ax.axhline(results['iou']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['iou']['mean']:.3f}")
    ax.axhline(0.70, color='orange', linestyle=':', linewidth=1.5, label='Min Target (0.70)')
    ax.axhline(0.75, color='green', linestyle=':', linewidth=1.5, label='Realistic Target (0.75)')
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('Per-Sample IoU Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_sample.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir / 'per_sample.png'}")
    plt.close()
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")


def main():
    """메인 실행"""
    
    # Best 모델 찾기
    checkpoint_dir = Path('checkpoints/baseline')
    best_checkpoints = sorted(checkpoint_dir.glob('best_iou*.pth'), 
                              key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not best_checkpoints:
        print("❌ No best checkpoint found!")
        return
    
    best_checkpoint = best_checkpoints[0]
    
    # 평가 실행
    results, per_sample = evaluate_test_set(
        checkpoint_path=str(best_checkpoint),
        test_json='training_data/splits/test.json',
        image_dir='training_data/images',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n✅ Test evaluation complete!")


if __name__ == '__main__':
    main()
