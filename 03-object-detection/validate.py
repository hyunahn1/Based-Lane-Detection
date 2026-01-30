"""
Model Validation Script
Test set 평가
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import json
import torch


def validate_model(
    weights: str,
    data: str = 'config/dataset.yaml',
    split: str = 'test',
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.25,
    iou: float = 0.45,
    save_json: bool = True,
    save_plots: bool = True
):
    """
    모델 검증 및 평가
    
    Parameters:
        weights: Model weights path
        data: Dataset configuration
        split: 'val' or 'test'
        imgsz: Input size
        batch: Batch size
        conf: Confidence threshold
        iou: NMS IoU threshold
        save_json: Save results as JSON
        save_plots: Save evaluation plots
    """
    print("="*80)
    print(f"📊 Model Validation - {split.upper()} SET")
    print("="*80)
    print(f"Weights:     {weights}")
    print(f"Data:        {data}")
    print(f"Split:       {split}")
    print(f"Batch size:  {batch}")
    print(f"Confidence:  {conf}")
    print("="*80 + "\n")
    
    # GPU 확인
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("⚠️ Using CPU (slower)\n")
    
    # 모델 로드
    model = YOLO(weights)
    
    # 검증 실행
    metrics = model.val(
        data=data,
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        save_json=save_json,
        plots=save_plots,
        verbose=True
    )
    
    # 결과 파싱
    results = {
        "mAP@0.5": float(metrics.box.map50),
        "mAP@0.5:0.95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(metrics.box.f1.mean() if hasattr(metrics.box.f1, 'mean') else 0.0)
    }
    
    # 클래스별 성능
    class_names = ['traffic_cone', 'obstacle', 'robot_car', 'traffic_sign', 'pedestrian']
    
    if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        results["per_class_ap50"] = {
            name: float(ap) 
            for name, ap in zip(class_names, metrics.box.ap50)
        }
    
    # 결과 출력
    print("\n" + "="*80)
    print(f"📊 {split.upper()} SET EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n【 Overall Metrics 】")
    print(f"  mAP@0.5:      {results['mAP@0.5']:.4f} {'✅' if results['mAP@0.5'] >= 0.90 else '❌'} (target: 0.90)")
    print(f"  mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f} {'✅' if results['mAP@0.5:0.95'] >= 0.70 else '❌'} (target: 0.70)")
    print(f"  Precision:    {results['precision']:.4f} {'✅' if results['precision'] >= 0.92 else '❌'} (target: 0.92)")
    print(f"  Recall:       {results['recall']:.4f} {'✅' if results['recall'] >= 0.88 else '❌'} (target: 0.88)")
    print(f"  F1-Score:     {results['f1']:.4f}")
    
    # 클래스별 성능
    if 'per_class_ap50' in results:
        print(f"\n【 Per-Class AP@0.5 】")
        for name, ap in results['per_class_ap50'].items():
            print(f"  {name:15s}: {ap:.4f}")
    
    # 목표 달성 여부
    goals_met = (
        results['mAP@0.5'] >= 0.90 and
        results['mAP@0.5:0.95'] >= 0.70 and
        results['precision'] >= 0.92 and
        results['recall'] >= 0.88
    )
    
    print("\n" + "="*80)
    if goals_met:
        print("✅ ALL GOALS MET!")
    else:
        print("❌ GOALS NOT MET - Further tuning needed")
    print("="*80)
    
    # JSON 저장
    if save_json:
        output_path = Path('results') / f'{split}_metrics.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved: {output_path}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLOv8 Validation')
    
    parser.add_argument('--weights', type=str, required=True,
                       help='Model weights path')
    parser.add_argument('--data', type=str, default='config/dataset.yaml',
                       help='Dataset configuration')
    parser.add_argument('--split', type=str, default='test',
                       choices=['val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IoU threshold')
    
    args = parser.parse_args()
    
    # 검증 실행
    validate_model(
        weights=args.weights,
        data=args.data,
        split=args.split,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou
    )


if __name__ == '__main__':
    main()
