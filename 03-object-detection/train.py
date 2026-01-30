"""
YOLOv8 Training Script
정확도 우선 학습
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch


def train_yolov8(
    model: str = 'yolov8l.pt',
    data: str = 'config/dataset.yaml',
    epochs: int = 200,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 50,
    project: str = 'runs/train',
    name: str = 'yolov8l_accuracy',
    device: int = 0,
    resume: bool = False
):
    """
    YOLOv8 모델 학습
    
    Parameters:
        model: Model variant ('yolov8l.pt' for accuracy)
        data: Dataset configuration path
        epochs: Training epochs
        batch: Batch size
        imgsz: Input image size
        patience: Early stopping patience
        project: Save directory
        name: Experiment name
        device: GPU device (0, 1, ...) or 'cpu'
        resume: Resume from last checkpoint
    """
    print("="*80)
    print("🚀 YOLOv8 Training - High Accuracy Configuration")
    print("="*80)
    print(f"Model:       {model}")
    print(f"Data:        {data}")
    print(f"Epochs:      {epochs}")
    print(f"Batch:       {batch}")
    print(f"Image size:  {imgsz}")
    print(f"Device:      {device}")
    print("="*80 + "\n")
    
    # GPU 확인
    if device != 'cpu':
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, using CPU")
            device = 'cpu'
        else:
            print(f"✅ Using GPU: {torch.cuda.get_device_name(device)}")
            print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB\n")
    
    # 모델 로드
    model = YOLO(model)
    
    # 학습 시작
    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        project=project,
        name=name,
        device=device,
        resume=resume,
        
        # Optimizer (정확도 우선)
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Loss weights
        box=7.5,          # High weight for accurate localization
        cls=0.5,
        dfl=1.5,
        
        # Augmentation (aggressive)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        # Training settings
        pretrained=True,
        freeze=10,        # Freeze first 10 layers
        save=True,
        save_period=10,
        verbose=True,
        
        # Validation
        val=True,
        plots=True,
        save_json=True,
        
        # Performance
        amp=True,         # Mixed precision
        workers=8
    )
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    print(f"Best mAP@0.5:      {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"Best mAP@0.5:0.95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"Best weights:      {project}/{name}/weights/best.pt")
    print("="*80)
    
    return results


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='YOLOv8 Training')
    
    parser.add_argument('--model', type=str, default='yolov8l.pt',
                       help='Model variant (yolov8l.pt for accuracy)')
    parser.add_argument('--data', type=str, default='config/dataset.yaml',
                       help='Dataset configuration')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Save directory')
    parser.add_argument('--name', type=str, default='yolov8l_accuracy',
                       help='Experiment name')
    parser.add_argument('--device', default=0,
                       help='GPU device (0, 1, ...) or cpu')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    
    args = parser.parse_args()
    
    # 학습 실행
    train_yolov8(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
