"""
Baseline 학습 (50 epoch)
빠른 확인용
"""
import sys
sys.path.append('.')

from src.training.train import Trainer

config = {
    # Data
    'train_json': 'training_data/splits/train.json',
    'val_json': 'training_data/splits/val.json',
    'image_dir': 'training_data/images',
    'input_size': (320, 320),  # (H, W) - OOM 방지 (480x640 → 320x320)
    
    # Model
    'num_classes': 2,
    'pretrained': True,
    
    # Training
    'batch_size': 8,  # ✅ 해상도 줄여서 배치 8 가능
    'epochs': 50,  # Baseline: 50 epoch
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'min_lr': 1e-6,
    
    # Loss
    'ce_weight': 1.0,
    'dice_weight': 3.0,
    
    # Early stopping
    'patience': 15,
    
    # System
    'num_workers': 8,
    'checkpoint_dir': 'checkpoints/baseline',
    'log_dir': 'logs/baseline'
}

print("🚀 Starting Baseline Training (50 epochs)")
print("="*60)

trainer = Trainer(config)
trainer.train()
