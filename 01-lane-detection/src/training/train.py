"""
메인 학습 스크립트
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys
import json

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import LaneDataset, get_train_transform, get_val_transform
from src.models.deeplabv3plus import get_model
from src.models.losses import get_loss_fn
from src.training.metrics import calculate_iou, calculate_pixel_accuracy, calculate_precision_recall


class Trainer:
    """학습 관리 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"🚀 Training Configuration")
        print(f"{'='*60}")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        # 데이터셋
        self.train_dataset = LaneDataset(
            coco_json_path=config['train_json'],
            image_dir=config['image_dir'],
            transform=get_train_transform(),
            target_size=config['input_size']
        )
        
        self.val_dataset = LaneDataset(
            coco_json_path=config['val_json'],
            image_dir=config['image_dir'],
            transform=get_val_transform(),
            target_size=config['input_size']
        )
        
        # 데이터로더
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        # 모델
        self.model = get_model(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        ).to(self.device)
        
        # 손실 함수
        self.criterion = get_loss_fn(
            ce_weight=config.get('ce_weight', 1.0),
            dice_weight=config.get('dice_weight', 3.0)
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Best model tracking
        self.best_val_iou = 0.0
        self.patience_counter = 0
        
        # Checkpoint dir
        Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch):
        """한 에포크 학습"""
        self.model.train()
        
        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        total_iou = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs, masks)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            iou = calculate_iou(outputs, masks)
            
            # Accumulate
            total_loss += loss.item()
            total_ce += loss_dict['ce']
            total_dice += loss_dict['dice']
            total_iou += iou
            
            # Progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })
        
        # 평균
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        avg_ce = total_ce / n_batches
        avg_dice = total_dice / n_batches
        avg_iou = total_iou / n_batches
        
        return {
            'loss': avg_loss,
            'ce': avg_ce,
            'dice': avg_dice,
            'iou': avg_iou
        }
    
    @torch.no_grad()
    def validate(self, epoch):
        """검증"""
        self.model.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        total_pixel_acc = 0.0
        total_precision = 0.0
        total_recall = 0.0
        
        pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss, _ = self.criterion(outputs, masks)
            
            # Metrics
            iou = calculate_iou(outputs, masks)
            pixel_acc = calculate_pixel_accuracy(outputs, masks)
            precision, recall = calculate_precision_recall(outputs, masks)
            
            # Accumulate
            total_loss += loss.item()
            total_iou += iou
            total_pixel_acc += pixel_acc
            total_precision += precision
            total_recall += recall
            
            pbar.set_postfix({'iou': f'{iou:.4f}'})
        
        # 평균
        n_batches = len(self.val_loader)
        metrics = {
            'loss': total_loss / n_batches,
            'iou': total_iou / n_batches,
            'pixel_acc': total_pixel_acc / n_batches,
            'precision': total_precision / n_batches,
            'recall': total_recall / n_batches
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Latest checkpoint
        latest_path = Path(self.config['checkpoint_dir']) / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Best checkpoint
        if is_best:
            best_path = Path(self.config['checkpoint_dir']) / f'best_iou{metrics["iou"]:.4f}_epoch{epoch}.pth'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model: {best_path}")
    
    def train(self):
        """전체 학습 루프"""
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train IoU: {train_metrics['iou']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Tensorboard
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/IoU', train_metrics['iou'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/IoU', val_metrics['iou'], epoch)
            self.writer.add_scalar('Val/Precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Val/Recall', val_metrics['recall'], epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # Best model
            is_best = val_metrics['iou'] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics['iou']
                self.patience_counter = 0
                print(f"  🎉 New best Val IoU: {self.best_val_iou:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\n⚠️ Early stopping triggered (patience={self.config['patience']})")
                break
            
            print(f"{'='*60}\n")
        
        print(f"\n✅ Training complete!")
        print(f"  Best Val IoU: {self.best_val_iou:.4f}")
        
        self.writer.close()


def main():
    """메인 함수"""
    config = {
        # Data
        'train_json': 'training_data/splits/train.json',
        'val_json': 'training_data/splits/val.json',
        'image_dir': 'training_data/images',
        'input_size': (480, 640),  # (H, W)
        
        # Model
        'num_classes': 2,
        'pretrained': True,
        
        # Training
        'batch_size': 16,  # ⚠️ RTX 5080 기준, 메모리에 따라 조정
        'epochs': 200,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'min_lr': 1e-6,
        
        # Loss
        'ce_weight': 1.0,
        'dice_weight': 3.0,
        
        # Early stopping
        'patience': 20,
        
        # System
        'num_workers': 8,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs'
    }
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
