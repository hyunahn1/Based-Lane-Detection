#!/usr/bin/env python3
"""
Module 06: End-to-End Learning Training Script
Vision Transformer (ViT) for direct image-to-control mapping
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Module 06 models
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from models.e2e_model import EndToEndModel


class DrivingDataset(Dataset):
    """CARLA E2E Driving Dataset"""
    
    def __init__(self, csv_path, image_dir=None, transform=None):
        self.df = pd.read_csv(csv_path)
        # If image_dir is provided, use it; otherwise use csv parent directory
        self.image_dir = Path(image_dir) if image_dir else Path(csv_path).parent / 'images'
        self.transform = transform
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"Images from: {self.image_dir}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row['image']
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        # Control labels
        steering = torch.tensor(row['steering'], dtype=torch.float32)
        throttle = torch.tensor(row['throttle'], dtype=torch.float32)
        control = torch.stack([steering, throttle])
        
        return image, control


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, controls in pbar:
        images = images.to(device)
        controls = controls.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, controls)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, controls in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            controls = controls.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, controls)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train E2E Model')
    parser.add_argument('--data', required=True, help='Path to labels.csv')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='runs/e2e_training')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Module 06: End-to-End Learning Training")
    print("="*80)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Split train/val (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    # Save splits
    train_csv = save_dir / 'train.csv'
    val_csv = save_dir / 'val.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # Datasets
    # Get original image directory from input data path
    original_data_dir = Path(args.data).parent
    image_dir = original_data_dir / 'images'
    
    train_dataset = DrivingDataset(train_csv, image_dir=image_dir)
    val_dataset = DrivingDataset(val_csv, image_dir=image_dir)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = EndToEndModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    model = model.to(args.device)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, args.device)
        val_losses.append(val_loss)
        
        # Scheduler
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / 'best_e2e_model.pth')
            print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Final save
    torch.save(model.state_dict(), save_dir / 'final_e2e_model.pth')
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_dir / 'training_curve.png')
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Models saved to: {save_dir}/")
    print(f"  - best_e2e_model.pth")
    print(f"  - final_e2e_model.pth")
    print("="*80)


if __name__ == '__main__':
    main()
