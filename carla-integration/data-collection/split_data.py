#!/usr/bin/env python3
"""
데이터 분할 스크립트 (Train/Val/Test)
"""

import argparse
import shutil
from pathlib import Path
import random
import pandas as pd


def split_data(data_dir, train_ratio=0.7, val_ratio=0.15):
    """Train/Val/Test 분할"""
    data_dir = Path(data_dir)
    output_dir = data_dir.parent / f'{data_dir.name}_split'
    
    print("="*80)
    print("✂️  Splitting data into Train/Val/Test")
    print("="*80)
    
    # 출력 디렉토리 생성
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 이미지 리스트
    images = sorted((data_dir / 'images').glob('*.jpg'))
    random.shuffle(images)
    
    # 분할 인덱스
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = images[:n_train]
    val_images = images[n_train:n_train+n_val]
    test_images = images[n_train+n_val:]
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_images)} ({len(train_images)/n_total*100:.1f}%)")
    print(f"   Val:   {len(val_images)} ({len(val_images)/n_total*100:.1f}%)")
    print(f"   Test:  {len(test_images)} ({len(test_images)/n_total*100:.1f}%)")
    
    # 복사 함수
    def copy_files(image_list, split_name):
        for img_path in image_list:
            # 이미지 복사
            dst_img = output_dir / split_name / 'images' / img_path.name
            shutil.copy(img_path, dst_img)
            
            # 라벨 복사 (있으면)
            label_path = data_dir / 'labels' / f'{img_path.stem}.txt'
            if label_path.exists():
                dst_label = output_dir / split_name / 'labels' / label_path.name
                shutil.copy(label_path, dst_label)
    
    print("\n📋 Copying files...")
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    copy_files(test_images, 'test')
    
    # CSV도 분할
    csv_path = data_dir / 'labels.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        # 파일명으로 분할
        train_names = [img.name for img in train_images]
        val_names = [img.name for img in val_images]
        test_names = [img.name for img in test_images]
        
        df_train = df[df['image'].isin(train_names)]
        df_val = df[df['image'].isin(val_names)]
        df_test = df[df['image'].isin(test_names)]
        
        df_train.to_csv(output_dir / 'train' / 'labels.csv', index=False)
        df_val.to_csv(output_dir / 'val' / 'labels.csv', index=False)
        df_test.to_csv(output_dir / 'test' / 'labels.csv', index=False)
        
        print("✅ CSV files split")
    
    print("\n" + "="*80)
    print(f"✅ Split complete!")
    print("="*80)
    print(f"\n📁 Output: {output_dir}/")
    print(f"   - train/images/ ({len(train_images)} images)")
    print(f"   - val/images/ ({len(val_images)} images)")
    print(f"   - test/images/ ({len(test_images)} images)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Val/Test 분할')
    parser.add_argument('--data', required=True, help='데이터 디렉토리')
    parser.add_argument('--train', type=float, default=0.7, help='Train 비율')
    parser.add_argument('--val', type=float, default=0.15, help='Val 비율')
    
    args = parser.parse_args()
    split_data(args.data, args.train, args.val)
