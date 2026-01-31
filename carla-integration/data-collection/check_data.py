#!/usr/bin/env python3
"""
데이터 품질 체크 스크립트
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import cv2


def check_data(data_dir):
    """수집된 데이터 품질 체크"""
    data_dir = Path(data_dir)
    
    print("="*80)
    print("📊 Data Quality Check")
    print("="*80)
    
    # 1. 파일 개수 확인
    images = list((data_dir / 'images').glob('*.jpg'))
    labels_yolo = list((data_dir / 'labels').glob('*.txt'))
    
    print(f"\n📁 Files:")
    print(f"   Images: {len(images)}")
    print(f"   YOLO labels: {len(labels_yolo)}")
    
    # 2. CSV 확인
    csv_path = data_dir / 'labels.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"   CSV records: {len(df)}")
        
        # 3. Steering 분포
        print(f"\n📈 Steering Statistics:")
        print(f"   Mean: {df['steering'].mean():.3f}")
        print(f"   Std: {df['steering'].std():.3f}")
        print(f"   Min: {df['steering'].min():.3f}")
        print(f"   Max: {df['steering'].max():.3f}")
        
        # 분포 그래프
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['steering'], bins=50, edgecolor='black')
        plt.title('Steering Distribution')
        plt.xlabel('Steering')
        plt.ylabel('Count')
        
        plt.subplot(1, 3, 2)
        plt.hist(df['throttle'], bins=50, edgecolor='black')
        plt.title('Throttle Distribution')
        plt.xlabel('Throttle')
        plt.ylabel('Count')
        
        plt.subplot(1, 3, 3)
        plt.hist(df['velocity'], bins=50, edgecolor='black')
        plt.title('Velocity Distribution (m/s)')
        plt.xlabel('Velocity')
        plt.ylabel('Count')
        
        plt.tight_layout()
        output_path = data_dir / 'distribution.png'
        plt.savefig(output_path)
        print(f"\n✅ Saved distribution plot: {output_path}")
        
        # 4. 경고
        straight_ratio = (df['steering'].abs() < 0.05).sum() / len(df)
        if straight_ratio > 0.7:
            print(f"\n⚠️  Warning: {straight_ratio*100:.1f}% frames are straight driving")
            print(f"   Consider collecting more curved road data")
        
        # 5. Object detection 통계
        if 'num_objects' in df.columns:
            print(f"\n🚗 Object Detection:")
            print(f"   Frames with objects: {(df['num_objects'] > 0).sum()} ({(df['num_objects'] > 0).sum()/len(df)*100:.1f}%)")
            print(f"   Avg objects per frame: {df['num_objects'].mean():.2f}")
    
    # 6. 샘플 이미지 확인
    if images:
        sample_img = cv2.imread(str(images[0]))
        print(f"\n🖼️  Image Info:")
        print(f"   Resolution: {sample_img.shape[1]}×{sample_img.shape[0]}")
        print(f"   Channels: {sample_img.shape[2]}")
    
    print("\n" + "="*80)
    print("✅ Quality check complete!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='데이터 품질 체크')
    parser.add_argument('--data', required=True, help='데이터 디렉토리')
    
    args = parser.parse_args()
    check_data(args.data)
