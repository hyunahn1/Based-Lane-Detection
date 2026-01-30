"""
Dataset Split Script
Train/Val/Test 분할 (70/15/15)
"""
import shutil
import random
import argparse
from pathlib import Path
from typing import Tuple


def split_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
):
    """
    데이터셋을 Train/Val/Test로 분할
    
    Parameters:
        image_dir: Raw images directory
        label_dir: Raw labels directory (YOLO format)
        output_dir: Output directory
        split_ratio: (train, val, test) ratio
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print("="*80)
    print("📂 Dataset Split")
    print("="*80)
    print(f"Image dir:   {image_dir}")
    print(f"Label dir:   {label_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Split ratio: {split_ratio}")
    print(f"Seed:        {seed}")
    print("="*80 + "\n")
    
    # 이미지 파일 목록
    image_files = list(Path(image_dir).glob('*.jpg')) + \
                  list(Path(image_dir).glob('*.png'))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {image_dir}")
        return
    
    random.shuffle(image_files)
    
    # 분할 계산
    total = len(image_files)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"📊 Dataset Statistics:")
    print(f"  Total:  {total} images")
    print(f"  Train:  {len(train_files)} ({len(train_files)/total*100:.1f}%)")
    print(f"  Val:    {len(val_files)} ({len(val_files)/total*100:.1f}%)")
    print(f"  Test:   {len(test_files)} ({len(test_files)/total*100:.1f}%)")
    print()
    
    # 파일 복사
    output_path = Path(output_dir)
    
    for split_name, files in [
        ('train', train_files),
        ('val', val_files),
        ('test', test_files)
    ]:
        print(f"Copying {split_name} split...")
        
        img_out = output_path / 'images' / split_name
        lbl_out = output_path / 'labels' / split_name
        
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        copied_images = 0
        copied_labels = 0
        missing_labels = 0
        
        for img_file in files:
            # 이미지 복사
            shutil.copy(img_file, img_out / img_file.name)
            copied_images += 1
            
            # 레이블 복사
            lbl_file = Path(label_dir) / f"{img_file.stem}.txt"
            
            if lbl_file.exists():
                shutil.copy(lbl_file, lbl_out / lbl_file.name)
                copied_labels += 1
            else:
                missing_labels += 1
        
        print(f"  ✅ {split_name:5s}: {copied_images} images, {copied_labels} labels")
        
        if missing_labels > 0:
            print(f"  ⚠️ Missing labels: {missing_labels}")
    
    print("\n" + "="*80)
    print("✅ Dataset split completed!")
    print("="*80)
    
    # 클래스 분포 확인
    print("\nChecking class distribution...")
    check_class_distribution(output_path)


def check_class_distribution(dataset_path: Path):
    """클래스 분포 확인"""
    class_counts = {i: 0 for i in range(5)}
    class_names = ['traffic_cone', 'obstacle', 'robot_car', 'traffic_sign', 'pedestrian']
    
    for split in ['train', 'val', 'test']:
        label_files = list((dataset_path / 'labels' / split).glob('*.txt'))
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    cls = int(line.split()[0])
                    class_counts[cls] += 1
    
    total = sum(class_counts.values())
    
    print(f"\n📊 Class Distribution:")
    for cls_id, name in enumerate(class_names):
        count = class_counts.get(cls_id, 0)
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {cls_id}. {name:15s}: {count:5d} ({percentage:5.1f}%)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Split dataset for YOLOv8')
    
    parser.add_argument('--images', type=str, required=True,
                       help='Raw images directory')
    parser.add_argument('--labels', type=str, required=True,
                       help='Raw labels directory (YOLO format)')
    parser.add_argument('--output', type=str, default='datasets',
                       help='Output directory')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Train split ratio')
    parser.add_argument('--val', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test', type=float, default=0.15,
                       help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 분할 실행
    split_dataset(
        image_dir=args.images,
        label_dir=args.labels,
        output_dir=args.output,
        split_ratio=(args.train, args.val, args.test),
        seed=args.seed
    )


if __name__ == '__main__':
    main()
