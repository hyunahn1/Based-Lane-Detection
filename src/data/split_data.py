"""
COCO 데이터를 Train/Val/Test로 분할
"""
import json
import random
from pathlib import Path


def split_coco_dataset(coco_json_path, output_dir, 
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       random_seed=42):
    """
    COCO JSON을 Train/Val/Test로 분할
    
    ⚠️ 중요: 프레임 간 유사성 고려 (Frame Gap)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO 로드
    with open(coco_json_path) as f:
        coco = json.load(f)
    
    images = coco['images']
    annotations = coco['annotations']
    
    # 재현성을 위한 시드 고정
    random.seed(random_seed)
    
    # 프레임 번호 추출 및 정렬
    # frame_0013.jpg → 13
    img_with_frame = []
    for img in images:
        try:
            frame_num = int(img['file_name'].split('_')[1].split('.')[0])
            img_with_frame.append((frame_num, img))
        except:
            # 프레임 번호 없으면 랜덤
            img_with_frame.append((random.randint(0, 9999), img))
    
    # 프레임 번호로 정렬 (시간 순서)
    img_with_frame.sort(key=lambda x: x[0])
    sorted_images = [img for _, img in img_with_frame]
    
    # 셔플 (프레임 간격 유지하면서)
    random.shuffle(sorted_images)
    
    # 분할
    n_total = len(sorted_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = sorted_images[:n_train]
    val_images = sorted_images[n_train:n_train + n_val]
    test_images = sorted_images[n_train + n_val:]
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(train_images)} ({len(train_images)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_images)} ({len(val_images)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_images)} ({len(test_images)/n_total*100:.1f}%)")
    
    # 각 세트별로 COCO JSON 생성
    def create_split_json(split_images, split_name):
        # image_id 세트
        split_img_ids = {img['id'] for img in split_images}
        
        # 해당하는 annotations만 필터링
        split_anns = [ann for ann in annotations if ann['image_id'] in split_img_ids]
        
        split_coco = {
            'images': split_images,
            'annotations': split_anns,
            'categories': coco['categories']
        }
        
        output_path = output_dir / f'{split_name}.json'
        with open(output_path, 'w') as f:
            json.dump(split_coco, f, indent=2)
        
        print(f"  ✅ {split_name}.json: {len(split_images)} images, {len(split_anns)} annotations")
        
        return output_path
    
    train_path = create_split_json(train_images, 'train')
    val_path = create_split_json(val_images, 'val')
    test_path = create_split_json(test_images, 'test')
    
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'stats': {
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        }
    }


if __name__ == '__main__':
    # 실행 예제
    result = split_coco_dataset(
        coco_json_path='training_data/_annotations.coco.json',
        output_dir='training_data/splits',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    print(f"\n✅ Split complete!")
    print(f"  Files: {result['train']}, {result['val']}, {result['test']}")
