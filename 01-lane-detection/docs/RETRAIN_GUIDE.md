# 데이터셋 교체 & 재학습 가이드

새로운 데이터셋으로 모델을 재학습하는 전체 과정입니다.

---

## 📋 목차

1. [데이터셋 교체](#1-데이터셋-교체)
2. [COCO 포맷 변환](#2-coco-포맷-변환)
3. [데이터 분할](#3-데이터-분할)
4. [학습 실행](#4-학습-실행)
5. [테스트 평가](#5-테스트-평가)
6. [트러블슈팅](#6-트러블슈팅)

---

## 1. 데이터셋 교체

### 📁 필요한 데이터 구조

```
training_data/
├── images/              ← 새 이미지들
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
└── annotations/         ← 새 어노테이션들
    ├── frame_0001.json  (이미지와 이름 매칭)
    ├── frame_0002.json
    └── ...
```

### ⚠️ 중요 사항

1. **이미지 파일명 = JSON 파일명** (확장자만 다름)
   ```
   frame_0001.png  ↔  frame_0001.json  ✅
   frame_0001.png  ↔  frame_0002.json  ❌
   ```

2. **JSON 포맷** (기존과 동일해야 함)
   ```json
   {
     "version": "1.0",
     "flags": {},
     "shapes": [
       {
         "label": "lane",
         "points": [[x1, y1], [x2, y2], ...],
         "shape_type": "polyline"
       }
     ],
     "imagePath": "frame_0001.png",
     "imageHeight": 480,
     "imageWidth": 640
   }
   ```

### 🔄 교체 방법

#### 방법 A: 자동 스크립트 (권장)

```bash
# 1. 새 데이터를 training_data/에 복사
# (기존 데이터는 자동 백업됨)

# 2. 스크립트 실행
./scripts/replace_dataset.sh
```

#### 방법 B: 수동 교체

```bash
# 1. 기존 데이터 백업
mv training_data training_data_backup_$(date +%Y%m%d_%H%M%S)

# 2. 새 데이터 복사
cp -r /path/to/new/data training_data/

# 3. 구조 확인
ls -lh training_data/images/ | head
ls -lh training_data/annotations/ | head

# 4. 개수 확인
echo "Images: $(find training_data/images -type f | wc -l)"
echo "Annotations: $(find training_data/annotations -type f | wc -l)"
```

---

## 2. COCO 포맷 변환

### 실행

```bash
cd /home/student/ads-skynet/hyunahn
python training_data/convert_coco.py
```

### 예상 출력

```
Processing annotations...
  Found 250 images
  Found 250 annotations
  Matched: 250

✅ COCO format saved: training_data/annotations_coco.json
📊 Statistics:
   Images: 250
   Annotations: 750 (3.0 per image)
   Categories: 1 (lane)
```

### ⚠️ 에러 발생 시

| 에러 | 원인 | 해결 |
|------|------|------|
| `Image not found` | 파일명 불일치 | 이미지와 JSON 이름 확인 |
| `Invalid JSON` | JSON 포맷 오류 | JSON 구조 확인 |
| `No shapes found` | 빈 어노테이션 | 차선이 표시되어 있는지 확인 |

---

## 3. 데이터 분할

### 실행

```bash
python src/data/split_data.py
```

### 예상 출력

```
🔀 Splitting dataset...
   Total: 250 images
   Train: 175 (70%)
   Val: 37 (15%)
   Test: 38 (15%)

✅ Splits saved:
   training_data/splits/train.json
   training_data/splits/val.json
   training_data/splits/test.json
```

### 분할 비율 변경 (선택)

```python
# src/data/split_data.py 수정

split_coco_dataset(
    coco_json_path='training_data/annotations_coco.json',
    output_dir='training_data/splits',
    train_ratio=0.70,  # ← 여기 수정
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
```

---

## 4. 학습 실행

### A. Baseline 학습 (빠른 테스트)

```bash
# 50 epochs, 320x320
python train_baseline.py
```

**예상 시간**: 30~40분  
**용도**: 데이터셋 품질 빠른 확인

### B. Optimized 학습 (최고 성능)

```bash
# 100 epochs, 384x384, Mixed Precision
nohup python train_optimized.py > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**예상 시간**: 2~3시간  
**용도**: 최종 모델 학습

### 학습 모니터링

```bash
# 실시간 로그
tail -f logs/training_*.log

# 현재 Epoch 확인
tail -50 logs/training_*.log | grep "Epoch"

# Best IoU 확인
tail -200 logs/training_*.log | grep "Best Val IoU" | tail -1

# 프로세스 확인
ps aux | grep train_optimized
```

### 학습 완료 확인

```bash
# 완료 메시지 확인
tail -50 logs/training_*.log | grep "Training complete"

# Best 모델 확인
ls -lht checkpoints/optimized/best*.pth | head -1
```

---

## 5. 테스트 평가

### Baseline 평가

```bash
python test_model.py \
  --checkpoint checkpoints/baseline/best*.pth
```

### Optimized 평가

```bash
python test_optimized.py
```

### 결과 확인

```bash
# JSON 결과
cat test_results_optimized/test_results.json

# 시각화
ls test_results_optimized/*.png
```

---

## 6. 트러블슈팅

### 문제 1: COCO 변환 실패

```bash
# 원인: JSON 포맷 오류
# 해결: 샘플 JSON 확인

python -c "
import json
with open('training_data/annotations/frame_0001.json') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
"
```

### 문제 2: 학습 중 OOM (Out of Memory)

```bash
# 해결 1: 배치 크기 감소
# train_optimized.py 수정:
'batch_size': 4,  # 원래 6 → 4

# 해결 2: 해상도 감소
'input_size': (320, 320),  # 원래 (384, 384)
```

### 문제 3: IoU가 너무 낮음 (<0.50)

```bash
# 원인 가능성:
1. 데이터 품질 문제
2. 어노테이션 오류
3. 데이터 부족

# 진단:
python scripts/check_data_quality.py  # (아래 참조)
```

### 문제 4: 특정 샘플에서 계속 실패

```bash
# 실패 샘플 확인
python -c "
import json
with open('test_results_optimized/test_results.json') as f:
    data = json.load(f)
    failures = [s for s in data['per_sample'] if s['iou'] < 0.5]
    print('Failure samples:')
    for f in failures:
        print(f'  Sample {f[\"index\"]}: IoU {f[\"iou\"]:.4f}')
"

# 해당 샘플 이미지 확인
# training_data/splits/test.json에서 인덱스 찾기
```

---

## 📊 데이터 품질 체크 스크립트

<details>
<summary>scripts/check_data_quality.py (클릭하여 펼치기)</summary>

```python
"""
데이터 품질 체크 스크립트
"""
import json
from pathlib import Path
from PIL import Image

def check_data_quality():
    img_dir = Path('training_data/images')
    ann_dir = Path('training_data/annotations')
    
    print("🔍 데이터 품질 체크\n")
    
    # 1. 개수 확인
    images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    jsons = list(ann_dir.glob('*.json'))
    
    print(f"📁 파일 개수:")
    print(f"   Images: {len(images)}")
    print(f"   JSONs: {len(jsons)}\n")
    
    # 2. 매칭 확인
    img_names = {p.stem for p in images}
    json_names = {p.stem for p in jsons}
    
    missing_json = img_names - json_names
    missing_img = json_names - img_names
    
    if missing_json:
        print(f"⚠️  어노테이션 없는 이미지: {len(missing_json)}개")
        for name in list(missing_json)[:5]:
            print(f"   - {name}")
    
    if missing_img:
        print(f"⚠️  이미지 없는 어노테이션: {len(missing_img)}개")
        for name in list(missing_img)[:5]:
            print(f"   - {name}")
    
    print()
    
    # 3. 어노테이션 품질
    empty_annotations = []
    invalid_jsons = []
    
    for json_path in jsons:
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            if not data.get('shapes'):
                empty_annotations.append(json_path.name)
            elif len(data['shapes']) == 0:
                empty_annotations.append(json_path.name)
        except Exception as e:
            invalid_jsons.append((json_path.name, str(e)))
    
    if empty_annotations:
        print(f"⚠️  빈 어노테이션: {len(empty_annotations)}개")
        for name in empty_annotations[:5]:
            print(f"   - {name}")
    
    if invalid_jsons:
        print(f"❌ 잘못된 JSON: {len(invalid_jsons)}개")
        for name, error in invalid_jsons[:5]:
            print(f"   - {name}: {error}")
    
    print()
    
    # 4. 이미지 크기 확인
    sizes = {}
    for img_path in list(images)[:10]:  # 샘플링
        img = Image.open(img_path)
        size = f"{img.width}x{img.height}"
        sizes[size] = sizes.get(size, 0) + 1
    
    print(f"📐 이미지 크기 (샘플 10개):")
    for size, count in sizes.items():
        print(f"   {size}: {count}개")
    
    print("\n✅ 체크 완료!")

if __name__ == '__main__':
    check_data_quality()
```
</details>

```bash
python scripts/check_data_quality.py
```

---

## 🎯 빠른 참조

### 전체 파이프라인 (한 번에)

```bash
# 1. 데이터 교체
./scripts/replace_dataset.sh

# 2. COCO 변환
python training_data/convert_coco.py

# 3. 데이터 분할
python src/data/split_data.py

# 4. 학습 (백그라운드)
nohup python train_optimized.py > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 5. 모니터링
tail -f logs/train_*.log

# 6. 완료 후 테스트
python test_optimized.py
```

### 주요 파일 경로

```
training_data/
├── images/                      ← 원본 이미지
├── annotations/                 ← 원본 JSON
├── annotations_coco.json        ← COCO 변환 결과
└── splits/
    ├── train.json               ← 학습 셋
    ├── val.json                 ← 검증 셋
    └── test.json                ← 테스트 셋

checkpoints/
├── baseline/
│   └── best_iou*.pth           ← Baseline 모델
└── optimized/
    └── best_iou*.pth           ← Optimized 모델

test_results_optimized/
├── test_results.json           ← 수치 결과
├── distribution.png            ← 분포 그래프
├── boxplot.png                 ← Box plot
└── per_sample.png              ← 샘플별 성능
```

---

## 📞 도움말

### 문제가 생기면?

1. **로그 확인**: `tail -100 logs/training_*.log`
2. **프로세스 확인**: `ps aux | grep python`
3. **GPU 메모리**: `nvidia-smi`
4. **디스크 공간**: `df -h`

### 긴급 중단

```bash
# 학습 중단
pkill -f train_optimized.py

# 프로세스 확인
ps aux | grep python | grep train
```

---

**작성일**: 2026-01-29  
**버전**: 1.0
