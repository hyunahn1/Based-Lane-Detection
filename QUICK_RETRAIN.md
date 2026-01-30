# ⚡ 빠른 재학습 가이드

새 데이터로 빠르게 재학습하는 치트시트입니다.

---

## 🚀 한 줄 명령어 (전체 파이프라인)

```bash
# 1단계씩 실행 (권장)
python scripts/check_data_quality.py && \
python training_data/convert_coco.py && \
python src/data/split_data.py && \
python train_optimized.py
```

---

## 📋 단계별 실행

### 1️⃣ 데이터 준비

```bash
# 새 데이터를 training_data/에 넣기
ls training_data/images/ | head     # 이미지 확인
ls training_data/annotations/ | head # JSON 확인

# 품질 체크 (필수!)
python scripts/check_data_quality.py
```

**예상 시간**: 5초  
**확인사항**: ✅ 모두 정상이어야 다음 단계 진행

---

### 2️⃣ COCO 변환

```bash
python training_data/convert_coco.py
```

**예상 시간**: 10초  
**결과 파일**: `training_data/annotations_coco.json`

---

### 3️⃣ 데이터 분할

```bash
python src/data/split_data.py
```

**예상 시간**: 5초  
**결과 파일**: 
- `training_data/splits/train.json` (70%)
- `training_data/splits/val.json` (15%)
- `training_data/splits/test.json` (15%)

---

### 4️⃣ 학습 시작

#### 옵션 A: Baseline (빠른 테스트)

```bash
python train_baseline.py
```

**시간**: 30~40분  
**해상도**: 320×320  
**Epochs**: 50

#### 옵션 B: Optimized (최고 성능) ⭐

```bash
# 백그라운드 실행
nohup python train_optimized.py > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 모니터링
tail -f logs/train_*.log
```

**시간**: 2~3시간  
**해상도**: 384×384  
**Epochs**: 100

---

### 5️⃣ 테스트 평가

```bash
# Baseline
python test_model.py

# Optimized
python test_optimized.py

# 결과 확인
cat test_results_optimized/test_results.json | grep "mean"
```

---

## 🔍 자주 쓰는 명령어

### 학습 상태 확인

```bash
# 완료 여부
tail -50 logs/train_*.log | grep "Training complete"

# 현재 Epoch
tail -30 logs/train_*.log | grep "Epoch"

# Best IoU
tail -200 logs/train_*.log | grep "Best Val IoU" | tail -1

# 프로세스 확인
ps aux | grep train_optimized | grep -v grep
```

### 학습 중단/재시작

```bash
# 중단
pkill -f train_optimized.py

# 재시작 (체크포인트에서)
# → 현재는 자동 재시작 없음, 처음부터 다시 학습
python train_optimized.py
```

### GPU 확인

```bash
# GPU 사용률
nvidia-smi

# 실시간 모니터링
watch -n 1 nvidia-smi
```

---

## ⚠️ 트러블슈팅

### "No such file or directory"

```bash
# 경로 확인
pwd
ls training_data/

# 상대 경로로 실행
cd /home/student/ads-skynet/hyunahn
python training_data/convert_coco.py
```

### "CUDA out of memory"

```python
# train_optimized.py 수정
'batch_size': 4,        # 6 → 4
'input_size': (320, 320)  # (384, 384) → (320, 320)
```

### "개수가 맞지 않습니다"

```bash
# 파일 개수 확인
find training_data/images -type f | wc -l
find training_data/annotations -type f -name "*.json" | wc -l

# 품질 체크
python scripts/check_data_quality.py
```

### IoU가 너무 낮음 (<0.50)

```bash
# 1. 데이터 확인
python scripts/check_data_quality.py

# 2. 샘플 이미지와 JSON 확인
# training_data/images/frame_0001.png
# training_data/annotations/frame_0001.json

# 3. 어노테이션 품질 확인
# → 차선이 정확히 표시되어 있는지
```

---

## 📂 주요 파일 위치

```
/home/student/ads-skynet/hyunahn/

├── training_data/
│   ├── images/              ← 이미지
│   ├── annotations/         ← JSON
│   ├── annotations_coco.json
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── checkpoints/
│   ├── baseline/
│   │   └── best_*.pth       ← Baseline 모델
│   └── optimized/
│       └── best_*.pth       ← Optimized 모델
│
├── test_results_optimized/
│   ├── test_results.json
│   └── *.png
│
└── logs/
    └── train_*.log          ← 학습 로그
```

---

## 💡 팁

### 1. 백업 먼저!

```bash
# 기존 데이터 백업
cp -r training_data training_data_backup_$(date +%Y%m%d)

# 기존 체크포인트 백업
cp -r checkpoints checkpoints_backup_$(date +%Y%m%d)
```

### 2. 작은 데이터로 먼저 테스트

```bash
# 처음 10개 샘플만으로 빠른 테스트
# → 데이터 품질 확인용
# → 10분이면 문제 발견 가능
```

### 3. 로그 저장 습관화

```bash
# 항상 날짜/시간 포함
nohup python train_optimized.py > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 나중에 비교 가능
ls -lht logs/
```

---

## ✅ 체크리스트

학습 전 확인사항:

- [ ] 새 데이터가 `training_data/`에 있음
- [ ] `python scripts/check_data_quality.py` 통과
- [ ] COCO 변환 완료 (`annotations_coco.json` 존재)
- [ ] 데이터 분할 완료 (`splits/` 폴더 확인)
- [ ] GPU 메모리 충분 (`nvidia-smi` 확인)
- [ ] 디스크 공간 충분 (`df -h` 확인)

---

**작성일**: 2026-01-29  
**전체 가이드**: `RETRAIN_GUIDE.md` 참조
