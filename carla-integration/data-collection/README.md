# 🚗 CARLA 자동 데이터 수집

**차량 1대로 Module 03, 06용 데이터를 자동 수집**

---

## 🎯 사용법 (3단계)

### 1단계: CARLA 실행

```bash
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh
```

**화면에 도시가 보이면 성공!**

---

### 2단계: 스크립트 실행

```bash
cd carla-integration/data-collection

# Dependencies 설치 (처음 1번만)
pip install -r requirements.txt

# 10분 동안 수집 (기본)
python auto_collect.py --duration 10
```

**그냥 실행하고 커피 마시면 됨 ☕**

---

### 3단계: 결과 확인

```bash
collected_data/
├── images/              # 이미지 (~6,000장)
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── labels/              # YOLO 라벨 (Module 03용)
│   ├── 000000.txt
│   ├── 000001.txt
│   └── ...
├── labels.csv          # E2E 라벨 (Module 06용)
└── stats.json          # 통계
```

---

## ⚙️ 옵션

```bash
# 30분 동안 수집
python auto_collect.py --duration 30

# 초당 20프레임으로 수집 (더 많이)
python auto_collect.py --duration 10 --fps 20

# 커스텀 출력 폴더
python auto_collect.py --duration 5 --output my_dataset
```

---

## 📊 수집되는 데이터

### Module 03 (YOLO) 용
- `images/*.jpg`: RGB 이미지 (640×480)
- `labels/*.txt`: Bounding box (YOLO format)
  ```
  0 0.512 0.345 0.123 0.089    # class x_center y_center width height
  0 0.723 0.412 0.098 0.076
  ```

### Module 06 (E2E) 용
- `labels.csv`: Steering/Throttle 라벨
  ```csv
  frame,image,steering,throttle,brake,velocity,num_objects,timestamp
  0,000000.jpg,-0.123,0.65,0.0,5.2,3,1706607123.12
  1,000001.jpg,-0.098,0.67,0.0,5.3,2,1706607123.22
  ```

---

## 🎮 실행 중 화면

```
================================================================================
🚗 CARLA Auto Data Collector
================================================================================
Duration: 10 minutes
FPS: 10
Output: collected_data/
================================================================================

🔌 Connecting to CARLA...
✅ Connected to CARLA

🚗 Spawning vehicle...
✅ Vehicle spawned at Location(x=123.4, y=56.7, z=0.3)

📷 Spawning camera...
✅ Camera attached

📊 Starting data collection for 10 minutes...
   Target FPS: 10
   Expected frames: ~6000
   Output: collected_data/

🤖 Autopilot enabled

[ 2341 frames] Elapsed: 3.9m | Remaining: 6.1m | FPS: 10.0 | Steering: -0.123 | Speed: 28.3 km/h
```

**Ctrl+C로 언제든 중단 가능**

---

## 📈 예상 수집량

| 시간 | 프레임 수 (10 FPS) | 디스크 용량 |
|------|------------------|-----------|
| 5분 | ~3,000장 | ~500 MB |
| 10분 | ~6,000장 | ~1 GB |
| 30분 | ~18,000장 | ~3 GB |
| 1시간 | ~36,000장 | ~6 GB |

---

## ✅ 권장 설정

### Module 03 (Object Detection)
```bash
# 최소 1,000장 필요
python auto_collect.py --duration 3

# 권장: 5,000장
python auto_collect.py --duration 10
```

### Module 06 (End-to-End)
```bash
# 최소 10,000장 필요
python auto_collect.py --duration 20

# 권장: 30,000장
python auto_collect.py --duration 60
```

---

## 🐛 문제 해결

### "Connection refused"
```bash
# CARLA가 실행 중인지 확인
ps aux | grep CarlaUE4

# CARLA 재시작
cd CARLA_0.9.15
./CarlaUE4.sh
```

### "No module named 'carla'"
```bash
# CARLA Python API 설치
pip install carla
```

### 너무 느림
```bash
# FPS 낮추기
python auto_collect.py --duration 10 --fps 5
```

---

## 🎯 다음 단계

### 1. 데이터 품질 확인
```bash
python check_data.py --data collected_data
```

### 2. Train/Val/Test 분할
```bash
python split_data.py --data collected_data
# → 70% train / 15% val / 15% test
```

### 3. Module 03 학습
```bash
cd ../../03-object-detection
cp -r ../carla-integration/data-collection/collected_data dataset/
python train.py
```

### 4. Module 06 학습
```bash
cd ../../06-end-to-end-learning
python train.py --data ../carla-integration/data-collection/collected_data/labels.csv
```

---

## 💡 팁

1. **여러 번 수집하기**
   ```bash
   # 날씨/시간/맵을 바꿔서 여러 번 수집
   python auto_collect.py --duration 10 --output data_sunny
   python auto_collect.py --duration 10 --output data_night
   python auto_collect.py --duration 10 --output data_rain
   ```

2. **수집 중 모니터링**
   - CARLA 창에서 차량 움직임 확인
   - Terminal에서 FPS/Steering 확인

3. **디스크 공간 확인**
   ```bash
   df -h  # 최소 5GB 여유 필요
   ```

---

## 📝 작성자

- 작성일: 2026-01-30
- 목적: Module 03, 06 학습용 데이터 자동 수집
- 테스트: ✅ CARLA 0.9.15

---

**문제가 생기면 언제든 물어보세요!** 🚀
