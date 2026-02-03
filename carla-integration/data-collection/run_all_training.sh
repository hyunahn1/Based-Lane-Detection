#!/bin/bash
# ============================================================
# 전체 자동 학습 스크립트
# Module 03, 06, 08 한번에 실행
# ============================================================

set -e  # 에러 발생시 중단

echo "============================================================"
echo "🚀 SEA_ME Autonomous Driving - Full Training Pipeline"
echo "============================================================"
echo ""

# ============================================================
# Phase 1: 데이터 수집 (10분)
# ============================================================
echo "📸 [Phase 1/4] 데이터 수집 시작..."
echo ""

cd ~/ads-skynet/hyunahn/carla-integration/data-collection

echo "✅ 1-1. Dependencies 설치"
pip3 install -r requirements.txt -q

echo "✅ 1-2. 테스트 수집 (1분)"
python3 auto_collect.py --duration 1 --output test_data

echo "✅ 1-3. 실제 수집 (10분 = ~1200-1500장)"
python3 auto_collect.py --duration 10 --output collected_data

echo "✅ 1-4. 데이터 품질 확인"
python3 check_data.py --data collected_data

echo "✅ 1-5. Train/Val/Test 분할"
python3 split_data.py --data collected_data

echo ""
echo "✅ Phase 1 완료: 데이터 수집 완료"
echo ""

# ============================================================
# Phase 2: Module 03 학습 (1-2시간)
# ============================================================
echo "🎯 [Phase 2/4] Module 03 학습 시작..."
echo ""

cd ~/ads-skynet/hyunahn/03-object-detection

echo "✅ 2-1. Dependencies 설치"
pip3 install -r requirements.txt -q

echo "✅ 2-2. 데이터셋 설정 파일 생성"
cat > config/carla_dataset.yaml << EOF
path: ../carla-integration/data-collection/collected_data_split
train: train/images
val: val/images
test: test/images

names:
  0: vehicle

nc: 1
EOF

echo "✅ 2-3. YOLOv8 학습 시작 (epochs=50, batch=16)"
python3 train.py \
    --data config/carla_dataset.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --name carla_yolo

echo ""
echo "✅ Phase 2 완료: Module 03 학습 완료"
echo ""

# ============================================================
# Phase 3: Module 06 학습 (1-2시간)
# ============================================================
echo "🧠 [Phase 3/4] Module 06 학습 시작..."
echo ""

cd ~/ads-skynet/hyunahn/06-end-to-end-learning

echo "✅ 3-1. Dependencies 설치"
pip3 install -r requirements.txt -q

echo "✅ 3-2. ViT E2E 학습 시작 (epochs=30, batch=32)"
python3 train.py \
    --data ../carla-integration/data-collection/collected_data/labels.csv \
    --epochs 30 \
    --batch 32 \
    --lr 1e-4

echo ""
echo "✅ Phase 3 완료: Module 06 학습 완료"
echo ""

# ============================================================
# Phase 4: Module 08 학습 (4-6시간)
# ============================================================
echo "🤖 [Phase 4/4] Module 08 학습 시작..."
echo ""

cd ~/ads-skynet/hyunahn/08-reinforcement-learning

echo "✅ 4-1. Dependencies 설치"
pip3 install -r requirements.txt -q

echo "✅ 4-2. PPO+ICM RL 학습 시작 (steps=1M)"
python3 train_rl.py \
    --carla-host localhost \
    --carla-port 2000 \
    --total-steps 1000000 \
    --save-interval 50000 \
    --use-curiosity

echo ""
echo "✅ Phase 4 완료: Module 08 학습 완료"
echo ""

# ============================================================
# Phase 5: 백업
# ============================================================
echo "💾 [Phase 5/5] 결과 백업..."
echo ""

cd ~/ads-skynet/hyunahn

echo "✅ Git commit & push"
git add .
git commit -m "Complete training: Module 03, 06, 08 on CARLA data

- Collected ~1200-1500 images (10min)
- Module 03: YOLOv8 (epochs=50)
- Module 06: ViT E2E (epochs=30)
- Module 08: PPO+ICM (steps=1M)
"
git push

echo ""
echo "============================================================"
echo "🎉 전체 학습 완료!"
echo "============================================================"
echo ""
echo "📊 결과 위치:"
echo "  - Module 03: 03-object-detection/runs/detect/carla_yolo/"
echo "  - Module 06: 06-end-to-end-learning/runs/e2e_training/"
echo "  - Module 08: 08-reinforcement-learning/runs/rl_training/"
echo ""
echo "⏱️  총 소요 시간: ~6-10시간"
echo "============================================================"
