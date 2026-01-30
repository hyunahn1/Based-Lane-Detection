#!/bin/bash
# 데이터셋 교체 스크립트

echo "🔄 데이터셋 교체 시작"
echo "================================"

# 1. 기존 데이터 백업
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="training_data_backup_${TIMESTAMP}"

echo "📦 기존 데이터 백업 중..."
if [ -d "training_data" ]; then
    mv training_data "$BACKUP_DIR"
    echo "✅ 백업 완료: $BACKUP_DIR"
else
    echo "⚠️  기존 training_data 폴더 없음 (첫 실행)"
fi

# 2. 새 데이터 준비 확인
echo ""
echo "📁 새 데이터 확인 중..."

if [ ! -d "training_data" ]; then
    echo "❌ training_data 폴더가 없습니다!"
    echo ""
    echo "다음과 같이 준비해주세요:"
    echo "training_data/"
    echo "├── images/         ← 이미지 파일들 (.png, .jpg)"
    echo "└── annotations/    ← JSON 파일들 (.json)"
    exit 1
fi

# 이미지와 어노테이션 개수 확인
IMG_COUNT=$(find training_data/images -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
JSON_COUNT=$(find training_data/annotations -type f -name "*.json" 2>/dev/null | wc -l)

echo "✅ 이미지: $IMG_COUNT 개"
echo "✅ 어노테이션: $JSON_COUNT 개"

if [ $IMG_COUNT -eq 0 ] || [ $JSON_COUNT -eq 0 ]; then
    echo "❌ 데이터가 없습니다!"
    exit 1
fi

if [ $IMG_COUNT -ne $JSON_COUNT ]; then
    echo "⚠️  경고: 이미지와 어노테이션 개수가 다릅니다!"
    echo "   이미지: $IMG_COUNT, 어노테이션: $JSON_COUNT"
    read -p "계속하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "✅ 데이터셋 교체 완료!"
echo ""
echo "다음 단계:"
echo "1. python training_data/convert_coco.py    # COCO 변환"
echo "2. python src/data/split_data.py           # 데이터 분할"
echo "3. python train_optimized.py               # 학습 시작"
echo ""
