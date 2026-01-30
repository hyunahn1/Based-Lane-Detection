# 🚗 자율주행 RC카 - 차선 인식 ML 프로젝트

딥러닝 기반 차선 인식 시스템으로 RC카의 자율주행을 구현한 프로젝트입니다.

## 📋 프로젝트 개요

- **목적**: 실내 RC 트랙에서 차선을 정확히 인식하여 자율주행 구현
- **모델**: DeepLabV3+ (ResNet-101 백본)
- **데이터**: 199개의 실내 트랙 이미지 (640×480 RGB)
- **성능**: 고성능 차선 세그멘테이션 및 후처리 알고리즘 적용

## 🏗️ 프로젝트 구조

```
autonomous-driving-ML/
├── src/                          # 소스 코드
│   ├── models/                   # 모델 정의 (DeepLabV3+, Loss 함수)
│   ├── data/                     # 데이터셋 로더 및 전처리
│   ├── training/                 # 학습 루프 및 메트릭
│   └── inference/                # 추론 및 후처리
├── docs/                         # 프로젝트 문서
│   ├── 01_아키텍처_설계서_v2_고성능.md
│   ├── 02_구현_명세서_v2_고성능.md
│   ├── 03_검증서_v2_고성능.md
│   ├── 04_구현_일치율_분석.md
│   ├── 05_테스트_성능_평가.md
│   └── 06_최적화_개선_보고서.md
├── test_results/                 # 베이스라인 테스트 결과
├── test_results_optimized/       # 최적화된 모델 테스트 결과
├── train_baseline.py             # 베이스라인 학습 스크립트
├── train_optimized.py            # 최적화된 학습 스크립트
├── test_with_postprocess.py      # 후처리 포함 테스트 스크립트
└── requirements.txt              # 의존성 패키지
```

## 🚀 주요 기능

### 1. 고성능 차선 세그멘테이션
- **DeepLabV3+ 아키텍처**: ASPP(Atrous Spatial Pyramid Pooling)를 통한 멀티스케일 특징 추출
- **ResNet-101 백본**: 사전 학습된 가중치 활용
- **Combined Loss**: Dice Loss + Focal Loss로 클래스 불균형 문제 해결

### 2. 데이터 증강 전략
- **학습 시**: RandomResizedCrop, ColorJitter, RandomRotation, GaussianBlur
- **검증 시**: 최소한의 전처리로 실제 성능 평가
- 소량 데이터(199개)를 효과적으로 활용

### 3. 고급 후처리 파이프라인
- **모폴로지 연산**: 노이즈 제거 및 차선 연결
- **연결 성분 분석**: 가장 큰 차선 영역 추출
- **폴리라인 피팅**: 차선 중심선 추출 및 스무딩
- **신뢰도 기반 필터링**: 저품질 예측 제거

### 4. 학습 최적화
- **혼합 정밀도 학습(AMP)**: 학습 속도 향상
- **학습률 스케줄링**: ReduceLROnPlateau로 적응적 조정
- **조기 종료**: 과적합 방지
- **체크포인트 저장**: 최상의 모델 자동 저장

## 📊 성능 결과

### 베이스라인 모델
- **평균 IoU**: 0.8924
- **평균 Dice Score**: 0.9430
- **평균 Pixel Accuracy**: 0.9852

### 최적화된 모델 (후처리 포함)
- **평균 IoU**: 0.9156
- **평균 Dice Score**: 0.9561
- **평균 Pixel Accuracy**: 0.9891

📈 상세 결과는 `test_results/` 및 `test_results_optimized/` 폴더 참조

## 🛠️ 기술 스택

- **프레임워크**: PyTorch
- **컴퓨터 비전**: OpenCV, PIL
- **데이터 처리**: NumPy, Albumentations
- **시각화**: Matplotlib
- **학습 모니터링**: TensorBoard

## 📦 설치 방법

```bash
# 저장소 클론
git clone [repository-url]
cd autonomous-driving-ML

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 🎯 사용 방법

### 학습
```bash
# 베이스라인 모델 학습
python train_baseline.py

# 최적화된 모델 학습
python train_optimized.py
```

### 테스트
```bash
# 후처리 포함 테스트
python test_with_postprocess.py
```

### 모델 재학습
상세한 재학습 가이드는 `docs/RETRAIN_GUIDE.md` 참조

## 📁 데이터셋 형식

```
dataset/
├── images/           # 원본 이미지 (640×480 RGB)
└── annotations/      # JSON 형식의 polyline 어노테이션
```

어노테이션 예시:
```json
{
  "shapes": [
    {
      "label": "lane",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polyline"
    }
  ]
}
```

## 🔧 모델 아키텍처

```
입력 이미지 (640×480×3)
    ↓
ResNet-101 Encoder
    ↓
ASPP (Atrous Spatial Pyramid Pooling)
    ↓
Decoder (저수준 특징과 융합)
    ↓
출력 (차선 세그멘테이션 맵)
    ↓
후처리 파이프라인
    ↓
최종 차선 폴리라인
```

## 📚 문서

프로젝트의 자세한 내용은 `docs/` 폴더의 문서를 참조하세요:

- **아키텍처 설계서**: 시스템 설계 및 기술 선택 근거
- **구현 명세서**: 코드 구현 상세 설명
- **검증서**: 모델 성능 검증 결과
- **성능 평가**: 정량적 성능 분석
- **최적화 보고서**: 성능 개선 과정

## 🎓 학습 포인트

이 프로젝트를 통해 다음을 학습하고 구현했습니다:

1. **세그멘테이션 모델**: DeepLabV3+ 아키텍처의 이해와 구현
2. **데이터 증강**: 소량 데이터 활용 전략
3. **손실 함수 설계**: 클래스 불균형 문제 해결
4. **후처리 알고리즘**: 컴퓨터 비전 기법 적용
5. **학습 최적화**: 혼합 정밀도, 학습률 스케줄링 등
6. **성능 평가**: 다양한 메트릭(IoU, Dice, Accuracy) 분석

## 📝 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

## 👤 작성자

SEA:ME 자율주행 프로젝트 팀

---

⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!
