# 🚗 차선 검출 모델 배포 패키지

**모델 버전**: v1.0 (Optimized)  
**학습 완료일**: 2026-01-30  
**성능**: IoU 69.45% (Test Set)

---

## 📦 패키지 구성

팀원에게 전달할 파일들:

```
delivery_package/
├── model/
│   └── best_model.pth                    # 학습된 모델 (681MB)
├── code/
│   ├── inference.py                      # 추론 코드 (간단 버전)
│   ├── test_model.py                     # 전체 평가 코드
│   └── requirements.txt                  # 필요한 패키지
├── results/
│   ├── test_results.json                 # 성능 지표
│   ├── distribution.png                  # 성능 분포 그래프
│   ├── boxplot.png                       # Box plot
│   └── per_sample.png                    # 샘플별 성능
├── sample_images/
│   └── (테스트용 샘플 이미지 5개)
├── README.md                             # 사용 방법 설명서
└── PERFORMANCE_REPORT.md                 # 성능 리포트
```

---

## 🎯 전달 옵션

### 옵션 1: 최소 패키지 (가장 추천)
**용도**: 모델 사용/테스트만  
**크기**: ~700MB

**포함 항목**:
- ✅ 학습된 모델 (`.pth`)
- ✅ 추론 코드 (`inference.py`)
- ✅ 사용 설명서 (`README.md`)
- ✅ 성능 리포트
- ✅ 샘플 이미지

### 옵션 2: 전체 패키지
**용도**: 추가 학습/수정 가능  
**크기**: ~2GB

**포함 항목**:
- ✅ 옵션 1의 모든 항목
- ✅ 전체 소스 코드
- ✅ 학습 데이터셋 (199개)
- ✅ 학습 로그

### 옵션 3: 클라우드 링크
**용도**: 대용량 파일 공유  
**방법**: Google Drive / GitHub Release

---

## 💻 팀원이 해야 할 것

### 1. 환경 설정 (5분)
```bash
# Python 3.10+ 필요
pip install -r requirements.txt
```

### 2. 모델 사용 (1줄!)
```python
from inference import LaneDetector

detector = LaneDetector('model/best_model.pth')
result = detector.predict('test_image.jpg')
```

### 3. 결과 확인
- 차선이 표시된 이미지 저장
- IoU, Precision, Recall 출력

---

## 📊 성능 요약

| 지표 | 값 |
|------|------|
| **IoU** | 69.45% ± 2.58% |
| **Pixel Accuracy** | 98.88% |
| **Recall** | 74.67% |
| **Precision** | 46.27% |

✅ **실전 사용 가능 수준**

---

## 🔧 트러블슈팅

**GPU 없으면?**  
→ CPU로도 동작 (느림)

**에러 발생 시?**  
→ 연락주세요!

---

**작성자**: [당신 이름]  
**연락처**: [이메일/슬랙]
