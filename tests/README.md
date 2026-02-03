# 🧪 Tests

프로젝트 테스트 스크립트 및 결과

## 📁 파일 목록

### 🔬 통합 테스트
- **[FACTCHECK.py](FACTCHECK.py)** - 전체 프로젝트 팩트체크 스크립트
- **[TEST_RESEARCH.py](TEST_RESEARCH.py)** - 연구 기능 통합 테스트

### 📊 성능 테스트
- **[performance_test.py](performance_test.py)** - 3가지 패러다임 성능 비교 테스트
- **[performance_results.json](performance_results.json)** - 성능 테스트 결과 (JSON)

---

## 🚀 사용 방법

### 전체 팩트체크 실행
```bash
python tests/FACTCHECK.py
```

### 성능 비교 테스트
```bash
python tests/performance_test.py
```

### 연구 기능 테스트
```bash
python tests/TEST_RESEARCH.py
```

---

## 📖 모듈별 테스트

각 모듈의 단위 테스트는 해당 모듈 디렉토리를 참고하세요:

- **[01-lane-detection/](../01-lane-detection/)** - `test_*.py` 파일들
- **[02-lane-keeping-assist/tests/](../02-lane-keeping-assist/tests/)** - 단위 테스트
- **[03-object-detection/tests/](../03-object-detection/tests/)** - 검출기 테스트
- **[06-end-to-end-learning/](../06-end-to-end-learning/)** - E2E 모델 테스트
- **[08-reinforcement-learning/](../08-reinforcement-learning/)** - RL 에이전트 테스트

---

**[← Back to Main README](../README.md)**
