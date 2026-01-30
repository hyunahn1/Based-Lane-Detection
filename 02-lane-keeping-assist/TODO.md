# 🚧 Module 02: Lane Keeping Assist - 남은 작업

**현재 상태:** 🔄 Phase 2 진행중 (25% 완료)  
**작성일:** 2026-01-30  
**다음 AI에게 전달할 작업 목록**

---

## ✅ 완료된 작업

### Phase 1: 문서화 ✅ 100%
- ✅ `docs/01_아키텍처_설계서.md` (완성)
- ✅ `docs/02_구현_명세서.md` (완성)
- ✅ `docs/03_검증서.md` (완성)
- ✅ `docs/00_팩트체크_및_수정사항.md` (완성)

### Phase 2: 구현 ✅ 25%
- ✅ **LaneTracker** (456줄)
  - ✅ Mask → Polyline 추출
  - ✅ 원근 보정 픽셀-미터 변환
  - ✅ Heading 추정 (IMU 없이)
  - ✅ 곡률 계산
  - ✅ **9개 테스트 케이스 ALL PASS**
  - ✅ 좌표 버그 수정 완료
  - 파일: `src/tracking/lane_tracker.py`

---

## 🔴 남은 작업 (우선순위 순)

### Phase 2: 구현 (계속) ⏳ 75%

#### 1. DepartureDetector (이탈 감지기) 🔴 HIGH PRIORITY
**파일:** `src/detection/departure_detector.py`  
**참고 문서:** `docs/02_구현_명세서.md` (§3.2)

**구현할 내용:**
```python
class DepartureDetector:
    """
    차선 이탈 감지 및 위험도 평가
    
    입력:
        - lateral_offset (float): 횡방향 오프셋 (meters)
        - heading_error (float): 헤딩 오차 (degrees)
        - vehicle_speed (float): 차량 속도 (m/s)
        - timestamp (float): 시각
    
    출력:
        - is_departing (bool): 이탈 여부
        - risk_level (int): 위험도 0-5
        - time_to_crossing (float): TTC (seconds)
        - departure_side (str): "left", "right", "none"
    """
```

**임계값 (RC 트랙 기준):**
```python
@dataclass
class DepartureThresholds:
    level_2_offset: float = 0.08  # 8cm
    level_3_offset: float = 0.12  # 12cm
    level_4_offset: float = 0.15  # 15cm
    level_5_offset: float = 0.18  # 18cm (트랙 경계)
    
    level_2_heading: float = 10.0  # degrees
    level_3_heading: float = 20.0
    level_4_heading: float = 30.0
    level_5_heading: float = 40.0
```

**핵심 알고리즘:**
1. `_calculate_risk_level()`: offset과 heading 중 최대값 선택
2. `_calculate_ttc()`: `remaining_distance / lateral_velocity`
3. `_determine_side()`: offset 부호로 방향 결정

**테스트:** `tests/test_departure_detector.py`
- Test Case 5: 안전 주행 (3cm offset)
- Test Case 6: 경고 레벨 (13cm offset)
- Test Case 7: 긴급 상황 (19cm offset)

**예상 시간:** 1시간

---

#### 2. PIDController (조향 제어기) 🔴 HIGH PRIORITY
**파일:** `src/control/pid_controller.py`  
**참고 문서:** `docs/02_구현_명세서.md` (§3.3)

**구현할 내용:**
```python
class PIDController:
    """
    PID 기반 조향 제어기
    
    제어 법칙:
        u(t) = Kp * e + Ki * ∫e dt + Kd * de/dt + FF
        
        where:
            e = lateral_offset + K_heading * heading_error
            FF = arctan(wheelbase * curvature)
    """
```

**파라미터 (RC 카 초기값):**
```python
@dataclass
class PIDParams:
    kp: float = 2.0           # 비례 게인
    ki: float = 0.2           # 적분 게인
    kd: float = 0.5           # 미분 게인
    k_heading: float = 0.2    # 헤딩 가중치 (낮음, 부정확하므로)
    
    max_steering_angle: float = 45.0   # RC 서보 범위
    max_steering_rate: float = 100.0   # deg/s (RC 빠름)
    windup_limit: float = 5.0          # Anti-windup
    wheelbase: float = 0.25            # PiRacer wheelbase
```

**핵심 알고리즘:**
1. P term: `kp * error`
2. I term: `ki * integral` (with anti-windup)
3. D term: `kd * derivative`
4. FF term: `arctan(wheelbase * curvature)` (clipped to ±15°)
5. Rate limiting: `max_steering_rate` 적용

**테스트:** `tests/test_pid_controller.py`
- Test Case 8: P 제어 단독
- Test Case 9: I 누적
- Test Case 10: Anti-windup
- Test Case 11: 조향각 제한

**예상 시간:** 1시간

---

#### 3. WarningSystem (경고 시스템) 🟡 MEDIUM PRIORITY
**파일:** `src/alert/warning_system.py`  
**참고 문서:** `docs/02_구현_명세서.md` (미완성, 아키텍처 참고)

**구현할 내용:**
```python
class WarningSystem:
    """
    위험도 기반 다단계 경고 시스템
    
    경고 타입:
        - Visual: OpenCV로 화면에 경고 표시
        - Audio: 비프음 (선택적, 구현 간단히)
        - Haptic: 미구현 (PiRacer 하드웨어 없음)
    """
```

**위험도별 경고:**
```python
Level 0-1: 경고 없음
Level 2:   시각 경고 (노란색)
Level 3:   시각 + 청각 (주황색 + 비프음 1회)
Level 4:   시각 + 청각 반복 (빨간색 + 비프음 2회)
Level 5:   전체 화면 경고 (깜빡임 + 연속 경보음)
```

**핵심 메서드:**
```python
def update(self, risk_level: int, departure_side: str):
    """위험도 업데이트"""

def render_visual_warning(self, frame: np.ndarray) -> np.ndarray:
    """프레임에 경고 오버레이"""

def trigger_audio_warning(self):
    """오디오 경고 재생 (선택적)"""
```

**테스트:** `tests/test_warning_system.py`
- Test Case 12: 레벨별 경고 활성화

**예상 시간:** 30분 (간단)

---

#### 4. Main Orchestrator (통합) 🟢 LOW PRIORITY
**파일:** `src/lkas.py`  
**참고 문서:** `docs/01_아키텍처_설계서.md` (§6.3)

**구현할 내용:**
```python
class LaneKeepingAssist:
    """
    LKAS 메인 오케스트레이터
    
    컴포넌트 통합:
        1. LaneTracker
        2. DepartureDetector
        3. WarningSystem
        4. PIDController
    """
    
    def process_frame(
        self,
        lane_detection: Dict,  # Module 01 출력
        vehicle_state: Dict    # 속도, 타임스탬프
    ) -> Dict:
        """
        전체 파이프라인 실행
        
        Returns:
            {
                "steering_angle": float,
                "throttle_adjustment": float,
                "warning_level": int,
                "is_intervening": bool,
                "lateral_offset": float,
                "heading_error": float,
                "timestamp": float
            }
        """
```

**통합 로직:**
```python
1. LaneTracker로 위치 추적
2. DepartureDetector로 위험도 계산
3. WarningSystem 업데이트
4. risk_level >= 4이면 PIDController로 조향
5. 결과 반환
```

**테스트:** `tests/test_integration.py`
- Test Case 13: E2E 파이프라인
- Test Case 14: 이탈 시나리오

**예상 시간:** 1시간

---

#### 5. 설정 파일 🟢 LOW PRIORITY
**파일:** `config/lkas_params.yaml`  
**참고:** `docs/01_아키텍처_설계서.md` (§6.2)

**내용:**
```yaml
# RC Car Environment
environment:
  wheelbase: 0.25
  track_width: 0.35
  max_speed: 2.0

# Tracking
tracking:
  smoothing_window: 5
  min_confidence: 0.6

# Departure Detection
departure:
  risk_thresholds:
    level_2: 0.08
    level_3: 0.12
    level_4: 0.15
    level_5: 0.18

# PID Controller
controller:
  kp: 2.0
  ki: 0.2
  kd: 0.5
  k_heading: 0.2
  max_steering_angle: 45.0
  max_steering_rate: 100.0

# Warning
warning:
  enable_visual: true
  enable_audio: true
```

**예상 시간:** 10분

---

### Phase 3: 검증 (구현 완료 후) ⏳ 0%

#### 6. 구현 일치율 분석 📝
**파일:** `docs/04_구현_일치율_분석.md`  
**참고:** Module 01의 `docs/04_구현_일치율_분석.md` 참고

**내용:**
```markdown
1. 설계 vs 구현 비교
   - 클래스/함수명 일치 여부
   - 파라미터 일치 여부
   - 알고리즘 일치 여부

2. 차이점 분석
   - 의도적 변경 사항
   - 개선 사항
   - 제약 사항

3. 변경 사항 정당화
```

**예상 시간:** 1-2시간

---

#### 7. 성능 평가 📝
**파일:** `docs/05_성능_평가.md`  
**참고:** Module 01의 `docs/05_테스트_성능_평가.md` 참고

**내용:**
```markdown
1. KPI 달성도
   - 처리 지연시간: < 30ms?
   - 차선 중심 MAE: < 5cm?
   - 이탈 감지 Precision: > 85%?
   - 이탈 감지 Recall: > 90%?

2. 정량적 측정
   - 벤치마크 결과
   - 메모리 사용량
   - CPU 사용률

3. 정성적 평가
   - 부드러운 제어
   - 예측 가능성
   - 강건성

4. 개선 방향
```

**예상 시간:** 2-3시간 (실차 테스트 포함)

---

## 📝 작업 순서 (권장)

### Step 1: 핵심 컴포넌트 구현 (3-4시간)
```
1. DepartureDetector 구현 (1h)
   └─ tests/test_departure_detector.py 작성 및 실행

2. PIDController 구현 (1h)
   └─ tests/test_pid_controller.py 작성 및 실행

3. WarningSystem 구현 (30m)
   └─ tests/test_warning_system.py 작성 및 실행

4. Main Orchestrator 구현 (1h)
   └─ tests/test_integration.py 작성 및 실행
```

### Step 2: 통합 테스트 (1시간)
```
5. 전체 파이프라인 테스트
   └─ Mock 데이터로 E2E 검증
   └─ 모든 테스트 케이스 통과 확인
```

### Step 3: 문서화 (2-3시간)
```
6. 구현 일치율 분석 작성
7. 성능 평가 (실차 테스트 시 작성)
```

---

## 🔧 환경 설정

### 가상환경 활성화
```bash
cd /Users/ahnhyunjun/Desktop/SEA_ME/-autonomous-driving_ML/02-lane-keeping-assist
source venv/bin/activate
```

### 테스트 실행
```bash
# 빠른 검증
python test_quick.py

# 개별 테스트
python -m pytest tests/test_departure_detector.py -v -s

# 전체 테스트
python -m pytest tests/ -v -s
```

---

## 📚 참고 문서

### 필수 읽기:
1. `docs/02_구현_명세서.md` - 모든 클래스/함수 명세
2. `docs/03_검증서.md` - 테스트 케이스
3. `docs/01_아키텍처_설계서.md` - 전체 구조

### 참고 코드:
1. `src/tracking/lane_tracker.py` - 구현 스타일 참고
2. `tests/test_lane_tracker.py` - 테스트 작성 스타일

### Module 01 참고:
1. `01-lane-detection/docs/04_구현_일치율_분석.md`
2. `01-lane-detection/docs/05_테스트_성능_평가.md`

---

## ⚠️ 주의사항

### 1. RC 카 파라미터 사용
```python
✅ wheelbase = 0.25m (NOT 2.5m!)
✅ track_width = 0.35m (NOT 3.5m!)
✅ max_steering_angle = 45° (NOT 30°!)
✅ max_steering_rate = 100°/s (NOT 5°/s!)
```

### 2. 좌표 순서 일관성
```python
✅ (X, Y) 순서 사용
   - vehicle_position: (320, 432) = (X, Y)
   - polyline: [(x, y), ...]
```

### 3. 단위 일치
```python
✅ lateral_offset: meters
✅ heading_error: degrees
✅ curvature: 1/m
✅ speed: m/s (NOT km/h in calculations!)
```

### 4. 테스트 우선
```
각 클래스 구현 후 즉시 테스트 작성 및 실행!
ALL PASS 확인 후 다음으로 진행!
```

---

## 🎯 완료 기준

### Phase 2 완료 조건:
- ✅ 4개 클래스 모두 구현
- ✅ 15개+ 테스트 케이스 ALL PASS
- ✅ Integration test 통과
- ✅ 문서와 100% 일치

### Phase 3 완료 조건:
- ✅ 구현 일치율 분석 문서
- ✅ 성능 평가 문서
- ✅ KPI 목표 달성 확인

---

## 🚀 예상 총 소요 시간

| Phase | 작업 | 예상 시간 |
|-------|------|-----------|
| Phase 2 | 컴포넌트 구현 | 3-4시간 |
| Phase 2 | 통합 & 테스트 | 1시간 |
| Phase 3 | 문서 작성 | 2-3시간 |
| **합계** | | **6-8시간** |

---

## 📞 질문 시 참고

### AI에게 제공할 컨텍스트:
```
"Module 02 (Lane Keeping Assist) 구현 중입니다.

현재 상태:
- ✅ LaneTracker 완성 (테스트 통과)
- ⏳ DepartureDetector 구현 필요

참고 문서:
- docs/02_구현_명세서.md (§3.2)
- docs/03_검증서.md (§4.2)

구현 스타일은 src/tracking/lane_tracker.py 참고
테스트 스타일은 tests/test_lane_tracker.py 참고
```

---

**작성자:** Previous AI Assistant  
**전달 대상:** Next AI Assistant  
**프로젝트:** SEA:ME Autonomous Driving ML (Module 02)  
**우선순위:** 🔴 HIGH - 포트폴리오 핵심 컴포넌트

**행운을 빕니다! 🚀**
