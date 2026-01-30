# Module 02: Lane Keeping Assist System

> **PID-based steering control system for autonomous lane keeping**

## 📌 Module Overview

This module implements a **Lane Keeping Assist System (LKAS)** that monitors vehicle position relative to lane center and provides warnings and corrective steering interventions to prevent lane departure.

**Status:** 🔄 **In Development** (Design Phase Complete)

**Dependencies:** Module 01 (Lane Detection)

## 🎯 Key Features

### Core Capabilities
- ✅ **Lane Center Tracking** - Real-time vehicle position estimation
- ✅ **Departure Detection** - 6-level risk assessment (0-5)
- ✅ **Multi-Modal Warning** - Visual, auditory, and haptic alerts
- ✅ **PID Steering Control** - Smooth corrective interventions
- ✅ **Safety Mechanisms** - Fail-safe, watchdog, emergency override

### Safety First Design
- **Gradual Intervention:** Progressive control from warning to steering
- **Fail-Safe:** Graceful degradation when sensors fail
- **Driver Override:** Immediate control return on driver input
- **Watchdog Timer:** 30ms timeout protection

## 📊 Performance Targets

| Metric | Target | Priority |
|--------|--------|----------|
| **Latency** | < 25ms | P0 (Critical) |
| **Lane Center Accuracy** | < 10cm MAE | P0 |
| **Departure Detection Precision** | > 95% | P0 |
| **Departure Detection Recall** | > 98% | P0 |
| **False Warning Rate** | < 1 per 100km | P1 |
| **Steering Jerk** | < 2°/s² | P1 |
| **CPU Usage** | < 30% | P2 |
| **Memory Usage** | < 100MB | P2 |

## 🏗️ System Architecture

```
Input (Lane Detection)
    ↓
┌─────────────────────────────┐
│  Lane Tracker               │  - Extract lane center
│                             │  - Calculate lateral offset
│                             │  - Estimate heading error
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Departure Detector         │  - Risk level assessment
│                             │  - Time-to-crossing calc
└──────────┬──────────────────┘
           │
      ┌────┴────┐
      │         │
      ▼         ▼
┌──────────┐ ┌──────────────┐
│ Warning  │ │ PID          │  - Steering angle
│ System   │ │ Controller   │  - Throttle adjust
└──────────┘ └──────────────┘
```

## 📂 Project Structure

```
02-lane-keeping-assist/
├── README.md                    # This file
├── requirements.txt
├── main.py                      # Entry point
│
├── config/
│   └── lkas_params.yaml        # Configuration
│
├── src/
│   ├── lkas.py                 # Main orchestrator
│   ├── tracking/
│   │   └── lane_tracker.py    # Lane tracking
│   ├── detection/
│   │   └── departure_detector.py  # Departure detection
│   ├── control/
│   │   ├── pid_controller.py  # PID control
│   │   └── safety_manager.py  # Safety mechanisms
│   ├── alert/
│   │   └── warning_system.py  # Warning system
│   └── utils/
│
├── tests/                       # Unit & integration tests
└── docs/                        # Documentation
    ├── 01_아키텍처_설계서.md      ✅ Complete
    ├── 02_구현_명세서.md          ✅ Complete
    ├── 03_검증서.md              ✅ Complete
    ├── 04_구현_일치율_분석.md     ⏳ Pending
    └── 05_성능_평가.md           ⏳ Pending
```

## 🚀 Quick Start

### Installation

```bash
cd 02-lane-keeping-assist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.lkas import LaneKeepingAssist

# Initialize
lkas = LaneKeepingAssist(config_path="config/lkas_params.yaml")

# Process frame (integrate with Module 01)
lane_detection_output = {
    "lane_polyline": [(x1, y1), (x2, y2), ...],
    "confidence": 0.95
}

vehicle_state = {
    "speed": 60.0  # km/h
}

# Get control commands
result = lkas.process_frame(lane_detection_output, vehicle_state)

# Output
print(f"Steering Angle: {result['steering_angle']}°")
print(f"Warning Level: {result['warning_level']}")
print(f"Intervening: {result['is_intervening']}")
```

### Configuration

Edit `config/lkas_params.yaml`:

```yaml
# PID Parameters
controller:
  kp: 0.5      # Proportional gain
  ki: 0.1      # Integral gain
  kd: 0.2      # Derivative gain
  k_heading: 0.3

# Risk Thresholds
departure:
  risk_thresholds:
    level_2: 0.4  # Caution (meters)
    level_3: 0.6  # Warning
    level_4: 0.8  # Critical
    level_5: 1.0  # Emergency

# Safety
safety:
  max_intervention_duration: 5.0  # seconds
  min_speed_for_lkas: 15.0       # km/h
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run benchmarks
pytest tests/test_benchmark.py --benchmark-only

# Run specific test
pytest tests/test_lane_tracker.py::test_lane_tracker_normal
```

## 📖 Documentation

### Design Documents
- **[Architecture Design](docs/01_아키텍처_설계서.md)** - System design, components, data flow
- **[Implementation Specification](docs/02_구현_명세서.md)** - Detailed class specs, algorithms, APIs
- **[Verification Plan](docs/03_검증서.md)** - Testing strategy, KPIs, test cases

### API Documentation

See [Implementation Specification](docs/02_구현_명세서.md) for detailed API docs.

## 🔗 Integration with Other Modules

### Module 01 (Lane Detection)
```python
# Module 01 provides lane detection
from module_01.inference import LaneDetector

detector = LaneDetector()
lane_output = detector.detect(image)

# Module 02 consumes the output
lkas = LaneKeepingAssist()
control_output = lkas.process_frame(lane_output, vehicle_state)
```

### Future Modules
- **Module 03 (Object Detection):** Collision avoidance integration
- **Module 06 (End-to-End Learning):** Compare with learned steering

## 📊 Development Progress

### Phase 1: Design ✅ **Complete**
- [x] Architecture Design Document
- [x] Implementation Specification
- [x] Verification Plan

### Phase 2: Implementation ⏳ **Next**
- [ ] Lane Tracker
- [ ] Departure Detector
- [ ] PID Controller
- [ ] Warning System
- [ ] Safety Manager
- [ ] Unit Tests

### Phase 3: Integration 📦 **Pending**
- [ ] Module 01 integration
- [ ] E2E testing
- [ ] Performance optimization

### Phase 4: Validation 📦 **Pending**
- [ ] Simulation testing (CARLA/Gazebo)
- [ ] Real-world testing (PiRacer)
- [ ] Performance evaluation report

## 🎓 Key Learnings

This module demonstrates:
- **Control Theory:** PID controller design and tuning
- **Safety Engineering:** Fail-safe mechanisms, watchdog timers
- **Real-time Systems:** Low-latency processing pipelines
- **System Integration:** Module interface design
- **Testing Methodology:** Comprehensive test strategy

## 📝 License

MIT License - See [LICENSE](../LICENSE) for details

## 🔗 Related Modules

- **[Module 01: Lane Detection](../01-lane-detection/)** - Upstream dependency
- **[Module 03: Object Detection](../03-object-detection/)** - Future integration

---

**Development Timeline:** 1 week (6-10 days)  
**Current Phase:** Design Complete, Implementation Starting  
**Last Updated:** 2026-01-30
