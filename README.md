# Lane Keeping Assist System

Implementation of lane keeping assist using three different approaches.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CARLA](https://img.shields.io/badge/Simulator-CARLA%200.9.15-blue.svg)](https://carla.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Modules](#modules)
- [CARLA Integration](#carla-integration)
- [Installation](#installation)
- [Documentation](#documentation)
- [Current Status](#current-status)

---

## Overview

This project implements Lane Keeping Assist System (LKAS) using three different approaches to compare traditional, modern deep learning, and reinforcement learning methods.

### Three Approaches

**1. Traditional Pipeline**
- Lane detection with semantic segmentation
- Control with PID and MPC
- Object detection for safety

**2. End-to-End Learning**
- Direct image-to-control using Vision Transformer
- No intermediate representations

**3. Reinforcement Learning**
- Self-learning with PPO algorithm
- Exploration with Intrinsic Curiosity Module

---

## Modules

### Module 01: Lane Detection

Semantic segmentation for lane marking detection.

**Implementation:**
- DeepLabV3+ with ResNet101 backbone
- CBAM attention mechanism
- Boundary-aware loss function
- Knowledge distillation for model compression

**Status:** Implemented and tested

---

### Module 02: Lane Keeping

Controllers for maintaining vehicle in lane.

**Implementation:**
- PID controller (fast response)
- Model Predictive Control (optimal trajectory)
- Risk assessment system
- Multi-level warning system

**Status:** Implemented and tested

---

### Module 03: Object Detection

Real-time object detection for obstacle awareness.

**Implementation:**
- YOLOv8 Large (43M parameters)
- CBAM attention in backbone
- Small object detection head
- 5 classes: traffic cone, obstacle, car, sign, pedestrian

**Status:** Trained on CARLA data

---

### Module 06: End-to-End Learning

Direct control from camera images.

**Implementation:**
- Vision Transformer (ViT-Base, 86M parameters)
- 12-layer transformer encoder
- Direct output: steering angle and throttle

**Status:** Trained on CARLA driving data

---

### Module 08: Reinforcement Learning

Self-learning autonomous agent.

**Implementation:**
- PPO (Proximal Policy Optimization)
- Actor-Critic architecture (10M parameters)
- ICM (Intrinsic Curiosity Module) for exploration
- Curiosity reward for sparse environments

**Status:** Trained in CARLA simulation

---

## CARLA Integration

Three simulation scenarios implemented in CARLA 0.9.15:

### Simulation 1: Traditional Pipeline

Uses Module 01 (Lane Detection) + Module 02 (Lane Keeping)

**Flow:**
```
Camera → DeepLabV3+ → Lane Tracking → PID/MPC → Vehicle Control
```

**Characteristics:**
- Interpretable intermediate steps
- Reliable in tested scenarios
- Fast inference (~30ms total)

---

### Simulation 2: End-to-End

Uses Module 06 (Vision Transformer)

**Flow:**
```
Camera → Vision Transformer → Direct Control → Vehicle
```

**Characteristics:**
- Simple pipeline
- Learns implicit representations
- Requires driving data

---

### Simulation 3: Reinforcement Learning

Uses Module 08 (PPO + ICM)

**Flow:**
```
Camera + State → PPO Agent → Vehicle Control
                    ↓
            Curiosity Module (ICM)
```

**Characteristics:**
- Learns through trial and error
- No demonstration data needed
- Exploration via curiosity rewards

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU)
- CARLA 0.9.15 (for simulation)

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/autonomous-driving-ML.git
cd autonomous-driving-ML

# Install dependencies
pip install -r requirements.txt

# For CARLA integration
cd carla-integration
pip install -r requirements.txt
```

### Download Models

Trained models are available separately (not in git due to size):

- Lane Detection: `best_lane_model.pth` (328MB)
- Object Detection: `best_carla_yolo.pt` (175MB)
- E2E Model: `best_e2e_model.pth` (328MB)
- RL Agent: `best_ppo_agent.pth` (11MB)

Place models in respective module directories.

---

## Documentation

### Project Documentation

- [Project Requirements](docs/problem.md)
- [Performance Comparison](docs/PERFORMANCE_COMPARISON.md)

### Module Documentation

Each module has detailed documentation:

```
XX-module-name/
├── README.md                    # Overview and usage
├── docs/
│   ├── 01_아키텍처_설계서.md      # Architecture design
│   ├── 02_구현_명세서.md          # Implementation spec
│   └── 03_검증서.md              # Verification plan
└── src/                         # Source code
```

### CARLA Integration Documentation

- [Simulation 1 (Traditional)](carla-integration/docs/01_Sim1_아키텍처_설계서.md)
- [Simulation 2 (E2E)](carla-integration/docs/04_Sim2_아키텍처_설계서.md)
- [Simulation 3 (RL)](carla-integration/docs/07_Sim3_아키텍처_설계서.md)

---

## Current Status

### Completed

- ✅ Module 01: Lane Detection (DeepLabV3+ implementation and training)
- ✅ Module 02: Lane Keeping (PID and MPC controllers)
- ✅ Module 03: Object Detection (YOLOv8 training on CARLA data)
- ✅ Module 06: End-to-End Learning (ViT implementation and training)
- ✅ Module 08: Reinforcement Learning (PPO + ICM training)
- ✅ CARLA integration code for all three approaches
- ✅ Documentation (32+ design documents)

### Testing Status

Module-level testing completed:
- Module 01: Inference and postprocessing tested
- Module 02: Controller tests passed
- Module 03: Detection inference tested
- Module 06: Model architecture verified (8/8 tests passed)
- Module 08: RL components tested (15/15 tests passed)

### Pending

- CARLA simulation execution and recording
- Real hardware deployment (PiRacer)
- Performance benchmarking across all three approaches

---

## Usage

### Module 01: Lane Detection

```bash
cd 01-lane-detection

# Test with sample image
python test_with_postprocess.py

# Train (if you have dataset)
python train_optimized.py
```

### Module 02: Lane Keeping

```bash
cd 02-lane-keeping-assist

# Run controller tests
python -m pytest tests/
```

### Module 06: End-to-End

```bash
cd 06-end-to-end-learning

# Verify implementation
python test_basic.py
```

### Module 08: Reinforcement Learning

```bash
cd 08-reinforcement-learning

# Test RL components
python test_basic.py
python test_curiosity.py
```

### CARLA Simulations

```bash
# Terminal 1: Start CARLA
cd CARLA_0.9.15
./CarlaUE4.sh

# Terminal 2: Run simulation
cd carla-integration/sim1-traditional
python main.py

# Or run other simulations
cd carla-integration/sim2-e2e
python main.py

cd carla-integration/sim3-rl
python main.py
```

---

## Project Structure

```
.
├── 01-lane-detection/          # DeepLabV3+ semantic segmentation
├── 02-lane-keeping-assist/     # PID and MPC controllers
├── 03-object-detection/        # YOLOv8 object detection
├── 06-end-to-end-learning/     # Vision Transformer E2E
├── 08-reinforcement-learning/  # PPO + ICM agent
├── carla-integration/          # CARLA simulation integration
│   ├── sim1-traditional/       # Traditional pipeline
│   ├── sim2-e2e/              # End-to-end learning
│   ├── sim3-rl/               # Reinforcement learning
│   └── data-collection/       # CARLA data collection scripts
├── docs/                       # Project-level documentation
└── tests/                      # Integration tests
```

---

## Technologies Used

### Deep Learning
- PyTorch 2.0+
- torchvision
- Ultralytics YOLOv8

### Computer Vision
- OpenCV
- PIL/Pillow

### Control
- NumPy
- SciPy (for MPC optimization)

### Simulation
- CARLA 0.9.15
- pygame (for CARLA interface)

---

## References

### Papers Implemented

1. **DeepLabV3+**: Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation," ECCV 2018
2. **CBAM**: Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018
3. **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR 2021
4. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms," 2017
5. **ICM**: Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction," ICML 2017

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- CARLA Simulator team for the simulation platform
- PyTorch team for the deep learning framework
- Ultralytics for YOLOv8 implementation

---
