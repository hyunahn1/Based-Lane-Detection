# Module 01: Lane Detection

> Deep Learning-based semantic segmentation for lane detection using DeepLabV3+

## 📌 Module Overview

This module provides **high-accuracy lane detection** using semantic segmentation. It serves as the foundational vision component for autonomous driving systems.

**Status:** ✅ Completed

**Key Features:**
- DeepLabV3+ with ResNet-101 backbone
- Combined loss function (CrossEntropy + Dice)
- Advanced post-processing pipeline
- IoU: 0.6945 (optimized model)

## 🔗 Integration Interface

This module can be used **standalone** or integrated with other modules.

### Input
- **Type:** RGB Image
- **Format:** `numpy.ndarray` or `PIL.Image`
- **Resolution:** 640×480 (recommended)

### Output
```python
{
    "lane_mask": np.ndarray,      # Binary mask (H, W)
    "lane_polyline": List[Point],  # Lane centerline points
    "confidence": float,           # Prediction confidence
    "processing_time": float       # Inference time (ms)
}
```

### Usage Example
```python
from src.models.deeplabv3plus import get_model
from src.inference.postprocess import PostProcessor

# Load model
model = get_model(num_classes=2, pretrained=True)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Detect lanes
image = cv2.imread('test.jpg')
mask = model(image)
polyline = PostProcessor()(mask)
```

## 📊 Performance

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| IoU | 0.6576 | 0.6945 |
| Dice Score | 0.7934 | 0.8198 |
| Pixel Accuracy | 98.49% | 98.88% |
| FPS (RTX 5090) | ~60 | ~50 |

## 📂 Directory Structure

```
01-lane-detection/
├── README.md                  # This file
├── docs/                      # Documentation
├── src/                       # Source code
├── tests/                     # Unit tests
├── test_results/              # Evaluation results
├── train_baseline.py          # Training script
├── test_with_postprocess.py   # Testing script
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train_optimized.py

# Test
python test_with_postprocess.py
```

## 📖 Documentation

See `docs/` folder for detailed documentation:
- [Architecture Design](docs/01_아키텍처_설계서_v2_고성능.md)
- [Implementation Specification](docs/02_구현_명세서_v2_고성능.md)
- [Verification Report](docs/03_검증서_v2_고성능.md)
- [Performance Evaluation](docs/05_테스트_성능_평가.md)

## 🔗 Related Modules

- **Module 02:** Lane Keeping Assist (uses this module's output)
- **Module 05:** Semantic Segmentation (extends this module)

## 📝 License

MIT License - See [LICENSE](../LICENSE) for details
