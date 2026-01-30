# YOLO Wine Bottle Detector

A cost-effective wine bottle detection and counting system using YOLO (You Only Look Once) models as an alternative to cloud-based API services.

## Overview

This project explores using YOLO object detection models to detect and count wine bottles in images without incurring API costs. It's designed as an alternative to using Claude or other cloud services for simple bottle counting tasks.

## Why YOLO?

- **Cost-effective**: Run locally without API fees
- **Fast**: Real-time detection capabilities
- **Accurate**: Modern YOLO models (v8, v11, v26) offer excellent accuracy
- **Privacy**: Process images locally without sending to external services

## YOLO Model Options

### YOLO26 (Latest - Recommended)
- **Released**: January 14, 2026
- **43% faster CPU inference** compared to previous versions
- **NMS-free design**: No post-processing needed, lower latency
- **Optimized for edge devices** and low-power hardware
- **Better small object detection** with improved loss functions
- Multiple model sizes (nano to extra-large)

### YOLO11
- Previous state-of-the-art model
- Enhanced accuracy and speed over v8
- Still excellent for general use

### YOLOv8 (Ultralytics)
- Most popular and well-documented
- Easy to use Python API
- Pre-trained on COCO dataset (includes "bottle" class)
- Mature ecosystem with lots of examples

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from ultralytics import YOLO

# Load pre-trained model (using latest YOLO26)
model = YOLO('yolo26n.pt')  # nano model for speed

# Detect bottles in an image
results = model('wine_bottles.jpg')

# Count bottles
bottle_count = sum(1 for det in results[0].boxes if det.cls == 39)  # 39 is bottle class in COCO
print(f"Found {bottle_count} bottles")
```

## Project Structure

```
yolo-wine-bottle-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── examples/
│   ├── simple_detection.py      # Basic bottle detection
│   ├── batch_processing.py      # Process multiple images
│   └── comparison.py            # Compare different YOLO models
├── utils/
│   └── detector.py              # Reusable detection utilities
└── sample_images/               # Test images (not in repo)
```

## Features

- Simple bottle detection and counting
- Batch image processing
- Visualization of detection results
- Model comparison utilities
- Export results to JSON/CSV

## Next Steps

1. Test with your existing wine bottle images
2. Fine-tune model if needed for specific bottle types
3. Integrate into your larger project
4. Add custom training data if generic bottle detection isn't accurate enough

## Comparison to Claude

| Feature | YOLO | Claude API |
|---------|------|------------|
| Cost | Free (local) | Pay per image |
| Speed | Very fast (local GPU) | Network latency |
| Privacy | Complete | Data sent to API |
| Accuracy | Good for bottles | Excellent for complex analysis |
| Setup | Requires model download | API key only |

## Use Claude when:
- You need complex image understanding
- You want natural language descriptions
- You need context about the scene

## Use YOLO when:
- Simple object counting
- Real-time processing needed
- Cost is a concern
- Privacy is important

## License

MIT

## Contributing

Pull requests welcome!
