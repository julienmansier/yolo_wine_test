# Usage Guide

## Installation

### Option 1: Using the setup script (Recommended)

```bash
python src/setup.py
```

This will guide you through creating a virtual environment and installing dependencies.

### Option 2: Manual installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Simple Detection (Single Image)

Detect bottles in a single image and view the results:

```bash
# Basic usage with nano model (fast, good for simple scenes)
python examples/simple_detection.py path/to/image.jpg

# Use medium model for better accuracy (recommended for dense/crowded scenes)
python examples/simple_detection.py path/to/image.jpg --model m

# Combine with other options
python examples/simple_detection.py path/to/shelf.jpg --model m --conf 0.20

# Available models: n (nano), s (small), m (medium), l (large), x (extra-large)
```

**Model Selection Guide:**
- `--model n` (nano): Fastest, best for 1-5 bottles, simple scenes
- `--model s` (small): Balanced speed/accuracy
- `--model m` (medium): **Recommended for shelves** - 50%+ more bottles detected
- `--model l` (large): High accuracy, slower
- `--model x` (extra-large): Best accuracy, slowest

The script will:
- Load the YOLO model
- Detect wine bottles
- Display detection details
- Save results to `output_detection.jpg`

### 2. Batch Processing (Multiple Images)

Process an entire folder of images:

```bash
# Basic usage
python examples/batch_processing.py sample_images/

# Use larger model for better accuracy
python examples/batch_processing.py sample_images/ --model m --conf 0.20
```

This will:
- Process all images in a directory
- Save annotated images to `batch_results/`
- Export detection data to CSV
- Print a summary report

### 3. Model Comparison

Compare different YOLO models to find the best one for your needs:

```bash
cd examples
python comparison.py
```

This will test YOLOv8n, YOLOv8s, YOLOv8m, and YOLO11n on the same image and show:
- Detection count
- Inference speed
- Average confidence
- Visual comparison charts

## Using the WineBottleDetector Class

You can also use the detector directly in your own Python code:

```python
from src.utils.detector import WineBottleDetector

# Initialize detector
detector = WineBottleDetector(
    model_name='models/yolo26n.pt',      # Latest model (recommended)
    # or: 'models/yolo26s.pt', 'models/yolo26m.pt', 'models/yolo11n.pt', 'models/yolov8n.pt'
    confidence_threshold=0.25      # minimum confidence (0-1)
)

# Detect bottles in a single image
count, detections = detector.detect_bottles(
    'path/to/image.jpg',
    visualize=True,               # show the image
    save_path='result.jpg'        # save annotated image
)

print(f"Found {count} bottles")

# Process multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.batch_detect(
    image_paths,
    output_dir='results'          # save annotated images here
)

for img_path, (count, detections) in results.items():
    print(f"{img_path}: {count} bottles")
```

## Model Selection Guide

### YOLO26n (Latest Nano - **Recommended**)
- **Speed**: Very fast (~43% faster on CPU than v11)
- **Accuracy**: Excellent, especially for small objects
- **Use for**: Edge devices, CPU inference, production deployments
- **Model file**: [models/yolo26n.pt](models/yolo26n.pt) (~5 MB)
- **Special features**: NMS-free (no post-processing), optimized for low-power devices

### YOLO26s (Latest Small)
- **Speed**: Fast
- **Accuracy**: Better than nano
- **Use for**: Balanced performance with latest improvements
- **Model file**: [models/yolo26s.pt](models/yolo26s.pt)

### YOLO26m (Latest Medium)
- **Speed**: Medium
- **Accuracy**: Very good
- **Use for**: When accuracy is more important than speed
- **Model file**: [models/yolo26m.pt](models/yolo26m.pt)

### YOLO11n (Previous Generation)
- **Speed**: Very fast
- **Accuracy**: Good
- **Use for**: If you need compatibility with older systems
- **Model file**: [models/yolo11n.pt](models/yolo11n.pt) (~5 MB)

### YOLOv8n (Nano)
- **Speed**: Fast (~100+ FPS on GPU)
- **Accuracy**: Good
- **Use for**: Well-documented, mature ecosystem
- **Model file**: [models/yolov8n.pt](models/yolov8n.pt) (~6 MB)

## Confidence Threshold

The confidence threshold (0-1) determines how certain the model must be to count a detection:

- **0.25** (default): Good balance, may include some false positives
- **0.5**: More conservative, fewer false positives
- **0.7**: Very strict, only high-confidence detections

Example:
```python
# Strict detection
detector = WineBottleDetector(
    model_name='models/yolov8n.pt',
    confidence_threshold=0.7
)
```

## Output Format

Each detection includes:
```python
{
    'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
    'confidence': 0.85,         # Confidence score (0-1)
    'class': 'bottle'           # Object class
}
```

## Tips for Best Results

1. **Image Quality**: Use clear, well-lit images
2. **Model Selection**: Start with YOLO26n (latest, fastest), upgrade to larger models if accuracy is insufficient
3. **CPU vs GPU**: YOLO26 is optimized for CPU inference (43% faster), making it ideal for edge devices
4. **Confidence Threshold**: Adjust based on your needs (lower for more detections, higher for more precision)
5. **Batch Processing**: Use batch processing for large datasets to get CSV exports
6. **GPU Acceleration**: Models run much faster on GPU (CUDA). Check if PyTorch detects your GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Troubleshooting

### Model download fails
Models are downloaded automatically on first use. If download fails:
- Check your internet connection
- Manually download from [Ultralytics GitHub](https://github.com/ultralytics/assets/releases)

### Out of memory errors
- Use a smaller model (nano instead of medium)
- Reduce image resolution
- Process images one at a time instead of batch

### Low detection accuracy
- Try a larger model (medium instead of nano)
- Lower the confidence threshold
- Ensure images are clear and bottles are visible
- Consider fine-tuning the model on your specific bottle types

## Advanced Usage

### Custom Training

If the pre-trained models aren't accurate enough for your specific bottle types, you can fine-tune:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('models/yolov8n.pt')

# Train on your custom dataset
model.train(
    data='path/to/dataset.yaml',
    epochs=100,
    imgsz=640
)
```

### Export to Other Formats

Export to ONNX, TensorRT, or other formats for deployment:

```python
model = YOLO('models/yolov8n.pt')
model.export(format='onnx')  # Creates yolov8n.onnx
```

## API Integration

Integrate into your web service:

```python
from flask import Flask, request, jsonify
from src.utils.detector import WineBottleDetector

app = Flask(__name__)
detector = WineBottleDetector(model_name='models/yolo26n.pt')

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files['image']
    image.save('temp.jpg')

    count, detections = detector.detect_bottles('temp.jpg')

    return jsonify({
        'bottle_count': count,
        'detections': detections
    })

if __name__ == '__main__':
    app.run()
```

## Further Reading

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO11 Release Notes](https://github.com/ultralytics/ultralytics)
- [COCO Dataset Classes](https://cocodataset.org/#explore)
