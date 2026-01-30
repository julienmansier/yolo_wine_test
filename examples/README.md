# Examples

This directory contains example scripts demonstrating different ways to use the YOLO Wine Bottle Detector.

## Available Examples

### 1. simple_detection.py
Basic single-image detection with visualization.

```bash
python simple_detection.py
```

**What it does:**
- Loads a YOLO model
- Detects bottles in one image
- Shows annotated image with bounding boxes
- Saves result to `output_detection.jpg`

**Best for:** Getting started, testing the detector

---

### 2. batch_processing.py
Process multiple images at once and export results.

```bash
python batch_processing.py
```

**What it does:**
- Processes all images in a directory
- Saves annotated images to `batch_results/`
- Exports detection data to CSV
- Prints summary statistics

**Best for:** Processing large datasets, generating reports

---

### 3. comparison.py
Compare different YOLO models on the same image.

```bash
python comparison.py
```

**What it does:**
- Tests multiple YOLO models (v8n, v8s, v8m, v11n)
- Measures detection count, speed, and confidence
- Generates comparison charts
- Saves visualization to `model_comparison.png`

**Best for:** Choosing the right model for your use case

---

## Tips

- Place test images in `../sample_images/` directory
- All examples prompt for image paths if not specified
- Annotated images are saved automatically
- Press any key to close the visualization window

## Output Files

- `output_detection.jpg` - Single detection result
- `batch_results/` - Batch processing output
- `model_comparison.png` - Model comparison chart
- `detection_results.csv` - Detailed detection data
