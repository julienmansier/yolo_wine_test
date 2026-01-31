# Project Notes

## Detection Results Summary

Successfully tested YOLO26 on wine bottle images with the following results:

### Batch Processing Results
- **Total images processed**: 6
- **Total bottles detected**: 55
- **Average per image**: 9.2 bottles
- **Processing speed**: ~20-24ms per image on CPU

### Per-Image Results
1. `shelf_horizontal_medium.jpg`: 24 bottles
2. `shelf_horizontal_close.jpg`: 21 bottles
3. `wine_bottles.jpeg`: 4 bottles
4. `shelf_horizontal_far.jpg`: 2 bottles
5. `one_wine.jpg`: 2 bottles
6. `wine_bottle.jpeg`: 2 bottles

## Important Findings

### Wine Bottles Detected as Vases
Wine bottles are often classified as **"vases" (COCO class 75)** rather than "bottles" (COCO class 39). This is expected behavior because:
- Wine bottles have a similar shape to decorative vases
- The COCO dataset "bottle" class was trained primarily on water/soda bottles
- Wine bottles (especially empty ones with long necks) visually resemble vases

**Solution**: The detector now checks for both:
- Class 39: "bottle"
- Class 75: "vase" (marked as "likely wine bottle")

### YOLO26 Performance
- Very fast on CPU (~20-25ms per image)
- Good accuracy for wine bottle detection
- Models download automatically on first run (~5.3 MB for nano model)
- No GPU required for good performance

### Model Size Impact on Accuracy

Testing on `shelf_horizontal_close.jpg` with `--conf 0.20`:

| Model | Bottles Detected | Inference Speed | Model Size | Improvement |
|-------|------------------|-----------------|------------|-------------|
| YOLO26n (nano) | 21 bottles | ~24ms | ~5 MB | Baseline |
| YOLO26s (small) | 31 bottles | ~35ms | ~12 MB | +48% |
| YOLO26m (medium) | 32 bottles | ~71ms | ~42 MB | +52% |

**Key Findings:**
- **Nano model**: Best for simple scenes (1-5 bottles), fastest
- **Medium model**: **Recommended for dense/crowded shelves** - detects 50%+ more bottles
- **Trade-off**: 3x slower (71ms vs 24ms) but still very fast on CPU
- **GPU impact**: None on accuracy - only affects speed, not detection count

**When to use larger models:**
- Dense scenes with many overlapping bottles
- Partial occlusions (bottles behind bottles)
- Small/distant bottles
- When accuracy is more important than speed

## CLI Usage Tips

All example scripts now support command-line arguments:

```bash
# Single image detection
python examples/simple_detection.py path/to/image.jpg

# Batch processing
python examples/batch_processing.py path/to/directory/

# Model comparison
python examples/comparison.py path/to/test_image.jpg
```

Or run interactively without arguments and you'll be prompted for paths.

## Output Files

### Simple Detection
- `output_detection.jpg` - Annotated image with bounding boxes

### Batch Processing
- `batch_results/` - Directory with all annotated images
- `batch_results/detection_results.csv` - Detailed CSV with:
  - Image names and paths
  - Bottle counts per image
  - Individual detection details (bbox coordinates, confidence)

## Virtual Environment

Always activate the virtual environment before running scripts:

```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

Then run:
```bash
python examples/simple_detection.py path/to/image.jpg
```

## Next Steps

Potential improvements:
1. Fine-tune the model on wine bottle dataset for better accuracy
2. Add support for video detection
3. Create a web API endpoint for remote detection
4. Export detection results in additional formats (JSON, XML)
5. Add confidence threshold adjustment via CLI arguments
