# Wine Bottle Detection API Documentation

A FastAPI-based REST API for detecting wine bottles in images using YOLO26 models.

## Features

- Base64-encoded image input (JSON)
- Multiple YOLO model sizes (nano, small, medium, large, extra-large)
- Configurable confidence threshold
- Returns detection count, bounding boxes, and confidence scores
- Optional annotated image with bounding boxes
- Wine-only detection (excludes vases via `--no-vases` flag)
- Docker support for easy deployment
- Automatic API documentation (Swagger/OpenAPI)

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API:**
   ```bash
   uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

2. **Or build manually:**
   ```bash
   docker build -t wine-bottle-api .
   docker run -p 8000:8000 wine-bottle-api
   ```

3. **Check health:**
   ```bash
   curl http://localhost:8000/health
   ```

## API Endpoints

### POST /api/v1/detect

Detect wine bottles in a base64-encoded image.

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "confidence": 0.25,
  "model": "n",
  "return_annotated": true
}
```

**Parameters:**
- `image` (required): Base64-encoded image string (with or without data URI prefix)
- `confidence` (optional): Confidence threshold (0.0-1.0), default: 0.25
- `model` (optional): Model size - "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (extra-large), default: "n"
- `return_annotated` (optional): Return annotated image with bounding boxes, default: true

**Response:**
```json
{
  "success": true,
  "total_count": 3,
  "detections": [
    {
      "confidence": 0.89,
      "bbox": [123.4, 56.7, 234.5, 456.8],
      "class_name": "bottle",
      "class_id": 39
    },
    {
      "confidence": 0.76,
      "bbox": [345.6, 78.9, 456.7, 567.8],
      "class_name": "bottle",
      "class_id": 39
    },
    {
      "confidence": 0.82,
      "bbox": [567.8, 90.1, 678.9, 678.9],
      "class_name": "bottle",
      "class_id": 39
    }
  ],
  "model_used": "yolo26n.pt",
  "confidence_threshold": 0.25,
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### GET /health

Health check endpoint for monitoring and load balancers.

**Response:**
```json
{
  "status": "healthy",
  "service": "wine-bottle-detection-api",
  "version": "1.0.0"
}
```

### GET /api/v1/info

Get API information and capabilities.

**Response:**
```json
{
  "service": "Wine Bottle Detection API",
  "version": "1.0.0",
  "description": "YOLO-based wine bottle detection with configurable models",
  "available_models": ["n (nano)", "s (small)", "m (medium)", "l (large)", "x (extra-large)"],
  "default_model": "n (nano)",
  "default_confidence": 0.25,
  "endpoints": ["/health", "/api/v1/info", "/api/v1/detect"]
}
```

### GET /

Root endpoint with basic API information and endpoint list.

## Usage Examples

### Python

```python
import base64
import requests
import json
import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).parent.parent))

# Read and encode image
with open("wine_bottle.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Prepare request
url = "http://localhost:8000/api/v1/detect"
payload = {
    "image": f"data:image/jpeg;base64,{image_base64}",
    "confidence": 0.25,
    "model": "n",
    "return_annotated": True
}

# Send request
response = requests.post(url, json=payload)
result = response.json()

print(f"Found {result['total_count']} wine bottles!")
for i, detection in enumerate(result['detections'], 1):
    print(f"Bottle {i}: {detection['confidence']:.2%} confidence")

# Save annotated image if returned
if result.get('annotated_image'):
    annotated_base64 = result['annotated_image'].split(',')[1]
    with open("output_annotated.jpg", "wb") as f:
        f.write(base64.b64decode(annotated_base64))
```

### cURL

```bash
# Convert image to base64
IMAGE_B64=$(base64 -i wine_bottle.jpg)

# Send request
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"data:image/jpeg;base64,$IMAGE_B64\",
    \"confidence\": 0.25,
    \"model\": \"n\",
    \"return_annotated\": true
  }"
```

### JavaScript/Node.js

```javascript
const fs = require('fs');
const axios = require('axios');

// Read and encode image
const imageBuffer = fs.readFileSync('wine_bottle.jpg');
const imageBase64 = imageBuffer.toString('base64');

// Send request
axios.post('http://localhost:8000/api/v1/detect', {
  image: `data:image/jpeg;base64,${imageBase64}`,
  confidence: 0.25,
  model: 'n',
  return_annotated: true
})
.then(response => {
  const result = response.data;
  console.log(`Found ${result.total_count} wine bottles!`);

  result.detections.forEach((det, i) => {
    console.log(`Bottle ${i+1}: ${(det.confidence * 100).toFixed(1)}% confidence`);
  });
})
.catch(error => {
  console.error('Error:', error.response?.data || error.message);
});
```

## Model Selection Guide

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| **n (nano)** | Fastest | Good | ~10MB | Default, real-time processing, low resources |
| **s (small)** | Fast | Better | ~40MB | Better accuracy with reasonable speed |
| **m (medium)** | Medium | Best | ~100MB | Production use with accuracy priority |
| **l (large)** | Slow | Excellent | ~200MB | High accuracy requirements |
| **x (extra-large)** | Slowest | Best | ~300MB | Maximum accuracy, offline processing |

**Recommendations:**
- **Development/Testing**: Use nano (n) for fast iteration
- **Production (balanced)**: Use small (s) or medium (m)
- **High accuracy**: Use medium (m) or large (l)
- **Edge devices**: Use nano (n) only

## Configuration

### Environment Variables

You can configure the API using environment variables:

```bash
# Example: Run on different port
export PORT=8080
uvicorn api:app --host 0.0.0.0 --port $PORT
```

### Performance Tuning

For production deployments:

```bash
# Use multiple workers (CPU-bound)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

# Or use Gunicorn with Uvicorn workers
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid image, parameters)
- `422`: Validation error (malformed request)
- `500`: Internal server error (detection failed)

**Example error response:**
```json
{
  "detail": "Detection failed: Invalid base64 image data"
}
```

## Security Considerations

1. **Image Size Limits**: Consider adding request size limits in production
2. **Rate Limiting**: Implement rate limiting for public APIs
3. **Authentication**: Add API key authentication if needed
4. **CORS**: Configure CORS for web client access

Example with rate limiting and CORS:
```bash
pip install slowapi fastapi-cors
```

## Monitoring

### Prometheus Metrics (Optional)

Add prometheus metrics:
```bash
pip install prometheus-fastapi-instrumentator
```

### Logging

Enable detailed logging:
```bash
uvicorn src.api:app --log-level debug
```

## Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure YOLO model files (yolo26*.pt) are in the [models/](models/) directory
   - Download missing models with: `yolo task=detect mode=predict model=yolo26n.pt`

2. **Out of memory errors**
   - Use smaller model (nano instead of medium)
   - Reduce image resolution before sending
   - Lower confidence threshold

3. **Slow detection**
   - Use nano model for faster inference
   - Consider GPU support (see below)

### GPU Support

For faster inference with NVIDIA GPUs:

1. Install PyTorch with CUDA support
2. Run with GPU flag:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```
   YOLO will automatically use GPU if available.

## API Versioning

Current version: `v1`

The API uses URL versioning (`/api/v1/...`) to support future updates without breaking existing clients.

## License

MIT License - See project LICENSE file for details.

## Support

- GitHub Issues: [Report bugs or request features]
- API Documentation: http://localhost:8000/docs
- Project README: See main repository README.md
